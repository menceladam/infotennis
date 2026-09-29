"""Live Challenger tournament/match scraper (tennisabstract.com).

This is intentionally separate from the daily TennisMyLife CSV pipeline
(tennis_model/update_ratings.py): TennisMyLife's downloadable CSV lags a
completed tournament by roughly a day, while tennisabstract.com's
/current/ pages update same-day as matches finish. robots.txt on
tennisabstract.com explicitly allows this (only /jsfrags/, /jsmatches/,
/jsplayers/ are disallowed); no ToS restriction on automated access was
found on the site.

Match data is embedded directly in inline JavaScript string literals on
each tournament's page (var completedSingles = '...'; var
upcomingSingles = '...';) -- no JS execution needed, just HTML/regex
parsing of the string content.
"""
import re
from dataclasses import dataclass

import requests

BASE_URL = "https://www.tennisabstract.com"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; senorsanj-tennis-model/1.0)"}

TOURNEY_LINK_RE = re.compile(r'current/(\d{4}[A-Za-z0-9()\.\-]*Challenger)\.html')
JS_VAR_RE = lambda name: re.compile(r"var " + name + r" = '(.*?)';", re.S)
PLAYER_RE = re.compile(r'<a href="[^"]*?p=(\d+)/[^"]*?"[^>]*>([^<]+)</a>\s*\(([A-Z]{3})\)')
ROUND_LINE_RE = re.compile(r'^(\w+):\s*(.*)')


@dataclass
class MatchRecord:
    tournament: str
    round: str
    status: str  # "completed" or "upcoming"
    p1_id: str
    p1_name: str
    p1_country: str
    p2_id: str
    p2_name: str
    p2_country: str
    winner_name: str | None  # None if upcoming
    score: str | None  # None if upcoming


def list_current_challenger_tournaments() -> list[str]:
    """Return tournament slugs (e.g. '2026Plovdiv4Challenger') currently
    listed on the homepage's Current Challenger Tour section."""
    resp = requests.get(BASE_URL + "/", headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return sorted(set(TOURNEY_LINK_RE.findall(resp.text)))


def _parse_entries(raw: str, tournament: str, status: str) -> list[MatchRecord]:
    if not raw:
        return []
    records = []
    for entry in raw.split("<br/>"):
        entry = entry.strip()
        if not entry or entry == "&nbsp;":
            continue
        m = ROUND_LINE_RE.match(entry)
        if not m:
            continue
        round_, rest = m.group(1), m.group(2)
        players = PLAYER_RE.findall(rest)
        if len(players) != 2:
            continue
        (p1_id, p1_name, p1_ctry), (p2_id, p2_name, p2_ctry) = players

        winner_name = score = None
        if status == "completed":
            winner_name = p1_name
            score_part = re.split(r"\([A-Z]{3}\)", rest)[-1]
            score = re.sub(r"<[^>]+>", "", score_part).strip()

        records.append(MatchRecord(
            tournament=tournament, round=round_, status=status,
            p1_id=p1_id, p1_name=p1_name, p1_country=p1_ctry,
            p2_id=p2_id, p2_name=p2_name, p2_country=p2_ctry,
            winner_name=winner_name, score=score,
        ))
    return records


def fetch_tournament_matches(slug: str) -> list[MatchRecord]:
    url = f"{BASE_URL}/current/{slug}.html"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    html = resp.text

    tournament_name = slug[4:].replace("Challenger", "").strip()

    records = []
    completed_m = JS_VAR_RE("completedSingles").search(html)
    if completed_m:
        records += _parse_entries(completed_m.group(1), tournament_name, "completed")
    upcoming_m = JS_VAR_RE("upcomingSingles").search(html)
    if upcoming_m:
        records += _parse_entries(upcoming_m.group(1), tournament_name, "upcoming")
    return records


def fetch_all_current_challenger_matches() -> list[MatchRecord]:
    all_records = []
    for slug in list_current_challenger_tournaments():
        try:
            all_records += fetch_tournament_matches(slug)
        except requests.RequestException as e:
            print(f"  warning: failed to fetch {slug}: {e}")
    return all_records


if __name__ == "__main__":
    records = fetch_all_current_challenger_matches()
    completed = [r for r in records if r.status == "completed"]
    upcoming = [r for r in records if r.status == "upcoming"]
    print(f"Found {len(completed)} completed, {len(upcoming)} upcoming matches across "
          f"{len(set(r.tournament for r in records))} tournaments")
    print("\nSample completed:")
    for r in completed[:5]:
        print(f"  [{r.tournament}] {r.round}: {r.p1_name} d. {r.p2_name} {r.score}")
    print("\nSample upcoming:")
    for r in upcoming[:5]:
        print(f"  [{r.tournament}] {r.round}: {r.p1_name} vs {r.p2_name}")
