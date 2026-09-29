"""Prediction tracking log: records what the model called before a match,
then checks it against the actual result once the match finishes.

This is the free, no-odds-needed way to build a real track record of
model accuracy over time (separate from whether any given price was
+EV against a bookmaker, which still needs a manual price check).
"""
import datetime as dt
from pathlib import Path

import pandas as pd

from tennis_model.live_scraper import fetch_all_current_challenger_matches
from tennis_model.name_match import normalize

LOG_PATH = Path(__file__).resolve().parent.parent / "data" / "predictions_log.csv"

COLUMNS = [
    "id", "logged_at", "tournament", "round", "surface",
    "p1", "p2", "p1_matches", "p2_matches",
    "win_prob_p1", "favorite", "target_line", "target_odds",
    "status", "actual_winner", "actual_score",
    "favorite_margin", "prediction_correct", "line_covered",
]


def _load_log() -> pd.DataFrame:
    if LOG_PATH.exists():
        return pd.read_csv(LOG_PATH)
    return pd.DataFrame(columns=COLUMNS)


def log_predictions(rows: list[dict]) -> None:
    """Append new pending predictions. Each row needs at least:
    tournament, round, surface, p1, p2, p1_matches, p2_matches,
    win_prob_p1, favorite, target_line, target_odds."""
    log = _load_log()
    next_id = (log["id"].max() + 1) if len(log) else 1
    now = dt.datetime.now().isoformat(timespec="seconds")

    new_rows = []
    for i, row in enumerate(rows):
        new_rows.append({
            "id": next_id + i,
            "logged_at": now,
            "status": "pending",
            "actual_winner": None, "actual_score": None,
            "favorite_margin": None, "prediction_correct": None, "line_covered": None,
            **row,
        })

    updated = pd.concat([log, pd.DataFrame(new_rows)], ignore_index=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    updated.to_csv(LOG_PATH, index=False)
    print(f"Logged {len(new_rows)} predictions to {LOG_PATH} (ids {next_id}-{next_id + len(rows) - 1})")


def _parse_final_margin(score: str, winner_is_p1: bool, favorite_is_p1: bool) -> float | None:
    """Parse a score string like '6-4 3-6 6-2' into (winner_games - loser_games),
    then flip to be favorite-relative. Returns None if unparseable (RET/W-O
    with no games, etc.)."""
    sets = score.replace("RET", "").replace("W/O", "").replace("DEF", "").strip().split()
    winner_games = loser_games = 0
    for s in sets:
        s = s.split("(")[0]  # drop tiebreak point counts
        if "-" not in s:
            continue
        try:
            w, l = s.split("-")
            winner_games += int(w)
            loser_games += int(l)
        except ValueError:
            continue
    if winner_games == 0 and loser_games == 0:
        return None
    winner_margin = winner_games - loser_games
    favorite_won = winner_is_p1 == favorite_is_p1
    return winner_margin if favorite_won else -winner_margin


def resolve_pending() -> int:
    """Check pending predictions against live completed results. Returns
    the number resolved."""
    log = _load_log()
    pending = log[log["status"] == "pending"]
    if pending.empty:
        print("No pending predictions to resolve.")
        return 0

    print("Fetching current completed results...")
    records = fetch_all_current_challenger_matches()
    completed = [r for r in records if r.status == "completed"]

    # Index completed results by (tournament, normalized name set) for lookup.
    by_key = {}
    for r in completed:
        key = (r.tournament, frozenset([normalize(r.p1_name), normalize(r.p2_name)]))
        by_key[key] = r

    resolved_count = 0
    for idx, row in pending.iterrows():
        key = (row["tournament"], frozenset([normalize(row["p1"]), normalize(row["p2"])]))
        match = by_key.get(key)
        if match is None:
            continue

        winner_is_p1 = normalize(match.winner_name) == normalize(row["p1"])
        favorite_is_p1 = row["favorite"] == row["p1"]
        margin = _parse_final_margin(match.score, winner_is_p1, favorite_is_p1)

        log.at[idx, "status"] = "resolved"
        log.at[idx, "actual_winner"] = match.winner_name
        log.at[idx, "actual_score"] = match.score
        log.at[idx, "favorite_margin"] = margin
        log.at[idx, "prediction_correct"] = bool(winner_is_p1 == favorite_is_p1)
        if margin is not None:
            log.at[idx, "line_covered"] = bool(margin > row["target_line"])
        resolved_count += 1

    log.to_csv(LOG_PATH, index=False)
    print(f"Resolved {resolved_count} of {len(pending)} pending predictions.")
    return resolved_count


def summary() -> None:
    log = _load_log()
    resolved = log[log["status"] == "resolved"]
    pending = log[log["status"] == "pending"]
    print(f"Total logged: {len(log)}  |  Resolved: {len(resolved)}  |  Pending: {len(pending)}")
    if len(resolved):
        acc = resolved["prediction_correct"].mean()
        print(f"Favorite-pick accuracy: {acc:.1%} ({resolved['prediction_correct'].sum()}/{len(resolved)})")
        line_resolved = resolved.dropna(subset=["line_covered"])
        if len(line_resolved):
            cover_rate = line_resolved["line_covered"].mean()
            print(f"Target-line cover rate: {cover_rate:.1%} ({line_resolved['line_covered'].sum()}/{len(line_resolved)})")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "resolve":
        resolve_pending()
    summary()
