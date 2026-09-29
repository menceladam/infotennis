"""Bridge player names from tennisabstract.com (live scraper) to the
TennisMyLife-derived ratings database (ratings_latest.csv).

The two sources both use "First [Middle] Last" naming but occasionally
render the same player differently (e.g. tennisabstract's "Igor Ribeiro
Marcondes" vs TennisMyLife's "Igor Marcondes"). This tries, in order:
1. exact match (case/accent-insensitive)
2. first-word + last-word match (handles dropped/added middle names)
3. fuzzy string match as a last resort, only when unambiguous

Anything that doesn't resolve cleanly is reported as unmatched rather
than guessed -- a wrong player match would silently corrupt a
prediction, which is worse than admitting "no rating available".
"""
import difflib
import unicodedata
from dataclasses import dataclass

import pandas as pd


def normalize(name: str) -> str:
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    name = name.lower().strip()
    name = name.replace(".", "").replace("'", "").replace("-", " ")
    name = " ".join(name.split())
    return name


@dataclass
class MatchResult:
    query: str
    row: pd.Series | None
    method: str  # "exact", "first_last", "fuzzy", "unmatched"


class NameMatcher:
    def __init__(self, ratings: pd.DataFrame):
        self.ratings = ratings.reset_index(drop=True)
        self.norm_names = self.ratings["name"].fillna("").map(normalize)
        self.exact_lookup = {}
        for idx, n in self.norm_names.items():
            self.exact_lookup.setdefault(n, []).append(idx)

    def match(self, query: str) -> MatchResult:
        nq = normalize(query)

        exact = self.exact_lookup.get(nq)
        if exact:
            if len(exact) == 1:
                return MatchResult(query, self.ratings.iloc[exact[0]], "exact")
            # Duplicate name entries happen when the source data assigns a
            # stray extra player_id to a single old match (a data-import
            # quirk, not a true career split -- confirmed by inspecting
            # several cases: one ID carries the full multi-year career,
            # the other has a single orphan match). Trust the dominant ID
            # only when it's not a close call; otherwise don't guess.
            rows = self.ratings.iloc[exact].sort_values("overall_matches", ascending=False)
            top, runner_up = rows.iloc[0], rows.iloc[1]
            if top["overall_matches"] >= 20 and top["overall_matches"] >= 5 * max(runner_up["overall_matches"], 1):
                return MatchResult(query, top, "exact_dominant")

        parts = nq.split()
        if len(parts) >= 2:
            first, last = parts[0], parts[-1]
            candidates = self.norm_names[
                self.norm_names.map(lambda n: n.split()[0] == first and n.split()[-1] == last if n else False)
            ]
            if len(candidates) == 1:
                return MatchResult(query, self.ratings.iloc[candidates.index[0]], "first_last")

        close = difflib.get_close_matches(nq, self.norm_names.tolist(), n=2, cutoff=0.88)
        if len(close) == 1:
            idx = self.norm_names[self.norm_names == close[0]].index[0]
            return MatchResult(query, self.ratings.iloc[idx], "fuzzy")

        return MatchResult(query, None, "unmatched")


if __name__ == "__main__":
    ratings = pd.read_csv("data/ratings_latest.csv")
    matcher = NameMatcher(ratings)

    from tennis_model.live_scraper import fetch_all_current_challenger_matches

    records = fetch_all_current_challenger_matches()
    all_names = sorted(set(n for r in records for n in (r.p1_name, r.p2_name)))

    counts = {"exact": 0, "exact_dominant": 0, "first_last": 0, "fuzzy": 0, "unmatched": 0}
    unmatched = []
    for name in all_names:
        result = matcher.match(name)
        counts[result.method] += 1
        if result.method == "unmatched":
            unmatched.append(name)
        elif result.method in ("first_last", "fuzzy"):
            print(f"  [{result.method}] '{name}' -> '{result.row['name']}'")

    print(f"\n{counts}")
    print(f"\nStill unmatched ({len(unmatched)}):")
    for n in unmatched:
        print(" ", n)
