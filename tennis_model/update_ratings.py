"""Daily pipeline: re-pull latest results, recompute Elo, write ratings snapshot.

Re-downloads the current year (plus prior year, in case of late corrections
to matches near a year boundary) for all three levels, then reprocesses the
full match history and writes a ratings snapshot to data/ratings_latest.csv.

Cheap to run fully from scratch every time: ~230k matches process in ~15s.
"""
import datetime as dt
from pathlib import Path

import pandas as pd

from scripts.download_tennismylife import download_year, LEVELS
from tennis_model.data_loader import load_all_matches
from tennis_model.elo import EloEngine

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "ratings_latest.csv"


def refresh_recent_years():
    current_year = dt.date.today().year
    for level in LEVELS:
        for year in (current_year - 1, current_year):
            ok = download_year(level, year)
            print(f"  refreshed {level} {year}: {'OK' if ok else 'missing'}")


def build_name_lookup(matches: pd.DataFrame) -> dict:
    """Most recent known name for each player_id."""
    w = matches[["winner_id", "winner_name", "tourney_date"]].rename(
        columns={"winner_id": "player_id", "winner_name": "name"}
    )
    l = matches[["loser_id", "loser_name", "tourney_date"]].rename(
        columns={"loser_id": "player_id", "loser_name": "name"}
    )
    both = pd.concat([w, l], ignore_index=True).sort_values("tourney_date")
    latest = both.drop_duplicates("player_id", keep="last")
    return dict(zip(latest["player_id"], latest["name"]))


def main():
    print("Refreshing current + prior year data from TennisMyLife...")
    refresh_recent_years()

    print("\nLoading full match history...")
    matches = load_all_matches()

    print("Processing Elo ratings...")
    engine = EloEngine()
    engine.process(matches)

    ratings = engine.ratings_table()
    names = build_name_lookup(matches)
    ratings.insert(1, "name", ratings["player_id"].map(names))
    ratings.insert(0, "as_of", dt.date.today().isoformat())

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ratings.to_csv(OUTPUT_PATH, index=False)
    print(f"\nWrote {len(ratings):,} player ratings to {OUTPUT_PATH}")
    print(f"Most recent match in dataset: {matches['tourney_date'].max().date()}")
    print(ratings.head(10)[["name", "overall", "overall_matches"]].to_string(index=False))


if __name__ == "__main__":
    main()
