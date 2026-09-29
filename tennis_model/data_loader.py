"""Load and merge tour/challenger/qualifying match CSVs into one chronological log."""
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

LEVEL_DIRS = {
    "tour": "tour",
    "challenger": "challenger",
    "quali": "quali",
}

DTYPES = {
    "tourney_id": str,
    "winner_id": str,
    "loser_id": str,
    "score": str,
    "round": str,
    "surface": str,
    "tourney_level": str,
}


def _load_level(level: str) -> pd.DataFrame:
    folder = DATA_DIR / LEVEL_DIRS[level]
    frames = []
    for path in sorted(folder.glob("*.csv")):
        df = pd.read_csv(path, dtype=DTYPES, low_memory=False)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["level"] = level
    return out


def load_all_matches() -> pd.DataFrame:
    """Load tour + challenger + quali matches into one chronologically sorted frame."""
    frames = [_load_level(level) for level in LEVEL_DIRS]
    matches = pd.concat(frames, ignore_index=True)

    matches["tourney_date"] = pd.to_datetime(matches["tourney_date"], format="%Y%m%d")
    matches["match_num"] = pd.to_numeric(matches["match_num"], errors="coerce").fillna(0)

    matches = matches.dropna(subset=["winner_id", "loser_id"])

    matches = matches.sort_values(
        ["tourney_date", "tourney_id", "match_num"]
    ).reset_index(drop=True)

    matches["retirement"] = matches["score"].fillna("").str.contains(
        "RET|W/O|DEF|ABN", case=False, regex=True
    )

    return matches


if __name__ == "__main__":
    df = load_all_matches()
    print(f"Total matches: {len(df):,}")
    print(df["level"].value_counts())
    print(f"Date range: {df['tourney_date'].min()} to {df['tourney_date'].max()}")
    print(f"Retirements/walkovers: {df['retirement'].sum():,} ({df['retirement'].mean():.1%})")
