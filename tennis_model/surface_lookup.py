"""Best-effort surface lookup for a currently-live tournament by name.

tennisabstract.com's live pages don't include surface directly. Challenger
venues rarely change surface year to year, so the most recent historical
edition of a same-named tournament is a reasonable proxy. Falls back to
Hard (the most common Challenger surface) when no history is found, and
flags that it did so.
"""
import pandas as pd


def lookup_surface(matches: pd.DataFrame, tournament_name: str, level: str = "challenger") -> tuple[str, bool]:
    """Returns (surface, is_fallback). is_fallback=True means no historical
    match was found and Hard was used as a default guess."""
    subset = matches[
        (matches["level"] == level)
        & matches["tourney_name"].str.contains(tournament_name, case=False, na=False, regex=False)
    ]
    if subset.empty:
        return "Hard", True
    latest = subset.sort_values("tourney_date").iloc[-1]
    surface = latest["surface"]
    if pd.isna(surface):
        return "Hard", True
    return surface, False
