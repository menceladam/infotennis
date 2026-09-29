"""Best-effort surface lookup for a currently-live tournament by name.

tennisabstract.com's live pages don't include surface directly. Challenger
venues rarely change surface year to year, so the most recent historical
edition of a same-named tournament is a reasonable proxy. Falls back to
Hard (the most common Challenger surface) when no history is found, and
flags that it did so.
"""
import pandas as pd

# Manual corrections for tournaments where the historical-lookup heuristic
# below got it wrong -- confirmed directly from a live odds site screenshot
# rather than guessed. Bari's most recent match in our data was from 2021
# (Hard), but the 2026 edition is actually Clay; a 5-year-old single data
# point isn't reliable enough to trust blindly.
CONFIRMED_SURFACES = {
    "bari": "Clay",  # confirmed via user screenshot, 2026-09-29
}

# Beyond this age, a single historical match is treated as too stale to
# trust outright (a venue/surface can change between editions).
MAX_RELIABLE_AGE_YEARS = 3


def lookup_surface(matches: pd.DataFrame, tournament_name: str, level: str = "challenger") -> tuple[str, bool]:
    """Returns (surface, is_fallback). is_fallback=True means either no
    historical match was found, or the only match found is old enough
    that the surface may have changed since -- Hard is used as the
    default guess in both cases."""
    for key, surface in CONFIRMED_SURFACES.items():
        if key in tournament_name.lower():
            return surface, False

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

    age_years = (matches["tourney_date"].max() - latest["tourney_date"]).days / 365.25
    if age_years > MAX_RELIABLE_AGE_YEARS:
        return "Hard", True

    return surface, False
