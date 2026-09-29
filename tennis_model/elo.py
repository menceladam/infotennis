"""Elo rating engine for ATP tour/challenger/qualifying matches.

Each player gets an overall Elo rating plus a surface-specific rating
(Hard/Clay/Grass/Carpet). K-factor decays with experience per player,
following the standard tennis-Elo formula (as used by FiveThirtyEight
and Jeff Sackmann's own implementation):

    K_i = 250 / (matches_played_i + 5) ** 0.4

Ratings are processed strictly in chronological order so every
pre-match rating reflects only information available before that
match was played (no lookahead).
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

START_RATING = 1500.0
K_BASE = 250.0
K_OFFSET = 5.0
K_POWER = 0.4

SURFACES = ["Hard", "Clay", "Grass", "Carpet"]

# Multiplier applied to the rating-change magnitude based on tournament
# importance. Grand Slams carry the most signal (best-of-5, deepest
# fields); Challengers and qualifying carry the least.
LEVEL_WEIGHT = {
    "G": 1.5,   # Grand Slam
    "M": 1.25,  # Masters 1000
    "500": 1.1,
    "250": 1.0,
    "A": 1.0,
    "O": 1.25,  # Olympics
    "F": 1.0,
    "D": 1.0,   # Davis Cup
    "C": 0.9,   # Challenger
}
QUALIFYING_WEIGHT = 0.85
RETIREMENT_WEIGHT = 0.5

# Blend weight for surface-adjusted rating: blended = SURFACE_WEIGHT *
# surface_rating + (1 - SURFACE_WEIGHT) * overall_rating.
SURFACE_WEIGHT = 0.65


@dataclass
class PlayerState:
    overall: float = START_RATING
    overall_matches: int = 0
    surface: dict = field(default_factory=lambda: {s: START_RATING for s in SURFACES})
    surface_matches: dict = field(default_factory=lambda: {s: 0 for s in SURFACES})


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def k_factor(matches_played: int) -> float:
    return K_BASE / (matches_played + K_OFFSET) ** K_POWER


def blended_rating(state: PlayerState, surface: str) -> float:
    if surface not in SURFACES:
        return state.overall
    return SURFACE_WEIGHT * state.surface[surface] + (1 - SURFACE_WEIGHT) * state.overall


class EloEngine:
    def __init__(self):
        self.players: dict[str, PlayerState] = {}

    def _get(self, player_id: str) -> PlayerState:
        if player_id not in self.players:
            self.players[player_id] = PlayerState()
        return self.players[player_id]

    def _match_weight(self, row) -> float:
        weight = LEVEL_WEIGHT.get(row["tourney_level"], 1.0)
        if row["level"] == "quali":
            weight *= QUALIFYING_WEIGHT
        if row["retirement"]:
            weight *= RETIREMENT_WEIGHT
        return weight

    def process(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Process matches in order, returning the match log annotated with
        pre-match ratings and blended win probability for the winner."""
        records = []
        for row in matches.itertuples(index=False):
            row = row._asdict() if hasattr(row, "_asdict") else dict(zip(matches.columns, row))
            w_state = self._get(row["winner_id"])
            l_state = self._get(row["loser_id"])
            surface = row["surface"]

            w_pre_overall, l_pre_overall = w_state.overall, l_state.overall
            w_pre_blend = blended_rating(w_state, surface)
            l_pre_blend = blended_rating(l_state, surface)
            win_prob_blend = expected_score(w_pre_blend, l_pre_blend)

            weight = self._match_weight(row)

            # Overall rating update
            e_w = expected_score(w_pre_overall, l_pre_overall)
            k_w = k_factor(w_state.overall_matches) * weight
            k_l = k_factor(l_state.overall_matches) * weight
            w_state.overall += k_w * (1 - e_w)
            l_state.overall += k_l * (0 - (1 - e_w))
            w_state.overall_matches += 1
            l_state.overall_matches += 1

            # Surface rating update
            if surface in SURFACES:
                e_w_surf = expected_score(w_state.surface[surface], l_state.surface[surface])
                k_w_surf = k_factor(w_state.surface_matches[surface]) * weight
                k_l_surf = k_factor(l_state.surface_matches[surface]) * weight
                w_state.surface[surface] += k_w_surf * (1 - e_w_surf)
                l_state.surface[surface] += k_l_surf * (0 - (1 - e_w_surf))
                w_state.surface_matches[surface] += 1
                l_state.surface_matches[surface] += 1

            records.append({
                "tourney_date": row["tourney_date"],
                "tourney_id": row["tourney_id"],
                "tourney_name": row["tourney_name"],
                "level": row["level"],
                "surface": surface,
                "round": row["round"],
                "winner_id": row["winner_id"],
                "loser_id": row["loser_id"],
                "winner_pre_overall": w_pre_overall,
                "loser_pre_overall": l_pre_overall,
                "winner_pre_blend": w_pre_blend,
                "loser_pre_blend": l_pre_blend,
                "win_prob_blend": win_prob_blend,
                "retirement": row["retirement"],
            })

        return pd.DataFrame.from_records(records)

    def ratings_table(self) -> pd.DataFrame:
        rows = []
        for pid, state in self.players.items():
            row = {
                "player_id": pid,
                "overall": state.overall,
                "overall_matches": state.overall_matches,
            }
            for s in SURFACES:
                row[f"{s.lower()}_rating"] = state.surface[s]
                row[f"{s.lower()}_matches"] = state.surface_matches[s]
            rows.append(row)
        return pd.DataFrame(rows).sort_values("overall", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    from tennis_model.data_loader import load_all_matches

    matches = load_all_matches()
    engine = EloEngine()
    log = engine.process(matches)

    ratings = engine.ratings_table()
    print("Top 15 by overall Elo (active + retired, all-time):")
    print(ratings.head(15).to_string(index=False))

    print(f"\nProcessed {len(log):,} matches, {len(ratings):,} players")
