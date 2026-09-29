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

# --- Recent form ---
# Tracks each player's exponentially-decayed over/under-performance vs
# their Elo-predicted win probability. Shrunk toward zero based on an
# "effective sample size" so a single upset doesn't swing it -- with no
# matches, effective_n=0 and the adjustment is exactly zero.
import math

FORM_HALF_LIFE_MATCHES = 12
FORM_DECAY = 0.5 ** (1 / FORM_HALF_LIFE_MATCHES)
FORM_SHRINKAGE_PRIOR = 6.0  # effective matches worth of "regress to zero"
FORM_ELO_SCALE = 400 / math.log(10)  # logit(p) -> Elo points
FORM_MAX_ELO = 100.0

# --- Head-to-head ---
# Beta-binomial shrinkage toward 50/50; with zero prior meetings this is
# exactly 0.5 (no adjustment). Most Challenger pairs have met 0-2 times,
# so a strong prior keeps this from overreacting to a single result.
H2H_PRIOR_ALPHA = 4.0
H2H_WEIGHT = 0.6  # extra global damping, tuned via backtest
H2H_MAX_ELO = 60.0


@dataclass
class PlayerState:
    overall: float = START_RATING
    overall_matches: int = 0
    surface: dict = field(default_factory=lambda: {s: START_RATING for s in SURFACES})
    surface_matches: dict = field(default_factory=lambda: {s: 0 for s in SURFACES})
    form_ewma: float = 0.0
    form_effective_n: float = 0.0


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def k_factor(matches_played: int) -> float:
    return K_BASE / (matches_played + K_OFFSET) ** K_POWER


def blended_rating(state: PlayerState, surface: str) -> float:
    if surface not in SURFACES:
        return state.overall
    return SURFACE_WEIGHT * state.surface[surface] + (1 - SURFACE_WEIGHT) * state.overall


def form_adjustment_elo(state: PlayerState) -> float:
    """Elo-point nudge from recent over/under-performance. Zero for a
    player with no match history (effective_n starts at 0)."""
    if state.form_effective_n <= 0:
        return 0.0
    shrink = state.form_effective_n / (state.form_effective_n + FORM_SHRINKAGE_PRIOR)
    shrunk_residual = state.form_ewma * shrink
    adjustment = shrunk_residual * FORM_ELO_SCALE
    return max(-FORM_MAX_ELO, min(FORM_MAX_ELO, adjustment))


def _update_form(state: PlayerState, residual: float) -> None:
    """residual = actual outcome (1 or 0) minus that match's predicted
    win probability for this player. Called after form_adjustment_elo
    has already been read for this match (no lookahead)."""
    prior_n = state.form_effective_n
    new_n = FORM_DECAY * prior_n + 1.0
    state.form_ewma = (FORM_DECAY * prior_n * state.form_ewma + residual) / new_n
    state.form_effective_n = new_n


def _pair_key(id_a: str, id_b: str) -> tuple:
    return tuple(sorted((id_a, id_b)))


def h2h_adjustment_elo(h2h_wins: dict, player_id: str, opponent_id: str) -> float:
    """Elo-point nudge from shrunk historical head-to-head record.
    Zero with no prior meetings (shrunk prob = exactly 0.5)."""
    key = _pair_key(player_id, opponent_id)
    record = h2h_wins.get(key, {})
    wins_p = record.get(player_id, 0)
    wins_o = record.get(opponent_id, 0)
    shrunk_prob = (wins_p + H2H_PRIOR_ALPHA) / (wins_p + wins_o + 2 * H2H_PRIOR_ALPHA)
    shrunk_prob = min(max(shrunk_prob, 1e-4), 1 - 1e-4)
    logit = math.log(shrunk_prob / (1 - shrunk_prob))
    adjustment = logit * FORM_ELO_SCALE * H2H_WEIGHT
    return max(-H2H_MAX_ELO, min(H2H_MAX_ELO, adjustment))


class EloEngine:
    def __init__(self):
        self.players: dict[str, PlayerState] = {}
        self.h2h_wins: dict[tuple, dict] = {}

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

            # Form- and H2H-adjusted variants, computed from pre-match state
            # only (no lookahead), for backtesting against the baseline.
            w_form_adj = form_adjustment_elo(w_state)
            l_form_adj = form_adjustment_elo(l_state)
            win_prob_form = expected_score(w_pre_blend + w_form_adj, l_pre_blend + l_form_adj)

            w_h2h_adj = h2h_adjustment_elo(self.h2h_wins, row["winner_id"], row["loser_id"])
            l_h2h_adj = h2h_adjustment_elo(self.h2h_wins, row["loser_id"], row["winner_id"])
            win_prob_h2h = expected_score(w_pre_blend + w_h2h_adj, l_pre_blend + l_h2h_adj)

            win_prob_full = expected_score(
                w_pre_blend + w_form_adj + w_h2h_adj, l_pre_blend + l_form_adj + l_h2h_adj
            )

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

            # Update form EWMA using this match's actual outcome vs the
            # blended-rating prediction (the adjustments above already used
            # the pre-update state, so this is safe to do now).
            _update_form(w_state, 1 - win_prob_blend)
            _update_form(l_state, win_prob_blend - 1)

            # Update H2H record for this pair.
            key = _pair_key(row["winner_id"], row["loser_id"])
            record = self.h2h_wins.setdefault(key, {})
            record[row["winner_id"]] = record.get(row["winner_id"], 0) + 1

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
                "win_prob_form": win_prob_form,
                "win_prob_h2h": win_prob_h2h,
                "win_prob_full": win_prob_full,
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
            row["form_adjustment"] = form_adjustment_elo(state)
            rows.append(row)
        return pd.DataFrame(rows).sort_values("overall", ascending=False).reset_index(drop=True)

    def h2h_lookup(self, player_id: str, opponent_id: str) -> float:
        return h2h_adjustment_elo(self.h2h_wins, player_id, opponent_id)


if __name__ == "__main__":
    from tennis_model.data_loader import load_all_matches

    matches = load_all_matches()
    engine = EloEngine()
    log = engine.process(matches)

    ratings = engine.ratings_table()
    print("Top 15 by overall Elo (active + retired, all-time):")
    print(ratings.head(15).to_string(index=False))

    print(f"\nProcessed {len(log):,} matches, {len(ratings):,} players")
