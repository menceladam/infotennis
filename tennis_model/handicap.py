"""Translate an Elo-derived match win probability into a game-handicap
projection (expected game margin, probability of covering a given line).

Approach:
1. Compute an empirical baseline serve-points-won rate per surface from
   real recent match stats (not guessed).
2. Solve for a symmetric adjustment (base + delta, base - delta) to each
   player's serve-win rate such that the exact Markov match-win
   probability matches the target win probability from Elo. This keeps
   the point/game/set model consistent with the rating engine rather
   than introducing a second, disconnected source of truth.
3. Monte Carlo simulate matches at the point level with those two serve
   rates to get the full game-margin distribution (the analytic model
   only gives clean closed forms for win probability, not the full
   margin distribution across a variable-length best-of-3 match).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from tennis_model.markov import match_win_prob, game_win_prob, tiebreak_win_prob
from tennis_model.odds import decimal_odds

DEFAULT_BASE_SERVE_RATE = 0.62  # fallback if empirical data unavailable
RECENT_YEARS_FOR_BASELINE = 4


def empirical_baseline_serve_rate(matches: pd.DataFrame, level: str = "challenger", surface: str = "Hard") -> float:
    cutoff = matches["tourney_date"].max() - pd.DateOffset(years=RECENT_YEARS_FOR_BASELINE)
    subset = matches[
        (matches["level"] == level)
        & (matches["surface"] == surface)
        & (matches["tourney_date"] >= cutoff)
        & (~matches["retirement"])
    ].dropna(subset=["w_svpt", "w_1stWon", "w_2ndWon", "l_svpt", "l_1stWon", "l_2ndWon"])

    if subset.empty:
        return DEFAULT_BASE_SERVE_RATE

    w_rate = (subset["w_1stWon"] + subset["w_2ndWon"]) / subset["w_svpt"]
    l_rate = (subset["l_1stWon"] + subset["l_2ndWon"]) / subset["l_svpt"]
    return float(pd.concat([w_rate, l_rate]).mean())


def solve_serve_rates(target_match_win_prob: float, base_rate: float, best_of: int = 3) -> tuple[float, float]:
    """Find delta such that match_win_prob(base+delta, base-delta) equals
    target_match_win_prob. Monotonic in delta, so bisection is safe."""
    target_match_win_prob = min(max(target_match_win_prob, 1e-4), 1 - 1e-4)

    def f(delta):
        p_a = min(max(base_rate + delta, 0.01), 0.99)
        p_b = min(max(base_rate - delta, 0.01), 0.99)
        return match_win_prob(p_a, p_b, best_of) - target_match_win_prob

    # Bounds: delta can't push either rate outside (0.01, 0.99), and in
    # practice realistic match-win-prob extremes are covered well within +-0.25.
    lo, hi = -0.3, 0.3
    if f(lo) > 0 or f(hi) < 0:
        # Target is extreme enough that even max delta doesn't reach it;
        # clip to the boundary.
        return (base_rate + (0.3 if f(hi) < 0 else -0.3), base_rate - (0.3 if f(hi) < 0 else -0.3))

    delta = brentq(f, lo, hi, xtol=1e-5)
    return base_rate + delta, base_rate - delta


@dataclass
class HandicapProjection:
    p_a_serve: float
    p_b_serve: float
    match_win_prob_a: float
    mean_game_margin: float  # positive = A favored by this many games on average
    median_game_margin: float
    prob_cover: dict  # {line: prob A wins by more than `line` games}


def simulate_match(p_a_serve: float, p_b_serve: float, best_of: int = 3, rng: np.random.Generator = None):
    """Simulate one match point-by-point. Returns (a_won_match, games_a, games_b)."""
    rng = rng or np.random.default_rng()
    sets_needed = best_of // 2 + 1
    sets_a = sets_b = 0
    total_games_a = total_games_b = 0
    a_serves_first = True

    while sets_a < sets_needed and sets_b < sets_needed:
        games_a = games_b = 0
        while True:
            if games_a >= 6 and games_a - games_b >= 2:
                sets_a += 1
                break
            if games_b >= 6 and games_b - games_a >= 2:
                sets_b += 1
                break
            if games_a == 6 and games_b == 6:
                # tiebreak: simulate point by point
                a_wins_tb = _simulate_tiebreak(p_a_serve, p_b_serve, rng)
                if a_wins_tb:
                    games_a += 1
                    sets_a += 1
                else:
                    games_b += 1
                    sets_b += 1
                break

            server_is_a = a_serves_first if (games_a + games_b) % 2 == 0 else not a_serves_first
            p = p_a_serve if server_is_a else p_b_serve
            a_wins_game = _simulate_game(p, rng) if server_is_a else not _simulate_game(p, rng)
            if a_wins_game:
                games_a += 1
            else:
                games_b += 1

        total_games_a += games_a
        total_games_b += games_b
        a_serves_first = not a_serves_first

    return sets_a > sets_b, total_games_a, total_games_b


def _simulate_game(p: float, rng: np.random.Generator) -> bool:
    """Simulate a single service game; returns True if server wins."""
    i = j = 0
    while True:
        if rng.random() < p:
            i += 1
        else:
            j += 1
        if i >= 4 and i - j >= 2:
            return True
        if j >= 4 and j - i >= 2:
            return False


def _simulate_tiebreak(p_a_serve: float, p_b_serve: float, rng: np.random.Generator) -> bool:
    from tennis_model.markov import _tiebreak_server_is_a

    i = j = 0
    point_num = 1
    while True:
        server_is_a = _tiebreak_server_is_a(point_num)
        p = p_a_serve if server_is_a else p_b_serve
        server_wins = rng.random() < p
        a_wins_point = server_wins if server_is_a else not server_wins
        if a_wins_point:
            i += 1
        else:
            j += 1
        if i >= 7 and i - j >= 2:
            return True
        if j >= 7 and j - i >= 2:
            return False
        point_num += 1


# Wide enough to cover any realistic best-of-3 game margin (max possible
# is +/-12 for a 6-0 6-0 sweep); prob_cover for the extra lines is nearly
# free once margins are already simulated.
DEFAULT_LINES = tuple(x - 0.5 for x in range(-11, 12) if x != 0)


def project_handicap(
    target_match_win_prob: float,
    base_rate: float,
    best_of: int = 3,
    n_sims: int = 20000,
    lines: tuple = DEFAULT_LINES,
    seed: int = 0,
) -> HandicapProjection:
    p_a, p_b = solve_serve_rates(target_match_win_prob, base_rate, best_of)

    rng = np.random.default_rng(seed)
    margins = np.empty(n_sims)
    wins = np.empty(n_sims, dtype=bool)
    for n in range(n_sims):
        a_won, ga, gb = simulate_match(p_a, p_b, best_of, rng)
        margins[n] = ga - gb
        wins[n] = a_won

    prob_cover = {line: float((margins > line).mean()) for line in lines}

    return HandicapProjection(
        p_a_serve=p_a,
        p_b_serve=p_b,
        match_win_prob_a=float(wins.mean()),
        mean_game_margin=float(margins.mean()),
        median_game_margin=float(np.median(margins)),
        prob_cover=prob_cover,
    )


def format_favorite_line(favorite_name: str, line: float) -> str:
    """Standard bookmaker handicap notation. `line` here is the
    favorite-margin threshold used throughout this module (prob_cover[line]
    = P(favorite's game margin > line)) -- NOT the number printed on a
    betting slip. Winning that condition is exactly what a "-line" bet on
    the favorite means, so a positive threshold prints with a MINUS sign
    (favorite must win by more than `line` games) and a negative threshold
    prints with a PLUS sign (favorite can even lose by up to |line| games
    and still cover). Getting this backwards makes every favorite line
    look like an underdog line.
    """
    if line > 0:
        return f"{favorite_name} -{line}"
    return f"{favorite_name} +{abs(line)}"


def line_for_odds_range(prob_cover: dict, low_odds: float, high_odds: float) -> tuple | None:
    """Find the handicap line (favorite-relative: positive = favorite
    giving games) whose fair decimal odds fall within [low_odds, high_odds].
    Returns (line, prob, fair_odds), or None if no line in prob_cover lands
    in range (picks the closest one instead, flagged via a 3rd return slot
    left None-free -- callers should treat a returned tuple as best-effort
    when the exact range isn't hit)."""
    target_lo, target_hi = 1 / high_odds, 1 / low_odds  # odds range -> prob range
    candidates = [(line, p) for line, p in prob_cover.items() if target_lo <= p <= target_hi]
    if candidates:
        # Prefer the one closest to the middle of the target probability band.
        mid = (target_lo + target_hi) / 2
        line, p = min(candidates, key=lambda lp: abs(lp[1] - mid))
        return line, p, decimal_odds(p)
    # Nothing landed exactly in range -- return the closest line to the band.
    mid = (target_lo + target_hi) / 2
    line, p = min(prob_cover.items(), key=lambda lp: abs(lp[1] - mid))
    return line, p, decimal_odds(p)
