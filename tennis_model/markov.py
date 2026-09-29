"""Point -> game -> tiebreak -> set -> match Markov model for tennis.

Standard iid-points model (Klaassen & Magnus-style): each player has a
fixed probability of winning a point on their own serve. Game, tiebreak,
set, and match win probabilities are derived exactly (no simulation)
from those two numbers via recursion with memoization.
"""
from functools import lru_cache


@lru_cache(maxsize=None)
def game_win_prob(p: float) -> float:
    """Probability the server wins a service game, given p = prob of
    winning a single point on serve."""
    return _game_recur(p, 0, 0)


@lru_cache(maxsize=None)
def _game_recur(p: float, i: int, j: int) -> float:
    q = 1 - p
    if i >= 4 and i - j >= 2:
        return 1.0
    if j >= 4 and j - i >= 2:
        return 0.0
    if i == j and i >= 3:
        return p * p / (p * p + q * q)
    return p * _game_recur(p, i + 1, j) + q * _game_recur(p, i, j + 1)


def _tiebreak_server_is_a(point_num: int) -> bool:
    """Standard tiebreak serving order: player A serves point 1 only,
    then serve alternates every 2 points (B: 2-3, A: 4-5, B: 6-7, ...)."""
    if point_num == 1:
        return True
    pair_index = (point_num - 2) // 2
    return pair_index % 2 == 1  # odd pair -> back to A


@lru_cache(maxsize=None)
def tiebreak_win_prob(p_a_serve: float, p_b_serve: float, i: int = 0, j: int = 0) -> float:
    """Probability A wins a first-to-7-win-by-2 tiebreak, given point-win
    probabilities on each player's own serve. A serves point 1."""
    if i >= 7 and i - j >= 2:
        return 1.0
    if j >= 7 and j - i >= 2:
        return 0.0
    if i == j and i >= 6:
        # From any tied state i==j, points_so_far = 2i is always even, and
        # the next two points (2i+1, 2i+2) are always served by different
        # players (verified for the standard 1-then-alternate-every-2
        # pattern), so this reduces to a stationary 2-point decider.
        return _tiebreak_deuce(p_a_serve, p_b_serve, i + j)
    point_num = i + j + 1
    server_is_a = _tiebreak_server_is_a(point_num)
    p = p_a_serve if server_is_a else p_b_serve
    p_a_wins_point = p if server_is_a else (1 - p)
    return p_a_wins_point * tiebreak_win_prob(p_a_serve, p_b_serve, i + 1, j) + (
        1 - p_a_wins_point
    ) * tiebreak_win_prob(p_a_serve, p_b_serve, i, j + 1)


@lru_cache(maxsize=None)
def _tiebreak_deuce(p_a_serve: float, p_b_serve: float, points_so_far: int) -> float:
    """P(A eventually wins tiebreak) from a tied-at->=6 state, where the
    next two points are served one each (order depends on parity)."""
    point_num = points_so_far + 1
    first_server_is_a = _tiebreak_server_is_a(point_num)
    p1 = p_a_serve if first_server_is_a else p_b_serve
    p1_a = p1 if first_server_is_a else (1 - p1)
    second_server_is_a = not first_server_is_a
    p2 = p_a_serve if second_server_is_a else p_b_serve
    p2_a = p2 if second_server_is_a else (1 - p2)

    # Both points to A -> A wins tiebreak outright. Both to B -> B wins
    # outright. Split -> back to an equivalent tied state (self-similar).
    both_a = p1_a * p2_a
    both_b = (1 - p1_a) * (1 - p2_a)
    split = 1 - both_a - both_b
    if split >= 1.0:
        return 0.5
    # x = both_a + split * x  =>  x = both_a / (1 - split)
    return both_a / (1 - split)


@lru_cache(maxsize=None)
def set_win_prob(p_a_serve: float, p_b_serve: float, a_serves_first: bool = True, i: int = 0, j: int = 0) -> float:
    """Probability A wins a set (to 6, win by 2, tiebreak at 6-6)."""
    if i >= 6 and i - j >= 2:
        return 1.0
    if j >= 6 and j - i >= 2:
        return 0.0
    if i == 6 and j == 6:
        return tiebreak_win_prob(p_a_serve, p_b_serve)

    games_played = i + j
    server_is_a = a_serves_first if games_played % 2 == 0 else not a_serves_first
    p_server_wins_game = p_a_serve if server_is_a else p_b_serve
    p_game = game_win_prob(p_server_wins_game)
    p_a_wins_this_game = p_game if server_is_a else (1 - p_game)

    return p_a_wins_this_game * set_win_prob(p_a_serve, p_b_serve, a_serves_first, i + 1, j) + (
        1 - p_a_wins_this_game
    ) * set_win_prob(p_a_serve, p_b_serve, a_serves_first, i, j + 1)


@lru_cache(maxsize=None)
def match_win_prob(
    p_a_serve: float,
    p_b_serve: float,
    best_of: int = 3,
    si: int = 0,
    sj: int = 0,
    a_serves_first: bool = True,
) -> float:
    """Probability A wins the match (best of 3 or 5 sets)."""
    sets_needed = best_of // 2 + 1
    if si == sets_needed:
        return 1.0
    if sj == sets_needed:
        return 0.0

    p_a_set = set_win_prob(p_a_serve, p_b_serve, a_serves_first)
    next_a_serves_first = not a_serves_first  # opponent opens next set

    return p_a_set * match_win_prob(p_a_serve, p_b_serve, best_of, si + 1, sj, next_a_serves_first) + (
        1 - p_a_set
    ) * match_win_prob(p_a_serve, p_b_serve, best_of, si, sj + 1, next_a_serves_first)
