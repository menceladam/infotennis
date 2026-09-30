"""CLI: predict a Challenger (or tour) matchup from the current ratings
snapshot, producing both a win probability and a game-handicap projection.

Usage:
    python -m tennis_model.predict "Player A" "Player B" --surface Hard
"""
import argparse
from pathlib import Path

import pandas as pd

from tennis_model.data_loader import load_all_matches
from tennis_model.elo import SURFACE_WEIGHT
from tennis_model.handicap import empirical_baseline_serve_rate, project_handicap, format_favorite_line
from tennis_model.markov import match_win_prob

RATINGS_PATH = Path(__file__).resolve().parent.parent / "data" / "ratings_latest.csv"

# See MIN_MATCHES in daily_report.py -- same reasoning, confirmed on a real
# match (Perot/Ostapenkov) where a sub-25-match rating produced a false edge.
MIN_MATCHES = 25


def find_player(ratings: pd.DataFrame, query: str) -> pd.Series:
    matches = ratings[ratings["name"].str.contains(query, case=False, na=False)]
    if matches.empty:
        raise SystemExit(f"No player found matching '{query}'")
    if len(matches) > 1:
        options = "\n".join(f"  - {n}" for n in matches["name"].tolist())
        raise SystemExit(f"Ambiguous query '{query}', matches:\n{options}")
    return matches.iloc[0]


def blended_rating(row: pd.Series, surface: str) -> float:
    surf_col = f"{surface.lower()}_rating"
    if surf_col not in row or row[f"{surface.lower()}_matches"] == 0:
        return row["overall"]
    return SURFACE_WEIGHT * row[surf_col] + (1 - SURFACE_WEIGHT) * row["overall"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("player_a")
    parser.add_argument("player_b")
    parser.add_argument("--surface", default="Hard", choices=["Hard", "Clay", "Grass", "Carpet"])
    parser.add_argument("--level", default="challenger", choices=["challenger", "tour", "quali"])
    parser.add_argument("--best-of", type=int, default=3)
    parser.add_argument("--sims", type=int, default=20000)
    args = parser.parse_args()

    if not RATINGS_PATH.exists():
        raise SystemExit(f"{RATINGS_PATH} not found — run `python -m tennis_model.update_ratings` first")
    ratings = pd.read_csv(RATINGS_PATH)

    a = find_player(ratings, args.player_a)
    b = find_player(ratings, args.player_b)

    a_blend = blended_rating(a, args.surface)
    b_blend = blended_rating(b, args.surface)
    win_prob_a = 1 / (1 + 10 ** ((b_blend - a_blend) / 400))

    print(f"\n{a['name']} (rating {a_blend:.0f}, {int(a['overall_matches'])} career matches) vs "
          f"{b['name']} (rating {b_blend:.0f}, {int(b['overall_matches'])} career matches)")
    print(f"Surface: {args.surface}  |  Elo match win probability: {a['name']}={win_prob_a:.1%}  {b['name']}={1-win_prob_a:.1%}")

    if a["overall_matches"] < MIN_MATCHES or b["overall_matches"] < MIN_MATCHES:
        thin = a["name"] if a["overall_matches"] < MIN_MATCHES else b["name"]
        print(f"\n*** WARNING: '{thin}' has under {MIN_MATCHES} career matches -- rating is still close to the "
              f"1500 starting point and hasn't stabilized. Do not treat this as a reliable prediction. ***")

    print("\nLoading match history to compute empirical serve baseline...")
    matches = load_all_matches()
    base_rate = empirical_baseline_serve_rate(matches, level=args.level, surface=args.surface)
    print(f"Baseline serve-points-won rate ({args.level}, {args.surface}): {base_rate:.3f}")

    proj = project_handicap(win_prob_a, base_rate, best_of=args.best_of, n_sims=args.sims)

    print(f"\nImplied serve-points-won: {a['name']}={proj.p_a_serve:.3f}  {b['name']}={proj.p_b_serve:.3f}")
    print(f"Simulated match win prob (sanity check): {proj.match_win_prob_a:.1%} (should be close to {win_prob_a:.1%})")
    print(f"Mean game margin: {a['name']} by {proj.mean_game_margin:+.2f} games (median {proj.median_game_margin:+.0f})")
    print("\nProbability A wins by more than N games (i.e. covers a -N.5 handicap):")
    for line, prob in sorted(proj.prob_cover.items()):
        print(f"  {format_favorite_line(a['name'], line)}: {prob:.1%}")

    if proj.set_score_probs:
        from tennis_model.odds import decimal_odds
        print("\nCorrect set score:")
        for score, key in [("2-0", "2-0"), ("2-1", "2-1"), ("0-2", "0-2"), ("1-2", "1-2")]:
            label = f"{a['name']} {score}" if score in ("2-0", "2-1") else f"{b['name']} {'2-0' if score=='0-2' else '2-1'}"
            prob = proj.set_score_probs[key]
            print(f"  {label}: fair odds {decimal_odds(prob):.2f} ({prob:.1%})")


if __name__ == "__main__":
    main()
