"""Daily report: what Challenger tournaments are live right now, upcoming
matches with model predictions, and recent results.

This is a READ of model confidence, not verified betting value -- there
is no odds feed wired in yet, so "most confident" and "positive EV" are
not the same thing until a real market price is checked against it.
"""
from collections import defaultdict

import pandas as pd

from tennis_model.data_loader import load_all_matches
from tennis_model.elo import SURFACE_WEIGHT
from tennis_model.handicap import empirical_baseline_serve_rate, project_handicap, line_for_odds_range
from tennis_model.live_scraper import fetch_all_current_challenger_matches
from tennis_model.name_match import NameMatcher
from tennis_model.odds import decimal_odds, american_odds
from tennis_model.surface_lookup import lookup_surface

RATINGS_PATH = "data/ratings_latest.csv"


def blended_rating(row: pd.Series, surface: str) -> float:
    surf_col = f"{surface.lower()}_rating"
    if surf_col not in row or row.get(f"{surface.lower()}_matches", 0) == 0:
        return row["overall"]
    return SURFACE_WEIGHT * row[surf_col] + (1 - SURFACE_WEIGHT) * row["overall"]


def main():
    print("Loading ratings and match history...")
    ratings = pd.read_csv(RATINGS_PATH)
    matcher = NameMatcher(ratings)
    matches = load_all_matches()

    print("Fetching live Challenger tournaments and matches...")
    records = fetch_all_current_challenger_matches()

    by_tournament = defaultdict(list)
    for r in records:
        by_tournament[r.tournament].append(r)

    predictions = []  # for the "best bets" ranking at the end

    for tourney, recs in sorted(by_tournament.items()):
        surface, is_fallback = lookup_surface(matches, tourney)
        base_rate = empirical_baseline_serve_rate(matches, "challenger", surface)
        flag = " (surface guessed, no history found)" if is_fallback else ""

        print(f"\n{'='*70}")
        print(f"{tourney}  [{surface}{flag}]")
        print(f"{'='*70}")

        completed = [r for r in recs if r.status == "completed"]
        upcoming = [r for r in recs if r.status == "upcoming"]

        if completed:
            print(f"\nRecent results ({len(completed)}):")
            for r in sorted(completed, key=lambda x: x.round)[-5:]:
                print(f"  {r.round}: {r.p1_name} d. {r.p2_name}  {r.score}")

        if upcoming:
            print(f"\nUpcoming ({len(upcoming)}):")
            for r in upcoming:
                m1 = matcher.match(r.p1_name)
                m2 = matcher.match(r.p2_name)
                if m1.row is None or m2.row is None:
                    missing = r.p1_name if m1.row is None else r.p2_name
                    print(f"  {r.round}: {r.p1_name} vs {r.p2_name}  -- no rating for '{missing}', skipped")
                    continue

                rating1 = blended_rating(m1.row, surface)
                rating2 = blended_rating(m2.row, surface)
                win_prob_1 = 1 / (1 + 10 ** ((rating2 - rating1) / 400))
                odds1, odds2 = decimal_odds(win_prob_1), decimal_odds(1 - win_prob_1)

                fav_name, fav_prob = (r.p1_name, win_prob_1) if win_prob_1 >= 0.5 else (r.p2_name, 1 - win_prob_1)

                # Handicap line whose fair odds land in the 1.80-2.00 band,
                # relative to the favorite -- built for a light Monte Carlo
                # run since this runs for every match, not just the top 10.
                proj = project_handicap(win_prob_1, base_rate, n_sims=4000)
                fav_is_p1 = fav_name == r.p1_name
                fav_cover = proj.prob_cover if fav_is_p1 else {
                    -line: 1 - prob for line, prob in proj.prob_cover.items()
                }
                target_line, target_prob, target_odds = line_for_odds_range(fav_cover, 1.80, 2.00)
                in_range = 1.80 <= target_odds <= 2.00
                range_note = "" if in_range else " (outside 1.80-2.00, closest available)"
                sign = "+" if target_line > 0 else ""

                print(f"  {r.round}: {r.p1_name} {win_prob_1:.0%} (fair {odds1:.2f}) vs "
                      f"{r.p2_name} {1-win_prob_1:.0%} (fair {odds2:.2f})"
                      f"  -> favorite: {fav_name} {fav_prob:.0%}")
                print(f"      target handicap: {fav_name} {sign}{target_line} games "
                      f"-> fair odds {target_odds:.2f} ({target_prob:.1%}){range_note}")

                predictions.append({
                    "tournament": tourney, "round": r.round, "surface": surface,
                    "p1": r.p1_name, "p2": r.p2_name,
                    "favorite": fav_name, "win_prob": fav_prob,
                    "win_prob_1": win_prob_1, "base_rate": base_rate,
                })

    if predictions:
        print(f"\n\n{'='*70}")
        print("MOST CONFIDENT PICKS -- fair (no-vig) odds from the model.")
        print("Bet only if Bet365's actual price is HIGHER (better payout) than")
        print("the 'fair' decimal odds below -- that gap is the edge. If Bet365")
        print("is lower/even, there's no value here even though the model likes it.")
        print("Set Bet365's odds format to Decimal (Account -> Settings -> Odds")
        print("Display) for a direct apples-to-apples comparison.")
        print(f"{'='*70}")
        top = sorted(predictions, key=lambda p: p["win_prob"], reverse=True)[:10]
        for p in top:
            proj = project_handicap(p["win_prob_1"], p["base_rate"], n_sims=5000)
            fav_is_p1 = p["favorite"] == p["p1"]
            underdog = p["p2"] if fav_is_p1 else p["p1"]
            margin = proj.mean_game_margin if fav_is_p1 else -proj.mean_game_margin
            fav_ml_decimal = decimal_odds(p["win_prob"])
            fav_ml_american = american_odds(p["win_prob"])

            print(f"\n  [{p['tournament']} {p['round']}] {p['favorite']} vs {underdog}  ({p['surface']})")
            print(f"    Moneyline -- {p['favorite']} to win: fair odds {fav_ml_decimal:.2f} "
                  f"(American {fav_ml_american:+d}), model prob {p['win_prob']:.1%}")
            # proj.prob_cover is relative to p1; flip to be relative to the
            # favorite when the favorite is p2, so both the sign and the
            # comparison against `margin` (already favorite-relative) line up.
            fav_cover = proj.prob_cover if fav_is_p1 else {
                -line: 1 - prob for line, prob in proj.prob_cover.items()
            }
            print(f"    Game handicap -- projected margin {p['favorite']} {margin:+.1f} games. "
                  f"Fair odds for {p['favorite']} to cover:")
            lines_by_dist = sorted(fav_cover.keys(), key=lambda L: abs(L - margin))[:4]
            for line in sorted(lines_by_dist):
                cover_prob = fav_cover[line]
                sign = "+" if line > 0 else ""
                print(f"      {p['favorite']} {sign}{line} games: fair odds {decimal_odds(cover_prob):.2f} "
                      f"({cover_prob:.1%})")


if __name__ == "__main__":
    main()
