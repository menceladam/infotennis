"""Walk-forward validation of the Elo win-probability model.

Ratings for every match are computed using only matches that happened
strictly before it (data_loader sorts chronologically, elo.py updates
in order), so there is no lookahead. This script scores how well the
resulting *pre-match* probabilities predicted the actual outcome.

Early matches in the dataset are excluded from scoring (but not from
rating updates) because Elo needs a multi-year "burn-in" period before
ratings are informative -- scoring 2000-2002 would unfairly penalize
the model for a cold start it can't avoid.
"""
import numpy as np
import pandas as pd

from tennis_model.data_loader import load_all_matches
from tennis_model.elo import EloEngine
from tennis_model.calibration import fit_temperature, apply_temperature

EVAL_START = "2023-01-01"


def evaluate(log: pd.DataFrame, label: str, prob_col: str = "win_prob_blend") -> None:
    if log.empty:
        print(f"{label}: no matches")
        return
    p = log[prob_col].clip(1e-6, 1 - 1e-6)
    accuracy = (p > 0.5).mean()
    log_loss = -np.log(p).mean()
    brier = ((p - 1) ** 2).mean()
    print(f"{label}: n={len(log):,}  accuracy={accuracy:.4f}  log_loss={log_loss:.4f}  brier={brier:.4f}")


def calibration_table(log: pd.DataFrame, bins=10) -> pd.DataFrame:
    df = log.copy()
    df["bucket"] = pd.cut(df["win_prob_blend"], bins=np.linspace(0.5, 1.0, bins + 1))
    grouped = df.groupby("bucket", observed=True).agg(
        n=("win_prob_blend", "size"),
        predicted=("win_prob_blend", "mean"),
    )
    # Actual empirical win rate for the higher-probability side in each bucket:
    # since win_prob_blend is always P(designated winner), empirical rate = 1.0
    # by construction unless we also fold in the mirrored loser-side probability.
    # Build a symmetric frame: one row per player-in-match with prob and outcome.
    return grouped


def calibration_symmetric(log: pd.DataFrame, bins=10, prob_col: str = "win_prob_blend") -> pd.DataFrame:
    winners = pd.DataFrame({"prob": log[prob_col], "won": 1})
    losers = pd.DataFrame({"prob": 1 - log[prob_col], "won": 0})
    both = pd.concat([winners, losers], ignore_index=True)
    both["bucket"] = pd.cut(both["prob"], bins=np.linspace(0.0, 1.0, bins + 1))
    grouped = both.groupby("bucket", observed=True).agg(
        n=("prob", "size"),
        predicted=("prob", "mean"),
        actual=("won", "mean"),
    )
    return grouped


def main():
    matches = load_all_matches()
    engine = EloEngine()
    log = engine.process(matches)

    eval_log = log[log["tourney_date"] >= EVAL_START]

    print(f"=== Full dataset (includes cold-start years, informational only) ===")
    evaluate(log, "all matches, all years")

    print(f"\n=== Walk-forward eval (matches from {EVAL_START} onward) ===")
    evaluate(eval_log, "all levels")
    for level in ["tour", "challenger", "quali"]:
        evaluate(eval_log[eval_log["level"] == level], level)

    print("\n=== Calibration (challenger only, eval window, raw) ===")
    chall = eval_log[eval_log["level"] == "challenger"]
    print(calibration_symmetric(chall).to_string())

    # Recalibrate: fit temperature on everything BEFORE the eval window,
    # apply to the held-out eval window only.
    train_log = log[log["tourney_date"] < EVAL_START]
    temperature = fit_temperature(train_log)
    print(f"\n=== Recalibration: fitted temperature = {temperature:.3f} (on pre-{EVAL_START} data) ===")

    eval_log = eval_log.copy()
    eval_log["win_prob_cal"] = apply_temperature(eval_log["win_prob_blend"], temperature)

    print(f"\n=== Walk-forward eval, RECALIBRATED (matches from {EVAL_START} onward) ===")
    evaluate(eval_log, "all levels", prob_col="win_prob_cal")
    for level in ["tour", "challenger", "quali"]:
        evaluate(eval_log[eval_log["level"] == level], level, prob_col="win_prob_cal")

    print("\n=== Calibration (challenger only, eval window, RECALIBRATED) ===")
    chall_cal = eval_log[eval_log["level"] == "challenger"]
    print(calibration_symmetric(chall_cal, prob_col="win_prob_cal").to_string())


if __name__ == "__main__":
    main()
