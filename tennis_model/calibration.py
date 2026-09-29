"""Recalibrate raw Elo win probabilities against actual outcomes.

The raw Elo formula (1 / (1 + 10**(-diff/400))) assumes the classic
400-point/10x-odds scaling is exactly correct. It isn't necessarily,
especially once matches are weighted by tournament level and
retirements as this engine does. This fits a 1-parameter logistic
recalibration (temperature scaling in logit space) on a training
window and applies it to a held-out window, so the correction itself
is never fit on the data it's evaluated against.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_temperature(train_log: pd.DataFrame) -> float:
    """Fit a scalar temperature T such that sigmoid(logit(p)/T) is
    better calibrated. T > 1 pulls predictions toward 0.5 (raw model
    overconfident); T < 1 pushes them further from 0.5."""
    raw_logit = _logit(train_log["win_prob_blend"].to_numpy())

    def neg_log_likelihood(temp: float) -> float:
        if temp <= 0:
            return np.inf
        p = 1 / (1 + np.exp(-raw_logit / temp))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return -np.log(p).mean()

    result = minimize_scalar(neg_log_likelihood, bounds=(0.1, 5.0), method="bounded")
    return float(result.x)


def apply_temperature(prob: pd.Series, temperature: float) -> pd.Series:
    raw_logit = _logit(prob.to_numpy())
    calibrated = 1 / (1 + np.exp(-raw_logit / temperature))
    return pd.Series(calibrated, index=prob.index)
