"""Convert model probabilities into fair (no-vig) odds for comparison
against a bookmaker's posted price."""


def decimal_odds(prob: float) -> float:
    prob = min(max(prob, 1e-6), 1.0)
    return 1.0 / prob


def american_odds(prob: float) -> int:
    d = decimal_odds(prob)
    if d >= 2.0:
        return round((d - 1) * 100)
    return round(-100 / (d - 1))


def fractional_odds(prob: float) -> str:
    d = decimal_odds(prob)
    num = d - 1
    # Round to nearest simple fraction (denominator up to 20) for readability
    best = (1, 1)
    best_err = abs(num - 1)
    for denom in range(1, 21):
        n = round(num * denom)
        if n <= 0:
            continue
        err = abs(num - n / denom)
        if err < best_err:
            best_err = err
            best = (n, denom)
    return f"{best[0]}/{best[1]}"
