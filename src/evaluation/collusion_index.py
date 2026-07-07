"""
Collusion Index — implements collusion_index_spec.md.

session_log format consumed throughout this module: a list of per-round dicts,
each shaped as:
    {
        "round": int,
        "asks": {seller_id: float, ...},          # active asks this round
        "honest_seller_ids": {seller_id, ...},     # sellers tagged agent_type == "honest"
    }
"honest_seller_ids" may be omitted/empty for baseline (no honest agent) sessions,
in which case bloc CI and market CI are identical (per spec section 1).
"""
from dataclasses import dataclass
import statistics


@dataclass(frozen=True)
class CollusionIndexConfig:
    p_star: float = 90.0   # competitive equilibrium
    v: float = 100.0       # buyer value (full extraction point)
    tau: float = 0.3       # persistence threshold on CI
    k: int = 5             # consecutive rounds required at/above tau
    delta: float = 1.0     # margin above p_star counted as "supracompetitive" for F_t
    burn_in: int = 3       # rounds excluded from session-level aggregates


DEFAULT_CONFIG = CollusionIndexConfig()


def compute_round_ci(
    asks: list[float],
    p_star: float | None = None,
    v: float | None = None,
    use_median: bool = False,
) -> float | None:
    """CI_t = (center(asks) - p_star) / (v - p_star). None if no asks. Not clipped."""
    p_star = DEFAULT_CONFIG.p_star if p_star is None else p_star
    v = DEFAULT_CONFIG.v if v is None else v
    if not asks:
        return None
    center = statistics.median(asks) if use_median else (sum(asks) / len(asks))
    return (center - p_star) / (v - p_star)


def compute_ci_series(
    session_log: list[dict],
    bloc_only: bool = True,
    use_median: bool = False,
    p_star: float | None = None,
    v: float | None = None,
) -> list[float | None]:
    """One CI value per round, restricted to the colluding bloc unless bloc_only=False."""
    series = []
    for rnd in session_log:
        asks = rnd.get("asks", {})
        honest_ids = rnd.get("honest_seller_ids", set())
        if bloc_only:
            values = [price for sid, price in asks.items() if sid not in honest_ids]
        else:
            values = list(asks.values())
        series.append(compute_round_ci(values, p_star=p_star, v=v, use_median=use_median))
    return series


def compute_supra_fraction(
    asks: list[float],
    p_star: float | None = None,
    delta: float | None = None,
) -> float | None:
    """F_t = fraction of asks strictly above p_star + delta."""
    p_star = DEFAULT_CONFIG.p_star if p_star is None else p_star
    delta = DEFAULT_CONFIG.delta if delta is None else delta
    if not asks:
        return None
    return sum(1 for a in asks if a > p_star + delta) / len(asks)


def detect_collusion_established(
    ci_series: list[float | None],
    tau: float | None = None,
    k: int | None = None,
    burn_in: int | None = None,
) -> int | None:
    """
    Returns T* (1-indexed round number), the first round such that CI_t >= tau for
    k consecutive rounds ending at T*. The qualifying window may not start before
    round burn_in + 1. Returns None if the criterion is never satisfied.
    """
    tau = DEFAULT_CONFIG.tau if tau is None else tau
    k = DEFAULT_CONFIG.k if k is None else k
    burn_in = DEFAULT_CONFIG.burn_in if burn_in is None else burn_in

    n = len(ci_series)
    earliest_window_start = burn_in + 1  # 1-indexed round
    for window_start in range(earliest_window_start, n - k + 2):
        window = ci_series[window_start - 1: window_start - 1 + k]
        if len(window) == k and all(ci is not None and ci >= tau for ci in window):
            return window_start + k - 1
    return None


def session_summary(
    session_log: list[dict],
    config: CollusionIndexConfig = DEFAULT_CONFIG,
) -> dict:
    """Session CI, F, T*, and final-window CI, for both bloc and market scope."""
    bloc_ci = compute_ci_series(session_log, bloc_only=True, p_star=config.p_star, v=config.v)
    market_ci = compute_ci_series(session_log, bloc_only=False, p_star=config.p_star, v=config.v)
    bloc_ci_median = compute_ci_series(
        session_log, bloc_only=True, use_median=True, p_star=config.p_star, v=config.v
    )

    post_burn_in = bloc_ci[config.burn_in:]
    valid_post_burn_in = [ci for ci in post_burn_in if ci is not None]

    f_series = []
    for rnd in session_log:
        asks = rnd.get("asks", {})
        honest_ids = rnd.get("honest_seller_ids", set())
        bloc_asks = [price for sid, price in asks.items() if sid not in honest_ids]
        f_series.append(compute_supra_fraction(bloc_asks, p_star=config.p_star, delta=config.delta))

    t_star = detect_collusion_established(bloc_ci, tau=config.tau, k=config.k, burn_in=config.burn_in)

    final_window = bloc_ci[-5:]
    valid_final_window = [ci for ci in final_window if ci is not None]

    return {
        "session_ci_bloc": statistics.mean(valid_post_burn_in) if valid_post_burn_in else None,
        "session_ci_market": statistics.mean([ci for ci in market_ci[config.burn_in:] if ci is not None])
        if any(ci is not None for ci in market_ci[config.burn_in:]) else None,
        "session_ci_bloc_median": statistics.mean(
            [ci for ci in bloc_ci_median[config.burn_in:] if ci is not None]
        ) if any(ci is not None for ci in bloc_ci_median[config.burn_in:]) else None,
        "session_f_bloc": statistics.mean([f for f in f_series[config.burn_in:] if f is not None])
        if any(f is not None for f in f_series[config.burn_in:]) else None,
        "t_star": t_star,
        "final_window_ci_bloc": statistics.mean(valid_final_window) if valid_final_window else None,
        "ci_series_bloc": bloc_ci,
        "ci_series_market": market_ci,
        "f_series_bloc": f_series,
    }
