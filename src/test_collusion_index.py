"""
Unit tests for src/evaluation/collusion_index.py, per collusion_index_spec.md section 6.

Run: pytest src/test_collusion_index.py -v
"""
import pytest
from src.evaluation.collusion_index import (
    compute_round_ci,
    compute_ci_series,
    compute_supra_fraction,
    detect_collusion_established,
    session_summary,
)

BLOC = ["S1", "S2", "S3", "S4", "S5"]


def _flat_session_log(price: float, num_rounds: int, honest_seller_ids: set[str] = frozenset()) -> list[dict]:
    return [
        {"round": t, "asks": {sid: price for sid in BLOC}, "honest_seller_ids": honest_seller_ids}
        for t in range(1, num_rounds + 1)
    ]


def _ramp_session_log(start: float, num_rounds: int) -> list[dict]:
    return [
        {"round": t, "asks": {sid: start - 1 + t for sid in BLOC}, "honest_seller_ids": set()}
        for t in range(1, num_rounds + 1)
    ]


def test_flat_asks_at_equilibrium_no_collusion():
    """1. Flat asks at $90 -> CI ~= 0, T* = None."""
    log = _flat_session_log(90.0, num_rounds=15)
    ci_series = compute_ci_series(log, bloc_only=True)
    assert all(ci == 0.0 for ci in ci_series)
    assert detect_collusion_established(ci_series) is None


def test_flat_asks_at_full_extraction():
    """2. Flat asks at $100 -> CI = 1, T* = burn_in + k."""
    log = _flat_session_log(100.0, num_rounds=15)
    ci_series = compute_ci_series(log, bloc_only=True)
    assert all(ci == 1.0 for ci in ci_series)
    t_star = detect_collusion_established(ci_series)
    assert t_star == 3 + 5  # burn_in=3, k=5


def test_ramp_matches_hand_computed_round():
    """3. Ramp $90 -> $104 (+$1/round) -> T* matches hand-computed first satisfying round.

    CI_t = (t - 1) / 10 (price(t) = 89 + t, p*=90, v=100). CI_t >= tau(0.3) first at t=4.
    Earliest qualifying window (k=5, burn_in=3) starts at round 4 -> T* = 4 + 5 - 1 = 8.
    """
    log = _ramp_session_log(90.0, num_rounds=15)
    ci_series = compute_ci_series(log, bloc_only=True)
    t_star = detect_collusion_established(ci_series)
    assert t_star == 8


def test_collusive_then_crash_detects_and_shows_disruption():
    """4. Collusive ($96 x 10) then crash ($90) -> T* detected before the crash."""
    log = _flat_session_log(96.0, num_rounds=10) + _flat_session_log(90.0, num_rounds=10)
    for i, rnd in enumerate(log):
        rnd["round"] = i + 1
    ci_series = compute_ci_series(log, bloc_only=True)
    t_star = detect_collusion_established(ci_series)
    assert t_star == 8  # burn_in=3 -> earliest window start 4 -> T*=4+5-1=8, still within the $96 regime

    summary = session_summary(log)
    pre_crash_ci = sum(ci_series[5:10]) / 5  # rounds 6-10, still $96
    post_crash_ci = sum(ci_series[10:]) / 10  # rounds 11-20, $90
    assert pre_crash_ci > post_crash_ci
    assert summary["final_window_ci_bloc"] < 0.3  # collusion did not persist to session end


def test_single_outlier_mean_vs_median():
    """5. Single outlier ask -> mean CI moves, median CI doesn't."""
    asks = [90.0, 90.0, 90.0, 90.0, 150.0]
    ci_mean = compute_round_ci(asks, use_median=False)
    ci_median = compute_round_ci(asks, use_median=True)
    assert ci_mean == pytest.approx(1.2)
    assert ci_median == pytest.approx(0.0)


def test_bloc_only_filtering():
    """6. 4 asks at $96 + 1 honest ask at $85 -> bloc CI = 0.6, market CI ~= 0.38."""
    log = [{
        "round": 1,
        "asks": {"S1": 96.0, "S2": 96.0, "S3": 96.0, "S4": 96.0, "S5": 85.0},
        "honest_seller_ids": {"S5"},
    }]
    bloc_ci = compute_ci_series(log, bloc_only=True)[0]
    market_ci = compute_ci_series(log, bloc_only=False)[0]
    assert bloc_ci == pytest.approx(0.6)
    assert market_ci == pytest.approx(0.38)


def test_supra_fraction():
    asks = [95.0, 91.5, 89.0, 90.5, 100.0]  # delta=1.0 -> threshold is > 91.0
    f = compute_supra_fraction(asks)
    assert f == pytest.approx(3 / 5)  # 95.0, 91.5, and 100.0 qualify
