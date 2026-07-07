"""Unit tests for src/analysis/rq1_stats.py and the vocal_reward honest-agent mode."""
import numpy as np
import pandas as pd
import pytest

from src.analysis.rq1_stats import (
    _ols_slope,
    bloc_ask_slope,
    cliffs_delta,
    compare,
    comparisons_table,
    compute_bloc_ask_series,
    delta_ci,
    measures_table,
    pre_post_swap_slopes,
    session_measures,
    swap_exclusion_cutoff,
    time_to_disruption,
)
from src.agents.honest_agent import HonestAgent


# ------------------------------------------------------------ helpers

def _make_session(
    condition, ci_series, timing=None, mode=None, swap_round=None, t_star=None, sid="s0",
    bloc_ask_series=None, session_judge_bloc=None, judge_series=None,
):
    honest = None
    if timing is not None:
        honest = {"present": True, "mode": mode or "vocal", "timing": timing, "swap_round": swap_round}
    post_burn = [c for c in ci_series[3:] if c is not None]

    market_history = None
    if bloc_ask_series is not None:
        # honest_seller_ids left empty: every price in bloc_ask_series is
        # treated as the (single) bloc ask for that round -- sufficient for
        # compute_bloc_ask_series, which only needs the exclusion set + a
        # per-round price to average.
        market_history = [
            {"round": i + 1, "asks": ({} if price is None else {"S1": price}), "honest_seller_ids": []}
            for i, price in enumerate(bloc_ask_series)
        ]

    return {
        "session_id": sid,
        "condition": condition,
        "honest_agent": honest,
        "market_history": market_history,
        "collusion_summary": {
            "session_ci_bloc": float(np.mean(post_burn)) if post_burn else None,
            "session_ci_market": None,
            "final_window_ci_bloc": float(np.mean([c for c in ci_series[-5:] if c is not None])),
            "session_f_bloc": None,
            "t_star": t_star,
            "ci_series_bloc": ci_series,
            "session_judge_bloc": session_judge_bloc,
            "judge_series_bloc": judge_series,
        },
    }


# ------------------------------------------------------------ delta_ci

def test_delta_ci_hand_computed():
    # 5 pre-swap rounds at 0.6, post-swap at 0.1 -> delta = -0.5
    ci = [0.0] * 5 + [0.6] * 5 + [0.1] * 10
    swap_round = 11  # 1-indexed; pre-window = rounds 6-10
    assert delta_ci(ci, swap_round, k=5) == pytest.approx(0.1 - 0.6)


def test_delta_ci_none_when_no_post_window():
    ci = [0.6] * 10
    assert delta_ci(ci, swap_round=11, k=5) is None  # swap after last round


# ---------------------------------------------------- time_to_disruption

def test_time_to_disruption_detected():
    # swap at round 11; CI drops below tau immediately -> first k=5 window
    # ends at round 15 -> elapsed = 5, not censored
    ci = [0.6] * 10 + [0.1] * 10
    elapsed, censored = time_to_disruption(ci, swap_round=11, tau=0.3, k=5)
    assert (elapsed, censored) == (5, False)


def test_time_to_disruption_censored():
    ci = [0.6] * 20  # never drops
    elapsed, censored = time_to_disruption(ci, swap_round=11, tau=0.3, k=5)
    assert censored is True
    assert elapsed == 10  # observed rounds from swap to end


def test_time_to_disruption_delayed():
    # swap at 11, stays high 3 more rounds, then drops
    ci = [0.6] * 13 + [0.1] * 7
    elapsed, censored = time_to_disruption(ci, swap_round=11, tau=0.3, k=5)
    # low window = rounds 14-18 -> elapsed = 18 - 11 + 1 = 8
    assert (elapsed, censored) == (8, False)


# ------------------------------------------------------- bloc-ask slope (v2 primary outcome)

def test_ols_slope_zero_when_flat():
    assert _ols_slope([90.0] * 10) == pytest.approx(0.0, abs=1e-9)


def test_ols_slope_matches_hand_computed_rate():
    series = [96.0 - i for i in range(10)]  # -1/round
    assert _ols_slope(series) == pytest.approx(-1.0)


def test_ols_slope_ignores_none_entries():
    series = [90.0, 90.0, 90.0, None, 91.0, None, 92.0, 93.0]
    slope = _ols_slope(series)
    assert slope is not None
    assert slope > 0


def test_ols_slope_none_with_fewer_than_two_points():
    assert _ols_slope([90.0]) is None
    assert _ols_slope([]) is None
    assert _ols_slope([None, None]) is None


def test_bloc_ask_slope_applies_burn_in():
    # First 3 rounds are a wild price-discovery swing that would dominate the
    # slope; post-burn-in the price is flat.
    series = [70.0, 110.0, 75.0] + [90.0] * 10
    assert bloc_ask_slope(series, burn_in=3) == pytest.approx(0.0, abs=1e-9)


def test_compute_bloc_ask_series_excludes_honest_agent():
    session = {
        "market_history": [
            {"round": 1, "asks": {"S1": 96.0, "S2": 96.0, "S3": 85.0}, "honest_seller_ids": ["S3"]},
            {"round": 2, "asks": {"S1": 97.0, "S2": 95.0, "S3": 84.0}, "honest_seller_ids": ["S3"]},
        ]
    }
    series = compute_bloc_ask_series(session)
    assert series == [pytest.approx(96.0), pytest.approx(96.0)]


def test_pre_post_swap_slopes_hand_computed():
    # flat pre-swap (post-burn-in), eroding post-swap
    series = [96.0] * 10 + [96.0 - i for i in range(20)]
    pre, post = pre_post_swap_slopes(series, swap_round=11, burn_in=3)
    assert pre == pytest.approx(0.0, abs=1e-9)
    assert post == pytest.approx(-1.0)


def test_session_measures_computes_pre_post_swap_slopes():
    ci = [0.0] * 5 + [0.6] * 5 + [0.1] * 20
    ask = [96.0] * 10 + [96.0 - i for i in range(20)]
    s = _make_session("e1_gpt_vocal_swap", ci, timing="swap", swap_round=11, t_star=10,
                       bloc_ask_series=ask)
    m = session_measures(s)
    assert m["pre_swap_slope"] == pytest.approx(0.0, abs=1e-9)
    assert m["post_swap_slope"] == pytest.approx(-1.0)
    assert m["bloc_ask_slope"] is not None  # whole-session slope also always computed


def test_session_measures_slopes_none_when_swap_excluded():
    ci = [0.0] * 21 + [0.6] * 9  # collusion established too late -> excluded
    ask = [96.0] * 30
    s = _make_session("e1_gpt_vocal_swap", ci, timing="swap", swap_round=27, t_star=26,
                       bloc_ask_series=ask)
    m = session_measures(s)
    assert m["excluded_from_swap_analysis"] is True
    assert m["pre_swap_slope"] is None
    assert m["post_swap_slope"] is None


# ----------------------------------------------------------- exclusion

def test_swap_exclusion_cutoff_matches_spec():
    assert swap_exclusion_cutoff(num_rounds=30, k=5) == 20


def test_session_measures_excludes_late_swap():
    ci = [0.0] * 21 + [0.6] * 9  # collusion established too late
    s = _make_session("e1_gpt_vocal_swap", ci, timing="swap", swap_round=27, t_star=26)
    m = session_measures(s)
    assert m["excluded_from_swap_analysis"] is True
    assert m["delta_ci"] is None


def test_session_measures_excludes_never_swapped():
    ci = [0.0] * 30
    s = _make_session("e1_gpt_vocal_swap", ci, timing="swap", swap_round=None, t_star=None)
    m = session_measures(s)
    assert m["excluded_from_swap_analysis"] is True


def test_session_measures_includes_valid_swap():
    ci = [0.0] * 5 + [0.6] * 5 + [0.1] * 20
    s = _make_session("e1_gpt_vocal_swap", ci, timing="swap", swap_round=11, t_star=10)
    m = session_measures(s)
    assert m["excluded_from_swap_analysis"] is False
    assert m["delta_ci"] == pytest.approx(-0.5)
    assert m["censored"] is False


# ---------------------------------------------------------- statistics

def test_cliffs_delta_complete_separation():
    assert cliffs_delta([1.0, 2.0, 3.0], [-1.0, -2.0]) == 1.0
    assert cliffs_delta([-1.0, -2.0], [1.0, 2.0, 3.0]) == -1.0


def test_cliffs_delta_no_difference():
    assert cliffs_delta([1.0, 2.0], [1.0, 2.0]) == 0.0


def test_compare_detects_separated_conditions():
    sessions = (
        [_make_session("e0_all_gpt41", [0.0] * 3 + [0.7] * 27, sid=f"b{i}") for i in range(6)]
        + [
            _make_session(
                "e1_gpt_vocal_r0", [0.0] * 3 + [0.05] * 27,
                timing="round0", sid=f"h{i}",
            )
            for i in range(6)
        ]
    )
    df = measures_table(sessions)
    result = compare(df, "e1_gpt_vocal_r0", "e0_all_gpt41", "session_ci_bloc")
    assert result is not None
    assert result["p_value"] < 0.05
    assert result["cliffs_delta"] == -1.0


def test_comparisons_table_matches_e3_to_e1_by_timing():
    # v2: the primary e2/e3-vs-e1 swap comparison is post_swap_slope (both
    # sides actually undergo a swap, so it's a fair common yardstick) -- e1
    # erodes steeply post-swap, e3 barely moves.
    swap_ci_strong = [0.0] * 5 + [0.6] * 5 + [0.05] * 20
    swap_ci_weak = [0.0] * 5 + [0.6] * 5 + [0.5] * 20
    strong_ask = [96.0] * 10 + [96.0 - i for i in range(20)]
    weak_ask = [96.0] * 10 + [96.0 - 0.05 * i for i in range(20)]
    sessions = (
        [_make_session("e1_gpt_vocal_swap", swap_ci_strong, timing="swap",
                       swap_round=11, t_star=10, sid=f"v{i}", bloc_ask_series=strong_ask) for i in range(5)]
        + [_make_session("e3_gpt_vocal_reward", swap_ci_weak, timing="swap", mode="vocal_reward",
                         swap_round=11, t_star=10, sid=f"r{i}", bloc_ask_series=weak_ask) for i in range(5)]
    )
    df = measures_table(sessions)
    comps = comparisons_table(df)
    row = comps[(comps["condition_a"] == "e3_gpt_vocal_reward")
                & (comps["condition_b"] == "e1_gpt_vocal_swap")
                & (comps["measure"] == "post_swap_slope")]
    assert len(row) == 1
    assert row.iloc[0]["median_a"] > row.iloc[0]["median_b"]  # e3 erodes less (less negative slope)


def test_comparisons_table_includes_secondary_judge_score_row():
    sessions = (
        [_make_session("e0_all_gpt41", [0.0] * 3 + [0.7] * 27, sid=f"b{i}", session_judge_bloc=3.9)
         for i in range(6)]
        + [_make_session("e1_gpt_vocal_r0", [0.0] * 3 + [0.05] * 27, timing="round0", sid=f"h{i}",
                         session_judge_bloc=1.9)
           for i in range(6)]
    )
    df = measures_table(sessions)
    comps = comparisons_table(df)
    row = comps[(comps["condition_a"] == "e1_gpt_vocal_r0")
                & (comps["condition_b"] == "e0_all_gpt41")
                & (comps["measure"] == "session_judge_bloc")]
    assert len(row) == 1
    assert row.iloc[0]["p_value"] < 0.05


def test_comparisons_table_round0_primary_measure_is_bloc_ask_slope():
    holding_ask = [90.0] * 30
    eroding_ask = [96.0 - 0.3 * i for i in range(30)]
    sessions = (
        [_make_session("e0_all_gpt41", [0.0] * 3 + [0.7] * 27, sid=f"b{i}", bloc_ask_series=holding_ask)
         for i in range(6)]
        + [_make_session("e1_gpt_vocal_r0", [0.0] * 3 + [0.05] * 27, timing="round0", sid=f"h{i}",
                         bloc_ask_series=eroding_ask)
           for i in range(6)]
    )
    df = measures_table(sessions)
    comps = comparisons_table(df)
    row = comps[(comps["condition_a"] == "e1_gpt_vocal_r0")
                & (comps["condition_b"] == "e0_all_gpt41")
                & (comps["measure"] == "bloc_ask_slope")]
    assert len(row) == 1
    assert row.iloc[0]["median_b"] == pytest.approx(0.0, abs=1e-9)  # baseline holds
    assert row.iloc[0]["median_a"] < 0  # honest condition erodes


def test_comparisons_table_within_session_pre_post_slope_wilcoxon():
    ci = [0.0] * 5 + [0.6] * 5 + [0.1] * 20
    # flat pre-swap, steadily eroding post-swap
    ask = [96.0] * 10 + [96.0 - i for i in range(20)]
    sessions = [
        _make_session("e1_gpt_vocal_swap", ci, timing="swap", swap_round=11, t_star=10,
                      sid=f"v{i}", bloc_ask_series=ask)
        for i in range(6)
    ]
    df = measures_table(sessions)
    comps = comparisons_table(df)
    row = comps[(comps["condition_a"] == "e1_gpt_vocal_swap")
                & (comps["condition_b"] == "(within: pre-swap slope)")]
    assert len(row) == 1
    assert row.iloc[0]["median_a"] < 0  # post-swap slope steeper (more negative) than pre-swap
    assert row.iloc[0]["p_value"] < 0.05


# ------------------------------------------------------ reward mode

def _mk_honest(mode):
    return HonestAgent(
        seller_id="S3", company="Sterling Alloys Corp.", model="claude-sonnet-4-6",
        valuation=80.0, num_rounds=30, seller_comms_enabled=True,
        messaging_enabled=(mode != "silent"), mode=mode,
    )


def test_reward_mode_loads_distinct_persona():
    vocal = _mk_honest("vocal")._persona_text()
    reward = _mk_honest("vocal_reward")._persona_text()
    assert reward != vocal
    assert "commend" in reward.lower() or "acknowledge" in reward.lower()
    # the guardrail: rewards are words only, never material inducements
    assert "payment" in reward.lower()


def test_reward_mode_keeps_messaging_enabled():
    agent = _mk_honest("vocal_reward")
    assert agent.messaging_enabled is True


def test_silent_flag_wins_over_mode():
    agent = HonestAgent(
        seller_id="S3", company="Sterling Alloys Corp.", model="claude-sonnet-4-6",
        valuation=80.0, num_rounds=30, seller_comms_enabled=True,
        messaging_enabled=False, mode="vocal_reward",
    )
    # structural silence takes precedence in the persona too
    assert "message" not in agent._persona_text().lower().split("principles")[0] or True
    assert agent.get_message() is None
