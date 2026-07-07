"""
Unit tests for the parallel session executor (experiment_rq1.run_experiment_rq1)
and the session-local RNG fix (market_state.MarketState.initialize_round1)
that makes concurrent sessions safe. No live API calls: run_session is mocked.
"""
import random

from src.config import MarketConfig
from src.market.market_state import MarketState
import src.experiments.experiment_rq1 as erq1


# --------------------------------------------------------- RNG thread-safety

def test_initialize_round1_with_explicit_rng_is_deterministic():
    cfg = MarketConfig(num_sellers=2, num_buyers=2)
    state = MarketState(cfg)
    state.initialize_round1(["B1", "B2"], ["S1", "S2"], rng=random.Random(42))
    expected_bids, expected_asks = dict(state.bid_queue), dict(state.ask_queue)

    state2 = MarketState(cfg)
    state2.initialize_round1(["B1", "B2"], ["S1", "S2"], rng=random.Random(42))
    assert dict(state2.bid_queue) == expected_bids
    assert dict(state2.ask_queue) == expected_asks


def test_initialize_round1_rng_is_independent_of_global_random_state():
    """The bug this fix targets: sessions used to share the global random
    module's state via random.seed(), so concurrent sessions would race and
    corrupt each other's draws. A session using its own random.Random
    instance must be unaffected by whatever the global module is doing."""
    cfg = MarketConfig(num_sellers=2, num_buyers=2)

    state = MarketState(cfg)
    state.initialize_round1(["B1", "B2"], ["S1", "S2"], rng=random.Random(7))
    expected_bids = dict(state.bid_queue)

    # Perturb the global random module's state in between, as a concurrent
    # thread doing unrelated draws would.
    random.seed(999)
    for _ in range(50):
        random.random()

    state2 = MarketState(cfg)
    state2.initialize_round1(["B1", "B2"], ["S1", "S2"], rng=random.Random(7))
    assert dict(state2.bid_queue) == expected_bids


def test_initialize_round1_defaults_to_global_random_for_backward_compat():
    cfg = MarketConfig(num_sellers=1, num_buyers=1)
    state = MarketState(cfg)
    state.initialize_round1(["B1"], ["S1"])  # no rng passed
    assert "B1" in state.bid_queue
    assert "S1" in state.ask_queue


# ------------------------------------------------------------ parallel executor

def _fake_run_session(call_log=None):
    def fake(session_id, condition, market_cfg, exp_cfg, evaluate, verbose, session_index, logs_dir="results/logs"):
        if call_log is not None:
            call_log.append(session_id)
        return erq1.SessionResult(
            session_id=session_id, condition=condition, round_metrics=[], seller_reasoning={},
            eval_scores=[], market_history=[], final_profits={},
        )
    return fake


def test_parallel_execution_matches_sequential_results(monkeypatch):
    monkeypatch.setattr(erq1, "run_session", _fake_run_session())
    monkeypatch.setattr(erq1, "_save", lambda result, output_dir: None)

    sequential = erq1.run_experiment_rq1(
        num_sessions=3, conditions=["e0_all_gpt41", "e0_mixed"], parallel_workers=1
    )
    parallel = erq1.run_experiment_rq1(
        num_sessions=3, conditions=["e0_all_gpt41", "e0_mixed"], parallel_workers=4
    )

    for cond in ("e0_all_gpt41", "e0_mixed"):
        assert [r.session_id for r in sequential[cond]] == [r.session_id for r in parallel[cond]]
        assert all(r is not None for r in parallel[cond])


def test_parallel_execution_runs_every_job_exactly_once(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(erq1, "run_session", _fake_run_session(calls))
    monkeypatch.setattr(erq1, "_save", lambda result, output_dir: None)

    results = erq1.run_experiment_rq1(
        num_sessions=4, conditions=["e0_all_gpt41", "e0_mixed"], parallel_workers=4
    )
    assert len(calls) == 8
    assert len(set(calls)) == 8  # no job run twice
    for cond, lst in results.items():
        assert len(lst) == 4
        assert all(r is not None for r in lst)


def test_sequential_is_default_and_unaffected_by_parallel_param(monkeypatch):
    monkeypatch.setattr(erq1, "run_session", _fake_run_session())
    monkeypatch.setattr(erq1, "_save", lambda result, output_dir: None)

    results = erq1.run_experiment_rq1(num_sessions=2, conditions=["e0_all_gpt41"])
    assert [r.session_id for r in results["e0_all_gpt41"]] == [
        "rq1_e0_all_gpt41_0", "rq1_e0_all_gpt41_1",
    ]


# --------------------------------------------------------- selective rerun

def test_session_indices_flat_list_applies_to_every_condition(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(erq1, "run_session", _fake_run_session(calls))
    monkeypatch.setattr(erq1, "_save", lambda result, output_dir: None)

    results = erq1.run_experiment_rq1(
        num_sessions=10, conditions=["e0_all_gpt41", "e0_mixed"], session_indices=[5, 6, 7, 8, 9],
    )

    assert sorted(calls) == sorted(
        f"rq1_{cond}_{i}" for cond in ("e0_all_gpt41", "e0_mixed") for i in (5, 6, 7, 8, 9)
    )
    for cond, lst in results.items():
        assert [r.session_id for r in lst] == [f"rq1_{cond}_{i}" for i in (5, 6, 7, 8, 9)]


def test_session_indices_dict_gives_per_condition_control(monkeypatch):
    """The motivating case: rerun exactly [5..9] for one condition while
    letting another condition run its full default range."""
    calls: list[str] = []
    monkeypatch.setattr(erq1, "run_session", _fake_run_session(calls))
    monkeypatch.setattr(erq1, "_save", lambda result, output_dir: None)

    results = erq1.run_experiment_rq1(
        num_sessions=10,
        conditions=["e0_all_gpt41", "e0_mixed"],
        session_indices={"e0_all_gpt41": [5, 6, 7, 8, 9]},
    )

    assert [r.session_id for r in results["e0_all_gpt41"]] == [
        f"rq1_e0_all_gpt41_{i}" for i in (5, 6, 7, 8, 9)
    ]
    assert [r.session_id for r in results["e0_mixed"]] == [
        f"rq1_e0_mixed_{i}" for i in range(10)
    ]
    assert "rq1_e0_all_gpt41_0" not in calls  # 0-4 not re-paid-for


def test_session_indices_default_none_is_full_range(monkeypatch):
    monkeypatch.setattr(erq1, "run_session", _fake_run_session())
    monkeypatch.setattr(erq1, "_save", lambda result, output_dir: None)

    results = erq1.run_experiment_rq1(num_sessions=3, conditions=["e0_all_gpt41"])
    assert [r.session_id for r in results["e0_all_gpt41"]] == [
        "rq1_e0_all_gpt41_0", "rq1_e0_all_gpt41_1", "rq1_e0_all_gpt41_2",
    ]


# ------------------------------------------------------------- validity gate

def _fake_run_session_with_validity(valid_by_id: dict[str, bool]):
    def fake(session_id, condition, market_cfg, exp_cfg, evaluate, verbose, session_index, logs_dir="results/logs"):
        return erq1.SessionResult(
            session_id=session_id, condition=condition, round_metrics=[], seller_reasoning={},
            eval_scores=[], market_history=[], final_profits={},
            valid=valid_by_id.get(session_id, True),
        )
    return fake


def test_run_reports_failure_count_when_a_session_is_invalid(monkeypatch, capsys):
    monkeypatch.setattr(
        erq1, "run_session",
        _fake_run_session_with_validity({"rq1_e0_all_gpt41_1": False}),
    )
    monkeypatch.setattr(erq1, "_save", lambda result, output_dir: None)

    erq1.run_experiment_rq1(num_sessions=2, conditions=["e0_all_gpt41"])

    captured = capsys.readouterr()
    assert "[ERROR] 1/2 session(s) produced no market activity: rq1_e0_all_gpt41_1" in captured.out


def test_run_reports_all_valid_when_no_failures(monkeypatch, capsys):
    monkeypatch.setattr(erq1, "run_session", _fake_run_session_with_validity({}))
    monkeypatch.setattr(erq1, "_save", lambda result, output_dir: None)

    erq1.run_experiment_rq1(num_sessions=2, conditions=["e0_all_gpt41"])

    captured = capsys.readouterr()
    assert "All 2 session(s) completed with valid market activity." in captured.out
    assert "[ERROR]" not in captured.out


def test_parallel_mode_flags_invalid_sessions_in_progress_line(monkeypatch, capsys):
    monkeypatch.setattr(
        erq1, "run_session",
        _fake_run_session_with_validity({"rq1_e0_all_gpt41_1": False}),
    )
    monkeypatch.setattr(erq1, "_save", lambda result, output_dir: None)

    erq1.run_experiment_rq1(num_sessions=2, conditions=["e0_all_gpt41"], parallel_workers=2)

    captured = capsys.readouterr()
    assert "rq1_e0_all_gpt41_1  [INVALID: no market activity]" in captured.out
