"""
Unit tests for runner.py's failure-visibility machinery: per-session log
files (always written, not gated by verbose), the failed_agent_calls
counter, and the end-of-session validity gate. No live API calls: call_llm
is mocked.

Run: pytest src/test_runner_reliability.py -v
"""
import src.agents.base_agent as base_agent_module
from src.config import ExperimentConfig, MarketConfig
from src.experiments.runner import run_session


def _mock_sleep(monkeypatch):
    monkeypatch.setattr(base_agent_module, "time", type("T", (), {"sleep": staticmethod(lambda s: None)}))


def test_failed_agent_calls_counted_and_logged(monkeypatch, tmp_path):
    def always_fail(model, prompt, temperature=1.0):
        raise ValueError("simulated agent failure")

    monkeypatch.setattr(base_agent_module, "call_llm", always_fail)
    _mock_sleep(monkeypatch)

    market_cfg = MarketConfig(num_sellers=2, num_buyers=2, num_rounds=2)
    exp_cfg = ExperimentConfig(seller_model="gpt-4.1-2025-04-14", buyer_model="gpt-4.1-2025-04-14")

    result = run_session(
        "test_reliability_session", "e0_test", market_cfg, exp_cfg,
        evaluate=True, verbose=False, session_index=0, logs_dir=str(tmp_path),
    )

    assert result.failed_agent_calls > 0

    log_path = tmp_path / "test_reliability_session.log"
    assert log_path.exists()
    content = log_path.read_text()
    assert "[WARN]" in content
    assert content.count("[WARN]") == result.failed_agent_calls


def test_log_file_written_even_when_verbose_is_false(monkeypatch, tmp_path):
    """The whole point: warnings must never be silently lost just because
    verbose=False (the default in parallel mode)."""
    def always_fail(model, prompt, temperature=1.0):
        raise ValueError("simulated agent failure")

    monkeypatch.setattr(base_agent_module, "call_llm", always_fail)
    _mock_sleep(monkeypatch)

    market_cfg = MarketConfig(num_sellers=2, num_buyers=2, num_rounds=1)
    exp_cfg = ExperimentConfig(seller_model="gpt-4.1-2025-04-14", buyer_model="gpt-4.1-2025-04-14")

    run_session(
        "test_silent_but_logged", "e0_test", market_cfg, exp_cfg,
        evaluate=True, verbose=False, session_index=0, logs_dir=str(tmp_path),
    )

    assert (tmp_path / "test_silent_but_logged.log").exists()


def test_no_failures_means_no_log_file(monkeypatch, tmp_path):
    def fake_call_llm(model, prompt, temperature=1.0):
        import json
        return json.dumps({
            "reflection": "r", "plan_for_this_hour": "p", "ask": 91.0,
            "new_memory": "m", "scratch_pad_update": "s",
        })

    monkeypatch.setattr(base_agent_module, "call_llm", fake_call_llm)
    import src.evaluation.evaluator as evaluator_module
    monkeypatch.setattr(evaluator_module, "call_llm", fake_call_llm)
    _mock_sleep(monkeypatch)

    market_cfg = MarketConfig(num_sellers=2, num_buyers=2, num_rounds=2)
    exp_cfg = ExperimentConfig(seller_model="gpt-4.1-2025-04-14", buyer_model="gpt-4.1-2025-04-14")

    result = run_session(
        "test_clean_session", "e0_test", market_cfg, exp_cfg,
        evaluate=True, verbose=False, session_index=0, logs_dir=str(tmp_path),
    )

    assert result.failed_agent_calls == 0
    assert not (tmp_path / "test_clean_session.log").exists()


# ------------------------------------------------------------- validity gate

def test_session_with_all_calls_failing_is_marked_invalid(monkeypatch, tmp_path, capsys):
    def always_fail(model, prompt, temperature=1.0):
        raise ValueError("simulated agent failure")

    monkeypatch.setattr(base_agent_module, "call_llm", always_fail)
    _mock_sleep(monkeypatch)

    market_cfg = MarketConfig(num_sellers=2, num_buyers=2, num_rounds=2)
    exp_cfg = ExperimentConfig(seller_model="gpt-4.1-2025-04-14", buyer_model="gpt-4.1-2025-04-14")

    result = run_session(
        "test_dead_session", "e0_test", market_cfg, exp_cfg,
        evaluate=True, verbose=False, session_index=0, logs_dir=str(tmp_path),
    )

    assert result.valid is False
    captured = capsys.readouterr()
    assert "[ERROR] session test_dead_session produced no market activity" in captured.out


def test_session_with_normal_activity_is_marked_valid(monkeypatch, tmp_path, capsys):
    import json

    def fake_call_llm(model, prompt, temperature=1.0):
        return json.dumps({
            "reflection": "r", "plan_for_this_hour": "p", "ask": 91.0,
            "new_memory": "m", "scratch_pad_update": "s",
        })

    monkeypatch.setattr(base_agent_module, "call_llm", fake_call_llm)
    import src.evaluation.evaluator as evaluator_module
    monkeypatch.setattr(evaluator_module, "call_llm", fake_call_llm)
    _mock_sleep(monkeypatch)

    market_cfg = MarketConfig(num_sellers=2, num_buyers=2, num_rounds=2)
    exp_cfg = ExperimentConfig(seller_model="gpt-4.1-2025-04-14", buyer_model="gpt-4.1-2025-04-14")

    result = run_session(
        "test_alive_session", "e0_test", market_cfg, exp_cfg,
        evaluate=True, verbose=False, session_index=0, logs_dir=str(tmp_path),
    )

    assert result.valid is True
    captured = capsys.readouterr()
    assert "[ERROR]" not in captured.out
