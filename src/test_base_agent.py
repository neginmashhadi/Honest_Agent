"""
Regression tests for src/agents/base_agent.py's LLM call/parse plumbing.

Covers a real failure observed in a live run: Claude occasionally returns a
response that is just an opening markdown fence marker (e.g. "```json") with
no closing fence and no JSON content -- a degenerate/truncated generation.
The old fence-stripping logic silently collapsed this to an empty string and
handed json.loads a misleading "Expecting value: line 1 column 1 (char 0)"
error. It should now raise a clear ValueError, and call_llm_and_parse should
retry on a fresh sample rather than exhausting on a single bad response.

Run: pytest src/test_base_agent.py -v
"""
import json

import anthropic
import httpx
import pytest

import src.agents.base_agent as base_agent
from src.agents.base_agent import call_llm_and_parse, parse_json_response


def _dummy_rate_limit_error() -> anthropic.RateLimitError:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(status_code=429, request=request)
    return anthropic.RateLimitError("rate limited", response=response, body=None)


def test_lone_opening_fence_raises_clear_value_error():
    with pytest.raises(ValueError, match="markdown fence"):
        parse_json_response("```json")


def test_lone_opening_fence_with_trailing_newline_raises_clear_value_error():
    with pytest.raises(ValueError, match="markdown fence"):
        parse_json_response("```json\n")


def test_well_formed_fenced_json_still_parses():
    text = '```json\n{"ask": 91.0}\n```'
    assert parse_json_response(text) == {"ask": 91.0}


def test_unfenced_preamble_before_json_still_parses():
    # Reproduces the "Expecting value: line 1 column 1" failure seen in
    # production: the model ignores the "output ONLY JSON" instruction and
    # adds a sentence of prose before the object, with no fence at all.
    text = 'I want to note that I price independently.\n{"ask": 91.0}'
    assert parse_json_response(text) == {"ask": 91.0}


def test_trailing_commentary_after_json_still_parses():
    text = '{"ask": 91.0}\nLet me know if you need anything else.'
    assert parse_json_response(text) == {"ask": 91.0}


def test_call_llm_and_parse_retries_past_degenerate_fence_responses(monkeypatch):
    responses = iter(["```json", "```", json.dumps({"ask": 91.0})])
    calls = {"n": 0}

    def fake_call_llm(model, prompt, temperature=1.0):
        calls["n"] += 1
        return next(responses)

    monkeypatch.setattr(base_agent, "call_llm", fake_call_llm)
    monkeypatch.setattr(base_agent, "time", type("T", (), {"sleep": staticmethod(lambda s: None)}))

    result = call_llm_and_parse("claude-sonnet-4-6", "prompt", max_retries=4)
    assert result == {"ask": 91.0}
    assert calls["n"] == 3  # first two attempts were degenerate, third succeeded


def test_call_llm_and_parse_raises_after_exhausting_retries(monkeypatch):
    def always_degenerate(model, prompt, temperature=1.0):
        return "```json"

    monkeypatch.setattr(base_agent, "call_llm", always_degenerate)
    monkeypatch.setattr(base_agent, "time", type("T", (), {"sleep": staticmethod(lambda s: None)}))

    # Wrapped in RuntimeError so a raw-response preview can be attached for
    # debugging; the original error's message is preserved inside it.
    with pytest.raises(RuntimeError, match="markdown fence"):
        call_llm_and_parse("claude-sonnet-4-6", "prompt", max_retries=2)


def test_call_llm_and_parse_retries_past_transient_api_error(monkeypatch):
    """The gap flagged in review: a rate-limit/5xx/connection error used to
    propagate uncaught and crash the whole session. It should now be retried
    on a fresh request just like a bad parse."""
    responses = iter([_dummy_rate_limit_error(), json.dumps({"ask": 91.0})])
    calls = {"n": 0}

    def fake_call_llm(model, prompt, temperature=1.0):
        calls["n"] += 1
        item = next(responses)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(base_agent, "call_llm", fake_call_llm)
    monkeypatch.setattr(base_agent, "time", type("T", (), {"sleep": staticmethod(lambda s: None)}))

    result = call_llm_and_parse("claude-sonnet-4-6", "prompt", max_retries=4)
    assert result == {"ask": 91.0}
    assert calls["n"] == 2  # first attempt rate-limited, second succeeded


def test_rate_limit_errors_get_a_separate_longer_retry_budget(monkeypatch):
    """Rate limits are worth waiting out longer than a bad parse -- they must
    not share max_retries' short generic budget."""
    fail_count = 6  # exceeds a tiny max_retries=2 generic budget (3 total attempts)
    responses = [_dummy_rate_limit_error() for _ in range(fail_count)] + [json.dumps({"ask": 91.0})]
    calls = {"n": 0}

    def fake_call_llm(model, prompt, temperature=1.0):
        item = responses[calls["n"]]
        calls["n"] += 1
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(base_agent, "call_llm", fake_call_llm)
    monkeypatch.setattr(base_agent, "time", type("T", (), {"sleep": staticmethod(lambda s: None)}))

    result = call_llm_and_parse("claude-sonnet-4-6", "prompt", max_retries=2)
    assert result == {"ask": 91.0}
    assert calls["n"] == fail_count + 1


def test_rate_limit_backoff_is_exponential_and_capped_at_60s(monkeypatch):
    sleeps = []

    def always_rate_limited(model, prompt, temperature=1.0):
        raise _dummy_rate_limit_error()

    monkeypatch.setattr(base_agent, "call_llm", always_rate_limited)
    monkeypatch.setattr(base_agent, "time", type("T", (), {"sleep": staticmethod(lambda s: sleeps.append(s))}))

    with pytest.raises(RuntimeError):
        call_llm_and_parse("claude-sonnet-4-6", "prompt", max_retries=0, rate_limit_max_retries=8)

    assert sleeps == [min(2 ** i, 60.0) for i in range(8)]
    assert sleeps[-1] == 60.0  # 2**7=128 gets capped


def test_call_llm_and_parse_does_not_retry_permanent_errors(monkeypatch):
    """Auth/bad-request errors would fail identically every time -- don't
    waste the backoff budget retrying them."""
    def always_auth_error(model, prompt, temperature=1.0):
        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        response = httpx.Response(status_code=401, request=request)
        raise anthropic.AuthenticationError("bad api key", response=response, body=None)

    monkeypatch.setattr(base_agent, "call_llm", always_auth_error)
    monkeypatch.setattr(base_agent, "time", type("T", (), {"sleep": staticmethod(lambda s: None)}))

    with pytest.raises(anthropic.AuthenticationError):
        call_llm_and_parse("claude-sonnet-4-6", "prompt", max_retries=4)
