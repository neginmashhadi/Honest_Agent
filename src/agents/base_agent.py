import json
import os
import time
from typing import Optional
import anthropic
import openai
from jinja2 import Environment, FileSystemLoader

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "prompts")
_jinja_env = Environment(loader=FileSystemLoader(PROMPTS_DIR), trim_blocks=True, lstrip_blocks=True)


def render_template(template_name: str, **kwargs) -> str:
    return _jinja_env.get_template(template_name).render(**kwargs)


def call_llm(model: str, prompt: str, temperature: float = 1.0) -> str:
    if "claude" in model:
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        msg = client.messages.create(
            model=model,
            max_tokens=2048,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        # Join all text blocks rather than indexing content[0] directly, so an
        # empty/non-text content list yields "" instead of an IndexError.
        return "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
    else:
        client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
        )
        return resp.choices[0].message.content or ""


def parse_json_response(text: str) -> dict:
    """Extract and parse JSON from LLM response, stripping markdown fences if present."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # strip opening fence
        lines = lines[1:]
        # strip closing fence
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
        if not text:
            # A response that was just an opening fence marker with nothing
            # after it (a degenerate/truncated generation) -- surface this
            # distinctly rather than letting json.loads raise a generic,
            # misleading "Expecting value" error at position 0.
            raise ValueError("LLM response was only a markdown fence marker, no JSON content")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # The model sometimes ignores the "output ONLY JSON" instruction and
        # adds stray prose before/after the object. Fall back to the
        # outermost {...} span rather than failing outright on a preamble.
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(text[start:end + 1])


# Rate limits get their own, longer retry budget (see call_llm_and_parse):
# they're worth waiting out -- a 429 usually clears within tens of seconds --
# unlike a bad parse, where retrying faster doesn't cost anything by waiting.
RATE_LIMIT_ERRORS = (
    anthropic.RateLimitError,
    openai.RateLimitError,
)

# Transient errors worth retrying on a fresh sample: momentary server
# overload/5xx, connection drops, timeouts, and bad/empty parses. Deliberately
# excludes permanent errors (bad request, auth, not-found) that would just
# fail identically every retry and waste the backoff.
RETRYABLE_ERRORS = (
    json.JSONDecodeError,
    ValueError,
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.InternalServerError,
    anthropic.OverloadedError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)


def call_llm_and_parse(
    model: str,
    prompt: str,
    temperature: float = 1.0,
    max_retries: int = 5,
    rate_limit_max_retries: int = 8,
    rate_limit_backoff_cap: float = 60.0,
) -> dict:
    """call_llm + parse_json_response, retrying the whole round-trip (not just the
    parse) on an empty/invalid response or a transient API error -- a fresh
    sample/request is what actually helps, since re-parsing the same bad text
    can't succeed.

    Rate limits (429s) get their own independent retry budget with a longer,
    capped exponential backoff (up to rate_limit_backoff_cap seconds) instead
    of sharing max_retries' short generic schedule -- a burst of 429s under
    high parallel_workers is worth waiting out rather than giving up on."""
    last_error: Exception = ValueError("call_llm_and_parse: no attempts made")
    last_raw = ""
    attempt = 0
    rate_limit_attempt = 0
    while True:
        try:
            raw = call_llm(model, prompt, temperature=temperature)
            last_raw = raw
            if not raw.strip():
                raise ValueError("empty response from LLM")
            return parse_json_response(raw)
        except RATE_LIMIT_ERRORS as e:
            last_error = e
            if rate_limit_attempt >= rate_limit_max_retries:
                break
            time.sleep(min(2 ** rate_limit_attempt, rate_limit_backoff_cap))
            rate_limit_attempt += 1
        except RETRYABLE_ERRORS as e:
            last_error = e
            if attempt >= max_retries:
                break
            time.sleep(2 ** attempt)  # 1s, 2s, 4s, ...
            attempt += 1
    # Surface a preview of the last raw response that failed to parse -- the
    # original exception alone (e.g. "Expecting value: line 1 column 1")
    # gives no way to tell what the model actually returned.
    preview = last_raw.strip()[:500]
    raise RuntimeError(
        f"call_llm_and_parse: giving up after {attempt} generic + {rate_limit_attempt} "
        f"rate-limit retries ({last_error}); last raw response: {preview!r}"
    ) from last_error
