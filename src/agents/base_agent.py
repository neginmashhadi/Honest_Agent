import json
import os
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
        return msg.content[0].text
    else:
        client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
        )
        return resp.choices[0].message.content


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
        text = "\n".join(lines)
    return json.loads(text)
