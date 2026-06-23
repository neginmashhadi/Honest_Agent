import json
import os
from src.agents.base_agent import render_template, call_llm, parse_json_response


EVALUATOR_MODEL = "gpt-4.1-mini"
EVALUATOR_TEMPERATURE = 0.1


def evaluate_seller_round(
    seller_id: str,
    round_number: int,
    reasoning: dict,
    evaluator_model: str = EVALUATOR_MODEL,
) -> dict:
    """Score a single seller's reasoning for a single round using an LLM judge."""
    prompt = render_template(
        "evaluator_prompt.j2",
        hour=round_number,
        hour_reasoning_data=reasoning,
    )

    raw = call_llm(evaluator_model, prompt, temperature=EVALUATOR_TEMPERATURE)
    try:
        result = parse_json_response(raw)
    except (json.JSONDecodeError, ValueError):
        # Fallback: return neutral score on parse failure
        result = {"coordination": "no", "score": 1, "type": "null", "evidence": []}

    result["seller_id"] = seller_id
    result["round_number"] = round_number
    return result


def batch_evaluate(
    seller_id: str,
    round_reasonings: list[tuple[int, dict]],
    evaluator_model: str = EVALUATOR_MODEL,
) -> list[dict]:
    """Evaluate all rounds for one seller."""
    results = []
    for round_number, reasoning in round_reasonings:
        result = evaluate_seller_round(seller_id, round_number, reasoning, evaluator_model)
        results.append(result)
    return results
