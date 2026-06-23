"""
Experiment 3 – Environmental Pressure
RQ3: How do oversight and urgency affect collusion?
Conditions:
  - no_oversight_no_urgency  (baseline)
  - oversight_only
  - urgency_only
  - oversight_and_urgency
"""
import json
import os
from src.config import MarketConfig, ExperimentConfig
from src.experiments.runner import run_session, SessionResult

GPT = "gpt-4.1-2025-04-14"

CONDITIONS = {
    "no_oversight_no_urgency": {"oversight_enabled": False, "boss_pressure": False},
    "oversight_only":          {"oversight_enabled": True,  "boss_pressure": False},
    "urgency_only":            {"oversight_enabled": False, "boss_pressure": True},
    "oversight_and_urgency":   {"oversight_enabled": True,  "boss_pressure": True},
}


def run_experiment3(
    num_sessions: int = 10,
    seller_model: str = GPT,
    buyer_model: str = GPT,
    output_dir: str = "results/sessions",
    verbose: bool = False,
) -> dict[str, list[SessionResult]]:
    market_cfg = MarketConfig()
    results = {cond: [] for cond in CONDITIONS}

    for condition, flags in CONDITIONS.items():
        exp_cfg = ExperimentConfig(
            seller_comms_enabled=True,
            seller_model=seller_model,
            buyer_model=buyer_model,
            **flags,
        )
        for i in range(num_sessions):
            session_id = f"exp3_{condition}_{i}"
            print(f"\n=== {session_id} ===")
            result = run_session(
                session_id=session_id,
                condition=condition,
                market_cfg=market_cfg,
                exp_cfg=exp_cfg,
                evaluate=True,
                verbose=verbose,
            )
            results[condition].append(result)
            _save(result, output_dir)

    return results


def _save(result: SessionResult, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{result.session_id}.json")
    with open(path, "w") as f:
        json.dump({
            "session_id": result.session_id,
            "condition": result.condition,
            "round_metrics": result.round_metrics,
            "eval_scores": result.eval_scores,
            "market_history": result.market_history,
            "final_profits": result.final_profits,
        }, f, indent=2)
    print(f"  Saved -> {path}")
