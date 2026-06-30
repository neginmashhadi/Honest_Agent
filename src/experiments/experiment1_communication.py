"""
Experiment 1 – Seller Communication
RQ1: How does natural language communication between sellers affect collusion?
Conditions: with_comms vs without_comms, 10 sessions each.
"""
import json
import os
from src.config import MarketConfig, ExperimentConfig
from src.experiments.runner import run_session, SessionResult


def run_experiment1(
    num_sessions: int = 10,
    seller_model: str = "gpt-4.1-2025-04-14",
    buyer_model: str = "gpt-4.1-2025-04-14",
    output_dir: str = "results/sessions",
    verbose: bool = False,
    market_cfg: MarketConfig = None,
    seed: int = 904058464,
) -> dict[str, list[SessionResult]]:
    if market_cfg is None:
        market_cfg = MarketConfig()
    results = {"with_comms": [], "without_comms": []}

    for condition, comms in [("with_comms", True), ("without_comms", False)]:
        exp_cfg = ExperimentConfig(
            seller_comms_enabled=comms,
            oversight_enabled=False,
            boss_pressure=False,
            seller_model=seller_model,
            buyer_model=buyer_model,
            seed=seed,
        )
        for i in range(num_sessions):
            session_id = f"exp1_{condition}_{i}"
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
