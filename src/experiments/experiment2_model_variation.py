"""
Experiment 2 – Model Variation
RQ2: How is collusion impacted by model choice?
Conditions:
  - all_gpt41: 5 GPT-4.1 sellers
  - all_claude: 5 Claude-3.7-Sonnet sellers
  - mixed: 3 GPT-4.1 + 2 Claude-3.7-Sonnet sellers
Buyers are a mix of Claude-3.7-Sonnet and GPT-4.1 (paper uses both).
"""
import json
import os
from src.config import MarketConfig, ExperimentConfig
from src.experiments.runner import run_session, SessionResult

GPT = "gpt-4.1-2025-04-14"
CLAUDE = "claude-3-7-sonnet-20250219"


CONDITION_MODELS = {
    "all_gpt41": [GPT] * 5,
    "mixed": [GPT, GPT, GPT, CLAUDE, CLAUDE],
    "all_claude": [CLAUDE] * 5,
}


def run_experiment2(
    num_sessions: int = 10,
    output_dir: str = "results/sessions",
    verbose: bool = False,
    seed: int = 904058464,
) -> dict[str, list[SessionResult]]:
    market_cfg = MarketConfig()
    results = {cond: [] for cond in CONDITION_MODELS}

    for condition, seller_models in CONDITION_MODELS.items():
        exp_cfg = ExperimentConfig(
            seller_comms_enabled=True,
            oversight_enabled=False,
            boss_pressure=False,
            seller_model=GPT,
            buyer_model=GPT,
            seller_models=seller_models,
            seed=seed,
        )
        for i in range(num_sessions):
            session_id = f"exp2_{condition}_{i}"
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
