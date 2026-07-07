"""
Experiment RQ1 - Honest Agent
RQ1: Can an honest LLM agent disrupt emergent price collusion in a continuous
double auction?

Conditions (honest_agent_spec.md section 5):
    e0_all_gpt41         5x GPT-4.1                       no honest agent    (baseline)
    e0_mixed             3x GPT-4.1 + 2x Claude            no honest agent    (baseline)
    e1_gpt_vocal_r0      4x GPT-4.1 + honest               vocal, round 0
    e1_gpt_vocal_swap    5x GPT-4.1 -> swap                vocal, swap at T*+1
    e1_mixed_vocal_r0    2x GPT-4.1 + 2x Claude + honest    vocal, round 0
    e1_mixed_vocal_swap  3x GPT-4.1 + 2x Claude -> swap    vocal, swap at T*+1
    e2_gpt_silent        4x GPT-4.1 + honest                silent (ablation)
    e3_gpt_vocal_reward  4x GPT-4.1 + honest                vocal + social reward (exploratory)

All conditions: oversight_enabled=False, boss_pressure=False, seller
communication ON (colluders need the channel to collude; the honest-vocal
agent needs it to persuade).

The honest agent always occupies seller slot index 2 ("S3"), fixed by index
for reproducibility (not chosen by price behavior), per section 4b.

e2's timing is an open item (honest_agent_spec.md section 5: "whichever timing
E1 favors") -- provisional default is "swap" (the harder, more informative
disruption case); override via `e2_timing` once E1 results decide it. e3
shares the same default (see CLAUDE.md: e1/e2/e3 form a mechanism ladder --
silent (undercut only) -> vocal (call-out) -> reward (call-out + praise)).
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
from src.config import MarketConfig, ExperimentConfig
from src.experiments.runner import run_session, SessionResult

GPT = "gpt-4.1-2025-04-14"
CLAUDE = "claude-sonnet-4-6"

GPT_BLOC = [GPT, GPT, GPT, GPT, GPT]
MIXED_BLOC = [GPT, GPT, GPT, CLAUDE, CLAUDE]

HONEST_SELLER_INDEX = 2  # "S3", fixed for reproducibility


def _condition_configs(e2_timing: str = "swap") -> dict[str, dict]:
    return {
        "e0_all_gpt41": {
            "seller_models": GPT_BLOC,
            "honest_agent_enabled": False,
        },
        "e0_mixed": {
            "seller_models": MIXED_BLOC,
            "honest_agent_enabled": False,
        },
        "e1_gpt_vocal_r0": {
            "seller_models": GPT_BLOC,
            "honest_agent_enabled": True,
            "honest_agent_mode": "vocal",
            "honest_agent_timing": "round0",
        },
        "e1_gpt_vocal_swap": {
            "seller_models": GPT_BLOC,
            "honest_agent_enabled": True,
            "honest_agent_mode": "vocal",
            "honest_agent_timing": "swap",
        },
        "e1_mixed_vocal_r0": {
            "seller_models": MIXED_BLOC,
            "honest_agent_enabled": True,
            "honest_agent_mode": "vocal",
            "honest_agent_timing": "round0",
        },
        "e1_mixed_vocal_swap": {
            "seller_models": MIXED_BLOC,
            "honest_agent_enabled": True,
            "honest_agent_mode": "vocal",
            "honest_agent_timing": "swap",
        },
        "e2_gpt_silent": {
            "seller_models": GPT_BLOC,
            "honest_agent_enabled": True,
            "honest_agent_mode": "silent",
            "honest_agent_timing": e2_timing,
        },
        # E3 (exploratory): social-reinforcement variant. Identical to
        # e1_gpt_vocal_* except the persona also PRAISES sellers who price
        # independently (positive reinforcement), rather than only calling
        # out coordination. Words only -- the environment has no transfer
        # mechanism and the prompt explicitly forbids material inducements,
        # so this stays on the right side of the "honest" framing.
        "e3_gpt_vocal_reward": {
            "seller_models": GPT_BLOC,
            "honest_agent_enabled": True,
            "honest_agent_mode": "vocal_reward",
            "honest_agent_timing": e2_timing,
        },
    }


def run_experiment_rq1(
    num_sessions: int = 10,
    output_dir: str = "results/sessions",
    verbose: bool = False,
    seed: int = 904058464,
    e2_timing: str = "swap",
    conditions: list[str] = None,
    parallel_workers: int = 1,
) -> dict[str, list[SessionResult]]:
    """Runs every (condition, session_index) cell. Sequential by default;
    pass parallel_workers > 1 to run sessions concurrently via
    ThreadPoolExecutor -- sessions are fully independent (each seeded off its
    own session_index, see runner.run_session) and each writes its own JSON
    file, so there's no shared state or write contention across threads."""
    market_cfg = MarketConfig()
    condition_configs = _condition_configs(e2_timing=e2_timing)
    if conditions is not None:
        condition_configs = {c: condition_configs[c] for c in conditions}

    results: dict[str, list[Optional[SessionResult]]] = {
        cond: [None] * num_sessions for cond in condition_configs
    }

    jobs = []
    for condition, overrides in condition_configs.items():
        exp_cfg = ExperimentConfig(
            seller_comms_enabled=True,
            oversight_enabled=False,
            boss_pressure=False,
            seller_model=GPT,
            buyer_model=GPT,
            seed=seed,
            honest_agent_seller_index=HONEST_SELLER_INDEX,
            **overrides,
        )
        for i in range(num_sessions):
            jobs.append((condition, exp_cfg, i))

    def _run_one(condition: str, exp_cfg: ExperimentConfig, i: int, session_verbose: bool) -> SessionResult:
        session_id = f"rq1_{condition}_{i}"
        result = run_session(
            session_id=session_id,
            condition=condition,
            market_cfg=market_cfg,
            exp_cfg=exp_cfg,
            evaluate=True,
            verbose=session_verbose,
            session_index=i,
        )
        _save(result, output_dir)
        return result

    if parallel_workers <= 1:
        for condition, exp_cfg, i in jobs:
            print(f"\n=== rq1_{condition}_{i} ===")
            result = _run_one(condition, exp_cfg, i, session_verbose=verbose)
            results[condition][i] = result
    else:
        # Per-session tqdm bars (and the WARN prints gated by the same verbose
        # flag) would garble the terminal if run concurrently -- they all
        # assume exclusive control of the cursor/stdout -- so progress in
        # parallel mode is a single completion counter instead.
        total = len(jobs)
        completed = 0
        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            futures = {
                executor.submit(_run_one, condition, exp_cfg, i, False): (condition, i)
                for condition, exp_cfg, i in jobs
            }
            for future in as_completed(futures):
                condition, i = futures[future]
                results[condition][i] = future.result()
                completed += 1
                print(f"[{completed}/{total}] done: rq1_{condition}_{i}")

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
            "honest_agent": result.honest_agent,
            "collusion_summary": result.collusion_summary,
        }, f, indent=2)
    print(f"  Saved -> {path}")
