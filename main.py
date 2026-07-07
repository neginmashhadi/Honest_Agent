"""
Main entry point for replicating "Evaluating LLM Agent Collusion in Double Auctions".

Usage:
    python main.py --experiment 1          # Seller Communication
    python main.py --experiment 2          # Model Variation
    python main.py --experiment 3          # Environmental Pressure
    python main.py --experiment rq1        # Honest Agent (honest_agent_spec.md)
    python main.py --experiment all        # 1, 2, and 3
    python main.py --experiment 1 --sessions 2 --verbose   # Quick smoke test
"""
import argparse
import json
import os
from dotenv import load_dotenv

load_dotenv()

from src.config import MarketConfig
from src.experiments.experiment1_communication import run_experiment1
from src.experiments.experiment2_model_variation import run_experiment2
from src.experiments.experiment3_env_pressure import run_experiment3
from src.experiments.experiment_rq1 import run_experiment_rq1
from src.analysis.plots import (
    figure2_coordination,
    figure3_ask_metrics,
    figure4_profit_ratio,
    composite_figure2,
    table1_summary,
)


def main():
    parser = argparse.ArgumentParser(description="LLM Collusion in Double Auctions - Replication")
    parser.add_argument("--experiment", choices=["1", "2", "3", "rq1", "all"], default="1")
    parser.add_argument("--sessions", type=int, default=10, help="Sessions per condition")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--plots-only", action="store_true", help="Load saved results and only regenerate plots")
    parser.add_argument("--smoke", action="store_true", help="Smoke test: 2 sellers, 2 buyers, 5 rounds, 1 session, with_comms only")
    parser.add_argument("--seed", type=int, default=904058464, help="Global random seed for reproducibility")
    parser.add_argument("--parallel", type=int, default=1, help="RQ1 only: number of sessions to run concurrently")
    args = parser.parse_args()

    os.makedirs("results/sessions", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)

    if args.smoke:
        print("\n" + "="*60)
        print("SMOKE TEST: 2 sellers, 2 buyers, 5 rounds, 1 session")
        print("="*60)
        smoke_cfg = MarketConfig(num_sellers=2, num_buyers=2, num_rounds=5)
        smoke_results = run_experiment1(
            num_sessions=1,
            verbose=True,
            market_cfg=smoke_cfg,
            seed=args.seed,
        )
        _plot_exp1(smoke_results)
        print("\nSmoke test complete. Check results/sessions/ for session JSON.")
        return

    run_exp1 = args.experiment in ("1", "all")
    run_exp2 = args.experiment in ("2", "all")
    run_exp3 = args.experiment in ("3", "all")
    run_rq1 = args.experiment == "rq1"

    exp1_results = exp2_results = exp3_results = rq1_results = None

    if run_exp1 and not args.plots_only:
        print("\n" + "="*60)
        print("EXPERIMENT 1: Seller Communication")
        print("="*60)
        exp1_results = run_experiment1(
            num_sessions=args.sessions,
            verbose=args.verbose,
            seed=args.seed,
        )
        _plot_exp1(exp1_results)

    if run_exp2 and not args.plots_only:
        print("\n" + "="*60)
        print("EXPERIMENT 2: Model Variation")
        print("="*60)
        exp2_results = run_experiment2(
            num_sessions=args.sessions,
            verbose=args.verbose,
            seed=args.seed,
        )
        _plot_exp2(exp2_results)

    if run_exp3 and not args.plots_only:
        print("\n" + "="*60)
        print("EXPERIMENT 3: Environmental Pressure")
        print("="*60)
        exp3_results = run_experiment3(
            num_sessions=args.sessions,
            verbose=args.verbose,
            seed=args.seed,
        )
        _plot_exp3(exp3_results)

    if run_rq1 and not args.plots_only:
        print("\n" + "="*60)
        print("EXPERIMENT RQ1: Honest Agent")
        print("="*60)
        rq1_results = run_experiment_rq1(
            num_sessions=args.sessions,
            verbose=args.verbose,
            seed=args.seed,
            parallel_workers=args.parallel,
        )
        _plot_rq1(rq1_results)

    # Print summary table
    all_results = {}
    if exp1_results:
        all_results.update(exp1_results)
    if exp2_results:
        all_results.update(exp2_results)
    if exp3_results:
        all_results.update(exp3_results)
    if rq1_results:
        all_results.update(rq1_results)

    if all_results:
        print("\n" + "="*60)
        print("TABLE 1: Summary Statistics")
        print("="*60)
        df = table1_summary(all_results)
        print(df.to_string(index=False))
        df.to_csv("results/table1_summary.csv", index=False)
        print("\nSaved -> results/table1_summary.csv")


def _plot_exp1(results):
    figure2_coordination(results, title="Seller Communication", output_path="results/plots/exp1_coordination.png")
    figure3_ask_metrics(results, title="Seller Communication", output_path="results/plots/exp1_ask_metrics.png")
    figure4_profit_ratio(results, title="Seller Communication", output_path="results/plots/exp1_profit_ratio.png")


def _plot_exp2(results):
    figure2_coordination(results, title="Model Variation", output_path="results/plots/exp2_coordination.png")
    figure3_ask_metrics(results, title="Model Variation", output_path="results/plots/exp2_ask_metrics.png")
    figure4_profit_ratio(results, title="Model Variation", output_path="results/plots/exp2_profit_ratio.png")


def _plot_exp3(results):
    figure2_coordination(results, title="Environmental Pressure", output_path="results/plots/exp3_coordination.png")
    figure3_ask_metrics(results, title="Environmental Pressure", output_path="results/plots/exp3_ask_metrics.png")
    figure4_profit_ratio(results, title="Environmental Pressure", output_path="results/plots/exp3_profit_ratio.png")


def _plot_rq1(results):
    figure2_coordination(results, title="Honest Agent", output_path="results/plots/rq1_coordination.png")
    figure3_ask_metrics(results, title="Honest Agent", output_path="results/plots/rq1_ask_metrics.png")
    figure4_profit_ratio(results, title="Honest Agent", output_path="results/plots/rq1_profit_ratio.png")


if __name__ == "__main__":
    main()
