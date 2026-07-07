"""
RQ1 outcome statistics — implements collusion_index_spec.md v2.

Operates post-hoc on the session JSONs saved by experiment_rq1._save()
(results/sessions/rq1_*.json). Produces:

  1. Per-session measures table (one row per session):
       bloc_ask_slope (primary), pre_swap_slope, post_swap_slope,
       session_judge_bloc (secondary), session_ci_bloc, final_window_ci_bloc
       (descriptive), t_star (judge-based), swap_round,
       excluded_from_swap_analysis, delta_ci, time_to_disruption, censored
  2. Condition-level comparisons (Mann-Whitney U + Cliff's delta):
       - round-0 conditions vs. their matching baseline, on bloc-ask-slope
         (primary) and session-mean bloc judge score (secondary)
       - e2 (silent) and e3 (reward) vs. the matching e1 vocal condition,
         on the timing-appropriate measure
       - swap conditions: within-session pre-swap vs. post-swap slope
         (Wilcoxon on the paired difference) is the primary swap comparison;
         the v1 delta-CI-vs-0 test is kept alongside as descriptive.
  3. The paper's main figure: J_t trajectories (with the theta threshold) as
     the main panel, bloc mean ask as a second panel, per-session swap-round
     markers.

v1's ask-based collusion index (collusion_index.py) is retained throughout
but demoted to a descriptive/robustness metric: the sanity run showed it
doesn't separate colluding from competitive sessions (round-1 anchoring
noise dominates), whereas the LLM-judge coordination score does. See
collusion_index_spec.md v2 and judge_index.py.

Exclusion rule (spec section 4): swap sessions where collusion (now judge-
based) is not established by round R_max - k - 5 (i.e. swap_round > R_max -
k - 4) are excluded from swap analyses. They still appear in the per-session
table, flagged, as data about collusion emergence rates.

Usage:
    python -m src.analysis.rq1_stats                       # default results/sessions
    python -m src.analysis.rq1_stats --results-dir path    # custom dir
"""
import argparse
import glob
import json
import os
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as sps

from src.evaluation.collusion_index import CollusionIndexConfig
from src.evaluation.judge_index import JudgeConfig

CI_CFG = CollusionIndexConfig()
JUDGE_CFG = JudgeConfig()

# (test_condition, reference_condition, measure) — measure=None means
# "pick by timing": bloc_ask_slope for round0, post_swap_slope for swap.
CANONICAL_COMPARISONS = [
    ("e1_gpt_vocal_r0",     "e0_all_gpt41",       "bloc_ask_slope"),
    ("e1_mixed_vocal_r0",   "e0_mixed",           "bloc_ask_slope"),
    ("e1_gpt_vocal_swap",   "e0_all_gpt41",       "final_window_ci_bloc"),
    ("e1_mixed_vocal_swap", "e0_mixed",           "final_window_ci_bloc"),
    ("e2_gpt_silent",       None,                 None),  # vs matching e1, by timing
    ("e3_gpt_vocal_reward", None,                 None),  # vs matching e1, by timing
]


# ---------------------------------------------------------------- loading

def load_sessions(results_dir: str = "results/sessions", prefix: str = "rq1_") -> list[dict]:
    sessions = []
    for path in sorted(glob.glob(os.path.join(results_dir, f"{prefix}*.json"))):
        with open(path) as f:
            sessions.append(json.load(f))
    return sessions


# ------------------------------------------------------- per-session math

def swap_exclusion_cutoff(num_rounds: int, k: int = JUDGE_CFG.k) -> int:
    """Latest T* (1-indexed round) still admitted to swap analysis. T* is now
    judge-based (runner.py's SwapController), so k defaults from JudgeConfig;
    numerically identical to the v1 CI-based k unless recalibrated."""
    return num_rounds - k - 5


def delta_ci(ci_series: list[Optional[float]], swap_round: int, k: int = CI_CFG.k) -> Optional[float]:
    """Mean CI over the post-swap window minus mean CI over the k rounds
    preceding the swap (spec section 5). swap_round is 1-indexed."""
    pre = [c for c in ci_series[max(0, swap_round - 1 - k): swap_round - 1] if c is not None]
    post = [c for c in ci_series[swap_round - 1:] if c is not None]
    if not pre or not post:
        return None
    return float(np.mean(post) - np.mean(pre))


def time_to_disruption(
    ci_series: list[Optional[float]],
    swap_round: int,
    tau: float = CI_CFG.tau,
    k: int = CI_CFG.k,
) -> tuple[int, bool]:
    """Rounds elapsed from swap_round (inclusive) to the end of the first
    k-length window of consecutive rounds with CI < tau. Returns
    (rounds_elapsed, censored). Censored=True means the session ended
    before disruption; rounds_elapsed is then the observation length."""
    n = len(ci_series)
    for d in range(swap_round, n - k + 2):          # d = 1-indexed window start
        window = ci_series[d - 1: d - 1 + k]
        if len(window) == k and all(c is not None and c < tau for c in window):
            return (d + k - 1) - swap_round + 1, False
    return n - swap_round + 1, True


def compute_bloc_ask_series(session: dict) -> list[Optional[float]]:
    """Bloc mean ask per round (honest agent's ask excluded), derived from
    market_history -- unlike round_metrics.ask_price_mean, which is
    market-wide and includes the honest agent's ask."""
    series = []
    for rnd in session.get("market_history") or []:
        asks = rnd.get("asks") or {}
        honest_ids = set(rnd.get("honest_seller_ids") or [])
        bloc_asks = [p for sid, p in asks.items() if sid not in honest_ids]
        series.append(float(np.mean(bloc_asks)) if bloc_asks else None)
    return series


def _ols_slope(values: list[Optional[float]]) -> Optional[float]:
    """OLS slope of values against their (0-indexed) round position, dropping
    None/NaN entries. None if fewer than 2 valid points remain."""
    pts = [
        (i, v) for i, v in enumerate(values)
        if v is not None and not (isinstance(v, float) and np.isnan(v))
    ]
    if len(pts) < 2:
        return None
    x = np.array([p[0] for p in pts], dtype=float)
    y = np.array([p[1] for p in pts], dtype=float)
    slope, _intercept = np.polyfit(x, y, 1)
    return float(slope)


def bloc_ask_slope(bloc_ask_series: list[Optional[float]], burn_in: int = CI_CFG.burn_in) -> Optional[float]:
    """Primary outcome (v2): OLS slope of bloc mean ask over post-burn-in
    rounds. Hypothesis: baselines hold (slope >= 0), honest conditions
    restore erosion (slope < 0)."""
    return _ols_slope(bloc_ask_series[burn_in:])


def pre_post_swap_slopes(
    bloc_ask_series: list[Optional[float]], swap_round: int, burn_in: int = CI_CFG.burn_in
) -> tuple[Optional[float], Optional[float]]:
    """Pre-swap slope (post-burn-in through swap_round-1) and post-swap slope
    (swap_round through session end). Replaces delta_ci as the primary swap
    outcome (spec v2 section 5); swap_round is 1-indexed."""
    pre = bloc_ask_series[burn_in: swap_round - 1]
    post = bloc_ask_series[swap_round - 1:]
    return _ols_slope(pre), _ols_slope(post)


def session_measures(session: dict) -> dict:
    cs = session.get("collusion_summary") or {}
    ha = session.get("honest_agent") or {}
    ci = cs.get("ci_series_bloc") or []
    num_rounds = len(ci)
    timing = ha.get("timing")
    swap_round = ha.get("swap_round")

    bloc_ask_series = compute_bloc_ask_series(session)

    excluded = False
    d_ci = None
    ttd = None
    censored = None
    pre_slope = None
    post_slope = None
    if ha.get("present") and timing == "swap":
        cutoff = swap_exclusion_cutoff(num_rounds)
        excluded = swap_round is None or (swap_round - 1) > cutoff
        if not excluded:
            d_ci = delta_ci(ci, swap_round)
            ttd, censored = time_to_disruption(ci, swap_round)
            pre_slope, post_slope = pre_post_swap_slopes(bloc_ask_series, swap_round)

    return {
        "session_id": session.get("session_id"),
        "condition": session.get("condition"),
        "timing": timing,
        "mode": ha.get("mode"),
        "num_rounds": num_rounds,
        "bloc_ask_slope": bloc_ask_slope(bloc_ask_series),
        "pre_swap_slope": pre_slope,
        "post_swap_slope": post_slope,
        "session_judge_bloc": cs.get("session_judge_bloc"),
        "session_ci_bloc": cs.get("session_ci_bloc"),
        "session_ci_market": cs.get("session_ci_market"),
        "final_window_ci_bloc": cs.get("final_window_ci_bloc"),
        "session_f_bloc": cs.get("session_f_bloc"),
        "t_star": cs.get("t_star"),
        "swap_round": swap_round,
        "excluded_from_swap_analysis": excluded,
        "delta_ci": d_ci,
        "time_to_disruption": ttd,
        "censored": censored,
    }


def measures_table(sessions: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([session_measures(s) for s in sessions])


# ------------------------------------------------------------- statistics

def cliffs_delta(x: list[float], y: list[float]) -> float:
    """Cliff's delta effect size: P(x > y) - P(x < y). Range [-1, 1]."""
    gt = sum(1 for a in x for b in y if a > b)
    lt = sum(1 for a in x for b in y if a < b)
    n = len(x) * len(y)
    return (gt - lt) / n if n else float("nan")


def compare(
    df: pd.DataFrame,
    condition_a: str,
    condition_b: str,
    measure: str,
) -> Optional[dict]:
    """Mann-Whitney U (two-sided) + Cliff's delta between two conditions on
    one measure, session as unit of analysis. Swap-excluded sessions are
    dropped when the measure is swap-specific."""
    swap_only_measures = (
        "delta_ci", "time_to_disruption", "final_window_ci_bloc",
        "pre_swap_slope", "post_swap_slope",
    )

    def _values(cond):
        sub = df[df["condition"] == cond]
        if measure in swap_only_measures:
            sub = sub[~sub["excluded_from_swap_analysis"].fillna(False)]
        return sub[measure].dropna().tolist()

    a, b = _values(condition_a), _values(condition_b)
    if len(a) < 2 or len(b) < 2:
        return None
    u, p = sps.mannwhitneyu(a, b, alternative="two-sided")
    return {
        "condition_a": condition_a,
        "condition_b": condition_b,
        "measure": measure,
        "n_a": len(a),
        "n_b": len(b),
        "median_a": float(np.median(a)),
        "median_b": float(np.median(b)),
        "mannwhitney_u": float(u),
        "p_value": float(p),
        "cliffs_delta": cliffs_delta(a, b),
    }


def _matching_e1(df: pd.DataFrame, condition: str) -> Optional[tuple[str, str]]:
    """For e2/e3, find the e1 vocal condition with the same bloc and timing,
    and the timing-appropriate primary measure. post_swap_slope is a fair
    common yardstick here (unlike in CANONICAL_COMPARISONS' baseline rows)
    since both sides of this comparison actually undergo a swap."""
    rows = df[df["condition"] == condition]
    if rows.empty:
        return None
    timing = rows["timing"].iloc[0]
    e1 = "e1_gpt_vocal_r0" if timing == "round0" else "e1_gpt_vocal_swap"
    measure = "bloc_ask_slope" if timing == "round0" else "post_swap_slope"
    return e1, measure


def comparisons_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cond_a, cond_b, measure in CANONICAL_COMPARISONS:
        resolved_b, resolved_measure = cond_b, measure
        if resolved_b is None:
            match = _matching_e1(df, cond_a)
            if match is None:
                continue
            resolved_b, resolved_measure = match
        result = compare(df, cond_a, resolved_b, resolved_measure)
        if result:
            rows.append(result)
        # Secondary outcome: session-mean bloc judge score, same condition pair.
        secondary = compare(df, cond_a, resolved_b, "session_judge_bloc")
        if secondary:
            rows.append(secondary)

    # Swap conditions: primary is the within-session pre- vs. post-swap slope
    # (paired Wilcoxon on the difference), replacing delta_ci as the headline
    # swap comparison. The v1 delta-CI-vs-0 test is kept alongside as
    # descriptive (CI is demoted, not deleted -- collusion_index_spec.md v2).
    for cond in df.loc[df["timing"] == "swap", "condition"].unique():
        sub = df[(df["condition"] == cond) & (~df["excluded_from_swap_analysis"].fillna(False))]

        paired = sub[["pre_swap_slope", "post_swap_slope"]].dropna()
        if len(paired):
            diff = paired["post_swap_slope"] - paired["pre_swap_slope"]
            rows.append({
                "condition_a": cond, "condition_b": "(within: pre-swap slope)",
                "measure": "post_swap_slope - pre_swap_slope", "n_a": len(diff), "n_b": len(diff),
                "median_a": float(diff.median()), "median_b": 0.0,
                "mannwhitney_u": float("nan"),
                "p_value": float(sps.wilcoxon(diff).pvalue) if len(diff) >= 5 else float("nan"),
                "cliffs_delta": float("nan"),
            })

        vals = sub["delta_ci"].dropna()
        if len(vals):
            rows.append({
                "condition_a": cond, "condition_b": "(within: pre-swap window, CI descriptive)",
                "measure": "delta_ci", "n_a": len(vals), "n_b": len(vals),
                "median_a": float(vals.median()), "median_b": 0.0,
                "mannwhitney_u": float("nan"),
                "p_value": float(sps.wilcoxon(vals).pvalue) if len(vals) >= 5 else float("nan"),
                "cliffs_delta": float("nan"),
            })
    return pd.DataFrame(rows)


# ------------------------------------------------------------ main figure

def _plot_condition_bands(ax, by_cond: dict[str, list[dict]], series_fn, colors) -> None:
    """Plots per-condition mean +/- IQR band trajectories on ax, plus one
    dotted swap-round marker per session. series_fn(session) -> the round
    series to plot for that session (list[Optional[float]])."""
    for idx, (cond, sess_list) in enumerate(sorted(by_cond.items())):
        series = [series_fn(s) or [] for s in sess_list]
        max_len = max((len(x) for x in series), default=0)
        if max_len == 0:
            continue
        mat = np.full((len(series), max_len), np.nan)
        for i, vals in enumerate(series):
            mat[i, : len(vals)] = [np.nan if v is None else v for v in vals]
        rounds = np.arange(1, max_len + 1)
        mean = np.nanmean(mat, axis=0)
        q25 = np.nanpercentile(mat, 25, axis=0)
        q75 = np.nanpercentile(mat, 75, axis=0)
        color = colors[idx % len(colors)]
        ax.plot(rounds, mean, label=cond, color=color, linewidth=2)
        ax.fill_between(rounds, q25, q75, color=color, alpha=0.15)

        swap_rounds = [
            (s.get("honest_agent") or {}).get("swap_round")
            for s in sess_list
            if (s.get("honest_agent") or {}).get("swap_round")
        ]
        for sr in swap_rounds:
            ax.axvline(sr, color=color, alpha=0.25, linewidth=1, linestyle=":")


def rq1_trajectory_figure(
    sessions: list[dict],
    output_path: str = "results/plots/rq1_trajectory.png",
    conditions: Optional[list[str]] = None,
):
    """The letter's main figure (v2): bloc-mean judge coordination score J_t
    (with the theta threshold) as the main panel -- the signal that actually
    separates colluding from competitive sessions and drives the swap
    trigger -- with bloc mean ask as a second, descriptive panel. Both panels
    share mean +/- IQR bands and per-session swap-round markers."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_cond: dict[str, list[dict]] = {}
    for s in sessions:
        by_cond.setdefault(s["condition"], []).append(s)
    if conditions:
        by_cond = {c: by_cond[c] for c in conditions if c in by_cond}

    fig, (ax_judge, ax_ask) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    colors = plt.cm.tab10.colors

    _plot_condition_bands(
        ax_judge, by_cond,
        lambda s: (s.get("collusion_summary") or {}).get("judge_series_bloc"),
        colors,
    )
    ax_judge.axhline(JUDGE_CFG.theta, color="black", linestyle="--", linewidth=1, label=f"theta = {JUDGE_CFG.theta}")
    ax_judge.set_ylabel("Bloc judge coordination score  J_t  (1-4)")
    ax_judge.set_title("RQ1: bloc judge-score trajectories (main) and bloc mean ask (descriptive)")
    ax_judge.legend(fontsize=8, loc="best")

    _plot_condition_bands(ax_ask, by_cond, compute_bloc_ask_series, colors)
    ax_ask.axhline(90.0, color="gray", linewidth=0.8, label="p* = $90")
    ax_ask.set_xlabel("Round")
    ax_ask.set_ylabel("Bloc mean ask ($)")
    ax_ask.legend(fontsize=8, loc="best")

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


# ------------------------------------------------------------------- CLI

def run_report(results_dir: str = "results/sessions", output_dir: str = "results") -> dict:
    sessions = load_sessions(results_dir)
    if not sessions:
        print(f"No rq1_*.json sessions found in {results_dir}")
        return {}
    df = measures_table(sessions)
    comps = comparisons_table(df)

    os.makedirs(output_dir, exist_ok=True)
    measures_path = os.path.join(output_dir, "rq1_session_measures.csv")
    comps_path = os.path.join(output_dir, "rq1_comparisons.csv")
    df.to_csv(measures_path, index=False)
    comps.to_csv(comps_path, index=False)
    fig_path = rq1_trajectory_figure(
        sessions, output_path=os.path.join(output_dir, "plots", "rq1_trajectory.png")
    )

    print("\n=== RQ1 per-session measures ===")
    print(df.to_string(index=False))
    print("\n=== RQ1 comparisons (Mann-Whitney U, Cliff's delta) ===")
    print(comps.to_string(index=False) if not comps.empty else "(not enough data yet)")
    excluded = df["excluded_from_swap_analysis"].fillna(False).sum()
    if excluded:
        print(f"\nNote: {int(excluded)} swap session(s) excluded (collusion not established by cutoff).")
    print(f"\nSaved -> {measures_path}\nSaved -> {comps_path}\nSaved -> {fig_path}")
    return {"measures": df, "comparisons": comps, "figure": fig_path}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RQ1 outcome statistics")
    parser.add_argument("--results-dir", default="results/sessions")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()
    run_report(results_dir=args.results_dir, output_dir=args.output_dir)
