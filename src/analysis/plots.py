"""
Reproduce paper Figures 2, 3, and Table 1.
All functions accept a dict mapping condition_name -> list of SessionResult.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

COMPETITIVE_EQ = 90.0


# ── helpers ─────────────────────────────────────────────────────────────────

def _coord_series(sessions) -> pd.DataFrame:
    """Mean coordination score per round across sellers and sessions."""
    all_rows = []
    for sess in sessions:
        by_round: dict[int, list[float]] = {}
        for ev in sess.eval_scores:
            r = ev["round_number"]
            s = ev.get("score", 1)
            by_round.setdefault(r, []).append(s)
        for r, scores in by_round.items():
            all_rows.append({"round": r, "score": np.mean(scores)})
    df = pd.DataFrame(all_rows)
    return df.groupby("round")["score"].agg(["mean", "sem"]).reset_index()


def _ask_series(sessions) -> pd.DataFrame:
    all_rows = []
    for sess in sessions:
        for rm in sess.round_metrics:
            if rm["ask_price_mean"] is not None:
                all_rows.append({"round": rm["round"], "ask": rm["ask_price_mean"]})
    df = pd.DataFrame(all_rows)
    return df.groupby("round")["ask"].agg(["mean", "sem"]).reset_index()


def _disp_series(sessions) -> pd.DataFrame:
    all_rows = []
    for sess in sessions:
        for rm in sess.round_metrics:
            all_rows.append({"round": rm["round"], "disp": rm["ask_dispersion"]})
    df = pd.DataFrame(all_rows)
    return df.groupby("round")["disp"].agg(["mean", "sem"]).reset_index()


def _profit_ratio_series(sessions) -> pd.DataFrame:
    """Total seller profit / total trade price per round (Figure 4)."""
    all_rows = []
    for sess in sessions:
        cum_profit = 0.0
        cum_trade_val = 0.0
        for rm in sess.round_metrics:
            cum_profit += rm["seller_profit"]
            if rm["trade_price_mean"] and rm["num_trades"]:
                cum_trade_val += rm["trade_price_mean"] * rm["num_trades"]
            ratio = cum_profit / cum_trade_val if cum_trade_val > 0 else np.nan
            all_rows.append({"round": rm["round"], "ratio": ratio})
    df = pd.DataFrame(all_rows)
    return df.groupby("round")["ratio"].agg(["mean", "sem"]).reset_index()


def _ci95(mean, sem):
    return 1.96 * sem


def _plot_band(ax, df, label, color):
    ci = _ci95(df["mean"], df["sem"])
    ax.plot(df["round"], df["mean"], label=label, color=color)
    ax.fill_between(df["round"], df["mean"] - ci, df["mean"] + ci, alpha=0.2, color=color)


COLORS = ["#2196F3", "#F44336", "#4CAF50", "#9C27B0", "#FF9800"]


# ── Figure 2: Coordination scores ───────────────────────────────────────────

def figure2_coordination(
    conditions: dict[str, list],
    title: str,
    output_path: str,
):
    fig, ax = plt.subplots(figsize=(5, 4))
    for (cond, sessions), color in zip(conditions.items(), COLORS):
        df = _coord_series(sessions)
        if not df.empty:
            _plot_band(ax, df, label=cond.replace("_", " "), color=color)

    ax.set_xlabel("Round")
    ax.set_ylabel("Coordination Score")
    ax.set_title(title)
    ax.set_ylim(1, 4)
    ax.legend(fontsize=8)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")


# ── Figure 3: Ask price + Ask dispersion ───────────────────────────────────

def figure3_ask_metrics(
    conditions: dict[str, list],
    title: str,
    output_path: str,
):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 7), sharex=True)

    for (cond, sessions), color in zip(conditions.items(), COLORS):
        ask_df = _ask_series(sessions)
        disp_df = _disp_series(sessions)
        label = cond.replace("_", " ")
        if not ask_df.empty:
            _plot_band(ax1, ask_df, label=label, color=color)
        if not disp_df.empty:
            _plot_band(ax2, disp_df, label=label, color=color)

    ax1.axhline(COMPETITIVE_EQ, color="gray", linestyle="--", linewidth=0.8, label="Competitive Eq.")
    ax1.set_ylabel("Ask Price ($)")
    ax1.set_title(title)
    ax1.legend(fontsize=7)

    ax2.set_xlabel("Round")
    ax2.set_ylabel("Ask Dispersion (std)")
    ax2.legend(fontsize=7)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")


# ── Figure 4: Profit/trade-price ratio ──────────────────────────────────────

def figure4_profit_ratio(
    conditions: dict[str, list],
    title: str,
    output_path: str,
):
    fig, ax = plt.subplots(figsize=(5, 4))
    for (cond, sessions), color in zip(conditions.items(), COLORS):
        df = _profit_ratio_series(sessions)
        if not df.empty:
            _plot_band(ax, df, label=cond.replace("_", " "), color=color)

    ax.set_xlabel("Round")
    ax.set_ylabel("Profit / Trade Price")
    ax.set_title(title)
    ax.legend(fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")


# ── Table 1: Summary statistics ─────────────────────────────────────────────

def table1_summary(all_conditions: dict[str, list]) -> pd.DataFrame:
    rows = []
    for cond, sessions in all_conditions.items():
        trade_prices, total_profits = [], []
        session_cis, t_stars = [], []
        for sess in sessions:
            tp = [rm["trade_price_mean"] for rm in sess.round_metrics if rm["trade_price_mean"] is not None]
            profit = sum(rm["seller_profit"] for rm in sess.round_metrics)
            if tp:
                trade_prices.append(np.mean(tp))
            total_profits.append(profit)

            ci_summary = getattr(sess, "collusion_summary", None)
            if ci_summary:
                if ci_summary.get("session_ci_bloc") is not None:
                    session_cis.append(ci_summary["session_ci_bloc"])
                t_stars.append(ci_summary.get("t_star"))

        row = {
            "Condition": cond.replace("_", " "),
            "Avg Trade Price (mean)": np.mean(trade_prices) if trade_prices else np.nan,
            "Avg Trade Price (95% CI low)": np.mean(trade_prices) - 1.96 * np.std(trade_prices) / np.sqrt(len(trade_prices)) if trade_prices else np.nan,
            "Avg Trade Price (95% CI high)": np.mean(trade_prices) + 1.96 * np.std(trade_prices) / np.sqrt(len(trade_prices)) if trade_prices else np.nan,
            "Total Profit (mean)": np.mean(total_profits),
            "Total Profit (95% CI low)": np.mean(total_profits) - 1.96 * np.std(total_profits) / np.sqrt(len(total_profits)),
            "Total Profit (95% CI high)": np.mean(total_profits) + 1.96 * np.std(total_profits) / np.sqrt(len(total_profits)),
        }
        if t_stars:
            row["Session CI (bloc, mean)"] = np.mean(session_cis) if session_cis else np.nan
            row["Sessions Establishing Collusion"] = f"{sum(t is not None for t in t_stars)}/{len(t_stars)}"
        rows.append(row)
    return pd.DataFrame(rows)


# ── Composite Figure 2 (3-panel, matching paper layout) ──────────────────────

def composite_figure2(
    exp1: dict[str, list],
    exp2: dict[str, list],
    exp3: dict[str, list],
    output_path: str,
):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    panels = [
        (axes[0], exp1, "Seller Communication"),
        (axes[1], exp2, "Model Variation"),
        (axes[2], exp3, "Environmental Pressure"),
    ]
    for ax, conditions, title in panels:
        for (cond, sessions), color in zip(conditions.items(), COLORS):
            df = _coord_series(sessions)
            if not df.empty:
                _plot_band(ax, df, label=cond.replace("_", " "), color=color)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Round")
        ax.set_ylim(1, 4)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("Coordination Score")
    plt.suptitle("Average Coordination Score by Round", fontsize=12)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")
