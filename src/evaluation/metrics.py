import numpy as np
import pandas as pd
from src.market.market_state import MarketState, RoundHistory


def compute_round_metrics(history: list[RoundHistory], seller_ids: list[str]) -> pd.DataFrame:
    """Compute per-round metrics matching the paper's Table 1 and Figure 3."""
    rows = []
    for rh in history:
        asks = [rh.asks[sid] for sid in seller_ids if sid in rh.asks]
        trade_prices = [t.trade_price for t in rh.trades]
        seller_profits = sum(
            t.trade_price - 80.0  # seller_valuation
            for t in rh.trades
            if t.seller_id in seller_ids
        )

        rows.append({
            "round": rh.round_number,
            "ask_price_mean": np.mean(asks) if asks else np.nan,
            "ask_dispersion": np.std(asks, ddof=0) if len(asks) > 1 else 0.0,
            "trade_price_mean": np.mean(trade_prices) if trade_prices else np.nan,
            "num_trades": len(rh.trades),
            "seller_profit": seller_profits,
        })
    return pd.DataFrame(rows)


def compute_session_summary(history: list[RoundHistory], seller_ids: list[str]) -> dict:
    """Aggregate metrics across all rounds in a session."""
    df = compute_round_metrics(history, seller_ids)
    return {
        "avg_trade_price": df["trade_price_mean"].mean(),
        "total_profit": df["seller_profit"].sum(),
        "avg_ask_price": df["ask_price_mean"].mean(),
        "avg_ask_dispersion": df["ask_dispersion"].mean(),
        "total_trades": df["num_trades"].sum(),
    }


def coordination_score_series(eval_results: list[dict], seller_ids: list[str], num_rounds: int) -> pd.DataFrame:
    """Build a (round x seller) coordination score DataFrame."""
    data = {}
    for sid in seller_ids:
        scores = {r["round_number"]: r.get("score", 1) for r in eval_results if r["seller_id"] == sid}
        data[sid] = [scores.get(r, np.nan) for r in range(1, num_rounds + 1)]
    df = pd.DataFrame(data, index=range(1, num_rounds + 1))
    df.index.name = "round"
    return df
