from dataclasses import dataclass, field
from typing import Optional
import random


@dataclass
class Order:
    agent_id: str
    price: float
    round_number: int


@dataclass
class Trade:
    round_number: int
    buyer_id: str
    seller_id: str
    bid_price: float
    ask_price: float
    trade_price: float

    @property
    def seller_profit(self) -> float:
        return self.trade_price - 80.0

    @property
    def buyer_profit(self) -> float:
        return 100.0 - self.trade_price


@dataclass
class RoundHistory:
    round_number: int
    bids: dict[str, float]   # agent_id -> price
    asks: dict[str, float]
    trades: list[Trade]


class MarketState:
    def __init__(self, config):
        self.config = config
        self.bid_queue: dict[str, float] = {}   # agent_id -> current bid
        self.ask_queue: dict[str, float] = {}   # agent_id -> current ask
        self.round_history: list[RoundHistory] = []
        self.agent_profits: dict[str, float] = {}
        self.agent_trades: dict[str, list[Trade]] = {}

    def initialize_round1(self, buyer_ids: list[str], seller_ids: list[str]):
        """Pre-populate round 1 queues with random bids/asks per paper."""
        cfg = self.config
        for bid in buyer_ids:
            price = round(random.uniform(cfg.initial_bid_low, cfg.initial_bid_high), 2)
            self.bid_queue[bid] = price
        for sid in seller_ids:
            price = round(random.uniform(cfg.initial_ask_low, cfg.initial_ask_high), 2)
            self.ask_queue[sid] = price

    def update_order(self, agent_id: str, is_seller: bool, price: Optional[float]):
        """Place or withdraw an order. None price = withdraw."""
        queue = self.ask_queue if is_seller else self.bid_queue
        if price is None:
            queue.pop(agent_id, None)
        else:
            queue[agent_id] = price

    def resolve_trades(self, round_number: int) -> list[Trade]:
        """Match highest bids against lowest asks; trade at average price."""
        trades = []
        bids_sorted = sorted(self.bid_queue.items(), key=lambda x: -x[1])
        asks_sorted = sorted(self.ask_queue.items(), key=lambda x: x[1])

        used_buyers = set()
        used_sellers = set()

        for buyer_id, bid in bids_sorted:
            for seller_id, ask in asks_sorted:
                if seller_id in used_sellers:
                    continue
                if bid >= ask:
                    trade_price = round((bid + ask) / 2, 2)
                    trade = Trade(
                        round_number=round_number,
                        buyer_id=buyer_id,
                        seller_id=seller_id,
                        bid_price=bid,
                        ask_price=ask,
                        trade_price=trade_price,
                    )
                    trades.append(trade)
                    used_buyers.add(buyer_id)
                    used_sellers.add(seller_id)

                    # Track profits
                    seller_profit = trade_price - self.config.seller_valuation
                    buyer_profit = self.config.buyer_valuation - trade_price
                    self.agent_profits[seller_id] = self.agent_profits.get(seller_id, 0) + seller_profit
                    self.agent_profits[buyer_id] = self.agent_profits.get(buyer_id, 0) + buyer_profit

                    # Track individual trades
                    self.agent_trades.setdefault(seller_id, []).append(trade)
                    self.agent_trades.setdefault(buyer_id, []).append(trade)
                    break

        self.round_history.append(RoundHistory(
            round_number=round_number,
            bids=dict(self.bid_queue),
            asks=dict(self.ask_queue),
            trades=trades,
        ))
        return trades

    def get_past_bids_and_asks(self, window: int = 5) -> str:
        """Format recent bid/ask history for agent prompts."""
        recent = self.round_history[-window:] if len(self.round_history) >= window else self.round_history
        if not recent:
            return "No history yet."
        lines = []
        for rh in recent:
            lines.append(f"Hour {rh.round_number}:")
            lines.append(f"  Bids: {', '.join(f'{k}: ${v:.2f}' for k, v in sorted(rh.bids.items()))}")
            lines.append(f"  Asks: {', '.join(f'{k}: ${v:.2f}' for k, v in sorted(rh.asks.items()))}")
        return "\n".join(lines)

    def get_past_trades(self, window: int = 5) -> str:
        recent = self.round_history[-window:]
        lines = []
        for rh in recent:
            if rh.trades:
                for t in rh.trades:
                    lines.append(f"Hour {rh.round_number}: {t.buyer_id} bought from {t.seller_id} at ${t.trade_price:.2f}")
            else:
                lines.append(f"Hour {rh.round_number}: No trades.")
        return "\n".join(lines) if lines else "No trades yet."

    def format_bid_queue(self) -> str:
        if not self.bid_queue:
            return "Empty"
        return ", ".join(f"{k}: ${v:.2f}" for k, v in sorted(self.bid_queue.items(), key=lambda x: -x[1]))

    def format_ask_queue(self) -> str:
        if not self.ask_queue:
            return "Empty"
        return ", ".join(f"{k}: ${v:.2f}" for k, v in sorted(self.ask_queue.items(), key=lambda x: x[1]))

    def format_agent_trades(self, agent_id: str) -> str:
        trades = self.agent_trades.get(agent_id, [])
        if not trades:
            return "No trades yet."
        lines = [f"Hour {t.round_number}: traded at ${t.trade_price:.2f}" for t in trades]
        return "\n".join(lines)
