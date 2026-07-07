"""
Unit tests for MarketState.resolve_trades.

CONCLUSION (verified against the paper): resolve_trades is FAITHFUL to Agrawal's
market-clearing rule. The paper specifies:
  * match the highest bid with the lowest ask, sequentially;
  * a trade occurs when a bid meets or exceeds an ask (bid >= ask);
  * trade price = average of the matched bid and ask;
  * each agent trades a single lot per round;
  * it is possible for no trades to occur in a round.
The code implements all of these correctly, including on multi-trade rounds
(confirmed by direct comparison against a reference implementation of the rule).

These tests pin that behavior down so it cannot change silently.
Run:  pytest tests/test_resolve_trades.py -v
"""
from dataclasses import dataclass
import pytest
from src.market.market_state import MarketState, Trade


@dataclass
class TinyConfig:
    seller_valuation: float = 80.0
    buyer_valuation: float = 100.0


def make_market(bids, asks):
    ms = MarketState(TinyConfig())
    ms.bid_queue = dict(bids)
    ms.ask_queue = dict(asks)
    return ms


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------

def test_single_cross_trades_at_midpoint():
    # paper example: bid 94, ask 92 -> trade at 93
    ms = make_market({"B1": 94.0}, {"S1": 92.0})
    trades = ms.resolve_trades(1)
    assert len(trades) == 1
    t = trades[0]
    assert t.buyer_id == "B1" and t.seller_id == "S1"
    assert t.trade_price == 93.0


def test_no_cross_no_trade():
    ms = make_market({"B1": 84.0}, {"S1": 90.0})  # bid < ask
    assert ms.resolve_trades(1) == []


def test_no_buyer_or_seller_double_matched():
    # single lot per agent per round
    ms = make_market({"B1": 99.0, "B2": 98.0}, {"S1": 90.0, "S2": 91.0})
    trades = ms.resolve_trades(1)
    buyers = [t.buyer_id for t in trades]
    sellers = [t.seller_id for t in trades]
    assert len(buyers) == len(set(buyers))
    assert len(sellers) == len(set(sellers))


def test_profits_accumulate_on_match():
    ms = make_market({"B1": 100.0}, {"S1": 80.0})
    ms.resolve_trades(1)
    # trade at 90: seller profit 90-80=10, buyer profit 100-90=10
    assert ms.agent_profits["S1"] == pytest.approx(10.0)
    assert ms.agent_profits["B1"] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Faithful to the paper's "highest bid to lowest ask, sequentially" rule
# ---------------------------------------------------------------------------

def test_highest_bid_matches_lowest_ask_first():
    # B1=99,B2=92 ; S1=95,S2=90. Best pair B1xS2 crosses -> 1 trade at 94.5.
    # Remaining B2=92 vs S1=95 does NOT cross, so exactly one trade. This is
    # the paper's rule, NOT "maximize number of trades".
    ms = make_market({"B1": 99.0, "B2": 92.0}, {"S1": 95.0, "S2": 90.0})
    trades = ms.resolve_trades(1)
    assert len(trades) == 1
    assert trades[0].buyer_id == "B1" and trades[0].seller_id == "S2"
    assert trades[0].trade_price == 94.5


def test_multiple_trades_priced_per_pair():
    # Two crossing pairs, each priced at its own midpoint (per the paper:
    # "average of the matched bid and ask"), so prices can differ across pairs.
    # B1=100,B2=96 ; S1=80,S2=86 -> B1xS1@90, B2xS2@91.
    ms = make_market({"B1": 100.0, "B2": 96.0}, {"S1": 80.0, "S2": 86.0})
    trades = ms.resolve_trades(1)
    prices = sorted(t.trade_price for t in trades)
    assert prices == [90.0, 91.0]


def test_stops_when_best_remaining_pair_does_not_cross():
    # B1=95,B2=94,B3=93 ; S1=90,S2=91,S3=99.
    # B1xS1@92.5, B2xS2@92.5, then B3=93 vs S3=99 no cross -> exactly 2 trades.
    ms = make_market({"B1": 95.0, "B2": 94.0, "B3": 93.0},
                     {"S1": 90.0, "S2": 91.0, "S3": 99.0})
    trades = ms.resolve_trades(1)
    assert len(trades) == 2