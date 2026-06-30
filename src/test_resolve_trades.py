"""
Unit tests for MarketState.resolve_trades.

Purpose: pin down EXACTLY what the matching rule does on rounds where multiple
bids and asks cross at once (the case the 5-round smoke test never exercised,
because it only ever produced one trade in the final round).

These tests document current behavior AND flag a real inefficiency: the greedy
"highest bidder takes the lowest ask" rule can consume a cheap seller on a buyer
who did not need it, leaving a feasible trade unmatched. Tests marked
`test_DOCUMENTS_*` capture current behavior so it cannot change silently.
Tests marked `test_BUG_*` assert the behavior a correct clearing SHOULD produce;
they FAIL against the current code and should pass once matching is fixed.

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
# Basic correctness (these should pass on the current code)
# ---------------------------------------------------------------------------

def test_single_cross_trades_at_midpoint():
    ms = make_market({"B1": 95.0}, {"S1": 85.0})
    trades = ms.resolve_trades(1)
    assert len(trades) == 1
    t = trades[0]
    assert t.buyer_id == "B1" and t.seller_id == "S1"
    assert t.trade_price == 90.0  # midpoint of 95 and 85


def test_no_cross_no_trade():
    ms = make_market({"B1": 84.0}, {"S1": 90.0})  # bid < ask
    assert ms.resolve_trades(1) == []


def test_no_buyer_or_seller_double_matched():
    ms = make_market({"B1": 99.0, "B2": 98.0}, {"S1": 90.0, "S2": 91.0})
    trades = ms.resolve_trades(1)
    buyers = [t.buyer_id for t in trades]
    sellers = [t.seller_id for t in trades]
    assert len(buyers) == len(set(buyers))   # no buyer twice
    assert len(sellers) == len(set(sellers)) # no seller twice


def test_profits_accumulate_on_match():
    ms = make_market({"B1": 100.0}, {"S1": 80.0})
    ms.resolve_trades(1)
    # trade at 90: seller profit 90-80=10, buyer profit 100-90=10
    assert ms.agent_profits["S1"] == pytest.approx(10.0)
    assert ms.agent_profits["B1"] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# DOCUMENTS current behavior: greedy matching can miss a feasible trade
# (these PASS now, encoding the quirk so it cannot change unnoticed)
# ---------------------------------------------------------------------------

def test_DOCUMENTS_greedy_starves_a_feasible_trade():
    # B1=99, B2=92 ; S1=95, S2=90.
    # Feasible pairs: B1xS1, B1xS2, B2xS2.  A good clearing makes 2 trades.
    # Greedy gives B1 the cheapest ask (S2@90), then B2(92) cannot afford S1(95).
    ms = make_market({"B1": 99.0, "B2": 92.0}, {"S1": 95.0, "S2": 90.0})
    trades = ms.resolve_trades(1)
    assert len(trades) == 1                      # only ONE trade (the quirk)
    assert trades[0].buyer_id == "B1"
    assert trades[0].seller_id == "S2"
    assert trades[0].trade_price == 94.5         # midpoint of 99 and 90


def test_DOCUMENTS_per_pair_midpoint_not_uniform_price():
    # Two clean simultaneous trades priced at DIFFERENT midpoints.
    # B1=100,S1=80 -> 90 ; B2=96,S2=86 -> 91.  Not a single clearing price.
    ms = make_market({"B1": 100.0, "B2": 96.0}, {"S1": 80.0, "S2": 86.0})
    trades = ms.resolve_trades(1)
    prices = sorted(t.trade_price for t in trades)
    assert prices == [90.0, 91.0]                # discriminatory, not uniform


# ---------------------------------------------------------------------------
# BUG: what a correct clearing SHOULD do. These FAIL on the current code and
# should PASS once matching is fixed (e.g. match cheapest ask to lowest
# *feasible* bid, or run a proper clearing that maximizes feasible trades).
# Marked xfail so the suite stays green but the gap is recorded.
# ---------------------------------------------------------------------------

@pytest.mark.xfail(reason="greedy high-low matching misses a feasible second trade", strict=True)
def test_BUG_should_match_all_feasible_trades():
    # Same market as the 'starves' case: a correct clearing yields 2 trades.
    ms = make_market({"B1": 99.0, "B2": 92.0}, {"S1": 95.0, "S2": 90.0})
    trades = ms.resolve_trades(1)
    assert len(trades) == 2
