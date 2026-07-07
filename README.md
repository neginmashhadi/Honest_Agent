# Honest_Agent

## Setup

Use the `honest-agent` conda environment:

```
conda activate honest-agent
```

## Running tests

Tests must be run from the repo root (not from inside `src/`), since test
files import via the `src` package, e.g. `from src.market.market_state import ...`.

```
cd /Users/neginmashhadi/Repos/Honest_Agent
pytest src/test_resolve_trades.py -v
```
