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


Quick sanity check that you're set up correctly — your prompt should show (honest-agent) and ls should show main.py and src/. Then paste the caffeinate command (the multi-line version with real line breaks, like the one that worked for the smoke test — the single-line squashed version in your message would break since Python needs those newlines).
Then detach: Ctrl+B, release both keys, press D. You'll drop back to your normal terminal with [detached (from session e0)] — the run continues inside. tmux attach -t e0 whenever you want to check on it.
One pre-flight check worth 10 seconds since you're in a fresh shell anyway: python -c "import dotenv, openai, anthropic" — if that errors, the conda env didn't activate properly and you'd rather find out now than 3 seconds into the run