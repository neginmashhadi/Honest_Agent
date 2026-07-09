"""
Quick per-session comparison tool for RQ1 session JSONs.

Prints, for any set of session files:
  - per-session row: judge mean / last-10, t*, round-1 ask, settle (last-5 mean),
    late slope (rounds 10+), trades, failed_agent_calls, honest-agent mode/timing
  - integrity flags: dead sessions, silent-mode violations (S3 sent a message),
    missing judge coverage
  - condition-level summary and pairwise Mann-Whitney U + Cliff's delta on
    session judge means and settle prices (whenever >= 2 conditions present)

Usage:
  python -m src.analysis.session_compare                       # all rq1_* sessions
  python -m src.analysis.session_compare results/sessions/rq1_e2*.json
  python -m src.analysis.session_compare 'results/sessions/rq1_e0_all*' 'results/sessions/rq1_e1*'
"""
import json
import sys
from glob import glob

import numpy as np
from scipy.stats import mannwhitneyu


def load(path):
    with open(path) as f:
        return json.load(f)


def late_slope(asks, start=9):
    pts = [(i, v) for i, v in enumerate(asks) if v is not None and i >= start]
    if len(pts) < 3:
        return float("nan")
    x, y = zip(*pts)
    return float(np.polyfit(x, y, 1)[0])


def s3_message_count(session):
    """Counts rounds in which S3 produced judge-visible message evidence.
    Used to verify silent mode: should be 0 when mode == 'silent'."""
    count = 0
    for e in session.get("eval_scores", []):
        if e.get("seller_id") == "S3" and e.get("evidence"):
            count += 1
    return count


def session_row(path):
    s = load(path)
    cs = s.get("collusion_summary") or {}
    j = [v for v in (cs.get("judge_series_bloc") or []) if v is not None]
    asks = [r["ask_price_mean"] for r in s.get("round_metrics", [])]
    asks_c = [a for a in asks if a is not None]
    honest = s.get("honest_agent") or {}
    total_asks = sum(len(r["asks"]) for r in s.get("market_history", []))
    row = {
        "session": s.get("session_id", path),
        "condition": s.get("condition", "?"),
        "mode": honest.get("mode") or "-",
        "timing": honest.get("timing") or "-",
        "judge": float(np.mean(j)) if j else None,
        "judge_l10": float(np.mean(j[-10:])) if j else None,
        "n_judge": len(j),
        "t_star": cs.get("t_star"),
        "r1_ask": asks_c[0] if asks_c else None,
        "settle": float(np.mean(asks_c[-5:])) if asks_c else None,
        "slope10": late_slope(asks),
        "trades": sum(r["num_trades"] for r in s.get("round_metrics", [])),
        "fails": s.get("failed_agent_calls", 0),
        "flags": [],
    }
    if total_asks == 0:
        row["flags"].append("DEAD")
    if row["n_judge"] < 20 and total_asks > 0:
        row["flags"].append(f"judge_cov={row['n_judge']}")
    if honest.get("mode") == "silent":
        msgs = s3_message_count(s)
        if msgs > 0:
            row["flags"].append(f"SILENT_VIOLATION({msgs} rounds)")
    if isinstance(row["fails"], int) and row["fails"] > 2:
        row["flags"].append("degraded")
    return row


def cliffs_delta(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    gt = sum((x > b).sum() for x in a)
    lt = sum((x < b).sum() for x in a)
    return (gt - lt) / (len(a) * len(b))


def main(argv):
    patterns = argv or ["results/sessions/rq1_*.json"]
    paths = sorted({p for pat in patterns for p in glob(pat)})
    if not paths:
        print(f"No session files match: {patterns}")
        return
    rows = [session_row(p) for p in paths]

    fmt = "{session:<26s} {mode:>6s} {timing:>7s} {judge:>5s} {jl10:>5s} {t:>4s} {settle:>7s} {slope:>8s} {trades:>6s} {fails:>5s}  {flags}"
    print(fmt.format(session="session", mode="mode", timing="timing", judge="judge",
                     jl10="j_l10", t="t*", settle="settle", slope="slope10+",
                     trades="trades", fails="fails", flags="flags"))
    for r in rows:
        print(fmt.format(
            session=r["session"], mode=r["mode"], timing=r["timing"],
            judge=f"{r['judge']:.2f}" if r["judge"] is not None else "-",
            jl10=f"{r['judge_l10']:.2f}" if r["judge_l10"] is not None else "-",
            t=str(r["t_star"]),
            settle=f"{r['settle']:.2f}" if r["settle"] is not None else "-",
            slope=f"{r['slope10']:+.3f}" if not np.isnan(r["slope10"]) else "-",
            trades=str(r["trades"]), fails=str(r["fails"]),
            flags=",".join(r["flags"]) or "ok"))

    # condition-level summary + pairwise comparisons
    conds = {}
    for r in rows:
        if r["judge"] is not None and "DEAD" not in r["flags"]:
            conds.setdefault(r["condition"], []).append(r)
    if len(conds) < 2:
        return
    print("\n--- condition summary ---")
    for c, rs in conds.items():
        js = [r["judge"] for r in rs]
        st = [r["settle"] for r in rs if r["settle"] is not None]
        print(f"{c:<22s} n={len(rs):2d}  judge={np.mean(js):.2f} [{min(js):.2f},{max(js):.2f}]"
              f"  settle={np.mean(st):.2f}")
    print("\n--- pairwise (judge means; settle in parens) ---")
    names = sorted(conds)
    for i in range(len(names)):
        for k in range(i + 1, len(names)):
            a = [r["judge"] for r in conds[names[i]]]
            b = [r["judge"] for r in conds[names[k]]]
            if len(a) < 2 or len(b) < 2:
                continue
            u, p = mannwhitneyu(a, b, alternative="two-sided")
            sa = [r["settle"] for r in conds[names[i]] if r["settle"] is not None]
            sb = [r["settle"] for r in conds[names[k]] if r["settle"] is not None]
            us, ps = mannwhitneyu(sa, sb, alternative="two-sided") if len(sa) > 1 and len(sb) > 1 else (float("nan"), float("nan"))
            print(f"{names[i]} vs {names[k]}: U={u:.0f} p={p:.2e} delta={cliffs_delta(a, b):+.2f}"
                  f"   (settle: U={us:.0f} p={ps:.3f})")


if __name__ == "__main__":
    main(sys.argv[1:])
