"""
Full parameter sweep for SPX/SPXW iron condors and iron butterflies --
UNRESTRICTED version: put short strike and call short strike are chosen
completely independently (not tied to each other by a fixed gap), so this
covers every possible short-strike pairing within your OTM range, including
asymmetric/skewed condors.

body_gap (= call_short - put_short) is computed and reported per row, not
used as a filter to generate combinations. You can still bound it with
--min-body-gap / --max-body-gap if you want to exclude nonsensical or
overlapping shapes after the fact.

Sweeps across:
  - DTE (expiration date window)
  - wing width(s)
  - every valid (put_short, call_short) pair within --max-otm-pct of spot

For every combination it prices the 4-leg combo at both mid
((bid+ask)/2 per leg) and "worst" (real bid/ask you could actually
transact at), and records liquidity (min open interest across all 4 legs,
min bid/ask size across the two short legs).

Usage:
    python3 sweep3.py SPX --min-dte 1 --max-dte 25 \
        --widths 5 10 --max-otm-pct 0.05 --min-body-gap 0 --top 20

Requires TRADIER_TOKEN in your environment.

NOTE: this generates far more rows than the tied-gap version (every put
short paired with every call short, not just gap-matched pairs) -- expect
tens of thousands of rows depending on your DTE/width/OTM settings. Full
results always go to CSV; only top-N per ranking prints to the terminal.
"""

import os
import sys
import csv
import argparse
from datetime import datetime, date

import requests

TRADIER_BASE = "https://api.tradier.com/v1"
TOKEN = os.environ.get("TRADIER_TOKEN")
if not TOKEN:
    sys.exit("Set TRADIER_TOKEN in your environment")

HEADERS = {"Authorization": f"Bearer {TOKEN}", "Accept": "application/json"}


# ---------------------------------------------------------------- API calls

def get_quote(symbol):
    r = requests.get(f"{TRADIER_BASE}/markets/quotes", params={"symbols": symbol},
                      headers=HEADERS, timeout=10)
    r.raise_for_status()
    return r.json()["quotes"]["quote"]


def get_expirations(symbol):
    r = requests.get(
        f"{TRADIER_BASE}/markets/options/expirations",
        params={"symbol": symbol, "includeAllRoots": "true", "strikes": "false"},
        headers=HEADERS, timeout=10,
    )
    r.raise_for_status()
    return r.json()["expirations"]["date"]


def get_chain(symbol, expiration):
    r = requests.get(
        f"{TRADIER_BASE}/markets/options/chains",
        params={"symbol": symbol, "expiration": expiration, "greeks": "false"},
        headers=HEADERS, timeout=10,
    )
    r.raise_for_status()
    return r.json()["options"]["option"] or []


# ------------------------------------------------------------- chain utils

def build_maps(opts, root):
    puts = {o["strike"]: o for o in opts if o["option_type"] == "put" and o["root_symbol"] == root}
    calls = {o["strike"]: o for o in opts if o["option_type"] == "call" and o["root_symbol"] == root}
    return puts, calls


def strike_increment(strikes):
    s = sorted(strikes)
    diffs = [round(b - a, 4) for a, b in zip(s, s[1:]) if b - a > 0]
    return min(diffs) if diffs else None


def mid(o):
    return (o["bid"] + o["ask"]) / 2


# --------------------------------------------------- precompute put/call legs

def precompute_put_legs(puts, upper_bound, width, lo_bound):
    """For every candidate short put strike (OTM, within range), precompute
    its spread economics against the fixed width. Returns dict keyed by
    short strike. upper_bound is the strike just above spot (atm_upper), not
    spot itself -- see precompute_call_legs for why."""
    out = {}
    for k_sp, short_put in puts.items():
        if not (lo_bound <= k_sp <= upper_bound):
            continue
        k_lp = round(k_sp - width, 4)
        long_put = puts.get(k_lp)
        if long_put is None:
            continue
        out[k_sp] = {
            "k_lp": k_lp,
            "credit_mid": mid(short_put) - mid(long_put),
            "credit_worst": short_put["bid"] - long_put["ask"],
            "oi": min(short_put["open_interest"], long_put["open_interest"]),
            "short_bidsize": short_put["bidsize"],
            "long_asksize": long_put["asksize"],
        }
    return out


def precompute_call_legs(calls, lower_bound, width, hi_bound):
    out = {}
    for k_sc, short_call in calls.items():
        # bounded by lower_bound (atm_lower, the strike just below spot), not
        # raw spot: spot is continuous and almost never lands exactly on a
        # strike, so a spot-based boundary leaves a gap between the highest
        # qualifying put strike and the lowest qualifying call strike --
        # gap=0 (true ATM butterfly) becomes structurally unreachable.
        # Both straddling strikes (atm_lower, atm_upper) are shared with the
        # put side's upper_bound, so either can appear as both a put-short
        # and a call-short -- the butterfly gets simulated at both.
        if not (lower_bound <= k_sc <= hi_bound):
            continue
        k_lc = round(k_sc + width, 4)
        long_call = calls.get(k_lc)
        if long_call is None:
            continue
        out[k_sc] = {
            "k_lc": k_lc,
            "credit_mid": mid(short_call) - mid(long_call),
            "credit_worst": short_call["bid"] - long_call["ask"],
            "oi": min(short_call["open_interest"], long_call["open_interest"]),
            "short_bidsize": short_call["bidsize"],
            "long_asksize": long_call["asksize"],
        }
    return out


# ------------------------------------------------------------ core sweep

def sweep_expiration(symbol, expiration, spot, widths, max_otm_pct, min_body_gap, max_body_gap):
    opts = get_chain(symbol, expiration)
    if not opts:
        return []

    today = date.today()
    dte = (datetime.strptime(expiration, "%Y-%m-%d").date() - today).days

    rows = []
    for root in sorted(set(o["root_symbol"] for o in opts)):
        puts, calls = build_maps(opts, root)
        if not puts or not calls:
            continue

        all_strikes = sorted(set(puts) | set(calls))
        inc = strike_increment(all_strikes)
        if not inc:
            continue

        # spot sits between two discrete strikes; simulate the ATM butterfly
        # at both straddling strikes rather than picking only the nearer one.
        strikes_at_or_below = [k for k in all_strikes if k <= spot]
        strikes_at_or_above = [k for k in all_strikes if k >= spot]
        atm_lower = max(strikes_at_or_below) if strikes_at_or_below else min(all_strikes)
        atm_upper = min(strikes_at_or_above) if strikes_at_or_above else max(all_strikes)
        lo_bound = spot * (1 - max_otm_pct)
        hi_bound = spot * (1 + max_otm_pct)

        for width in widths:
            w = round(inc * round(width / inc), 4) if inc else width
            if w <= 0:
                continue

            put_legs = precompute_put_legs(puts, atm_upper, w, lo_bound)
            call_legs = precompute_call_legs(calls, atm_lower, w, hi_bound)
            if not put_legs or not call_legs:
                continue

            # fully independent pairing: every put short x every call short
            for k_sp, p in put_legs.items():
                for k_sc, c in call_legs.items():
                    gap = round(k_sc - k_sp, 4)
                    if gap < min_body_gap:
                        continue
                    if max_body_gap is not None and gap > max_body_gap:
                        continue

                    credit_mid = p["credit_mid"] + c["credit_mid"]
                    credit_worst = p["credit_worst"] + c["credit_worst"]

                    rows.append({
                        "expiration": expiration,
                        "dte": dte,
                        "root": root,
                        "width": w,
                        "body_gap": gap,
                        "structure": "butterfly" if gap == 0 else "condor",
                        "k_lp": p["k_lp"], "k_sp": k_sp, "k_sc": k_sc, "k_lc": c["k_lc"],
                        "put_pct_otm": (spot - k_sp) / spot,
                        "call_pct_otm": (k_sc - spot) / spot,
                        "credit_mid": round(credit_mid, 3),
                        "edge_mid": round(credit_mid - w, 3),
                        "credit_worst": round(credit_worst, 3),
                        "edge_worst": round(credit_worst - w, 3),
                        "min_oi": min(p["oi"], c["oi"]),
                        "min_short_bidsize": min(p["short_bidsize"], c["short_bidsize"]),
                        "min_long_asksize": min(p["long_asksize"], c["long_asksize"]),
                        "spot": spot,
                    })
    return rows


def main():
    ap = argparse.ArgumentParser(description="Unrestricted sweep: every put-short x call-short pairing, across DTE and width")
    ap.add_argument("symbol", nargs="?", default="SPX")
    ap.add_argument("--min-dte", type=int, default=1)
    ap.add_argument("--max-dte", type=int, default=25)
    ap.add_argument("--widths", type=float, nargs="+", default=[5, 10])
    ap.add_argument("--max-otm-pct", type=float, default=0.05,
                     help="How far OTM (as a fraction of spot) to consider short strikes, each side.")
    ap.add_argument("--min-body-gap", type=float, default=0,
                     help="Minimum call_short - put_short. 0 excludes inverted/crossed shapes; keeps butterflies (gap=0) and all real condors (gap>0).")
    ap.add_argument("--max-body-gap", type=float, default=None,
                     help="Optional cap on call_short - put_short, to exclude absurdly wide/asymmetric shapes if desired.")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--min-oi", type=int, default=0)
    ap.add_argument("--csv", default="sweep3_results.csv")
    args = ap.parse_args()

    quote = get_quote(args.symbol)
    spot = quote["last"]
    print(f"{args.symbol} spot: {spot}")
    print(f"Widths: {args.widths}   Max OTM: {args.max_otm_pct*100:.1f}%   "
          f"Body gap range: [{args.min_body_gap}, {args.max_body_gap if args.max_body_gap is not None else 'unbounded'}]\n")

    today = date.today()
    expirations = get_expirations(args.symbol)
    target_exps = [
        e for e in expirations
        if args.min_dte <= (datetime.strptime(e, "%Y-%m-%d").date() - today).days <= args.max_dte
    ]
    print(f"Expirations in {args.min_dte}-{args.max_dte} DTE window: {len(target_exps)} found\n")

    all_rows = []
    for exp in target_exps:
        rows = sweep_expiration(args.symbol, exp, spot, args.widths, args.max_otm_pct,
                                 args.min_body_gap, args.max_body_gap)
        all_rows.extend(rows)
        print(f"  {exp}: {len(rows)} combinations")

    print(f"\nTotal combinations across all expirations/widths: {len(all_rows)}")

    if not all_rows:
        print("No combinations found -- try widening --max-otm-pct or --max-dte.")
        return

    fieldnames = list(all_rows[0].keys())
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)
    print(f"Full results written to {args.csv}\n")

    filtered = [r for r in all_rows if r["min_oi"] >= args.min_oi]
    if args.min_oi > 0:
        print(f"({len(filtered)} of {len(all_rows)} combos have min_oi >= {args.min_oi})\n")

    def print_top(rows, key, label, n):
        ranked = sorted(rows, key=lambda r: r[key], reverse=True)[:n]
        print("=" * 100)
        print(f"TOP {n} BY {label}")
        print("=" * 100)
        for r in ranked:
            print(f"  {r['root']} {r['expiration']} ({r['dte']} DTE)  {r['structure']:9s} gap ${r['body_gap']:g}  "
                  f"width ${r['width']:g}  "
                  f"put {r['k_lp']:g}/{r['k_sp']:g} ({r['put_pct_otm']*100:.2f}% OTM)  "
                  f"call {r['k_sc']:g}/{r['k_lc']:g} ({r['call_pct_otm']*100:.2f}% OTM)")
            print(f"      credit_mid ${r['credit_mid']:.2f}  edge_mid ${r['edge_mid']:.2f}  |  "
                  f"credit_worst ${r['credit_worst']:.2f}  edge_worst ${r['edge_worst']:.2f}  |  "
                  f"min_oi {r['min_oi']}  min_short_bidsize {r['min_short_bidsize']}  min_long_asksize {r['min_long_asksize']}")
        print()

    print_top(filtered, "edge_mid", "MID-PRICE EDGE (reference only, not executable)", args.top)
    print_top(filtered, "edge_worst", "REAL EXECUTABLE EDGE (bid/ask -- trust this one)", args.top)

    print("This version pairs every valid put-short strike with every valid call-short strike")
    print("independently -- body_gap is reported per row, not used to generate combinations.")
    print("Open the CSV and group by body_gap to see the real, complete tradeoff between")
    print("butterfly (gap=0) and condors of every width, across every DTE, all at once.")


if __name__ == "__main__":
    main()