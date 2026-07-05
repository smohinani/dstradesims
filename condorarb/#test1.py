import os
import sys
from datetime import datetime, date

import requests

TRADIER_BASE = "https://api.tradier.com/v1"
TOKEN = os.environ.get("TRADIER_TOKEN")
if not TOKEN:
    sys.exit("Set TRADIER_TOKEN in your environment")

HEADERS = {"Authorization": f"Bearer {TOKEN}", "Accept": "application/json"}


def get_quote(symbol):
    r = requests.get(f"{TRADIER_BASE}/markets/quotes", params={"symbols": symbol}, headers=HEADERS, timeout=10)
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


def build_maps(opts, root):
    # A single expiration date can list multiple root symbols (e.g. SPX
    # monthly + SPXW weekly both expire on the same third-Friday date).
    # Those are non-fungible contracts, so a spread must stay within one root.
    puts = {o["strike"]: o for o in opts if o["option_type"] == "put" and o["root_symbol"] == root}
    calls = {o["strike"]: o for o in opts if o["option_type"] == "call" and o["root_symbol"] == root}
    return puts, calls


def strike_increment(strikes):
    s = sorted(strikes)
    diffs = [round(b - a, 4) for a, b in zip(s, s[1:]) if b - a > 0]
    return min(diffs) if diffs else None


def mid(o):
    return (o["bid"] + o["ask"]) / 2


def put_spread_candidates(puts, spot, width, moneyness_pct):
    """Put credit spreads with the short strike OTM (below spot), priced at
    real bid/ask -- this is the leg you'd actually enter today."""
    lo = spot * (1 - moneyness_pct)
    out = []
    for k_sp, short_put in puts.items():
        if not (lo <= k_sp < spot):
            continue
        k_lp = round(k_sp - width, 4)
        long_put = puts.get(k_lp)
        if long_put is None:
            continue
        credit_worst = short_put["bid"] - long_put["ask"]
        credit_mid = mid(short_put) - mid(long_put)
        out.append({
            "k_sp": k_sp, "k_lp": k_lp,
            "credit_worst": credit_worst, "credit_mid": credit_mid,
            "pct_otm": (spot - k_sp) / spot,
            "min_oi": min(short_put["open_interest"], long_put["open_interest"]),
            "short_bidsize": short_put["bidsize"], "long_asksize": long_put["asksize"],
        })
    return out


def call_spread_candidates(calls, spot, width, moneyness_pct):
    """Call credit spreads with the short strike OTM (above spot), priced at
    real bid/ask as of right now -- i.e. before any assumed rally."""
    hi = spot * (1 + moneyness_pct)
    out = []
    for k_sc, short_call in calls.items():
        if not (spot < k_sc <= hi):
            continue
        k_lc = round(k_sc + width, 4)
        long_call = calls.get(k_lc)
        if long_call is None:
            continue
        credit_worst = short_call["bid"] - long_call["ask"]
        credit_mid = mid(short_call) - mid(long_call)
        out.append({
            "k_sc": k_sc, "k_lc": k_lc,
            "credit_worst": credit_worst, "credit_mid": credit_mid,
            "pct_otm": (k_sc - spot) / spot,
            "min_oi": min(short_call["open_interest"], long_call["open_interest"]),
            "short_bidsize": short_call["bidsize"], "long_asksize": long_call["asksize"],
        })
    return out


def analyze_expiration(symbol, expiration, spot, moneyness_pct, width_steps, watch_n):
    opts = get_chain(symbol, expiration)
    if not opts:
        return []

    out = []
    for root in sorted(set(o["root_symbol"] for o in opts)):
        puts, calls = build_maps(opts, root)
        if not puts or not calls:
            continue
        common = sorted(set(puts) | set(calls))
        inc = strike_increment(common)
        if not inc:
            continue
        width = round(inc * width_steps, 4)

        puts_c = put_spread_candidates(puts, spot, width, moneyness_pct)
        calls_c = call_spread_candidates(calls, spot, width, moneyness_pct)
        if not puts_c or not calls_c:
            continue

        # menu sorted by OTM distance -- credit/width alone always favors the
        # closest-to-the-money strike (highest premium), which is a coin-flip
        # on direction, not a "leaning bullish" trade. Let the trader pick
        # their own OTM comfort level instead of silently choosing ATM.
        puts_c.sort(key=lambda p: p["pct_otm"])
        put_menu = puts_c[:watch_n]

        # for the call watchlist demo, anchor on the median-OTM put in the menu
        anchor_put = put_menu[len(put_menu) // 2]
        required_call_credit = width - anchor_put["credit_worst"]
        for c in calls_c:
            c["shortfall"] = required_call_credit - c["credit_worst"]
        calls_c.sort(key=lambda c: c["shortfall"])

        # every put x call combo, priced at mid, for the global "closest to
        # completing right now" ranking (mid is not executable -- see caveat
        # printed with the results -- but it's the best apples-to-apples
        # measure of how close a combo is before bid/ask friction).
        combos = []
        for p in puts_c:
            for c in calls_c:
                credit_mid = p["credit_mid"] + c["credit_mid"]
                credit_worst = p["credit_worst"] + c["credit_worst"]
                combos.append({
                    "root": root, "width": width,
                    "k_lp": p["k_lp"], "k_sp": p["k_sp"], "k_sc": c["k_sc"], "k_lc": c["k_lc"],
                    "put_pct_otm": p["pct_otm"], "call_pct_otm": c["pct_otm"],
                    "credit_mid": credit_mid, "edge_mid": credit_mid - width,
                    "credit_worst": credit_worst, "edge_worst": credit_worst - width,
                    "min_oi": min(p["min_oi"], c["min_oi"]),
                })

        out.append({
            "root": root, "width": width,
            "put_menu": put_menu,
            "anchor_put": anchor_put,
            "required_call_credit": required_call_credit,
            "call_watchlist": calls_c[:watch_n],
            "combos": combos,
        })
    return out


def main():
    symbol = sys.argv[1] if len(sys.argv) > 1 else "SPX"
    min_dte = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    max_dte = int(sys.argv[3]) if len(sys.argv) > 3 else 25

    quote = get_quote(symbol)
    spot = quote["last"]
    print(f"{symbol} spot: {spot}\n")

    today = date.today()
    expirations = get_expirations(symbol)
    target_exps = [
        e for e in expirations
        if min_dte <= (datetime.strptime(e, "%Y-%m-%d").date() - today).days <= max_dte
    ]
    print(f"Expirations in {min_dte}-{max_dte} DTE window: {len(target_exps)} found\n")

    all_combos = []
    for exp in target_exps:
        dte = (datetime.strptime(exp, "%Y-%m-%d").date() - today).days
        entries = analyze_expiration(symbol, exp, spot, moneyness_pct=0.05, width_steps=1, watch_n=5)
        for entry in entries:
            for c in entry["combos"]:
                c["expiration"] = exp
                c["dte"] = dte
            all_combos.extend(entry["combos"])
        for entry in entries:
            width = entry["width"]
            print(f"=== {entry['root']} {exp} ({dte} DTE)  width=${width:g}  spot={spot} ===")
            print(f"  Put spread menu (pick your own OTM comfort level -- credit/width is highest near ATM):")
            for p in entry["put_menu"]:
                print(f"    {p['k_lp']:g}/{p['k_sp']:g}  ({p['pct_otm']*100:.2f}% OTM)  "
                      f"real credit: {p['credit_worst']:.3f} (${p['credit_worst']*100:.2f}/contract)  "
                      f"[OI {p['min_oi']}, shortBid {p['short_bidsize']}, longAsk {p['long_asksize']}]")
            bp = entry["anchor_put"]
            print(f"  Using median-OTM pick {bp['k_lp']:g}/{bp['k_sp']:g} ({bp['pct_otm']*100:.2f}% OTM, "
                  f"credit {bp['credit_worst']:.3f}) as the anchor below:")
            print(f"  Call side needed to complete the condor: >= {entry['required_call_credit']:.3f} credit "
                  f"(width {width:g} - put credit {bp['credit_worst']:.3f})")
            print(f"  Call spread watchlist (closest to completing first, priced as of now):")
            for c in entry["call_watchlist"]:
                status = "COMPLETES NOW" if c["shortfall"] <= 0 else f"needs +{c['shortfall']:.3f} more credit"
                print(f"    {c['k_sc']:g}/{c['k_lc']:g}  (+{c['pct_otm']*100:.2f}% above spot)  "
                      f"credit now: {c['credit_worst']:.3f}  -> {status}  "
                      f"[OI {c['min_oi']}, shortBid {c['short_bidsize']}, longAsk {c['long_asksize']}]")
            print()

    def print_top5(combos, label):
        combos.sort(key=lambda c: c["edge_mid"], reverse=True)
        print("=" * 70)
        print(f"TOP 5 CLOSEST TO COMPLETING RIGHT NOW -- {label} (mid price, NOT executable)")
        print("=" * 70)
        for c in combos[:5]:
            print(f"  {c['root']} {c['expiration']} ({c['dte']} DTE)  "
                  f"put {c['k_lp']:g}/{c['k_sp']:g} ({c['put_pct_otm']*100:.2f}% OTM)  "
                  f"call {c['k_sc']:g}/{c['k_lc']:g} ({c['call_pct_otm']*100:.2f}% OTM)  "
                  f"width ${c['width']:g}")
            print(f"    credit_mid: {c['credit_mid']:.3f}  edge_mid: {c['edge_mid']:.3f}  "
                  f"|  credit_worst: {c['credit_worst']:.3f}  edge_worst: {c['edge_worst']:.3f}  "
                  f"|  min OI: {c['min_oi']}")
        print()

    print_top5(all_combos, "$5 wings")

    print("Reminder: mid price is the (bid+ask)/2 of each leg -- nobody can actually "
          "transact there, it's a reference point. The edge_worst column next to each "
          "row shows what's left once you price it at real bid/ask; that gap is the "
          "friction you'd have to beat to actually capture any of this.")


if __name__ == "__main__":
    main()
