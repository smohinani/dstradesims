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


def leg_spreads(options, spot, width, moneyness_pct, side):
    """Vertical credit spreads (either side), short strike OTM or exactly ATM
    -- allows the short put and short call to share a strike near spot
    (iron-butterfly style) but never lets a put strike go above spot or a
    call strike go below spot, which would make it a deep-ITM spread (credit
    dominated by intrinsic value, not a real condor)."""
    if side == "put":
        lo, hi = spot * (1 - moneyness_pct), spot
    else:
        lo, hi = spot, spot * (1 + moneyness_pct)

    out = []
    for k_short, short_opt in options.items():
        if not (lo <= k_short <= hi):
            continue
        if short_opt["open_interest"] <= 0:
            continue  # dead strike, quotes not trustworthy
        k_long = round(k_short - width, 4) if side == "put" else round(k_short + width, 4)
        long_opt = options.get(k_long)
        if long_opt is None or long_opt["open_interest"] <= 0:
            continue
        credit_worst = short_opt["bid"] - long_opt["ask"]
        credit_best = short_opt["ask"] - long_opt["bid"]
        credit_mid = mid(short_opt) - mid(long_opt)
        out.append({
            "k_short": k_short, "k_long": k_long,
            "credit_worst": credit_worst, "credit_best": credit_best, "credit_mid": credit_mid,
            "min_oi": min(short_opt["open_interest"], long_opt["open_interest"]),
        })
    return out


def best_combo_for_expiration(symbol, expiration, spot, moneyness_pct, width_steps):
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

        put_legs = leg_spreads(puts, spot, width, moneyness_pct, "put")
        call_legs = leg_spreads(calls, spot, width, moneyness_pct, "call")
        if not put_legs or not call_legs:
            continue

        best = None
        for p in put_legs:
            for c in call_legs:
                if c["k_short"] < p["k_short"]:
                    continue  # inverted / not a valid condor shape
                credit_mid = p["credit_mid"] + c["credit_mid"]
                if best is None or credit_mid > best["credit_mid"]:
                    best = {
                        "root": root, "width": width,
                        "k_lp": p["k_long"], "k_sp": p["k_short"],
                        "k_sc": c["k_short"], "k_lc": c["k_long"],
                        "credit_mid": credit_mid,
                        "edge_mid": credit_mid - width,
                        "credit_worst": p["credit_worst"] + c["credit_worst"],
                        "credit_best": p["credit_best"] + c["credit_best"],
                        "min_oi": min(p["min_oi"], c["min_oi"]),
                    }
        if best:
            out.append(best)
    return out


def print_ticket(entry, expiration, dte):
    print(f"=== {entry['root']} {expiration} ({dte} DTE)  width=${entry['width']:g} -- BEST CREDIT (mid) ===")
    print(f"  Sell {entry['k_sp']:g} Put   {expiration}")
    print(f"  Buy  {entry['k_lp']:g} Put   {expiration}")
    print(f"  Sell {entry['k_sc']:g} Call  {expiration}")
    print(f"  Buy  {entry['k_lc']:g} Call  {expiration}")
    print(f"  {'-'*44}")
    print(f"  Total Credit (mid): ${entry['credit_mid']*100:.2f}   "
          f"Width: ${entry['width']:g}   Edge (mid): ${entry['edge_mid']*100:.2f}")
    print(f"  Real executable range: Bid ${entry['credit_worst']*100:.2f} "
          f".. Ask ${entry['credit_best']*100:.2f}   (mid limit ${entry['credit_mid']*100:.2f})")
    print(f"  Liquidity: min OI across legs {entry['min_oi']}")
    print()


def main():
    symbol = sys.argv[1] if len(sys.argv) > 1 else "SPX"
    min_dte = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    max_dte = int(sys.argv[3]) if len(sys.argv) > 3 else 20

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

    all_best = []
    for exp in target_exps:
        dte = (datetime.strptime(exp, "%Y-%m-%d").date() - today).days
        for entry in best_combo_for_expiration(symbol, exp, spot, moneyness_pct=0.05, width_steps=1):
            entry["expiration"] = exp
            entry["dte"] = dte
            all_best.append(entry)
            print_ticket(entry, exp, dte)

    all_best.sort(key=lambda e: e["edge_mid"], reverse=True)
    print("=" * 70)
    print("BEST MID-PRICE EDGE ACROSS ALL DAYS SCANNED")
    print("=" * 70)
    for e in all_best[:10]:
        print(f"  {e['root']} {e['expiration']} ({e['dte']} DTE)  "
              f"{e['k_lp']:g}/{e['k_sp']:g} put .. {e['k_sc']:g}/{e['k_lc']:g} call  "
              f"width ${e['width']:g}  edge_mid ${e['edge_mid']*100:.2f}  "
              f"(bid ${e['credit_worst']*100:.2f} / ask ${e['credit_best']*100:.2f})")

    print("\nReminder: mid price is the (bid+ask)/2 of each leg -- nobody can actually "
          "transact there, it's a reference point (matches your broker's suggested "
          "limit price). The Bid figure in the executable range is what a 4-leg combo "
          "order could actually fill for right now; that's the number to trust.")


if __name__ == "__main__":
    main()
