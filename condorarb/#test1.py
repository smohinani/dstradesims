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
    opts = r.json()["options"]["option"] or []
    puts = {o["strike"]: o for o in opts if o["option_type"] == "put"}
    calls = {o["strike"]: o for o in opts if o["option_type"] == "call"}
    return puts, calls


def strike_increment(strikes):
    s = sorted(strikes)
    diffs = [round(b - a, 4) for a, b in zip(s, s[1:]) if b - a > 0]
    return min(diffs) if diffs else None


def mid(o):
    return (o["bid"] + o["ask"]) / 2


def scan_expiration(symbol, expiration, spot, moneyness_pct, max_gap_steps, width_steps_list):
    puts, calls = get_chain(symbol, expiration)
    if not puts or not calls:
        return []

    common_strikes = sorted(set(puts) & set(calls))
    if not common_strikes:
        return []
    inc = strike_increment(common_strikes)
    if not inc:
        return []

    lo = spot * (1 - moneyness_pct)
    hi = spot * (1 + moneyness_pct)
    near_strikes = [k for k in common_strikes if lo <= k <= hi]

    results = []
    for width_steps in width_steps_list:
        width = round(inc * width_steps, 4)
        for k_sp in near_strikes:
            k_lp = round(k_sp - width, 4)
            if k_lp not in puts:
                continue
            short_put, long_put = puts[k_sp], puts[k_lp]

            for gap_steps in range(0, max_gap_steps + 1):
                gap = round(inc * gap_steps, 4)
                k_sc = round(k_sp + gap, 4)
                k_lc = round(k_sc + width, 4)
                if k_sc not in calls or k_lc not in calls:
                    continue
                short_call, long_call = calls[k_sc], calls[k_lc]

                credit_put_worst = short_put["bid"] - long_put["ask"]
                credit_call_worst = short_call["bid"] - long_call["ask"]
                total_worst = credit_put_worst + credit_call_worst

                credit_put_mid = mid(short_put) - mid(long_put)
                credit_call_mid = mid(short_call) - mid(long_call)
                total_mid = credit_put_mid + credit_call_mid

                edge_worst = total_worst - width
                edge_mid = total_mid - width

                results.append({
                    "expiration": expiration,
                    "width": width,
                    "gap": gap,
                    "k_lp": k_lp, "k_sp": k_sp, "k_sc": k_sc, "k_lc": k_lc,
                    "credit_worst": total_worst,
                    "credit_mid": total_mid,
                    "edge_worst": edge_worst,
                    "edge_mid": edge_mid,
                })
    return results


def main():
    symbol = sys.argv[1] if len(sys.argv) > 1 else "XSP"
    min_dte = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    max_dte = int(sys.argv[3]) if len(sys.argv) > 3 else 18

    quote = get_quote(symbol)
    spot = quote["last"]
    print(f"{symbol} spot: {spot}")

    today = date.today()
    expirations = get_expirations(symbol)
    target_exps = [
        e for e in expirations
        if min_dte <= (datetime.strptime(e, "%Y-%m-%d").date() - today).days <= max_dte
    ]
    print(f"Expirations in {min_dte}-{max_dte} DTE window: {target_exps}")

    all_results = []
    for exp in target_exps:
        res = scan_expiration(
            symbol, exp, spot,
            moneyness_pct=0.03,
            max_gap_steps=3,
            width_steps_list=[1],
        )
        all_results.extend(res)

    all_results.sort(key=lambda r: r["edge_worst"], reverse=True)

    print(f"\nScanned {len(all_results)} condor combos.\n")
    header = f"{'exp':10} {'w':>5} {'gap':>5} {'strikes':>26} {'credit(worst)':>13} {'edge(worst)':>11} {'credit(mid)':>11} {'edge(mid)':>9}"
    print(header)
    for r in all_results[:25]:
        strikes = f"{r['k_lp']}/{r['k_sp']}..{r['k_sc']}/{r['k_lc']}"
        print(f"{r['expiration']:10} {r['width']:>5} {r['gap']:>5} {strikes:>26} "
              f"{r['credit_worst']:>13.3f} {r['edge_worst']:>11.3f} "
              f"{r['credit_mid']:>11.3f} {r['edge_mid']:>9.3f}")

    real_arbs = [r for r in all_results if r["edge_worst"] > 0]
    print(f"\n{len(real_arbs)} combos show a real (worst-case executable) arbitrage edge (bid/ask).")
    mid_arbs = [r for r in all_results if r["edge_mid"] > 0]
    print(f"{len(mid_arbs)} combos show an edge at mid-price (not necessarily executable).")


if __name__ == "__main__":
    main()
