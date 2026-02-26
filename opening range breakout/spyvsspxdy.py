#lab01.py

import pandas as pd
import numpy as np
import yfinance as yf
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# ---------------------------
# Config
# ---------------------------
TICKERS = {"SPY": "SPY", "SPX": "^GSPC"}
INTERVAL = "5m"       # 5m bars: ORB 15m=3 bars, 30m=6 bars, 60m=12 bars
PERIOD = "60d"        # yfinance intraday limit
WINDOWS_MIN = [15, 30, 60]

RR_LIST = [0.5, 1.0, 1.5, 2.0]
K_LIST  = [0.5, 0.75, 1.0]

MARKET_OPEN = "09:30"
MARKET_CLOSE = "16:00"
ONE_TRADE_PER_DAY_PER_WINDOW = True

# Conservative intrabar rule:
# for LONG: if both SL and TP touched same bar -> count SL first
# for SHORT: if both touched same bar -> count SL first
CONSERVATIVE_SAME_BAR = True


# ---------------------------
# Data
# ---------------------------
def load_intraday(ticker: str) -> pd.DataFrame:
    df = yf.download(
        ticker,
        period=PERIOD,
        interval=INTERVAL,
        auto_adjust=False,
        progress=False,
        prepost=False
    )

    if df is None or df.empty:
        raise ValueError(f"No data returned for {ticker}. yfinance may be throttling.")

    df = df.reset_index()

    # yfinance sometimes returns multiindex column names
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    # ---- FIX: normalize timestamps to US/Eastern BEFORE filtering ----
    dt_col = "Datetime" if "Datetime" in df.columns else "Date"
    df[dt_col] = pd.to_datetime(df[dt_col], utc=True).dt.tz_convert("US/Eastern").dt.tz_localize(None)
    df = df.rename(columns={dt_col: "Datetime"})

    # RTH filter now works correctly
    df["time"] = df["Datetime"].dt.strftime("%H:%M")
    df = df[(df["time"] >= MARKET_OPEN) & (df["time"] <= MARKET_CLOSE)].copy()

    df["date"] = df["Datetime"].dt.date
    df = df.sort_values("Datetime").reset_index(drop=True)

    # Quick sanity prints (remove once confirmed)
    print(f"{ticker} rows after RTH filter:", len(df))
    if len(df) > 0:
        print(f"{ticker} first/last:", df['Datetime'].iloc[0], "->", df['Datetime'].iloc[-1])

    return df


# ---------------------------
# ORB helpers
# ---------------------------
def bars_needed(window_min: int) -> int:
    # For 5m bars
    return window_min // 5

def compute_orb(day_df: pd.DataFrame, window_min: int) -> Optional[Tuple[float, float]]:
    n = bars_needed(window_min)
    if len(day_df) < n:
        return None
    orb_slice = day_df.iloc[:n]
    return float(orb_slice["High"].max()), float(orb_slice["Low"].min())

def find_first_breakout_index(day_df: pd.DataFrame, start_idx: int, orb_high: float, orb_low: float):
    # scan from start_idx onward (bars after ORB window)
    for i in range(start_idx, len(day_df)-1):  # -1 because we enter next bar open
        c = float(day_df.iloc[i]["Close"])
        if c > orb_high:
            return i, "LONG"
        if c < orb_low:
            return i, "SHORT"
    return None, None

def simulate_trade(day_df: pd.DataFrame, entry_bar_i: int, side: str,
                   sl: float, tp: float) -> str:
    """
    entry_bar_i is the bar where we ENTER (we enter at its Open).
    We start evaluating exits on entry_bar_i and onward.
    Returns: "WIN", "LOSS", or "EOD"
    """
    for j in range(entry_bar_i, len(day_df)):
        hi = float(day_df.iloc[j]["High"])
        lo = float(day_df.iloc[j]["Low"])

        if side == "LONG":
            hit_sl = lo <= sl
            hit_tp = hi >= tp
            if hit_sl and hit_tp and CONSERVATIVE_SAME_BAR:
                return "LOSS"
            if hit_sl:
                return "LOSS"
            if hit_tp:
                return "WIN"

        else:  # SHORT
            hit_sl = hi >= sl
            hit_tp = lo <= tp
            if hit_sl and hit_tp and CONSERVATIVE_SAME_BAR:
                return "LOSS"
            if hit_sl:
                return "LOSS"
            if hit_tp:
                return "WIN"

    return "EOD"


# ---------------------------
# Backtests
# ---------------------------
def backtest_fixed_rr(df: pd.DataFrame, window_min: int, rr: float) -> pd.DataFrame:
    rows = []
    for d, day_df in df.groupby("date"):
        day_df = day_df.reset_index(drop=True)

        orb = compute_orb(day_df, window_min)
        if orb is None:
            continue
        orb_high, orb_low = orb
        n = bars_needed(window_min)

        trigger_i, side = find_first_breakout_index(day_df, start_idx=n, orb_high=orb_high, orb_low=orb_low)
        if trigger_i is None:
            continue

        # enter next bar open
        entry_i = trigger_i + 1
        entry = float(day_df.iloc[entry_i]["Open"])

        # SL = opposite ORB boundary
        if side == "LONG":
            sl = orb_low
            risk = entry - sl
            if risk <= 0:
                continue
            tp = entry + rr * risk
        else:
            sl = orb_high
            risk = sl - entry
            if risk <= 0:
                continue
            tp = entry - rr * risk

        outcome = simulate_trade(day_df, entry_i, side, sl, tp)

        rows.append({
            "date": d,
            "window_min": window_min,
            "method": "fixed_rr",
            "param": rr,
            "side": side,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "outcome": outcome
        })

    return pd.DataFrame(rows)

def backtest_dynamic(df: pd.DataFrame, window_min: int, k: float) -> pd.DataFrame:
    rows = []
    for d, day_df in df.groupby("date"):
        day_df = day_df.reset_index(drop=True)

        orb = compute_orb(day_df, window_min)
        if orb is None:
            continue
        orb_high, orb_low = orb
        rng = orb_high - orb_low
        if rng <= 0:
            continue

        n = bars_needed(window_min)
        trigger_i, side = find_first_breakout_index(day_df, start_idx=n, orb_high=orb_high, orb_low=orb_low)
        if trigger_i is None:
            continue

        entry_i = trigger_i + 1
        entry = float(day_df.iloc[entry_i]["Open"])

        # SL = opposite boundary, TP = k * ORB range
        if side == "LONG":
            sl = orb_low
            tp = entry + k * rng
        else:
            sl = orb_high
            tp = entry - k * rng

        # sanity: ensure risk is positive
        if side == "LONG" and entry <= sl:
            continue
        if side == "SHORT" and entry >= sl:
            continue

        outcome = simulate_trade(day_df, entry_i, side, sl, tp)

        rows.append({
            "date": d,
            "window_min": window_min,
            "method": "dynamic",
            "param": k,
            "side": side,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "outcome": outcome
        })

    return pd.DataFrame(rows)

def summarize(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades

    # By default, treat EOD as not a win
    trades["is_win"] = (trades["outcome"] == "WIN").astype(int)
    trades["is_loss"] = (trades["outcome"] == "LOSS").astype(int)
    trades["is_eod"] = (trades["outcome"] == "EOD").astype(int)

    g = trades.groupby(["window_min", "method", "param"]).agg(
        trades=("outcome", "count"),
        wins=("is_win", "sum"),
        losses=("is_loss", "sum"),
        eod=("is_eod", "sum"),
    ).reset_index()

    g["win_rate_all"] = g["wins"] / g["trades"]
    # Optional: win rate excluding EOD exits
    denom = (g["wins"] + g["losses"]).replace(0, np.nan)
    g["win_rate_ex_eod"] = g["wins"] / denom
    return g.sort_values(["window_min", "method", "param"]).reset_index(drop=True)

def run_symbol(name: str, ticker: str) -> None:
    df = load_intraday(ticker)

    all_trades = []
    for w in WINDOWS_MIN:
        for rr in RR_LIST:
            all_trades.append(backtest_fixed_rr(df, w, rr))
        for k in K_LIST:
            all_trades.append(backtest_dynamic(df, w, k))

    trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    print(f"\n===== {name} ({ticker}) =====")
    if trades.empty:
        print("No trades found.")
        return
    summary = summarize(trades)
    print(summary.to_string(index=False))

def main():
    for name, ticker in TICKERS.items():
        run_symbol(name, ticker)

if __name__ == "__main__":
    main()
