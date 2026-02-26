# orb_expectancy_backtest_1hr_hold.py
# Same as your script, but with a HARD MAX HOLD of 60 minutes per trade.
# If neither TP nor SL is hit within 60 minutes, we exit at that bar's CLOSE.

import pandas as pd
import numpy as np
import yfinance as yf

# ---------------------------
# Config
# ---------------------------
TICKERS = {"SPY": "SPY", "SPX": "^GSPC"}

INTERVAL = "5m"        # 5-minute bars
PERIOD = "60d"         # yfinance intraday limit
WINDOWS_MIN = [15, 30, 60]

RR_LIST = [0.5, 1.0, 1.5, 2.0]
K_LIST  = [0.5, 0.75, 1.0]

MARKET_OPEN  = "09:30"
MARKET_CLOSE = "16:00"

CONSERVATIVE_SAME_BAR = True

# NEW: max time in trade
MAX_HOLD_MIN = 60  # <-- you asked for 1 hour max hold
MAX_HOLD_BARS = MAX_HOLD_MIN // 5  # since interval is 5m; 60m -> 12 bars


# ---------------------------
# Data loading (timezone-fixed)
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
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    dt_col = "Datetime" if "Datetime" in df.columns else "Date"

    df[dt_col] = (
        pd.to_datetime(df[dt_col], utc=True)
          .dt.tz_convert("US/Eastern")
          .dt.tz_localize(None)
    )
    df = df.rename(columns={dt_col: "Datetime"})

    df["time"] = df["Datetime"].dt.strftime("%H:%M")
    df = df[(df["time"] >= MARKET_OPEN) & (df["time"] <= MARKET_CLOSE)].copy()

    df["date"] = df["Datetime"].dt.date
    df = df.sort_values("Datetime").reset_index(drop=True)
    return df


# ---------------------------
# ORB helpers
# ---------------------------
def bars_needed(window_min: int) -> int:
    return window_min // 5

def compute_orb(day_df: pd.DataFrame, window_min: int):
    n = bars_needed(window_min)
    if len(day_df) < n:
        return None
    orb_slice = day_df.iloc[:n]
    orb_high = float(orb_slice["High"].max())
    orb_low  = float(orb_slice["Low"].min())
    return orb_high, orb_low

def find_first_breakout_index(day_df: pd.DataFrame, start_idx: int, orb_high: float, orb_low: float):
    for i in range(start_idx, len(day_df) - 1):
        c = float(day_df.iloc[i]["Close"])
        if c > orb_high:
            return i, "LONG"
        if c < orb_low:
            return i, "SHORT"
    return None, None

def simulate_trade(day_df: pd.DataFrame, entry_bar_i: int, side: str, sl: float, tp: float):
    """
    Evaluate exits from entry_bar_i onward.
    NEW: Hard time stop at MAX_HOLD_BARS (60 min for 5m bars).
    Returns (outcome, exit_price).
      WIN  -> exit_price = tp
      LOSS -> exit_price = sl
      TIME -> exit_price = close of the bar where time stop triggers
      EOD  -> exit_price = last bar close (only possible if near end-of-day before time stop)
    """
    for j in range(entry_bar_i, len(day_df)):
        hi = float(day_df.iloc[j]["High"])
        lo = float(day_df.iloc[j]["Low"])

        if side == "LONG":
            hit_sl = lo <= sl
            hit_tp = hi >= tp
            if hit_sl and hit_tp and CONSERVATIVE_SAME_BAR:
                return "LOSS", sl
            if hit_sl:
                return "LOSS", sl
            if hit_tp:
                return "WIN", tp
        else:
            hit_sl = hi >= sl
            hit_tp = lo <= tp
            if hit_sl and hit_tp and CONSERVATIVE_SAME_BAR:
                return "LOSS", sl
            if hit_sl:
                return "LOSS", sl
            if hit_tp:
                return "WIN", tp

        # NEW: time stop after holding MAX_HOLD_BARS bars (including entry bar)
        bars_held = (j - entry_bar_i) + 1
        if bars_held >= MAX_HOLD_BARS:
            return "TIME", float(day_df.iloc[j]["Close"])

    return "EOD", float(day_df.iloc[-1]["Close"])


# ---------------------------
# Backtests
# ---------------------------
def backtest_fixed_rr(df: pd.DataFrame, window_min: int, rr: float) -> pd.DataFrame:
    rows = []
    n = bars_needed(window_min)

    for d, day_df in df.groupby("date"):
        day_df = day_df.reset_index(drop=True)

        orb = compute_orb(day_df, window_min)
        if orb is None:
            continue
        orb_high, orb_low = orb

        trigger_i, side = find_first_breakout_index(day_df, start_idx=n, orb_high=orb_high, orb_low=orb_low)
        if trigger_i is None:
            continue

        entry_i = trigger_i + 1
        entry = float(day_df.iloc[entry_i]["Open"])

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

        outcome, exit_price = simulate_trade(day_df, entry_i, side, sl, tp)

        if outcome == "WIN":
            r_mult = ((tp - entry) / risk) if side == "LONG" else ((entry - tp) / risk)
        elif outcome == "LOSS":
            r_mult = -1.0
        else:  # TIME or EOD
            r_mult = ((exit_price - entry) / risk) if side == "LONG" else ((entry - exit_price) / risk)

        rows.append({
            "date": d,
            "window_min": window_min,
            "method": "fixed_rr",
            "param": rr,
            "side": side,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "outcome": outcome,
            "exit_price": exit_price,
            "risk": risk,
            "r_mult": float(r_mult),
        })

    return pd.DataFrame(rows)

def backtest_dynamic(df: pd.DataFrame, window_min: int, k: float) -> pd.DataFrame:
    rows = []
    n = bars_needed(window_min)

    for d, day_df in df.groupby("date"):
        day_df = day_df.reset_index(drop=True)

        orb = compute_orb(day_df, window_min)
        if orb is None:
            continue
        orb_high, orb_low = orb
        rng = orb_high - orb_low
        if rng <= 0:
            continue

        trigger_i, side = find_first_breakout_index(day_df, start_idx=n, orb_high=orb_high, orb_low=orb_low)
        if trigger_i is None:
            continue

        entry_i = trigger_i + 1
        entry = float(day_df.iloc[entry_i]["Open"])

        if side == "LONG":
            sl = orb_low
            risk = entry - sl
            if risk <= 0:
                continue
            tp = entry + k * rng
        else:
            sl = orb_high
            risk = sl - entry
            if risk <= 0:
                continue
            tp = entry - k * rng

        outcome, exit_price = simulate_trade(day_df, entry_i, side, sl, tp)

        if outcome == "WIN":
            r_mult = ((tp - entry) / risk) if side == "LONG" else ((entry - tp) / risk)
        elif outcome == "LOSS":
            r_mult = -1.0
        else:  # TIME or EOD
            r_mult = ((exit_price - entry) / risk) if side == "LONG" else ((entry - exit_price) / risk)

        rows.append({
            "date": d,
            "window_min": window_min,
            "method": "dynamic",
            "param": k,
            "side": side,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "outcome": outcome,
            "exit_price": exit_price,
            "risk": risk,
            "r_mult": float(r_mult),
        })

    return pd.DataFrame(rows)


# ---------------------------
# Summaries (Expectancy in R)
# ---------------------------
def profit_factor_r(g: pd.DataFrame) -> float:
    wins_sum = g.loc[g["r_mult"] > 0, "r_mult"].sum()
    losses_sum = g.loc[g["r_mult"] < 0, "r_mult"].sum()
    if losses_sum == 0:
        return float("inf") if wins_sum > 0 else 0.0
    return float(wins_sum / abs(losses_sum))

def summarize(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades

    trades = trades.copy()
    trades["is_win"]  = (trades["outcome"] == "WIN").astype(int)
    trades["is_loss"] = (trades["outcome"] == "LOSS").astype(int)
    trades["is_time"] = (trades["outcome"] == "TIME").astype(int)
    trades["is_eod"]  = (trades["outcome"] == "EOD").astype(int)

    out_rows = []
    for (w, m, p), g in trades.groupby(["window_min", "method", "param"]):
        n = len(g)
        wins = int(g["is_win"].sum())
        losses = int(g["is_loss"].sum())
        time_exits = int(g["is_time"].sum())
        eod = int(g["is_eod"].sum())

        expectancy = float(g["r_mult"].mean())
        avg_win_r = float(g.loc[g["outcome"] == "WIN", "r_mult"].mean()) if wins > 0 else np.nan
        avg_loss_r = float(g.loc[g["outcome"] == "LOSS", "r_mult"].mean()) if losses > 0 else np.nan
        avg_time_r = float(g.loc[g["outcome"] == "TIME", "r_mult"].mean()) if time_exits > 0 else np.nan

        out_rows.append({
            "window_min": w,
            "method": m,
            "param": p,
            "trades": n,
            "wins": wins,
            "losses": losses,
            "time": time_exits,
            "eod": eod,
            "win_rate_all": wins / n if n else np.nan,
            "win_rate_ex_resolved": (wins / (wins + losses)) if (wins + losses) else np.nan,
            "expectancy_R": expectancy,
            "avg_win_R": avg_win_r,
            "avg_loss_R": avg_loss_r,
            "avg_time_R": avg_time_r,
            "profit_factor_R": profit_factor_r(g),
        })

    return (pd.DataFrame(out_rows)
            .sort_values(["window_min", "method", "param"])
            .reset_index(drop=True))


# ---------------------------
# Runner
# ---------------------------
def run_symbol(name: str, ticker: str) -> None:
    df = load_intraday(ticker)
    print(f"\n{name} rows after RTH filter: {len(df)}")
    if len(df) > 0:
        print(f"{name} first/last: {df['Datetime'].iloc[0]} -> {df['Datetime'].iloc[-1]}")

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
    print(f"Max hold: {MAX_HOLD_MIN} minutes ({MAX_HOLD_BARS} bars @ {INTERVAL})")
    for name, ticker in TICKERS.items():
        run_symbol(name, ticker)

if __name__ == "__main__":
    main()
