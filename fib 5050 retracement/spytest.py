#01.py

import math
from dataclasses import dataclass
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import yfinance as yf


# ----------------------------
# If you already have your own clean_yf_data(), import + use it here.
# From your prior preference: ALWAYS run it on yfinance output.
# ----------------------------
def clean_yf_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal cleaning helper. Replace with your own clean_yf_data() if you have it.
    - Flattens columns if needed
    - Ensures OHLCV are standard columns
    """
    if df is None or df.empty:
        return df

    df = df.copy()

    # Handle MultiIndex columns (common with yfinance)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    # Standardize names
    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    df = df.rename(columns=rename_map)

    # Keep only what we need
    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[keep]

    df = df.dropna()
    return df


@dataclass
class Config:
    ticker: str = "SPY"
    period: str = "60d"
    interval: str = "5m"

    # Impulse detection
    LOOKBACK_BARS: int = 24            # how far back to look for a "local low" (e.g., 24 bars = 2 hours on 5m)
    IMPULSE_SIZE: float = 1.50         # price units; for SPY maybe 1.5; for SPX you might use 40.0
    PULLBACK_TRIGGER: float = 0.30     # how far off the impulse high to consider that "pullback started"

    # Fib entry
    FIB_RETRACE: float = 0.50          # 0.50 = 50% retrace

    # Trade management
    STOP_PTS: float = 0.30             # fixed stop distance in price units
    TARGET_PTS: float = 0.60           # fixed target distance in price units
    MAX_HOLD_BARS: int = 24            # time stop, e.g., 24 bars = 2 hours

    # Execution assumptions
    ENTER_ON_TOUCH: bool = True        # True = enter when bar low <= entry <= bar high
    CONSERVATIVE_SAME_BAR: bool = True # if stop+target hit same bar, assume stop first (conservative)


def fetch_5m_data(cfg: Config) -> pd.DataFrame:
    df = yf.download(
        tickers=cfg.ticker,
        period=cfg.period,
        interval=cfg.interval,
        auto_adjust=False,
        progress=False,
        threads=True,
    )
    df = clean_yf_data(df)
    if df is None or df.empty:
        raise ValueError("No data returned. Try a different ticker/period/interval.")
    return df


def backtest_fib_retrace_long(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    State machine:
    IDLE -> BUILD_IMPULSE -> WAIT_RETRACE -> IN_TRADE
    """
    # Ensure index is datetime-like
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    trades: List[Dict] = []

    state = "IDLE"

    # Impulse tracking
    impulse_start_idx: Optional[int] = None
    L = None  # impulse low
    H = None  # impulse high
    locked_entry = None

    # Trade tracking
    entry_price = stop_price = target_price = None
    entry_time = None
    hold_bars = 0

    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    times = df.index

    for i in range(len(df)):
        hi = float(highs[i])
        lo = float(lows[i])
        cl = float(closes[i])
        t = times[i]

        # --- IDLE: look for a potential impulse start ---
        if state == "IDLE":
            if i < cfg.LOOKBACK_BARS:
                continue

            # Define a "local low" over lookback
            recent_low = float(np.min(lows[i - cfg.LOOKBACK_BARS:i]))
            recent_high = float(np.max(highs[i - cfg.LOOKBACK_BARS:i]))

            # Simple start condition: take out recent high (momentum kick)
            if hi > recent_high:
                state = "BUILD_IMPULSE"
                impulse_start_idx = i
                L = recent_low
                H = hi
                locked_entry = None
                continue

        # --- BUILD_IMPULSE: keep updating L/H until impulse size achieved + pullback starts ---
        if state == "BUILD_IMPULSE":
            # Update L/H since start
            if impulse_start_idx is None:
                state = "IDLE"
                continue

            # Update L with current low; H with current high
            L = min(L, lo) if L is not None else lo
            H = max(H, hi) if H is not None else hi

            impulse_range = (H - L) if (H is not None and L is not None) else 0.0

            # If not big enough yet, keep building
            if impulse_range < cfg.IMPULSE_SIZE:
                continue

            # Once big enough, wait for evidence pullback started:
            # pullback if close is below (H - trigger)
            if cl <= (H - cfg.PULLBACK_TRIGGER):
                # Lock fib entry based on the impulse L/H
                locked_entry = H - cfg.FIB_RETRACE * (H - L)
                state = "WAIT_RETRACE"
                continue

        # --- WAIT_RETRACE: wait until price reaches fib level, then enter ---
        if state == "WAIT_RETRACE":
            if locked_entry is None or L is None or H is None:
                state = "IDLE"
                continue

            # Optional invalidation: if price makes a NEW high, update H and recompute entry (or just invalidate).
            # We'll invalidate to avoid chasing a shifting impulse.
            if hi > H:
                state = "IDLE"
                impulse_start_idx = None
                L = H = locked_entry = None
                continue

            if cfg.ENTER_ON_TOUCH:
                touched = (lo <= locked_entry <= hi)
                entry_hit = touched and (cl >= locked_entry)   # touch + close back above
            else:
                entry_hit = (cl >= locked_entry)

            if entry_hit:
                entry_price = float(locked_entry)
                stop_price = entry_price - cfg.STOP_PTS
                target_price = entry_price + cfg.TARGET_PTS
                entry_time = t
                hold_bars = 0
                state = "IN_TRADE"
                continue

        # --- IN_TRADE: manage stop/target/time-stop ---
        if state == "IN_TRADE":
            hold_bars += 1

            hit_stop = (lo <= stop_price)
            hit_target = (hi >= target_price)

            exit_reason = None
            exit_price = None
            exit_time = t

            if hit_stop and hit_target:
                if cfg.CONSERVATIVE_SAME_BAR:
                    exit_reason = "stop_and_target_same_bar_stop_first"
                    exit_price = float(stop_price)
                else:
                    exit_reason = "stop_and_target_same_bar_target_first"
                    exit_price = float(target_price)

            elif hit_stop:
                exit_reason = "stop"
                exit_price = float(stop_price)

            elif hit_target:
                exit_reason = "target"
                exit_price = float(target_price)

            elif hold_bars >= cfg.MAX_HOLD_BARS:
                exit_reason = "time_stop"
                exit_price = float(cl)

            if exit_reason is not None:
                pnl = exit_price - entry_price
                r_mult = pnl / (entry_price - stop_price) if (entry_price - stop_price) != 0 else np.nan

                trades.append({
                    "ticker": cfg.ticker,
                    "entry_time": entry_time,
                    "exit_time": exit_time,
                    "entry": entry_price,
                    "stop": stop_price,
                    "target": target_price,
                    "exit": exit_price,
                    "pnl": pnl,
                    "R": r_mult,
                    "hold_bars": hold_bars,
                    "exit_reason": exit_reason,
                    "impulse_L": float(L) if L is not None else np.nan,
                    "impulse_H": float(H) if H is not None else np.nan,
                    "fib": cfg.FIB_RETRACE,
                })

                # Reset
                state = "IDLE"
                impulse_start_idx = None
                L = H = locked_entry = None
                entry_price = stop_price = target_price = None
                entry_time = None
                hold_bars = 0

    return pd.DataFrame(trades)


def summarize(trades: pd.DataFrame) -> None:
    if trades is None or trades.empty:
        print("No trades.")
        return

    wins = (trades["pnl"] > 0).sum()
    losses = (trades["pnl"] <= 0).sum()
    win_rate = wins / len(trades)

    avg_pnl = trades["pnl"].mean()
    avg_r = trades["R"].mean()
    med_r = trades["R"].median()
    pf = trades.loc[trades["pnl"] > 0, "pnl"].sum() / abs(trades.loc[trades["pnl"] <= 0, "pnl"].sum() or 1e-9)

    print(f"Trades: {len(trades)}")
    print(f"Wins / Losses: {wins} / {losses} | Win rate: {win_rate:.2%}")
    print(f"Avg PnL: {avg_pnl:.4f} | Avg R: {avg_r:.3f} | Median R: {med_r:.3f}")
    print(f"Profit Factor: {pf:.3f}")
    print("\nExit reasons:")
    print(trades["exit_reason"].value_counts())


if __name__ == "__main__":
    cfg = Config(
        ticker="SPY",
        period="60d",
        interval="5m",

        LOOKBACK_BARS=24,
        IMPULSE_SIZE=1.50,
        PULLBACK_TRIGGER=0.30,

        FIB_RETRACE=0.50,

        STOP_PTS=0.30,
        TARGET_PTS=0.60,
        MAX_HOLD_BARS=24,

        ENTER_ON_TOUCH=True,
        CONSERVATIVE_SAME_BAR=True,
    )

    df = fetch_5m_data(cfg)
    trades = backtest_fib_retrace_long(df, cfg)

    print(trades.tail(10))
    summarize(trades)

    # Save trades
    trades.to_csv(f"{cfg.ticker}_fib50_5m_trades.csv", index=False)
    print(f"\nSaved: {cfg.ticker}_fib50_5m_trades.csv")
