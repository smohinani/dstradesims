import numpy as np
import pandas as pd
import yfinance as yf
from dataclasses import dataclass
from typing import Optional, List, Dict


# ----------------------------
# Minimal cleaner (swap in your own clean_yf_data if you prefer)
# ----------------------------
def clean_yf_data(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    })
    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[keep].dropna()
    return df


@dataclass
class Config:
    ticker: str = "ES=F"
    period: str = "60d"          # yfinance 5m limit ~60d; keep this for 5m
    interval: str = "5m"

    # Session filter (RTH only)
    USE_RTH: bool = True
    RTH_TZ: str = "US/Eastern"
    RTH_START: str = "09:30"
    RTH_END: str = "16:00"

    # Impulse detection
    LOOKBACK_BARS: int = 24

    # Bigger moves OK: loosen max
    IMPULSE_SIZE_MIN: float = 40.0
    IMPULSE_SIZE_MAX: float = 160.0
    MAX_IMPULSE_BARS: int = 24
    PULLBACK_TRIGGER: float = 10.0

    # Fib entry
    FIB_RETRACE: float = 0.50
    INVALIDATE_BELOW_618: bool = True
    RECLAIM_BUFFER: float = 0.50

    # Trade management (fixed points)
    STOP_PTS: float = 5.0
    TARGET_PTS: float = 10.0
    MAX_HOLD_BARS: int = 24

    # Entry timing filter
    BLOCK_LATE_ENTRIES: bool = True
    LATE_ENTRY_HOUR: int = 15
    LATE_ENTRY_MINUTE: int = 30

    # Execution assumptions
    CONSERVATIVE_SAME_BAR: bool = True

    # Costs (ES tick = 0.25)
    COST_POINTS: float = 0.25


def _is_late_entry(t, cfg: Config) -> bool:
    if not cfg.BLOCK_LATE_ENTRIES:
        return False
    h, m = t.hour, t.minute
    return (h > cfg.LATE_ENTRY_HOUR) or (h == cfg.LATE_ENTRY_HOUR and m >= cfg.LATE_ENTRY_MINUTE)


def _profit_factor(pnl: pd.Series) -> float:
    gross_win = pnl[pnl > 0].sum()
    gross_loss = pnl[pnl <= 0].sum()
    if gross_loss == 0:
        return float("inf")
    return float(gross_win / abs(gross_loss))


def fetch_5m_data(cfg: Config) -> pd.DataFrame:
    # Hard cap for yfinance intraday
    if cfg.interval.endswith("m") and cfg.interval in {"1m", "2m", "5m", "15m", "30m", "60m", "90m"}:
        if cfg.period != "60d":
            print(f"[info] yfinance intraday ({cfg.interval}) is limited; overriding period {cfg.period} -> 60d")
            cfg.period = "60d"

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
        raise ValueError("No data returned from yfinance. For 5m, use period='60d'.")

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    if cfg.USE_RTH:
        df = df.tz_convert(cfg.RTH_TZ)
        df = df.between_time(cfg.RTH_START, cfg.RTH_END)

    return df


def backtest_fib_retrace_long_short(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    times = df.index

    trades: List[Dict] = []

    # Scanners (long + short) + single-position rule
    state_L, state_S = "IDLE_L", "IDLE_S"

    # Long impulse
    L_low = H_high = entry_L = None
    impulse_start_L: Optional[int] = None

    # Short impulse
    H_high_S = L_low_S = entry_S = None
    impulse_start_S: Optional[int] = None

    # Position
    pos: Optional[str] = None  # "LONG" / "SHORT"
    entry_price = stop_price = target_price = None
    entry_time = None
    hold_bars = 0
    impulse_L = impulse_H = None

    for i in range(len(df)):
        hi = float(highs[i])
        lo = float(lows[i])
        cl = float(closes[i])
        t = times[i]

        # --------------------
        # Manage open position
        # --------------------
        if pos is not None:
            hold_bars += 1

            if pos == "LONG":
                hit_stop = (lo <= stop_price)
                hit_target = (hi >= target_price)
            else:
                hit_stop = (hi >= stop_price)
                hit_target = (lo <= target_price)

            exit_reason = None
            exit_price = None

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
                if pos == "LONG":
                    pnl_pts = (exit_price - entry_price) - cfg.COST_POINTS
                else:
                    pnl_pts = (entry_price - exit_price) - cfg.COST_POINTS

                r_mult = pnl_pts / cfg.STOP_PTS if cfg.STOP_PTS != 0 else np.nan

                trades.append({
                    "ticker": cfg.ticker,
                    "side": pos,
                    "entry_time": entry_time,
                    "exit_time": t,
                    "entry": entry_price,
                    "stop": stop_price,
                    "target": target_price,
                    "exit": exit_price,
                    "pnl_points": pnl_pts,
                    "R": r_mult,
                    "hold_bars": hold_bars,
                    "exit_reason": exit_reason,
                    "impulse_L": float(impulse_L),
                    "impulse_H": float(impulse_H),
                    "impulse_range": float(impulse_H - impulse_L),
                    "fib": cfg.FIB_RETRACE,
                })

                # Reset everything after a trade
                pos = None
                entry_price = stop_price = target_price = None
                entry_time = None
                hold_bars = 0
                impulse_L = impulse_H = None

                state_L, state_S = "IDLE_L", "IDLE_S"
                L_low = H_high = entry_L = None
                impulse_start_L = None
                H_high_S = L_low_S = entry_S = None
                impulse_start_S = None

            continue

        # --------------------
        # LONG scanner
        # --------------------
        if state_L == "IDLE_L":
            if i >= cfg.LOOKBACK_BARS:
                recent_low = float(np.min(lows[i - cfg.LOOKBACK_BARS:i]))
                recent_high = float(np.max(highs[i - cfg.LOOKBACK_BARS:i]))
                if hi > recent_high:
                    state_L = "BUILD_L"
                    impulse_start_L = i
                    L_low = recent_low
                    H_high = hi
                    entry_L = None

        elif state_L == "BUILD_L":
            if impulse_start_L is None:
                state_L = "IDLE_L"
            else:
                if (i - impulse_start_L) > cfg.MAX_IMPULSE_BARS:
                    state_L = "IDLE_L"
                    impulse_start_L = None
                    L_low = H_high = entry_L = None
                else:
                    L_low = min(L_low, lo) if L_low is not None else lo
                    H_high = max(H_high, hi) if H_high is not None else hi
                    rng = (H_high - L_low) if (H_high is not None and L_low is not None) else 0.0

                    if rng > cfg.IMPULSE_SIZE_MAX:
                        state_L = "IDLE_L"
                        impulse_start_L = None
                        L_low = H_high = entry_L = None
                    elif rng >= cfg.IMPULSE_SIZE_MIN:
                        if cl <= (H_high - cfg.PULLBACK_TRIGGER):
                            entry_L = H_high - cfg.FIB_RETRACE * (H_high - L_low)
                            state_L = "WAIT_L"

        elif state_L == "WAIT_L":
            if entry_L is None or L_low is None or H_high is None:
                state_L = "IDLE_L"
                impulse_start_L = None
                L_low = H_high = entry_L = None
            else:
                if hi > H_high:
                    state_L = "IDLE_L"
                    impulse_start_L = None
                    L_low = H_high = entry_L = None
                else:
                    if cfg.INVALIDATE_BELOW_618:
                        lvl_618 = H_high - 0.618 * (H_high - L_low)
                        if lo < lvl_618:
                            state_L = "IDLE_L"
                            impulse_start_L = None
                            L_low = H_high = entry_L = None

                    if state_L == "WAIT_L":
                        if _is_late_entry(t, cfg):
                            state_L = "IDLE_L"
                            impulse_start_L = None
                            L_low = H_high = entry_L = None
                        else:
                            touched = (lo <= entry_L <= hi)
                            enter = touched and (cl >= entry_L + cfg.RECLAIM_BUFFER)
                            if enter:
                                pos = "LONG"
                                entry_price = float(entry_L)
                                stop_price = entry_price - cfg.STOP_PTS
                                target_price = entry_price + cfg.TARGET_PTS
                                entry_time = t
                                hold_bars = 0
                                impulse_L = float(L_low)
                                impulse_H = float(H_high)

        # --------------------
        # SHORT scanner (only if we didn't enter long this bar)
        # --------------------
        if pos is None:
            if state_S == "IDLE_S":
                if i >= cfg.LOOKBACK_BARS:
                    recent_low = float(np.min(lows[i - cfg.LOOKBACK_BARS:i]))
                    recent_high = float(np.max(highs[i - cfg.LOOKBACK_BARS:i]))
                    if lo < recent_low:
                        state_S = "BUILD_S"
                        impulse_start_S = i
                        H_high_S = recent_high
                        L_low_S = lo
                        entry_S = None

            elif state_S == "BUILD_S":
                if impulse_start_S is None:
                    state_S = "IDLE_S"
                else:
                    if (i - impulse_start_S) > cfg.MAX_IMPULSE_BARS:
                        state_S = "IDLE_S"
                        impulse_start_S = None
                        H_high_S = L_low_S = entry_S = None
                    else:
                        H_high_S = max(H_high_S, hi) if H_high_S is not None else hi
                        L_low_S = min(L_low_S, lo) if L_low_S is not None else lo
                        rng = (H_high_S - L_low_S) if (H_high_S is not None and L_low_S is not None) else 0.0

                        if rng > cfg.IMPULSE_SIZE_MAX:
                            state_S = "IDLE_S"
                            impulse_start_S = None
                            H_high_S = L_low_S = entry_S = None
                        elif rng >= cfg.IMPULSE_SIZE_MIN:
                            if cl >= (L_low_S + cfg.PULLBACK_TRIGGER):
                                entry_S = L_low_S + cfg.FIB_RETRACE * (H_high_S - L_low_S)
                                state_S = "WAIT_S"

            elif state_S == "WAIT_S":
                if entry_S is None or L_low_S is None or H_high_S is None:
                    state_S = "IDLE_S"
                    impulse_start_S = None
                    H_high_S = L_low_S = entry_S = None
                else:
                    if lo < L_low_S:
                        state_S = "IDLE_S"
                        impulse_start_S = None
                        H_high_S = L_low_S = entry_S = None
                    else:
                        if cfg.INVALIDATE_BELOW_618:
                            lvl_618 = L_low_S + 0.618 * (H_high_S - L_low_S)
                            if hi > lvl_618:
                                state_S = "IDLE_S"
                                impulse_start_S = None
                                H_high_S = L_low_S = entry_S = None

                        if state_S == "WAIT_S":
                            if _is_late_entry(t, cfg):
                                state_S = "IDLE_S"
                                impulse_start_S = None
                                H_high_S = L_low_S = entry_S = None
                            else:
                                touched = (lo <= entry_S <= hi)
                                enter = touched and (cl <= entry_S - cfg.RECLAIM_BUFFER)
                                if enter:
                                    pos = "SHORT"
                                    entry_price = float(entry_S)
                                    stop_price = entry_price + cfg.STOP_PTS
                                    target_price = entry_price - cfg.TARGET_PTS
                                    entry_time = t
                                    hold_bars = 0
                                    impulse_L = float(L_low_S)
                                    impulse_H = float(H_high_S)

    return pd.DataFrame(trades)


def summarize(trades: pd.DataFrame, title: str = "") -> None:
    if title:
        print(f"\n=== {title} ===")
    if trades is None or trades.empty:
        print("No trades.")
        return

    wins = (trades["pnl_points"] > 0).sum()
    losses = (trades["pnl_points"] <= 0).sum()
    win_rate = wins / len(trades)
    pf = _profit_factor(trades["pnl_points"])

    print(f"Trades: {len(trades)}")
    print(f"Wins / Losses: {wins} / {losses} | Win rate: {win_rate:.2%}")
    print(f"Avg pnl (pts): {trades['pnl_points'].mean():.3f} | Avg R: {trades['R'].mean():.3f} | Median R: {trades['R'].median():.3f}")
    print(f"Profit Factor: {pf:.3f}")
    print("\nExit reasons:")
    print(trades["exit_reason"].value_counts())

    if "impulse_range" in trades.columns:
        print("\nImpulse range (pts):")
        print(trades["impulse_range"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]))


if __name__ == "__main__":
    cfg = Config(
        ticker="ES=F",
        period="60d",     # must be 60d for yfinance 5m
        interval="5m",

        USE_RTH=True,

        LOOKBACK_BARS=24,

        IMPULSE_SIZE_MIN=40.0,
        IMPULSE_SIZE_MAX=160.0,   # bigger legs allowed
        MAX_IMPULSE_BARS=24,
        PULLBACK_TRIGGER=10.0,

        FIB_RETRACE=0.50,
        INVALIDATE_BELOW_618=True,
        RECLAIM_BUFFER=0.50,

        STOP_PTS=5.0,
        TARGET_PTS=10.0,
        MAX_HOLD_BARS=24,

        BLOCK_LATE_ENTRIES=True,
        LATE_ENTRY_HOUR=15,
        LATE_ENTRY_MINUTE=30,

        CONSERVATIVE_SAME_BAR=True,
        COST_POINTS=0.25,
    )

    df = fetch_5m_data(cfg)
    trades = backtest_fib_retrace_long_short(df, cfg)

    print(trades.tail(20))

    summarize(trades, "ALL (Long + Short)")
    summarize(trades[trades["side"] == "LONG"], "LONG ONLY")
    summarize(trades[trades["side"] == "SHORT"], "SHORT ONLY")

    out = f"{cfg.ticker.replace('=','_')}_fib50_5m_trades.csv"
    trades.to_csv(out, index=False)
    print(f"\nSaved: {out}")
