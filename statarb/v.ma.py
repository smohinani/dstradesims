# statarb_backtest_rth_holdbars_fixed.py
# Intraday stat-arb backtest (yfinance only) with:
# - RTH-only (09:30–16:00 ET)
# - Strict 1-minute grid per day (no overnight minutes)
# - HOLD TIME measured in RTH minutes (bar count)
# - TIME stop + OVERNIGHT_FLAT cannot be skipped (managed before filters)
# - Trade log CSV output

import yfinance as yf
import pandas as pd
import numpy as np
from dataclasses import dataclass
import math
from datetime import time as dtime


@dataclass
class PairConfig:
    a: str
    b: str
    lookback: int = 90
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float = 3.0
    min_corr: float = 0.75
    max_hold_min: int = 30
    max_notional: float = 9000.0


PAIRS = [
    # ⭐ Payments / Financial Infra
    PairConfig("V", "MA",   min_corr=0.75, entry_z=2.0, exit_z=0.5, stop_z=3.0, max_hold_min=30),
    PairConfig("V", "AXP",  min_corr=0.75, entry_z=2.1, exit_z=0.5, stop_z=3.2, max_hold_min=30),
    PairConfig("MA", "AXP", min_corr=0.75, entry_z=2.1, exit_z=0.5, stop_z=3.2, max_hold_min=30),

    # ⭐ Gold / Precious Metals
    PairConfig("GLD", "IAU", min_corr=0.85, entry_z=2.0, exit_z=0.5, stop_z=3.0, max_hold_min=35),
    PairConfig("GLD", "SLV", min_corr=0.80, entry_z=2.2, exit_z=0.6, stop_z=3.4, max_hold_min=35),
    PairConfig("IAU", "SLV", min_corr=0.80, entry_z=2.2, exit_z=0.6, stop_z=3.4, max_hold_min=35),

    # ⭐ Broad Market ETFs (sanity-check pairs)
    PairConfig("SPY", "IVV", min_corr=0.95, entry_z=2.0, exit_z=0.4, stop_z=3.0, max_hold_min=25),
    PairConfig("SPY", "VOO", min_corr=0.95, entry_z=2.0, exit_z=0.4, stop_z=3.0, max_hold_min=25),
    PairConfig("QQQ", "SPY", min_corr=0.90, entry_z=2.3, exit_z=0.6, stop_z=3.5, max_hold_min=35),
    PairConfig("QQQ", "VTI", min_corr=0.88, entry_z=2.4, exit_z=0.6, stop_z=3.6, max_hold_min=35),

    # ⭐ Financial Sector ETFs
    PairConfig("XLF", "KBE", min_corr=0.88, entry_z=2.2, exit_z=0.6, stop_z=3.4, max_hold_min=35),
    PairConfig("XLF", "KRE", min_corr=0.85, entry_z=2.3, exit_z=0.6, stop_z=3.5, max_hold_min=35),
    PairConfig("KBE", "KRE", min_corr=0.85, entry_z=2.3, exit_z=0.6, stop_z=3.5, max_hold_min=35),

    # ⭐ Tech / Semis (liquid, but spikier)
    PairConfig("XLK",  "QQQ",  min_corr=0.90, entry_z=2.3, exit_z=0.6, stop_z=3.5, max_hold_min=35),
    PairConfig("SMH",  "SOXX", min_corr=0.90, entry_z=2.2, exit_z=0.5, stop_z=3.3, max_hold_min=35),
    PairConfig("NVDA", "AMD",  min_corr=0.75, entry_z=2.6, exit_z=0.7, stop_z=3.9, max_hold_min=25),

    # Energy (often better on 2–5m bars, but try anyway)
    PairConfig("XLE", "VDE", min_corr=0.88, entry_z=2.3, exit_z=0.6, stop_z=3.6, max_hold_min=40),
    PairConfig("XOM", "CVX", min_corr=0.80, entry_z=2.6, exit_z=0.7, stop_z=3.9, max_hold_min=40),

    # Banks (single-name; keep stricter filters)
    PairConfig("GS",  "JPM", min_corr=0.85, entry_z=2.3, exit_z=0.6, stop_z=3.2, max_hold_min=25),
    PairConfig("BAC", "C",   min_corr=0.85, entry_z=2.4, exit_z=0.6, stop_z=3.6, max_hold_min=30),
    PairConfig("MS",  "GS",  min_corr=0.80, entry_z=2.5, exit_z=0.7, stop_z=3.8, max_hold_min=30),

    # Mega-cap tech (works, but slower/less “pure” intraday)
    PairConfig("AAPL",  "MSFT", min_corr=0.80, entry_z=2.6, exit_z=0.7, stop_z=3.9, max_hold_min=35),
    PairConfig("GOOGL", "META", min_corr=0.75, entry_z=2.7, exit_z=0.7, stop_z=4.0, max_hold_min=35),
    PairConfig("AMZN",  "META", min_corr=0.75, entry_z=2.7, exit_z=0.7, stop_z=4.0, max_hold_min=35),
]

START_EQUITY = 10_000.0
YF_INTERVAL = "1m"
YF_PERIOD = "5d"
TZ = "America/New_York"

RTH_START = dtime(9, 30)
RTH_END = dtime(16, 0)


# ---------------------
# Data helpers
# ---------------------

def clean_yf_intraday(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["close"])

    df = df.copy()

    # Flatten MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(c[0]).lower() for c in df.columns]
    else:
        df.columns = [str(c).lower() for c in df.columns]

    if "close" not in df.columns:
        if "adj close" in df.columns:
            df["close"] = df["adj close"]
        else:
            return pd.DataFrame(columns=["close"])

    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()]

    # Convert index to ET
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert(TZ)
    else:
        df.index = df.index.tz_convert(TZ)

    return df[["close"]].dropna().sort_index()


def rth_only(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df.between_time("09:30", "16:00")


def regularize_per_day_rth(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build strict 1-minute grid per trading day from 09:30 to 16:00 ET.
    Forward-fill inside the day only (no overnight fill).
    """
    if df.empty:
        return df

    chunks = []
    for day, g in df.groupby(df.index.date):
        g = g.sort_index()
        tz = g.index.tz

        day_start = pd.Timestamp.combine(pd.Timestamp(day), RTH_START).tz_localize(tz)
        day_end = pd.Timestamp.combine(pd.Timestamp(day), RTH_END).tz_localize(tz)

        idx = pd.date_range(day_start, day_end, freq="1min", tz=tz)
        gg = g.reindex(idx).ffill()
        gg = gg.loc[day_start:day_end].dropna()
        chunks.append(gg)

    return pd.concat(chunks).dropna().sort_index()


# ---------------------
# Math helpers
# ---------------------

def rolling_beta(a: np.ndarray, b: np.ndarray) -> float:
    vb = np.var(b)
    if len(a) < 10 or vb <= 0:
        return float("nan")
    cov = np.cov(b, a, ddof=0)[0, 1]
    return float(cov / vb)

def zscore(x: np.ndarray) -> float:
    if len(x) < 10:
        return float("nan")
    s = float(np.std(x, ddof=0))
    if s == 0:
        return float("nan")
    m = float(np.mean(x))
    return float((x[-1] - m) / s)

def corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 10:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])

def size_positions(price_a: float, price_b: float, beta: float, max_notional: float):
    """
    Dollar-neutral-ish sizing using beta magnitude:
      allocate notionals so A ~ B*|beta| and A+B = max_notional
    """
    beta_abs = max(1e-6, abs(beta))
    notional_b = max_notional / (1.0 + beta_abs)
    notional_a = max_notional - notional_b
    qa = max(1, int(notional_a / price_a))
    qb = max(1, int(notional_b / price_b))
    return qa, qb


# ---------------------
# Backtest
# ---------------------

def backtest_pair(pc: PairConfig, df_a: pd.DataFrame, df_b: pd.DataFrame):
    # RTH + strict 1m grid per day (no overnight minutes)
    df_a = regularize_per_day_rth(rth_only(df_a)).rename(columns={"close": "close_a"})
    df_b = regularize_per_day_rth(rth_only(df_b)).rename(columns={"close": "close_b"})

    df = df_a.join(df_b, how="inner").dropna()
    if df.empty or len(df) < pc.lookback + 10:
        return START_EQUITY, pd.DataFrame()

    equity = START_EQUITY
    rows = []
    pos = None
    pos_day = None

    for i in range(pc.lookback, len(df)):
        price_a = float(df["close_a"].iloc[i])
        price_b = float(df["close_b"].iloc[i])
        t = df.index[i]
        cur_day = t.date()

        # =========================
        # ALWAYS manage open position FIRST
        # (prevents TIME/OVERNIGHT from being skipped by filter continues)
        # =========================
        if pos is not None:
            hold_min_rth = float(i - pos["entry_i"])  # 1 row == 1 RTH minute on our grid

            # Overnight flat: first bar of a new day
            if pos_day is not None and cur_day != pos_day:
                pnl = (
                    pos["dir"] * pos["qa"] * (price_a - pos["entry_a"])
                    - pos["dir"] * pos["qb"] * (price_b - pos["entry_b"])
                )
                equity += pnl
                rows.append({
                    "pair": f"{pc.a}-{pc.b}",
                    "side": "LONG_SPREAD" if pos["dir"] == 1 else "SHORT_SPREAD",
                    "entry_time_et": pos["entry_t"],
                    "exit_time_et": t,
                    "hold_min": hold_min_rth,
                    "entry_z": pos["entry_z"],
                    "exit_z": np.nan,
                    "beta": pos["beta"],
                    "qty_a": pos["qa"] * pos["dir"],
                    "qty_b": -pos["qb"] * pos["dir"],
                    "entry_a": pos["entry_a"],
                    "entry_b": pos["entry_b"],
                    "exit_a": price_a,
                    "exit_b": price_b,
                    "pnl": pnl,
                    "exit_reason": "OVERNIGHT_FLAT",
                })
                pos = None
                pos_day = None
                continue

            # Time stop: enforced no matter what
            if hold_min_rth >= pc.max_hold_min:
                pnl = (
                    pos["dir"] * pos["qa"] * (price_a - pos["entry_a"])
                    - pos["dir"] * pos["qb"] * (price_b - pos["entry_b"])
                )
                equity += pnl
                rows.append({
                    "pair": f"{pc.a}-{pc.b}",
                    "side": "LONG_SPREAD" if pos["dir"] == 1 else "SHORT_SPREAD",
                    "entry_time_et": pos["entry_t"],
                    "exit_time_et": t,
                    "hold_min": hold_min_rth,
                    "entry_z": pos["entry_z"],
                    "exit_z": np.nan,
                    "beta": pos["beta"],
                    "qty_a": pos["qa"] * pos["dir"],
                    "qty_b": -pos["qb"] * pos["dir"],
                    "entry_a": pos["entry_a"],
                    "entry_b": pos["entry_b"],
                    "exit_a": price_a,
                    "exit_b": price_b,
                    "pnl": pnl,
                    "exit_reason": "TIME",
                })
                pos = None
                pos_day = None
                continue

        # =========================
        # Compute corr/beta/z for entry + TP/STOP
        # =========================
        window = df.iloc[i - pc.lookback:i]
        a = window["close_a"].to_numpy(float)
        b = window["close_b"].to_numpy(float)

        c = corr(a, b)
        if not math.isfinite(c) or c < pc.min_corr:
            continue

        beta = rolling_beta(a, b)
        if not math.isfinite(beta):
            continue

        spread = a - beta * b
        z = zscore(spread)
        if not math.isfinite(z):
            continue

        # ENTRY
        if pos is None and abs(z) >= pc.entry_z:
            direction = 1 if z < 0 else -1
            qa, qb = size_positions(price_a, price_b, beta, pc.max_notional)
            pos = {
                "dir": direction,
                "qa": qa,
                "qb": qb,
                "beta": beta,
                "entry_a": price_a,
                "entry_b": price_b,
                "entry_t": t,
                "entry_z": z,
                "entry_i": i,
            }
            pos_day = cur_day
            continue

        # TP/STOP
        if pos is not None:
            exit_tp = abs(z) <= pc.exit_z
            exit_stop = abs(z) >= pc.stop_z
            if exit_tp or exit_stop:
                hold_min_rth = float(i - pos["entry_i"])
                pnl = (
                    pos["dir"] * pos["qa"] * (price_a - pos["entry_a"])
                    - pos["dir"] * pos["qb"] * (price_b - pos["entry_b"])
                )
                equity += pnl
                reason = "TP" if exit_tp else "STOP"
                rows.append({
                    "pair": f"{pc.a}-{pc.b}",
                    "side": "LONG_SPREAD" if pos["dir"] == 1 else "SHORT_SPREAD",
                    "entry_time_et": pos["entry_t"],
                    "exit_time_et": t,
                    "hold_min": hold_min_rth,
                    "entry_z": pos["entry_z"],
                    "exit_z": z,
                    "beta": pos["beta"],
                    "qty_a": pos["qa"] * pos["dir"],
                    "qty_b": -pos["qb"] * pos["dir"],
                    "entry_a": pos["entry_a"],
                    "entry_b": pos["entry_b"],
                    "exit_a": price_a,
                    "exit_b": price_b,
                    "pnl": pnl,
                    "exit_reason": reason,
                })
                pos = None
                pos_day = None

    return equity, pd.DataFrame(rows)


# ---------------------
# Reporting
# ---------------------

def summarize(pc: PairConfig, equity: float, trades_df: pd.DataFrame):
    print(f"\n=== {pc.a}-{pc.b} ===")
    if trades_df.empty:
        print("No trades.")
        return

    pnls = trades_df["pnl"].to_numpy(float)
    holds = trades_df["hold_min"].to_numpy(float)

    wins = int((pnls > 0).sum())
    losses = int((pnls <= 0).sum())

    print(f"Trades:           {len(pnls)}")
    print(f"Wins/Losses:      {wins}/{losses}  (win rate {wins/len(pnls):.2%})")
    print(f"Avg PnL:          {pnls.mean():.2f}")
    print(f"Median PnL:       {np.median(pnls):.2f}")
    print(f"Total PnL:        {pnls.sum():.2f}")
    print(f"Final Equity:     {equity:.2f}")

    print(f"Avg hold (min):   {holds.mean():.2f}")
    print(f"Median hold (m):  {np.median(holds):.2f}")
    print(f"90% hold (m):     {np.percentile(holds, 90):.2f}")
    print(f"Max hold (m):     {holds.max():.2f}")

    reasons = trades_df["exit_reason"].value_counts()
    print("Exit reasons:     " + ", ".join([f"{k}:{v}" for k, v in reasons.items()]))


def run():
    print(f"Downloading data ({YF_INTERVAL}, {YF_PERIOD})...")
    symbols = sorted({p.a for p in PAIRS} | {p.b for p in PAIRS})

    raw = {}
    for s in symbols:
        df = yf.download(
            s,
            interval=YF_INTERVAL,
            period=YF_PERIOD,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        raw[s] = clean_yf_intraday(df)
        if raw[s].empty:
            print(f"WARNING: {s} returned empty data.")

    all_trades = []
    for pc in PAIRS:
        eq, tdf = backtest_pair(pc, raw[pc.a], raw[pc.b])
        summarize(pc, eq, tdf)
        if not tdf.empty:
            all_trades.append(tdf)

    if all_trades:
        out = pd.concat(all_trades, ignore_index=True)
        out.to_csv("statarb_trade_log.csv", index=False)
        print("\nWrote trade log: statarb_trade_log.csv")
    else:
        print("\nNo trades to log.")


if __name__ == "__main__":
    run()
