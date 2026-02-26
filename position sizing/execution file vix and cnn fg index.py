"""
SPX 0DTE Regime Execution + Position Sizing Bot (PAPER)
- Regime inputs: CNN Fear & Greed + VIX
- Bias: leans bearish when either deteriorates
- Trades: 0DTE defined-risk vertical debit spreads (puts/calls) using Black–Scholes
- Data: yfinance for SPX + VIX
- CNN FG: strict JSON fetch (NO GUESSING)
"""

from __future__ import annotations

import math
import time
import json
import datetime as dt
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import requests

try:
    from math import erf
except ImportError:
    erf = None


# =============================
# Exceptions
# =============================

class FearGreedUnavailable(Exception):
    pass


# =============================
# Config
# =============================

@dataclass
class BotConfig:
    tz: str = "US/Eastern"

    entry_after_et: dt.time = dt.time(10, 0)
    last_entry_et: dt.time = dt.time(14, 30)
    hard_exit_et: dt.time = dt.time(15, 30)

    account_equity: float = 100_000.0
    base_risk_frac: float = 0.005
    max_contracts: int = 50
    min_risk_frac_to_trade: float = 0.0008

    vix_low: float = 18.0
    vix_high: float = 25.0
    fg_fear: int = 25
    fg_greed: int = 75

    rf_rate: float = 0.00

    spread_width_points: int = 25
    long_leg_offset_points: int = 0

    take_profit_frac_of_max: float = 0.50
    stop_loss_frac_of_max: float = 0.70

    USE_CNN_SCRAPE: bool = True
    cnn_timeout_sec: int = 8

    poll_interval_sec: int = 60


# =============================
# Time utilities
# =============================

def now_et(cfg: BotConfig) -> dt.datetime:
    ts = pd.Timestamp.utcnow()
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert(cfg.tz).to_pydatetime()

def is_weekday(d: dt.date) -> bool:
    return d.weekday() < 5

def within_time_window(t: dt.time, start: dt.time, end: dt.time) -> bool:
    return start <= t <= end


# =============================
# CNN Fear & Greed (STRICT)
# =============================

def fetch_cnn_fear_greed(cfg: BotConfig) -> int:
    if not cfg.USE_CNN_SCRAPE:
        raise FearGreedUnavailable("FG disabled by config")

    url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"

    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Origin": "https://www.cnn.com",
        "Referer": "https://www.cnn.com/markets/fear-and-greed",
        "Connection": "keep-alive",
    })

    r = session.get(url, timeout=cfg.cnn_timeout_sec)
    r.raise_for_status()

    data = r.json()

    if "fear_and_greed" not in data:
        raise FearGreedUnavailable("FG object missing")

    fg = data["fear_and_greed"].get("score")
    if fg is None:
        raise FearGreedUnavailable("FG score missing")

    fg = int(round(float(fg)))

    if not (0 <= fg <= 100):
        raise FearGreedUnavailable(f"FG out of bounds: {fg}")

    return fg



# =============================
# Market data
# =============================

def fetch_spx_vix() -> Tuple[float, float]:
    spx = yf.Ticker("^SPX")
    vix = yf.Ticker("^VIX")
    return float(spx.fast_info["last_price"]), float(vix.fast_info["last_price"])


# =============================
# Regime logic
# =============================

def compute_direction_weights(fg: int, vix: float, cfg: BotConfig) -> Dict[str, float]:
    long_w = 1.0
    short_w = 1.0

    if fg < cfg.fg_fear:
        long_w *= 0.30
        short_w *= 1.50
    elif fg > cfg.fg_greed:
        long_w *= 1.30
        short_w *= 0.70

    if vix >= cfg.vix_high:
        long_w *= 0.40
        short_w *= 1.60
    elif vix >= cfg.vix_low:
        long_w *= 0.70
        short_w *= 1.20

    return {"long_weight": long_w, "short_weight": short_w}

def decide_bias(weights: Dict[str, float]) -> str:
    score = math.log(weights["short_weight"] + 1e-9) - math.log(weights["long_weight"] + 1e-9)
    if score > 0.35:
        return "BEAR"
    if score < -0.35:
        return "BULL"
    return "FLAT"


# =============================
# Black–Scholes
# =============================

def norm_cdf(x: float) -> float:
    if erf:
        return 0.5 * (1 + erf(x / math.sqrt(2)))
    return 1 / (1 + math.exp(-1.702 * x))

def bs_price(S, K, T, r, sigma, kind):
    if T <= 0:
        return max(0, S - K) if kind == "C" else max(0, K - S)

    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if kind == "C":
        return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    return K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)

def vix_to_sigma(vix: float) -> float:
    return max(0.01, vix / 100.0)

def minutes_to_close_et(now):
    close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return max(0, int((close - now).total_seconds() // 60))


# =============================
# Trade structures
# =============================

@dataclass
class SpreadTrade:
    direction: str
    entry_time: dt.datetime
    long_strike: float
    short_strike: float
    entry_debit: float
    contracts: int
    max_loss: float
    max_profit: float
    is_open: bool = True


def round_to_5(x): return 5 * round(x / 5)

def build_spread(S, direction, cfg):
    if direction == "BEAR":
        longK = round_to_5(S)
        return longK, longK - cfg.spread_width_points
    longK = round_to_5(S)
    return longK, longK + cfg.spread_width_points


# =============================
# Bot loop
# =============================

def run_paper_bot(cfg: BotConfig):
    open_trade = None
    print("=== SPX 0DTE REGIME BOT (STRICT FG) ===\n")

    while True:
        now = now_et(cfg)

        try:
            spx, vix = fetch_spx_vix()
            fg = fetch_cnn_fear_greed(cfg)
        except FearGreedUnavailable as e:
            print(f"[{now:%H:%M}] ⚠️ FG ERROR: {e} — SKIPPING")
            time.sleep(cfg.poll_interval_sec)
            continue
        except Exception as e:
            print(f"[{now:%H:%M}] DATA ERROR: {e}")
            time.sleep(cfg.poll_interval_sec)
            continue

        weights = compute_direction_weights(fg, vix, cfg)
        bias = decide_bias(weights)

        print(f"[{now:%H:%M}] SPX={spx:.2f} VIX={vix:.2f} FG={fg} bias={bias}")

        time.sleep(cfg.poll_interval_sec)


# =============================
# Main
# =============================

if __name__ == "__main__":
    cfg = BotConfig()
    run_paper_bot(cfg)
