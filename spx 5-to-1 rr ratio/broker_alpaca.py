from dataclasses import dataclass
from typing import Optional, Dict, Any
import os
import time
import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# ---- Alpaca (alpaca-py) ----
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.requests import StockBarsRequest
from alpaca.data.enums import DataFeed


# ==========================
# CONFIG
# ==========================
SYMBOL = "SPY"                 # we actually trade SPY
DATA_TIMEFRAME_MIN = 5         # 5-minute bars

# "SPX-like" logic (SPX ≈ SPY * 10)
LEVEL_STEP_SPX = 50.0          # 50 SPX points  -> 5 USD in SPY
STOP_OFFSET_SPX = 2.0          # 2 SPX points   -> 0.2 USD in SPY
RR = 5.0                       # 5:1 reward:risk
RISK_PER_TRADE = 0.01          # 1% of equity per trade
MAX_HOLD_MIN = 30              # max hold time in minutes
POLL_INTERVAL_SEC = 10         # how often to poll Alpaca (seconds)
INITIAL_EQUITY_FALLBACK = 10000.0


# ==========================
# STRATEGY STATE
# ==========================
@dataclass
class OpenTrade:
    direction: str     # "long" or "short"
    qty: int
    entry_spy: float
    stop_spy: float
    target_spy: float
    entry_time: datetime


class StrategyState:
    def __init__(self) -> None:
        self.levels_spx: Optional[np.ndarray] = None      # array of SPX levels
        self.level_state: Dict[float, Dict[str, Any]] = {}  # per-level FSM
        self.open_trade: Optional[OpenTrade] = None
        self.last_bar_time: Optional[datetime] = None     # last processed bar


state = StrategyState()


# ==========================
# HELPERS
# ==========================
def synthetic_spx_from_spy(spy_price: float) -> float:
    """Approximate SPX as 10x SPY."""
    return spy_price * 10.0


def spx_to_spy_price(spx_value: float) -> float:
    """Approximate SPY as SPX / 10."""
    return spx_value / 10.0


def init_clients():
    """
    Load keys from .env and create Trading + Data clients.
    .env must contain:
      ALPACA_API_KEY=...
      ALPACA_API_SECRET=...
    """
    load_dotenv()
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_API_SECRET")

    if not api_key or not secret_key:
        raise ValueError(
            "Missing ALPACA_API_KEY or ALPACA_API_SECRET in environment. "
            "Check your .env and that you're running this from the same folder."
        )

    trading_client = TradingClient(api_key, secret_key, paper=True)
    data_client = StockHistoricalDataClient(api_key, secret_key)
    return trading_client, data_client


def init_levels_from_history(data_client: StockHistoricalDataClient,
                             state: StrategyState,
                             days: int = 10) -> None:
    """
    Pull last `days` of 5m SPY bars (IEX feed) and build SPX-like levels.
    """
    print(f"Initializing levels from last {days} days of SPY data...")

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    req = StockBarsRequest(
        symbol_or_symbols=SYMBOL,
        timeframe=TimeFrame(DATA_TIMEFRAME_MIN, TimeFrameUnit.Minute),
        start=start,
        end=end,
        feed=DataFeed.IEX,   # FREE PLAN: use IEX feed
    )

    bars = data_client.get_stock_bars(req)
    df = bars.df

    if df.empty:
        raise RuntimeError("No historical SPY data returned when initializing levels.")

    # For a single symbol, df has MultiIndex (symbol, time)
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(SYMBOL, level="symbol")

    closes = df["close"].astype(float)
    spx_series = closes * 10.0

    price_min = float(spx_series.min())
    price_max = float(spx_series.max())

    level_start = math.floor(price_min / LEVEL_STEP_SPX) * LEVEL_STEP_SPX
    level_end = math.ceil(price_max / LEVEL_STEP_SPX) * LEVEL_STEP_SPX
    levels = np.arange(level_start, level_end + LEVEL_STEP_SPX, LEVEL_STEP_SPX).astype(float)

    state.levels_spx = levels
    state.level_state = {float(L): {"mode": "idle", "direction": None} for L in levels}

    print("Levels initialized (SPX space):", levels)


# ==========================
# ENTRY LOGIC (CONTINUATION)
# ==========================
def check_entry_signal_continuation_spx(prev_spx_close: float,
                                       spx_row_low: float,
                                       spx_row_high: float,
                                       spx_row_close: float,
                                       state: StrategyState):
    """
    Continuation breakout + retest in SPX space.
    Returns dict with direction / entry_spx / stop_spx, or None.
    """
    RETEST_TOLERANCE_SPX = 0.5  # how close the retest needs to be

    for L in state.levels_spx:
        L = float(L)
        info = state.level_state[L]
        mode = info["mode"]

        if mode == "idle":
            # Bullish breakout
            if (prev_spx_close < L) and (spx_row_close > L) and (spx_row_high >= L):
                info["mode"] = "waiting_retest"
                info["direction"] = "long"

            # Bearish breakdown
            elif (prev_spx_close > L) and (spx_row_close < L) and (spx_row_low <= L):
                info["mode"] = "waiting_retest"
                info["direction"] = "short"

        elif mode == "waiting_retest":
            direction = info["direction"]

            if direction == "long":
                if spx_row_low <= L + RETEST_TOLERANCE_SPX:
                    entry = L
                    stop = L - STOP_OFFSET_SPX
                    info["mode"] = "idle"
                    info["direction"] = None
                    return {
                        "direction": "long",
                        "entry_spx": entry,
                        "stop_spx": stop,
                    }

            elif direction == "short":
                if spx_row_high >= L - RETEST_TOLERANCE_SPX:
                    entry = L
                    stop = L + STOP_OFFSET_SPX
                    info["mode"] = "idle"
                    info["direction"] = None
                    return {
                        "direction": "short",
                        "entry_spx": entry,
                        "stop_spx": stop,
                    }

    return None


# ==========================
# LIVE DATA (LATEST 5m BARS)
# ==========================
def get_latest_5m_bars(data_client: StockHistoricalDataClient):
    """
    Fetch a small window of recent 5m SPY bars (IEX feed) and return (prev_bar, last_bar).
    """
    end = datetime.now(timezone.utc)
    start = end - timedelta(minutes=15)

    req = StockBarsRequest(
        symbol_or_symbols=SYMBOL,
        timeframe=TimeFrame(DATA_TIMEFRAME_MIN, TimeFrameUnit.Minute),
        start=start,
        end=end,
        feed=DataFeed.IEX,
    )

    bars = data_client.get_stock_bars(req)

    # bars is dict-like: symbol -> list[Bar]
    try:
        bar_list = list(bars[SYMBOL])
    except Exception:
        df = bars.df
        if df.empty:
            return None, None
        if isinstance(df.index, pd.MultiIndex):
            df_sym = df.xs(SYMBOL, level="symbol")
        else:
            df_sym = df
        df_sym = df_sym.sort_index()
        if len(df_sym) < 2:
            return None, None
        return None, None  # fallback not implemented cleanly here

    if len(bar_list) < 2:
        return None, None

    return bar_list[-2], bar_list[-1]


# ==========================
# TRADE OPEN / MANAGE
# ==========================
def open_new_trade_if_signal(prev_bar, last_bar,
                             trading_client: TradingClient,
                             state: StrategyState):
    """
    From prev+last bar, generate signal in SPX-space and open SPY trade.
    """
    if state.levels_spx is None:
        raise RuntimeError("Levels not initialized. Call init_levels_from_history first.")

    prev_spx_close = synthetic_spx_from_spy(prev_bar.close)
    spx_row_low = synthetic_spx_from_spy(last_bar.low)
    spx_row_high = synthetic_spx_from_spy(last_bar.high)
    spx_row_close = synthetic_spx_from_spy(last_bar.close)

    signal = check_entry_signal_continuation_spx(
        prev_spx_close,
        spx_row_low,
        spx_row_high,
        spx_row_close,
        state,
    )
    if signal is None:
        return

    direction = signal["direction"]
    entry_spx = signal["entry_spx"]
    stop_spx = signal["stop_spx"]

    entry_spy = spx_to_spy_price(entry_spx)
    stop_spy = spx_to_spy_price(stop_spx)

    if direction == "long":
        target_spx = entry_spx + (entry_spx - stop_spx) * RR
    else:
        target_spx = entry_spx - (stop_spx - entry_spx) * RR
    target_spy = spx_to_spy_price(target_spx)

    # Position sizing: risk 1% of equity per trade
    account = trading_client.get_account()
    try:
        equity = float(account.equity)
    except Exception:
        equity = INITIAL_EQUITY_FALLBACK

    risk_amount = equity * RISK_PER_TRADE
    risk_per_share = abs(entry_spy - stop_spy)
    if risk_per_share <= 0:
        return

    qty = int(risk_amount / risk_per_share)
    if qty <= 0:
        return

    print(
        f"[SIGNAL] {direction.upper()} SPY qty={qty} "
        f"entry~{entry_spy:.2f} stop~{stop_spy:.2f} target~{target_spy:.2f}"
    )

    order_side = OrderSide.BUY if direction == "long" else OrderSide.SELL
    order = MarketOrderRequest(
        symbol=SYMBOL,
        qty=qty,
        side=order_side,
        time_in_force=TimeInForce.DAY,
    )
    trading_client.submit_order(order)

    state.open_trade = OpenTrade(
        direction=direction,
        qty=qty,
        entry_spy=entry_spy,
        stop_spy=stop_spy,
        target_spy=target_spy,
        entry_time=last_bar.timestamp,
    )


def manage_open_trade(last_bar, trading_client: TradingClient, state: StrategyState):
    """
    Manage an open trade with stop, 5R target, and 30-min max hold.
    """
    trade = state.open_trade
    if trade is None:
        return

    now_price = last_bar.close
    t_now = last_bar.timestamp
    direction = trade.direction
    qty = trade.qty
    stop = trade.stop_spy
    tgt = trade.target_spy

    reason = None

    if direction == "long":
        if last_bar.low <= stop:
            reason = "stop"
        elif last_bar.high >= tgt:
            reason = "target"
    else:
        if last_bar.high >= stop:
            reason = "stop"
        elif last_bar.low <= tgt:
            reason = "target"

    dur_minutes = (t_now - trade.entry_time).total_seconds() / 60.0
    if reason is None and dur_minutes >= MAX_HOLD_MIN:
        reason = "time_exit"

    if reason is None:
        return

    exit_side = OrderSide.SELL if direction == "long" else OrderSide.BUY
    print(f"[EXIT] {reason} closing {direction.upper()} {qty} SPY at ~{now_price:.2f}")

    order = MarketOrderRequest(
        symbol=SYMBOL,
        qty=qty,
        side=exit_side,
        time_in_force=TimeInForce.DAY,
    )
    trading_client.submit_order(order)

    state.open_trade = None


# ==========================
# MAIN LOOP  🔁
# ==========================
def main_loop():
    trading_client, data_client = init_clients()

    # Build SPX levels from last 10 days of SPY data
    init_levels_from_history(data_client, state, days=10)

    print("Starting SPX→SPY continuation 5R bot (paper)...")

    while True:
        try:
            prev_bar, last_bar = get_latest_5m_bars(data_client)

            if prev_bar is None or last_bar is None:
                print("[LOOP] No bars returned (market may be closed). Sleeping...")
                time.sleep(POLL_INTERVAL_SEC)
                continue

            # Only act when we see a NEW completed bar
            if state.last_bar_time is not None and last_bar.timestamp <= state.last_bar_time:
                time.sleep(POLL_INTERVAL_SEC)
                continue

            state.last_bar_time = last_bar.timestamp

            ts_str = last_bar.timestamp.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M")
            print(
                f"\n[BAR] New 5m bar @ {ts_str}  "
                f"O:{last_bar.open:.2f} H:{last_bar.high:.2f} "
                f"L:{last_bar.low:.2f} C:{last_bar.close:.2f}"
            )

            # 1) Manage existing position
            if state.open_trade is not None:
                manage_open_trade(last_bar, trading_client, state)

            # 2) If flat, look for new entry
            if state.open_trade is None:
                open_new_trade_if_signal(prev_bar, last_bar, trading_client, state)

        except Exception as e:
            print("[ERROR] in main loop:", repr(e))
            time.sleep(POLL_INTERVAL_SEC)


if __name__ == "__main__":
    main_loop()
