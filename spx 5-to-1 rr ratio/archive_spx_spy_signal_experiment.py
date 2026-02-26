import yfinance as yf
import numpy as np
import pandas as pd

# ==========================
# CONFIG
# ==========================
TICKER_SPX = "^GSPC"     # signal source
TICKER_SPY = "SPY"       # execution asset

PERIOD = "60d"
INTERVAL = "5m"

LEVEL_STEP = 50          # 6400, 6450, ...
STOP_OFFSET = 2.0        # 2 SPX points
RR = 5.0                 # 5:1 reward:risk

RETEST_TOLERANCE = 0.5   # how close to level counts as retest
MAX_HOLD_MIN = 30        # max time in trade (minutes); None to disable


# ==========================
# DATA LOADING
# ==========================
def load_data():
    print("Downloading SPX (^GSPC)...")
    spx = yf.download(
        TICKER_SPX,
        period=PERIOD,
        interval=INTERVAL,
        auto_adjust=True,
    )

    if spx.empty:
        raise ValueError("No SPX data downloaded.")

    print("Downloading SPY...")
    spy = yf.download(
        TICKER_SPY,
        period=PERIOD,
        interval=INTERVAL,
        auto_adjust=True,
    )

    if spy.empty:
        raise ValueError("No SPY data downloaded.")

    # Flatten MultiIndex columns if needed
    def flatten_cols(df):
        if isinstance(df.columns, pd.MultiIndex):
            new_cols = []
            for col in df.columns:
                parts = [str(x) for x in col if str(x) != ""]
                new_cols.append(parts[0] if parts else str(col))
            df.columns = new_cols
        return df

    spx = flatten_cols(spx)
    spy = flatten_cols(spy)

    spx = spx[["Open", "High", "Low", "Close"]].dropna()
    spy = spy[["Open", "High", "Low", "Close"]].dropna()

    spx = spx.astype(float)
    spy = spy.astype(float)

    spx.reset_index(inplace=True)
    spy.reset_index(inplace=True)

    spx.rename(columns={spx.columns[0]: "Datetime"}, inplace=True)
    spy.rename(columns={spy.columns[0]: "Datetime"}, inplace=True)

    spx["Datetime"] = pd.to_datetime(spx["Datetime"])
    spy["Datetime"] = pd.to_datetime(spy["Datetime"])

    # Merge SPX + SPY on timestamp
    df = pd.merge(
        spx,
        spy,
        on="Datetime",
        how="inner",
        suffixes=("_SPX", "_SPY"),
    )

    df["date"] = df["Datetime"].dt.date

    print(f"Bars after merge: {len(df)}")
    print(f"Start: {df['Datetime'].iloc[0]}, End: {df['Datetime'].iloc[-1]}")

    # Build SPX levels
    price_min = float(df["Low_SPX"].min())
    price_max = float(df["High_SPX"].max())
    level_start = np.floor(price_min / LEVEL_STEP) * LEVEL_STEP
    level_end = np.ceil(price_max / LEVEL_STEP) * LEVEL_STEP
    levels = np.arange(level_start, level_end + LEVEL_STEP, LEVEL_STEP).astype(float)
    print("SPX Levels:", levels)

    return df, levels


# ==========================
# CONTINUATION ENTRY LOGIC (SPX-based)
# ==========================
def check_entry_signal_continuation(i, row, prev_close_spx, levels, level_state):
    """
    Continuation (breakout + retest) on SPX:

    Long:
      - prev_close < L and close > L and high >= L  (breakout)
      - later bar retests near L from above -> long

    Short:
      - prev_close > L and close < L and low <= L   (breakdown)
      - later bar retests near L from below -> short
    """
    row_low = float(row["Low_SPX"])
    row_high = float(row["High_SPX"])
    row_close = float(row["Close_SPX"])
    prev_close_spx = float(prev_close_spx)

    for L in levels:
        L = float(L)
        state = level_state[L]
        mode = state["mode"]

        if mode == "idle":
            # Bullish breakout
            if (prev_close_spx < L) and (row_close > L) and (row_high >= L):
                state["mode"] = "waiting_retest"
                state["direction"] = "long"

            # Bearish breakdown
            elif (prev_close_spx > L) and (row_close < L) and (row_low <= L):
                state["mode"] = "waiting_retest"
                state["direction"] = "short"

        elif mode == "waiting_retest":
            direction = state["direction"]

            if direction == "long":
                # Retest from above
                if row_low <= L + RETEST_TOLERANCE:
                    entry = L
                    stop = L - STOP_OFFSET
                    state["mode"] = "idle"
                    state["direction"] = None
                    return {
                        "index": i,
                        "direction": "long",
                        "level": L,
                        "entry_spx": entry,
                        "stop_spx": stop,
                    }

            elif direction == "short":
                # Retest from below
                if row_high >= L - RETEST_TOLERANCE:
                    entry = L
                    stop = L + STOP_OFFSET
                    state["mode"] = "idle"
                    state["direction"] = None
                    return {
                        "index": i,
                        "direction": "short",
                        "level": L,
                        "entry_spx": entry,
                        "stop_spx": stop,
                    }

    return None


# ==========================
# SIGNAL GENERATION (SPX → SPY)
# ==========================
def generate_signals(df, levels):
    """
    Uses SPX continuation logic to generate buy/sell signals for SPY.

    Returns a DataFrame of signal events:
      - Datetime
      - action: "open" / "close"
      - side: "long" / "short"
      - reason: "target", "stop", "time_exit", "overnight_flat"
      - entry/exit SPX level
      - entry/exit SPY price
      - duration (for closes)
    """
    # State for each SPX level
    level_state = {float(L): {"mode": "idle", "direction": None} for L in levels}

    signals = []
    open_pos = None  # current open position

    print("\nRunning SPX→SPY signal generation (continuation 5:1, 30-min max hold)...")

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i - 1]

        prev_close_spx = prev_row["Close_SPX"]
        time = row["Datetime"]
        prev_time = prev_row["Datetime"]

        # ---------- Handle overnight flat ----------
        if open_pos is not None and row["date"] != prev_row["date"]:
            # Close at previous bar SPY close
            exit_price_spy = float(prev_row["Close_SPY"])
            exit_time = prev_time
            duration = exit_time - open_pos["entry_time"]

            signals.append({
                "Datetime": exit_time,
                "action": "close",
                "side": open_pos["side"],
                "reason": "overnight_flat",
                "entry_spx": open_pos["entry_spx"],
                "stop_spx": open_pos["stop_spx"],
                "target_spx": open_pos["target_spx"],
                "entry_spy": open_pos["entry_spy"],
                "exit_spy": exit_price_spy,
                "pnl_per_share": (exit_price_spy - open_pos["entry_spy"]) if open_pos["side"] == "long"
                                 else (open_pos["entry_spy"] - exit_price_spy),
                "duration": duration,
            })
            open_pos = None

        # ---------- Manage open position (SPX-based exits) ----------
        if open_pos is not None:
            # Check SPX stop/target on this bar
            high_spx = float(row["High_SPX"])
            low_spx = float(row["Low_SPX"])
            side = open_pos["side"]
            exit_reason = None

            if side == "long":
                # Stop first, then target
                if low_spx <= open_pos["stop_spx"]:
                    exit_reason = "stop"
                elif high_spx >= open_pos["target_spx"]:
                    exit_reason = "target"
            else:  # short
                if high_spx >= open_pos["stop_spx"]:
                    exit_reason = "stop"
                elif low_spx <= open_pos["target_spx"]:
                    exit_reason = "target"

            # Check max hold time
            if exit_reason is None and MAX_HOLD_MIN is not None:
                elapsed_min = (time - open_pos["entry_time"]) / pd.Timedelta(minutes=1)
                if elapsed_min >= MAX_HOLD_MIN:
                    exit_reason = "time_exit"

            if exit_reason is not None:
                exit_price_spy = float(row["Close_SPY"])
                exit_time = time
                duration = exit_time - open_pos["entry_time"]

                signals.append({
                    "Datetime": exit_time,
                    "action": "close",
                    "side": side,
                    "reason": exit_reason,
                    "entry_spx": open_pos["entry_spx"],
                    "stop_spx": open_pos["stop_spx"],
                    "target_spx": open_pos["target_spx"],
                    "entry_spy": open_pos["entry_spy"],
                    "exit_spy": exit_price_spy,
                    "pnl_per_share": (exit_price_spy - open_pos["entry_spy"]) if side == "long"
                                     else (open_pos["entry_spy"] - exit_price_spy),
                    "duration": duration,
                })

                open_pos = None

        # ---------- Look for new entry if flat ----------
        if open_pos is None:
            signal = check_entry_signal_continuation(i, row, prev_close_spx, levels, level_state)
            if signal is None:
                continue

            direction = signal["direction"]
            entry_spx = signal["entry_spx"]
            stop_spx = signal["stop_spx"]

            if direction == "long":
                target_spx = entry_spx + (entry_spx - stop_spx) * RR
                side = "long"
            else:
                target_spx = entry_spx - (stop_spx - entry_spx) * RR
                side = "short"

            entry_spy = float(row["Close_SPY"])

            # Open position (SPY)
            open_pos = {
                "side": side,
                "entry_time": time,
                "entry_spx": entry_spx,
                "stop_spx": stop_spx,
                "target_spx": target_spx,
                "entry_spy": entry_spy,
            }

            signals.append({
                "Datetime": time,
                "action": "open",
                "side": side,
                "reason": "entry",
                "entry_spx": entry_spx,
                "stop_spx": stop_spx,
                "target_spx": target_spx,
                "entry_spy": entry_spy,
                "exit_spy": None,
                "pnl_per_share": None,
                "duration": pd.NaT,
            })

    # If still open at last bar, close it
    if open_pos is not None:
        last_row = df.iloc[-1]
        exit_price_spy = float(last_row["Close_SPY"])
        exit_time = last_row["Datetime"]
        duration = exit_time - open_pos["entry_time"]

        signals.append({
            "Datetime": exit_time,
            "action": "close",
            "side": open_pos["side"],
            "reason": "forced_exit",
            "entry_spx": open_pos["entry_spx"],
            "stop_spx": open_pos["stop_spx"],
            "target_spx": open_pos["target_spx"],
            "entry_spy": open_pos["entry_spy"],
            "exit_spy": exit_price_spy,
            "pnl_per_share": (exit_price_spy - open_pos["entry_spy"]) if open_pos["side"] == "long"
                             else (open_pos["entry_spy"] - exit_price_spy),
            "duration": duration,
        })

    signals_df = pd.DataFrame(signals).sort_values("Datetime").reset_index(drop=True)
    return signals_df


# ==========================
# SUMMARY
# ==========================
def summarize_signals(signals_df):
    print("\n===== SIGNAL SUMMARY (SPX→SPY continuation 5:1, 30-min max hold) =====")

    opens = signals_df[signals_df["action"] == "open"]
    closes = signals_df[signals_df["action"] == "close"]

    print(f"Total opens:  {len(opens)}")
    print(f"Total closes: {len(closes)}")

    # Basic PnL stats (per share)
    if not closes.empty:
        pnl = closes["pnl_per_share"]
        wins = (closes["reason"] == "target")
        stops = (closes["reason"] == "stop")

        print(f"\nPnL per share (SPY):")
        print(f"Average: {pnl.mean():.3f}")
        print(f"Median:  {pnl.median():.3f}")
        print(f"Win rate (target hits): {(wins.sum() / len(closes)) * 100:.2f}%")
        print(f"Stop-outs: {stops.sum()} of {len(closes)}")

        # Duration stats
        durations = closes["duration"].dropna()
        if not durations.empty:
            to_min = lambda td: td / pd.Timedelta(minutes=1)
            print("\nTime in trade (closes only):")
            print(f"Average: {to_min(durations.mean()):.2f} minutes")
            print(f"Median:  {to_min(durations.median()):.2f} minutes")
            print(f"Min:     {to_min(durations.min()):.2f} minutes")
            print(f"Max:     {to_min(durations.max()):.2f} minutes")

        print("\nOutcome breakdown:")
        print(closes["reason"].value_counts())


# ==========================
# MAIN
# ==========================
def main():
    df, levels = load_data()
    signals_df = generate_signals(df, levels)
    summarize_signals(signals_df)

    out_file = "spx_to_spy_signals_5R_30min.csv"
    signals_df.to_csv(out_file, index=False)
    print(f"\nSaved signals to {out_file}")


if __name__ == "__main__":
    main()
