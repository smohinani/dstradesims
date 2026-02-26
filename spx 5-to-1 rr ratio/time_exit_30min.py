import yfinance as yf
import numpy as np
import pandas as pd

# ==========================
# CONFIG
# ==========================
# Stick to SPX index
TICKER = "^GSPC"        # S&P 500 index
PERIOD = "60d"
INTERVAL = "5m"

LEVEL_STEP = 50         # 6500, 6550, 6600, ...
STOP_OFFSET = 2.0       # 2 points
RR = 5.0                # 5:1 fixed

RETEST_TOLERANCE = 0.5
RISK_PER_TRADE = 0.01   # 1% of equity per trade
INITIAL_CAPITAL = 10000.0

MAX_HOLD_MINUTES = 30   # max time in trade
MAX_HOLD_DELTA = pd.Timedelta(minutes=MAX_HOLD_MINUTES)


# ==========================
# DATA
# ==========================
def load_data():
    print(f"Downloading {TICKER}...")
    df = yf.download(
        TICKER,
        period=PERIOD,
        interval=INTERVAL,
        auto_adjust=True,
    )

    if df.empty:
        raise ValueError("No data downloaded.")

    # Flatten MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        new_cols = []
        for col in df.columns:
            parts = [str(x) for x in col if str(x) != ""]
            new_cols.append(parts[0] if parts else str(col))
        df.columns = new_cols

    df = df[["Open", "High", "Low", "Close"]].dropna()
    df = df.astype(float)
    df.reset_index(inplace=True)
    df.rename(columns={df.columns[0]: "Datetime"}, inplace=True)
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df["date"] = df["Datetime"].dt.date

    print(f"Bars: {len(df)}")
    print(f"Start: {df['Datetime'].iloc[0]}, End: {df['Datetime'].iloc[-1]}")

    price_min = float(df["Low"].min())
    price_max = float(df["High"].max())
    level_start = np.floor(price_min / LEVEL_STEP) * LEVEL_STEP
    level_end = np.ceil(price_max / LEVEL_STEP) * LEVEL_STEP
    levels = np.arange(level_start, level_end + LEVEL_STEP, LEVEL_STEP).astype(float)
    print("Levels:", levels)

    return df, levels


# ==========================
# CONTINUATION ENTRY LOGIC
# ==========================
def check_entry_signal_continuation(i, row, prev_close, levels, level_state):
    """
    Continuation (breakout + retest)
    """
    row_low = float(row["Low"])
    row_high = float(row["High"])
    row_close = float(row["Close"])
    prev_close = float(prev_close)

    for L in levels:
        L = float(L)
        state = level_state[L]
        mode = state["mode"]

        if mode == "idle":
            # Bullish breakout
            if (prev_close < L) and (row_close > L) and (row_high >= L):
                state["mode"] = "waiting_retest"
                state["direction"] = "long"
            # Bearish breakdown
            elif (prev_close > L) and (row_close < L) and (row_low <= L):
                state["mode"] = "waiting_retest"
                state["direction"] = "short"

        elif mode == "waiting_retest":
            direction = state["direction"]

            if direction == "long":
                if row_low <= L + RETEST_TOLERANCE:
                    entry = L
                    stop = L - STOP_OFFSET
                    state["mode"] = "idle"
                    state["direction"] = None
                    return {
                        "index": i,
                        "direction": "long",
                        "level": L,
                        "entry": entry,
                        "stop": stop,
                    }
            elif direction == "short":
                if row_high >= L - RETEST_TOLERANCE:
                    entry = L
                    stop = L + STOP_OFFSET
                    state["mode"] = "idle"
                    state["direction"] = None
                    return {
                        "index": i,
                        "direction": "short",
                        "level": L,
                        "entry": entry,
                        "stop": stop,
                    }

    return None


# ==========================
# TRADE MANAGEMENT
# ==========================
def manage_open_trade(i, row, trade, capital):
    """
    Manage open trade:
    - If stop or target hit, close with that outcome.
    (Max-hold timer is checked outside and may close it earlier at MOC.)
    """
    entry = float(trade["entry"])
    stop = float(trade["stop"])
    target = float(trade["target"])
    direction = trade["direction"]

    low = float(row["Low"])
    high = float(row["High"])

    exit_price = None
    outcome = None

    if direction == "long":
        if low <= stop:
            exit_price = stop
            outcome = "stop"
        elif high >= target:
            exit_price = target
            outcome = "target"
        if exit_price is not None:
            pnl = (exit_price - entry) * trade["units"]
    else:  # short
        if high >= stop:
            exit_price = stop
            outcome = "stop"
        elif low <= target:
            exit_price = target
            outcome = "target"
        if exit_price is not None:
            pnl = (entry - exit_price) * trade["units"]

    if exit_price is not None:
        capital += pnl
        closed_trade = {
            **trade,
            "exit_index": i,
            "exit_price": exit_price,
            "pnl": pnl,
            "outcome": outcome,
            "capital_after": capital,
        }
        return None, capital, closed_trade

    return trade, capital, None


# ==========================
# BACKTEST
# ==========================
def run_backtest(df, levels):
    level_state = {float(L): {"mode": "idle", "direction": None} for L in levels}
    capital = INITIAL_CAPITAL
    open_trade = None
    trades = []

    print("\nRunning continuation 5:1 backtest with 30-minute max hold...")

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i - 1]
        prev_close = prev_row["Close"]

        current_time = row["Datetime"]
        prev_time = prev_row["Datetime"]

        # Overnight flat
        if open_trade is not None and row["date"] != prev_row["date"]:
            prev_close_val = float(prev_close)
            if open_trade["direction"] == "long":
                pnl = (prev_close_val - float(open_trade["entry"])) * open_trade["units"]
            else:
                pnl = (float(open_trade["entry"]) - prev_close_val) * open_trade["units"]
            capital += pnl
            trades.append({
                **open_trade,
                "exit_index": i - 1,
                "exit_price": prev_close_val,
                "pnl": pnl,
                "outcome": "overnight_flat",
                "capital_after": capital,
            })
            open_trade = None

        # If still open, check stop/target first
        if open_trade is not None:
            open_trade, capital, closed = manage_open_trade(i, row, open_trade, capital)
            if closed is not None:
                trades.append(closed)

        # If still open, check max hold time (30 min)
        if open_trade is not None:
            entry_time = open_trade["entry_time"]
            if current_time - entry_time >= MAX_HOLD_DELTA:
                # Exit at current close as "time_exit"
                exit_price = float(row["Close"])
                if open_trade["direction"] == "long":
                    pnl = (exit_price - float(open_trade["entry"])) * open_trade["units"]
                else:
                    pnl = (float(open_trade["entry"]) - exit_price) * open_trade["units"]
                capital += pnl
                trades.append({
                    **open_trade,
                    "exit_index": i,
                    "exit_price": exit_price,
                    "pnl": pnl,
                    "outcome": "time_exit",
                    "capital_after": capital,
                })
                open_trade = None

        # If flat, look for new entry
        if open_trade is None:
            signal = check_entry_signal_continuation(i, row, prev_close, levels, level_state)
            if signal is None:
                continue

            entry = signal["entry"]
            stop = signal["stop"]
            direction = signal["direction"]

            if direction == "long":
                target = entry + (entry - stop) * RR
            else:
                target = entry - (stop - entry) * RR

            risk_per_unit = abs(entry - stop)
            if risk_per_unit <= 0:
                continue

            risk_amount = capital * RISK_PER_TRADE
            units = risk_amount / risk_per_unit

            open_trade = {
                **signal,
                "target": target,
                "units": units,
                "risk_amount": risk_amount,
                "entry_time": current_time,  # store for duration
            }

    # Forced exit at final bar if still open
    if open_trade is not None:
        last_row = df.iloc[-1]
        last_close = float(last_row["Close"])
        if open_trade["direction"] == "long":
            pnl = (last_close - open_trade["entry"]) * open_trade["units"]
        else:
            pnl = (open_trade["entry"] - last_close) * open_trade["units"]
        capital += pnl
        trades.append({
            **open_trade,
            "exit_index": len(df) - 1,
            "exit_price": last_close,
            "pnl": pnl,
            "outcome": "forced_exit",
            "capital_after": capital,
        })

    trades_df = pd.DataFrame(trades)

    # Map entry/exit indices to timestamps for duration calc
    if not trades_df.empty:
        trades_df["entry_time"] = trades_df["index"].apply(
            lambda idx: df["Datetime"].iloc[int(idx)]
        )
        trades_df["exit_time"] = trades_df["exit_index"].apply(
            lambda idx: df["Datetime"].iloc[int(idx)]
        )
        trades_df["duration"] = trades_df["exit_time"] - trades_df["entry_time"]

    return capital, trades_df


# ==========================
# STATS (incl. time)
# ==========================
def summarize(capital, trades_df):
    print("\n===== RESULTS (Continuation 5:1, 30-min max hold) =====")
    print(f"TICKER:           {TICKER}")
    print(f"Initial capital:  {INITIAL_CAPITAL:,.2f}")
    print(f"Final capital:    {capital:,.2f}")
    print(f"Net return:       {(capital / INITIAL_CAPITAL - 1) * 100:.2f}%")

    if trades_df.empty:
        print("No trades taken.")
        return

    total = len(trades_df)
    wins_df = trades_df[trades_df["outcome"] == "target"]
    losses_df = trades_df[trades_df["outcome"] == "stop"]

    wins = len(wins_df)
    losses = len(losses_df)
    win_rate = wins / total * 100 if total > 0 else 0.0

    long_trades = trades_df[trades_df["direction"] == "long"]
    short_trades = trades_df[trades_df["direction"] == "short"]
    long_pct = len(long_trades) / total * 100
    short_pct = len(short_trades) / total * 100

    print(f"Trades: {total}")
    print(f"Wins:   {wins}, Losses: {losses}")
    print(f"Win rate: {win_rate:.2f}%")
    print(f"% Longs: {long_pct:.2f}%, % Shorts: {short_pct:.2f}%")

    if not wins_df.empty:
        avg_win = wins_df["pnl"].mean()
        avg_win_R = (wins_df["pnl"] / wins_df["risk_amount"]).mean()
        print(f"\nAverage win:  {avg_win:.2f} USD ({avg_win_R:.2f} R)")
    if not losses_df.empty:
        avg_loss = losses_df["pnl"].mean()
        avg_loss_R = (losses_df["pnl"] / losses_df["risk_amount"]).mean()
        print(f"Average loss: {avg_loss:.2f} USD ({avg_loss_R:.2f} R)")

    # Time stats
    durations = trades_df["duration"]
    avg_dur = durations.mean()
    med_dur = durations.median()
    min_dur = durations.min()
    max_dur = durations.max()

    to_min = lambda td: td / pd.Timedelta(minutes=1)

    print("\n--- Time in Trade ---")
    print(f"Average duration: {to_min(avg_dur):.2f} minutes")
    print(f"Median duration:  {to_min(med_dur):.2f} minutes")
    print(f"Min duration:     {to_min(min_dur):.2f} minutes")
    print(f"Max duration:     {to_min(max_dur):.2f} minutes")

    # Outcome breakdown (including time_exit)
    print("\n--- Outcome breakdown ---")
    print(trades_df["outcome"].value_counts())

    out_name = f"gspc_cont_5R_30min_trades.csv"
    trades_df.to_csv(out_name, index=False)
    print(f"\nSaved trades to {out_name}")


# ==========================
# MAIN
# ==========================
def main():
    df, levels = load_data()
    capital, trades_df = run_backtest(df, levels)
    summarize(capital, trades_df)


if __name__ == "__main__":
    main()
