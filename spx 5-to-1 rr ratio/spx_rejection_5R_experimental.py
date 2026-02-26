import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================
# CONFIG
# ==========================
TICKER = "^GSPC"            # S&P 500 index on Yahoo
PERIOD = "60d"              # 5m data only available for last ~60 days
INTERVAL = "5m"             # intraday resolution

LEVEL_STEP = 50             # 50-point levels: 6400, 6450, 6500, ...
STOP_OFFSET = 2.0           # 2 points beyond level
RR = 5.0                    # risk:reward = 1:5 → target 10 points away if risk is 2
RETEST_TOLERANCE = 0.5      # how close to the level counts as a retest
RISK_PER_TRADE = 0.01       # 1% of current equity per trade

INITIAL_CAPITAL = 10000.0


# ==========================
# DATA DOWNLOAD
# ==========================
print("Downloading data...")
df = yf.download(
    TICKER,
    period=PERIOD,
    interval=INTERVAL,
    auto_adjust=True,
)

if df.empty:
    raise ValueError("No data downloaded. Check ticker, period, or interval.")

# ---- Flatten MultiIndex columns if needed ----
if isinstance(df.columns, pd.MultiIndex):
    new_cols = []
    for col in df.columns:
        parts = [str(x) for x in col if str(x) != ""]
        new_cols.append(parts[0] if parts else str(col))
    df.columns = new_cols

# Keep only OHLC
df = df[["Open", "High", "Low", "Close"]].dropna()
df = df.astype(float)
df.reset_index(inplace=True)

time_col = df.columns[0]  # usually "Datetime"
print(f"Downloaded {len(df)} bars of data")
print(f"Data start: {df[time_col].iloc[0]}")
print(f"Data end:   {df[time_col].iloc[-1]}")


# ==========================
# LEVEL GENERATION
# ==========================
price_min = float(df["Low"].min())
price_max = float(df["High"].max())

level_start = np.floor(price_min / LEVEL_STEP) * LEVEL_STEP
level_end   = np.ceil(price_max / LEVEL_STEP) * LEVEL_STEP

levels = np.arange(level_start, level_end + LEVEL_STEP, LEVEL_STEP).astype(float)
print("Levels:", levels)


# ==========================
# STATE PER LEVEL (for bounce+retest)
# ==========================
level_state = {
    float(L): {"mode": "idle", "direction": None}
    for L in levels
}


# ==========================
# BACKTEST LOOP
# ==========================
capital = INITIAL_CAPITAL
open_trade = None
trades = []


def check_entry_signal(i, row, prev_close):
    """
    Try to find an entry based on bounce + retest.
    """
    global level_state

    row_low = float(row["Low"])
    row_high = float(row["High"])
    row_close = float(row["Close"])
    prev_close = float(prev_close)

    for L in levels:
        L = float(L)
        state = level_state[L]
        mode = state["mode"]

        touched = (row_low <= L) and (L <= row_high)

        # --- First touch → bounce setup ---
        if mode == "idle" and touched:
            # Coming down from above → support → long
            if prev_close > L and row_close > L:
                state["mode"] = "waiting_retest"
                state["direction"] = "long"
            # Coming up from below → resistance → short
            elif prev_close < L and row_close < L:
                state["mode"] = "waiting_retest"
                state["direction"] = "short"

        # --- Retest = actual entry ---
        elif mode == "waiting_retest":
            direction = state["direction"]

            if direction == "long":
                # Retest: price dips back near the level
                if row_low <= L + RETEST_TOLERANCE:
                    entry = L
                    stop = L - STOP_OFFSET
                    target = L + STOP_OFFSET * RR
                    state["mode"] = "idle"
                    state["direction"] = None
                    return {
                        "index": i,
                        "direction": "long",
                        "level": L,
                        "entry": entry,
                        "stop": stop,
                        "target": target,
                    }

            elif direction == "short":
                # Retest: price rallies back near the level
                if row_high >= L - RETEST_TOLERANCE:
                    entry = L
                    stop = L + STOP_OFFSET
                    target = L - STOP_OFFSET * RR
                    state["mode"] = "idle"
                    state["direction"] = None
                    return {
                        "index": i,
                        "direction": "short",
                        "level": L,
                        "entry": entry,
                        "stop": stop,
                        "target": target,
                    }

    return None


def manage_open_trade(i, row, trade, capital):
    """
    Check if stop or target hit.
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
        # Stop has priority if both hit same bar
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
        return None, capital, {
            **trade,
            "exit_index": i,
            "exit_price": exit_price,
            "pnl": pnl,
            "outcome": outcome,
            "capital_after": capital,
        }

    return trade, capital, None


print("\nRunning backtest...")
for i in range(1, len(df)):
    row = df.iloc[i]
    prev_row = df.iloc[i - 1]
    prev_close = prev_row["Close"]

    # Manage open trade first
    if open_trade is not None:
        open_trade, capital, closed = manage_open_trade(i, row, open_trade, capital)
        if closed is not None:
            trades.append(closed)

    # Look for new entry if flat
    if open_trade is None:
        signal = check_entry_signal(i, row, prev_close)
        if signal is not None:
            entry = signal["entry"]
            stop = signal["stop"]

            risk_per_unit = abs(entry - stop)
            if risk_per_unit <= 0:
                continue

            risk_amount = capital * RISK_PER_TRADE
            units = risk_amount / risk_per_unit

            open_trade = {
                **signal,
                "units": units,
                "risk_amount": risk_amount,
            }

print("Backtest complete.")

# Forced exit at last candle if still in a trade
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
    open_trade = None


# ==========================
# RESULTS + STATS
# ==========================
trades_df = pd.DataFrame(trades)

print("\n===== RESULTS =====")
print(f"Initial capital:  {INITIAL_CAPITAL:,.2f}")
print(f"Final capital:    {capital:,.2f}")
print(f"Net return:       {((capital / INITIAL_CAPITAL) - 1)*100:.2f}%")

if not trades_df.empty:
    wins_df = trades_df[trades_df["outcome"] == "target"]
    losses_df = trades_df[trades_df["outcome"] == "stop"]

    wins = len(wins_df)
    losses = len(losses_df)
    total = len(trades_df)

    print(f"Trades: {total}")
    print(f"Wins: {wins}, Losses: {losses}")
    if total > 0:
        print(f"Win rate: {wins / total * 100:.2f}%")

    # Profit / loss per trade
    if not wins_df.empty:
        avg_win = wins_df["pnl"].mean()
        avg_win_R = (wins_df["pnl"] / wins_df["risk_amount"]).mean()
        print(f"\nAverage profit per winning trade: {avg_win:.2f} USD ({avg_win_R:.2f} R)")
    if not losses_df.empty:
        avg_loss = losses_df["pnl"].mean()
        avg_loss_R = (losses_df["pnl"] / losses_df["risk_amount"]).mean()
        print(f"Average loss per losing trade:   {avg_loss:.2f} USD ({avg_loss_R:.2f} R)")

    # Save trades for later inspection
    trades_df.to_csv("spx_5to1_trades.csv", index=False)
    print("\nSaved trades to spx_5to1_trades.csv")
else:
    print("No trades were taken.")


# ==========================
# EQUITY CURVE
# ==========================
if not trades_df.empty:
    # Build equity curve over time based on exit_index & capital_after
    trades_sorted = trades_df.sort_values("exit_index").reset_index(drop=True)

    equity_curve = []
    equity = INITIAL_CAPITAL
    trade_idx = 0

    for i in range(len(df)):
        # If a trade closes at this bar, update equity
        if trade_idx < len(trades_sorted) and trades_sorted.loc[trade_idx, "exit_index"] == i:
            equity = trades_sorted.loc[trade_idx, "capital_after"]
            trade_idx += 1
        equity_curve.append(equity)

    df["equity"] = equity_curve

    plt.figure(figsize=(10, 4))
    plt.plot(df[time_col], df["equity"])
    plt.xlabel("Time")
    plt.ylabel("Equity")
    plt.title("Equity Curve - 5:1 RR Bounce & Retest Strategy")
    plt.tight_layout()
    plt.show()
