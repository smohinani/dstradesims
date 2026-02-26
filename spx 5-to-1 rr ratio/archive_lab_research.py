import yfinance as yf
import numpy as np
import pandas as pd

# ==========================
# CONFIG
# ==========================
TICKER = "^GSPC"            # S&P 500 index on Yahoo
PERIOD = "60d"              # 5m data only available for last ~60 days
INTERVAL = "5m"             # intraday resolution

LEVEL_STEP = 50             # 50-point levels: 6400, 6450, 6500, ...
STOP_OFFSET = 2.0           # 2 points beyond level
RETEST_TOLERANCE = 0.5      # how close to the level counts as a retest
RISK_PER_TRADE = 0.01       # 1% of current equity per trade

INITIAL_CAPITAL = 10000.0


# ==========================
# DATA DOWNLOAD & PREP
# ==========================
def load_data():
    print("Downloading data...")
    df = yf.download(
        TICKER,
        period=PERIOD,
        interval=INTERVAL,
        auto_adjust=True,
    )

    if df.empty:
        raise ValueError("No data downloaded. Check ticker, period, or interval.")

    # Flatten MultiIndex columns if needed
    if isinstance(df.columns, pd.MultiIndex):
        new_cols = []
        for col in df.columns:
            parts = [str(x) for x in col if str(x) != ""]
            new_cols.append(parts[0] if parts else str(col))
        df.columns = new_cols

    df = df[["Open", "High", "Low", "Close"]].dropna()
    df = df.astype(float)
    df.reset_index(inplace=True)

    time_col = df.columns[0]  # usually "Datetime"
    df[time_col] = pd.to_datetime(df[time_col])
    df["date"] = df[time_col].dt.date

    print(f"Downloaded {len(df)} bars of data")
    print(f"Data start: {df[time_col].iloc[0]}")
    print(f"Data end:   {df[time_col].iloc[-1]}")

    # Levels
    price_min = float(df["Low"].min())
    price_max = float(df["High"].max())

    level_start = np.floor(price_min / LEVEL_STEP) * LEVEL_STEP
    level_end = np.ceil(price_max / LEVEL_STEP) * LEVEL_STEP

    levels = np.arange(level_start, level_end + LEVEL_STEP, LEVEL_STEP).astype(float)
    print("Levels:", levels)

    return df, levels, time_col


# ==========================
# ENTRY LOGIC
# ==========================
def check_entry_signal_rejection(i, row, prev_close, levels, level_state):
    """
    Rejection (bounce + retest) logic.
    """
    row_low = float(row["Low"])
    row_high = float(row["High"])
    row_close = float(row["Close"])
    prev_close = float(prev_close)

    for L in levels:
        L = float(L)
        state = level_state[L]
        mode = state["mode"]

        touched = (row_low <= L) and (L <= row_high)

        # First touch & bounce
        if mode == "idle" and touched:
            # Coming down from above, close above -> support bounce, long bias
            if prev_close > L and row_close > L:
                state["mode"] = "waiting_retest"
                state["direction"] = "long"
            # Coming up from below, close below -> resistance bounce, short bias
            elif prev_close < L and row_close < L:
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


def check_entry_signal_continuation(i, row, prev_close, levels, level_state):
    """
    Continuation (breakout + retest) logic.
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
# BACKTEST FOR GIVEN R:R AND MODE
# ==========================
def run_backtest(df, levels, mode, rr):
    """
    mode: "continuation" or "rejection"
    rr: float, risk:reward ratio (e.g. 5.0 for 5:1)
    """
    level_state = {float(L): {"mode": "idle", "direction": None} for L in levels}

    capital = INITIAL_CAPITAL
    open_trade = None
    trades = []

    print(f"\nRunning backtest for mode={mode}, RR={rr}:1 ...")

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i - 1]
        prev_close = prev_row["Close"]

        # Overnight flat: if date changes, close open trade at previous bar's close
        day = row["date"]
        prev_day = prev_row["date"]

        if open_trade is not None and day != prev_day:
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

        # Manage open trade during the day
        if open_trade is not None:
            open_trade, capital, closed = manage_open_trade(i, row, open_trade, capital)
            if closed is not None:
                trades.append(closed)

        # Look for new entry if flat
        if open_trade is None:
            if mode == "rejection":
                signal = check_entry_signal_rejection(i, row, prev_close, levels, level_state)
            else:
                signal = check_entry_signal_continuation(i, row, prev_close, levels, level_state)

            if signal is not None:
                entry = signal["entry"]
                stop = signal["stop"]

                if signal["direction"] == "long":
                    target = entry + (entry - stop) * rr
                else:
                    target = entry - (stop - entry) * rr

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
                }

    # Forced exit at last bar if still open
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

    trades_df = pd.DataFrame(trades)
    result = {
        "mode": mode,
        "RR": rr,
        "final_capital": capital,
        "net_return_pct": (capital / INITIAL_CAPITAL - 1.0) * 100.0,
        "num_trades": 0,
        "wins": 0,
        "losses": 0,
        "win_rate": 0.0,
        "avg_win": None,
        "avg_loss": None,
        "avg_win_R": None,
        "avg_loss_R": None,
    }

    if trades_df.empty:
        return result

    wins_df = trades_df[trades_df["outcome"] == "target"]
    losses_df = trades_df[trades_df["outcome"] == "stop"]

    wins = len(wins_df)
    losses = len(losses_df)
    total = len(trades_df)

    result["num_trades"] = total
    result["wins"] = wins
    result["losses"] = losses
    if total > 0:
        result["win_rate"] = wins / total * 100.0

    if not wins_df.empty:
        result["avg_win"] = wins_df["pnl"].mean()
        result["avg_win_R"] = (wins_df["pnl"] / wins_df["risk_amount"]).mean()

    if not losses_df.empty:
        result["avg_loss"] = losses_df["pnl"].mean()
        result["avg_loss_R"] = (losses_df["pnl"] / losses_df["risk_amount"]).mean()

    return result


# ==========================
# MAIN
# ==========================
def main():
    df, levels, time_col = load_data()

    modes = ["continuation", "rejection"]
    all_mode_results = {}

    for mode in modes:
        mode_results = []
        for rr in range(1, 11):  # 1:1 to 10:1
            res = run_backtest(df, levels, mode, rr)
            mode_results.append(res)
        all_mode_results[mode] = mode_results

    # Print summary per mode
    for mode in modes:
        print(f"\n\nMode: {mode}")
        print("RR  | Final Cap | Return % | Trades | Win%  | Avg Win ($, R)         | Avg Loss ($, R)")
        print("-" * 90)
        for res in all_mode_results[mode]:
            rr = res["RR"]
            cap = res["final_capital"]
            ret = res["net_return_pct"]
            trades = res["num_trades"]
            winrate = res["win_rate"]

            if res["avg_win"] is not None:
                avg_win_str = f"{res['avg_win']:.2f}, {res['avg_win_R']:.2f}R"
            else:
                avg_win_str = "n/a"

            if res["avg_loss"] is not None:
                avg_loss_str = f"{res['avg_loss']:.2f}, {res['avg_loss_R']:.2f}R"
            else:
                avg_loss_str = "n/a"

            print(
                f"{rr:>2}:1 | {cap:9.2f} | {ret:7.2f}% | {trades:6d} | {winrate:5.2f}% | "
                f"{avg_win_str:<22} | {avg_loss_str}"
            )


if __name__ == "__main__":
    main()
