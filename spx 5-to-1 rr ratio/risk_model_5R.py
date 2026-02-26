import yfinance as yf
import numpy as np
import pandas as pd

# ==========================
# GLOBAL CONFIG (fixed)
# ==========================
TICKER_SPX = "^GSPC"
TICKER_VIX = "^VIX"

PERIOD = "60d"          # 5m limited to ~60d
INTERVAL = "5m"

LEVEL_STEP = 50         # 6400, 6450, 6500, ...
STOP_OFFSET = 2.0       # 2 points
RR = 5.0                # 5:1 fixed

RETEST_TOLERANCE = 0.5  # how close to level counts as retest
RISK_PER_TRADE = 0.01   # 1% of equity per trade

INITIAL_CAPITAL = 10000.0
EMA_PERIOD = 50         # EMA span for trend filter

# VIX thresholds to test when filter is ON
VIX_THRESHOLDS = [15.0, 18.0, 20.0]

SIDE_MODES = ["both", "long_only", "short_only"]


# ==========================
# DATA DOWNLOAD & PREP
# ==========================
def load_data():
    print("Downloading SPX...")
    spx = yf.download(
        TICKER_SPX,
        period=PERIOD,
        interval=INTERVAL,
        auto_adjust=True,
    )

    if spx.empty:
        raise ValueError("No SPX data downloaded.")

    print("Downloading VIX...")
    vix = yf.download(
        TICKER_VIX,
        period=PERIOD,
        interval=INTERVAL,
        auto_adjust=True,
    )

    # Flatten MultiIndex if needed
    if isinstance(spx.columns, pd.MultiIndex):
        new_cols = []
        for col in spx.columns:
            parts = [str(x) for x in col if str(x) != ""]
            new_cols.append(parts[0] if parts else str(col))
        spx.columns = new_cols

    spx = spx[["Open", "High", "Low", "Close"]].dropna()
    spx = spx.astype(float)
    spx.reset_index(inplace=True)
    spx.rename(columns={spx.columns[0]: "Datetime"}, inplace=True)
    spx["Datetime"] = pd.to_datetime(spx["Datetime"])
    spx["date"] = spx["Datetime"].dt.date

    if vix.empty:
        print("WARNING: No VIX data downloaded. VIX filter will effectively do nothing.")
        spx["VIX_Close"] = np.nan
    else:
        if isinstance(vix.columns, pd.MultiIndex):
            new_cols = []
            for col in vix.columns:
                parts = [str(x) for x in col if str(x) != ""]
                new_cols.append(parts[0] if parts else str(col))
            vix.columns = new_cols

        vix = vix[["Close"]].dropna()
        vix = vix.astype(float)
        vix.reset_index(inplace=True)
        vix.rename(columns={vix.columns[0]: "Datetime"}, inplace=True)
        vix["Datetime"] = pd.to_datetime(vix["Datetime"])

        spx = pd.merge_asof(
            spx.sort_values("Datetime"),
            vix.sort_values("Datetime").rename(columns={"Close": "VIX_Close"}),
            on="Datetime",
            direction="backward",
        )

    # EMA for trend filter
    spx["EMA"] = spx["Close"].ewm(span=EMA_PERIOD).mean()

    print(f"SPX bars: {len(spx)}")
    print(f"Start: {spx['Datetime'].iloc[0]}, End: {spx['Datetime'].iloc[-1]}")

    # Levels
    price_min = float(spx["Low"].min())
    price_max = float(spx["High"].max())
    level_start = np.floor(price_min / LEVEL_STEP) * LEVEL_STEP
    level_end = np.ceil(price_max / LEVEL_STEP) * LEVEL_STEP
    levels = np.arange(level_start, level_end + LEVEL_STEP, LEVEL_STEP).astype(float)
    print("Levels:", levels)

    return spx, levels


# ==========================
# CONTINUATION ENTRY LOGIC
# ==========================
def check_entry_signal_continuation(i, row, prev_close, levels, level_state):
    """
    Continuation (breakout + retest):

    Long:
      - prev_close < L and close > L and high >= L
      - later retest near L from above -> long

    Short:
      - prev_close > L and close < L and low <= L
      - later retest near L from below -> short
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
        closed = {
            **trade,
            "exit_index": i,
            "exit_price": exit_price,
            "pnl": pnl,
            "outcome": outcome,
            "capital_after": capital,
        }
        return None, capital, closed

    return trade, capital, None


# ==========================
# BACKTEST FOR ONE COMBO
# ==========================
def run_backtest(spx, levels, use_vix_filter, vix_min, use_trend_filter, side_mode):
    # level state for continuation
    level_state = {float(L): {"mode": "idle", "direction": None} for L in levels}

    capital = INITIAL_CAPITAL
    open_trade = None
    trades = []

    for i in range(1, len(spx)):
        row = spx.iloc[i]
        prev_row = spx.iloc[i - 1]
        prev_close = prev_row["Close"]

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

        # Manage existing trade
        if open_trade is not None:
            open_trade, capital, closed = manage_open_trade(i, row, open_trade, capital)
            if closed is not None:
                trades.append(closed)

        # Look for new trade
        if open_trade is None:
            signal = check_entry_signal_continuation(i, row, prev_close, levels, level_state)
            if signal is None:
                continue

            direction = signal["direction"]

            # Side filter
            if side_mode == "long_only" and direction == "short":
                continue
            if side_mode == "short_only" and direction == "long":
                continue

            # Trend filter: EMA
            if use_trend_filter:
                ema = float(row["EMA"])
                close = float(row["Close"])
                if np.isnan(ema):
                    continue
                if direction == "long" and close <= ema:
                    continue
                if direction == "short" and close >= ema:
                    continue

            # VIX filter
            if use_vix_filter:
                vix_val = row.get("VIX_Close", np.nan)
                if np.isnan(vix_val) or vix_val < vix_min:
                    continue

            entry = signal["entry"]
            stop = signal["stop"]

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
            }

    # Forced exit at final bar
    if open_trade is not None:
        last_row = spx.iloc[-1]
        last_close = float(last_row["Close"])
        if open_trade["direction"] == "long":
            pnl = (last_close - open_trade["entry"]) * open_trade["units"]
        else:
            pnl = (open_trade["entry"] - last_close) * open_trade["units"]
        capital += pnl
        trades.append({
            **open_trade,
            "exit_index": len(spx) - 1,
            "exit_price": last_close,
            "pnl": pnl,
            "outcome": "forced_exit",
            "capital_after": capital,
        })
        open_trade = None

    trades_df = pd.DataFrame(trades)
    stats = compute_stats(capital, trades_df)
    return stats


# ==========================
# STATS
# ==========================
def compute_stats(capital, trades_df):
    stats = {
        "final_capital": capital,
        "net_return_pct": (capital / INITIAL_CAPITAL - 1.0) * 100.0,
        "num_trades": 0,
        "wins": 0,
        "losses": 0,
        "win_rate": 0.0,
        "long_pct": 0.0,
        "short_pct": 0.0,
        "avg_win": None,
        "avg_win_R": None,
        "avg_loss": None,
        "avg_loss_R": None,
    }

    if trades_df.empty:
        return stats

    total = len(trades_df)
    stats["num_trades"] = total

    wins_df = trades_df[trades_df["outcome"] == "target"]
    losses_df = trades_df[trades_df["outcome"] == "stop"]

    stats["wins"] = len(wins_df)
    stats["losses"] = len(losses_df)
    if total > 0:
        stats["win_rate"] = len(wins_df) / total * 100.0

    long_trades = trades_df[trades_df["direction"] == "long"]
    short_trades = trades_df[trades_df["direction"] == "short"]
    stats["long_pct"] = len(long_trades) / total * 100.0
    stats["short_pct"] = len(short_trades) / total * 100.0

    if not wins_df.empty:
        stats["avg_win"] = wins_df["pnl"].mean()
        stats["avg_win_R"] = (wins_df["pnl"] / wins_df["risk_amount"]).mean()
    if not losses_df.empty:
        stats["avg_loss"] = losses_df["pnl"].mean()
        stats["avg_loss_R"] = (losses_df["pnl"] / losses_df["risk_amount"]).mean()

    return stats


# ==========================
# MAIN: SWEEP ALL COMBOS
# ==========================
def main():
    spx, levels = load_data()

    rows = []

    combos = []
    # VIX filter OFF combos
    combos.append((False, None, False))
    combos.append((False, None, True))
    # VIX filter ON combos with thresholds
    for vmin in VIX_THRESHOLDS:
        combos.append((True, vmin, False))
        combos.append((True, vmin, True))

    total_runs = len(combos) * len(SIDE_MODES)
    run_id = 1

    for (use_vix, vmin, use_trend) in combos:
        for side_mode in SIDE_MODES:
            print(f"\n=== Running combo {run_id}/{total_runs} ===")
            print(f"VIX_FILTER={use_vix}, VIX_MIN={vmin}, TREND_FILTER={use_trend}, SIDE_MODE={side_mode}")
            run_id += 1

            stats = run_backtest(
                spx,
                levels,
                use_vix_filter=use_vix,
                vix_min=vmin if vmin is not None else 0.0,
                use_trend_filter=use_trend,
                side_mode=side_mode,
            )

            row = {
                "use_vix_filter": use_vix,
                "vix_min": vmin,
                "use_trend_filter": use_trend,
                "side_mode": side_mode,
                **stats,
            }
            rows.append(row)

    results_df = pd.DataFrame(rows)

    print("\n==================== SUMMARY (Continuation 5:1) ====================")
    print("VIX? Vmin  EMA?  Side        | FinalCap   Return%  Trades  Win%   Long%  Short%  AvgWin($,R)         AvgLoss($,R)")
    print("-" * 120)
    for _, r in results_df.iterrows():
        vix_flag = "Y" if r["use_vix_filter"] else "N"
        vmin_str = f"{r['vix_min']:.0f}" if pd.notna(r["vix_min"]) else "-"
        ema_flag = "Y" if r["use_trend_filter"] else "N"
        side = r["side_mode"]

        if pd.notna(r["avg_win"]):
            avg_win_str = f"{r['avg_win']:.2f}, {r['avg_win_R']:.2f}R"
        else:
            avg_win_str = "n/a"

        if pd.notna(r["avg_loss"]):
            avg_loss_str = f"{r['avg_loss']:.2f}, {r['avg_loss_R']:.2f}R"
        else:
            avg_loss_str = "n/a"

        print(
            f"{vix_flag:>3}  {vmin_str:>4}  {ema_flag:>3}  {side:<10} | "
            f"{r['final_capital']:9.2f}  {r['net_return_pct']:7.2f}%  "
            f"{int(r['num_trades']):6d}  {r['win_rate']:5.2f}%  "
            f"{r['long_pct']:6.2f}%  {r['short_pct']:6.2f}%  "
            f"{avg_win_str:<20} {avg_loss_str}"
        )

    # Optional: save the grid
    results_df.to_csv("spx_continuation_5R_grid_results.csv", index=False)
    print("\nSaved all results to spx_continuation_5R_grid_results.csv")


if __name__ == "__main__":
    main()
