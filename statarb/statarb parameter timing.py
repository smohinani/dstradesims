import warnings
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)


PAIR_CONFIG = {
    "V/MA": {
        "tickers": ["V", "MA"],
        "labels": ["Visa", "Mastercard"],
    },
    "GLD/SLV": {
        "tickers": ["GLD", "SLV"],
        "labels": ["Gold", "Silver"],
    },
}


def download_close_prices(tickers):
    data = yf.download(tickers, start="1900-01-01", auto_adjust=True, progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        close = data.xs("Close", axis=1, level=0)
    else:
        close = data[["Close"]].copy()

    close = close.dropna()
    if close.shape[1] < 2:
        raise ValueError(f"Not enough columns returned for {tickers}")

    return close


def compute_return_correlation_by_years(close, labels):
    close.columns = labels
    returns = np.log(close).diff().dropna()

    end_date = returns.index.max()
    start_date = returns.index.min()
    max_years = max(1, int((end_date - start_date).days / 365.25))

    results = []
    for years_back in range(1, max_years + 1):
        cutoff = end_date - pd.DateOffset(years=years_back)
        window = returns[returns.index >= cutoff]

        if len(window) < 30:
            continue

        corr = float(window.iloc[:, 0].corr(window.iloc[:, 1]))
        results.append((years_back, corr))

    return results, max_years


def analyze_spread_signal(close, labels, z_threshold=2.0):
    close.columns = labels
    spread = np.log(close.iloc[:, 0]) - np.log(close.iloc[:, 1])
    spread = spread.dropna()

    rolling_mean = spread.rolling(window=60, min_periods=60).mean()
    rolling_std = spread.rolling(window=60, min_periods=60).std()
    zscore = (spread - rolling_mean) / rolling_std
    zscore = zscore.dropna()

    abs_z = zscore.abs()
    max_divergence = float(abs_z.max())
    avg_abs_z = float(abs_z.mean())

    exceedances = (abs_z >= z_threshold).astype(int)
    signal_count = int(exceedances.sum())
    signal_rate = float(signal_count / len(exceedances)) if len(exceedances) else np.nan

    durations = []
    current_run = 0
    for is_signal in exceedances:
        if is_signal:
            current_run += 1
        else:
            if current_run > 0:
                durations.append(current_run)
                current_run = 0
    if current_run > 0:
        durations.append(current_run)

    avg_duration = float(np.mean(durations)) if durations else np.nan
    max_duration = int(np.max(durations)) if durations else 0

    reversion_days = []
    for i in range(len(zscore)):
        if abs_z.iloc[i] >= z_threshold:
            j = i + 1
            while j < len(zscore) and abs_z.iloc[j] < z_threshold:
                j += 1
            if j < len(zscore):
                reversion_days.append((zscore.index[j] - zscore.index[i]).days)

    avg_reversion_days = float(np.mean(reversion_days)) if reversion_days else np.nan
    median_reversion_days = float(np.median(reversion_days)) if reversion_days else np.nan

    return {
        "max_divergence_z": max_divergence,
        "avg_abs_z": avg_abs_z,
        "signal_count": signal_count,
        "signal_rate": signal_rate,
        "avg_duration": avg_duration,
        "max_duration": max_duration,
        "avg_reversion_days": avg_reversion_days,
        "median_reversion_days": median_reversion_days,
        "threshold": z_threshold,
    }


def build_ml_features(close, labels, lookback=60, horizon=5):
    close = close.copy()
    close.columns = labels
    prices = np.log(close)
    spread = prices.iloc[:, 0] - prices.iloc[:, 1]
    spread = spread.dropna()

    rolling_mean = spread.rolling(window=lookback, min_periods=lookback).mean()
    rolling_std = spread.rolling(window=lookback, min_periods=lookback).std()
    zscore = (spread - rolling_mean) / rolling_std

    features = pd.DataFrame(index=spread.index)
    features["spread_z"] = zscore
    features["spread_z_lag1"] = zscore.shift(1)
    features["spread_z_lag2"] = zscore.shift(2)
    features["ret_a"] = prices.iloc[:, 0].diff().shift(1)
    features["ret_b"] = prices.iloc[:, 1].diff().shift(1)
    features["ret_a_lag2"] = prices.iloc[:, 0].diff().shift(2)
    features["ret_b_lag2"] = prices.iloc[:, 1].diff().shift(2)
    features["spread_change"] = spread.diff().shift(1)

    target = (spread.shift(-horizon) - spread).dropna()
    features = features.loc[target.index]
    features["target"] = target.values
    features = features.dropna()
    return features[[c for c in features.columns if c != "target"]], features["target"]


def backtest_pairs_trade(close, labels, start_balance=10000.0, z_entry=2.0, z_exit=0.5, lookback=60, max_hold_days=20, transaction_cost=0.001):
    close = close.copy()
    close.columns = labels
    end_date = close.index.max()
    start_date = end_date - pd.DateOffset(years=10)
    close = close[close.index >= start_date].copy()

    prices = np.log(close)
    spread = prices.iloc[:, 0] - prices.iloc[:, 1]
    spread = spread.dropna()
    rolling_mean = spread.rolling(window=lookback, min_periods=lookback).mean()
    rolling_std = spread.rolling(window=lookback, min_periods=lookback).std()
    zscore = (spread - rolling_mean) / rolling_std
    zscore = zscore.dropna()

    balance = float(start_balance)
    equity_curve = [balance]
    trades = []
    active_trade = None
    entry_date = None
    entry_prices = None
    entry_z = None

    def pnl_from_trade(side, entry_prices, exit_prices, notional=10000.0):
        short_symbol = labels[0]
        long_symbol = labels[1]
        if side == "short_a_long_b":
            short_pnl = notional * (entry_prices[short_symbol] / exit_prices[short_symbol] - 1.0)
            long_pnl = notional * (exit_prices[long_symbol] / entry_prices[long_symbol] - 1.0)
            return short_pnl + long_pnl - 2 * notional * transaction_cost
        else:
            long_pnl = notional * (exit_prices[short_symbol] / entry_prices[short_symbol] - 1.0)
            short_pnl = notional * (entry_prices[long_symbol] / exit_prices[long_symbol] - 1.0)
            return long_pnl + short_pnl - 2 * notional * transaction_cost

    for date, z in zscore.items():
        if active_trade is None:
            if z > z_entry:
                active_trade = "short_a_long_b"
                entry_date = date
                entry_prices = {
                    labels[0]: float(close.loc[date, labels[0]]),
                    labels[1]: float(close.loc[date, labels[1]]),
                }
                entry_z = float(z)
            elif z < -z_entry:
                active_trade = "long_a_short_b"
                entry_date = date
                entry_prices = {
                    labels[0]: float(close.loc[date, labels[0]]),
                    labels[1]: float(close.loc[date, labels[1]]),
                }
                entry_z = float(z)
        else:
            exit_now = abs(z) < z_exit or (date - entry_date).days >= max_hold_days
            if exit_now:
                exit_prices = {
                    labels[0]: float(close.loc[date, labels[0]]),
                    labels[1]: float(close.loc[date, labels[1]]),
                }
                trade_pnl = pnl_from_trade(active_trade, entry_prices, exit_prices)
                balance += trade_pnl
                equity_curve.append(balance)
                trades.append((entry_date, date, entry_z, float(z), trade_pnl))
                active_trade = None
                entry_date = None
                entry_prices = None
                entry_z = None

    if active_trade is not None:
        last_date = zscore.index[-1]
        exit_prices = {
            labels[0]: float(close.loc[last_date, labels[0]]),
            labels[1]: float(close.loc[last_date, labels[1]]),
        }
        trade_pnl = pnl_from_trade(active_trade, entry_prices, exit_prices)
        balance += trade_pnl
        equity_curve.append(balance)
        trades.append((entry_date, last_date, entry_z, None, trade_pnl))

    if not trades:
        return {"final_balance": start_balance, "total_return_pct": 0.0, "trades": 0, "avg_trade_pnl": 0.0, "max_drawdown_pct": 0.0, "equity_curve": [start_balance]}

    equity_array = np.array(equity_curve, dtype=float)
    peak = np.maximum.accumulate(equity_array)
    drawdown = 1 - (equity_array / peak)
    max_drawdown_pct = float(drawdown.max() * 100)
    total_return_pct = float((balance / start_balance - 1.0) * 100)
    avg_trade_pnl = float(np.mean([t[4] for t in trades]))
    return {"final_balance": float(balance), "total_return_pct": total_return_pct, "trades": len(trades), "avg_trade_pnl": avg_trade_pnl, "max_drawdown_pct": max_drawdown_pct, "equity_curve": equity_curve, "trades_data": trades}


def benchmark_buy_hold(close, labels, start_balance=10000.0):
    close = close.copy()
    close.columns = labels
    end_date = close.index.max()
    start_date = end_date - pd.DateOffset(years=10)
    close = close[close.index >= start_date].copy()
    first = close.iloc[0]
    last = close.iloc[-1]
    total = 0.5 * start_balance * (last.iloc[0] / first.iloc[0]) + 0.5 * start_balance * (last.iloc[1] / first.iloc[1])
    return float(total)


def backtest_neural_net(close, labels, start_balance=10000.0, lookback=60, z_entry=2.0, z_exit=1.0, max_hold_days=20, pred_threshold=0.002, horizon=5, transaction_cost=0.001):
    close = close.copy()
    close.columns = labels
    end_date = close.index.max()
    start_date = end_date - pd.DateOffset(years=10)
    close = close[close.index >= start_date].copy()

    X, y = build_ml_features(close, labels, lookback=lookback, horizon=horizon)
    split_idx = int(len(X) * 0.7)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(32, 16), activation="relu", max_iter=500, random_state=42, early_stopping=True, n_iter_no_change=25))
    model.fit(X_train, y_train)

    prices = np.log(close)
    spread = prices.iloc[:, 0] - prices.iloc[:, 1]
    spread = spread.dropna()
    rolling_mean = spread.rolling(window=lookback, min_periods=lookback).mean()
    rolling_std = spread.rolling(window=lookback, min_periods=lookback).std()
    zscore = (spread - rolling_mean) / rolling_std
    zscore = zscore.dropna()

    test_dates = X_test.index
    predictions = model.predict(X_test)
    balance = float(start_balance)
    equity_curve = [balance]
    trades = []
    active_trade = None
    entry_date = None
    entry_prices = None
    entry_z = None
    entry_pred = None

    for idx, date in enumerate(test_dates):
        z = float(zscore.loc[date])
        pred = float(predictions[idx])
        if active_trade is None:
            if (z > z_entry and pred < -pred_threshold) or (z < -z_entry and pred > pred_threshold):
                active_trade = "short_a_long_b" if z > z_entry else "long_a_short_b"
                entry_date = date
                entry_prices = {
                    labels[0]: float(close.loc[date, labels[0]]),
                    labels[1]: float(close.loc[date, labels[1]]),
                }
                entry_z = z
                entry_pred = pred
        else:
            exit_now = abs(z) < z_exit or (date - entry_date).days >= max_hold_days
            if exit_now:
                exit_prices = {
                    labels[0]: float(close.loc[date, labels[0]]),
                    labels[1]: float(close.loc[date, labels[1]]),
                }
                notional = 10000.0
                if active_trade == "short_a_long_b":
                    short_pnl = notional * (entry_prices[labels[0]] / exit_prices[labels[0]] - 1.0)
                    long_pnl = notional * (exit_prices[labels[1]] / entry_prices[labels[1]] - 1.0)
                    trade_pnl = short_pnl + long_pnl - 2 * notional * transaction_cost
                else:
                    long_pnl = notional * (exit_prices[labels[0]] / entry_prices[labels[0]] - 1.0)
                    short_pnl = notional * (entry_prices[labels[1]] / exit_prices[labels[1]] - 1.0)
                    trade_pnl = long_pnl + short_pnl - 2 * notional * transaction_cost
                balance += trade_pnl
                equity_curve.append(balance)
                trades.append((entry_date, date, entry_z, entry_pred, pred, trade_pnl))
                active_trade = None
                entry_date = None
                entry_prices = None
                entry_z = None
                entry_pred = None

    if active_trade is not None:
        last_date = test_dates[-1]
        exit_prices = {
            labels[0]: float(close.loc[last_date, labels[0]]),
            labels[1]: float(close.loc[last_date, labels[1]]),
        }
        notional = 10000.0
        if active_trade == "short_a_long_b":
            short_pnl = notional * (entry_prices[labels[0]] / exit_prices[labels[0]] - 1.0)
            long_pnl = notional * (exit_prices[labels[1]] / entry_prices[labels[1]] - 1.0)
            trade_pnl = short_pnl + long_pnl - 2 * notional * transaction_cost
        else:
            long_pnl = notional * (exit_prices[labels[0]] / entry_prices[labels[0]] - 1.0)
            short_pnl = notional * (entry_prices[labels[1]] / exit_prices[labels[1]] - 1.0)
            trade_pnl = long_pnl + short_pnl - 2 * notional * transaction_cost
        balance += trade_pnl
        equity_curve.append(balance)
        trades.append((entry_date, last_date, entry_z, entry_pred, None, trade_pnl))

    if not trades:
        return {"final_balance": start_balance, "total_return_pct": 0.0, "trades": 0, "avg_trade_pnl": 0.0, "max_drawdown_pct": 0.0, "equity_curve": [start_balance]}

    equity_array = np.array(equity_curve, dtype=float)
    peak = np.maximum.accumulate(equity_array)
    drawdown = 1 - (equity_array / peak)
    max_drawdown_pct = float(drawdown.max() * 100)
    total_return_pct = float((balance / start_balance - 1.0) * 100)
    avg_trade_pnl = float(np.mean([t[5] for t in trades]))
    return {"final_balance": float(balance), "total_return_pct": total_return_pct, "trades": len(trades), "avg_trade_pnl": avg_trade_pnl, "max_drawdown_pct": max_drawdown_pct, "equity_curve": equity_curve, "trades_data": trades}


for pair_name, cfg in PAIR_CONFIG.items():
    print(f"\n=== {pair_name} ===")
    close = download_close_prices(cfg["tickers"])
    results, max_years = compute_return_correlation_by_years(close, cfg["labels"])

    print(f"Longest available lookback window: {max_years} years")
    print("Years back | Correlation of daily returns")
    for years_back, corr in results:
        print(f"{years_back:>10} | {corr:>28.4f}")

    if results:
        best_years, best_corr = max(results, key=lambda x: x[1])
        worst_years, worst_corr = min(results, key=lambda x: x[1])
        print(f"\nBest correlation: {best_corr:.4f} over the last {best_years} years")
        print(f"Worst correlation: {worst_corr:.4f} over the last {worst_years} years")
    else:
        print("No valid windows were available.")

    stats = analyze_spread_signal(close, cfg["labels"])
    print("\nSpread-based signal stats")
    print(f"Max divergence (|z|): {stats['max_divergence_z']:.2f}")
    print(f"Average |z|: {stats['avg_abs_z']:.2f}")
    print(f"Signal threshold: |z| >= {stats['threshold']}")
    print(f"Signal count: {stats['signal_count']}")
    print(f"Signal rate: {stats['signal_rate']:.2%} of days")
    print(f"Average signal duration: {stats['avg_duration']:.1f} days")
    print(f"Max signal duration: {stats['max_duration']} days")
    print(f"Average time to revert below threshold: {stats['avg_reversion_days']:.1f} days")
    print(f"Median time to revert below threshold: {stats['median_reversion_days']:.1f} days")

    benchmark = benchmark_buy_hold(close, cfg["labels"])
    print(f"\n10-year benchmark (5k long in each ticker): ${benchmark:,.2f}")

    best_result = None
    best_params = None
    for lookback in [60, 90, 120]:
        for z_entry in [2.0, 2.5]:
            for z_exit in [0.5, 1.0]:
                for pred_threshold in [0.001, 0.002]:
                    for max_hold_days in [10, 20, 40]:
                        for horizon in [3, 5]:
                            result = backtest_neural_net(close, cfg["labels"], lookback=lookback, z_entry=z_entry, z_exit=z_exit, max_hold_days=max_hold_days, pred_threshold=pred_threshold, horizon=horizon)
                            if best_result is None or result["final_balance"] > best_result["final_balance"]:
                                best_result = result
                                best_params = (lookback, z_entry, z_exit, pred_threshold, max_hold_days, horizon)

    if best_result is not None and best_result["final_balance"] > benchmark:
        lookback, z_entry, z_exit, pred_threshold, max_hold_days, horizon = best_params
        print(f"Best benchmark-beating neural-net combo -> lookback={lookback}, z_entry={z_entry}, z_exit={z_exit}, pred_threshold={pred_threshold}, max_hold_days={max_hold_days}, horizon={horizon}")
        print(f"Ending balance: ${best_result['final_balance']:,.2f}")
        print(f"Total return: {best_result['total_return_pct']:.2f}%")
        print(f"Max drawdown: {best_result['max_drawdown_pct']:.2f}%")
    else:
        print("No neural-net combo beat the simple 5k-long-in-each-ticker benchmark over the last 10 years.")

    base = backtest_pairs_trade(close, cfg["labels"], z_entry=2.5, z_exit=1.0, lookback=120, max_hold_days=60)
    print(f"\nSimple z-score pair trade")
    print(f"Ending balance: ${base['final_balance']:,.2f}")
    print(f"Total return: {base['total_return_pct']:.2f}%")
    print(f"Max drawdown: {base['max_drawdown_pct']:.2f}%")
