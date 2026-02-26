#!/usr/bin/env python3
"""
Monte Carlo simulation on daily prices using yfinance.

Models:
- GBM: calibrates drift/vol from log returns, simulates log-normal paths
- Bootstrap: resamples historical daily log returns with replacement

Outputs:
- Summary stats to stdout
- Plot of simulated price fan chart
- Optional CSV of terminal price distribution (toggle SAVE_CSV)

Usage:
  python mc_price_sim.py --ticker SPY --lookback 756 --days 252 --sims 20000 --model gbm
"""
import argparse
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

SAVE_CSV = False  # flip True if you want terminal prices saved

def fetch_adj_close(ticker: str, lookback_days: int) -> pd.Series:
    # Pull a bit extra for safety (market holidays)
    hist = yf.download(ticker, period="max", auto_adjust=True, progress=False)
    if hist.empty:
        raise ValueError(f"No data for {ticker}")
    prices = hist["Close"].dropna()
    # keep only last N trading days
    return prices.iloc[-lookback_days-1:]  # +1 to compute returns

def calibrate_gbm(log_rets: np.ndarray):
    """
    Calibrate GBM using daily log returns r_t = ln(S_t/S_{t-1}).
    Drift (mu) and vol (sigma) are *daily* parameters.
    """
    mu_hat = log_rets.mean()            # daily mean of log return
    sigma_hat = log_rets.std(ddof=1)    # daily vol
    # In continuous-time GBM, expected log return per step = (mu - 0.5*sigma^2)
    # Here mu_hat already includes that effect empirically. For simulation we use:
    drift = mu_hat
    vol = sigma_hat
    return drift, vol

def simulate_gbm(S0: float, drift: float, vol: float, days: int, sims: int, seed: int = 42) -> np.ndarray:
    """
    Vectorized GBM in log space.
    Returns array shape (days+1, sims)
    """
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((days, sims))
    # log returns for each step
    log_steps = (drift - 0.5 * vol**2) + vol * Z
    # cumulative sum across time
    log_path = np.vstack([np.zeros((1, sims)), np.cumsum(log_steps, axis=0)])
    return S0 * np.exp(log_path)

def simulate_bootstrap(S0: float, log_rets_hist: np.ndarray, days: int, sims: int, seed: int = 42) -> np.ndarray:
    """
    Block-free bootstrap: i.i.d. resampling of historical daily log returns.
    Returns array shape (days+1, sims)
    """
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(log_rets_hist), size=(days, sims))
    sampled = log_rets_hist[idx]
    log_path = np.vstack([np.zeros((1, sims)), np.cumsum(sampled, axis=0)])
    return S0 * np.exp(log_path)

def summarize_paths(paths: np.ndarray):
    terminal = paths[-1, :]
    out = {
        "mean_terminal": float(terminal.mean()),
        "median_terminal": float(np.median(terminal)),
        "p05": float(np.percentile(terminal, 5)),
        "p25": float(np.percentile(terminal, 25)),
        "p75": float(np.percentile(terminal, 75)),
        "p95": float(np.percentile(terminal, 95)),
        "exp_return_%": float((terminal.mean() / paths[0, :].mean() - 1) * 100),
        "downside_prob_(terminal<S0)_%": float((terminal < paths[0, :]).mean() * 100),
    }
    return out, terminal

def plot_fan_chart(paths: np.ndarray, ticker: str, model: str):
    qs = [5, 25, 50, 75, 95]
    qmat = np.percentile(paths, qs, axis=1)  # shape (len(qs), T)
    t = np.arange(paths.shape[0])
    plt.figure(figsize=(10, 5))
    plt.plot(t, qmat[2, :], label="Median")          # 50th
    plt.fill_between(t, qmat[1, :], qmat[3, :], alpha=0.3, label="IQR (25–75)")
    plt.fill_between(t, qmat[0, :], qmat[4, :], alpha=0.15, label="5–95%")
    plt.title(f"{ticker} Monte Carlo ({model.upper()}) — {paths.shape[0]-1} days, {paths.shape[1]:,} sims")
    plt.xlabel("Day")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", type=str, default="SPY")
    ap.add_argument("--lookback", type=int, default=756, help="Calibration window in trading days (~3y)")
    ap.add_argument("--days", type=int, default=252, help="Simulation horizon in trading days")
    ap.add_argument("--sims", type=int, default=10000, help="Number of simulated paths")
    ap.add_argument("--model", type=str, default="gbm", choices=["gbm", "bootstrap"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    prices = fetch_adj_close(args.ticker, args.lookback)
    S0 = float(prices.iloc[-1])
    log_rets = np.log(prices / prices.shift(1)).dropna().to_numpy()

    if args.model == "gbm":
        drift, vol = calibrate_gbm(log_rets)
        paths = simulate_gbm(S0, drift, vol, args.days, args.sims, seed=args.seed)
    else:
        paths = simulate_bootstrap(S0, log_rets, args.days, args.sims, seed=args.seed)

    stats, terminal = summarize_paths(paths)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"[{ts}] {args.ticker} S0={S0:.2f} | model={args.model} | lookback={args.lookback}d | horizon={args.days}d | sims={args.sims:,}")
    for k, v in stats.items():
        print(f"{k:>32}: {v:,.2f}")

    if SAVE_CSV:
        out = pd.DataFrame({"terminal_price": terminal})
        out.to_csv(f"terminal_{args.ticker}_{args.model}_{args.days}d_{args.sims}x.csv", index=False)

    plot_fan_chart(paths, args.ticker, args.model)

if __name__ == "__main__":
    main()
