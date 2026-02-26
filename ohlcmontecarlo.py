#!/usr/bin/env python3
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf

def fetch_prices(start="2004-01-01", end=None, ticker="SPY", vix_ticker="^VIX"):
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")
    spy = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    vix = yf.download(vix_ticker, start=start, end=end, auto_adjust=False, progress=False)

    # Flatten MultiIndex
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = [c[0] for c in spy.columns]
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = [c[0] for c in vix.columns]

    if "Close" not in spy.columns or "Close" not in vix.columns:
        raise ValueError("Missing Close column in downloads. Check tickers or connectivity.")

    spy = spy[["Close"]].rename(columns={"Close": "spy_close"}).dropna()
    vix = vix[["Close"]].rename(columns={"Close": "vix_close"}).dropna()

    df = spy.join(vix, how="inner").dropna()
    if df.empty:
        raise ValueError("Joined SPY/VIX dataframe is empty. Try different dates.")
    return df

def prepare_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["spy_next_close"] = df["spy_close"].shift(-1)
    df["ret_1d"] = df["spy_next_close"] / df["spy_close"] - 1.0
    df["vix_daily_vol_imp"] = df["vix_close"] / np.sqrt(252.0)
    return df.dropna(subset=["spy_next_close", "ret_1d", "vix_daily_vol_imp"])

def calibrate_k(df: pd.DataFrame) -> float:
    # r^2 ~= k^2 * (vix/√252)^2  => k = sqrt(sum r^2 / sum sig_imp^2)
    r2 = (df["ret_1d"]**2).sum()
    s2 = (df["vix_daily_vol_imp"]**2).sum()
    if s2 <= 0:
        raise ValueError("VIX variance denominator non-positive.")
    return float(np.sqrt(r2 / s2))

def student_t_draws(size, df=7, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    return rng.standard_t(df=df, size=size)

def simulate_next_day_probs(
    x: float,
    vix_level: float,
    mu_daily: float,
    k_scale: float,
    nsims: int = 200_000,
    seed: int = 42,
    dist: str = "normal",
    df_t: int = 7,
    horizon_days: int = 1
):
    rng = np.random.default_rng(seed)
    sigma = k_scale * (vix_level / np.sqrt(252.0))

    # thresholds using integer-dollar logic
    t1 = np.floor(x) - 1.0
    t2 = np.floor(x) - 2.0
    r_cut_up = 0.0               # up day: C_{t+1} > x  <=>  R > 0
    r_cut_1  = t1 / x - 1.0      # > floor(x)-1
    r_cut_2  = t2 / x - 1.0      # > floor(x)-2

    # multi-day horizon (compound)
    def draw_returns(n):
        if dist == "normal":
            steps = rng.normal(loc=mu_daily, scale=sigma, size=(n, horizon_days))
        else:
            # Student-t then scale to match stdev sigma
            z = student_t_draws((n, horizon_days), df=df_t, rng=rng)
            # variance of t is df/(df-2) for df>2; scale so std = 1
            t_std = np.sqrt(df_t/(df_t-2)) if df_t > 2 else 1.0
            steps = (z / t_std) * sigma + mu_daily
        # compound returns over horizon
        tot = np.prod(1.0 + steps, axis=1) - 1.0
        return tot

    rets = draw_returns(nsims)

    # probabilities
    p_up = float((rets > r_cut_up).mean())
    p1   = float((rets > r_cut_1).mean())
    p2   = float((rets > r_cut_2).mean())

    # bucketed probabilities (use $ buckets on price change)
    # Map return to dollar move relative to x
    dollar_move = rets * x
    buckets = {
        "(-inf, -2$]": (dollar_move <= -2.0).mean(),
        "(-2$, -1$]": ((dollar_move > -2.0) & (dollar_move <= -1.0)).mean(),
        "(-1$, 0]":   ((dollar_move > -1.0) & (dollar_move <= 0.0)).mean(),
        "(0, +1$]":   ((dollar_move > 0.0) & (dollar_move <= +1.0)).mean(),
        "(+1$, +2$]": ((dollar_move > +1.0) & (dollar_move <= +2.0)).mean(),
        "(+2$, inf)": (dollar_move > +2.0).mean(),
    }
    # simple directional prediction
    prediction = "Up" if p_up >= 0.5 else "Down"

    return {
        "x": float(x),
        "vix_level": float(vix_level),
        "mu_daily": float(mu_daily),
        "sigma_daily": float(sigma),
        "k_scale": float(k_scale),
        "nsims": int(nsims),
        "dist": dist,
        "df_t": int(df_t),
        "horizon_days": int(horizon_days),
        "thresholds": {
            "floor(x)-1": float(t1),
            "floor(x)-2": float(t2),
            "return_cut_up": float(r_cut_up),
            "return_cut_1": float(r_cut_1),
            "return_cut_2": float(r_cut_2),
        },
        "probabilities": {
            "P(up)": p_up,
            "P(> floor(x)-1)": p1,
            "P(> floor(x)-2)": p2,
        },
        "buckets": {k: float(v) for k, v in buckets.items()},
        "prediction": prediction,
    }

def main():
    p = argparse.ArgumentParser(description="Predict next-day SPY with VIX-conditioned Monte Carlo.")
    p.add_argument("--x", type=float, default=None, help="Today's SPY close. If omitted, use last close.")
    p.add_argument("--vix", type=float, default=None, help="VIX level. If omitted, use last VIX close.")
    p.add_argument("--start", default="2004-01-01", help="Data start date (VIX robust from ~2004).")
    p.add_argument("--end", default=datetime.today().strftime("%Y-%m-%d"), help="Data end date.")
    p.add_argument("--lookback", type=int, default=252, help="Lookback days for mu calibration (hist).")
    p.add_argument("--mu_mode", choices=["hist","zero"], default="hist", help="Daily drift mode.")
    p.add_argument("--nsims", type=int, default=200000, help="Monte Carlo simulations.")
    p.add_argument("--seed", type=int, default=42, help="RNG seed.")
    p.add_argument("--dist", choices=["normal","student"], default="normal", help="Shock distribution.")
    p.add_argument("--df_t", type=int, default=7, help="Student-t degrees of freedom (if dist=student).")
    p.add_argument("--horizon", type=int, default=1, help="Horizon in trading days (default 1).")
    args = p.parse_args()

    try:
        df_all = fetch_prices(start=args.start, end=args.end)
    except Exception as e:
        print(f"ERROR: {e}")
        return

    # Defaults for x and vix: last available
    x = args.x if args.x is not None else float(df_all["spy_close"].iloc[-1])
    vix_level = args.vix if args.vix is not None else float(df_all["vix_close"].iloc[-1])

    df = prepare_returns(df_all)
    # use recent lookback for mean
    if args.mu_mode == "hist":
        mu_daily = float(df["ret_1d"].tail(args.lookback).mean())
    else:
        mu_daily = 0.0

    k_scale = calibrate_k(df)

    res = simulate_next_day_probs(
        x=x,
        vix_level=vix_level,
        mu_daily=mu_daily,
        k_scale=k_scale,
        nsims=args.nsims,
        seed=args.seed,
        dist=args.dist,
        df_t=args.df_t,
        horizon_days=args.horizon
    )

    print("\n=== VIX-Conditioned Monte Carlo Prediction ===")
    print(f"Last SPY close (x): {res['x']:.2f}")
    print(f"VIX level used   : {res['vix_level']:.2f}")
    print(f"Daily mu (mode)  : {res['mu_daily']:.6f}  ({args.mu_mode})")
    print(f"Daily sigma      : {res['sigma_daily']:.6f}  (k={res['k_scale']:.4f})")
    print(f"Dist/H/NSims     : {res['dist']}, H={res['horizon_days']}, N={res['nsims']:,}")

    th = res["thresholds"]
    print("\nThresholds (integer-dollar):")
    print(f" floor(x)-1: {th['floor(x)-1']:.2f}  -> return cutoff {th['return_cut_1']:.6f}")
    print(f" floor(x)-2: {th['floor(x)-2']:.2f}  -> return cutoff {th['return_cut_2']:.6f}")

    probs = res["probabilities"]
    print("\nEstimated probabilities for next close:")
    print(f" P(Up day)              : {100*probs['P(up)']:.2f}%")
    print(f" P(> floor(x)-1)        : {100*probs['P(> floor(x)-1)']:.2f}%")
    print(f" P(> floor(x)-2)        : {100*probs['P(> floor(x)-2)']:.2f}%")

    print("\nDollar-move buckets:")
    for k, v in res["buckets"].items():
        print(f" {k:>12}: {100*v:5.2f}%")

    print(f"\nSimple prediction (direction): {res['prediction']}")
    print("Note: This is a probabilistic estimate, not advice. Consider vol regimes & tail risk.\n")

if __name__ == "__main__":
    main()
