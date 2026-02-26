import argparse
from datetime import datetime
import pandas as pd
import numpy as np
import yfinance as yf

def fetch_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf
    import pandas as pd

    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)

    # Handle multi-index columns if they appear
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    # Keep only the 'Close' column
    df = df[['Close']].rename(columns={'Close': 'close'}).dropna()

    # Shift to get next day's close
    df['next_close'] = df['close'].shift(-1)

    # Compute next-day close-to-close return
    df['ret_1d'] = df['next_close'] / df['close'] - 1.0

    return df.dropna(subset=['next_close'])


def probs_dynamic_by_day(df: pd.DataFrame, ks=(1, 2)) -> pd.DataFrame:
    out = []
    n = len(df)
    for k in ks:
        threshold = np.floor(df['close']) - k
        hits = (df['next_close'] > threshold).sum()
        p = hits / n
        out.append({'k_dollars': k, 'probability': p, 'count': n})
    return pd.DataFrame(out)

def probs_fixed_x_return_cutoff(df: pd.DataFrame, x: float, ks=(1, 2)) -> pd.DataFrame:
    """
    If today's close = x, then 'close above x-k' is R > -k/x.
    Measure the proportion of historical next-day returns exceeding that cutoff.
    """
    out = []
    rets = df['ret_1d'].dropna()
    n = len(rets)
    for k in ks:
        cutoff = -k / x
        p = (rets > cutoff).mean()
        out.append({'k_dollars': k, 'x_level': x, 'return_cutoff': cutoff, 'probability': p, 'count': n})
    return pd.DataFrame(out)

def probs_by_year(df: pd.DataFrame, ks=(1, 2)) -> pd.DataFrame:
    """
    Optional diagnostics: dynamic-by-day probabilities per calendar year.
    """
    tmp = df.copy()
    tmp['year'] = tmp.index.year
    rows = []
    for y, g in tmp.groupby('year'):
        for k in ks:
            p = (g['next_close'] > (g['close'] - k)).mean()
            rows.append({'year': int(y), 'k_dollars': k, 'probability': p, 'count': len(g)})
    return pd.DataFrame(rows).sort_values(['year', 'k_dollars'])

def main():
    parser = argparse.ArgumentParser(description="Empirical next-day probability SPY closes above x-1 and x-2.")
    parser.add_argument('--ticker', default='SPY', help='Ticker symbol (default: SPY)')
    parser.add_argument('--start', default='1993-01-29', help='Start date YYYY-MM-DD (default: SPY inception)')
    parser.add_argument('--end', default=datetime.today().strftime('%Y-%m-%d'), help='End date YYYY-MM-DD (default: today)')
    parser.add_argument('--x', type=float, default=None, help='Optional fixed x level (e.g., 600) to evaluate return cutoffs -1/x and -2/x.')
    parser.add_argument('--no_year_breakdown', action='store_true', help='Skip per-year breakdown.')
    args = parser.parse_args()

    df = fetch_data(args.ticker, args.start, args.end)

    print("\n=== Dynamic-by-day (exact) probabilities ===")
    dyn = probs_dynamic_by_day(df, ks=(1, 2))
    for _, row in dyn.iterrows():
        print(f"k=${row['k_dollars']:<1}: P(next_close > close - {int(row['k_dollars'])}) = {row['probability']*100:5.2f}%  (n={int(row['count'])})")

    if args.x is not None:
        print(f"\n=== Fixed-x approximation using return cutoffs (x={args.x}) ===")
        fx = probs_fixed_x_return_cutoff(df, x=args.x, ks=(1, 2))
        for _, row in fx.iterrows():
            k = int(row['k_dollars'])
            print(f"k=${k}: cutoff R > {-k/args.x:.6f} -> Probability = {row['probability']*100:5.2f}%  (n={int(row['count'])})")

    if not args.no_year_breakdown:
        print("\n=== Year-by-year (dynamic-by-day) diagnostics ===")
        yearly = probs_by_year(df, ks=(1, 2))
        # Pretty print
        for y in yearly['year'].unique():
            sub = yearly[yearly['year'] == y]
            row1 = sub[sub['k_dollars'] == 1].iloc[0]
            row2 = sub[sub['k_dollars'] == 2].iloc[0]
            print(f"{y}:  >close-1: {row1['probability']*100:5.2f}%   >close-2: {row2['probability']*100:5.2f}%   (n={int(row1['count'])})")

if __name__ == '__main__':
    main()