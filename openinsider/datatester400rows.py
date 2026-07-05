import pandas as pd
import numpy as np
import requests
from io import StringIO

def get_insider_data_raw():
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/123.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
        }
    )

    urls = [
        "http://openinsider.com/latest-ceo-cfo-purchases-25k",
        "http://openinsider.com/latest-officer-purchases-25k",
        "http://openinsider.com/latest-cluster-buys",
        "http://openinsider.com/latest-insider-purchases-25k",
    ]

    all_frames = []

    for url in urls:
        print(f"\n[DEBUG] Fetching raw: {url}")

        try:
            response = session.get(url, timeout=(5, 12))
            response.raise_for_status()
            tables = pd.read_html(StringIO(response.text))
        except Exception as e:
            print(f"[DEBUG] Failed {url}: {repr(e)}")
            continue

        best_df = pick_best_table(tables)
        if best_df is None:
            continue

        if isinstance(best_df.columns, pd.MultiIndex):
            best_df.columns = [
                " ".join([str(x) for x in col if str(x) != "nan"]).replace("\xa0", " ").strip()
                for col in best_df.columns
            ]
        else:
            best_df.columns = [str(c).replace("\xa0", " ").strip() for c in best_df.columns]

        rename_map = {}
        for col in best_df.columns:
            c = str(col).replace("\xa0", " ").strip().lower()

            if "filing date" in c:
                rename_map[col] = "FilingDate"
            elif "trade date" in c:
                rename_map[col] = "TradeDate"
            elif "ticker" in c:
                rename_map[col] = "Ticker"
            elif "company" in c:
                rename_map[col] = "Company"
            elif "insider name" in c or c == "insider":
                rename_map[col] = "InsiderName"
            elif "title" in c:
                rename_map[col] = "Title"
            elif "trade type" in c:
                rename_map[col] = "TradeType"
            elif "price" in c:
                rename_map[col] = "Price"
            elif "qty" in c:
                rename_map[col] = "Qty"
            elif c == "owned":
                rename_map[col] = "Owned"
            elif "δown" in c or "Δown" in col or "deltaown" in c:
                rename_map[col] = "DeltaOwn"
            elif "value" in c:
                rename_map[col] = "Value"

        best_df = best_df.rename(columns=rename_map)
        best_df = best_df.loc[:, ~best_df.columns.duplicated()].copy()

        best_df["SourceURL"] = url

        all_frames.append(best_df)

    if not all_frames:
        return pd.DataFrame()

    raw_df = pd.concat(all_frames, ignore_index=True)
    raw_df = raw_df.loc[:, ~raw_df.columns.duplicated()].copy()

    return raw_df

def pick_best_table(tables):
    best_df = None
    best_score = -1

    for i, t in enumerate(tables):
        cols = [str(c).replace("\xa0", " ").strip().lower() for c in t.columns]

        score = 0
        wanted = [
            "filing date", "trade date", "ticker", "company",
            "insider", "title", "trade type", "price",
            "qty", "owned", "value"
        ]
        for k in wanted:
            if any(k in c for c in cols):
                score += 1

        print(f"[DEBUG] Table {i} score={score}")

        if score > best_score:
            best_score = score
            best_df = t.copy()

    return best_df

if __name__ == "__main__":
    raw_df = get_insider_data_raw()
    raw_df.to_csv("insider_raw_debug.csv", index=False)
    print(f"Saved insider_raw_debug.csv with {len(raw_df)} rows")