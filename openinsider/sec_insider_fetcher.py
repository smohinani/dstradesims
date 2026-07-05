import pandas as pd
import numpy as np
import requests
from io import StringIO

# =========================
# CONFIG
# =========================
MIN_PRICE = 2
MAX_PRICE = 20
MIN_VALUE = 100000
MAX_RESULTS = 10

URLS = [
    "http://openinsider.com/latest-ceo-cfo-purchases-25k",
    "http://openinsider.com/latest-officer-purchases-25k",
    "http://openinsider.com/latest-cluster-buys",
    "http://openinsider.com/latest-insider-purchases-25k"
]

# =========================
# HELPERS
# =========================
def clean_numeric(series):
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]

    return pd.to_numeric(
        series.astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace("+", "", regex=False)
        .str.strip()
        .replace({"New": "100", "new": "100"}),
        errors="coerce",
    )


def role_score(title):
    t = str(title).lower()

    if any(x in t for x in ["ceo", "co-ceo", "chief executive", "chair", "chairman", "chairperson", "cob"]):
        return 5
    if any(x in t for x in ["cfo", "chief financial", "pres", "president"]):
        return 4
    if any(x in t for x in ["coo", "director", "10%"]):
        return 3
    if any(x in t for x in ["vp", "evp", "svp"]):
        return 2
    return 1


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

        #print(f"[DEBUG] Table {i} score={score}")

        if score > best_score:
            best_score = score
            best_df = t.copy()

    return best_df


# =========================
# MAIN FETCHER
# =========================
def get_insider_data():
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

    all_frames = []

    for url in URLS:
        #print(f"\n[DEBUG] Fetching: {url}")

        try:
            response = session.get(url, timeout=(5, 12))
            response.raise_for_status()
        except Exception as e:
            print(f"[DEBUG] Request failed for {url}: {repr(e)}")
            continue

        try:
            tables = pd.read_html(StringIO(response.text))
        except Exception as e:
            print(f"[DEBUG] read_html failed for {url}: {repr(e)}")
            continue

        #print(f"[DEBUG] Tables found: {len(tables)}")

        best_df = pick_best_table(tables)
        if best_df is None:
            print("[DEBUG] No usable table found.")
            continue

        if isinstance(best_df.columns, pd.MultiIndex):
            best_df.columns = [
                " ".join([str(x) for x in col if str(x) != "nan"]).replace("\xa0", " ").strip()
                for col in best_df.columns
            ]
        else:
            best_df.columns = [str(c).replace("\xa0", " ").strip() for c in best_df.columns]

        #print(f"[DEBUG] Raw columns: {best_df.columns.tolist()}")

        rename_map = {}
        for col in best_df.columns:
            c = str(col).replace("\xa0", " ").strip().lower()

            if c == "x" or c.startswith("x "):
                rename_map[col] = "X"
            elif "filing date" in c:
                rename_map[col] = "FilingDate"
            elif "trade date" in c:
                rename_map[col] = "TradeDate"
            elif "ticker" in c:
                rename_map[col] = "Ticker"
            elif "company name" in c or c == "company" or "company" in c:
                rename_map[col] = "Company"
            elif "insider name" in c or c == "insider":
                rename_map[col] = "InsiderName"
            elif c == "title" or "title" in c:
                rename_map[col] = "Title"
            elif "trade type" in c:
                rename_map[col] = "TradeType"
            elif c == "price" or "price" in c:
                rename_map[col] = "Price"
            elif "qty" in c:
                rename_map[col] = "Qty"
            elif c == "owned":
                rename_map[col] = "Owned"
            elif "δown" in c or "Δown" in col or "deltaown" in c:
                rename_map[col] = "DeltaOwn"
            elif c == "value" or "value" in c:
                rename_map[col] = "Value"
            elif c in ["1d", "1w", "1m", "6m"]:
                rename_map[col] = c.upper()

        best_df = best_df.rename(columns=rename_map)

        # Drop duplicate column names after rename
        best_df = best_df.loc[:, ~best_df.columns.duplicated()].copy()

        #(f"[DEBUG] Renamed columns: {best_df.columns.tolist()}")

        # Required text columns
        for col in ["FilingDate", "TradeDate", "Ticker", "Company", "InsiderName", "Title", "TradeType"]:
            if col not in best_df.columns:
                best_df[col] = ""

        # Required numeric columns
        for col in ["Price", "Qty", "Owned", "DeltaOwn", "Value"]:
            if col not in best_df.columns:
                best_df[col] = np.nan

        # Clean numerics
        best_df["Price"] = clean_numeric(best_df["Price"])
        best_df["Qty"] = clean_numeric(best_df["Qty"])
        best_df["Owned"] = clean_numeric(best_df["Owned"])
        best_df["DeltaOwn"] = clean_numeric(best_df["DeltaOwn"])
        best_df["Value"] = clean_numeric(best_df["Value"])

        # Normalize trade type and keep purchases only
        best_df["TradeType"] = (
            best_df["TradeType"]
            .astype(str)
            .str.replace("\xa0", " ", regex=False)
            .str.strip()
        )
        if best_df["TradeType"].str.len().gt(0).any():
            best_df = best_df[best_df["TradeType"].str.contains("P", na=False)].copy()

        # Local filters
        best_df = best_df[
            (best_df["Price"] >= MIN_PRICE) &
            (best_df["Price"] <= MAX_PRICE) &
            (best_df["Value"] >= MIN_VALUE)
        ].copy()

        #print(f"[DEBUG] Rows kept from {url}: {len(best_df)}")

        if not best_df.empty:
            preview_cols = [
                c for c in [
                    "Ticker", "Company", "InsiderName", "Title",
                    "TradeType", "Price", "Qty", "Owned", "DeltaOwn", "Value"
                ] if c in best_df.columns
            ]
            #print("[DEBUG] Preview:")
            #print(best_df[preview_cols].head(8).to_string(index=False))
            all_frames.append(best_df)

    if not all_frames:
        print("[DEBUG] No qualifying rows from any source.")
        return pd.DataFrame()

    df = pd.concat(all_frames, ignore_index=True)

    # Drop duplicate columns again after concat
    df = df.loc[:, ~df.columns.duplicated()].copy()

    # Drop duplicate rows
    dedupe_cols = [c for c in ["Ticker", "InsiderName", "TradeDate", "Price", "Qty", "Value"] if c in df.columns]
    if dedupe_cols:
        df = df.drop_duplicates(subset=dedupe_cols, keep="first").copy()

    # Score
    df["RoleScore"] = df["Title"].apply(role_score)

    cluster_counts = df.groupby("Ticker")["InsiderName"].nunique().reset_index()
    cluster_counts.columns = ["Ticker", "ClusterCount"]
    df = df.merge(cluster_counts, on="Ticker", how="left")

    df["ValueScore"] = np.log10(df["Value"].clip(lower=1))
    df["OwnershipScore"] = df["DeltaOwn"].fillna(0).clip(lower=0, upper=200) / 20.0
    df["ClusterScore"] = df["ClusterCount"].clip(lower=1, upper=5) * 1.5

    df["Score"] = (
        df["RoleScore"] * 4
        + df["ValueScore"] * 3
        + df["OwnershipScore"] * 2
        + df["ClusterScore"] * 2
    )

    df = df.sort_values(["Score", "Value"], ascending=[False, False]).copy()
    
    return df


# =========================
# DEBUG MAIN
# =========================
if __name__ == "__main__":
    df = get_insider_data()

    filename = f"insider_debug_{pd.Timestamp.today().strftime('%Y-%m-%d')}.csv"
    df.to_csv(filename, index=False)

    print(f"Saved CSV: {filename}")
    print(f"Rows saved: {len(df)}")