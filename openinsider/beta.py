import pandas as pd
import numpy as np
import yfinance as yf

from sec_insider_fetcher import get_insider_data
from alpha import build_trade_plan


# =========================
# CONFIG
# =========================
DEFAULT_BENCHMARK = "SPY"
TOP_N_ALPHA = 10

# Sector-level ETF map from yfinance "sector" field
SECTOR_NAME_TO_ETF = {
    "technology": "XLK",
    "consumer cyclical": "XLY",
    "consumer defensive": "XLP",
    "financial services": "XLF",
    "healthcare": "XLV",
    "industrials": "XLI",
    "energy": "XLE",
    "utilities": "XLU",
    "basic materials": "XLB",
    "real estate": "XLRE",
    "communication services": "XLC",
}

# More precise industry keyword mapping from yfinance "industry" field
INDUSTRY_KEYWORDS_TO_ETF = {
    "asset management": "XLF",
    "banks": "XLF",
    "capital markets": "XLF",
    "insurance": "XLF",
    "reit": "XLRE",

    "biotechnology": "XBI",
    "drug manufacturers": "XLV",
    "medical devices": "XLV",
    "medical instruments": "XLV",
    "health information services": "XLV",
    "diagnostics & research": "XLV",

    "oil & gas": "XLE",
    "uranium": "XLE",
    "solar": "ICLN",
    "renewable": "ICLN",

    "aerospace": "XLI",
    "airlines": "XLI",
    "railroads": "XLI",
    "trucking": "XLI",
    "engineering & construction": "XLI",

    "internet retail": "XLY",
    "auto manufacturers": "XLY",
    "restaurants": "XLY",
    "travel services": "XLY",

    "grocery stores": "XLP",
    "discount stores": "XLP",
    "household": "XLP",
    "beverages": "XLP",
    "packaged foods": "XLP",

    "telecom": "XLC",
    "entertainment": "XLC",
    "broadcasting": "XLC",
    "publishing": "XLC",

    "semiconductors": "SOXX",
    "software": "IGV",
}

# Special hardcoded overrides for names where normal sector mapping is not ideal
SPECIAL_TICKER_ETF_MAP = {
    "PFLT": "BIZD",
    "ECC": "BIZD",
    "SCM": "BIZD",
    "ARCC": "BIZD",
    "GO": "XLP",
    "BLCO": "XLV",
    "MYGN": "XBI",
    "NP": "XLI",
    "VENU": "XLC",
}

INFO_CACHE = {}


# =========================
# HELPERS
# =========================
def clean_yf_data(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(how="all")
    return df


def download_ohlc(ticker: str, period: str = "3mo") -> pd.DataFrame:
    df = yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )

    df = clean_yf_data(df)

    if df.empty:
        return pd.DataFrame()

    needed = ["Open", "High", "Low", "Close", "Volume"]
    for col in needed:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna(how="all")


def download_close_series(ticker: str, period: str = "3mo") -> pd.Series:
    df = download_ohlc(ticker, period=period)

    if df.empty or "Close" not in df.columns:
        return pd.Series(dtype=float)

    close = df["Close"].dropna()

    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    return pd.to_numeric(close, errors="coerce").dropna()


def get_ticker_info_cached(ticker: str) -> dict:
    ticker = str(ticker).strip().upper()

    if ticker in INFO_CACHE:
        return INFO_CACHE[ticker]

    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        info = {}

    INFO_CACHE[ticker] = info
    return info


def infer_etf_from_info(ticker: str) -> tuple[str, str, str]:
    """
    Returns:
        (industry_etf, mapping_source, mapping_detail)
    """
    ticker = str(ticker).strip().upper()

    # 1. Special overrides
    if ticker in SPECIAL_TICKER_ETF_MAP:
        return SPECIAL_TICKER_ETF_MAP[ticker], "special_ticker_map", ticker

    info = get_ticker_info_cached(ticker)

    sector = str(info.get("sector", "") or "").strip().lower()
    industry = str(info.get("industry", "") or "").strip().lower()
    quote_type = str(info.get("quoteType", "") or "").strip().lower()
    long_name = str(info.get("longName", "") or "").strip().lower()
    category = str(info.get("category", "") or "").strip().lower()

    combined_text = " | ".join([sector, industry, quote_type, long_name, category])

    # 2. BDC / closed-end / fund-like detection
    if any(x in combined_text for x in ["bdc", "business development", "closed-end", "closed end"]):
        return "BIZD", "industry_keyword", combined_text

    # 3. Industry keyword match
    for keyword, etf in INDUSTRY_KEYWORDS_TO_ETF.items():
        if keyword in industry or keyword in combined_text:
            return etf, "industry_keyword", keyword

    # 4. Sector match
    if sector in SECTOR_NAME_TO_ETF:
        return SECTOR_NAME_TO_ETF[sector], "sector_name", sector

    # 5. Fallback to market benchmark
    return DEFAULT_BENCHMARK, "fallback_benchmark", "unknown"


def get_industry_etf(ticker: str) -> str:
    industry_etf, _, _ = infer_etf_from_info(ticker)
    return industry_etf


def get_mapping_confidence(mapping_source: str, industry_etf: str) -> float:
    if mapping_source == "special_ticker_map":
        return 1.00
    if mapping_source == "industry_keyword":
        return 0.95
    if mapping_source == "sector_name":
        return 0.85
    if mapping_source == "fallback_benchmark" and industry_etf == DEFAULT_BENCHMARK:
        return 0.25
    return 0.50


def safe_round(x, n=4):
    if x is None or pd.isna(x):
        return None
    return round(float(x), n)


def score_label(score_5: float) -> str:
    if score_5 >= 4.2:
        return "STRONG BUY"
    elif score_5 >= 3.4:
        return "BUY"
    elif score_5 >= 2.5:
        return "WATCH"
    elif score_5 >= 1.5:
        return "WEAK"
    else:
        return "AVOID"


def explain_decision(score_5: float) -> str:
    if score_5 >= 4.2:
        return "Good buy"
    elif score_5 >= 3.4:
        return "Buyable"
    elif score_5 >= 2.5:
        return "Watch / small size"
    else:
        return "Not a good buy"


# =========================
# ALPHA INPUT
# keep repeated tickers so frequency matters
# =========================
def get_alpha_top_candidates(top_n: int = TOP_N_ALPHA) -> pd.DataFrame:
    df = get_insider_data()

    if df is None or df.empty:
        return pd.DataFrame()

    candidates = (
        df.sort_values(["ClusterCount", "Score", "Value"], ascending=[False, False, False])
        .drop_duplicates(subset=["Ticker", "InsiderName"])
        .copy()
    )

    selected = []

    for _, row in candidates.iterrows():
        plan = build_trade_plan(row)
        if "error" in plan:
            continue

        selected.append(row.to_dict())

        if len(selected) == top_n:
            break

    if not selected:
        return pd.DataFrame()

    return pd.DataFrame(selected)


def collapse_to_unique_tickers(alpha_df: pd.DataFrame) -> pd.DataFrame:
    if alpha_df is None or alpha_df.empty:
        return pd.DataFrame()

    working = alpha_df.copy()
    working["Ticker"] = working["Ticker"].astype(str).str.strip().str.upper()

    freq = (
        working.groupby("Ticker")
        .size()
        .reset_index(name="AlphaFrequency")
    )

    best_rows = (
        working.sort_values(["Ticker", "Score", "Value"], ascending=[True, False, False])
        .drop_duplicates(subset=["Ticker"], keep="first")
        .copy()
    )

    best_rows = best_rows.merge(freq, on="Ticker", how="left")
    best_rows["AlphaFrequency"] = best_rows["AlphaFrequency"].fillna(1).astype(int)

    return best_rows


# =========================
# METRICS
# =========================
def compute_relative_metrics(ticker: str, industry_etf: str) -> dict:
    try:
        stock = download_close_series(ticker, period="3mo")
        sector = download_close_series(industry_etf, period="3mo")
        market = download_close_series(DEFAULT_BENCHMARK, period="3mo")

        if stock.empty or sector.empty or market.empty:
            return {
                "beta": None,
                "stock_return_3m": None,
                "sector_return_3m": None,
                "market_return_3m": None,
                "rs_vs_sector": None,
                "rs_vs_market": None,
                "sector_vs_market": None,
            }

        df = pd.concat([stock, sector, market], axis=1, join="inner").dropna()
        df.columns = ["stock", "sector", "market"]

        if len(df) < 20:
            return {
                "beta": None,
                "stock_return_3m": None,
                "sector_return_3m": None,
                "market_return_3m": None,
                "rs_vs_sector": None,
                "rs_vs_market": None,
                "sector_vs_market": None,
            }

        returns = df.pct_change().dropna()
        if len(returns) < 10:
            return {
                "beta": None,
                "stock_return_3m": None,
                "sector_return_3m": None,
                "market_return_3m": None,
                "rs_vs_sector": None,
                "rs_vs_market": None,
                "sector_vs_market": None,
            }

        market_var = np.var(returns["market"])
        beta = None
        if not pd.isna(market_var) and market_var != 0:
            beta = float(np.cov(returns["stock"], returns["market"])[0][1] / market_var)

        stock_return_3m = float(df["stock"].iloc[-1] / df["stock"].iloc[0] - 1.0)
        sector_return_3m = float(df["sector"].iloc[-1] / df["sector"].iloc[0] - 1.0)
        market_return_3m = float(df["market"].iloc[-1] / df["market"].iloc[0] - 1.0)

        rs_vs_sector = stock_return_3m - sector_return_3m
        rs_vs_market = stock_return_3m - market_return_3m
        sector_vs_market = sector_return_3m - market_return_3m

        return {
            "beta": beta,
            "stock_return_3m": stock_return_3m,
            "sector_return_3m": sector_return_3m,
            "market_return_3m": market_return_3m,
            "rs_vs_sector": rs_vs_sector,
            "rs_vs_market": rs_vs_market,
            "sector_vs_market": sector_vs_market,
        }

    except Exception:
        return {
            "beta": None,
            "stock_return_3m": None,
            "sector_return_3m": None,
            "market_return_3m": None,
            "rs_vs_sector": None,
            "rs_vs_market": None,
            "sector_vs_market": None,
        }


def compute_volatility(ticker: str):
    try:
        data = download_ohlc(ticker, period="2mo")
        if data.empty:
            return None

        required = ["High", "Low", "Close"]
        for col in required:
            if col not in data.columns:
                return None

        high = pd.to_numeric(data["High"], errors="coerce")
        low = pd.to_numeric(data["Low"], errors="coerce")
        close = pd.to_numeric(data["Close"], errors="coerce")

        tr = pd.concat(
            [
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)

        atr = tr.rolling(14).mean().iloc[-1]
        price = close.iloc[-1]

        if pd.isna(atr) or pd.isna(price) or price == 0:
            return None

        return float(atr / price)

    except Exception:
        return None


def compute_drawdown(ticker: str):
    try:
        close = download_close_series(ticker, period="3mo")
        if close.empty or len(close) < 20:
            return None

        rolling_high = close.rolling(60, min_periods=20).max()
        dd = (close / rolling_high) - 1.0
        latest_dd = dd.iloc[-1]

        if pd.isna(latest_dd):
            return None

        return float(latest_dd)

    except Exception:
        return None


# =========================
# SCORE OUT OF 5
# =========================
def score_buy_out_of_5(alpha_freq, rs_vs_sector, sector_vs_market, beta, vol, drawdown, mapping_confidence):
    score = 3.0

    # Alpha conviction from repeated appearance in top list
    if alpha_freq >= 4:
        score += 1.0
    elif alpha_freq == 3:
        score += 0.75
    elif alpha_freq == 2:
        score += 0.40
    else:
        score += 0.10

    # Reduce sector-based scoring if ETF mapping is weak
    sector_weight = mapping_confidence

    # Stock relative to sector
    if rs_vs_sector is not None and not pd.isna(rs_vs_sector):
        if rs_vs_sector > 0.08:
            score += 0.90 * sector_weight
        elif rs_vs_sector > 0.04:
            score += 0.60 * sector_weight
        elif rs_vs_sector > 0.00:
            score += 0.30 * sector_weight
        elif rs_vs_sector < -0.08:
            score -= 0.90 * sector_weight
        elif rs_vs_sector < -0.04:
            score -= 0.60 * sector_weight
        elif rs_vs_sector < 0.00:
            score -= 0.30 * sector_weight

    # Sector relative to market
    if sector_vs_market is not None and not pd.isna(sector_vs_market):
        if sector_vs_market > 0.05:
            score += 0.60 * sector_weight
        elif sector_vs_market > 0.00:
            score += 0.30 * sector_weight
        elif sector_vs_market < -0.05:
            score -= 0.35 * sector_weight
        elif sector_vs_market < 0.00:
            score -= 0.20 * sector_weight

    # Beta
    if beta is not None and not pd.isna(beta):
        if 0.80 <= beta <= 1.20:
            score += 0.30
        elif beta > 1.60:
            score -= 0.35
        elif beta < 0.50:
            score -= 0.15

    # Volatility
    if vol is not None and not pd.isna(vol):
        if vol < 0.025:
            score += 0.35
        elif vol < 0.05:
            score += 0.15
        elif vol > 0.09:
            score -= 0.40
        elif vol > 0.07:
            score -= 0.30
        elif vol > 0.05:
            score -= 0.15

    # Drawdown
    if drawdown is not None and not pd.isna(drawdown):
        if drawdown > -0.03:
            score += 0.35
        elif drawdown < -0.20:
            score -= 0.70
        elif drawdown < -0.10:
            score -= 0.35

    score = max(0.0, min(5.0, score))
    return round(score, 2)


# =========================
# MAIN
# =========================
def run_beta_model():
    alpha_raw = get_alpha_top_candidates(TOP_N_ALPHA)

    if alpha_raw.empty:
        print("No alpha trades found.")
        return

    alpha_df = collapse_to_unique_tickers(alpha_raw)

    if alpha_df.empty:
        print("No unique tickers found.")
        return

    results = []

    for _, row in alpha_df.iterrows():
        ticker = str(row["Ticker"]).strip().upper()
        company = row.get("Company", "")
        alpha_frequency = int(row.get("AlphaFrequency", 1))

        industry_etf, mapping_source, mapping_detail = infer_etf_from_info(ticker)
        mapping_confidence = get_mapping_confidence(mapping_source, industry_etf)

        metrics = compute_relative_metrics(ticker, industry_etf)
        beta = metrics["beta"]
        stock_return_3m = metrics["stock_return_3m"]
        sector_return_3m = metrics["sector_return_3m"]
        market_return_3m = metrics["market_return_3m"]
        rs_vs_sector = metrics["rs_vs_sector"]
        rs_vs_market = metrics["rs_vs_market"]
        sector_vs_market = metrics["sector_vs_market"]

        # If the ETF mapping is just broad-market fallback, do not pretend it is a sector comparison
        if mapping_source == "fallback_benchmark" and industry_etf == DEFAULT_BENCHMARK:
            rs_vs_sector = None
            sector_vs_market = None

        vol = compute_volatility(ticker)
        drawdown = compute_drawdown(ticker)

        buy_score_5 = score_buy_out_of_5(
            alpha_freq=alpha_frequency,
            rs_vs_sector=rs_vs_sector,
            sector_vs_market=sector_vs_market,
            beta=beta,
            vol=vol,
            drawdown=drawdown,
            mapping_confidence=mapping_confidence,
        )

        label = score_label(buy_score_5)
        decision = explain_decision(buy_score_5)

        results.append({
            "Ticker": ticker,
            "Company": company,
            "AlphaFrequency": alpha_frequency,
            "IndustryETF": industry_etf,
            "ETF_MapSource": mapping_source,
            "ETF_MapDetail": mapping_detail,
            "ETF_MapConfidence": safe_round(mapping_confidence, 2),
            "StockRet_3M": safe_round(stock_return_3m, 4),
            "SectorRet_3M": safe_round(sector_return_3m, 4),
            "MarketRet_3M": safe_round(market_return_3m, 4),
            "RS_vs_Sector": safe_round(rs_vs_sector, 4),
            "Sector_vs_Market": safe_round(sector_vs_market, 4),
            "RS_vs_Market": safe_round(rs_vs_market, 4),
            "Beta": safe_round(beta, 2),
            "Volatility": safe_round(vol, 4),
            "Drawdown_60D": safe_round(drawdown, 4),
            "BuyScore_5": buy_score_5,
            "Label": label,
            "Decision": decision,
        })

    result_df = pd.DataFrame(results)

    if result_df.empty:
        print("No beta results generated.")
        return

    sort_rs = "RS_vs_Sector" if "RS_vs_Sector" in result_df.columns else "BuyScore_5"
    result_df = result_df.sort_values(
        ["BuyScore_5", "AlphaFrequency", sort_rs],
        ascending=[False, False, False],
        na_position="last",
    )

    final_df = result_df[["Ticker", "BuyScore_5"]].copy()

    # Sort best → worst
    final_df = final_df.sort_values("BuyScore_5", ascending=False)

    print("\nFINAL TRADE SCORES:\n")
    print(final_df.to_string(index=False))

def get_trade_scores():
    alpha_raw = get_alpha_top_candidates(TOP_N_ALPHA)

    if alpha_raw.empty:
        return pd.DataFrame(columns=["Ticker", "BuyScore_5"])

    alpha_df = collapse_to_unique_tickers(alpha_raw)

    if alpha_df.empty:
        return pd.DataFrame(columns=["Ticker", "BuyScore_5"])

    results = []

    for _, row in alpha_df.iterrows():
        ticker = str(row["Ticker"]).strip().upper()
        company = row.get("Company", "")
        alpha_frequency = int(row.get("AlphaFrequency", 1))

        industry_etf, mapping_source, mapping_detail = infer_etf_from_info(ticker)
        mapping_confidence = get_mapping_confidence(mapping_source, industry_etf)

        metrics = compute_relative_metrics(ticker, industry_etf)
        beta = metrics["beta"]
        stock_return_3m = metrics["stock_return_3m"]
        sector_return_3m = metrics["sector_return_3m"]
        market_return_3m = metrics["market_return_3m"]
        rs_vs_sector = metrics["rs_vs_sector"]
        rs_vs_market = metrics["rs_vs_market"]
        sector_vs_market = metrics["sector_vs_market"]

        if mapping_source == "fallback_benchmark" and industry_etf == DEFAULT_BENCHMARK:
            rs_vs_sector = None
            sector_vs_market = None

        vol = compute_volatility(ticker)
        drawdown = compute_drawdown(ticker)

        buy_score_5 = score_buy_out_of_5(
            alpha_freq=alpha_frequency,
            rs_vs_sector=rs_vs_sector,
            sector_vs_market=sector_vs_market,
            beta=beta,
            vol=vol,
            drawdown=drawdown,
            mapping_confidence=mapping_confidence,
        )

        results.append({
            "Ticker": ticker,
            "BuyScore_5": buy_score_5,
        })

    result_df = pd.DataFrame(results)

    if result_df.empty:
        return pd.DataFrame(columns=["Ticker", "BuyScore_5"])

    final_df = result_df[["Ticker", "BuyScore_5"]].copy()
    final_df = final_df.sort_values("BuyScore_5", ascending=False).reset_index(drop=True)
    return final_df

if __name__ == "__main__":
    run_beta_model()