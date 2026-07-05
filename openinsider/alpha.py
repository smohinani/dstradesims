import yfinance as yf
import pandas as pd
import numpy as np
import smtplib
from email.message import EmailMessage
from pandas.tseries.offsets import BDay
from sec_insider_fetcher import get_insider_data


# =========================
# CONFIG
# =========================
FROM_EMAIL = "samuelmohinanihkg@gmail.com"
APP_PASSWORD = "jccn gigr yebd aycl"
RECIPIENT_LIST = ["samuelmohinanihkg@gmail.com"]


# =========================
# HELPERS
# =========================
def get_next_trading_day():
    return (pd.Timestamp.today() + BDay(1)).strftime("%Y-%m-%d")


def get_price_data(ticker: str):
    df = yf.download(
        ticker,
        period="6mo",
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(how="all")

    required = ["Open", "High", "Low", "Close", "Volume"]
    for col in required:
        if col not in df.columns:
            return pd.DataFrame()

    return df


def calculate_atr(data, period=14):
    if data.empty or len(data) < period + 2:
        return np.nan

    high = data["High"]
    low = data["Low"]
    close = data["Close"]

    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.rolling(period).mean().iloc[-1]


# =========================
# TRADE PLAN (FILINGS ONLY)
# =========================
_PLAN_CACHE: dict = {}


def build_trade_plan(row):
    ticker = str(row["Ticker"]).strip().upper()

    if ticker in _PLAN_CACHE:
        return _PLAN_CACHE[ticker]

    data = get_price_data(ticker)

    if data.empty or len(data) < 30:
        _PLAN_CACHE[ticker] = {"error": f"Not enough data for {ticker}"}
        return _PLAN_CACHE[ticker]

    close = float(data["Close"].iloc[-1])
    atr = calculate_atr(data)

    if pd.isna(atr) or atr <= 0:
        atr = close * 0.05

    value = float(row.get("Value", 0))
    delta = float(row.get("DeltaOwn", 0))
    title = str(row.get("Title", "")).lower()
    cluster = int(row.get("ClusterCount", 1))

    conviction = 1.0

    if value > 5_000_000:
        conviction += 0.30
    elif value > 1_000_000:
        conviction += 0.20
    elif value > 500_000:
        conviction += 0.10

    if delta > 50:
        conviction += 0.20
    elif delta > 20:
        conviction += 0.10
    elif delta > 10:
        conviction += 0.05

    if cluster > 2:
        conviction += 0.15
    elif cluster > 1:
        conviction += 0.10

    if "ceo" in title or "chair" in title:
        conviction += 0.10
    elif "cfo" in title or "president" in title:
        conviction += 0.08

    entry = close * 0.98
    stop = max(close - 1.2 * atr, 0.01)
    risk = max(close - stop, 0.01)

    tp1 = close + risk * 2 * conviction
    tp2 = close + risk * 4 * conviction

    plan = {
        "entry": round(entry, 2),
        "stop": round(stop, 2),
        "tp1": round(tp1, 2),
        "tp2": round(tp2, 2),
        "upside1": round((tp1 / close - 1) * 100, 1),
        "upside2": round((tp2 / close - 1) * 100, 1),
    }
    _PLAN_CACHE[ticker] = plan
    return plan


# =========================
# EMAIL FORMAT
# =========================
def format_email(df):
    if df.empty:
        return "No insider buys today."

    df = df.copy()

    candidates = (
        df.sort_values(["ClusterCount", "Score", "Value"], ascending=[False, False, False])
        .drop_duplicates(subset=["Ticker", "InsiderName"])
        .copy()
    )

    out = []
    out.append("Morning Insider Scan")
    out.append(f"Focus for next session: {get_next_trading_day()}")
    out.append("")
    out.append("TOP 10 BEST TRADES")
    out.append("")

    added = 0
    for _, row in candidates.iterrows():
        plan = build_trade_plan(row)
        if "error" in plan:
            continue

        ticker_link = f"http://openinsider.com/search?q={str(row['Ticker']).lower()}"

        out.append(f"{row['Ticker']} | {row['Company']}")
        out.append(f"Insider: {row['InsiderName']}")
        out.append(f"Title: {row['Title']}")
        out.append(f"OpenInsider Link: {ticker_link}")
        out.append(
            f"Buy Value: ${float(row['Value']):,.0f} | ΔOwn: {float(row.get('DeltaOwn', 0)):.1f}% | Cluster: {int(row.get('ClusterCount', 1))}"
        )

        if "VerifiedPrice" in row.index and pd.notna(row["VerifiedPrice"]):
            out.append(
                f"Verified Row: {row.get('VerifiedInsiderName', '')} | {row.get('VerifiedTitle', '')} | "
                f"{row.get('VerifiedTradeType', '')} | "
                f"Price: {float(row.get('VerifiedPrice', 0)):.2f} | "
                f"Qty: {int(float(row.get('VerifiedQty', 0))):,} | "
                f"Value: ${float(row.get('VerifiedValue', 0)):,.0f}"
            )

        out.append(f"Entry: {plan['entry']}")
        out.append(f"SL: {plan['stop']}")
        out.append(f"TP1: {plan['tp1']} ({plan['upside1']}%)")
        out.append(f"TP2: {plan['tp2']} ({plan['upside2']}%)")
        out.append("-" * 50)

        added += 1
        if added == 10:
            break

    if added == 0:
        return "No insider buys with valid trade plans found."

    return "\n".join(out)

# =========================
# EMAIL SENDER
# =========================
def send_mail(subject, message):
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = FROM_EMAIL
    msg["To"] = ", ".join(RECIPIENT_LIST) if isinstance(RECIPIENT_LIST, list) else RECIPIENT_LIST
    msg.set_content(message)

    smtp = smtplib.SMTP("smtp.gmail.com", 587)
    smtp.starttls()
    smtp.login(FROM_EMAIL, APP_PASSWORD)
    smtp.send_message(msg)
    smtp.quit()


def verify_ticker_page(ticker, insider_name=None):
    url = f"http://openinsider.com/search?q={str(ticker).lower()}"

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/123.0.0.0 Safari/537.36"
        )
    }

    try:
        response = requests.get(url, headers=headers, timeout=(5, 12))
        response.raise_for_status()
        tables = pd.read_html(StringIO(response.text))
    except Exception:
        return None

    best_df = None
    best_score = -1

    for t in tables:
        cols = [str(c).replace("\xa0", " ").strip().lower() for c in t.columns]
        score = 0
        wanted = ["filing date", "trade date", "ticker", "insider", "title", "trade type", "price", "qty", "owned", "value"]
        for k in wanted:
            if any(k in c for c in cols):
                score += 1
        if score > best_score:
            best_score = score
            best_df = t.copy()

    if best_df is None:
        return None

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

    for col in ["Ticker", "InsiderName", "Title", "TradeType"]:
        if col not in best_df.columns:
            best_df[col] = ""

    for col in ["Price", "Qty", "Owned", "DeltaOwn", "Value"]:
        if col not in best_df.columns:
            best_df[col] = np.nan

    best_df["Price"] = pd.to_numeric(best_df["Price"].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False), errors="coerce")
    best_df["Qty"] = pd.to_numeric(best_df["Qty"].astype(str).str.replace("+", "", regex=False).str.replace(",", "", regex=False), errors="coerce")
    best_df["Owned"] = pd.to_numeric(best_df["Owned"].astype(str).str.replace(",", "", regex=False), errors="coerce")
    best_df["DeltaOwn"] = pd.to_numeric(
        best_df["DeltaOwn"].astype(str)
        .str.replace("%", "", regex=False)
        .str.replace("+", "", regex=False)
        .replace({"New": "100", "new": "100"}),
        errors="coerce"
    )
    best_df["Value"] = pd.to_numeric(best_df["Value"].astype(str).str.replace("$", "", regex=False).str.replace("+", "", regex=False).str.replace(",", "", regex=False), errors="coerce")

    best_df["TradeType"] = best_df["TradeType"].astype(str).str.strip()
    best_df = best_df[best_df["TradeType"].str.contains("P", na=False)].copy()

    if best_df.empty:
        return None

    if insider_name:
        match_df = best_df[
            best_df["InsiderName"].astype(str).str.lower().str.contains(str(insider_name).lower(), na=False)
        ].copy()
        if not match_df.empty:
            best_df = match_df

    best_df = best_df.sort_values(["TradeDate", "Value"], ascending=[False, False], na_position="last")
    return best_df.iloc[0].to_dict()


def add_verification_columns(df):
    df = df.copy()

    verified_rows = []
    for _, row in df.iterrows():
        verified = verify_ticker_page(row["Ticker"], row.get("InsiderName", None))
        verified_rows.append(verified if verified is not None else {})

    verified_df = pd.DataFrame(verified_rows).add_prefix("Verified")
    return pd.concat([df.reset_index(drop=True), verified_df.reset_index(drop=True)], axis=1)

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    try:
        df = get_insider_data()
        df = add_verification_columns(df)
        email_body = format_email(df)

        send_mail(
            subject=f"Morning Insider Signals - {pd.Timestamp.today().strftime('%Y-%m-%d')}",
            message=email_body
        )
        print("Email sent successfully.")

    except Exception as e:
        send_mail(
            subject="Insider Bot Error",
            message=str(e)
        )
        print(f"Error occurred: {repr(e)}")