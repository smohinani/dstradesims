import math
import os
import sys
import time
import importlib
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple, Dict

import pandas as pd
from dotenv import load_dotenv

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    LimitOrderRequest,
    StopLossRequest,
    TakeProfitRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus, OrderClass
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestTradeRequest, StockLatestQuoteRequest

from alpha import build_trade_plan
from beta import get_alpha_top_candidates, collapse_to_unique_tickers


SCORING_MODULE = "beta"
SCORING_FUNCTION = "get_trade_scores"

load_dotenv()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")
ALPACA_PAPER = os.getenv("ALPACA_PAPER", "true").lower() == "true"

MAX_POSITION_PCT = float(os.getenv("MAX_POSITION_PCT", "0.20"))
ALLOW_FRACTIONAL = os.getenv("ALLOW_FRACTIONAL", "false").lower() == "true"
MIN_SCORE_TO_TRADE = float(os.getenv("MIN_SCORE_TO_TRADE", "2.5"))

POSITIONS_CSV = os.getenv(
    "POSITIONS_CSV",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "open_positions.csv"),
)
MAX_HOLD_DAYS = int(os.getenv("MAX_HOLD_DAYS", "30"))

ENTRY_FILL_TIMEOUT_SEC = int(os.getenv("ENTRY_FILL_TIMEOUT_SEC", "60"))
ENTRY_POLL_INTERVAL_SEC = float(os.getenv("ENTRY_POLL_INTERVAL_SEC", "2.0"))

if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
    raise ValueError(
        "Missing Alpaca keys in .env. Need ALPACA_API_KEY / ALPACA_SECRET_KEY "
        "or APCA_API_KEY_ID / APCA_API_SECRET_KEY."
    )

trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=ALPACA_PAPER)
data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

TRACKER_COLUMNS = [
    "Ticker",
    "OpenedAt",
    "Status",
    "EntryPrice",
    "InitialQty",
    "CurrentQty",
    "StopPrice",
    "TP1Price",
    "TP2Price",
    "TP1Qty",
    "TP2Qty",
    "LastSeenAt",
    "ClosedAt",
    "ExitPrice",
    "PnL",
    "ExitReason",
]


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def dt_to_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def parse_dt(value) -> Optional[datetime]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    text = text.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def load_tracker() -> pd.DataFrame:
    if not os.path.exists(POSITIONS_CSV):
        return pd.DataFrame(columns=TRACKER_COLUMNS)

    try:
        df = pd.read_csv(POSITIONS_CSV)
    except Exception:
        return pd.DataFrame(columns=TRACKER_COLUMNS)

    for col in TRACKER_COLUMNS:
        if col not in df.columns:
            df[col] = None

    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    df["Status"] = df["Status"].astype(str).str.upper().str.strip()
    return df[TRACKER_COLUMNS].copy()


def save_tracker(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        pd.DataFrame(columns=TRACKER_COLUMNS).to_csv(POSITIONS_CSV, index=False)
        return

    out = df.copy()
    for col in TRACKER_COLUMNS:
        if col not in out.columns:
            out[col] = None

    out = out[TRACKER_COLUMNS].copy()
    # OPEN rows first, then CLOSED; within each group sort by OpenedAt
    out["_sort_status"] = (out["Status"].astype(str).str.upper().str.strip() != "OPEN").astype(int)
    out = out.sort_values(["_sort_status", "OpenedAt"], ascending=[True, True])
    out = out.drop(columns=["_sort_status"])
    out.to_csv(POSITIONS_CSV, index=False)


def tracker_has_open_position(symbol: str, tracker: pd.DataFrame) -> bool:
    if tracker is None or tracker.empty:
        return False

    symbol = symbol.upper().strip()
    mask = (
        (tracker["Ticker"] == symbol) &
        (tracker["Status"].astype(str).str.upper().str.strip() == "OPEN")
    )
    return bool(mask.any())


def upsert_open_tracker_row(
    tracker: pd.DataFrame,
    symbol: str,
    opened_at: datetime,
    entry_price: float,
    initial_qty: float,
    current_qty: float,
    stop_price: float,
    tp1_price: float,
    tp2_price: float,
    tp1_qty: float,
    tp2_qty: float,
) -> pd.DataFrame:
    symbol = symbol.upper().strip()
    opened_at_iso = dt_to_iso(opened_at)
    now_iso = dt_to_iso(now_utc())

    new_row = {
        "Ticker": symbol,
        "OpenedAt": opened_at_iso,
        "Status": "OPEN",
        "EntryPrice": round(float(entry_price), 4),
        "InitialQty": round(float(initial_qty), 4),
        "CurrentQty": round(float(current_qty), 4),
        "StopPrice": round(float(stop_price), 4),
        "TP1Price": round(float(tp1_price), 4),
        "TP2Price": round(float(tp2_price), 4),
        "TP1Qty": round(float(tp1_qty), 4),
        "TP2Qty": round(float(tp2_qty), 4),
        "LastSeenAt": now_iso,
    }

    if tracker is None or tracker.empty:
        return pd.DataFrame([new_row], columns=TRACKER_COLUMNS)

    tracker = tracker.copy()
    tracker = tracker[tracker["Ticker"] != symbol].copy()
    tracker = pd.concat([tracker, pd.DataFrame([new_row])], ignore_index=True)
    return tracker


def rebuild_tracker_from_alpaca(previous_tracker: pd.DataFrame) -> pd.DataFrame:
    now_iso = dt_to_iso(now_utc())

    try:
        positions = trading_client.get_all_positions()
    except Exception as e:
        print(f"[ERROR] rebuild_tracker_from_alpaca: get_all_positions() failed: {e}")
        return previous_tracker if previous_tracker is not None else pd.DataFrame(columns=TRACKER_COLUMNS)

    rows = []

    for p in positions:
        symbol = str(p.symbol).upper().strip()
        qty = abs(float(p.qty))
        avg_entry_price = float(p.avg_entry_price)

        existing = pd.DataFrame()
        if previous_tracker is not None and not previous_tracker.empty:
            existing = previous_tracker[previous_tracker["Ticker"] == symbol]

        if not existing.empty:
            row0 = existing.iloc[0]
            opened_at = row0.get("OpenedAt", now_iso)
            stop_price = row0.get("StopPrice", "")
            tp1_price = row0.get("TP1Price", "")
            tp2_price = row0.get("TP2Price", "")
            tp1_qty = row0.get("TP1Qty", "")
            tp2_qty = row0.get("TP2Qty", "")
            initial_qty = row0.get("InitialQty", qty)
        else:
            opened_at = now_iso
            stop_price = ""
            tp1_price = ""
            tp2_price = ""
            tp1_qty = ""
            tp2_qty = ""
            initial_qty = qty

        rows.append({
            "Ticker": symbol,
            "OpenedAt": opened_at,
            "Status": "OPEN",
            "EntryPrice": avg_entry_price,
            "InitialQty": initial_qty,
            "CurrentQty": qty,
            "StopPrice": stop_price,
            "TP1Price": tp1_price,
            "TP2Price": tp2_price,
            "TP1Qty": tp1_qty,
            "TP2Qty": tp2_qty,
            "LastSeenAt": now_iso,
        })

    result = pd.DataFrame(rows, columns=TRACKER_COLUMNS)

    # Preserve closed/historical rows from previous tracker
    if previous_tracker is not None and not previous_tracker.empty:
        closed_rows = previous_tracker[
            previous_tracker["Status"].astype(str).str.upper().str.strip() != "OPEN"
        ].copy()
        if not closed_rows.empty:
            for col in TRACKER_COLUMNS:
                if col not in closed_rows.columns:
                    closed_rows[col] = None
            result = pd.concat([result, closed_rows[TRACKER_COLUMNS]], ignore_index=True)

    save_tracker(result)
    return result


def load_scores_from_old_function() -> pd.DataFrame:
    try:
        module = importlib.import_module(SCORING_MODULE)
    except Exception as e:
        raise ImportError(
            f"Could not import module '{SCORING_MODULE}'. "
            f"Make sure execution.py is in the same folder and the filename matches.\n"
            f"Import error: {e}"
        )

    if not hasattr(module, SCORING_FUNCTION):
        available = [x for x in dir(module) if not x.startswith("_")]
        raise AttributeError(
            f"Module '{SCORING_MODULE}' does not have function '{SCORING_FUNCTION}'.\n"
            f"Available names: {available}"
        )

    func = getattr(module, SCORING_FUNCTION)
    if not callable(func):
        raise TypeError(f"'{SCORING_FUNCTION}' exists but is not callable.")

    result = func()

    if isinstance(result, pd.DataFrame):
        df = result.copy()
    elif isinstance(result, list):
        df = pd.DataFrame(result)
    elif isinstance(result, dict):
        df = pd.DataFrame([{"Ticker": k, "BuyScore_5": v} for k, v in result.items()])
    else:
        raise TypeError(
            f"Unsupported return type from {SCORING_FUNCTION}(): {type(result)}\n"
            f"Return a DataFrame, a list of dicts, or a dict."
        )

    rename_map = {}
    cols_lower = {str(c).lower(): c for c in df.columns}

    if "ticker" not in cols_lower:
        raise ValueError(f"Your scoring output must include a ticker column. Current columns: {list(df.columns)}")

    if "buyscore_5" not in cols_lower:
        possible_score_cols = [
            "score", "buyscore", "buy_score", "score_5", "finalscore", "final_score"
        ]
        found_score = None
        for c in possible_score_cols:
            if c in cols_lower:
                found_score = cols_lower[c]
                break
        if found_score is None:
            raise ValueError(
                f"Your scoring output must include BuyScore_5 or similar. Current columns: {list(df.columns)}"
            )
        rename_map[found_score] = "BuyScore_5"

    rename_map[cols_lower["ticker"]] = "Ticker"
    df = df.rename(columns=rename_map)

    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    df["BuyScore_5"] = pd.to_numeric(df["BuyScore_5"], errors="coerce")
    df = df.dropna(subset=["Ticker", "BuyScore_5"])
    df = df[df["Ticker"] != ""]

    if df.empty:
        raise ValueError("Scoring function returned no valid rows after cleaning.")

    return df[["Ticker", "BuyScore_5"]].sort_values("BuyScore_5", ascending=False).reset_index(drop=True)


def load_alpha_plan_map() -> Dict[str, dict]:
    alpha_raw = get_alpha_top_candidates()

    if alpha_raw is None or alpha_raw.empty:
        return {}

    alpha_unique = collapse_to_unique_tickers(alpha_raw)
    if alpha_unique is None or alpha_unique.empty:
        return {}

    plan_map = {}

    for _, row in alpha_unique.iterrows():
        ticker = str(row["Ticker"]).strip().upper()
        plan = build_trade_plan(row)

        if isinstance(plan, dict) and "error" not in plan:
            plan_map[ticker] = plan

    return plan_map


def get_latest_price(symbol: str) -> Optional[float]:
    try:
        req = StockLatestTradeRequest(symbol_or_symbols=symbol)
        resp = data_client.get_stock_latest_trade(req)
        if symbol in resp and resp[symbol].price is not None:
            return float(resp[symbol].price)
    except Exception:
        pass

    try:
        req = StockLatestQuoteRequest(symbol_or_symbols=symbol)
        resp = data_client.get_stock_latest_quote(req)
        if symbol in resp:
            bid = resp[symbol].bid_price
            ask = resp[symbol].ask_price
            if bid is not None and ask is not None and bid > 0 and ask > 0:
                return float((bid + ask) / 2)
            if ask is not None and ask > 0:
                return float(ask)
            if bid is not None and bid > 0:
                return float(bid)
    except Exception:
        pass

    return None


def get_account_equity_and_buying_power() -> Tuple[float, float]:
    acct = trading_client.get_account()
    return float(acct.equity), float(acct.buying_power)


def score_to_risk_budget_pct(score: float) -> float:
    if score >= 4.0:
        return 0.02
    if score >= 2.5:
        return 0.01
    return 0.0


def calc_position_size_from_alpha_stop(
    equity: float,
    buying_power: float,
    current_price: float,
    stop_price: float,
    risk_budget_pct: float,
):
    if current_price <= 0 or stop_price <= 0 or risk_budget_pct <= 0:
        return 0

    risk_per_share = current_price - stop_price
    if risk_per_share <= 0:
        return 0

    risk_dollars = equity * risk_budget_pct
    shares_by_risk = risk_dollars / risk_per_share
    shares_by_cap = (equity * MAX_POSITION_PCT) / current_price
    shares_by_bp = buying_power / current_price

    shares = min(shares_by_risk, shares_by_cap, shares_by_bp)

    if ALLOW_FRACTIONAL:
        return round(shares, 4)

    return math.floor(shares)


def split_qty(total_qty: float) -> Tuple[float, float]:
    if ALLOW_FRACTIONAL:
        tp1_qty = round(total_qty * 0.80, 4)
        tp2_qty = round(total_qty - tp1_qty, 4)
        if tp1_qty <= 0:
            tp1_qty = 0
        if tp2_qty <= 0:
            tp2_qty = 0
        return tp1_qty, tp2_qty

    total_qty = int(total_qty)

    if total_qty <= 1:
        return total_qty, 0

    tp1_qty = math.floor(total_qty * 0.80)
    tp2_qty = total_qty - tp1_qty

    if tp1_qty <= 0:
        tp1_qty = 1
        tp2_qty = total_qty - 1

    return tp1_qty, tp2_qty


def round_price(price: float) -> float:
    if price < 1:
        return round(price, 4)
    return round(price, 2)


def has_open_position(symbol: str) -> bool:
    try:
        positions = trading_client.get_all_positions()
        return any(str(p.symbol).upper() == symbol.upper() for p in positions)
    except Exception:
        return False


def has_open_order(symbol: str) -> bool:
    try:
        orders = trading_client.get_orders(filter=QueryOrderStatus.OPEN)
        return any(str(o.symbol).upper() == symbol.upper() for o in orders)
    except Exception:
        return False


def cancel_open_orders_for_symbol(symbol: str) -> None:
    try:
        orders = trading_client.get_orders(filter=QueryOrderStatus.OPEN)
    except Exception:
        return

    for o in orders:
        if str(o.symbol).upper() == symbol.upper():
            try:
                trading_client.cancel_order_by_id(o.id)
                print(f"[CANCELLED OPEN ORDER] {symbol} | order_id={o.id}")
            except Exception as e:
                print(f"[WARN] Could not cancel open order for {symbol}: {e}")


def wait_for_fill(order_id: str, timeout_sec: int = ENTRY_FILL_TIMEOUT_SEC) -> Tuple[Optional[float], Optional[float], Optional[datetime], str]:
    deadline = time.time() + timeout_sec
    last_status = "unknown"

    while time.time() < deadline:
        try:
            order = trading_client.get_order_by_id(order_id)
            raw_status = getattr(order, "status", None)
            last_status = str(raw_status).lower()

            filled_qty = getattr(order, "filled_qty", None)
            filled_avg_price = getattr(order, "filled_avg_price", None)
            filled_at = getattr(order, "filled_at", None)

            if filled_qty is not None and float(filled_qty) > 0 and filled_avg_price is not None:
                return float(filled_qty), float(filled_avg_price), parse_dt(filled_at) or now_utc(), last_status

            if any(x in last_status for x in ["canceled", "cancelled", "rejected", "expired"]):
                return None, None, None, last_status

        except Exception:
            pass

        time.sleep(ENTRY_POLL_INTERVAL_SEC)

    return None, None, None, last_status


def submit_entry_order(symbol: str, qty: float):
    order = MarketOrderRequest(
        symbol=symbol,
        qty=qty if ALLOW_FRACTIONAL else int(qty),
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY,
    )
    return trading_client.submit_order(order_data=order)


def submit_oco_exit(symbol: str, qty: float, tp_price: float, stop_price: float):
    if qty <= 0:
        return None

    tp_price = round_price(float(tp_price))
    stop_price = round_price(float(stop_price))

    order = LimitOrderRequest(
        symbol=symbol,
        qty=qty if ALLOW_FRACTIONAL else int(qty),
        side=OrderSide.SELL,
        time_in_force=TimeInForce.GTC,
        limit_price=tp_price,
        order_class=OrderClass.OCO,
        take_profit=TakeProfitRequest(limit_price=tp_price),
        stop_loss=StopLossRequest(stop_price=stop_price),
    )
    return trading_client.submit_order(order_data=order)


def submit_entry_and_exits(symbol: str, qty: float, plan: dict, tracker: pd.DataFrame) -> pd.DataFrame:
    if qty <= 0:
        print(f"[SKIP] {symbol}: qty <= 0")
        return tracker

    stop_price = float(plan["stop"])
    tp1_price = float(plan["tp1"])
    tp2_price = float(plan["tp2"])

    print(
        f"[ENTRY ORDER] {symbol} | qty={qty} | "
        f"market buy | stop={stop_price:.2f} | tp1={tp1_price:.2f} | tp2={tp2_price:.2f}"
    )

    entry_resp = submit_entry_order(symbol, qty)
    print(f"[ENTRY SUBMITTED] {symbol} | order_id={entry_resp.id}")

    filled_qty, filled_avg_price, filled_at, status = wait_for_fill(str(entry_resp.id))

    if not filled_qty or not filled_avg_price:
        print(f"[WARN] {symbol}: entry not confirmed filled. last_status={status}")
        return tracker

    tp1_qty, tp2_qty = split_qty(filled_qty)

    print(
        f"[FILLED] {symbol} | qty={filled_qty} | avg_fill={filled_avg_price:.2f} | "
        f"tp1_qty={tp1_qty} | tp2_qty={tp2_qty}"
    )

    if tp1_qty > 0:
        o1 = submit_oco_exit(symbol, tp1_qty, tp1_price, stop_price)
        print(
            f"[TP1 OCO SUBMITTED] {symbol} | order_id={o1.id} | "
            f"qty={tp1_qty} | tp={tp1_price:.2f} | stop={stop_price:.2f}"
        )

    if tp2_qty > 0:
        o2 = submit_oco_exit(symbol, tp2_qty, tp2_price, stop_price)
        print(
            f"[TP2 OCO SUBMITTED] {symbol} | order_id={o2.id} | "
            f"qty={tp2_qty} | tp={tp2_price:.2f} | stop={stop_price:.2f}"
        )

    tracker = upsert_open_tracker_row(
        tracker=tracker,
        symbol=symbol,
        opened_at=filled_at or now_utc(),
        entry_price=filled_avg_price,
        initial_qty=filled_qty,
        current_qty=filled_qty,
        stop_price=stop_price,
        tp1_price=tp1_price,
        tp2_price=tp2_price,
        tp1_qty=tp1_qty,
        tp2_qty=tp2_qty,
    )
    save_tracker(tracker)

    return tracker


def close_stale_positions(tracker: pd.DataFrame) -> pd.DataFrame:
    if tracker is None or tracker.empty:
        return tracker

    cutoff = now_utc() - timedelta(days=MAX_HOLD_DAYS)

    current_positions = {}
    try:
        for p in trading_client.get_all_positions():
            current_positions[str(p.symbol).upper()] = p
    except Exception:
        return tracker

    tracker = tracker.copy()

    for idx, row in tracker.iterrows():
        symbol = str(row["Ticker"]).upper().strip()
        status = str(row.get("Status", "")).upper().strip()

        if status != "OPEN":
            continue

        if symbol not in current_positions:
            continue

        opened_at = parse_dt(row.get("OpenedAt"))
        if opened_at is None:
            continue

        if opened_at <= cutoff:
            print(f"[STALE CLOSE] {symbol} | opened_at={row.get('OpenedAt')} | > {MAX_HOLD_DAYS} days")
            try:
                cancel_open_orders_for_symbol(symbol)
                trading_client.close_position(symbol)
            except Exception as e:
                print(f"[ERROR] Could not close stale position {symbol}: {e}")

    time.sleep(1)
    tracker = rebuild_tracker_from_alpaca(tracker)
    return tracker


def main():
    tracker = load_tracker()
    tracker = rebuild_tracker_from_alpaca(tracker)
    tracker = close_stale_positions(tracker)

    df = load_scores_from_old_function()
    alpha_plan_map = load_alpha_plan_map()

    # Fill in missing stop/tp for open positions that were entered in a prior run
    tracker_updated = False
    for idx, row in tracker.iterrows():
        if str(row.get("Status", "")).upper().strip() != "OPEN":
            continue
        symbol = str(row["Ticker"]).upper().strip()
        if str(row.get("StopPrice", "")).strip() in ("", "nan"):
            plan = alpha_plan_map.get(symbol)
            if plan:
                tracker.at[idx, "StopPrice"] = round(float(plan["stop"]), 4)
                tracker.at[idx, "TP1Price"] = round(float(plan["tp1"]), 4)
                tracker.at[idx, "TP2Price"] = round(float(plan["tp2"]), 4)
                tracker_updated = True
                print(f"[FILL] {symbol}: restored stop/tp1/tp2 from alpha plan")
    if tracker_updated:
        save_tracker(tracker)

    equity, buying_power = get_account_equity_and_buying_power()

    print("=" * 90)
    print("ALPACA EXECUTION START")
    print(f"Paper Trading:   {ALPACA_PAPER}")
    print(f"Equity:          ${equity:,.2f}")
    print(f"Buying Power:    ${buying_power:,.2f}")
    print(f"Scoring From:    {SCORING_MODULE}.{SCORING_FUNCTION}()")
    print(f"Rows Loaded:     {len(df)}")
    print(f"Max Pos %:       {MAX_POSITION_PCT:.1%}")
    print(f"Tracker CSV:     {POSITIONS_CSV}")
    print(f"Max Hold Days:   {MAX_HOLD_DAYS}")
    print("=" * 90)
    print(df.to_string(index=False))
    print("=" * 90)

    for _, row in df.iterrows():
        symbol = str(row["Ticker"]).upper().strip()
        score = float(row["BuyScore_5"])

        if score < MIN_SCORE_TO_TRADE:
            print(f"[IGNORE] {symbol}: score={score:.2f} < {MIN_SCORE_TO_TRADE}")
            continue

        if tracker_has_open_position(symbol, tracker):
            print(f"[SKIP] {symbol}: already marked OPEN in {POSITIONS_CSV}")
            continue

        if has_open_position(symbol):
            print(f"[SKIP] {symbol}: already open at Alpaca")
            continue

        if has_open_order(symbol):
            print(f"[SKIP] {symbol}: already has open order")
            continue

        plan = alpha_plan_map.get(symbol)
        if not plan:
            print(f"[SKIP] {symbol}: no alpha trade plan found")
            continue

        price = get_latest_price(symbol)
        if price is None or price <= 0:
            print(f"[SKIP] {symbol}: could not fetch valid price")
            continue

        stop_price = float(plan["stop"])
        tp1_price = float(plan["tp1"])
        tp2_price = float(plan["tp2"])

        if stop_price >= price - 0.01:
            print(f"[SKIP] {symbol}: alpha stop too close to or above current price")
            continue

        if tp1_price <= stop_price or tp2_price <= tp1_price:
            print(f"[SKIP] {symbol}: invalid alpha tp/stop structure")
            continue

        risk_budget_pct = score_to_risk_budget_pct(score)
        if risk_budget_pct <= 0:
            print(f"[IGNORE] {symbol}: invalid risk bucket")
            continue

        qty = calc_position_size_from_alpha_stop(
            equity=equity,
            buying_power=buying_power,
            current_price=price,
            stop_price=stop_price,
            risk_budget_pct=risk_budget_pct,
        )

        if qty <= 0:
            print(f"[SKIP] {symbol}: calculated qty <= 0")
            continue

        est_cost = qty * price
        tp1_qty, tp2_qty = split_qty(qty)
        est_risk = qty * max(price - stop_price, 0)

        print(
            f"[READY] {symbol} | "
            f"score={score:.2f} | "
            f"price={price:.2f} | "
            f"qty={qty} | "
            f"position≈${est_cost:,.2f} | "
            f"risk≈${est_risk:,.2f} | "
            f"stop={stop_price:.2f} | "
            f"tp1={tp1_price:.2f} ({tp1_qty}) | "
            f"tp2={tp2_price:.2f} ({tp2_qty})"
        )

        try:
            tracker = submit_entry_and_exits(symbol, qty, plan, tracker)
            tracker = rebuild_tracker_from_alpaca(tracker)

            buying_power -= est_cost
            if buying_power <= 0:
                print("[INFO] Buying power exhausted, stopping.")
                break

        except Exception as e:
            print(f"[ERROR] {symbol}: {e}")

    tracker = rebuild_tracker_from_alpaca(load_tracker())

    print("=" * 90)
    print("ALL POSITIONS CSV UPDATED")
    if tracker.empty:
        print("(no positions)")
    else:
        print(tracker.to_string(index=False))
    print("=" * 90)
    print("DONE")
    print("=" * 90)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nFatal error: {e}")
        sys.exit(1)