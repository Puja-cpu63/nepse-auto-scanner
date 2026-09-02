import os
import time
import argparse
from datetime import date, datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from nepse_data_api import Nepse


# =========================
# Daily Current-Data Scanner
# =========================

MIN_HISTORY = 220
MIN_AVG_VALUE = 1_000_000
SLEEP_BETWEEN_REQUESTS = 0.10


def first_value(row: Dict[str, Any], keys: List[str], default=np.nan):
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return default


def normalize_history(raw: Any) -> pd.DataFrame:
    """Convert common NEPSE API history formats into OHLCV dataframe."""
    if raw is None:
        return pd.DataFrame()

    if isinstance(raw, dict):
        for key in ("data", "content", "items", "result", "records"):
            if isinstance(raw.get(key), list):
                raw = raw[key]
                break

    if not isinstance(raw, list):
        return pd.DataFrame()

    rows = []
    for x in raw:
        if not isinstance(x, dict):
            continue

        dt = first_value(
            x,
            ["date", "businessDate", "tradeDate", "timestamp", "time", "t"]
        )
        close = first_value(
            x,
            ["close", "closePrice", "closingPrice", "ltp", "lastPrice"]
        )
        high = first_value(x, ["high", "highPrice"])
        low = first_value(x, ["low", "lowPrice"])
        volume = first_value(
            x,
            ["volume", "totalTradedQuantity", "totalTradeQuantity"]
        )
        value = first_value(
            x,
            ["value", "turnover", "totalTradedValue", "totalTradeValue"]
        )

        if dt is None or pd.isna(close):
            continue

        rows.append(
            {
                "Date": dt,
                "Close": close,
                "High": high,
                "Low": low,
                "Volume": volume,
                "Value": value,
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Unix timestamp support
    if pd.api.types.is_numeric_dtype(df["Date"]):
        try:
            sample = pd.to_numeric(df["Date"], errors="coerce").dropna()
            if not sample.empty and sample.median() > 10_000_000_000:
                df["Date"] = pd.to_datetime(
                    df["Date"], unit="ms", errors="coerce"
                )
            elif not sample.empty and sample.median() > 1_000_000_000:
                df["Date"] = pd.to_datetime(
                    df["Date"], unit="s", errors="coerce"
                )
            else:
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        except Exception:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    else:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    for col in ["Close", "High", "Low", "Volume", "Value"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.dropna(subset=["Date", "Close"])
        .sort_values("Date")
        .drop_duplicates("Date")
        .reset_index(drop=True)
    )

    if df["High"].isna().all():
        df["High"] = df["Close"]
    else:
        df["High"] = df["High"].fillna(df["Close"])

    if df["Low"].isna().all():
        df["Low"] = df["Close"]
    else:
        df["Low"] = df["Low"].fillna(df["Close"])

    df["Volume"] = df["Volume"].fillna(0)
    df["Value"] = df["Value"].fillna(0)

    return df


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))

    out = out.where(avg_loss != 0, 100)
    return out.fillna(50)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    c = df["Close"]

    df["EMA9"] = ema(c, 9)
    df["EMA20"] = ema(c, 20)
    df["EMA50"] = ema(c, 50)
    df["EMA200"] = ema(c, 200)

    df["RSI14"] = rsi(c, 14)

    macd = df["EMA12"] if "EMA12" in df else ema(c, 12)
    signal_base = df["EMA26"] if "EMA26" in df else ema(c, 26)

    df["MACD"] = macd - signal_base
    df["MACD_SIGNAL"] = ema(df["MACD"], 9)
    df["MACD_HIST"] = df["MACD"] - df["MACD_SIGNAL"]

    prev_close = c.shift(1)
    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["ATR14"] = tr.rolling(14).mean()

    df["VOL20"] = df["Volume"].rolling(20).mean()
    df["VOL_RATIO"] = df["Volume"] / df["VOL20"].replace(0, np.nan)

    df["VALUE20"] = df["Value"].rolling(20).mean()
    if df["Value"].sum() == 0:
        df["VALUE20"] = (df["Close"] * df["Volume"]).rolling(20).mean()

    df["HIGH20_PREV"] = df["High"].shift(1).rolling(20).max()
    df["LOW20_PREV"] = df["Low"].shift(1).rolling(20).min()

    df["RET1M"] = c.pct_change(21) * 100
    df["RET3M"] = c.pct_change(63) * 100
    df["RET6M"] = c.pct_change(126) * 100

    df["EMA20_SLOPE"] = df["EMA20"].pct_change(10) * 100

    return df


def get_last_valid(row: pd.Series, key: str, default=0.0) -> float:
    value = row.get(key, default)
    try:
        value = float(value)
        return default if not np.isfinite(value) else value
    except Exception:
        return default


def bullish_score(r: pd.Series, market_ret_3m: float) -> float:
    score = 0.0
    close = get_last_valid(r, "Close")
    ema20 = get_last_valid(r, "EMA20")
    ema50 = get_last_valid(r, "EMA50")
    ema200 = get_last_valid(r, "EMA200")
    slope = get_last_valid(r, "EMA20_SLOPE")
    ret1 = get_last_valid(r, "RET1M")
    ret3 = get_last_valid(r, "RET3M")
    ret6 = get_last_valid(r, "RET6M")
    rv = get_last_valid(r, "RSI14", 50)
    vol = get_last_valid(r, "VOL_RATIO")
    high20 = get_last_valid(r, "HIGH20_PREV")
    macd = get_last_valid(r, "MACD")
    sig = get_last_valid(r, "MACD_SIGNAL")

    # Trend: 25
    if close > ema20:
        score += 5
    if ema20 > ema50:
        score += 5
    if ema50 > ema200:
        score += 5
    if slope > 0:
        score += 5
    if close > ema50:
        score += 5

    # Momentum: 20
    if ret1 > 0:
        score += 5
    if ret3 > 0:
        score += 5
    if ret6 > 0:
        score += 5
    if ret3 > market_ret_3m:
        score += 5

    # RSI: 10
    if 55 <= rv <= 65:
        score += 10
    elif 65 < rv <= 70:
        score += 8
    elif 50 <= rv < 55:
        score += 6
    elif 45 <= rv < 50:
        score += 3

    # Volume: 15
    if vol >= 2.0:
        score += 15
    elif vol >= 1.5:
        score += 12
    elif vol >= 1.2:
        score += 8
    elif vol >= 1.0:
        score += 5

    # Breakout / near breakout: 10
    if high20 > 0:
        if close >= high20:
            score += 10
        elif close >= high20 * 0.97:
            score += 7
        elif close >= high20 * 0.93:
            score += 4

    # MACD bonus: 5
    if macd > sig and macd > 0:
        score += 5
    elif macd > sig:
        score += 3

    return round(min(score, 100), 2)


def bearish_score(r: pd.Series) -> float:
    score = 0.0

    close = get_last_valid(r, "Close")
    ema9 = get_last_valid(r, "EMA9")
    ema20 = get_last_valid(r, "EMA20")
    ema50 = get_last_valid(r, "EMA50")
    ema200 = get_last_valid(r, "EMA200")
    slope = get_last_valid(r, "EMA20_SLOPE")
    ret1 = get_last_valid(r, "RET1M")
    ret3 = get_last_valid(r, "RET3M")
    ret6 = get_last_valid(r, "RET6M")
    rv = get_last_valid(r, "RSI14", 50)
    vol = get_last_valid(r, "VOL_RATIO")
    low20 = get_last_valid(r, "LOW20_PREV")
    macd = get_last_valid(r, "MACD")
    sig = get_last_valid(r, "MACD_SIGNAL")

    if close < ema20:
        score += 15
    if ema20 < ema50:
        score += 15
    if ema50 < ema200:
        score += 10
    if slope < 0:
        score += 10
    if close < ema50:
        score += 10

    if ret1 < 0:
        score += 7
    if ret3 < 0:
        score += 7
    if ret6 < 0:
        score += 5

    if rv < 40:
        score += 10
    elif rv < 45:
        score += 7
    elif rv < 50:
        score += 3

    if vol < 0.70:
        score += 5

    if macd < sig:
        score += 5
    if macd < 0:
        score += 3

    if low20 > 0 and close <= low20:
        score += 10

    if ema9 < ema20:
        score += 3

    return round(min(score, 100), 2)


def watch_trigger(r: pd.Series) -> str:
    close = get_last_valid(r, "Close")
    ema20 = get_last_valid(r, "EMA20")
    ema50 = get_last_valid(r, "EMA50")
    rv = get_last_valid(r, "RSI14", 50)
    vol = get_last_valid(r, "VOL_RATIO")
    macd = get_last_valid(r, "MACD")
    sig = get_last_valid(r, "MACD_SIGNAL")

    triggers = []

    if close <= ema20:
        triggers.append(f"above EMA20 {ema20:.2f}")
    if ema20 <= ema50:
        triggers.append(f"EMA20 > EMA50 ({ema50:.2f})")
    if macd <= sig:
        triggers.append("MACD bullish crossover")
    if vol < 1.20:
        triggers.append("volume >= 1.2x")
    if rv < 50:
        triggers.append("RSI > 50")

    if not triggers:
        triggers.append("breakout confirmation")

    return " + ".join(triggers[:3])


def score_row(
    symbol: str,
    df: pd.DataFrame,
    market_ret_3m: float,
) -> Dict[str, Any]:

    r = df.iloc[-1]

    close = get_last_valid(r, "Close")
    atr = get_last_valid(r, "ATR14")
    ema20 = get_last_valid(r, "EMA20")
    ema50 = get_last_valid(r, "EMA50")
    ema200 = get_last_valid(r, "EMA200")
    ema9 = get_last_valid(r, "EMA9")
    macd = get_last_valid(r, "MACD")
    macd_sig = get_last_valid(r, "MACD_SIGNAL")
    rv = get_last_valid(r, "RSI14", 50)
    vol = get_last_valid(r, "VOL_RATIO")
    value20 = get_last_valid(r, "VALUE20")

    score = bullish_score(r, market_ret_3m)
    bear = bearish_score(r)

    macd_bull = macd > macd_sig
    trend_bull = close > ema20 > ema50
    volume_confirm = vol >= 1.20

    high20 = get_last_valid(r, "HIGH20_PREV")
    breakout = high20 > 0 and close >= high20 * 0.97

    # Strong BUY: current-price momentum confirmation.
    buy = (
        score >= 80
        and trend_bull
        and macd_bull
        and volume_confirm
        and 50 <= rv <= 72
        and breakout
    )

    # WATCH: positive setup, but not yet fully confirmed.
    watch = (
        not buy
        and score >= 55
        and (
            (close > ema20 and ema20 >= ema50)
            or (macd_bull and vol >= 1.0)
            or (rv >= 50 and get_last_valid(r, "EMA20_SLOPE") > 0)
        )
    )

    # SELL only when weakness is confirmed.
    sell = (
        not buy
        and not watch
        and bear >= 60
        and close < ema20
        and ema9 < ema20
        and (ema20 < ema50 or macd < macd_sig or rv < 45)
    )

    if buy:
        signal = "BUY"
    elif watch:
        signal = "WATCH"
    elif sell:
        signal = "SELL"
    else:
        signal = "NO ACTION"

    if atr <= 0:
        atr = max(close * 0.03, 0.01)

    support = get_last_valid(r, "LOW20_PREV")
    if support <= 0:
        support = close - atr

    entry_low = max(support, close - 0.50 * atr)
    entry_high = close + 0.25 * atr

    stop_loss = max(support * 0.98, close - 1.00 * atr)

    risk = max(close - stop_loss, 0.01)
    target1 = close + 1.50 * risk
    target2 = close + 2.50 * risk

    if buy:
        reason = "Trend + MACD + volume + breakout confirmed"
    elif watch:
        reason = "Momentum building; waiting for trigger"
    elif sell:
        reason = "Confirmed bearish trend / momentum weakness"
    else:
        reason = "No strong confirmed setup"

    return {
        "Symbol": symbol,
        "Date": r["Date"].date().isoformat()
        if hasattr(r["Date"], "date")
        else str(r["Date"]),
        "LTP": round(close, 2),
        "Score": score,
        "Bear_Score": bear,
        "Signal": signal,
        "RSI": round(rv, 2),
        "EMA9": round(ema9, 2),
        "EMA20": round(ema20, 2),
        "EMA50": round(ema50, 2),
        "EMA200": round(ema200, 2),
        "MACD": round(macd, 4),
        "MACD_Signal": round(macd_sig, 4),
        "Volume_Ratio": round(vol, 2),
        "Return_1M_%": round(get_last_valid(r, "RET1M"), 2),
        "Return_3M_%": round(get_last_valid(r, "RET3M"), 2),
        "Return_6M_%": round(get_last_valid(r, "RET6M"), 2),
        "EMA20_Slope_%": round(get_last_valid(r, "EMA20_SLOPE"), 2),
        "Avg_Value_20D": round(value20, 2),
        "Support": round(support, 2),
        "Entry_Low": round(entry_low, 2),
        "Entry_High": round(entry_high, 2),
        "Stop_Loss": round(stop_loss, 2),
        "Target_1": round(target1, 2),
        "Target_2": round(target2, 2),
        "Watch_Trigger": watch_trigger(r) if watch else "",
        "Reason": reason,
    }


def get_symbols(api: Nepse) -> List[str]:
    securities = api.get_security_list()

    symbols = []
    if isinstance(securities, list):
        for item in securities:
            if not isinstance(item, dict):
                continue
            symbol = (
                item.get("symbol")
                or item.get("stockSymbol")
                or item.get("ticker")
            )
            if symbol:
                symbols.append(str(symbol).upper())

    return sorted(set(symbols))


def get_security_id(api: Nepse, symbol: str):
    if getattr(api, "security_map", None) is None:
        api.get_security_list()

    security_map = getattr(api, "security_map", {}) or {}

    for sid, sym in security_map.items():
        if str(sym).upper() == symbol.upper():
            return sid

    return None


def fetch_market_return(api: Nepse) -> float:
    """Use NEPSE index history when available; otherwise return 0."""
    try:
        raw = api.get_nepse_index()
        if isinstance(raw, dict):
            data = raw.get("data", raw.get("content", raw))
        else:
            data = raw

        if isinstance(data, list) and len(data) >= 64:
            values = []
            for x in data:
                if isinstance(x, dict):
                    v = (
                        x.get("value")
                        or x.get("index")
                        or x.get("close")
                        or x.get("closingValue")
                    )
                    try:
                        values.append(float(v))
                    except Exception:
                        pass

            if len(values) >= 64:
                return float((values[-1] / values[-64] - 1) * 100)
    except Exception:
        pass

    return 0.0


def save_outputs(out: pd.DataFrame, top_n: int = 10):
    os.makedirs("output", exist_ok=True)

    buys = (
        out[out["Signal"] == "BUY"]
        .sort_values(["Score", "Return_3M_%"], ascending=False)
        .head(top_n)
    )

    watches = (
        out[out["Signal"] == "WATCH"]
        .sort_values(["Score", "Return_3M_%"], ascending=False)
        .head(top_n)
    )

    sells = (
        out[out["Signal"] == "SELL"]
        .sort_values(["Bear_Score", "Score"], ascending=[False, True])
        .head(top_n)
    )

    candidates = out[out["Signal"].isin(["BUY", "WATCH"])].copy()
    top10 = candidates.sort_values(
        ["Score", "Return_3M_%"], ascending=False
    ).head(top_n)

    no_action = out[out["Signal"] == "NO ACTION"]

    out.to_csv("output/latest_all.csv", index=False)
    buys.to_csv("output/latest_buy.csv", index=False)
    watches.to_csv("output/latest_watch.csv", index=False)
    sells.to_csv("output/latest_sell.csv", index=False)
    top10.to_csv("output/latest_top10.csv", index=False)
    no_action.to_csv("output/latest_no_action.csv", index=False)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out.to_csv(f"output/scan_{stamp}_all.csv", index=False)

    print("\n" + "=" * 60)
    print("DAILY CURRENT-DATA NEPSE SCANNER")
    print("=" * 60)

    print(f"\n🟢 BUY ({len(buys)})")
    if buys.empty:
        print("None")
    else:
        print(
            buys[
                ["Symbol", "LTP", "Score", "RSI", "Volume_Ratio",
                 "Entry_Low", "Entry_High", "Stop_Loss",
                 "Target_1", "Target_2"]
            ].to_string(index=False)
        )

    print(f"\n🟡 WATCH ({len(watches)})")
    if watches.empty:
        print("None")
    else:
        print(
            watches[
                ["Symbol", "LTP", "Score", "RSI",
                 "Volume_Ratio", "Watch_Trigger"]
            ].to_string(index=False)
        )

    print(f"\n🔴 SELL ({len(sells)})")
    if sells.empty:
        print("None")
    else:
        print(
            sells[
                ["Symbol", "LTP", "Score", "Bear_Score",
                 "RSI", "Volume_Ratio", "Reason"]
            ].to_string(index=False)
        )

    print(f"\n⚪ NO ACTION: {len(no_action)} stocks")
    print("\nSaved:")
    print("  output/latest_all.csv")
    print("  output/latest_buy.csv")
    print("  output/latest_watch.csv")
    print("  output/latest_sell.csv")
    print("  output/latest_top10.csv")
    print("  output/latest_no_action.csv")


def scan(top_n: int = 10):
    api = Nepse()

    print("Fetching current NEPSE security list...")
    symbols = get_symbols(api)
    print(f"Found {len(symbols)} securities.")

    market_ret_3m = fetch_market_return(api)
    print(f"Market 3M return used for relative strength: {market_ret_3m:.2f}%")

    results = []

    for i, symbol in enumerate(symbols, 1):
        try:
            sid = get_security_id(api, symbol)
            if sid is None:
                print(f"[{i}/{len(symbols)}] {symbol}: security id not found")
                continue

            raw = api.get_historical_chart(
                sid,
                start_date=None,
                end_date=None,
            )

            df = normalize_history(raw)

            if len(df) < MIN_HISTORY:
                print(
                    f"[{i}/{len(symbols)}] {symbol}: "
                    f"skip, history={len(df)}"
                )
                continue

            df = add_indicators(df)

            avg_value = get_last_valid(df.iloc[-1], "VALUE20")
            if avg_value < MIN_AVG_VALUE:
                print(
                    f"[{i}/{len(symbols)}] {symbol}: "
                    f"skip, low liquidity"
                )
                continue

            row = score_row(symbol, df, market_ret_3m)
            results.append(row)

            print(
                f"[{i}/{len(symbols)}] {symbol}: "
                f"{row['Signal']} | "
                f"Score={row['Score']} | "
                f"Bear={row['Bear_Score']} | "
                f"LTP={row['LTP']}"
            )

        except Exception as e:
            print(f"[{i}/{len(symbols)}] {symbol}: ERROR {e}")

        time.sleep(SLEEP_BETWEEN_REQUESTS)

    if not results:
        raise RuntimeError("No valid stock data was returned by the API.")

    out = pd.DataFrame(results)

    signal_order = {
        "BUY": 0,
        "WATCH": 1,
        "SELL": 2,
        "NO ACTION": 3,
    }

    out["_order"] = out["Signal"].map(signal_order).fillna(9)

    out = (
        out.sort_values(
            ["_order", "Score", "Return_3M_%"],
            ascending=[True, False, False],
        )
        .drop(columns=["_order"])
        .reset_index(drop=True)
    )

    save_outputs(out, top_n=top_n)

    return out


def main():
    parser = argparse.ArgumentParser(
        description="Daily current-data NEPSE momentum scanner"
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of BUY/WATCH/SELL rows to keep in shortlist",
    )
    args = parser.parse_args()

    scan(top_n=args.top)


if __name__ == "__main__":
    main()
