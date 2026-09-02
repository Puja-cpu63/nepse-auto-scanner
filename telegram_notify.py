import os
import json
import requests
import pandas as pd

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
STATE_FILE = "output/telegram_state.json"
URL = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"


def load_csv(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def send(text):
    if not BOT_TOKEN or not CHAT_ID:
        raise RuntimeError("Telegram secrets are missing.")
    r = requests.post(URL, json={
        "chat_id": CHAT_ID,
        "text": text,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True
    }, timeout=30)
    r.raise_for_status()


def state_load():
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            x = json.load(f)
        return {k: x.get(k, []) for k in ("BUY", "WATCH", "SELL")}
    except Exception:
        return {"BUY": [], "WATCH": [], "SELL": []}


def state_save(x):
    os.makedirs("output", exist_ok=True)
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(x, f, indent=2)


def n(row, key, default=0):
    try:
        x = row.get(key, default)
        return default if pd.isna(x) else x
    except Exception:
        return default


def f(row, key, d=2):
    try:
        return f"{float(n(row,key)):,.{d}f}"
    except Exception:
        return str(n(row,key,"-"))


def buy_msg(r):
    return (f"🟢 *BUY CANDIDATE: {r['Symbol']}*\n"
            f"Score: *{f(r,'Score')}/100*\nLTP: Rs {f(r,'LTP')}\n"
            f"RSI: {f(r,'RSI')} | Volume: {f(r,'Volume_Ratio')}x\n"
            f"1M: {f(r,'Return_1M_%')}% | 3M: {f(r,'Return_3M_%')}%\n\n"
            f"Entry: Rs {f(r,'Entry_Low')} - {f(r,'Entry_High')}\n"
            f"SL: Rs {f(r,'Stop_Loss')}\n"
            f"Target 1: Rs {f(r,'Target_1')}\nTarget 2: Rs {f(r,'Target_2')}\n\n"
            f"EMA20: {f(r,'EMA20')} | EMA50: {f(r,'EMA50')}\n"
            f"MACD: {f(r,'MACD',4)} | Signal: {f(r,'MACD_Signal',4)}\n"
            f"Reason: {n(r,'Reason','Momentum confirmed.')}\n"
            f"⚠️ Algorithmic signal. Confirm before trading.")


def watch_msg(r):
    return (f"🟡 *WATCHLIST: {r['Symbol']}*\n"
            f"Score: *{f(r,'Score')}/100*\nLTP: Rs {f(r,'LTP')}\n"
            f"RSI: {f(r,'RSI')} | Volume: {f(r,'Volume_Ratio')}x\n"
            f"Trigger: *{n(r,'Watch_Trigger','Confirmation required')}*\n"
            f"EMA20: {f(r,'EMA20')} | EMA50: {f(r,'EMA50')}\n"
            f"MACD: {f(r,'MACD',4)} | Signal: {f(r,'MACD_Signal',4)}\n"
            f"⚠️ Wait for confirmation before entry.")


def sell_msg(r):
    return (f"🔴 *SELL / EXIT WARNING: {r['Symbol']}*\n"
            f"Bear Score: *{f(r,'Bear_Score')}/100*\n"
            f"Bull Score: {f(r,'Score')}/100\nLTP: Rs {f(r,'LTP')}\n"
            f"RSI: {f(r,'RSI')} | Volume: {f(r,'Volume_Ratio')}x\n"
            f"EMA20: {f(r,'EMA20')} | EMA50: {f(r,'EMA50')}\n"
            f"MACD: {f(r,'MACD',4)} | Signal: {f(r,'MACD_Signal',4)}\n"
            f"Reason: {n(r,'Reason','Confirmed weakness.')}\n"
            f"⚠️ Algorithmic exit warning. Confirm before trading.")


def syms(df):
    if df.empty or "Symbol" not in df.columns:
        return []
    return sorted(df["Symbol"].astype(str).str.upper().unique())


def main():
    buy = load_csv("output/latest_buy.csv")
    watch = load_csv("output/latest_watch.csv")
    sell = load_csv("output/latest_sell.csv")

    if "Signal" in buy: buy = buy[buy.Signal == "BUY"]
    if "Signal" in watch: watch = watch[watch.Signal == "WATCH"]
    if "Signal" in sell: sell = sell[sell.Signal == "SELL"]

    cur = {"BUY": syms(buy), "WATCH": syms(watch), "SELL": syms(sell)}
    old = state_load()

    for kind, df, maker in [
        ("BUY", buy, buy_msg), ("WATCH", watch, watch_msg), ("SELL", sell, sell_msg)
    ]:
        for symbol in sorted(set(cur[kind]) - set(old[kind])):
            row = df[df.Symbol.astype(str).str.upper() == symbol].iloc[0]
            send(maker(row))

    text = (f"📊 *NEPSE DAILY CURRENT-DATA SCAN*\n\n"
            f"🟢 BUY: *{len(buy)}*\n🟡 WATCH: *{len(watch)}*\n"
            f"🔴 SELL: *{len(sell)}*\n\n")

    if not buy.empty:
        text += "*Top BUY:*\n"
        for _, r in buy.sort_values("Score", ascending=False).head(5).iterrows():
            text += f"🟢 {r['Symbol']} — {f(r,'Score')} | Rs {f(r,'LTP')}\n"
        text += "\n"

    if not watch.empty:
        text += "*Top WATCH:*\n"
        for _, r in watch.sort_values("Score", ascending=False).head(5).iterrows():
            text += f"🟡 {r['Symbol']} — {f(r,'Score')} | Rs {f(r,'LTP')}\n"
        text += "\n"

    if not sell.empty:
        text += "*Top SELL / EXIT:*\n"
        for _, r in sell.sort_values("Bear_Score", ascending=False).head(5).iterrows():
            text += f"🔴 {r['Symbol']} — Bear {f(r,'Bear_Score')} | Rs {f(r,'LTP')}\n"
        text += "\n"

    text += "⚠️ Algorithmic scan only. Confirm before trading."
    send(text)
    state_save(cur)


if __name__ == "__main__":
    main()
