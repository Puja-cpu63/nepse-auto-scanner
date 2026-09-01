
import os
import json
import urllib.request
from pathlib import Path
import pandas as pd

# Set these as GitHub Actions Secrets / environment variables:
# TELEGRAM_BOT_TOKEN = BotFather token
# TELEGRAM_CHAT_ID   = your Telegram chat/group/channel ID

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
STATE_FILE = Path("output/telegram_state.json")

def send(text):
    if not TOKEN or not CHAT_ID:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID")
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = json.dumps({
        "chat_id": CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }).encode()
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read())

def load_state():
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {"buy": [], "sell": []}

def save_state(state):
    STATE_FILE.parent.mkdir(exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))

def fmt(v):
    if pd.isna(v):
        return "-"
    return f"{float(v):.2f}"

def main():
    buy_path = Path("output/latest_buy.csv")
    sell_path = Path("output/latest_sell.csv")

    if not buy_path.exists() and not sell_path.exists():
        raise RuntimeError("Scanner output not found. Run scanner.py first.")

    buys = pd.read_csv(buy_path) if buy_path.exists() else pd.DataFrame()
    sells = pd.read_csv(sell_path) if sell_path.exists() else pd.DataFrame()
    state = load_state()

    current_buy = buys["Symbol"].astype(str).tolist() if "Symbol" in buys else []
    current_sell = sells["Symbol"].astype(str).tolist() if "Symbol" in sells else []

    new_buys = [s for s in current_buy if s not in state.get("buy", [])]
    new_sells = [s for s in current_sell if s not in state.get("sell", [])]

    messages = []

    for symbol in new_buys:
        row = buys[buys["Symbol"].astype(str) == symbol].iloc[0]
        messages.append(
            f"🟢 <b>NEW BUY: {symbol}</b>\n"
            f"Score: <b>{fmt(row['Score'])}/100</b>\n"
            f"LTP: Rs {fmt(row['LTP'])}\n"
            f"RSI: {fmt(row['RSI'])}\n"
            f"Volume: {fmt(row['Volume_Ratio'])}x\n"
            f"1M Return: {fmt(row['Return_1M_%'])}%\n"
            f"3M Return: {fmt(row['Return_3M_%'])}%\n"
            f"Entry: Rs {fmt(row['Entry_Low'])} - {fmt(row['Entry_High'])}\n"
            f"Stop Loss: Rs {fmt(row['Stop_Loss'])}\n"
            f"Target 1: Rs {fmt(row['Target_1'])}\n"
            f"Target 2: Rs {fmt(row['Target_2'])}\n\n"
            f"⚠️ Algorithmic signal. Confirm before trading."
        )

    for symbol in new_sells:
        row = sells[sells["Symbol"].astype(str) == symbol].iloc[0]
        messages.append(
            f"🔴 <b>NEW SELL: {symbol}</b>\n"
            f"Score: <b>{fmt(row['Score'])}/100</b>\n"
            f"LTP: Rs {fmt(row['LTP'])}\n"
            f"RSI: {fmt(row['RSI'])}\n"
            f"Reason: trend/momentum weakness detected.\n\n"
            f"⚠️ Algorithmic signal. Confirm before trading."
        )

    # Send a daily summary even when there is no new signal.
    top = buys.head(10) if not buys.empty else pd.DataFrame()
    summary = "📊 <b>NEPSE Daily Scanner</b>\n\n"
    if not top.empty:
        summary += "<b>Top BUY candidates:</b>\n"
        for _, r in top.iterrows():
            summary += f"🟢 {r['Symbol']} | {fmt(r['Score'])}/100 | Rs {fmt(r['LTP'])}\n"
    else:
        summary += "No BUY signal today.\n"

    if not new_buys and not new_sells:
        messages.insert(0, summary)

    for m in messages:
        send(m)

    state = {"buy": current_buy, "sell": current_sell}
    save_state(state)

if __name__ == "__main__":
    main()
