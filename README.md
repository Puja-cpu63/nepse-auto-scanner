
# NEPSE Automatic BUY/SELL Scanner

This project scans the currently available NEPSE security universe and calculates a
0-100 technical score. It produces:

- latest_all.csv
- latest_top10.csv
- latest_buy.csv
- latest_sell.csv
- dated copies of every scan
- scan_failures.csv

## 1. Install

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## 2. Run

```bash
python scanner.py
```

Top 20 instead of 10:

```bash
python scanner.py --top 20
```

## 3. Score

100 points total:

- Trend: 25
- Momentum: 20
- RSI: 10
- Volume: 15
- Breakout: 10
- Fundamentals: 15
- Relative/sector strength: 5

The current code leaves FundamentalScore at 0 unless you create an optional
`fundamentals.csv`:

```csv
Symbol,FundamentalScore
NABIL,13
CHCL,12
API,10
```

This is intentional because public NEPSE API schemas for financial statements
can change. Do not blindly treat missing fundamental data as a perfect score.

## 4. Signals

BUY:
- Score >= 80
- MACD bullish
- RSI 50-72
- Price > EMA20

HOLD:
- Score >= 65
- Price > EMA50
- MACD bullish

SELL:
- Price < EMA20
- EMA9 < EMA20
- RSI <45 or MACD bearish

Everything else is WATCH.

## 5. Automation

Run after the NEPSE session closes using Windows Task Scheduler, cron, or GitHub
Actions. Example cron (adjust for your server timezone):

```cron
30 12 * * 0-5 cd /path/to/nepse_auto_scanner && /path/to/.venv/bin/python scanner.py >> scanner.log 2>&1
```

For a monthly scan on the 1st:

```cron
30 12 1 * * cd /path/to/nepse_auto_scanner && /path/to/.venv/bin/python scanner.py >> monthly.log 2>&1
```

## Important

This is an algorithmic scanner, not a guarantee of future returns. Backtest it
and paper-trade it before risking money. The underlying data source is a
third-party open-source Python package and is not an official NEPSE product.


## Telegram notifications

### 1. Create a Telegram bot
In Telegram, open **@BotFather**, create a new bot with `/newbot`, and copy
the bot token.

### 2. Get your Chat ID
Start a chat with your bot and send `/start`.
For a private chat, use:
`https://api.telegram.org/botYOUR_TOKEN/getUpdates`
and read the `chat.id` value.

For a group/channel, add the bot and use the corresponding chat ID. Keep the
bot token private.

### 3. GitHub Secrets
In your GitHub repository:
Settings -> Secrets and variables -> Actions -> New repository secret

Create:
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

### 4. Run
The workflow scans after the scheduled time and sends:
- 🟢 NEW BUY
- 🔴 NEW SELL
- daily Top BUY summary

It stores the previous BUY/SELL symbols in `output/telegram_state.json` so the
same signal is not repeatedly labelled "NEW".

You can also manually run the workflow from GitHub Actions using
"Run workflow".
