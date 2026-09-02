import os
import time
import argparse
from datetime import date, datetime
from typing import Any, Dict

import numpy as np
import pandas as pd
from nepse_data_api import Nepse

MIN_HISTORY = 220
MIN_AVG_VALUE = 1_000_000
SLEEP_BETWEEN_REQUESTS = 0.10

def pick(d: Dict[str, Any], *keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default

def num(x):
    try:
        if x is None or x == "":
            return np.nan
        return float(str(x).replace(",", ""))
    except Exception:
        return np.nan

def normalize_history(raw):
    if raw is None:
        return pd.DataFrame()
    if isinstance(raw, dict):
        for key in ("data","content","result","records","prices","chartData"):
            if isinstance(raw.get(key), list):
                raw = raw[key]
                break
    if not isinstance(raw, list):
        return pd.DataFrame()
    rows = []
    for r in raw:
        if not isinstance(r, dict):
            continue
        dt = pick(r,"date","businessDate","tradingDate","timestamp","time")
        if isinstance(dt,(int,float)) and dt > 10_000_000_000:
            dt = pd.to_datetime(dt, unit="ms", errors="coerce")
        elif isinstance(dt,(int,float)) and dt > 1_000_000_000:
            dt = pd.to_datetime(dt, unit="s", errors="coerce")
        rows.append({
            "Date": dt,
            "Open": num(pick(r,"open","openPrice","openingPrice")),
            "High": num(pick(r,"high","highPrice")),
            "Low": num(pick(r,"low","lowPrice")),
            "Close": num(pick(r,"close","closePrice","closingPrice","lastTradedPrice")),
            "Volume": num(pick(r,"volume","totalTradedQuantity","totalTradeQuantity","quantity")),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for c in ["Open","High","Low","Close","Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return (df.dropna(subset=["Date","Close"])
              .drop_duplicates("Date")
              .sort_values("Date")
              .reset_index(drop=True))

def ema(s,n):
    return s.ewm(span=n, adjust=False).mean()

def rsi(s,n=14):
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    ag = gain.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    al = loss.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    rs = ag / al.replace(0,np.nan)
    return (100 - 100/(1+rs)).fillna(50)

def macd(s):
    line = ema(s,12) - ema(s,26)
    sig = ema(line,9)
    return line, sig, line-sig

def atr(df,n=14):
    pc = df["Close"].shift(1)
    tr = pd.concat([
        df["High"]-df["Low"],
        (df["High"]-pc).abs(),
        (df["Low"]-pc).abs()
    ],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False, min_periods=n).mean()

def add_indicators(df):
    x = df.copy()
    x["EMA9"] = ema(x["Close"],9)
    x["EMA20"] = ema(x["Close"],20)
    x["EMA50"] = ema(x["Close"],50)
    x["EMA200"] = ema(x["Close"],200)
    x["RSI14"] = rsi(x["Close"],14)
    x["MACD"],x["MACD_SIGNAL"],x["MACD_HIST"] = macd(x["Close"])
    x["ATR14"] = atr(x,14)
    x["VOL20"] = x["Volume"].rolling(20).mean()
    x["VOL_RATIO"] = x["Volume"] / x["VOL20"].replace(0,np.nan)
    x["VALUE20"] = (x["Close"]*x["Volume"]).rolling(20).mean()
    x["HIGH20"] = x["High"].rolling(20).max().shift(1)
    x["LOW20"] = x["Low"].rolling(20).min().shift(1)
    x["RET1M"] = x["Close"].pct_change(21)*100
    x["RET3M"] = x["Close"].pct_change(63)*100
    x["RET6M"] = x["Close"].pct_change(126)*100
    x["EMA20_SLOPE"] = x["EMA20"].pct_change(10)*100
    return x

def score_row(x, market_ret_3m=0.0):
    r = x.iloc[-1]
    score = 0
    reasons = []

    if r.Close > r.EMA20: score += 5; reasons.append("Price>EMA20")
    if r.EMA20 > r.EMA50: score += 5; reasons.append("EMA20>EMA50")
    if r.EMA50 > r.EMA200: score += 5; reasons.append("EMA50>EMA200")
    if r.EMA20_SLOPE > 0: score += 5; reasons.append("EMA20 rising")
    if r.Close > r.EMA50: score += 5; reasons.append("Price>EMA50")

    if r.RET1M > 0: score += 5
    if r.RET3M > 0: score += 5
    if r.RET6M > 0: score += 5
    if r.RET3M > market_ret_3m: score += 5

    rv = float(r.RSI14)
    if 55 <= rv <= 65: score += 10
    elif 65 < rv <= 70: score += 8
    elif 50 <= rv < 55: score += 6
    elif 45 <= rv < 50: score += 3

    vr = r.VOL_RATIO
    if pd.notna(vr):
        if vr >= 2.0: score += 15
        elif vr >= 1.5: score += 12
        elif vr >= 1.2: score += 8
        elif vr >= 1.0: score += 5

    if pd.notna(r.HIGH20) and r.Close >= r.HIGH20:
        score += 10; reasons.append("20D breakout")
    elif pd.notna(r.HIGH20) and r.HIGH20 > 0:
        distance = (r.HIGH20-r.Close)/r.HIGH20*100
        if distance <= 3: score += 7
        elif distance <= 7: score += 4

    relative_score = 5 if r.RET3M > market_ret_3m else 0
    score += relative_score

    macd_bull = bool(r.MACD > r.MACD_SIGNAL)
    volume_confirm = bool(pd.notna(vr) and vr >= 1.2)

    close = float(r.Close)
    atrv = float(r.ATR14) if pd.notna(r.ATR14) else close*0.04
    support = float(r.LOW20) if pd.notna(r.LOW20) else close-2*atrv
    stop = min(support*0.98, close-1.5*atrv)
    if stop <= 0 or stop >= close:
        stop = close*0.92
    risk = close-stop
    target1 = close+1.5*risk
    target2 = close+2.5*risk

    # Strict BUY: strong score + trend + MACD + RSI + volume.
    buy = (score >= 80 and macd_bull and volume_confirm
           and 50 <= rv <= 72 and close > r.EMA20 > r.EMA50)

    # Strict SELL: weak score + confirmed bearish trend.
    sell = (score <= 35 and close < r.EMA20 and r.EMA9 < r.EMA20
            and (rv < 45 or not macd_bull))

    if buy:
        signal = "BUY"
    elif score >= 65 and close > r.EMA50 and macd_bull and volume_confirm:
        signal = "HOLD"
    elif sell:
        signal = "SELL"
    else:
        signal = "WATCH"

    return {
        "Score": round(float(score),1),
        "Signal": signal,
        "LTP": round(close,2),
        "RSI": round(rv,2),
        "EMA9": round(float(r.EMA9),2),
        "EMA20": round(float(r.EMA20),2),
        "EMA50": round(float(r.EMA50),2),
        "EMA200": round(float(r.EMA200),2),
        "MACD": round(float(r.MACD),4),
        "MACD_Signal": round(float(r.MACD_SIGNAL),4),
        "MACD_Bull": macd_bull,
        "Volume_Ratio": round(float(vr),2) if pd.notna(vr) else np.nan,
        "Return_1M_%": round(float(r.RET1M),2),
        "Return_3M_%": round(float(r.RET3M),2),
        "Return_6M_%": round(float(r.RET6M),2),
        "20D_High": round(float(r.HIGH20),2) if pd.notna(r.HIGH20) else np.nan,
        "Support": round(support,2),
        "Entry_Low": round(close-0.5*atrv,2),
        "Entry_High": round(close+0.25*atrv,2),
        "Stop_Loss": round(stop,2),
        "Target_1": round(target1,2),
        "Target_2": round(target2,2),
        "Reasons": ", ".join(reasons[:8]),
    }

def get_symbol_and_id(item):
    if not isinstance(item,dict): return None,None
    return pick(item,"symbol","stockSymbol","securitySymbol"), pick(item,"id","securityId","securityID")

def fetch_universe(nepse):
    raw = nepse.get_security_list()
    if isinstance(raw,dict):
        for key in ("data","content","result","records"):
            if isinstance(raw.get(key),list):
                raw=raw[key]; break
    out=[]
    for item in raw or []:
        symbol,sid=get_symbol_and_id(item)
        if symbol and sid: out.append((str(symbol).upper(),sid))
    seen=set(); clean=[]
    for symbol,sid in out:
        if symbol not in seen:
            seen.add(symbol); clean.append((symbol,sid))
    return clean

def fetch_market_3m(nepse):
    try:
        raw=nepse.get_nepse_index()
        if isinstance(raw,dict):
            for key in ("data","content","result","records"):
                if isinstance(raw.get(key),list) and raw[key]:
                    raw=raw[key]; break
        if isinstance(raw,list) and len(raw)>=64:
            vals=[]
            for item in raw:
                if isinstance(item,dict):
                    vals.append(num(pick(item,"value","close","index","nepseIndex")))
            vals=[v for v in vals if not pd.isna(v)]
            if len(vals)>=64 and vals[-64]!=0:
                return (vals[-1]/vals[-64]-1)*100
    except Exception:
        pass
    return 0.0

def load_fundamentals():
    path="fundamentals.csv"
    if not os.path.exists(path): return {}
    try:
        df=pd.read_csv(path)
        if "Symbol" not in df or "FundamentalScore" not in df: return {}
        return dict(zip(df["Symbol"].astype(str).str.upper(),
                        df["FundamentalScore"].clip(0,15)))
    except Exception:
        return {}

def scan():
    parser=argparse.ArgumentParser()
    parser.add_argument("--top",type=int,default=10)
    parser.add_argument("--out",default="output")
    args=parser.parse_args()
    os.makedirs(args.out,exist_ok=True)
    nepse=Nepse(cache_ttl=120,enable_cache=True)

    print("Fetching NEPSE security universe...")
    universe=fetch_universe(nepse)
    print(f"Found {len(universe)} securities.")
    market_ret_3m=fetch_market_3m(nepse)
    fundamentals=load_fundamentals()
    results=[]; failures=[]

    for i,(symbol,sid) in enumerate(universe,1):
        try:
            raw=nepse.get_historical_chart(
                sid,start_date="2025-01-01",
                end_date=date.today().strftime("%Y-%m-%d"))
            df=normalize_history(raw)
            if len(df)<MIN_HISTORY:
                failures.append((symbol,"insufficient_history")); continue
            df=add_indicators(df).dropna(subset=["EMA200","RSI14","VOL20","ATR14"])
            if len(df)<30:
                failures.append((symbol,"insufficient_indicators")); continue
            last=df.iloc[-1]
            if pd.isna(last.VALUE20) or last.VALUE20<MIN_AVG_VALUE:
                failures.append((symbol,"low_liquidity")); continue

            result=score_row(df,market_ret_3m)
            fs=float(fundamentals.get(symbol,0))
            result["Fundamental_Score"]=fs
            result["Score"]=min(100,round(result["Score"]+fs,1))

            # Re-evaluate after optional fundamentals.
            score=result["Score"]; rv=result["RSI"]
            buy=(score>=80 and result["MACD_Bull"]
                 and result["Volume_Ratio"]>=1.2
                 and 50<=rv<=72 and result["LTP"]>result["EMA20"]>result["EMA50"])
            sell=(score<=35 and result["LTP"]<result["EMA20"]
                  and result["EMA9"]<result["EMA20"]
                  and (rv<45 or not result["MACD_Bull"]))
            if buy: result["Signal"]="BUY"
            elif score>=65 and result["LTP"]>result["EMA50"] and result["MACD_Bull"] and result["Volume_Ratio"]>=1.2:
                result["Signal"]="HOLD"
            elif sell: result["Signal"]="SELL"
            else: result["Signal"]="WATCH"

            result["Symbol"]=symbol
            result["Data_Date"]=str(df["Date"].iloc[-1].date())
            results.append(result)
        except Exception as e:
            failures.append((symbol,str(e)[:150]))
        if i%25==0: print(f"Processed {i}/{len(universe)}")
        time.sleep(SLEEP_BETWEEN_REQUESTS)

    if not results:
        raise RuntimeError("No stocks were successfully scanned. Check API connectivity/schema.")

    out=pd.DataFrame(results).sort_values(["Score","Return_3M_%"],ascending=[False,False]).reset_index(drop=True)
    out.insert(0,"Rank",np.arange(1,len(out)+1))
    buys=out[out.Signal=="BUY"].head(args.top).copy()
    sells=out[out.Signal=="SELL"].sort_values("Score").head(args.top).copy()
    top10=out.head(args.top).copy()

    stamp=datetime.now().strftime("%Y%m%d_%H%M%S")
    out.to_csv(f"{args.out}/all_scanned_{stamp}.csv",index=False)
    top10.to_csv(f"{args.out}/top10_{stamp}.csv",index=False)
    buys.to_csv(f"{args.out}/buy_signals_{stamp}.csv",index=False)
    sells.to_csv(f"{args.out}/sell_signals_{stamp}.csv",index=False)
    pd.DataFrame(failures,columns=["Symbol","Reason"]).to_csv(f"{args.out}/scan_failures_{stamp}.csv",index=False)
    out.to_csv(f"{args.out}/latest_all.csv",index=False)
    buys.to_csv(f"{args.out}/latest_buy.csv",index=False)
    sells.to_csv(f"{args.out}/latest_sell.csv",index=False)
    top10.to_csv(f"{args.out}/latest_top10.csv",index=False)

    print("\n=== TOP 10 ===")
    print(top10[["Rank","Symbol","LTP","Score","Signal","RSI","Volume_Ratio","Return_3M_%"]].to_string(index=False))
    print("\n=== BUY ===")
    print(buys[["Rank","Symbol","LTP","Score","Signal","RSI","Entry_Low","Entry_High","Stop_Loss","Target_1","Target_2"]].to_string(index=False) if len(buys) else "No BUY signals.")
    print("\n=== SELL ===")
    print(sells[["Rank","Symbol","LTP","Score","Signal","RSI","EMA20","EMA50","MACD","MACD_Signal"]].to_string(index=False) if len(sells) else "No SELL signals.")
    return out

if __name__=="__main__":
    scan()
