
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import requests
from io import StringIO
from datetime import datetime
import math

# === utils.py hooks ===
from utils import calculate_indicators, generate_signal

ALPHA_VANTAGE_API_KEY = "2QD3PFZE54GZO088"

app = FastAPI(title="LKBUY2 API (render-fix)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalysisRequest(BaseModel):
    symbol: str
    decision: str

def _safe_float(x):
    try:
        if x is None:
            return 0.0
        s = str(x).strip()
        if s.endswith('%'):
            s = s[:-1]
        v = float(s)
        return v if math.isfinite(v) else 0.0
    except Exception:
        return 0.0

def fetch_from_yahoo(symbol: str):
    try:
        import yfinance as yf
        tk = yf.Ticker(symbol)
        df = tk.history(period="6mo", interval="1d", auto_adjust=False)
        if df is None or df.empty:
            return None
        df = df.rename(columns=str.title)[["Open","High","Low","Close","Volume"]]
        df = df.dropna().reset_index(drop=False)
        # Ensure monotonic by date
        if "Date" in df.columns:
            df = df.sort_values("Date").reset_index(drop=True)
        return df.tail(120)
    except Exception as e:
        print("Yahoo fetch fail:", e)
        return None

def fetch_from_alpha_vantage(symbol: str):
    try:
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "apikey": ALPHA_VANTAGE_API_KEY,
            "outputsize": "compact"
        }
        r = requests.get(url, params=params, timeout=15)
        data = r.json()
        ts = data.get("Time Series (Daily)")
        if not ts:
            print("Alpha response has no TS:", list(data.keys()))
            return None
        df = pd.DataFrame(ts).T.astype(float)
        df = df.rename(columns={
            "1. open": "Open", "2. high": "High", "3. low": "Low",
            "4. close": "Close", "5. volume": "Volume"
        })[["Open","High","Low","Close","Volume"]]
        df = df.iloc[::-1].reset_index(drop=False).rename(columns={"index":"Date"})
        df["Date"] = pd.to_datetime(df["Date"])
        return df.tail(120)
    except Exception as e:
        print("Alpha fetch fail:", e)
        return None

def fetch_from_krx_naver(code: str, pages: int = 5):
    try:
        code = code.zfill(6)
        headers = {"User-Agent": "Mozilla/5.0"}
        dfs = []
        for page in range(1, pages+1):
            url = f"https://finance.naver.com/item/sise_day.naver?code={code}&page={page}"
            res = requests.get(url, headers=headers, timeout=12)
            tbls = pd.read_html(StringIO(res.text))
            if not tbls:
                continue
            df = tbls[0]
            dfs.append(df)
        if not dfs:
            return None
        df = pd.concat(dfs, ignore_index=True)
        df = df.dropna().reset_index(drop=True)
        df = df.rename(columns={
            "날짜":"Date","종가":"Close","고가":"High","저가":"Low","거래량":"Volume"
        })
        # Some pages don't have Open; approximate using H/L/C mean
        for col in ["Close","High","Low","Volume"]:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",",""), errors="coerce")
        df["Open"] = df[["Close","High","Low"]].mean(axis=1)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
        return df[["Date","Open","High","Low","Close","Volume"]].tail(180)
    except Exception as e:
        print("KRX fetch fail:", e)
        return None

@app.post("/analyze")
def analyze(req: AnalysisRequest):
    symbol = (req.symbol or "").strip()
    decision = (req.decision or "매수").strip()

    if not symbol:
        return JSONResponse({"error":"symbol required"}, status_code=400)

    df = None
    source = None

    # Domestic numeric -> KRX
    if symbol.isdigit() and len(symbol) <= 6:
        df = fetch_from_krx_naver(symbol)
        source = "krx_naver"

    # Overseas / non-numeric -> Yahoo first, then Alpha
    if df is None and not symbol.isdigit():
        df = fetch_from_yahoo(symbol)
        source = "yahoo" if df is not None else None
    if df is None:
        df = fetch_from_alpha_vantage(symbol)
        if df is not None:
            source = "alpha_vantage"

    if df is None or df.empty:
        return JSONResponse({"symbol": symbol, "message": "데이터 없음 또는 제공처 제한", "source": source}, status_code=404)

    # Prepare OHLCV for utils
    data = df[["Open","High","Low","Close","Volume"]].copy()
    indicators = calculate_indicators(data)
    result = generate_signal(indicators, decision)

    # Compose debug fields to verify change by symbol/decision
    debug = {
        "symbol": symbol,
        "decision_requested": decision,
        "data_source": source,
        "rows": int(len(df)),
        "last_date": df["Date"].iloc[-1].strftime("%Y-%m-%d") if "Date" in df.columns else None,
        "raw_indicators": {k: _safe_float(v) for k,v in indicators.items()},
        "buy_score": result.get("components",{}),  # component contribution for selected side
        "thresholds": result.get("thresholds",{}),
    }

    resp = {
        "symbol": symbol,
        "decision_requested": decision,
        "recommendation": str(result.get("recommendation","")),
        "conviction_score": _safe_float(result.get("score")),
        "strength_pct": int(result.get("strength_pct", 0)),
        "strength": str(result.get("strength", f"{int(result.get('strength_pct',0))}%")),
        "level": str(result.get("level","")),
        "color": str(result.get("color","#F44336")),
        "reason": str(result.get("reason","")),
        "thresholds": result.get("thresholds", {}),
        "weights": result.get("weights", {}),
        "indicators": {
            "CCI": _safe_float(indicators.get("CCI")),
            "OBV_trend": _safe_float(indicators.get("OBV_trend")),
            "RSI": _safe_float(indicators.get("RSI")),
        },
        "debug": debug,
    }
    return JSONResponse(resp, media_type="application/json; charset=utf-8")
