
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import requests
from io import StringIO
from datetime import datetime
import math
import time

# === utils.py hooks ===
from utils import calculate_indicators, generate_signal

# === Use Finnhub (user-provided key) ===
FINNHUB_API_KEY = "d2jqag1r01qj8a5kmhigd2jqag1r01qj8a5kmhj0"

app = FastAPI(title="LKBUY2 API (finnhub)")

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

def fetch_from_finnhub(symbol: str):
    """
    Daily OHLCV from Finnhub /stock/candle (resolution=D)
    """
    try:
        url = "https://finnhub.io/api/v1/stock/candle"
        now = int(time.time())
        past = now - 60*60*24*220  # ~220 days window
        params = {
            "symbol": symbol,
            "resolution": "D",
            "from": past,
            "to": now,
            "token": FINNHUB_API_KEY
        }
        r = requests.get(url, params=params, timeout=20)
        data = r.json()
        # Expected response has s == "ok"
        if data.get("s") != "ok":
            print("Finnhub response:", data)
            return None
        df = pd.DataFrame({
            "Date": pd.to_datetime(data.get("t", []), unit="s"),
            "Open": data.get("o", []),
            "High": data.get("h", []),
            "Low": data.get("l", []),
            "Close": data.get("c", []),
            "Volume": data.get("v", [])
        })
        if df.empty:
            return None
        df = df.dropna().sort_values("Date").reset_index(drop=True)
        return df.tail(120)
    except Exception as e:
        print("Finnhub fetch fail:", e)
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

    # Overseas / non-numeric -> Yahoo first, then Finnhub
    if df is None and not symbol.isdigit():
        df = fetch_from_yahoo(symbol)
        source = "yahoo" if df is not None else None
    if df is None:
        df = fetch_from_finnhub(symbol)
        if df is not None:
            source = "finnhub"

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
