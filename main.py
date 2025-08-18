from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from utils import calculate_indicators, generate_signal
import traceback
import requests
import pandas as pd
from io import StringIO
import math

ALPHA_VANTAGE_API_KEY = "demo"  # 필요 시 실제 키로 교체

app = FastAPI(title="LKBUY2 API")

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

def safe_float(value):
    try:
        v = float(value)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return 0.0

def fetch_from_alpha_vantage(symbol: str):
    try:
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "apikey": ALPHA_VANTAGE_API_KEY,
            "outputsize": "compact"
        }
        response = requests.get(url, params=params)
        if response.status_code != 200:
            return None
        data = response.json()
        if "Time Series (Daily)" not in data:
            return None

        df = pd.DataFrame(data["Time Series (Daily)"]).T
        df = df.astype(float)
        df = df.rename(columns={
            "1. open": "Open", "2. high": "High", "3. low": "Low",
            "4. close": "Close", "5. volume": "Volume"
        })[["Open", "High", "Low", "Close", "Volume"]]
        return df.tail(90)
    except Exception:
        return None

def fetch_from_krx(symbol: str, pages: int = 5):
    try:
        symbol = symbol.zfill(6)
        headers = {"User-Agent": "Mozilla/5.0"}
        all_dfs = []
        for page in range(1, pages + 1):
            url = f"https://finance.naver.com/item/sise_day.nhn?code={symbol}&page={page}"
            response = requests.get(url, headers=headers)
            dfs = pd.read_html(StringIO(response.text), header=0)
            if dfs:
                all_dfs.append(dfs[0])
        df = pd.concat(all_dfs)
        df = df.dropna(how="all")
        df = df.rename(columns={"종가": "Close", "거래량": "Volume", "고가": "High", "저가": "Low"})
        df["Close"] = df["Close"].astype(str).str.replace(",", "").astype(float)
        df["Volume"] = df["Volume"].astype(str).str.replace(",", "").astype(float)
        df["High"] = df["High"].astype(str).str.replace(",", "").astype(float)
        df["Low"] = df["Low"].astype(str).str.replace(",", "").astype(float)
        df["Open"] = df[["Close", "High", "Low"]].mean(axis=1)
        result = df.iloc[::-1].reset_index(drop=True)
        return result
    except Exception:
        return None

def fetch_from_stooq(symbol: str):
    try:
        sym = symbol.lower()
        if '.' not in sym:
            sym = f"{sym}.us"  # 미국주식 기본
        url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
        df = pd.read_csv(url)
        df = df.dropna()
        df = df.sort_values('Date')
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        return df.tail(90).reset_index(drop=True)
    except Exception:
        return None

@app.post("/analyze")
def analyze_stock(req: AnalysisRequest, request: Request):
    try:
        data = fetch_from_alpha_vantage(req.symbol)

        if data is None or getattr(data, 'empty', True):
            if req.symbol.isdigit() and len(req.symbol.zfill(6)) == 6:
                data = fetch_from_krx(req.symbol, pages=5)
            else:
                data = fetch_from_stooq(req.symbol)

        if data is None or getattr(data, 'empty', True):
            return JSONResponse(
                content={"symbol": req.symbol, "message": "검출안됨"},
                media_type="application/json; charset=utf-8",
                status_code=404
            )

        indicators = calculate_indicators(data)
        result = generate_signal(indicators, req.decision)

        response = {
            "symbol": req.symbol,
            "decision_requested": req.decision,
            "recommendation": str(result.get("recommendation", "")),
            "conviction_score": safe_float(result.get("score")),
            "strength_level": safe_float(result.get("strength")),
            "color": result.get("color"),
            "level": result.get("level"),
            "reason": str(result.get("reason", "")),
            "indicators": {
                "CCI": safe_float(indicators.get("CCI")),
                "OBV": safe_float(indicators.get("OBV")),
                "OBV_trend": safe_float(indicators.get("OBV_trend")),
                "RSI": safe_float(indicators.get("RSI")),
            }
        }
        return JSONResponse(content=response, media_type="application/json; charset=utf-8")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "LKBUY2 API is running"}
