
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

ALPHA_VANTAGE_API_KEY = "2QD3PFZE54GZO088"

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
        if value is None:
            return 0.0
        # allow strings like "85%"
        if isinstance(value, str) and value.endswith('%'):
            value = value[:-1]
        v = float(value)
        return v if math.isfinite(v) else 0.0
    except Exception:
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
        print(f"🔍 Alpha 요청 URL: {response.url}")
        if response.status_code != 200:
            print(f"❌ Alpha 요청 실패 (status: {response.status_code})")
            return None
        data = response.json()
        if "Time Series (Daily)" not in data:
            print("❌ Alpha 응답에 Time Series 없음:", data)
            return None

        df = pd.DataFrame(data["Time Series (Daily)"]).T
        df = df.astype(float)
        df = df.rename(columns={
            "1. open": "Open", "2. high": "High", "3. low": "Low",
            "4. close": "Close", "5. volume": "Volume"
        })[["Open", "High", "Low", "Close", "Volume"]]
        return df.tail(90)
    except Exception as e:
        print("❌ Alpha 처리 중 오류:", e)
        return None

def fetch_from_krx(symbol: str, pages: int = 5):
    try:
        symbol = symbol.zfill(6)
        headers = {"User-Agent": "Mozilla/5.0"}
        all_dfs = []
        for page in range(1, pages + 1):
            url = f"https://finance.naver.com/item/sise_day.nhn?code={symbol}&page={page}"
            print(f"📡 KRX 요청 중 (p{page}): {url}")
            response = requests.get(url, headers=headers)
            dfs = pd.read_html(StringIO(response.text), header=0)
            if dfs:
                all_dfs.append(dfs[0])
        if not all_dfs:
            return None
        df = pd.concat(all_dfs)
        print(f"📋 누적 수집된 DataFrame 크기 (raw): {df.shape}")

        df = df.dropna(how="all")
        df = df.rename(columns={"종가": "Close", "거래량": "Volume", "고가": "High", "저가": "Low"})
        for col in ["Close", "Volume", "High", "Low"]:
            df[col] = df[col].astype(str).str.replace(",", "").astype(float)
        df["Open"] = df[["Close", "High", "Low"]].mean(axis=1)
        result = df.iloc[::-1].reset_index(drop=True)
        print(f"✅ 정제된 DataFrame 크기: {result.shape}")
        return result
    except Exception as e:
        print("❌ KRX 다중 페이지 파싱 실패:", e)
        return None

@app.post("/analyze")
def analyze_stock(req: AnalysisRequest, request: Request):
    try:
        print(f"✅ 요청 받음: symbol={req.symbol}, decision={req.decision}")
        data = fetch_from_alpha_vantage(req.symbol)

        if data is None or data.empty:
            if req.symbol.isdigit() and len(req.symbol.zfill(6)) == 6:
                print("⚠️ Alpha 실패, 숫자형 종목으로 KRX 시도")
                data = fetch_from_krx(req.symbol, pages=5)
            else:
                print("🚫 Alpha 실패 + KRX 우회 차단 (비숫자 종목코드)")

        if data is None or (hasattr(data, "empty") and data.empty):
            return JSONResponse(
                content={"symbol": req.symbol, "message": "검출안됨"},
                media_type="application/json; charset=utf-8",
                status_code=404
            )

        indicators = calculate_indicators(data)
        result = generate_signal(indicators, req.decision)

        # NOTE: utils.generate_signal now returns:
        # - strength_pct (int, 0~100)
        # - strength (str, "85%")
        response = {
            "symbol": req.symbol,
            "decision_requested": req.decision,
            "recommendation": str(result.get("recommendation", "")),
            "conviction_score": safe_float(result.get("score")),
            "strength_pct": int(result.get("strength_pct", 0)),
            "strength": str(result.get("strength", f"{int(result.get('strength_pct', 0))}%")),
            "level": str(result.get("level", "")),
            "color": str(result.get("color", "#F44336")),
            "reason": str(result.get("reason", "")),
            "thresholds": result.get("thresholds", {}),
            "weights": result.get("weights", {}),
            "indicators": {
                "CCI": safe_float(indicators.get("CCI")),       # present
                "OBV_trend": safe_float(indicators.get("OBV_trend")), # present
                "RSI": safe_float(indicators.get("RSI")),       # present
            }
        }

        return JSONResponse(content=response, media_type="application/json; charset=utf-8")
    except Exception as e:
        print("❌ 서버 오류:", str(e))
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "LKBUY2 API is running"}
