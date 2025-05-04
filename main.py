
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import requests
import pandas as pd
import traceback

app = FastAPI(title="LKBUY2 API - Alpha + KRX Fallback")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = "2QD3PFZE54GZO088"

class AnalysisRequest(BaseModel):
    symbol: str  # 종목 코드 또는 심볼 (예: AAPL, 069500)
    decision: str

def fetch_indicator(symbol, function, extra_params):
    url = "https://www.alphavantage.co/query"
    params = {
        "function": function,
        "symbol": symbol,
        "apikey": API_KEY,
        "interval": "daily"
    }
    params.update(extra_params)
    response = requests.get(url, params=params)
    return response.json()

def fetch_krx_etf_data(etf_code):
    try:
        url = "http://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd"
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        data = {
            "bld": "dbms/MDC/STAT/standard/MDCSTAT01701",
            "mktId": "ETF",
            "etfTab": "1",
            "code": etf_code
        }
        response = requests.post(url, data=data, headers=headers)
        result = response.json()
        df = pd.DataFrame(result["output"])
        df["종가"] = pd.to_numeric(df["종가"].str.replace(",", ""), errors="coerce")
        df["일자"] = pd.to_datetime(df["일자"])
        df = df.sort_values("일자")
        return df
    except Exception as e:
        print("❌ KRX ETF fallback 실패:", e)
        return None

def calculate_indicators(symbol):
    try:
        cci_data = fetch_indicator(symbol, "CCI", {"time_period": 20})
        obv_data = fetch_indicator(symbol, "OBV", {})
        rsi_data = fetch_indicator(symbol, "RSI", {"time_period": 14, "series_type": "close"})

        cci_series = cci_data.get("Technical Analysis: CCI", {})
        obv_series = obv_data.get("Technical Analysis: OBV", {})
        rsi_series = rsi_data.get("Technical Analysis: RSI", {})

        df = pd.DataFrame({
            "CCI": {d: float(v["CCI"]) for d, v in cci_series.items()},
            "OBV": {d: float(v["OBV"]) for d, v in obv_series.items()},
            "RSI": {d: float(v["RSI"]) for d, v in rsi_series.items()},
        }).sort_index(ascending=True)

        if df is None or df.empty or len(df) < 2:
            raise ValueError("Alpha Vantage 데이터 부족")

        latest = df.iloc[-1]
        prev = df.iloc[-8] if len(df) >= 8 else df.iloc[-2]
        obv_trend = latest["OBV"] - prev["OBV"]

        return {
            "CCI": latest["CCI"],
            "OBV": latest["OBV"],
            "OBV_trend": obv_trend,
            "RSI": latest["RSI"]
        }

    except Exception as e:
        print("⚠️ Alpha 실패, KRX fallback 시도")
        df = fetch_krx_etf_data(symbol)
        if df is None or df.empty or len(df) < 10:
            raise ValueError("Alpha/KRX 모두 실패")

        close = df["종가"]
        high = close.rolling(2).max()
        low = close.rolling(2).min()
        volume = pd.Series([1000000] * len(close))  # KRX는 거래량 제공 안 함: 임시값

        import ta
        cci = ta.trend.CCIIndicator(high=high, low=low, close=close, window=20).cci().iloc[-1]
        obv = ta.volume.OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()
        rsi = ta.momentum.RSIIndicator(close=close, window=14).rsi().iloc[-1]
        obv_trend = obv.iloc[-1] - obv.iloc[-8] if len(obv) >= 8 else 0

        return {
            "CCI": cci,
            "OBV": obv.iloc[-1],
            "OBV_trend": obv_trend,
            "RSI": rsi,
        }

def generate_signal(indicators: dict, decision: str):
    cci = indicators.get("CCI", 0)
    obv_trend = indicators.get("OBV_trend", 0)
    rsi = indicators.get("RSI", 0)

    reasons = []

    if decision == "매수":
        cci_score = 40 if cci < -100 else 10
        obv_score = 30 if obv_trend > 0 else 10
        rsi_score = 30 if rsi < 30 else 10
    else:
        cci_score = 40 if cci > 100 else 10
        obv_score = 30 if obv_trend < 0 else 10
        rsi_score = 30 if rsi > 70 else 10

    if decision == "매수":
        reasons.append("CCI가 과매도 구간입니다." if cci < -100 else "CCI가 과매도 구간이 아닙니다.")
        reasons.append("OBV가 상승세로 자금 유입 신호입니다." if obv_trend > 0 else "OBV가 하락세로 자금 유출 신호입니다.")
        reasons.append("RSI가 30 이하로 과매도 구간입니다." if rsi < 30 else "RSI가 과매도 구간이 아닙니다.")
    else:
        reasons.append("CCI가 과매수 구간입니다." if cci > 100 else "CCI가 과매수 구간이 아닙니다.")
        reasons.append("OBV가 하락세로 자금 이탈 신호입니다." if obv_trend < 0 else "OBV가 상승세로 자금 유입 신호입니다.")
        reasons.append("RSI가 70 이상으로 과매수 구간입니다." if rsi > 70 else "RSI가 과매수 구간이 아닙니다.")

    score = round(cci_score * 0.45 + obv_score * 0.35 + rsi_score * 0.20)

    if decision == "매수":
        recommendation = "매수" if score >= 80 else "매수X"
    else:
        recommendation = "매도" if score >= 80 else "매도X"

    if score >= 80:
        color = "#4CAF50"
        level = "매우 강함"
    elif score >= 60:
        color = "#FFEB3B"
        level = "보통"
    elif score >= 40:
        color = "#FF9800"
        level = "약함"
    else:
        color = "#F44336"
        level = "매수X" if decision == "매수" else "매도X"

    return {
        "recommendation": recommendation,
        "score": score,
        "color": color,
        "level": level,
        "reason": " ".join(reasons)
    }

@app.post("/analyze")
def analyze_stock(req: AnalysisRequest, request: Request):
    try:
        indicators = calculate_indicators(req.symbol)
        result = generate_signal(indicators, req.decision)
        return JSONResponse(content={
            "symbol": req.symbol,
            "decision_requested": req.decision,
            "recommendation": result["recommendation"],
            "conviction_score": result["score"],
            "strength_level": result["level"],
            "color": result["color"],
            "reason": result["reason"],
            "indicators": indicators
        }, media_type="application/json; charset=utf-8")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "LKBUY2 API (Alpha + KRX fallback) is running"}
