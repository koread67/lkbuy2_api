
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import requests
import pandas as pd
import traceback

app = FastAPI(title="LKBUY2 API - AlphaVantage Edition")

# ✅ CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = "2QD3PFZE54GZO088"

class AnalysisRequest(BaseModel):
    symbol: str
    decision: str

def fetch_indicator(symbol, function, extra_params):
    base_url = "https://www.alphavantage.co/query"
    params = {
        "function": function,
        "symbol": symbol,
        "apikey": API_KEY,
        "interval": "daily"
    }
    params.update(extra_params)
    response = requests.get(base_url, params=params)
    return response.json()

def calculate_indicators(symbol):
    cci_data = fetch_indicator(symbol, "CCI", {"time_period": 20})
    cci_series = cci_data.get("Technical Analysis: CCI", {})

    obv_data = fetch_indicator(symbol, "OBV", {})
    obv_series = obv_data.get("Technical Analysis: OBV", {})

    rsi_data = fetch_indicator(symbol, "RSI", {"time_period": 14, "series_type": "close"})
    rsi_series = rsi_data.get("Technical Analysis: RSI", {})

    df = pd.DataFrame({
        "CCI": {date: float(val["CCI"]) for date, val in cci_series.items()},
        "OBV": {date: float(val["OBV"]) for date, val in obv_series.items()},
        "RSI": {date: float(val["RSI"]) for date, val in rsi_series.items()},
    }).sort_index(ascending=True)

    latest = df.iloc[-1]
    prev = df.iloc[-8] if len(df) >= 8 else df.iloc[-2]

    obv_trend = latest["OBV"] - prev["OBV"]

    return {
        "CCI": latest["CCI"],
        "OBV": latest["OBV"],
        "OBV_trend": obv_trend,
        "RSI": latest["RSI"]
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
        print(f"✅ 요청 받음: symbol={req.symbol}, decision={req.decision}")
        indicators = calculate_indicators(req.symbol)
        result = generate_signal(indicators, req.decision)

        response = {
            "symbol": req.symbol,
            "decision_requested": req.decision,
            "recommendation": str(result["recommendation"]),
            "conviction_score": float(result["score"]),
            "strength_level": str(result["level"]),
            "color": result["color"],
            "reason": str(result["reason"]),
            "indicators": indicators
        }

        return JSONResponse(content=response, media_type="application/json; charset=utf-8")

    except HTTPException as e:
        raise e
    except Exception as e:
        print("❌ 서버 오류 발생:", str(e))
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "LKBUY2 API with Alpha Vantage is running"}
