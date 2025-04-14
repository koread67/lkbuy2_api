from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import yfinance as yf
import traceback
from utils import calculate_indicators, generate_signal

app = FastAPI(title="LKBUY2 API")

# ✅ CORS 설정
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

@app.post("/analyze")
def analyze_stock(req: AnalysisRequest, request: Request):
    try:
        print(f"✅ 요청 받음: symbol={req.symbol}, decision={req.decision}")
        data = yf.download(req.symbol, period="3mo", interval="1d")
        print("📊 데이터 다운로드 완료:", data.shape)

        if data.empty:
            raise HTTPException(status_code=404, detail=f"종목 '{req.symbol}' 을 찾을 수 없거나 데이터가 없습니다.")

        indicators = calculate_indicators(data)
        signal_result = generate_signal(indicators, req.decision)

        response = {
            "symbol": req.symbol,
            "decision_requested": str(req.decision),
            "recommendation": str(signal_result["recommendation"]),
            "conviction_score": float(signal_result["score"]),
            "strength_level": str(signal_result["level"]),
            "color": signal_result["color"],
            "reason": str(signal_result["reason"]),
            "indicators": {
                "CCI": float(indicators["CCI"]),
                "OBV": float(indicators["OBV"]),
                "OBV_trend": float(indicators["OBV_trend"]),
                "RSI": float(indicators["RSI"]),
            }
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
    return {"message": "LKBUY2 API is running"}