from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yfinance as yf
from utils import calculate_indicators, generate_signal

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

@app.post("/analyze")
def analyze_stock(req: AnalysisRequest, request: Request):
    try:
        data = yf.download(req.symbol, period="3mo", interval="1d")
        if data.empty:
            raise HTTPException(status_code=404, detail="종목을 찾을 수 없거나 데이터가 없습니다.")

        indicators = calculate_indicators(data)
        signal_result = generate_signal(indicators, req.decision)

        return {
            "symbol": req.symbol,
            "decision_requested": req.decision,
            "recommendation": signal_result["recommendation"],
            "conviction_score": signal_result["score"],
            "strength_level": signal_result["level"] if signal_result["level"] else "해당없음",
            "color": signal_result["color"],
            "reason": signal_result["reason"],
            "indicators": indicators
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "LKBUY2 API is running"}