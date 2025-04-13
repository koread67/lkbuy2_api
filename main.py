from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import yfinance as yf
from utils import calculate_indicators, generate_signal

app = FastAPI(title="LKBUY2 API")

class AnalysisRequest(BaseModel):
    symbol: str         # 예: "AAPL" 또는 "005930.KS"
    decision: str       # "매수" 또는 "매도"

@app.post("/analyze")
def analyze_stock(req: AnalysisRequest):
    try:
        # 야후 파이낸스에서 최근 3개월치 일봉 데이터 다운로드
        data = yf.download(req.symbol, period="3mo", interval="1d")
        if data.empty:
            raise HTTPException(status_code=404, detail="종목을 찾을 수 없거나 데이터가 없습니다.")

        # 기술적 지표 계산 (ADX, CCI, OBV 및 OBV 추세)
        indicators = calculate_indicators(data)
        # 계산된 지표를 기반으로 요청한 판단(매수/매도)과 신뢰도 점수를 산출
        recommendation, conviction_score = generate_signal(indicators, req.decision)
        
        result = {
            "symbol": req.symbol,
            "decision_requested": req.decision,
            "recommendation": recommendation,
            "conviction_score": round(conviction_score, 2),
            "indicators": indicators
        }
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
@app.get("/")
def home():
    return {"message": "LKBUY2 API is running"}
