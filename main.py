from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import yfinance as yf
from utils import calculate_indicators, generate_signal

app = FastAPI(title="LKBUY2 API")

class AnalysisRequest(BaseModel):
    symbol: str         # 예: "AAPL" 또는 "005930.KS"
    decision: str       # "매수" 또는 "매도"

@app.post("/analyze")
def analyze_stock(req: AnalysisRequest, request: Request):
    try:
        print(f"✅ 요청 받음: symbol={req.symbol}, decision={req.decision}")

        data = yf.download(req.symbol, period="3mo", interval="1d")
        print("📊 데이터 다운로드 완료:", data.shape)

        if data.empty:
            raise HTTPException(status_code=404, detail="종목을 찾을 수 없거나 데이터가 없습니다.")

        indicators = calculate_indicators(data)
        recommendation, conviction_score = generate_signal(indicators, req.decision)

        return {
            "symbol": req.symbol,
            "decision_requested": req.decision,
            "recommendation": recommendation,
            "conviction_score": round(conviction_score, 2),
            "indicators": indicators
        }

    except HTTPException as e:
        raise e  # 404는 그대로 전달
    except Exception as e:
        print("❌ 서버 오류 발생:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "LKBUY2 API is running"}
