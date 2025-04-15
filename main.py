from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import yfinance as yf
from utils import calculate_indicators, generate_signal
import traceback

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
            raise HTTPException(status_code=404, detail=f"종목 '{req.symbol}'을 찾을 수 없거나 데이터가 없습니다.")

        indicators = calculate_indicators(data)
        result = generate_signal(indicators, req.decision)

        score = result.get("score")
        level = result.get("level")

        # 예외 없이 숫자/문자 형태로 반환되도록 보장
        try:
            score = float(score)
        except:
            score = 0.0

        if not isinstance(level, str):
            level = "해당없음"

        response = {
            "symbol": req.symbol,
            "decision_requested": req.decision,
            "recommendation": str(result.get("recommendation", "분석불가")),
            "conviction_score": score,
            "strength_level": level,
            "color": result.get("color", "#888888"),
            "reason": str(result.get("reason", "분석 결과 없음")),
            "indicators": {
                "CCI": float(indicators.get("CCI", 0)),
                "OBV": float(indicators.get("OBV", 0)),
                "OBV_trend": float(indicators.get("OBV_trend", 0)),
                "RSI": float(indicators.get("RSI", 0)),
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