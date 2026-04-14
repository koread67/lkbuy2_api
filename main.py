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
    return float(value) if value is not None and math.isfinite(value) else 0.0

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
        return df.iloc[::-1].reset_index(drop=True)
    except Exception:
        return None

@app.post("/analyze")
def analyze_stock(req: AnalysisRequest, request: Request):
    try:
        data = fetch_from_krx(req.symbol)

        if data is None or data.empty:
            return JSONResponse(
                content={"symbol": req.symbol, "message": "검출안됨"},
                status_code=404
            )

        indicators = calculate_indicators(data)
        result = generate_signal(indicators, req.decision)

        return {
            "symbol": req.symbol,
            "recommendation": result["recommendation"],
            "conviction_score": safe_float(result["score"]),
            "strength_level": safe_float(result["strength"]),
            "color": result["color"],
            "reason": result["reason"],
            "indicators": indicators
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "LKBUY2 API running"}
