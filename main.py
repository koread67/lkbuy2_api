# Finnhub + Yahoo fallback version

import math
import os
import traceback
import pandas as pd
import requests
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from utils import calculate_indicators, generate_signal

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "").strip()

class Req(BaseModel):
    symbol: str
    decision: str

def fetch_finnhub(symbol):
    try:
        if not FINNHUB_API_KEY:
            return None
        url = "https://finnhub.io/api/v1/stock/candle"
        now = int(pd.Timestamp.utcnow().timestamp())
        past = int((pd.Timestamp.utcnow() - pd.Timedelta(days=365)).timestamp())

        res = requests.get(url, params={
            "symbol": symbol,
            "resolution": "D",
            "from": past,
            "to": now,
            "token": FINNHUB_API_KEY
        })

        print("FINNHUB:", res.status_code, res.text[:100])

        data = res.json()
        if data.get("s") != "ok":
            return None

        df = pd.DataFrame({
            "Date": pd.to_datetime(data["t"], unit="s"),
            "Open": data["o"],
            "High": data["h"],
            "Low": data["l"],
            "Close": data["c"],
            "Volume": data["v"],
        })
        return df
    except:
        return None

def fetch_yahoo(symbol):
    try:
        df = yf.download(symbol, period="1y", interval="1d", progress=False)
        if df.empty:
            return None
        df = df.reset_index()
        return df
    except:
        return None

def fetch(symbol):
    df = fetch_finnhub(symbol)
    if df is not None:
        return df, "FINNHUB"
    df = fetch_yahoo(symbol)
    if df is not None:
        return df, "YAHOO"
    return None, "NONE"

@app.post("/analyze")
def analyze(req: Req):
    df, source = fetch(req.symbol)

    if df is None:
        return JSONResponse(
            content={"symbol": req.symbol, "message": "검출안됨", "source": source},
            status_code=404
        )

    ind = calculate_indicators(df)
    res = generate_signal(ind, req.decision)

    return {
        "symbol": req.symbol,
        "source": source,
        "result": res
    }
