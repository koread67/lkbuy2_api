import math
import os
import traceback
from io import StringIO

import pandas as pd
import requests
import yfinance as yf
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from utils import calculate_indicators, generate_signal, generate_dual_signal

app = FastAPI(title="LKBUY2 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

NAVER_HEADERS = {"User-Agent": "Mozilla/5.0"}
FINNHUB_BASE_URL = "https://finnhub.io/api/v1"
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "").strip()


class AnalysisRequest(BaseModel):
    symbol: str
    decision: str | None = None


def safe_float(value):
    return float(value) if value is not None and math.isfinite(value) else 0.0


def is_krx_symbol(symbol: str) -> bool:
    s = symbol.strip()
    return s.isdigit() and len(s) <= 6


def fetch_from_krx(symbol: str, pages: int = 12) -> pd.DataFrame | None:
    try:
        symbol = symbol.zfill(6)
        all_dfs = []

        for page in range(1, pages + 1):
            url = f"https://finance.naver.com/item/sise_day.nhn?code={symbol}&page={page}"
            response = requests.get(url, headers=NAVER_HEADERS, timeout=10)
            response.raise_for_status()
            dfs = pd.read_html(StringIO(response.text), header=0)
            if dfs:
                all_dfs.append(dfs[0])

        if not all_dfs:
            return None

        df = pd.concat(all_dfs, ignore_index=True)
        df = df.dropna(how="all")

        if "날짜" not in df.columns:
            return None

        df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
        df = df.dropna(subset=["날짜"])

        df = df.rename(
            columns={
                "날짜": "Date",
                "종가": "Close",
                "거래량": "Volume",
                "고가": "High",
                "저가": "Low",
                "시가": "Open",
            }
        )

        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(",", "", regex=False)
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["Date", "Open", "High", "Low", "Close", "Volume"])
        df = df.sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
        return df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    except Exception:
        return None


def fetch_from_finnhub(symbol: str, lookback_days: int = 400) -> pd.DataFrame | None:
    if not FINNHUB_API_KEY:
        return None

    try:
        symbol = symbol.strip().upper()
        now = pd.Timestamp.utcnow().floor("s")
        date_from = int((now - pd.Timedelta(days=lookback_days)).timestamp())
        date_to = int(now.timestamp())

        response = requests.get(
            f"{FINNHUB_BASE_URL}/stock/candle",
            params={
                "symbol": symbol,
                "resolution": "D",
                "from": date_from,
                "to": date_to,
                "token": FINNHUB_API_KEY,
            },
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()

        if not isinstance(payload, dict) or payload.get("s") != "ok":
            return None

        required_keys = ["t", "o", "h", "l", "c", "v"]
        if any(k not in payload for k in required_keys):
            return None

        df = pd.DataFrame(
            {
                "Date": pd.to_datetime(payload["t"], unit="s", utc=True).tz_localize(None),
                "Open": pd.to_numeric(payload["o"], errors="coerce"),
                "High": pd.to_numeric(payload["h"], errors="coerce"),
                "Low": pd.to_numeric(payload["l"], errors="coerce"),
                "Close": pd.to_numeric(payload["c"], errors="coerce"),
                "Volume": pd.to_numeric(payload["v"], errors="coerce"),
            }
        )

        df = df.dropna(subset=["Date", "Open", "High", "Low", "Close", "Volume"])
        df = df.sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
        return df
    except Exception:
        return None


def normalize_yahoo_df(df: pd.DataFrame) -> pd.DataFrame | None:
    try:
        if df is None or df.empty:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

        df = df.reset_index()
        rename_map = {"Adj Close": "Adj_Close"}
        df = df.rename(columns=rename_map)

        required = ["Date", "Open", "High", "Low", "Close", "Volume"]
        for col in required:
            if col not in df.columns:
                return None

        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date", "Open", "High", "Low", "Close", "Volume"])
        df = df.sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
        return df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    except Exception:
        return None


def fetch_from_yahoo(symbol: str) -> pd.DataFrame | None:
    try:
        symbol = symbol.strip().upper()
        candidates = [symbol]

        if is_krx_symbol(symbol):
            code = symbol.zfill(6)
            candidates = [f"{code}.KS", f"{code}.KQ", code]

        for candidate in candidates:
            df = yf.download(candidate, period="1y", interval="1d", progress=False, auto_adjust=False)
            normalized = normalize_yahoo_df(df)
            if normalized is not None and not normalized.empty:
                return normalized

        return None
    except Exception:
        return None


def fetch_stock_data(symbol: str) -> tuple[pd.DataFrame | None, str]:
    symbol = symbol.strip()

    if is_krx_symbol(symbol):
        df = fetch_from_krx(symbol)
        if df is not None and not df.empty:
            return df, "KRX"

        df = fetch_from_yahoo(symbol)
        if df is not None and not df.empty:
            return df, "YAHOO_KRX"

        return None, "KRX"

    df = fetch_from_finnhub(symbol)
    if df is not None and not df.empty:
        return df, "FINNHUB"

    df = fetch_from_yahoo(symbol)
    if df is not None and not df.empty:
        return df, "YAHOO"

    return None, "NONE"


def fetch_vix() -> pd.DataFrame | None:
    try:
        response = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS", timeout=10)
        response.raise_for_status()

        vix = pd.read_csv(StringIO(response.text))
        vix.columns = ["Date", "VIX"]
        vix["Date"] = pd.to_datetime(vix["Date"], errors="coerce")
        vix["VIX"] = pd.to_numeric(vix["VIX"], errors="coerce")
        vix = vix.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
        return vix
    except Exception:
        return None


def merge_vix(stock_df: pd.DataFrame, vix_df: pd.DataFrame | None) -> pd.DataFrame:
    if stock_df is None or stock_df.empty:
        return stock_df

    merged = stock_df.copy()
    if vix_df is None or vix_df.empty:
        merged["VIX"] = pd.NA
        return merged

    merged = pd.merge(merged, vix_df, on="Date", how="left")
    merged["VIX"] = merged["VIX"].ffill()
    return merged


@app.post("/analyze")
def analyze_stock(req: AnalysisRequest, request: Request):
    try:
        if req.decision not in ["매수", "매도"]:
            raise HTTPException(status_code=400, detail="decision은 '매수' 또는 '매도'여야 합니다.")

        data, source = fetch_stock_data(req.symbol)
        if data is None or data.empty:
            return JSONResponse(
                content={"symbol": req.symbol, "message": "검출안됨", "source": source},
                status_code=404,
            )

        vix = fetch_vix()
        data = merge_vix(data, vix)
        indicators = calculate_indicators(data)
        result = generate_signal(indicators, req.decision)

        return {
            "symbol": req.symbol,
            "source": source,
            "recommendation": result["recommendation"],
            "conviction_score": safe_float(result.get("score")),
            "strength_level": safe_float(result.get("strength")),
            "position_size": safe_float(result.get("position_size")),
            "color": result.get("color"),
            "reason": result.get("reason"),
            "indicators": indicators,
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze-dual")
def analyze_stock_dual(req: AnalysisRequest, request: Request):
    try:
        data, source = fetch_stock_data(req.symbol)
        if data is None or data.empty:
            return JSONResponse(
                content={"symbol": req.symbol, "message": "검출안됨", "source": source},
                status_code=404,
            )

        vix = fetch_vix()
        data = merge_vix(data, vix)
        indicators = calculate_indicators(data)
        result = generate_dual_signal(indicators)

        return {
            "symbol": req.symbol,
            "source": source,
            "final_decision": result["final_decision"],
            "buy_signal": result["buy_signal"],
            "sell_signal": result["sell_signal"],
            "indicators": indicators,
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    return {
        "message": "LKBUY2 API running",
        "finnhub_configured": bool(FINNHUB_API_KEY),
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "finnhub_configured": bool(FINNHUB_API_KEY),
    }
