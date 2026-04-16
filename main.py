import math
import os
import re
import traceback
from io import StringIO
from urllib.parse import quote

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

NAVER_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://finance.naver.com/",
}
FINNHUB_BASE_URL = "https://finnhub.io/api/v1"
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "").strip()


class AnalysisRequest(BaseModel):
    symbol: str
    decision: str | None = None


def safe_float(value):
    return float(value) if value is not None and math.isfinite(value) else 0.0


def normalize_symbol(symbol: str) -> str:
    if symbol is None:
        return ""

    s = str(symbol).strip().upper()
    # 해외/국내 혼합 입력용: 영문/숫자/.-:_ 만 유지
    s = re.sub(r"[^A-Z0-9\.\-:_]", "", s)
    return s


def normalize_search_query(symbol: str) -> str:
    if symbol is None:
        return ""

    s = str(symbol).strip()
    # 네이버 검색용: 한글, 영문, 숫자, 공백, 괄호, .-_:+ 허용
    s = re.sub(r"[^0-9A-Za-z가-힣\s\-\._\(\):+]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def is_krx_symbol(symbol: str) -> bool:
    s = str(symbol).strip().upper()
    return re.fullmatch(r"[0-9A-Z]{6,12}", s) is not None

def is_possible_domestic_query(symbol: str) -> bool:
    if symbol is None:
        return False

    raw = str(symbol).strip()
    if not raw:
        return False

    # 6자리 숫자 코드는 물론 국내로 본다
    if raw.isdigit() and len(raw) == 6:
        return True

    # 숫자/영문/특수문자가 섞인 6문자 이상 입력도 국내 검색 후보로 허용
    compact = re.sub(r"[^0-9A-Za-z가-힣]", "", raw)
    has_digit = bool(re.search(r"\d", raw))
    has_alpha = bool(re.search(r"[A-Za-z]", raw))
    has_special = bool(re.search(r"[^0-9A-Za-z가-힣\s]", raw))

    if len(compact) >= 6 and (has_alpha or has_special) and has_digit:
        return True

    # 국내 ETF/ETN 계열 검색어도 허용
    upper = raw.upper()
    keywords = ["ETF", "ETN", "KODEX", "TIGER", "ACE", "KBSTAR", "ARIRANG", "KOSEF", "SOL", "HANARO", "TIMEFOLIO", "PLUS", "RISE"]
    if any(k in upper for k in keywords):
        return True

    return False



def extract_krx_code(symbol: str) -> str | None:
    if symbol is None:
        return None

    raw = str(symbol).strip().upper()

    # 1) 영문+숫자 혼합 6~12자리 코드 자체
    exact_alnum = re.fullmatch(r"[0-9A-Z]{6,12}", raw)
    if exact_alnum:
        return raw

    # 2) code=XXXXX 형태에서 영문 포함 코드 추출
    m = re.search(r"code=([0-9A-Z]{6,12})", raw)
    if m:
        return m.group(1)

    # 3) 숫자 6자리 코드
    exact_num = re.search(r"(?<!\d)(\d{6})(?!\d)", raw)
    if exact_num:
        return exact_num.group(1)

    return None


KRX_CONFUSABLE_CHAR_MAP = {
    "O": "0",
    "Q": "0",
    "D": "0",
    "I": "1",
    "L": "1",
    "Z": "2",
    "S": "5",
    "B": "8",
    "G": "6",
}


def extract_krx_candidates(symbol: str) -> list[str]:
    """
    국내 종목코드 후보를 넓게 생성.
    예:
    - 005930.KS -> 005930
    - KRX:005930 -> 005930
    - 0026E0 -> 002660 처럼 혼동 문자 보정 후보 생성
    """
    if symbol is None:
        return []

    raw = str(symbol).strip().upper()
    candidates: list[str] = []
    seen: set[str] = set()

    def add(candidate: str | None):
        if candidate and re.fullmatch(r"\d{6}", candidate) and candidate not in seen:
            seen.add(candidate)
            candidates.append(candidate)

    # 1) 기존 규칙 기반 직접 추출
    add(extract_krx_code(raw))

    # 2) 숫자만 남겼을 때 5~6자리면 0패딩도 고려
    digits_only = re.sub(r"[^0-9]", "", raw)
    if len(digits_only) == 6:
        add(digits_only)
    elif len(digits_only) == 5:
        add(digits_only.zfill(6))

    # 3) 영문 혼합형에 대해 혼동문자 치환 후 숫자화
    translated_chars = []
    for ch in raw:
        if ch.isdigit():
            translated_chars.append(ch)
        elif ch in KRX_CONFUSABLE_CHAR_MAP:
            translated_chars.append(KRX_CONFUSABLE_CHAR_MAP[ch])
    translated_digits = "".join(translated_chars)

    if len(translated_digits) == 6:
        add(translated_digits)
    elif len(translated_digits) > 6:
        for i in range(len(translated_digits) - 5):
            add(translated_digits[i:i+6])
    elif len(translated_digits) == 5:
        add(translated_digits.zfill(6))

    # 4) 원문에서 6글자 블록 단위로 잘라 보고 혼동문자 치환
    compact = re.sub(r"[^A-Z0-9]", "", raw)
    if len(compact) >= 6:
        for i in range(len(compact) - 5):
            block = compact[i:i+6]
            converted = "".join(ch if ch.isdigit() else KRX_CONFUSABLE_CHAR_MAP.get(ch, "") for ch in block)
            add(converted if len(converted) == 6 else None)

    return candidates


def search_krx_code_from_naver(query: str) -> str | None:
    """
    국내 ETF/ETN/레버리지/종목명을 네이버 금융 검색으로 6자리 종목코드로 해석.
    입력이 영문+숫자 혼합이어도 검색 결과의 code=###### 를 우선 사용.
    """
    try:
        q = normalize_search_query(query)
        if not q:
            return None

        # 1차: 네이버 금융 통합검색
        search_url = f"https://finance.naver.com/search/search.naver?query={quote(q)}"
        response = requests.get(search_url, headers=NAVER_HEADERS, timeout=10)
        response.raise_for_status()
        text = response.text

        codes = re.findall(r"code=([0-9A-Z]{6,12})", text)
        if codes:
            return codes[0]

        # 2차: 모바일 검색 fallback
        mobile_url = f"https://m.stock.naver.com/search/index?q={quote(q)}"
        response = requests.get(mobile_url, headers=NAVER_HEADERS, timeout=10)
        response.raise_for_status()
        text = response.text

        codes = re.findall(r'"stockCode"\s*:\s*"([0-9A-Z]{6,12})"', text)
        if codes:
            return codes[0]

        codes = re.findall(r"/stock/([0-9A-Z]{6,12})", text)
        if codes:
            return codes[0]

        return None
    except Exception:
        return None



def fetch_from_krx(symbol: str, pages: int = 12) -> pd.DataFrame | None:
    try:
        symbol = str(symbol).strip().upper()
        if symbol.isdigit():
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



def build_yahoo_candidates(symbol: str) -> list[str]:
    symbol = symbol.strip().upper()
    candidates: list[str] = []

    # 국내 숫자 6자리 코드면 한국 suffix 우선
    code = extract_krx_code(symbol)
    if code and code.isdigit() and len(code) == 6:
        candidates.extend([f"{code}.KS", f"{code}.KQ", code])
    elif code:
        candidates.append(code)

    # 원문 자체도 시도
    candidates.append(symbol)

    # 중복 제거
    deduped = []
    seen = set()
    for candidate in candidates:
        if candidate and candidate not in seen:
            deduped.append(candidate)
            seen.add(candidate)
    return deduped



def fetch_from_yahoo(symbol: str) -> pd.DataFrame | None:
    try:
        for candidate in build_yahoo_candidates(symbol):
            df = yf.download(candidate, period="1y", interval="1d", progress=False, auto_adjust=False)
            normalized = normalize_yahoo_df(df)
            if normalized is not None and not normalized.empty:
                return normalized
        return None
    except Exception:
        return None



def resolve_krx_code(symbol: str) -> str | None:
    # 1차: 입력값에서 가능한 국내 코드 후보들을 직접 생성
    for code in extract_krx_candidates(symbol):
        return code

    # 2차: 네이버 검색으로 국내 종목/ETF/ETN/레버리지 해석
    return search_krx_code_from_naver(symbol)



def fetch_stock_data(symbol: str) -> tuple[pd.DataFrame | None, str]:
    raw_symbol = str(symbol).strip()
    normalized_symbol = normalize_symbol(raw_symbol)

    if not raw_symbol and not normalized_symbol:
        return None, "INVALID"

    # 1) 국내 문자열 검색 우선
    # 숫자 6자리뿐 아니라 숫자+영문+특수문자 조합 6문자 이상도 국내 검색 대상으로 본다.
    if is_possible_domestic_query(raw_symbol):
        resolved_code = search_krx_code_from_naver(raw_symbol)
        if resolved_code:
            df = fetch_from_krx(resolved_code)
            if df is not None and not df.empty:
                return df, "KRX_SEARCH"

            df = fetch_from_yahoo(resolved_code)
            if df is not None and not df.empty:
                return df, "YAHOO_KRX_SEARCH"

    # 2) 명시적인 6자리 코드 또는 코드 내장 입력 처리
    krx_candidates = []
    direct_code = extract_krx_code(raw_symbol)
    if direct_code:
        krx_candidates.append(direct_code)

    for krx_code in krx_candidates:
        df = fetch_from_krx(krx_code)
        if df is not None and not df.empty:
            return df, "KRX"

        df = fetch_from_yahoo(krx_code)
        if df is not None and not df.empty:
            return df, "YAHOO_KRX"

    # 3) 해외/일반 티커 처리
    if normalized_symbol:
        df = fetch_from_finnhub(normalized_symbol)
        if df is not None and not df.empty:
            return df, "FINNHUB"

        df = fetch_from_yahoo(normalized_symbol)
        if df is not None and not df.empty:
            return df, "YAHOO"

    # 4) 마지막으로 원문을 야후 후보군에 넣어서 한 번 더 시도
    df = fetch_from_yahoo(raw_symbol)
    if df is not None and not df.empty:
        return df, "YAHOO_RAW"

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
