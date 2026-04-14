from __future__ import annotations

from io import StringIO
from typing import Dict, Optional

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

KOREA_ETF_CODES = [
    "069500", "102110", "114800", "117700", "118260", "122630", "139260", "157450",
    "214980", "219480", "229200", "232080", "233740", "237370", "238720", "251340",
    "252670", "253150", "263750", "272220", "278530", "292150", "295820", "305720",
    "307510", "310970", "352560", "364980", "371160", "379800", "394280", "411060",
    "426410", "438420", "461340"
]

# 최적 조합 반영
CCI_WEIGHT = 0.33
RSI_WEIGHT = 0.33
OBV_WEIGHT = 0.34

STRONG_BUY_THRESHOLD = 50
STRONG_SELL_THRESHOLD = 90


def get_naver_etf_price(code: str, pages: int = 5) -> Optional[pd.DataFrame]:
    """
    네이버 금융 ETF 일별 시세를 읽어 OHLCV 데이터프레임으로 반환한다.
    """
    try:
        if not str(code).isdigit():
            return None

        url = f"https://finance.naver.com/item/sise_day.naver?code={code}"
        headers = {"User-Agent": "Mozilla/5.0"}
        dfs = []

        for page in range(1, pages + 1):
            res = requests.get(f"{url}&page={page}", headers=headers, timeout=10)
            res.raise_for_status()

            soup = BeautifulSoup(res.text, "html.parser")
            table = soup.select_one("table[type='N']")
            if table is None:
                return None

            df = pd.read_html(StringIO(str(table)))[0].dropna()
            dfs.append(df)

        df_all = pd.concat(dfs, ignore_index=True)
        df_all.columns = ["Date", "Close", "Change", "Open", "High", "Low", "Volume"]
        df_all["Date"] = pd.to_datetime(df_all["Date"])
        df_all = df_all.sort_values("Date").set_index("Date")

        return df_all[["Open", "High", "Low", "Close", "Volume"]]

    except Exception:
        return None


def _calculate_cci(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
    typical_price = (high + low + close) / 3.0
    sma = typical_price.rolling(window).mean()

    def _mad(values: pd.Series) -> float:
        avg = values.mean()
        return float(np.mean(np.abs(values - avg)))

    mad = typical_price.rolling(window).apply(_mad, raw=False)
    return (typical_price - sma) / (0.015 * mad)


def _calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    if len(close) == 0:
        return pd.Series(dtype="float64")

    obv_values = [float(volume.iloc[0])]
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i - 1]:
            obv_values.append(obv_values[-1] + float(volume.iloc[i]))
        elif close.iloc[i] < close.iloc[i - 1]:
            obv_values.append(obv_values[-1] - float(volume.iloc[i]))
        else:
            obv_values.append(obv_values[-1])

    return pd.Series(obv_values, index=close.index, dtype="float64")


def _calculate_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    diff = close.diff(1)
    up = diff.clip(lower=0)
    down = -diff.clip(upper=0)

    avg_up = up.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_down = down.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    rs = avg_up / avg_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def calculate_indicators(data: pd.DataFrame) -> Dict[str, float]:
    """
    CCI, OBV, RSI와 OBV 20일 평균 및 OBV 추세를 계산한다.
    """
    close = pd.Series(data["Close"].to_numpy().ravel(), dtype="float64")
    volume = pd.Series(data["Volume"].to_numpy().ravel(), dtype="float64")
    high = pd.Series(data["High"].to_numpy().ravel(), dtype="float64")
    low = pd.Series(data["Low"].to_numpy().ravel(), dtype="float64")

    cci = _calculate_cci(high=high, low=low, close=close, window=20)
    obv = _calculate_obv(close=close, volume=volume)
    obv_ma20 = obv.rolling(20).mean()
    rsi = _calculate_rsi(close=close, window=14)

    obv_trend = obv.iloc[-1] - obv.iloc[-8] if len(obv) >= 8 else 0.0

    return {
        "CCI": float(cci.iloc[-1]),
        "OBV": float(obv.iloc[-1]),
        "OBV_MA20": float(obv_ma20.iloc[-1]) if pd.notna(obv_ma20.iloc[-1]) else float(obv.iloc[-1]),
        "OBV_trend": float(obv_trend),
        "RSI": float(rsi.iloc[-1]),
    }


def _build_result(decision: str, cci_ok: bool, rsi_ok: bool, obv_ok: bool, reasons: list[str]) -> Dict[str, object]:
    """
    조건 충족 여부를 가중치 점수로 환산한다.
    """
    score = round(
        cci_ok * CCI_WEIGHT * 100
        + rsi_ok * RSI_WEIGHT * 100
        + obv_ok * OBV_WEIGHT * 100
    )

    matched = sum([cci_ok, rsi_ok, obv_ok])
    strength = round((matched / 3) * 100)

    if decision == "매수":
        recommendation = "매수" if score >= STRONG_BUY_THRESHOLD else "매수X"
        normalized_direction = 1
    else:
        recommendation = "매도" if score >= STRONG_SELL_THRESHOLD else "매도X"
        normalized_direction = -1

    if strength >= 80:
        color = "#4CAF50"
        level = "매우 강함"
    elif strength >= 60:
        color = "#8BC34A"
        level = "강함"
    elif strength >= 40:
        color = "#FFC107"
        level = "보통"
    else:
        color = "#F44336"
        level = "약함"

    return {
        "recommendation": recommendation,
        "score": score,
        "strength": strength,
        "color": color,
        "level": level,
        "reason": " ".join(reasons),
        "normalized_direction": normalized_direction,
    }


def generate_signal(indicators: Dict[str, float], decision: str) -> Dict[str, object]:
    """
    수정된 매매판별 로직.

    매수 조건
    - CCI >= +100
    - RSI >= 50
    - OBV 상승전환 및 OBV >= OBV_MA20

    매도 조건
    - CCI <= -100
    - RSI <= 50
    - OBV 하락전환 및 OBV <= OBV_MA20
    """
    cci = float(indicators.get("CCI", 0))
    obv = float(indicators.get("OBV", 0))
    obv_ma20 = float(indicators.get("OBV_MA20", obv))
    obv_trend = float(indicators.get("OBV_trend", 0))
    rsi = float(indicators.get("RSI", 50))

    if decision == "매수":
        cci_ok = cci >= 100
        rsi_ok = rsi >= 50
        obv_ok = (obv_trend > 0) and (obv >= obv_ma20)

        reasons = [
            "CCI가 +100 이상으로 상방 추세가 강합니다." if cci_ok else "CCI가 +100 미만이라 상승 추세 확정이 약합니다.",
            "RSI가 50 이상으로 상승 탄력이 유지됩니다." if rsi_ok else "RSI가 50 미만이라 상승 탄력이 약합니다.",
            "OBV가 상승전환되었고 20일 평균 이상입니다." if obv_ok else "OBV가 상승전환이 아니거나 20일 평균 아래입니다.",
        ]
    else:
        cci_ok = cci <= -100
        rsi_ok = rsi <= 50
        obv_ok = (obv_trend < 0) and (obv <= obv_ma20)

        reasons = [
            "CCI가 -100 이하로 하방 추세가 강합니다." if cci_ok else "CCI가 -100 초과라 하락 추세 확정이 약합니다.",
            "RSI가 50 이하로 하락 탄력이 유지됩니다." if rsi_ok else "RSI가 50 초과라 하락 탄력이 약합니다.",
            "OBV가 하락전환되었고 20일 평균 이하입니다." if obv_ok else "OBV가 하락전환이 아니거나 20일 평균 위입니다.",
        ]

    return _build_result(decision, cci_ok, rsi_ok, obv_ok, reasons)


def calculate_effect(decision: str, return_after_7d: float) -> float:
    """
    판별효과를 방향 정규화 기준으로 계산한다.
    매수는 그대로, 매도는 부호를 반전한다.
    """
    direction = 1 if decision == "매수" else -1
    return float(return_after_7d) * direction


def simulate_trade_log(file_path: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
    """
    거래일지 엑셀을 읽어 방향 정규화 기준의 성과를 계산한다.
    """
    df = pd.read_excel(file_path, sheet_name=0 if sheet_name is None else sheet_name)

    if "매수/매도" not in df.columns or "7일후 등락" not in df.columns:
        raise ValueError("엑셀에 '매수/매도'와 '7일후 등락' 열이 필요합니다.")

    action_col = None
    for col in df.columns:
        if str(col) == "매수/매도":
            sample = df[col].dropna().astype(str)
            if sample.str.contains("매수|매도").any():
                action_col = col
                break

    if action_col is None:
        raise ValueError("매수/매도 방향 열을 찾지 못했습니다.")

    sim = df.copy()
    sim["방향"] = sim[action_col].astype(str).str.contains("매수").map({True: 1, False: -1})
    sim["정규화등락"] = sim["7일후 등락"].astype(float) * sim["방향"]
    sim["성공여부"] = sim["정규화등락"] > 0

    columns = [action_col, "7일후 등락", "정규화등락", "성공여부"]
    if "점수" in sim.columns:
        columns.insert(1, "점수")

    return sim[columns]


def summarize_simulation(sim_df: pd.DataFrame) -> pd.DataFrame:
    """
    방향 정규화 기준의 평균효과와 승률을 요약한다.
    """
    action_col = sim_df.columns[0]
    return (
        sim_df.groupby(action_col)
        .agg(
            건수=("정규화등락", "size"),
            평균판별효과=("정규화등락", "mean"),
            중앙판별효과=("정규화등락", "median"),
            승률=("성공여부", "mean"),
        )
        .reset_index()
    )
