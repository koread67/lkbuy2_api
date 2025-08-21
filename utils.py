
# trade_decider_v2.py
# 매매판별 2.0 (English filename for download)
# - Implements "최적조합" weighted scoring (CCI, RSI, OBV)
# - Strength is reported as 0~100%: below min-threshold -> 0%, between min~max -> 1~100%
# - Keeps generate_signal(indicators, decision) signature for drop-in use
from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from typing import Dict, Optional

import pandas as pd
import requests
import ta
from bs4 import BeautifulSoup


# === Constants ===
KOREA_ETF_CODES = [
    "069500", "102110", "114800", "117700", "118260", "122630", "139260", "157450",
    "214980", "219480", "229200", "232080", "233740", "237370", "238720", "251340",
    "252670", "253150", "263750", "272220", "278530", "292150", "295820", "305720",
    "307510", "310970", "352560", "364980", "371160", "379800", "394280", "411060",
    "426410", "438420", "461340"
]


@dataclass
class OptimalCombo:
    w_cci: float = 0.33
    w_rsi: float = 0.33
    w_obv: float = 0.34
    buy_min: float = 50.0   # "강매수 임계값"
    sell_min: float = 90.0  # "강매도 임계값"
    max_threshold: float = 100.0  # Upper bound for percent mapping

    @classmethod
    def from_text(cls, text: str) -> "OptimalCombo":
        # Very tolerant parser for the provided "최적 조합.txt" format
        import re
        w_cci = 0.33
        w_rsi = 0.33
        w_obv = 0.34
        buy_min = 50.0
        sell_min = 90.0
        for line in text.splitlines():
            s = line.strip()
            if not s:
                continue
            if "CCI" in s and "가중치" in s:
                m = re.search(r"([0-9]*\.?[0-9]+)", s)
                if m: w_cci = float(m.group(1))
            elif "RSI" in s and "가중치" in s:
                m = re.search(r"([0-9]*\.?[0-9]+)", s)
                if m: w_rsi = float(m.group(1))
            elif "OBV" in s and "가중치" in s:
                m = re.search(r"([0-9]*\.?[0-9]+)", s)
                if m: w_obv = float(m.group(1))
            elif "강매수" in s and "임계값" in s:
                m = re.search(r"([0-9]*\.?[0-9]+)", s)
                if m: buy_min = float(m.group(1))
            elif "강매도" in s and "임계값" in s:
                m = re.search(r"([0-9]*\.?[0-9]+)", s)
                if m: sell_min = float(m.group(1))
        return cls(w_cci=w_cci, w_rsi=w_rsi, w_obv=w_obv, buy_min=buy_min, sell_min=sell_min)


def load_optimal_combo(file_path: Optional[str] = None) -> OptimalCombo:
    # Attempts to read the on-disk "최적 조합.txt"; falls back to defaults if missing.
    if not file_path:
        file_path = "최적 조합.txt"
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return OptimalCombo.from_text(f.read())
    except Exception:
        return OptimalCombo()


# === Data & Indicators ===
def get_naver_etf_price(code: str, pages: int = 5) -> Optional[pd.DataFrame]:
    try:
        if not code.isdigit():
            return None
        url = f"https://finance.naver.com/item/sise_day.naver?code={code}"
        headers = {"User-Agent": "Mozilla/5.0"}
        dfs = []
        for page in range(1, pages + 1):
            res = requests.get(f"{url}&page={page}", headers=headers, timeout=10)
            soup = BeautifulSoup(res.text, "html.parser")
            table = soup.select_one("table[type='N']")
            if table is None:
                continue
            df = pd.read_html(StringIO(str(table)))[0].dropna()
            dfs.append(df)
        if not dfs:
            return None
        df_all = pd.concat(dfs, ignore_index=True)
        df_all.columns = ["Date", "Close", "Change", "Open", "High", "Low", "Volume"]
        df_all["Date"] = pd.to_datetime(df_all["Date"])
        df_all = df_all.sort_values("Date")
        df_all.set_index("Date", inplace=True)
        return df_all[["Open", "High", "Low", "Close", "Volume"]]
    except Exception:
        return None


def calculate_indicators(data: pd.DataFrame) -> Dict[str, float]:
    close = pd.Series(data["Close"].to_numpy().ravel())
    volume = pd.Series(data["Volume"].to_numpy().ravel())
    high = pd.Series(data["High"].to_numpy().ravel())
    low = pd.Series(data["Low"].to_numpy().ravel())

    cci_indicator = ta.trend.CCIIndicator(high=high, low=low, close=close, window=20)
    cci = float(cci_indicator.cci().iloc[-1])

    obv_indicator = ta.volume.OnBalanceVolumeIndicator(close=close, volume=volume)
    obv = obv_indicator.on_balance_volume()
    obv_trend = float(obv.iloc[-1] - obv.iloc[-8]) if len(obv) >= 8 else 0.0

    rsi_indicator = ta.momentum.RSIIndicator(close=close, window=14)
    rsi = float(rsi_indicator.rsi().iloc[-1])

    return {"CCI": cci, "OBV_trend": obv_trend, "RSI": rsi}


# === Normalization helpers (0~100 bullish/bearish scores) ===
def _clip01(x: float) -> float:
    return 0.0 if x < 0 else (1.0 if x > 1 else x)


def _bullish_components(cci: float, rsi: float, obv_trend: float) -> Dict[str, float]:
    # Lower CCI and RSI are bullish for "매수" side.
    cci_bull = _clip01((0.0 - cci) / 200.0) * 100.0          # cci <= -200 -> 100, cci >= 0 -> 0
    rsi_bull = _clip01((50.0 - rsi) / 20.0) * 100.0          # rsi <= 30 -> 100, rsi >= 50 -> 0
    obv_bull = 100.0 if obv_trend > 0 else 0.0               # sign-based
    return {"CCI": cci_bull, "RSI": rsi_bull, "OBV": obv_bull}


def _bearish_components(cci: float, rsi: float, obv_trend: float) -> Dict[str, float]:
    # Higher CCI and RSI are bearish for "매도" side.
    cci_bear = _clip01(cci / 200.0) * 100.0                  # cci >= +200 -> 100, cci <= 0 -> 0
    rsi_bear = _clip01((rsi - 50.0) / 20.0) * 100.0          # rsi >= 70 -> 100, rsi <= 50 -> 0
    obv_bear = 100.0 if obv_trend < 0 else 0.0               # sign-based
    return {"CCI": cci_bear, "RSI": rsi_bear, "OBV": obv_bear}


def _weighted_sum(components: Dict[str, float], combo: OptimalCombo) -> float:
    return components["CCI"] * combo.w_cci + components["RSI"] * combo.w_rsi + components["OBV"] * combo.w_obv


def _percent_from_threshold(score: float, min_thr: float, max_thr: float) -> int:
    # Below min -> 0%; at min -> 1%; at/above max -> 100%
    if score < min_thr:
        return 0
    if score >= max_thr:
        return 100
    # Map linearly to 1..100
    return int(1 + (score - min_thr) * 99.0 / (max_thr - min_thr))


def _level_from_percent(p: int) -> str:
    if p == 0:
        return "미달"
    if p >= 71:
        return "매우강함"
    if p >= 41:
        return "강함"
    return "적정"  # 1~40%


def _color_from_percent(decision: str, p: int) -> str:
    # Green-ish when strong, yellow mid, orange low, red when 0%
    if p == 0:
        return "#F44336"  # red
    if p >= 71:
        return "#2E7D32"  # strong green
    if p >= 41:
        return "#FBC02D"  # amber
    return "#FB8C00"      # orange


# === Public API ===
def generate_signal(indicators: Dict[str, float], decision: str) -> Dict[str, object]:
    """
    Args:
        indicators: {"CCI": float, "OBV_trend": float, "RSI": float}
        decision: "매수" 또는 "매도"
    Returns:
        {
            "recommendation": "매수"/"매도" or "매수X"/"매도X",
            "score": int,            # raw weighted score (0~100 rounded)
            "strength_pct": int,     # 0~100% (0 when below min threshold; 1~100 within min~max)
            "color": str,
            "level": str,            # 적정/강함/매우강함/미달
            "reason": str,           # 간단 사유
            "components": {...},     # 지표별 0~100 기여도
            "thresholds": {"min": float, "max": float}
        }
    """
    combo = load_optimal_combo()  # Falls back to defaults if the file is missing.
    cci = float(indicators.get("CCI", 0.0))
    obv_trend = float(indicators.get("OBV_trend", 0.0))
    rsi = float(indicators.get("RSI", 0.0))

    bull = _bullish_components(cci, rsi, obv_trend)
    bear = _bearish_components(cci, rsi, obv_trend)

    buy_score = _weighted_sum(bull, combo)     # 0~100
    sell_score = _weighted_sum(bear, combo)    # 0~100

    if decision == "매수":
        min_thr = combo.buy_min
        max_thr = combo.max_threshold
        strength_pct = _percent_from_threshold(buy_score, min_thr, max_thr)
        score_to_report = int(round(buy_score))
        recommendation = "매수" if strength_pct > 0 else "매수X"
        components = bull
        reason = []
        reason.append(f"CCI={cci:.0f} → 매수 관점 {bull['CCI']:.0f}점")
        reason.append(f"RSI={rsi:.0f} → 매수 관점 {bull['RSI']:.0f}점")
        reason.append(f"OBV 추세={'상승' if obv_trend>0 else '하락/정체'} → {bull['OBV']:.0f}점")
    else:
        min_thr = combo.sell_min
        max_thr = combo.max_threshold
        strength_pct = _percent_from_threshold(sell_score, min_thr, max_thr)
        score_to_report = int(round(sell_score))
        recommendation = "매도" if strength_pct > 0 else "매도X"
        components = bear
        reason = []
        reason.append(f"CCI={cci:.0f} → 매도 관점 {bear['CCI']:.0f}점")
        reason.append(f"RSI={rsi:.0f} → 매도 관점 {bear['RSI']:.0f}점")
        reason.append(f"OBV 추세={'하락' if obv_trend<0 else '상승/정체'} → {bear['OBV']:.0f}점")

    level = _level_from_percent(strength_pct)
    color = _color_from_percent(decision, strength_pct)

    return {
        "recommendation": recommendation,
        "score": score_to_report,
        "strength_pct": int(strength_pct),
        "color": color,
        "level": level,
        "reason": "\n".join(reason),  # 줄바꿈으로 임계/사유 구분 표시 용이
        "components": {k: int(round(v)) for k, v in components.items()},
        "thresholds": {"min": float(min_thr), "max": float(max_thr)},
        "weights": {"CCI": combo.w_cci, "RSI": combo.w_rsi, "OBV": combo.w_obv},
    }
