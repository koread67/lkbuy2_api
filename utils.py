
# -*- coding: utf-8 -*-
"""
utils_strength_level_pct_final.py
- BUY 임계값 50, SELL 임계값 90 (최적 조합 기준 반영)
- 강도 표기 형식: "레벨 / XX%"
- 매수/매도 각각 임계값~100 구간을 0~100%로 선형 환산
- 기존 출력 키: recommendation / score / strength / color / level / reason
"""

import pandas as pd
import ta
import requests
from bs4 import BeautifulSoup
from io import StringIO
import numpy as np

# ======================== 기본 설정 ========================
KOREA_ETF_CODES = [
    "069500", "102110", "114800", "117700", "118260", "122630", "139260", "157450",
    "214980", "219480", "229200", "232080", "233740", "237370", "238720", "251340",
    "252670", "253150", "263750", "272220", "278530", "292150", "295820", "305720",
    "307510", "310970", "352560", "364980", "371160", "379800", "394280", "411060",
    "426410", "438420", "461340"
]

# --- 민감도 프로파일 (연 10회 내외 목표) ---
CCI_LEN   = 14
RSI_LEN   = 9
OBV_DIFF  = 2    # OBV의 변화폭을 짧게 보아 단기 자금유입/이탈에 더 민감
BUY_THRESHOLD  = 50
SELL_THRESHOLD = 90  # 최적 조합 기준으로 상향

# 가중치
W_CCI = 0.33
W_RSI = 0.33
W_OBV = 0.34

# ======================== 데이터 수집 ========================
def get_naver_etf_price(code: str, pages: int = 5):
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
            df = pd.read_html(StringIO(str(table)))[0].dropna()
            dfs.append(df)
        df_all = pd.concat(dfs, ignore_index=True)
        df_all.columns = ['Date', 'Close', 'Change', 'Open', 'High', 'Low', 'Volume']
        df_all['Date'] = pd.to_datetime(df_all['Date'])
        df_all = df_all.sort_values('Date')
        df_all.set_index('Date', inplace=True)
        return df_all[['Open', 'High', 'Low', 'Close', 'Volume']]
    except Exception:
        return None

# ======================== 스코어링 유틸 ========================
def _scale_linear(x, x0, x1, y0=0.0, y1=100.0):
    if x0 == x1:
        return (y0 + y1) / 2.0
    if x <= min(x0, x1):
        return y0 if x0 < x1 else y1
    if x >= max(x0, x1):
        return y1 if x0 < x1 else y0
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

def _obv_continuous_score(obv_series: pd.Series) -> pd.Series:
    """
    OBV 연속 점수(0~100) 생성:
      - OBV_trend = OBV.diff(OBV_DIFF)
      - 20일 표준편차로 z-score → tanh로 매핑 → 0~100
    """
    obv_trend = obv_series.diff(OBV_DIFF)
    rolling_std = obv_trend.rolling(20, min_periods=20).std()
    z = obv_trend / (rolling_std.replace(0, np.nan))
    score = 50.0 + 50.0 * np.tanh((z.fillna(0.0)) / 1.5)
    return score.clip(0, 100), obv_trend

# ======================== 지표 계산 ========================
def calculate_indicators(data: pd.DataFrame) -> dict:
    """
    Returns dict with keys: CCI, RSI, OBV, OBV_trend, OBV_SCORE
    (OBV_SCORE는 0~100 연속 점수)
    """
    close = pd.Series(data["Close"].to_numpy().ravel(), index=data.index)
    volume = pd.Series(data["Volume"].to_numpy().ravel(), index=data.index)
    high = pd.Series(data["High"].to_numpy().ravel(), index=data.index)
    low = pd.Series(data["Low"].to_numpy().ravel(), index=data.index)

    # CCI, RSI (민감도 상향 기간)
    cci_indicator = ta.trend.CCIIndicator(high=high, low=low, close=close, window=CCI_LEN)
    cci = cci_indicator.cci()

    rsi_indicator = ta.momentum.RSIIndicator(close=close, window=RSI_LEN)
    rsi = rsi_indicator.rsi()

    # OBV + 연속점수
    obv_indicator = ta.volume.OnBalanceVolumeIndicator(close=close, volume=volume)
    obv = obv_indicator.on_balance_volume()
    obv_score, obv_trend = _obv_continuous_score(obv)

    return {
        "CCI": float(cci.iloc[-1]) if len(cci) else 0.0,
        "RSI": float(rsi.iloc[-1]) if len(rsi) else 0.0,
        "OBV": float(obv.iloc[-1]) if len(obv) else 0.0,
        "OBV_trend": float(obv_trend.iloc[-1]) if len(obv_trend) else 0.0,
        "OBV_SCORE": float(obv_score.iloc[-1]) if len(obv_score) else 50.0,
    }

# ======================== 컴포넌트 스코어 ========================
def _component_scores_for_buy(cci, rsi, obv_score):
    cci_s = _scale_linear(cci, -100, 100, 100, 0)   # 과매도일수록 가산
    rsi_s = _scale_linear(rsi, 30, 70, 100, 0)      # 30 이하 우수
    obv_s = float(obv_score)                        # 0~100
    return cci_s, rsi_s, obv_s

def _component_scores_for_sell(cci, rsi, obv_score):
    cci_s = _scale_linear(cci, -100, 100, 0, 100)   # 과매수일수록 가산
    rsi_s = _scale_linear(rsi, 30, 70, 0, 100)      # 70 이상 우수
    obv_s = 100.0 - float(obv_score)                # 유출 쪽에 가산
    return cci_s, rsi_s, obv_s

def score_buy(cci, rsi, obv_score):
    cci_s, rsi_s, obv_s = _component_scores_for_buy(cci, rsi, obv_score)
    return W_CCI * cci_s + W_RSI * rsi_s + W_OBV * obv_s

def score_sell(cci, rsi, obv_score):
    cci_s, rsi_s, obv_s = _component_scores_for_sell(cci, rsi, obv_score)
    return W_CCI * cci_s + W_RSI * rsi_s + W_OBV * obv_s

# ======================== 신호 생성 ========================
def _percent_from_threshold(score_f: float, threshold: float) -> float:
    """임계값~100 구간을 0~100%로 환산"""
    if score_f < threshold:
        return 0.0
    if threshold >= 100:
        return 100.0 if score_f >= threshold else 0.0
    return max(0.0, min(100.0, (score_f - threshold) / (100.0 - threshold) * 100.0))

def generate_signal(indicators: dict, decision: str):
    """
    decision: "매수" 또는 "매도"
    반환: { recommendation, score, strength, color, level, reason }
    """
    cci = indicators.get("CCI", 0.0)
    rsi = indicators.get("RSI", 0.0)
    obv_score = indicators.get("OBV_SCORE", None)

    # 하위호환: OBV_SCORE가 없으면 방향성으로만 0/50/100 근사
    if obv_score is None:
        obv_trend = indicators.get("OBV_trend", 0.0)
        obv_score = 100.0 if obv_trend > 0 else (0.0 if obv_trend < 0 else 50.0)

    if decision == "매수":
        score_f = score_buy(cci, rsi, obv_score)
        recommendation = "매수" if score_f >= BUY_THRESHOLD else "매수X"
        # 단계/색상
        if score_f >= 90:
            level, color = "매수 매우 강함", "#2E7D32"
        elif score_f >= 70:
            level, color = "매수 강함", "#4CAF50"
        elif score_f >= BUY_THRESHOLD:
            level, color = "매수 적정", "#FFEB3B"
        else:
            level, color = "약함", "#F44336"
        percent = _percent_from_threshold(score_f, BUY_THRESHOLD)
        reasons = [
            ("CCI 과매도" if cci <= -100 else "CCI 중립" if -100 < cci < 100 else "CCI 과매수"),
            ("OBV 유입 강함" if obv_score >= 60 else "OBV 정체" if 40 <= obv_score < 60 else "OBV 유출"),
            ("RSI 과매도(≤30)" if rsi <= 30 else "RSI 중립(30~70)" if 30 < rsi < 70 else "RSI 과매수(≥70)"),
        ]
    else:
        score_f = score_sell(cci, rsi, obv_score)
        recommendation = "매도" if score_f >= SELL_THRESHOLD else "매도X"
        if score_f >= 90:
            level, color = "매도 매우 강함", "#B71C1C"
        elif score_f >= 80:
            level, color = "매도 강함", "#E53935"
        elif score_f >= SELL_THRESHOLD:
            level, color = "매도 적정", "#FF9800"
        else:
            level, color = "약함", "#9E9E9E"
        percent = _percent_from_threshold(score_f, SELL_THRESHOLD)
        reasons = [
            ("CCI 과매수" if cci >= 100 else "CCI 중립" if -100 < cci < 100 else "CCI 과매도"),
            ("OBV 유출 강함" if obv_score <= 40 else "OBV 정체" if 40 < obv_score < 60 else "OBV 유입"),
            ("RSI 과매수(≥70)" if rsi >= 70 else "RSI 중립(30~70)" if 30 < rsi < 70 else "RSI 과매도(≤30)"),
        ]

    score_int = int(round(score_f))
    strength_text = f"{level} / {percent:.0f}%"
    return {
        "recommendation": recommendation,
        "score": score_int,
        "strength": strength_text,
        "color": color,
        "level": level,
        "reason": " / ".join(reasons)
    }
