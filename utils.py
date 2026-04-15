import math
from typing import Any

import numpy as np
import pandas as pd

# =========================
# Trading Logic 2.0
# =========================
# 사용자 입력 조건을 점수화하는 구조
# - RSI / OBV / DMI 가중치 반영
# - VIX는 독립 지표가 아니라 필터로 반영
# - 최종적으로 추천, 점수, 강도, 비중, 사유를 반환

RSI_WEIGHT = 17
OBV_WEIGHT = 33
DMI_WEIGHT = 40

BUY_THRESHOLD = 81
SELL_THRESHOLD = 104

VIX_BUY_PENALTY = 15
VIX_SELL_BONUS = 10

BUY_SCORE_MAX = RSI_WEIGHT + OBV_WEIGHT + DMI_WEIGHT  # 90
SELL_SCORE_MAX = RSI_WEIGHT + OBV_WEIGHT + DMI_WEIGHT + VIX_SELL_BONUS  # 100


def _safe_last(series: pd.Series) -> float:
    """마지막 값을 float로 반환. 값이 없으면 NaN 반환."""
    if series is None or len(series) == 0:
        return float("nan")
    value = series.iloc[-1]
    if pd.isna(value):
        return float("nan")
    return float(value)


def _clean_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        if math.isfinite(result):
            return result
    except Exception:
        pass
    return default


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _to_bool_label(flag: bool) -> str:
    return "True" if flag else "False"


def _build_position_size(strength: float) -> int:
    """강도에 따른 추천 비중(%)"""
    if strength <= 0:
        return 0
    if strength < 20:
        return 30
    if strength < 50:
        return 50
    if strength < 80:
        return 70
    return 100


def _build_reason_text(
    *,
    decision: str,
    rsi_ok: bool,
    obv_ok: bool,
    dmi_ok: bool,
    vix_risk: bool,
    buy_score: int,
    sell_score: int,
    applied_filter: bool,
) -> str:
    reasons: list[str] = []

    if decision == "매수":
        if rsi_ok:
            reasons.append("RSI 매수")
        if obv_ok:
            reasons.append("OBV 매수")
        if dmi_ok:
            reasons.append("DMI 매수")
        if not vix_risk:
            reasons.append("VIX 안정")
        elif applied_filter:
            reasons.append("VIX 위험으로 매수점수 감점")
        reasons.append(f"매수점수 {buy_score}")
        reasons.append(f"매도점수 {sell_score}")
    else:
        if rsi_ok:
            reasons.append("RSI 매도")
        if obv_ok:
            reasons.append("OBV 매도")
        if dmi_ok:
            reasons.append("DMI 매도")
        if vix_risk:
            reasons.append("VIX 위험")
        reasons.append(f"매수점수 {buy_score}")
        reasons.append(f"매도점수 {sell_score}")
        if applied_filter:
            reasons.append("VIX 필터로 매도점수 가점")

    return ", ".join(reasons) if reasons else "판단 근거 부족"


def _score_to_strength(score: int, threshold: int, max_score: int) -> int:
    if score < threshold:
        return 0
    denominator = max(max_score - threshold, 1)
    strength = ((score - threshold) / denominator) * 100
    return int(round(_clamp(strength, 0, 100)))


def calculate_indicators(data: pd.DataFrame) -> dict:
    """
    입력 컬럼
    - 필수: Close, High, Low, Volume
    - 선택: VIX
    """
    required_cols = ["Close", "High", "Low", "Volume"]
    missing = [col for col in required_cols if col not in data.columns]
    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {missing}")

    df = data.copy()

    close = pd.to_numeric(df["Close"], errors="coerce")
    high = pd.to_numeric(df["High"], errors="coerce")
    low = pd.to_numeric(df["Low"], errors="coerce")
    volume = pd.to_numeric(df["Volume"], errors="coerce")

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    # OBV
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    obv_ma20 = obv.rolling(20).mean()
    obv_trend = obv.diff(7)

    # DMI / ADX
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=df.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=df.index,
    )

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr.replace(0, np.nan))

    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    adx = dx.rolling(14).mean()

    # VIX
    if "VIX" in df.columns:
        vix = pd.to_numeric(df["VIX"], errors="coerce")
        vix_5ma = vix.rolling(5).mean()
    else:
        vix = pd.Series(np.nan, index=df.index)
        vix_5ma = pd.Series(np.nan, index=df.index)

    return {
        "RSI": _safe_last(rsi),
        "OBV": _safe_last(obv),
        "OBV_MA20": _safe_last(obv_ma20),
        "OBV_trend": _safe_last(obv_trend),
        "PLUS_DI": _safe_last(plus_di),
        "MINUS_DI": _safe_last(minus_di),
        "ADX": _safe_last(adx),
        "VIX": _safe_last(vix),
        "VIX_5MA": _safe_last(vix_5ma),
    }


def generate_signal(indicators: dict, decision: str) -> dict:
    """decision: '매수' 또는 '매도'"""
    if decision not in ["매수", "매도"]:
        raise ValueError("decision은 '매수' 또는 '매도'여야 합니다.")

    rsi = indicators.get("RSI", np.nan)
    obv = indicators.get("OBV", np.nan)
    obv_ma20 = indicators.get("OBV_MA20", np.nan)
    obv_trend = indicators.get("OBV_trend", np.nan)
    plus_di = indicators.get("PLUS_DI", np.nan)
    minus_di = indicators.get("MINUS_DI", np.nan)
    vix = indicators.get("VIX", np.nan)
    vix_5ma = indicators.get("VIX_5MA", np.nan)

    base_values = [rsi, obv, obv_ma20, obv_trend, plus_di, minus_di]
    if any(pd.isna(v) for v in base_values):
        return {
            "recommendation": "관망",
            "score": 0,
            "strength": 0,
            "position_size": 0,
            "color": "#9E9E9E",
            "reason": "지표 계산 데이터 부족",
        }

    vix_risk = bool(pd.notna(vix) and pd.notna(vix_5ma) and vix > vix_5ma)

    # 사용자가 입력한 매수/매도 조건을 점수화하는 구조
    buy_rsi_ok = bool(_clean_float(rsi) >= 50)
    buy_obv_ok = bool(_clean_float(obv_trend) > 0 and _clean_float(obv) >= _clean_float(obv_ma20))
    buy_dmi_ok = bool(_clean_float(plus_di) > _clean_float(minus_di))

    sell_rsi_ok = bool(_clean_float(rsi) <= 50)
    sell_obv_ok = bool(_clean_float(obv_trend) < 0 and _clean_float(obv) <= _clean_float(obv_ma20))
    sell_dmi_ok = bool(_clean_float(minus_di) > _clean_float(plus_di))

    buy_score = 0
    if buy_rsi_ok:
        buy_score += RSI_WEIGHT
    if buy_obv_ok:
        buy_score += OBV_WEIGHT
    if buy_dmi_ok:
        buy_score += DMI_WEIGHT

    sell_score = 0
    if sell_rsi_ok:
        sell_score += RSI_WEIGHT
    if sell_obv_ok:
        sell_score += OBV_WEIGHT
    if sell_dmi_ok:
        sell_score += DMI_WEIGHT

    buy_filter_applied = False
    sell_filter_applied = False
    if vix_risk:
        buy_score = max(0, buy_score - VIX_BUY_PENALTY)
        sell_score += VIX_SELL_BONUS
        buy_filter_applied = True
        sell_filter_applied = True

    buy_strength = _score_to_strength(buy_score, BUY_THRESHOLD, BUY_SCORE_MAX)
    sell_strength = _score_to_strength(sell_score, SELL_THRESHOLD, SELL_SCORE_MAX)

    buy_position = _build_position_size(buy_strength)
    sell_position = _build_position_size(sell_strength)

    buy_result = {
        "recommendation": "매수" if buy_score >= BUY_THRESHOLD else "관망",
        "score": int(buy_score),
        "strength": int(buy_strength),
        "position_size": int(buy_position),
        "color": "#2196F3" if buy_score >= BUY_THRESHOLD else "#9E9E9E",
        "reason": _build_reason_text(
            decision="매수",
            rsi_ok=buy_rsi_ok,
            obv_ok=buy_obv_ok,
            dmi_ok=buy_dmi_ok,
            vix_risk=vix_risk,
            buy_score=int(buy_score),
            sell_score=int(sell_score),
            applied_filter=buy_filter_applied,
        ),
        "matched": {
            "RSI": _to_bool_label(buy_rsi_ok),
            "OBV": _to_bool_label(buy_obv_ok),
            "DMI": _to_bool_label(buy_dmi_ok),
            "VIX_RISK": _to_bool_label(vix_risk),
        },
    }

    sell_result = {
        "recommendation": "매도" if sell_score >= SELL_THRESHOLD else "관망",
        "score": int(sell_score),
        "strength": int(sell_strength),
        "position_size": int(sell_position),
        "color": "#F44336" if sell_score >= SELL_THRESHOLD else "#9E9E9E",
        "reason": _build_reason_text(
            decision="매도",
            rsi_ok=sell_rsi_ok,
            obv_ok=sell_obv_ok,
            dmi_ok=sell_dmi_ok,
            vix_risk=vix_risk,
            buy_score=int(buy_score),
            sell_score=int(sell_score),
            applied_filter=sell_filter_applied,
        ),
        "matched": {
            "RSI": _to_bool_label(sell_rsi_ok),
            "OBV": _to_bool_label(sell_obv_ok),
            "DMI": _to_bool_label(sell_dmi_ok),
            "VIX_RISK": _to_bool_label(vix_risk),
        },
    }

    return buy_result if decision == "매수" else sell_result


def generate_dual_signal(indicators: dict) -> dict:
    """매수/매도를 동시에 평가해서 더 강한 쪽을 반환. 둘 다 기준 미달이면 관망."""
    buy_signal = generate_signal(indicators, "매수")
    sell_signal = generate_signal(indicators, "매도")

    buy_ok = buy_signal["recommendation"] == "매수"
    sell_ok = sell_signal["recommendation"] == "매도"

    if buy_ok and not sell_ok:
        final_decision = "매수"
    elif sell_ok and not buy_ok:
        final_decision = "매도"
    elif buy_ok and sell_ok:
        if buy_signal["score"] > sell_signal["score"]:
            final_decision = "매수"
        elif sell_signal["score"] > buy_signal["score"]:
            final_decision = "매도"
        elif buy_signal["strength"] > sell_signal["strength"]:
            final_decision = "매수"
        elif sell_signal["strength"] > buy_signal["strength"]:
            final_decision = "매도"
        else:
            final_decision = "관망"
    else:
        final_decision = "관망"

    final_position = 0
    final_reason = "매수/매도 임계값 미달"
    final_color = "#9E9E9E"

    if final_decision == "매수":
        final_position = int(buy_signal.get("position_size", 0))
        final_reason = str(buy_signal.get("reason", ""))
        final_color = str(buy_signal.get("color", "#2196F3"))
    elif final_decision == "매도":
        final_position = int(sell_signal.get("position_size", 0))
        final_reason = str(sell_signal.get("reason", ""))
        final_color = str(sell_signal.get("color", "#F44336"))

    return {
        "final_decision": final_decision,
        "position_size": final_position,
        "color": final_color,
        "reason": final_reason,
        "buy_signal": buy_signal,
        "sell_signal": sell_signal,
    }
