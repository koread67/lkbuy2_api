import math
from typing import Any

import numpy as np
import pandas as pd

# =========================
# Trading Logic 2.3
# =========================
# - 00통합.xlsx 자동 탐색 결과 반영
# - 추격형 매수/매도 완화를 위해 '위치 필터'를 독립 점수로 반영
# - 매수: 최근 20일 고점 대비 -5% 이하에서 전환 신호가 나올 때 우대
# - 매도: 최근 20일 고점 대비 -1% 이내 또는 상승 피로 구간에서 우대
# - 기존 앱 연동을 위해 calculate_indicators / generate_signal / generate_dual_signal 구조 유지

RSI_WEIGHT = 25
OBV_WEIGHT = 20
CCI_WEIGHT = 15
POSITION_WEIGHT = 40

BUY_THRESHOLD = 70
SELL_THRESHOLD = 65

SCORE_MAX = RSI_WEIGHT + OBV_WEIGHT + CCI_WEIGHT + POSITION_WEIGHT  # 100

BUY_HIGH_DISCOUNT = -5.0      # 최근 20일 고점 대비 -5% 이하
SELL_HIGH_NEAR = -1.0         # 최근 20일 고점 대비 -1% 이내
RSI_DELTA_BUY_MIN = 2.0       # RSI 3일 변화율
OBV_GAP_MIN = 0.5             # OBV와 OBV MA20 간 갭(%)

VIX_BUY_PENALTY = 10
VIX_SELL_BONUS = 5


def _safe_last(series: pd.Series) -> float:
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
    if strength <= 0:
        return 0
    if strength < 20:
        return 30
    if strength < 50:
        return 50
    if strength < 80:
        return 70
    return 100


def _score_to_strength(score: int, threshold: int, max_score: int = SCORE_MAX) -> int:
    if score < threshold:
        return 0
    denominator = max(max_score - threshold, 1)
    strength = ((score - threshold) / denominator) * 100
    return int(round(_clamp(strength, 0, 100)))


def _gap_percent(value: float, base: float) -> float:
    if pd.isna(value) or pd.isna(base) or abs(_clean_float(base)) < 1e-12:
        return 0.0
    return ((_clean_float(value) / _clean_float(base)) - 1.0) * 100.0


def calculate_indicators(data: pd.DataFrame) -> dict:
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
    obv_gap = ((obv / obv_ma20.replace(0, np.nan)) - 1) * 100

    # CCI
    typical_price = (high + low + close) / 3
    tp_ma20 = typical_price.rolling(20).mean()
    mean_dev = (typical_price - tp_ma20).abs().rolling(20).mean()
    cci = (typical_price - tp_ma20) / (0.015 * mean_dev.replace(0, np.nan))
    cci_delta3 = cci.diff(3)

    # DMI / ADX: 보조 참고 지표로 유지
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
    adx_delta3 = adx.diff(3)

    # 위치 필터
    price_ma20 = close.rolling(20).mean()
    dist20 = ((close / price_ma20.replace(0, np.nan)) - 1) * 100
    high20 = high.rolling(20).max()
    high20_dist = ((close / high20.replace(0, np.nan)) - 1) * 100
    low20 = low.rolling(20).min()
    low20_dist = ((close / low20.replace(0, np.nan)) - 1) * 100

    rsi_delta3 = rsi.diff(3)
    close_prev1 = close.shift(1)
    ma5 = close.rolling(5).mean()

    if "VIX" in df.columns:
        vix = pd.to_numeric(df["VIX"], errors="coerce")
        vix_5ma = vix.rolling(5).mean()
    else:
        vix = pd.Series(np.nan, index=df.index)
        vix_5ma = pd.Series(np.nan, index=df.index)

    return {
        "RSI": _safe_last(rsi),
        "RSI_DELTA3": _safe_last(rsi_delta3),
        "OBV": _safe_last(obv),
        "OBV_MA20": _safe_last(obv_ma20),
        "OBV_trend": _safe_last(obv_trend),
        "OBV_GAP": _safe_last(obv_gap),
        "CCI": _safe_last(cci),
        "CCI_DELTA3": _safe_last(cci_delta3),
        "PLUS_DI": _safe_last(plus_di),
        "MINUS_DI": _safe_last(minus_di),
        "ADX": _safe_last(adx),
        "ADX_DELTA3": _safe_last(adx_delta3),
        "PRICE_MA20": _safe_last(price_ma20),
        "DIST20": _safe_last(dist20),
        "HIGH20": _safe_last(high20),
        "HIGH20_DIST": _safe_last(high20_dist),
        "LOW20": _safe_last(low20),
        "LOW20_DIST": _safe_last(low20_dist),
        "CLOSE_PREV1": _safe_last(close_prev1),
        "MA5": _safe_last(ma5),
        "CLOSE": _safe_last(close),
        "VIX": _safe_last(vix),
        "VIX_5MA": _safe_last(vix_5ma),
    }


def _build_reason_text(
    *,
    decision: str,
    buy_rsi_ok: bool,
    buy_obv_ok: bool,
    buy_cci_ok: bool,
    buy_position_ok: bool,
    sell_rsi_ok: bool,
    sell_obv_ok: bool,
    sell_cci_ok: bool,
    sell_position_ok: bool,
    vix_risk: bool,
    buy_score: int,
    sell_score: int,
) -> str:
    reasons: list[str] = []

    if decision == "매수":
        if buy_rsi_ok:
            reasons.append("RSI 전환")
        if buy_obv_ok:
            reasons.append("OBV 수급 개선")
        if buy_cci_ok:
            reasons.append("CCI 반등")
        if buy_position_ok:
            reasons.append("저위치 매수 허용")
        if vix_risk:
            reasons.append("VIX 위험으로 매수 감점")
    else:
        if sell_rsi_ok:
            reasons.append("RSI 둔화")
        if sell_obv_ok:
            reasons.append("OBV 수급 약화")
        if sell_cci_ok:
            reasons.append("CCI 약화")
        if sell_position_ok:
            reasons.append("고위치 매도 허용")
        if vix_risk:
            reasons.append("VIX 위험으로 매도 가점")

    reasons.append(f"매수점수 {buy_score}")
    reasons.append(f"매도점수 {sell_score}")
    return ", ".join(reasons) if reasons else "판단 근거 부족"


def generate_signal(indicators: dict, decision: str) -> dict:
    if decision not in ["매수", "매도"]:
        raise ValueError("decision은 '매수' 또는 '매도'여야 합니다.")

    rsi = indicators.get("RSI", np.nan)
    rsi_delta3 = indicators.get("RSI_DELTA3", np.nan)
    obv = indicators.get("OBV", np.nan)
    obv_ma20 = indicators.get("OBV_MA20", np.nan)
    obv_trend = indicators.get("OBV_trend", np.nan)
    obv_gap = indicators.get("OBV_GAP", np.nan)
    cci = indicators.get("CCI", np.nan)
    cci_delta3 = indicators.get("CCI_DELTA3", np.nan)
    high20_dist = indicators.get("HIGH20_DIST", np.nan)
    plus_di = indicators.get("PLUS_DI", np.nan)
    minus_di = indicators.get("MINUS_DI", np.nan)
    close = indicators.get("CLOSE", np.nan)
    vix = indicators.get("VIX", np.nan)
    vix_5ma = indicators.get("VIX_5MA", np.nan)

    base_values = [rsi, rsi_delta3, obv, obv_ma20, obv_trend, cci, high20_dist, close]
    if any(pd.isna(v) for v in base_values):
        return {
            "recommendation": "관망",
            "score": 0,
            "strength": 0,
            "position_size": 0,
            "color": "#9E9E9E",
            "reason": "지표 계산 데이터 부족",
        }

    rsi_f = _clean_float(rsi)
    rsi_delta_f = _clean_float(rsi_delta3)
    obv_f = _clean_float(obv)
    obv_ma20_f = _clean_float(obv_ma20)
    obv_trend_f = _clean_float(obv_trend)
    obv_gap_f = _clean_float(obv_gap, _gap_percent(obv_f, obv_ma20_f))
    cci_f = _clean_float(cci)
    cci_delta_f = _clean_float(cci_delta3)
    high20_dist_f = _clean_float(high20_dist)
    plus_di_f = _clean_float(plus_di)
    minus_di_f = _clean_float(minus_di)

    vix_risk = bool(pd.notna(vix) and pd.notna(vix_5ma) and _clean_float(vix) > _clean_float(vix_5ma))

    # 매수: 단순 상승 추격이 아니라 '저위치 + 전환'을 우선한다.
    buy_rsi_ok = bool(45 <= rsi_f <= 68 and rsi_delta_f >= RSI_DELTA_BUY_MIN)
    buy_obv_ok = bool(obv_trend_f > 0 and obv_gap_f >= OBV_GAP_MIN)
    buy_cci_ok = bool(-120 <= cci_f <= 120 and cci_delta_f > 0)
    buy_position_ok = bool(high20_dist_f <= BUY_HIGH_DISCOUNT)

    # 매도: 하락 추격을 줄이기 위해 고위치/피로 구간을 중심으로 판단한다.
    sell_rsi_ok = bool((rsi_f >= 60 and rsi_delta_f <= -2) or rsi_f <= 45)
    sell_obv_ok = bool(obv_trend_f < 0 and obv_gap_f <= -OBV_GAP_MIN)
    sell_cci_ok = bool((cci_f >= 100 and cci_delta_f < 0) or (cci_f < -100 and cci_delta_f < 0))
    sell_position_ok = bool(high20_dist_f >= SELL_HIGH_NEAR)

    buy_score = 0
    if buy_rsi_ok:
        buy_score += RSI_WEIGHT
    if buy_obv_ok:
        buy_score += OBV_WEIGHT
    if buy_cci_ok:
        buy_score += CCI_WEIGHT
    if buy_position_ok:
        buy_score += POSITION_WEIGHT

    sell_score = 0
    if sell_rsi_ok:
        sell_score += RSI_WEIGHT
    if sell_obv_ok:
        sell_score += OBV_WEIGHT
    if sell_cci_ok:
        sell_score += CCI_WEIGHT
    if sell_position_ok:
        sell_score += POSITION_WEIGHT

    # DMI는 독립 점수가 아니라 동점/경계 상황 보정으로만 사용한다.
    if pd.notna(plus_di) and pd.notna(minus_di):
        if plus_di_f > minus_di_f and buy_score >= BUY_THRESHOLD - 5:
            buy_score += 5
        if minus_di_f > plus_di_f and sell_score >= SELL_THRESHOLD - 5:
            sell_score += 5

    if vix_risk:
        buy_score = max(0, buy_score - VIX_BUY_PENALTY)
        sell_score += VIX_SELL_BONUS

    buy_score = int(_clamp(buy_score, 0, SCORE_MAX))
    sell_score = int(_clamp(sell_score, 0, SCORE_MAX))

    buy_strength = _score_to_strength(buy_score, BUY_THRESHOLD)
    sell_strength = _score_to_strength(sell_score, SELL_THRESHOLD)
    buy_position = _build_position_size(buy_strength)
    sell_position = _build_position_size(sell_strength)

    buy_result = {
        "recommendation": "매수" if buy_score >= BUY_THRESHOLD else "관망",
        "score": buy_score,
        "strength": int(buy_strength),
        "position_size": int(buy_position),
        "color": "#2196F3" if buy_score >= BUY_THRESHOLD else "#9E9E9E",
        "reason": _build_reason_text(
            decision="매수",
            buy_rsi_ok=buy_rsi_ok,
            buy_obv_ok=buy_obv_ok,
            buy_cci_ok=buy_cci_ok,
            buy_position_ok=buy_position_ok,
            sell_rsi_ok=sell_rsi_ok,
            sell_obv_ok=sell_obv_ok,
            sell_cci_ok=sell_cci_ok,
            sell_position_ok=sell_position_ok,
            vix_risk=vix_risk,
            buy_score=buy_score,
            sell_score=sell_score,
        ),
        "matched": {
            "RSI": _to_bool_label(buy_rsi_ok),
            "OBV": _to_bool_label(buy_obv_ok),
            "CCI": _to_bool_label(buy_cci_ok),
            "POSITION": _to_bool_label(buy_position_ok),
            "VIX_RISK": _to_bool_label(vix_risk),
        },
    }

    sell_result = {
        "recommendation": "매도" if sell_score >= SELL_THRESHOLD else "관망",
        "score": sell_score,
        "strength": int(sell_strength),
        "position_size": int(sell_position),
        "color": "#F44336" if sell_score >= SELL_THRESHOLD else "#9E9E9E",
        "reason": _build_reason_text(
            decision="매도",
            buy_rsi_ok=buy_rsi_ok,
            buy_obv_ok=buy_obv_ok,
            buy_cci_ok=buy_cci_ok,
            buy_position_ok=buy_position_ok,
            sell_rsi_ok=sell_rsi_ok,
            sell_obv_ok=sell_obv_ok,
            sell_cci_ok=sell_cci_ok,
            sell_position_ok=sell_position_ok,
            vix_risk=vix_risk,
            buy_score=buy_score,
            sell_score=sell_score,
        ),
        "matched": {
            "RSI": _to_bool_label(sell_rsi_ok),
            "OBV": _to_bool_label(sell_obv_ok),
            "CCI": _to_bool_label(sell_cci_ok),
            "POSITION": _to_bool_label(sell_position_ok),
            "VIX_RISK": _to_bool_label(vix_risk),
        },
    }

    return buy_result if decision == "매수" else sell_result


def generate_dual_signal(indicators: dict) -> dict:
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
