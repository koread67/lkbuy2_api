import math
from typing import Any

import numpy as np
import pandas as pd

# =========================
# Trading Logic 2.4
# =========================
# - 00통합.xlsx 자동 탐색 결과 반영
# - 추격형 매수/매도 완화를 위해 '위치 필터'를 독립 점수로 반영
# - 매수: 최근 20일 고점 대비 -5% 이하에서 전환 신호가 나올 때 우대
# - 매도: 원자재 제외 ETF 기준으로 전체 청산이 아니라 매도 추천 신호로 전환
# - 기존 앱 연동을 위해 calculate_indicators / generate_signal / generate_dual_signal 구조 유지
# - 상승 시 매수권유가 누적되는 문제를 줄이기 위해 과열/고점 추격 패널티 반영
# - 신심리도 + VIX 동시 최적화 필터 반영
# - ATR은 매도 조건에서만 변동성 확장 확인 보조지표로 반영
# - 상승 말기 구간은 추가 매수가 아니라 매도 추천으로 전환

RSI_WEIGHT = 25
OBV_WEIGHT = 20
CCI_WEIGHT = 15
POSITION_WEIGHT = 40

BUY_THRESHOLD = 70
SELL_THRESHOLD = 90

SCORE_MAX = RSI_WEIGHT + OBV_WEIGHT + CCI_WEIGHT + POSITION_WEIGHT  # 100

BUY_HIGH_DISCOUNT = -5.0      # 최근 20일 고점 대비 -5% 이하
SELL_HIGH_NEAR = -0.5         # 최근 20일 고점 대비 -0.5% 이내: 고점권에서만 매도 검토
RSI_DELTA_BUY_MIN = 2.0       # RSI 3일 변화율
OBV_GAP_MIN = 0.5             # OBV와 OBV MA20 간 갭(%)

VIX_BUY_PENALTY = 10
VIX_SELL_BONUS = 0            # VIX만으로 매도 가점 부여하지 않음

# 상승추격 방지 보정값
CHASE_BUY_PENALTY = 25       # 고점권 상승 추격 매수 감점
OVERHEAT_BUY_PENALTY = 15    # RSI/CCI 과열 매수 감점
LOW_VIX_BUY_PENALTY = 10     # VIX 과도 안정 구간 매수 감점
EXHAUST_SELL_BONUS = 10      # 상승 피로 구간 일부 비중 축소 가점

# 신심리도 최적화 조건: 00통합 원자재 제외 7거래일 검증 기준
SENTIMENT_LOOKBACK = 20
SENTIMENT_BUY_MIN = -50.0
SENTIMENT_BUY_MAX = 100.0
SENTIMENT_DELTA_MIN = -5.0

# 신심리도 + VIX 동시 최적화 조건: 00통합 원자재 제외 7거래일 검증 기준
VIX_BUY_MAX = 35.0
VIX_BUY_RATIO_MAX = 1.0

# 원자재 제외 ETF용 매도 제한 조건
SELL_RSI_HIGH = 72.0          # 과열 후 둔화 확인
SELL_RSI_DELTA_MAX = -2.5
SELL_OBV_GAP_MIN = -1.5       # OBV가 평균 대비 충분히 약해야 함
SELL_CCI_HIGH = 180.0         # CCI 과열 후 꺾임
SELL_CCI_DELTA_MAX = -10.0
SELL_ADX_MIN = 20.0           # 추세 약화/하락 확인 최소 ADX
SELL_REDUCTION_MAX = 100      # 매도 추천 시 전체 매도 가능

# 상승 말기 매도 추천 조건
LATE_RISE_RSI_MIN = 65.0      # 상승 말기 RSI 하한
LATE_RISE_CCI_MIN = 100.0     # 상승 말기 CCI 하한
LATE_RISE_5D_CHANGE = 3.0     # 최근 5거래일 상승률 기준
LATE_RISE_BUY_PENALTY = 35    # 상승 말기 매수 차단 감점
LATE_RISE_SELL_BONUS = 60     # 상승 말기 매도 추천 강제 가점

# ATR은 매수에는 반영하지 않고, 매도 조건에서만 변동성 확장 확인용으로 사용
ATR_SELL_RATIO_MIN = 1.30     # ATR / ATR 20MA가 1.30 이상이면 변동성 확장
ATR_SELL_DELTA_MIN = 0.0      # ATR 3일 변화가 0 이상이면 변동성 증가


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
    atr_ma20 = atr.rolling(20).mean()
    atr_ratio = atr / atr_ma20.replace(0, np.nan)
    atr_delta3 = atr.diff(3)

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
    ma5_delta3 = ma5.diff(3)
    close_change5 = ((close / close.shift(5).replace(0, np.nan)) - 1) * 100

    # 신심리도: 가격 등락률을 거래량으로 가중한 심리 압력 지표
    # 범위는 -100~+100이며, 0보다 낮으면 침체/약심리, 0보다 높으면 과열/강심리로 해석한다.
    ret = close.pct_change()
    weighted_signed = (ret * volume).rolling(SENTIMENT_LOOKBACK).sum()
    weighted_abs = (ret.abs() * volume).rolling(SENTIMENT_LOOKBACK).sum()
    new_sentiment = (100 * weighted_signed / weighted_abs.replace(0, np.nan)).clip(-100, 100)
    new_sentiment_score = (new_sentiment + 100) / 2
    new_sentiment_delta3 = new_sentiment.diff(3)

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
        "ATR": _safe_last(atr),
        "ATR_MA20": _safe_last(atr_ma20),
        "ATR_RATIO": _safe_last(atr_ratio),
        "ATR_DELTA3": _safe_last(atr_delta3),
        "PRICE_MA20": _safe_last(price_ma20),
        "DIST20": _safe_last(dist20),
        "HIGH20": _safe_last(high20),
        "HIGH20_DIST": _safe_last(high20_dist),
        "LOW20": _safe_last(low20),
        "LOW20_DIST": _safe_last(low20_dist),
        "CLOSE_PREV1": _safe_last(close_prev1),
        "MA5": _safe_last(ma5),
        "MA5_DELTA3": _safe_last(ma5_delta3),
        "CLOSE_CHANGE5": _safe_last(close_change5),
        "NEW_SENTIMENT": _safe_last(new_sentiment),
        "NEW_SENTIMENT_SCORE": _safe_last(new_sentiment_score),
        "NEW_SENTIMENT_DELTA3": _safe_last(new_sentiment_delta3),
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
    chase_risk: bool,
    overheat_risk: bool,
    low_vix_risk: bool,
    exhaustion_risk: bool,
    late_rise_sell: bool,
    atr_sell_ok: bool,
    sentiment_buy_ok: bool,
    vix_buy_ok: bool,
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
        if chase_risk:
            reasons.append("고점권 추격매수 감점")
        if overheat_risk:
            reasons.append("과열구간 매수 감점")
        if low_vix_risk:
            reasons.append("VIX 과도안정 매수 감점")
        if not vix_buy_ok:
            reasons.append("VIX 매수 필터 미통과")
        if not sentiment_buy_ok:
            reasons.append("신심리도 매수필터 미충족")
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
        if exhaustion_risk:
            reasons.append("상승 피로 매도 가점")
        if late_rise_sell:
            reasons.append("상승 말기 매도 추천")
        if atr_sell_ok:
            reasons.append("ATR 변동성 확장 확인")

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
    adx = indicators.get("ADX", np.nan)
    adx_delta3 = indicators.get("ADX_DELTA3", np.nan)
    close = indicators.get("CLOSE", np.nan)
    close_change5 = indicators.get("CLOSE_CHANGE5", np.nan)
    ma5_delta3 = indicators.get("MA5_DELTA3", np.nan)
    atr_ratio = indicators.get("ATR_RATIO", np.nan)
    atr_delta3 = indicators.get("ATR_DELTA3", np.nan)
    new_sentiment = indicators.get("NEW_SENTIMENT", np.nan)
    new_sentiment_delta3 = indicators.get("NEW_SENTIMENT_DELTA3", np.nan)
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
    adx_f = _clean_float(adx)
    adx_delta3_f = _clean_float(adx_delta3)
    close_change5_f = _clean_float(close_change5)
    ma5_delta3_f = _clean_float(ma5_delta3)
    atr_ratio_f = _clean_float(atr_ratio, float("nan"))
    atr_delta3_f = _clean_float(atr_delta3, float("nan"))
    new_sentiment_f = _clean_float(new_sentiment, float("nan"))
    new_sentiment_delta_f = _clean_float(new_sentiment_delta3, float("nan"))

    sentiment_buy_ok = bool(
        pd.notna(new_sentiment)
        and pd.notna(new_sentiment_delta3)
        and SENTIMENT_BUY_MIN <= new_sentiment_f <= SENTIMENT_BUY_MAX
        and new_sentiment_delta_f >= SENTIMENT_DELTA_MIN
    )

    vix_available = bool(pd.notna(vix) and pd.notna(vix_5ma) and _clean_float(vix_5ma) > 0)
    vix_risk = bool(vix_available and _clean_float(vix) > _clean_float(vix_5ma))
    low_vix_risk = bool(vix_available and _clean_float(vix) < _clean_float(vix_5ma) * 0.92)
    vix_buy_ok = bool(
        (not vix_available)
        or (
            _clean_float(vix) <= VIX_BUY_MAX
            and (_clean_float(vix) / _clean_float(vix_5ma)) <= VIX_BUY_RATIO_MAX
        )
    )

    # 매수: 단순 상승 추격이 아니라 '저위치 + 전환'을 우선한다.
    buy_rsi_ok = bool(45 <= rsi_f <= 68 and rsi_delta_f >= RSI_DELTA_BUY_MIN)
    buy_obv_ok = bool(obv_trend_f > 0 and obv_gap_f >= OBV_GAP_MIN)
    buy_cci_ok = bool(-120 <= cci_f <= 120 and cci_delta_f > 0)
    buy_position_ok = bool(high20_dist_f <= BUY_HIGH_DISCOUNT)

    # 매도: 원자재 제외 ETF에서는 매도 후 7일 상승 오판이 컸으므로,
    # 단순 약화/하락 추격 매도는 제거하고 고점권 피로 + 수급/추세 약화를 동시에 요구한다.
    sell_position_ok = bool(high20_dist_f >= SELL_HIGH_NEAR)
    sell_rsi_ok = bool(rsi_f >= SELL_RSI_HIGH and rsi_delta_f <= SELL_RSI_DELTA_MAX)
    sell_obv_ok = bool(obv_trend_f < 0 and obv_gap_f <= SELL_OBV_GAP_MIN)
    sell_cci_ok = bool(cci_f >= SELL_CCI_HIGH and cci_delta_f <= SELL_CCI_DELTA_MAX)
    sell_trend_ok = bool(minus_di_f > plus_di_f and adx_f >= SELL_ADX_MIN and adx_delta3_f >= -2.0)

    # 상승추격 방지: 가격이 이미 20일 고점권에 있고 단기 상승률이 큰 경우,
    # 매수 신호가 누적되더라도 감점한다.
    chase_risk = bool(high20_dist_f >= SELL_HIGH_NEAR and close_change5_f >= 3.0 and rsi_f >= 58)
    overheat_risk = bool(rsi_f >= 70 or cci_f >= 180)
    exhaustion_risk = bool(
        sell_position_ok
        and (sell_rsi_ok or sell_cci_ok)
        and (sell_obv_ok or sell_trend_ok or ma5_delta3_f <= -1.0)
    )

    # ATR 변동성 확장 확인: 매수에는 쓰지 않고 매도 조건에서만 사용한다.
    atr_sell_ok = bool(
        pd.notna(atr_ratio)
        and pd.notna(atr_delta3)
        and atr_ratio_f >= ATR_SELL_RATIO_MIN
        and atr_delta3_f >= ATR_SELL_DELTA_MIN
    )

    # 상승 말기: 가격이 고점권이고 RSI/CCI/단기상승률이 동시에 높으며
    # ATR 변동성 확장이 확인될 때만 매도 추천으로 전환한다.
    late_rise_sell = bool(
        sell_position_ok
        and rsi_f >= LATE_RISE_RSI_MIN
        and (cci_f >= LATE_RISE_CCI_MIN or close_change5_f >= LATE_RISE_5D_CHANGE)
        and not sell_trend_ok
        and atr_sell_ok
    )

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

    # 상승 말기에는 부분매도가 아니라 명확한 매도 추천으로 전환한다.
    # 이 경우 점수가 반드시 SELL_THRESHOLD 이상이 되도록 강제한다.
    if late_rise_sell:
        sell_score = max(sell_score + LATE_RISE_SELL_BONUS, SELL_THRESHOLD)
        buy_score = max(0, buy_score - LATE_RISE_BUY_PENALTY)
    if not (exhaustion_risk or late_rise_sell):
        sell_score = min(sell_score, SELL_THRESHOLD - 10)

    # DMI는 매도 방향에서는 sell_trend_ok일 때만 보정한다.
    if pd.notna(plus_di) and pd.notna(minus_di):
        if plus_di_f > minus_di_f and buy_score >= BUY_THRESHOLD - 5:
            buy_score += 5
        if sell_trend_ok and sell_score >= SELL_THRESHOLD - 5:
            sell_score += 5

    if vix_risk:
        buy_score = max(0, buy_score - VIX_BUY_PENALTY)
        sell_score += VIX_SELL_BONUS

    if chase_risk:
        buy_score = max(0, buy_score - CHASE_BUY_PENALTY)
    if overheat_risk:
        buy_score = max(0, buy_score - OVERHEAT_BUY_PENALTY)
    if low_vix_risk:
        buy_score = max(0, buy_score - LOW_VIX_BUY_PENALTY)
    if exhaustion_risk:
        sell_score += EXHAUST_SELL_BONUS

    # 신심리도 최적화: 매수 신호가 임계값 이상이어도
    # 신심리도 필터를 통과하지 못하면 매수 신호를 관망으로 낮춘다.
    if buy_score >= BUY_THRESHOLD and not sentiment_buy_ok:
        buy_score = min(buy_score, BUY_THRESHOLD - 1)
    if buy_score >= BUY_THRESHOLD and not vix_buy_ok:
        buy_score = min(buy_score, BUY_THRESHOLD - 1)

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
            chase_risk=chase_risk,
            overheat_risk=overheat_risk,
            low_vix_risk=low_vix_risk,
            exhaustion_risk=exhaustion_risk,
            late_rise_sell=late_rise_sell,
            atr_sell_ok=atr_sell_ok,
            sentiment_buy_ok=sentiment_buy_ok,
            vix_buy_ok=vix_buy_ok,
            buy_score=buy_score,
            sell_score=sell_score,
        ),
        "matched": {
            "RSI": _to_bool_label(buy_rsi_ok),
            "OBV": _to_bool_label(buy_obv_ok),
            "CCI": _to_bool_label(buy_cci_ok),
            "POSITION": _to_bool_label(buy_position_ok),
            "VIX_RISK": _to_bool_label(vix_risk),
            "CHASE_RISK": _to_bool_label(chase_risk),
            "OVERHEAT_RISK": _to_bool_label(overheat_risk),
            "LOW_VIX_RISK": _to_bool_label(low_vix_risk),
            "LATE_RISE_SELL": _to_bool_label(late_rise_sell),
            "ATR_SELL_OK": _to_bool_label(atr_sell_ok),
            "NEW_SENTIMENT": _to_bool_label(sentiment_buy_ok),
            "VIX_BUY_FILTER": _to_bool_label(vix_buy_ok),
        },
    }

    sell_result = {
        "recommendation": "매도" if sell_score >= SELL_THRESHOLD else "관망",
        "score": sell_score,
        "strength": int(sell_strength),
        "position_size": int(100 if (sell_score >= SELL_THRESHOLD and late_rise_sell) else min(sell_position, SELL_REDUCTION_MAX)),
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
            chase_risk=chase_risk,
            overheat_risk=overheat_risk,
            low_vix_risk=low_vix_risk,
            exhaustion_risk=exhaustion_risk,
            late_rise_sell=late_rise_sell,
            atr_sell_ok=atr_sell_ok,
            sentiment_buy_ok=sentiment_buy_ok,
            vix_buy_ok=vix_buy_ok,
            buy_score=buy_score,
            sell_score=sell_score,
        ),
        "matched": {
            "RSI": _to_bool_label(sell_rsi_ok),
            "OBV": _to_bool_label(sell_obv_ok),
            "CCI": _to_bool_label(sell_cci_ok),
            "POSITION": _to_bool_label(sell_position_ok),
            "VIX_RISK": _to_bool_label(vix_risk),
            "EXHAUSTION_RISK": _to_bool_label(exhaustion_risk),
            "LATE_RISE_SELL": _to_bool_label(late_rise_sell),
            "ATR_SELL_OK": _to_bool_label(atr_sell_ok),
            "SELL_TREND": _to_bool_label(sell_trend_ok),
            "NEW_SENTIMENT": _to_bool_label(sentiment_buy_ok),
            "VIX_BUY_FILTER": _to_bool_label(vix_buy_ok),
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
