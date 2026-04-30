# -*- coding: utf-8 -*-
"""
매매판별 2604 - 상승추격 방지형 판별 로직

목적
- 상승할수록 매수점수가 누적되어 고점 추격매수가 발생하는 문제를 줄입니다.
- VIX 안정, OBV 개선, DMI 상승추세는 매수에 반영하되,
  RSI 과열, VIX 과도 안정, ADX 과열 추세, OBV 급등은 매수 감점으로 처리합니다.
- 매수/매도 점수는 0~100 범위로 산출합니다.

주요 함수
- judge_trade_2604(row): 단일 행 데이터 판별
- calculate_scores(...): 지표값 직접 입력 판별
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class TradeDecision:
    buy_score: int
    sell_score: int
    decision: str
    strength: str
    reason: str


def _to_float(value: Any, default: float = 0.0) -> float:
    """숫자 변환 실패 시 기본값을 반환합니다."""
    try:
        if value is None:
            return default
        if isinstance(value, str) and value.strip() == "":
            return default
        return float(value)
    except Exception:
        return default


def _clamp(value: float, low: int = 0, high: int = 100) -> int:
    """점수를 0~100 범위로 제한합니다."""
    return int(max(low, min(high, round(value))))


def calculate_scores(
    rsi: Any = 50,
    obv: Any = 0,
    obv_ma20: Any = 0,
    obv_trend: Any = 0,
    plus_di: Any = 0,
    minus_di: Any = 0,
    adx: Any = 0,
    vix: Any = 20,
    vix_5ma: Any = 20,
) -> TradeDecision:
    """
    상승추격 방지형 매매 점수를 계산합니다.

    입력값
    - rsi: RSI 값
    - obv: OBV 현재값
    - obv_ma20: OBV 20일 평균
    - obv_trend: OBV 추세값. 양수면 유입, 음수면 이탈
    - plus_di: +DI
    - minus_di: -DI
    - adx: ADX
    - vix: VIX 현재값
    - vix_5ma: VIX 5일 평균

    판별 원칙
    - 상승 확인 지표: OBV, DMI, VIX 안정
    - 추격매수 차단 지표: RSI 과열, ADX 과열, VIX 과도 안정, OBV 급등
    - 매도는 하락 우위, 자금 이탈, VIX 악화, RSI 과열 후 둔화를 중점 반영
    """

    rsi = _to_float(rsi, 50)
    obv = _to_float(obv, 0)
    obv_ma20 = _to_float(obv_ma20, 0)
    obv_trend = _to_float(obv_trend, 0)
    plus_di = _to_float(plus_di, 0)
    minus_di = _to_float(minus_di, 0)
    adx = _to_float(adx, 0)
    vix = _to_float(vix, 20)
    vix_5ma = _to_float(vix_5ma, 20)

    buy = 0.0
    sell = 0.0
    reasons = []

    # 1) RSI: 중립권은 약한 가점, 과열권은 매수 감점 및 매도 가점
    if rsi < 30:
        buy += 18
        reasons.append("RSI 과매도권: 반등 가능성 반영")
    elif 30 <= rsi < 45:
        buy += 8
        reasons.append("RSI 저중립권: 약한 매수 가점")
    elif 45 <= rsi <= 58:
        buy += 5
        reasons.append("RSI 중립권: 방향성 제한")
    elif 58 < rsi <= 65:
        buy += 3
        sell += 4
        reasons.append("RSI 상승권: 추격매수 주의")
    elif 65 < rsi <= 72:
        buy -= 10
        sell += 12
        reasons.append("RSI 과열 접근: 매수 감점")
    else:
        buy -= 20
        sell += 22
        reasons.append("RSI 과열권: 고점 추격 위험")

    # 2) OBV: 단순 상승은 매수 가점, 급등은 추격매수 감점
    obv_gap = obv - obv_ma20
    obv_gap_ratio = 0.0
    if abs(obv_ma20) > 1:
        obv_gap_ratio = obv_gap / abs(obv_ma20)

    if obv > obv_ma20 and obv_trend > 0:
        buy += 22
        reasons.append("OBV가 평균 위이고 추세 양수: 자금 유입")
    elif obv > obv_ma20:
        buy += 12
        reasons.append("OBV가 평균 위: 제한적 자금 유입")
    elif obv < obv_ma20 and obv_trend < 0:
        sell += 22
        reasons.append("OBV가 평균 아래이고 추세 음수: 자금 이탈")
    else:
        sell += 8
        reasons.append("OBV가 평균 아래: 매수 신뢰도 약화")

    if obv_gap_ratio > 0.35 and rsi > 60:
        buy -= 12
        sell += 8
        reasons.append("OBV 급등과 RSI 상승 동반: 추격매수 감점")

    # 3) DMI/ADX: 방향성 반영. ADX가 너무 높고 RSI 과열이면 매수 감점
    di_gap = plus_di - minus_di
    if di_gap > 8:
        buy += 20
        reasons.append("+DI가 -DI보다 우위: 상승 방향성")
    elif 0 < di_gap <= 8:
        buy += 10
        reasons.append("+DI 소폭 우위: 약한 상승 방향성")
    elif -8 <= di_gap <= 0:
        sell += 10
        reasons.append("-DI 소폭 우위: 약한 하락 방향성")
    else:
        sell += 20
        reasons.append("-DI가 +DI보다 우위: 하락 방향성")

    if adx < 18:
        buy -= 5
        sell -= 5
        reasons.append("ADX 낮음: 추세 신뢰도 낮음")
    elif 18 <= adx <= 30:
        buy += 6 if di_gap > 0 else 0
        sell += 6 if di_gap < 0 else 0
        reasons.append("ADX 정상 추세권: 방향성 신뢰도 보강")
    elif 30 < adx <= 42:
        buy += 8 if di_gap > 0 and rsi <= 65 else -6
        sell += 8 if di_gap < 0 or rsi > 65 else 0
        reasons.append("ADX 강한 추세권: 과열 여부에 따라 조정")
    else:
        buy -= 12 if rsi > 60 else 0
        sell += 12 if rsi > 60 else 0
        reasons.append("ADX 과도 상승: 막판 추세 가능성 반영")

    # 4) VIX: 안정은 매수 가점이지만, 과도한 안정은 과열 경고로 전환
    vix_gap = vix - vix_5ma
    if vix > vix_5ma + 2:
        sell += 18
        reasons.append("VIX가 5일 평균보다 높음: 위험 확대")
    elif vix_5ma - 2 <= vix <= vix_5ma + 2:
        buy += 8
        reasons.append("VIX 안정권: 시장 위험 중립")
    else:
        buy += 10
        reasons.append("VIX가 5일 평균보다 낮음: 위험 완화")

    if vix < 14 and rsi > 60:
        buy -= 15
        sell += 10
        reasons.append("VIX 과도 안정과 RSI 상승 동반: 안도 과열 경고")
    elif vix < 16 and rsi > 65:
        buy -= 10
        sell += 8
        reasons.append("낮은 VIX와 RSI 과열 접근: 매수 제한")

    # 5) 상승추격 방지 핵심 조건
    chase_risk = 0
    if rsi > 65:
        chase_risk += 1
    if adx > 30 and di_gap > 0:
        chase_risk += 1
    if vix < vix_5ma and vix < 17:
        chase_risk += 1
    if obv_gap_ratio > 0.25:
        chase_risk += 1

    if chase_risk >= 3:
        buy -= 20
        sell += 12
        reasons.append("상승추격 위험 3개 이상 충족: 매수 강도 강제 감점")
    elif chase_risk == 2:
        buy -= 10
        reasons.append("상승추격 위험 2개 충족: 매수 강도 제한")

    buy_score = _clamp(buy)
    sell_score = _clamp(sell)

    # 최종 판정
    if buy_score >= 70 and buy_score - sell_score >= 20:
        decision = "강매수"
        strength = "3/3"
    elif buy_score >= 55 and buy_score > sell_score:
        decision = "매수"
        strength = "2/3"
    elif sell_score >= 70 and sell_score - buy_score >= 20:
        decision = "강매도"
        strength = "3/3"
    elif sell_score >= 50 and sell_score >= buy_score:
        decision = "매도"
        strength = "2/3"
    else:
        decision = "관망"
        strength = "1/3"

    # 매수와 매도가 동시에 높으면 관망 우선
    if buy_score >= 50 and sell_score >= 45 and abs(buy_score - sell_score) < 15:
        decision = "관망"
        strength = "1/3"
        reasons.append("매수·매도 점수 동시 상승: 방향성 충돌로 관망")

    return TradeDecision(
        buy_score=buy_score,
        sell_score=sell_score,
        decision=decision,
        strength=strength,
        reason=" / ".join(reasons),
    )


def judge_trade_2604(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    엑셀 또는 데이터프레임 한 행을 받아 판별 결과를 반환합니다.

    허용 컬럼명 예시
    - RSI
    - OBV
    - OBV_MA20 또는 OBV MA20
    - OBV 추세 또는 OBV_TREND
    - +DI 또는 PLUS_DI
    - -DI 또는 MINUS_DI
    - ADX
    - VIX
    - VIX 5MA 또는 VIX_5MA
    """

    def pick(*names: str, default: Any = 0) -> Any:
        for name in names:
            if name in row:
                return row.get(name)
        return default

    result = calculate_scores(
        rsi=pick("RSI", default=50),
        obv=pick("OBV", default=0),
        obv_ma20=pick("OBV_MA20", "OBV MA20", "OBV_MA", default=0),
        obv_trend=pick("OBV 추세", "OBV_TREND", "OBV Trend", default=0),
        plus_di=pick("+DI", "PLUS_DI", "PDI", default=0),
        minus_di=pick("-DI", "MINUS_DI", "MDI", default=0),
        adx=pick("ADX", default=0),
        vix=pick("VIX", default=20),
        vix_5ma=pick("VIX 5MA", "VIX_5MA", "VIX_MA5", default=20),
    )

    return {
        "판별값": result.decision,
        "강도": result.strength,
        "매수점수": result.buy_score,
        "매도점수": result.sell_score,
        "판단사유": result.reason,
    }


# 이미지 예시값 테스트
if __name__ == "__main__":
    sample = {
        "RSI": 49.3,
        "OBV": -247569,
        "OBV_MA20": -203123,
        "OBV 추세": -7736,
        "+DI": 27.9,
        "-DI": 30.5,
        "ADX": 16.9,
        "VIX": 17.8,
        "VIX 5MA": 18,
    }
    print(judge_trade_2604(sample))
