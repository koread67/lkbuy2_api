# trade_decider_v2.py
# Improved decision engine with strong-signal filter
# - Automatic decision: compares buy_score and sell_score
# - Weak signals below action threshold are forced to 관망

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import pandas as pd
import ta


@dataclass
class OptimalCombo:
    w_cci: float = 0.33
    w_rsi: float = 0.33
    w_obv: float = 0.34
    buy_min: float = 30.0
    sell_min: float = 90.0
    max_threshold: float = 100.0
    min_action_strength: int = 20   # strong signal filter


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


def _clip01(x: float) -> float:
    if x < 0:
        return 0.0
    if x > 1:
        return 1.0
    return x


def _bullish_components(cci: float, rsi: float, obv_trend: float) -> Dict[str, float]:
    cci_bull = _clip01((0.0 - cci) / 200.0) * 100.0
    rsi_bull = _clip01((50.0 - rsi) / 20.0) * 100.0
    obv_bull = 100.0 if obv_trend > 0 else 0.0
    return {"CCI": cci_bull, "RSI": rsi_bull, "OBV": obv_bull}


def _bearish_components(cci: float, rsi: float, obv_trend: float) -> Dict[str, float]:
    cci_bear = _clip01(cci / 200.0) * 100.0
    rsi_bear = _clip01((rsi - 50.0) / 20.0) * 100.0
    obv_bear = 100.0 if obv_trend < 0 else 0.0
    return {"CCI": cci_bear, "RSI": rsi_bear, "OBV": obv_bear}


def _weighted_sum(components: Dict[str, float], combo: OptimalCombo) -> float:
    return (
        components["CCI"] * combo.w_cci
        + components["RSI"] * combo.w_rsi
        + components["OBV"] * combo.w_obv
    )


def _percent_from_threshold(score: float, min_thr: float, max_thr: float) -> int:
    if score < min_thr:
        return 0
    if score >= max_thr:
        return 100
    return int(1 + (score - min_thr) * 99.0 / (max_thr - min_thr))


def _level_from_percent(p: int) -> str:
    if p == 0:
        return "미달"
    if p >= 71:
        return "매우강함"
    if p >= 41:
        return "강함"
    return "적정"


def _color_from_percent(p: int) -> str:
    if p == 0:
        return "#F44336"
    if p >= 71:
        return "#2E7D32"
    if p >= 41:
        return "#FBC02D"
    return "#FB8C00"


def auto_generate_signal(indicators: Dict[str, float]) -> Dict[str, object]:
    combo = OptimalCombo()

    cci = float(indicators.get("CCI", 0.0))
    obv_trend = float(indicators.get("OBV_trend", 0.0))
    rsi = float(indicators.get("RSI", 0.0))

    bull = _bullish_components(cci, rsi, obv_trend)
    bear = _bearish_components(cci, rsi, obv_trend)

    buy_score = _weighted_sum(bull, combo)
    sell_score = _weighted_sum(bear, combo)

    buy_strength = _percent_from_threshold(buy_score, combo.buy_min, combo.max_threshold)
    sell_strength = _percent_from_threshold(sell_score, combo.sell_min, combo.max_threshold)

    if buy_strength == 0 and sell_strength == 0:
        recommendation = "관망"
        chosen_score = max(buy_score, sell_score)
        strength_pct = 0
        side = "중립"
        chosen_components = bull if buy_score >= sell_score else bear
    elif buy_strength >= sell_strength:
        recommendation = "매수"
        chosen_score = buy_score
        strength_pct = buy_strength
        side = "매수"
        chosen_components = bull
    else:
        recommendation = "매도"
        chosen_score = sell_score
        strength_pct = sell_strength
        side = "매도"
        chosen_components = bear

    # strong signal filter
    if recommendation in ("매수", "매도") and strength_pct < combo.min_action_strength:
        recommendation = "관망"
        side = "중립"

    return {
        "recommendation": recommendation,
        "decision_side": side,
        "score": int(round(chosen_score)),
        "buy_score": round(buy_score, 2),
        "sell_score": round(sell_score, 2),
        "strength_pct": int(strength_pct),
        "strength": f"{int(strength_pct)}%",
        "color": _color_from_percent(strength_pct),
        "level": _level_from_percent(strength_pct),
        "reason": f"CCI={cci:.2f}, RSI={rsi:.2f}, OBV_trend={obv_trend:.2f}",
        "components": {k: int(round(v)) for k, v in chosen_components.items()},
        "thresholds": {
            "buy_min": combo.buy_min,
            "sell_min": combo.sell_min,
            "max": combo.max_threshold,
            "min_action_strength": combo.min_action_strength,
        },
        "weights": {"CCI": combo.w_cci, "RSI": combo.w_rsi, "OBV": combo.w_obv},
    }
