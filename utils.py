import pandas as pd
import ta

def calculate_indicators(data: pd.DataFrame) -> dict:
    close = pd.Series(data["Close"].to_numpy().ravel())
    volume = pd.Series(data["Volume"].to_numpy().ravel())
    high = pd.Series(data["High"].to_numpy().ravel())
    low = pd.Series(data["Low"].to_numpy().ravel())

    cci_indicator = ta.trend.CCIIndicator(high=high, low=low, close=close, window=20)
    cci = cci_indicator.cci()

    obv_indicator = ta.volume.OnBalanceVolumeIndicator(close=close, volume=volume)
    obv = obv_indicator.on_balance_volume()

    rsi_indicator = ta.momentum.RSIIndicator(close=close, window=14)
    rsi = rsi_indicator.rsi()

    obv_trend = obv.iloc[-1] - obv.iloc[-8] if len(obv) >= 8 else 0

    return {
        "CCI": cci.iloc[-1],
        "OBV": obv.iloc[-1],
        "OBV_trend": obv_trend,
        "RSI": rsi.iloc[-1],
    }

def generate_signal(indicators: dict, decision: str):
    cci = indicators.get("CCI", 0)
    obv_trend = indicators.get("OBV_trend", 0)
    rsi = indicators.get("RSI", 0)

    reasons = []

    cci_score = 40 if cci < -100 else 10
    if cci < -100:
        reasons.append("CCI가 과매도 구간입니다.")
    else:
        reasons.append("CCI가 과매도 구간이 아닙니다.")

    obv_score = 30 if obv_trend > 0 else 10
    if obv_trend > 0:
        reasons.append("OBV가 상승세로 자금 유입 신호입니다.")
    else:
        reasons.append("OBV가 하락세로 자금 유출 신호입니다.")

    rsi_score = 30 if rsi < 30 else 10
    if rsi < 30:
        reasons.append("RSI가 30 이하로 과매도 구간입니다.")
    else:
        reasons.append("RSI가 과매도 구간이 아닙니다.")

    score = round(cci_score * 0.45 + obv_score * 0.35 + rsi_score * 0.20)

    recommendation = "매수" if score >= 80 else "매수X"

    # 색상 분류 (Flutter UI 연동용)
    if score >= 80:
        color = "#4CAF50"  # green
        level = "매우 강함"
    elif score >= 60:
        color = "#FFEB3B"  # yellow
        level = "보통"
    elif score >= 40:
        color = "#FF9800"  # orange
        level = "약함"
    else:
        color = "#F44336"  # red
        level = "매수X"

    return {
        "recommendation": recommendation,
        "score": score,
        "color": color,
        "level": level,
        "reason": " ".join(reasons)
    }