import pandas as pd
import numpy as np

CCI_WEIGHT = 0.33
RSI_WEIGHT = 0.33
OBV_WEIGHT = 0.34

def calculate_indicators(data: pd.DataFrame):
    close = data["Close"]
    high = data["High"]
    low = data["Low"]
    volume = data["Volume"]

    tp = (high + low + close) / 3
    sma = tp.rolling(20).mean()
    mad = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())))
    cci = (tp - sma) / (0.015 * mad)

    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    obv_ma20 = obv.rolling(20).mean()
    obv_trend = obv.diff(7)

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(14).mean() / loss.rolling(14).mean()
    rsi = 100 - (100 / (1 + rs))

    return {
        "CCI": float(cci.iloc[-1]),
        "RSI": float(rsi.iloc[-1]),
        "OBV": float(obv.iloc[-1]),
        "OBV_MA20": float(obv_ma20.iloc[-1]),
        "OBV_trend": float(obv_trend.iloc[-1]),
    }

def generate_signal(indicators, decision):
    cci = indicators["CCI"]
    rsi = indicators["RSI"]
    obv = indicators["OBV"]
    obv_ma20 = indicators["OBV_MA20"]
    obv_trend = indicators["OBV_trend"]

    if decision == "매수":
        cci_ok = cci >= 100
        rsi_ok = rsi >= 50
        obv_ok = obv_trend > 0 and obv >= obv_ma20
    else:
        cci_ok = cci <= -100
        rsi_ok = rsi <= 50
        obv_ok = obv_trend < 0 and obv <= obv_ma20

    score = round(
        cci_ok * CCI_WEIGHT * 100 +
        rsi_ok * RSI_WEIGHT * 100 +
        obv_ok * OBV_WEIGHT * 100
    )

    matched = sum([cci_ok, rsi_ok, obv_ok])
    strength = round((matched / 3) * 100)

    recommendation = decision if score >= (50 if decision == "매수" else 90) else decision + "X"

    return {
        "recommendation": recommendation,
        "score": score,
        "strength": strength,
        "color": "#2196F3",
        "reason": f"CCI:{cci_ok}, RSI:{rsi_ok}, OBV:{obv_ok}"
    }
