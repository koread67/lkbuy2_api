import pandas as pd
import ta

def calculate_indicators(data: pd.DataFrame) -> dict:
    """
    야후 파이낸스 데이터를 이용해 ADX, CCI, OBV 지표와 OBV 7일 추세를 계산합니다.
    """

    # ✅ 모든 컬럼을 1차원 Series로 강제 변환
    for col in ["Close", "Volume", "High", "Low"]:
        if isinstance(data[col], pd.DataFrame):
            data[col] = data[col].iloc[:, 0]
        elif hasattr(data[col], "values") and data[col].values.ndim == 2:
            data[col] = pd.Series(data[col].values.squeeze())

    # ADX 계산 (14일)
    adx_indicator = ta.trend.ADXIndicator(
        high=data["High"],
        low=data["Low"],
        close=data["Close"],
        window=14
    )
    data["ADX"] = adx_indicator.adx()

    # CCI 계산 (20일)
    cci_indicator = ta.trend.CCIIndicator(
        high=data["High"],
        low=data["Low"],
        close=data["Close"],
        window=20
    )
    data["CCI"] = cci_indicator.cci()

    # OBV 계산
    obv_indicator = ta.volume.OnBalanceVolumeIndicator(
        close=data["Close"],
        volume=data["Volume"]
    )
    data["OBV"] = obv_indicator.on_balance_volume()

    # OBV 추세 계산 (최근 값 - 7일 전)
    if len(data) >= 8:
        obv_trend = data["OBV"].iloc[-1] - data["OBV"].iloc[-8]
    else:
        obv_trend = 0

    latest = data.iloc[-1]
    return {
        "ADX": latest["ADX"],
        "CCI": latest["CCI"],
        "OBV": latest["OBV"],
        "OBV_trend": obv_trend
    }

def generate_signal(indicators: dict, decision: str):
    """
    기술적 지표를 바탕으로 매수 또는 매도 신호와 100점 만점의 신뢰도(찬단 강도)를 산출합니다.
    """
    score = 0
    adx = indicators.get("ADX", 0)
    if adx >= 25:
        score += 30
    else:
        score += 10

    cci = indicators.get("CCI", 0)
    if decision == "매수":
        if cci > 100:
            score += 40
        else:
            score += 10
    elif decision == "매도":
        if cci < -100:
            score += 40
        else:
            score += 10

    obv_trend = indicators.get("OBV_trend", 0)
    if decision == "매수":
        if obv_trend > 0:
            score += 30
        else:
            score += 10
    elif decision == "매도":
        if obv_trend < 0:
            score += 30
        else:
            score += 10

    recommendation = decision if score >= 80 else ("매도" if decision == "매수" else "매수")
    return recommendation, score
