import pandas as pd
import numpy as np

# =========================
# Trading Logic 2.0
# =========================
RSI_WEIGHT = 0.01
OBV_WEIGHT = 0.17
DMI_WEIGHT = 0.82

BUY_THRESHOLD = 84
SELL_THRESHOLD = 110

VIX_BUY_PENALTY = 15
VIX_SELL_BONUS = 10


def _safe_last(series: pd.Series) -> float:
    """마지막 값을 float로 반환. 값이 없으면 NaN 반환."""
    if series is None or len(series) == 0:
        return float("nan")
    value = series.iloc[-1]
    if pd.isna(value):
        return float("nan")
    return float(value)


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
    """
    decision: "매수" 또는 "매도"
    """
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
            "recommendation": decision + "X",
            "score": np.nan,
            "strength": np.nan,
            "color": "#9E9E9E",
            "reason": "지표 계산 데이터 부족",
        }

    if decision == "매수":
        rsi_ok = rsi >= 50
        obv_ok = (obv_trend > 0) and (obv >= obv_ma20)
        dmi_ok = plus_di > minus_di

        score = round(
            rsi_ok * RSI_WEIGHT * 100 +
            obv_ok * OBV_WEIGHT * 100 +
            dmi_ok * DMI_WEIGHT * 100
        )

        vix_filter_applied = False
        if pd.notna(vix) and pd.notna(vix_5ma) and vix > vix_5ma:
            score -= VIX_BUY_PENALTY
            vix_filter_applied = True

        threshold = BUY_THRESHOLD
        matched = sum([rsi_ok, obv_ok, dmi_ok])

        recommendation = "매수" if score >= threshold else "매수X"
        color = "#2196F3" if recommendation == "매수" else "#9E9E9E"

        return {
            "recommendation": recommendation,
            "score": int(score),
            "strength": round((matched / 3) * 100),
            "color": color,
            "reason": (
                f"RSI:{rsi_ok}, OBV:{obv_ok}, DMI:{dmi_ok}, "
                f"VIX필터적용:{vix_filter_applied}"
            ),
        }

    rsi_ok = rsi <= 50
    obv_ok = (obv_trend < 0) and (obv <= obv_ma20)
    dmi_ok = minus_di > plus_di

    score = round(
        rsi_ok * RSI_WEIGHT * 100 +
        obv_ok * OBV_WEIGHT * 100 +
        dmi_ok * DMI_WEIGHT * 100
    )

    vix_filter_applied = False
    if pd.notna(vix) and pd.notna(vix_5ma) and vix > vix_5ma:
        score += VIX_SELL_BONUS
        vix_filter_applied = True

    threshold = SELL_THRESHOLD
    matched = sum([rsi_ok, obv_ok, dmi_ok])

    recommendation = "매도" if score >= threshold else "매도X"
    color = "#F44336" if recommendation == "매도" else "#9E9E9E"

    return {
        "recommendation": recommendation,
        "score": int(score),
        "strength": round((matched / 3) * 100),
        "color": color,
        "reason": (
            f"RSI:{rsi_ok}, OBV:{obv_ok}, DMI:{dmi_ok}, "
            f"VIX필터적용:{vix_filter_applied}"
        ),
    }


def generate_dual_signal(indicators: dict) -> dict:
    """
    매수 / 매도를 동시에 평가해서 더 강한 쪽을 반환.
    둘 다 기준 미달이면 관망.
    """
    buy_signal = generate_signal(indicators, "매수")
    sell_signal = generate_signal(indicators, "매도")

    buy_ok = buy_signal["recommendation"] == "매수"
    sell_ok = sell_signal["recommendation"] == "매도"

    if buy_ok and not sell_ok:
        return {"final_decision": "매수", "buy_signal": buy_signal, "sell_signal": sell_signal}

    if sell_ok and not buy_ok:
        return {"final_decision": "매도", "buy_signal": buy_signal, "sell_signal": sell_signal}

    if buy_ok and sell_ok:
        if buy_signal["score"] > sell_signal["score"]:
            final = "매수"
        elif sell_signal["score"] > buy_signal["score"]:
            final = "매도"
        else:
            if buy_signal["strength"] > sell_signal["strength"]:
                final = "매수"
            elif sell_signal["strength"] > buy_signal["strength"]:
                final = "매도"
            else:
                final = "관망"
        return {"final_decision": final, "buy_signal": buy_signal, "sell_signal": sell_signal}

    return {"final_decision": "관망", "buy_signal": buy_signal, "sell_signal": sell_signal}
