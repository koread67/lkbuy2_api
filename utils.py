
import pandas as pd
import ta
import requests
from bs4 import BeautifulSoup

KOREA_ETF_CODES = [
    "069500", "102110", "114800", "117700", "118260", "122630", "139260", "157450",
    "214980", "219480", "229200", "232080", "233740", "237370", "238720", "251340",
    "252670", "253150", "263750", "272220", "278530", "292150", "295820", "305720",
    "307510", "310970", "352560", "364980", "371160", "379800", "394280", "411060",
    "426410", "438420", "461340"
]

def get_naver_etf_price(code: str, pages: int = 5):
    try:
        if not code.isdigit():
            return None
        url = f"https://finance.naver.com/item/sise_day.naver?code={code}"
        headers = {"User-Agent": "Mozilla/5.0"}
        dfs = []
        for page in range(1, pages + 1):
            res = requests.get(f"{url}&page={page}", headers=headers)
            soup = BeautifulSoup(res.text, "html.parser")
            table = soup.select_one("table[type='N']")
            df = pd.read_html(StringIO(str(table)))[0].dropna()
            dfs.append(df)
        df_all = pd.concat(dfs)
        df_all.columns = ['Date', 'Close', 'Change', 'Open', 'High', 'Low', 'Volume']
        df_all['Date'] = pd.to_datetime(df_all['Date'])
        df_all = df_all.sort_values('Date')
        df_all.set_index('Date', inplace=True)
        return df_all[['Open', 'High', 'Low', 'Close', 'Volume']]
    except:
        return None

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
    if decision == "매수":
        cci_score = 40 if cci < -100 else 10
        obv_score = 30 if obv_trend > 0 else 10
        rsi_score = 30 if rsi < 30 else 10
    else:
        cci_score = 40 if cci > 100 else 10
        obv_score = 30 if obv_trend < 0 else 10
        rsi_score = 30 if rsi > 70 else 10
    if decision == "매수":
        reasons.append("CCI가 과매도 구간입니다." if cci < -100 else "CCI가 과매도 구간이 아닙니다.")
        reasons.append("OBV가 상승세로 자금 유입 신호입니다." if obv_trend > 0 else "OBV가 하락세로 자금 유출 신호입니다.")
        reasons.append("RSI가 30 이하로 과매도 구간입니다." if rsi < 30 else "RSI가 과매도 구간이 아닙니다.")
    else:
        reasons.append("CCI가 과매수 구간입니다." if cci > 100 else "CCI가 과매수 구간이 아닙니다.")
        reasons.append("OBV가 하락세로 자금 이탈 신호입니다." if obv_trend < 0 else "OBV가 상승세로 자금 유입 신호입니다.")
        reasons.append("RSI가 70 이상으로 과매수 구간입니다." if rsi > 70 else "RSI가 과매수 구간이 아닙니다.")
    
    score = round(cci_score * 0.4 + obv_score * 0.3 + rsi_score * 0.3)
    strength = round(obv_score * 0.6 + cci_score * 0.4)

    if decision == "매수":
        recommendation = "매수" if score >= 80 else "매수X"
    else:
        recommendation = "매도" if score >= 80 else "매도X"

    if strength >= 80:
        color = "#4CAF50"
        level = "매우 강함"
    elif strength >= 60:
        color = "#FFEB3B"
        level = "보통"
    elif strength >= 40:
        color = "#FF9800"
        level = "약함"
    else:
        color = "#F44336"
        level = "매수X" if decision == "매수" else "매도X"

    return {
        "recommendation": recommendation,
        "score": score,
        "strength": strength,
        "color": color,
        "level": level,
        "reason": " ".join(reasons)
    }

