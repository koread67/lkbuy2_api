
import pandas as pd
import ta
import requests
from bs4 import BeautifulSoup
from io import StringIO

# (Optional) 국내 ETF 코드 목록 - 현재 로직에서는 사용하지 않음
KOREA_ETF_CODES = [
    "069500", "102110", "114800", "117700", "118260", "122630", "139260", "157450",
    "214980", "219480", "229200", "232080", "233740", "237370", "238720", "251340",
    "252670", "253150", "263750", "272220", "278530", "292150", "295820", "305720",
    "307510", "310970", "352560", "364980", "371160", "379800", "394280", "411060",
    "426410", "438420", "461340"
]

def get_naver_etf_price(code: str, pages: int = 5):
    """선택형: 네이버에서 국내 ETF 일별시세 수집"""
    try:
        if not code.isdigit():
            return None
        url = f"https://finance.naver.com/item/sise_day.naver?code={code}"
        headers = {"User-Agent": "Mozilla/5.0"}
        dfs = []
        for page in range(1, pages + 1):
            res = requests.get(f"{url}&page={page}", headers=headers, timeout=15)
            soup = BeautifulSoup(res.text, "html.parser")
            table = soup.select_one("table[type='N']")
            if table is None:
                continue
            df = pd.read_html(StringIO(str(table)))[0].dropna(how="all").dropna()
            dfs.append(df)
        if not dfs:
            return None
        df_all = pd.concat(dfs)
        df_all.columns = ['Date', 'Close', 'Change', 'Open', 'High', 'Low', 'Volume']
        df_all['Date'] = pd.to_datetime(df_all['Date'])
        df_all = df_all.sort_values('Date')
        df_all.set_index('Date', inplace=True)
        return df_all[['Open', 'High', 'Low', 'Close', 'Volume']]
    except Exception:
        return None

def calculate_indicators(data: pd.DataFrame) -> dict:
    """CCI(20), RSI(14), OBV 및 8일 추세를 계산"""
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

    obv_trend = float(obv.iloc[-1] - obv.iloc[-8]) if len(obv) >= 8 else 0.0
    return {
        "CCI": float(cci.iloc[-1]),
        "OBV": float(obv.iloc[-1]),
        "OBV_trend": obv_trend,
        "RSI": float(rsi.iloc[-1]),
    }

# ====== 최적조합(권장 로직) ======
# 연속형 스코어링 + 비대칭 임계값
W_CCI = 0.33
W_RSI = 0.33
W_OBV = 0.34

BUY_THRESHOLD = 50    # 진입은 민첩
SELL_THRESHOLD = 90   # 청산은 보수

def _scale_linear(x: float, x0: float, x1: float, y0: float = 0.0, y1: float = 100.0) -> float:
    """구간 [x0,x1] -> [y0,y1] 선형 매핑(클램프 포함)"""
    if x0 == x1:
        return (y0 + y1) / 2.0
    # 클램핑
    if x <= min(x0, x1):
        return y0 if x0 < x1 else y1
    if x >= max(x0, x1):
        return y1 if x0 < x1 else y0
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

def _component_scores_for_buy(cci: float, rsi: float, obv_trend: float):
    """매수용: 과매도/유입에 가산"""
    cci_s = _scale_linear(cci, -100, 100, 100, 0)     # 과매도일수록 100
    rsi_s = _scale_linear(rsi, 30, 70, 100, 0)        # 30 이하일수록 100
    obv_s = 100.0 if obv_trend > 0 else (0.0 if obv_trend < 0 else 50.0)
    return cci_s, rsi_s, obv_s

def _component_scores_for_sell(cci: float, rsi: float, obv_trend: float):
    """매도용: 과매수/이탈에 가산"""
    cci_s = _scale_linear(cci, -100, 100, 0, 100)     # 과매수일수록 100
    rsi_s = _scale_linear(rsi, 30, 70, 0, 100)        # 70 이상일수록 100
    obv_s = 100.0 if obv_trend < 0 else (0.0 if obv_trend > 0 else 50.0)
    return cci_s, rsi_s, obv_s

def generate_signal(indicators: dict, decision: str):
    """권장 로직 기반 신호 생성"""
    cci = float(indicators.get("CCI", 0.0))
    obv_trend = float(indicators.get("OBV_trend", 0.0))
    rsi = float(indicators.get("RSI", 0.0))

    if decision == "매수":
        cci_s, rsi_s, obv_s = _component_scores_for_buy(cci, rsi, obv_trend)
        score_f = W_CCI * cci_s + W_RSI * rsi_s + W_OBV * obv_s
        recommendation = "매수" if score_f >= BUY_THRESHOLD else "매수X"
        # 색/레벨 (진입용 팔레트)
        if score_f >= 90:
            level, color = "매우 강함", "#2E7D32"
        elif score_f >= 70:
            level, color = "강함", "#4CAF50"
        elif score_f >= 50:
            level, color = "보통", "#FFEB3B"
        else:
            level, color = "약함", "#F44336"
        reasons = [
            ("CCI 과매도" if cci <= -100 else "CCI 중립" if -100 < cci < 100 else "CCI 과매수"),
            ("OBV 상승(유입)" if obv_trend > 0 else "OBV 정체" if obv_trend == 0 else "OBV 하락(이탈)"),
            ("RSI 과매도(≤30)" if rsi <= 30 else "RSI 중립(30~70)" if 30 < rsi < 70 else "RSI 과매수(≥70)"),
        ]
    else:
        cci_s, rsi_s, obv_s = _component_scores_for_sell(cci, rsi, obv_trend)
        score_f = W_CCI * cci_s + W_RSI * rsi_s + W_OBV * obv_s
        recommendation = "매도" if score_f >= SELL_THRESHOLD else "매도X"
        # 색/레벨 (청산용 팔레트)
        if score_f >= 90:
            level, color = "매우 강함", "#B71C1C"
        elif score_f >= 70:
            level, color = "강함", "#E53935"
        elif score_f >= 50:
            level, color = "보통", "#FF9800"
        else:
            level, color = "약함", "#9E9E9E"
        reasons = [
            ("CCI 과매수" if cci >= 100 else "CCI 중립" if -100 < cci < 100 else "CCI 과매도"),
            ("OBV 하락(이탈)" if obv_trend < 0 else "OBV 정체" if obv_trend == 0 else "OBV 상승(유입)"),
            ("RSI 과매수(≥70)" if rsi >= 70 else "RSI 중립(30~70)" if 30 < rsi < 70 else "RSI 과매도(≤30)"),
        ]

    score_int = int(round(score_f))
    return {
        "recommendation": recommendation,
        "score": score_int,      # 0~100
        "strength": score_int,   # UI에서 강도로 재사용
        "color": color,
        "level": level,
        "reason": " / ".join(reasons),
    }
