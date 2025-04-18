# LKBUY2 API

이 API는 야후 파이낸스 데이터를 기반으로 ADX, CCI, OBV 지표를 계산하여 매수/매도 판단과 100점 만점의 신뢰도 점수를 제공합니다.

## 기능

- **데이터 수집:** yfinance를 통해 최근 3개월치 일봉 데이터를 다운로드
- **지표 계산:** ADX, CCI, OBV 및 OBV 7일 추세 계산
- **투자 판단:** 단순 기준을 기반으로 요청된 판단(매수/매도)과 신뢰도 점수 산출
- **API 제공:** FastAPI를 이용한 REST API 엔드포인트 제공

## 설치 및 실행

1. Python 가상환경을 생성 및 활성화합니다.
2. 필요한 패키지를 설치합니다:
