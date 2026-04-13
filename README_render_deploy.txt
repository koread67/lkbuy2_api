Render 배포용 파일

포함 파일
- main_improved.py
- trade_decider_v2.py
- requirements.txt
- render.yaml

배포 개요
1. Render에 새 Web Service 생성
2. 이 파일들을 GitHub 저장소에 업로드
3. Render가 자동 배포
4. 기존 앱의 BASE 주소를 그대로 사용하려면
   Render 서비스 URL 또는 커스텀 도메인을 기존 주소와 맞춰야 함

주의
- 기존 onrender 서비스에 같은 URL로 덮어쓰려면 기존 Render 서비스의 소스 저장소를 이 파일들로 교체해야 함
- 새 서비스로 만들면 URL이 달라질 수 있음
