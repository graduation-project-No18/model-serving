## 녹음본 옥타브 분석 모델 서빙 (개발 중)

- ### framework and libraries:
  1. FastAPI
  2. S3(boto3)
  3. ffmpeg


- ### 역할 : 

    1. ResponseBody에 담긴 정보를 바탕으로 s3에서 회원의 녹음본 바이너리 파일을 가져오기 
    2. 가져온 녹음본 바이너리 파일을 오디오 포맷(.wav)로 변환하기
    3. 변환된 .wav 파일을 모델에 전달
    4. 모델로부터 반환된 옥타브 및 노래 추천 분류 결과 Backend에 전달

- ### 참고 사항:
    - AWS_ACCESS_KEY와 AWS_SECRET_ACCESS_KEY는 .env에 존재
    - ffmpeg 설치 필요
    - 