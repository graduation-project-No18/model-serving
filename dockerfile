# 나의 python 버전
FROM python:3.9.6

# /code 폴더 만들기
WORKDIR /modelserving

# ./requirements.txt 를 /requirements.txt 로 복사
COPY requirements.txt ./requirements.txt

# requirements.txt 를 보고 모듈 전체 설치(-r)
RUN pip install -r requirements.txt
RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get install -y ffmpeg
RUN mkdir recordings

# 이제 app 에 있는 파일들을 /code/app 에 복사
COPY ./main.py /modelserving/
COPY ./.env /modelserving/
COPY ./DF.csv /modelserving/
COPY ./doremi_fre_boy.csv /modelserving/
COPY ./doremi_fre_girl.csv /modelserving/
COPY ./recommendSong.py /modelserving/
COPY ./Recording.py /modelserving/
COPY ./s3service.py /modelserving/
COPY ./Song.csv /modelserving/
COPY ./Song_details.xlsx /modelserving/


# 실행
CMD ["uvicorn", "main:app", "--host","0.0.0.0", "--port", "8000"]

#build
#docker build -t no18_modelserving:0.1 -f dockerfile . --platform linux/amd64