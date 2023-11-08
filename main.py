import os

from fastapi import FastAPI
from Recording import Recording
import s3service
from pydub import AudioSegment
from recommendSong import *
import sys
import json

sys.path.append('/path/to/ffmpeg-6.0')

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


@app.post("/api/classification")
async def classify_octave_and_recommendation_songs(recording: Recording):
    member_nickname = recording.memberNickname
    file_name = recording.fileName
    s3_file_name = s3service.downloads_recording(member_nickname, file_name)

    audio = AudioSegment.from_file('recordings/'+s3_file_name)
    audio.export(member_nickname+'.wav', format='wav')
    os.remove('recordings/'+s3_file_name)
    recommendation = Recommendation(member_nickname)

    result = recommendation.get_result()
    os.remove(member_nickname+'.wav')
    response = dict()

    response["tone"] = result["tone"]
    response["octave"] = result["octave"]
    songs = result["songs"]
    for i in range(len(songs)):
        k = dict()
        tmp = songs[i].split(',')
        k["singer"] = tmp[0]
        k["title"] = tmp[1]
        k["songOctave"] = tmp[2]
        k["youtubeURL"] = tmp[3]
        k["albumCover"] = tmp[4]
        k["lyrics"] = tmp[5]
        k["duration"] = tmp[6];
        response["result"+str(i+1)] = k

    ret = json.dumps(response)
    return ret






    

