from pydantic import BaseModel


class Recording(BaseModel):
    recordingId: str
    memberId: str
    memberNickname: str
    fileName: str


class Response(BaseModel):
    member_octave: str
    song_singer1: str
    song_title1: str
    sing_octave1: str
    song_youtube_link1: str
    song_singer2: str
    song_title2: str
    sing_octave2: str
    song_youtube_link2: str
    song_singer3: str
    song_title3: str
    sing_octave3: str
    song_youtube_link3: str