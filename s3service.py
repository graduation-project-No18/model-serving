import boto3
from dotenv import load_dotenv
import os

load_dotenv()
bucket = 'no18'


def downloads_recording(member_nickname, file_name):
    s3 = boto3.resource('s3',
                        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY'),
                        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'))

    key = file_name
    f_name = file_name.split("/")[1]
    s3.Bucket(bucket).download_file(Key=key, Filename='recordings/'+f_name)
    return f_name
