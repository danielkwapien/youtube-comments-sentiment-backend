import os
from dotenv import load_dotenv, dotenv_values
from transformers import pipeline
import googleapiclient.discovery
import googleapiclient.errors
import pandas as pd

def obtain_sentiment(text):
    pipe = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions")
    emotion = pipe(text)
    return emotion

def call_api(videoId):
    api_service_name = 'youtube'
    api_version = 'v3'
    api_key = os.getenv('YOUTUBE_API_KEY')

    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, developerKey = api_key  
    )
    request = youtube.commentThreads().list(
        part='snippet',
        videoId=videoId,
        maxResults=100
    )

    res = request.execute()

    comments = []
    for item in res['items']:
        comment = item['snippet']['topLevelComment']['snippet']
        comments.append([
            comment['authorDisplayName'],
            comment['publishedAt'],
            comment['updatedAt'],
            comment['likeCount'],
            comment['textDisplay']
        ])

    return pd.DataFrame(comments, columns=['author', 'published_at', 'updated_at', 'likeCount', 'text'])

