import os
from dotenv import load_dotenv, dotenv_values
from transformers import pipeline
import googleapiclient.discovery
import googleapiclient.errors
import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
import lxml
import contractions

def obtain_sentiment(text):
    pipe = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions")
    emotion = pipe(text)
    return emotion

def call_api(videoId):
    load_dotenv()
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

def wrangle_text(text):
  # Remove HTML tags using BeautifulSoup with lxml parser
  soup = BeautifulSoup(text, 'lxml')
  print(soup)
  text_without_tags = soup.get_text(separator='\n')


  # Remove URLs using regular expression
  url_pattern = r'https?://\S+|www\.\S+'
  text_without_urls = re.sub(url_pattern, '', text_without_tags)

  # Expand contractions
  wrangled_text = contractions.fix(text_without_urls)

  return wrangled_text


def get_sentiment(videoId):
    comments = call_api(videoId)
    comments['wrangled_text'] = comments['text'].apply(wrangle_text)
    comments['sentiment'] = comments['wrangled_text'].apply(obtain_sentiment)
    return comments

def obtain_proportion(comments):
    sentiment_proportion = comments['sentiment'].apply(lambda x: x[0]['label']).value_counts(normalize=True)
    return sentiment_proportion
