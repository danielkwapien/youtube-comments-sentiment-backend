import os
import json
from dotenv import load_dotenv, dotenv_values
from flask import jsonify
from transformers import pipeline
import googleapiclient.discovery
import googleapiclient.errors
import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
import lxml
import contractions
import pickle
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from scipy.special import softmax
import numpy as np

"""with open("server/roberta-go-emotions.bin", "rb") as f:
  model = torch.load(f, map_location=torch.device('cpu'))"""

tokenizer = AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")
config = AutoConfig.from_pretrained("SamLowe/roberta-base-go_emotions")
model = AutoModelForSequenceClassification.from_pretrained("SamLowe/roberta-base-go_emotions")

def obtain_sentiment(text):
    encoded_text = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        output = model(**encoded_text)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        ranking = np.argsort(scores)
        ranking = ranking[::-1]
        emotion = config.id2label[ranking[0]]
        if emotion == 'neutral' and ranking[1]>0.01:
            emotion = config.id2label[ranking[1]]
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
    print(comments)
    sentiment_proportion = comments['sentiment'].value_counts(normalize=True)
    return sentiment_proportion

def get_proportion(videoId):
    sentiments = get_sentiment(videoId)
    proportion = obtain_proportion(sentiments)
    proportion_json = proportion.to_json(orient='index')
    return json.loads(proportion_json)

def get_timeline(videoId):
    comments = call_api(videoId)
    timeline = pd.to_datetime(comments['published_at']).dt.date.value_counts().sort_index()
    timeline_json = timeline.to_json(orient='index')
    return json.loads(timeline_json)