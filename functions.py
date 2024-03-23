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

class CommentAnalysis:

    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def obtain_sentiment(self, text):
        encoded_text = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            output = self.model(**encoded_text)
            scores = output[0][0].detach().numpy()
            scores = softmax(scores)
            ranking = np.argsort(scores)
            ranking = ranking[::-1]
            emotion = self.config.id2label[ranking[0]]
            if emotion == 'neutral' and ranking[1]>0.01:
                emotion = self.config.id2label[ranking[1]]
            return emotion

    def call_api(self, videoId):
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
            maxResults=1000
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
                comment['textDisplay'],
                comment['likeCount'],
                comment['parentId']
            ])

        return pd.DataFrame(comments, columns=['author', 'published_at', 'updated_at', 'likeCount', 'text'])

    def wrangle_text(self, text):
        soup = BeautifulSoup(text, 'lxml')
        text_without_tags = soup.get_text(separator='\n')
        url_pattern = r'https?://\S+|www\.\S+'
        text_without_urls = re.sub(url_pattern, '', text_without_tags)
        wrangled_text = contractions.fix(text_without_urls)
        return wrangled_text


    def get_sentiment(self, videoId):
        comments = self.call_api(videoId)
        comments['wrangled_text'] = comments['text'].apply(self.wrangle_text)
        comments['sentiment'] = comments['wrangled_text'].apply(self.obtain_sentiment)
        return comments

    def obtain_proportion(self, comments):
        print(comments)
        sentiment_proportion = comments['sentiment'].value_counts(normalize=True)
        return sentiment_proportion

    def get_proportion(self, videoId):
        sentiments = self.get_sentiment(videoId)
        proportion = self.obtain_proportion(sentiments)
        proportion_json = proportion.to_json(orient='index')
        return json.loads(proportion_json)

    def get_timeline(self, videoId):
        comments = self.call_api(videoId)
        timeline = pd.to_datetime(comments['published_at']).dt.date.value_counts().sort_index()
        timeline_json = timeline.to_json(orient='index')
        return json.loads(timeline_json)
    
class ExtractFeatures:

    def call_api(self, videoId):
        load_dotenv()
        api_service_name = 'youtube'
        api_version = 'v3'
        api_key = os.getenv('YOUTUBE_API_KEY')

        youtube = googleapiclient.discovery.build(
            api_service_name, api_version, developerKey = api_key  
        )

        request = youtube.videos().list(
            part='snippet',
            id=videoId
        )

        res = request.execute()

        item = res['items'][0]['snippet']['thumbnails']['medium']
        thumbnail  = [{
            'url': item['url'], 'width': item['width'], 'height': item['height']
            }]
        thumbnail = pd.DataFrame(thumbnail, columns=['url', 'width', 'height'])
        thumbnail_json = thumbnail.to_json(orient='index')

        item = res['items'][0]['snippet']['title']
        title = [{
            'title': item
            }]
        title = pd.DataFrame(title, columns=['title'])
        title_json = title.to_json(orient='index')

        return json.loads(thumbnail_json), json.loads(title_json) 