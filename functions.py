from transformers import pipeline

def obtain_sentiment(text):
    pipe = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions")
    emotion = pipe(text)
    return emotion

