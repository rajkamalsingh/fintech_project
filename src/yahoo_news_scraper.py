import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
import nltk
import os
from datetime import datetime, timedelta
from nltk.sentiment import SentimentIntensityAnalyzer
#from stocknews import StockNews


# Load stock ticker from config file
with open("config.json", "r") as f:
    config = json.load(f)
stock_ticker = config["stock_ticker"]  # Read ticker


'''key = os.getenv("MY_WORLD_TRADING_DATA_KEY")
stocks = ['AAPL']
sn = StockNews(stocks, wt_key=key)
df = sn.summarize()'''

def get_news_data():
    # api_key = os.getenv("NEWS_API_KEY")
    api_key = '994bde3408734b91b5a38e18bc6ab41a'
    print(api_key)
    url = 'https://newsapi.org/v2/everything'
    params = {
        'q': "Apple",
        'from': (datetime.now() - timedelta(days=120)).strftime('%y-%m-%d'),  # get articles from last 52 weeks
        'sortBy': 'relevancy',
        'apiKey': api_key,
        'pageSize': 100,  # maximum number of results per page
        'language': 'en'
    }
    # Making the request
    response = requests.get(url, params=params)
    data = response.json()
    # Check for errors
    if data['status'] != 'ok':
        raise Exception(f"NewsaPI error: {data['message']}")
    # Extract Articles
    articles = data['articles']
    df_news = pd.DataFrame(articles)
    df_news = df_news[['publishedAt', 'title']]
    df_news.columns = ['Date', 'Headline']
    return df_news

# Initialize Sentiment Analyzer
sia = nltk.sentiment.SentimentIntensityAnalyzer()
# ✅ Function to perform sentiment analysis on news headlines
def analyze_sentiment(df_news):
    if df_news.empty:
        print("⚠️ No news data available for sentiment analysis.")
        return df_news

    df_news["News_Sentiment"] = df_news["Headline"].apply(lambda text: sia.polarity_scores(text)["compound"])
    return df_news


def maintain_historical_news():
    file_path = "news_sentiment.csv"

    # ✅ Load existing news sentiment data if available
    if os.path.exists(file_path):
        df_existing = pd.read_csv(file_path)
        df_existing["Date"] = pd.to_datetime(df_existing["Date"])
    else:
        df_existing = pd.DataFrame(columns=["Date", "Headline", "Sentiment_Score"])

    # ✅ Scrape new news headlines
    news_data = get_news_data()
    # Convert to DataFrame
    #print(news_data.head())

    # ✅ Perform sentiment analysis
    news_data = analyze_sentiment(news_data)

    # ✅ Append new data to existing dataset
    df_combined = pd.concat([df_existing, news_data])

    # ✅ Save updated dataset
    df_combined.to_csv(file_path, index=False)
    print(df_combined.tail())  # Show latest entries


# ✅ Main execution
if __name__ == "__main__":
    maintain_historical_news()
# Get news headlines for the stock'''
