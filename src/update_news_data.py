import os
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import json, requests

#  Load existing dataset
df = pd.read_csv("news_sentiment.csv")

#  Get today's date
today = datetime.today().strftime("%Y-%m-%d")

#  Fetch latest stock data
# Load stock ticker from config file
with open("config.json", "r") as f:
    config = json.load(f)
stock_ticker = config["stock_ticker"]  # Read ticker
#print((datetime.now() - timedelta(days=1)).strftime('%y-%m-%d'))
def get_news_data():
    # api_key = os.getenv("NEWS_API_KEY")
    api_key = '994bde3408734b91b5a38e18bc6ab41a'
    #print(api_key)
    url = 'https://newsapi.org/v2/everything'
    params = {
        'q': "Apple",
        'from': today,  # get articles from today
        'sortBy': 'relevancy',
        'apiKey': api_key,
        'pageSize': 100,  # maximum number of results per page
        'language': 'en'
    }
    # Making the request
    response = requests.get(url, params=params)
    data = response.json()
    if data['totalResults']==0:
        print(" No new data available for today.")
        exit()
    print(data)
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
'''
# used for docker on ec2 as python is installed on docker and not ec2
# Set the NLTK data directory to the mounted volume inside the container
nltk.data.path.append("/app/data/nltk_data")  # Replace with the correct path based on your mount

# Download vader_lexicon to the mounted volume
if not os.path.exists("/app/data/nltk_data/vader_lexicon"):
    nltk.download('vader_lexicon', download_dir='/app/data/nltk_data')
'''
sia = nltk.sentiment.SentimentIntensityAnalyzer()
# Function to perform sentiment analysis on news headlines
def analyze_sentiment(df_news):
    if df_news.empty:
        print("No news data available for sentiment analysis.")
        return df_news

    df_news["News_Sentiment"] = df_news["Headline"].apply(lambda text: sia.polarity_scores(text)["compound"])
    return df_news

def maintain_historical_news():
    file_path = "news_sentiment.csv"

    # Load existing news sentiment data if available
    if os.path.exists(file_path):
        df_existing = pd.read_csv(file_path)
        df_existing["Date"] = pd.to_datetime(df_existing["Date"])
        #df_existing["Date"] = df_existing["Date"].dt.strftime("%Y-%m-%d %H:%M:%S+00:00")
    else:
        df_existing = pd.DataFrame(columns=["Date", "Headline", "Sentiment_Score"])

    # Scrape new news headlines
    news_data = get_news_data()
    news_data["Date"] = pd.to_datetime(news_data["Date"], format='mixed', utc=True)
    news_data["Date"] = news_data["Date"].dt.strftime("%Y-%m-%d %H:%M:%S+00:00")
    # Convert to DataFrame

    #  Perform sentiment analysis
    news_data = analyze_sentiment(news_data)

    #  Append new data to existing dataset
    df_combined = pd.concat([df_existing, news_data])

    #  Save updated dataset
    df_combined.to_csv(file_path, index=False)
    print(df_combined.tail())  # Show latest entries


# Main execution
if __name__ == "__main__":
    maintain_historical_news()
# Get news headlines for the stock'''


