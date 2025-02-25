import nltk
import pandas as pd
from dotenv import load_dotenv
import json
import os
import tweepy
from nltk.sentiment import SentimentIntensityAnalyzer

# Load the environment variables from .env
load_dotenv()

# Get API credentials from .env
api_key = os.getenv("TWITTER_API_KEY")
api_secret = os.getenv("TWITTER_API_SECRET")
access_token = os.getenv("TWITTER_ACCESS_TOKEN")
access_secret = os.getenv("TWITTER_ACCESS_SECRET")

# Authenticate Twitter API
auth = tweepy.OAuthHandler(api_key, api_secret)
auth.set_access_token(access_token, access_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)

# Download VADER sentiment analyzer (only required once)
nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

# Function to fetch tweets for a stock ticker
def get_twitter_sentiment(stock_ticker, num_tweets=10):
    try:
        # Search for recent tweets mentioning the stock
        tweets = api.search_tweets(q=f"{stock_ticker} stock", lang="en", count=num_tweets, tweet_mode="extended")

        # Extract tweet text and compute sentiment score
        tweet_data = []
        for tweet in tweets:
            text = tweet.full_text
            sentiment_score = sia.polarity_scores(text)["compound"]  # VADER sentiment score
            tweet_data.append({"Tweet": text, "Sentiment_Score": sentiment_score})

        # Convert to DataFrame
        df_tweets = pd.DataFrame(tweet_data)
        return df_tweets

    except Exception as e:
        print(f"Error fetching tweets: {e}")
        return pd.DataFrame()

# Load stock ticker from config file
with open("config.json", "r") as f:
    config = json.load(f)
stock_ticker = config["stock_ticker"]  # Read ticker
# Example usage
tweets = get_twitter_sentiment(stock_ticker)
for i, tweet in enumerate(tweets, 1):
    print(f"{i}. {tweet}")
