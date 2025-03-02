import nltk
import pandas as pd
from dotenv import load_dotenv
import json
import os
import tweepy
from datetime import datetime
from nltk.sentiment import SentimentIntensityAnalyzer

# Load the environment variables from .env
load_dotenv()

# Load API keys from .env file
load_dotenv()
bearer_token = os.getenv("TWITTER_BEARER_TOKEN")  # Use Bearer Token for v2 API

# Authenticate using Twitter API v2
client = tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=True)

# Download VADER sentiment analyzer
nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

# Function to fetch tweets using Twitter API v2
def get_twitter_sentiment_v2(stock_ticker, num_tweets=10):
    try:
        # Search for recent tweets mentioning the stock (v2 API)
        query = f"{stock_ticker} stock -is:retweet lang:en"
        tweets = client.search_recent_tweets(query=query, max_results=num_tweets, tweet_fields=["text"])

        # Extract tweet text and compute sentiment scores
        tweet_data = []
        for tweet in tweets.data:
            text = tweet.text
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
df_sentiment = get_twitter_sentiment_v2(stock_ticker, num_tweets=10)
print(df_sentiment)
# Assume df_tweets contains sentiment scores
df_sentiment["Date"] = datetime.today().strftime("%Y-%m-%d")  # Add today's date

# Define file path
file_path = "twitter_sentiment.csv"

# Overwrite the file if it exists
if os.path.exists(file_path):
    os.remove(file_path)  # Delete the old file

df_sentiment.to_csv(file_path, index=False)

print("✅ Twitter sentiment data saved and overwritten.")
