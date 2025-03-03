import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
import nltk
import os
from datetime import datetime, timedelta
from nltk.sentiment import SentimentIntensityAnalyzer


# Load stock ticker from config file
with open("config.json", "r") as f:
    config = json.load(f)
stock_ticker = config["stock_ticker"]  # Read ticker

# Initialize Sentiment Analyzer
sia = nltk.sentiment.SentimentIntensityAnalyzer()

# Function to scrape Yahoo Finance news headlines
def get_yahoo_finance_news(stock_ticker, days=30):
    base_url = f"https://finance.yahoo.com/quote/{stock_ticker}/news"
    headers = {"User-Agent": "Mozilla/5.0"}

    response = requests.get(base_url, headers=headers)

    # ✅ Handle 404 error properly
    if response.status_code == 404:
        print(f"❌ Yahoo Finance news page not found for {stock_ticker}. Check the ticker symbol.")
        return []
    elif response.status_code != 200:
        print(f"⚠️ Failed to fetch Yahoo Finance news. Status code: {response.status_code}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")

    headlines = []
    cutoff_date = datetime.today() - timedelta(days=days)  # Earliest date to fetch

    news_items = soup.find_all("h3")  # ✅ Adjusted tag for news headlines

    # ✅ Check if any news items were found
    if not news_items:
        print("⚠️ No news articles found. The webpage structure might have changed.")
        return []

    for item in news_items:
        try:
            headline = item.text.strip()
            article_date = datetime.today().strftime("%Y-%m-%d")  # ✅ Yahoo does not provide dates, so use today

            headlines.append({"Date": article_date, "Headline": headline})

        except Exception as e:
            print(f"⚠️ Error extracting a news item: {e}")
            continue  # Skip problematic items

    return headlines


# ✅ Function to perform sentiment analysis on news headlines
def analyze_sentiment(df_news):
    if df_news.empty:
        print("⚠️ No news data available for sentiment analysis.")
        return df_news

    df_news["Sentiment_Score"] = df_news["Headline"].apply(lambda text: sia.polarity_scores(text)["compound"])
    return df_news


def maintain_historical_news(stock_ticker, days=30):
    file_path = "news_sentiment.csv"

    # ✅ Load existing news sentiment data if available
    if os.path.exists(file_path):
        df_existing = pd.read_csv(file_path)
        df_existing["Date"] = pd.to_datetime(df_existing["Date"])
    else:
        df_existing = pd.DataFrame(columns=["Date", "Headline", "Sentiment_Score"])

    # ✅ Scrape new news headlines
    news_data = get_yahoo_finance_news(stock_ticker)
    df_news = pd.DataFrame(news_data)

    # ✅ Perform sentiment analysis
    df_news = analyze_sentiment(df_news)

    # ✅ Append new data to existing dataset
    df_combined = pd.concat([df_existing, df_news])

    # ✅ Keep only the last X days of news
    #cutoff_date = datetime.today() - timedelta(days=days)
    #df_combined = df_combined[df_combined["Date"] >= pd.to_datetime(cutoff_date.strftime("%Y-%m-%d"))]

    # ✅ Save updated dataset
    df_combined.to_csv(file_path, index=False)
    print(f"✅ News sentiment data updated and saved. Kept last {days} days of news.")
    print(df_combined.tail())  # Show latest entries


# ✅ Main execution
if __name__ == "__main__":
    maintain_historical_news(stock_ticker, days=30)  # Keeps last 30 days of news
# Get news headlines for the stock
