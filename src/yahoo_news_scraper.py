import requests
from bs4 import BeautifulSoup
import json
import pandas as pd

# Load stock ticker from config file
with open("config.json", "r") as f:
    config = json.load(f)
stock_ticker = config["stock_ticker"]  # Read ticker

# Function to scrape Yahoo Finance news headlines
def get_yahoo_finance_news(stock_ticker):
    url = f"https://finance.yahoo.com/quote/{stock_ticker}/news?p={stock_ticker}"
    headers = {"User-Agent": "Mozilla/5.0"}

    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    # Extract news headlines
    headlines = [item.text.strip() for item in soup.find_all("h3") if item.text.strip()]

    return headlines

# Get news headlines for the stock
news_headlines = get_yahoo_finance_news(stock_ticker)

# Convert to DataFrame
df_news = pd.DataFrame(news_headlines, columns=["Headline"])
print(df_news.head())  # Display first few headlines
