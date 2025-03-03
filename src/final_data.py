import pandas as pd
import os
# Load stock price data
df_stock = pd.read_csv("stock_data.csv")
df_stock["Date"] = pd.to_datetime(df_stock["Date"])  # Convert Date column to datetime format
df_stock = df_stock.sort_values(by="Date")

# Load Twitter Sentiment Data
#df_twitter = pd.read_csv("twitter_sentiment.csv")
#df_twitter["Date"] = pd.to_datetime(df_twitter["Date"])

# Load News Sentiment Data
#df_news = pd.read_csv("news_sentiment.csv")
#df_news["Date"] = pd.to_datetime(df_news["Date"])

# Aggregate daily sentiment scores (mean score per day)
#df_twitter = df_twitter.groupby("Date")["Sentiment_Score"].mean().reset_index()
#df_news = df_news.groupby("Date")["Sentiment_Score"].mean().reset_index()

# Rename columns for clarity
#df_twitter.rename(columns={"Sentiment_Score": "Twitter_Sentiment"}, inplace=True)
#df_news.rename(columns={"Sentiment_Score": "News_Sentiment"}, inplace=True)

# Merge sentiment data with stock prices
#df_final = df_stock.merge(df_twitter, on="Date", how="left")
#df_final = df_final.merge(df_news, on="Date", how="left")

# Fill NaN values (if sentiment data is missing for some days)
df_final=df_stock
df_final.fillna(0, inplace=True)

# Save merged dataset
final_file_path = "final_dataset.csv"
if os.path.exists(final_file_path):
    os.remove(final_file_path)  # Delete old file if exists

df_final.to_csv(final_file_path, index=False)

print("✅ Final dataset saved as final_stock_dataset.csv")
print(df_stock.head())  # Preview the dataset
