import pandas as pd
import os
# Load stock price data
df_stock = pd.read_csv("stock_data.csv")
df_stock["Date"] = pd.to_datetime(df_stock["Date"]).dt.normalize()  # Convert Date column to datetime format
df_stock = df_stock.sort_values(by="Date")

# Load Twitter Sentiment Data
#df_twitter = pd.read_csv("twitter_sentiment.csv")
#df_twitter["Date"] = pd.to_datetime(df_twitter["Date"])

# Load News Sentiment Data
df_news = pd.read_csv("news_sentiment.csv")
df_news["Date"] = pd.to_datetime(df_news["Date"], utc=True).dt.tz_localize(None)
df_news["Date"] = pd.to_datetime(df_news["Date"]).dt.normalize()
df_news = df_news.groupby("Date", as_index=False)["News_Sentiment"].mean()


# Aggregate daily sentiment scores (mean score per day)
#df_twitter = df_twitter.groupby("Date")["Sentiment_Score"].mean().reset_index()
#df_news = df_news.groupby("Date")["News_Sentiment"].mean().reset_index()
print(df_news.tail())

df_news = df_news.sort_values(by="Date")
# Initialize final dataframe with News_Sentiment column
df_final = df_stock.copy()
df_final["News_Sentiment"] = 0.0  # Initialize with zeros

# Create complete date range
all_dates = pd.DataFrame({
    "Date": pd.date_range(
        start=min(df_stock["Date"].min(), df_news["Date"].min()),
        end=max(df_stock["Date"].max(), df_news["Date"].max())
    )
})

# Mark trading days
all_dates = all_dates.merge(
    df_stock[["Date"]].assign(IsTradingDay=True),
    how="left",
    on="Date"
)
all_dates["IsTradingDay"] = all_dates["IsTradingDay"].fillna(False)

# Add news sentiment
all_dates = all_dates.merge(df_news, how="left", on="Date")

# Process in reverse chronological order
pending_sentiments = []

for date in sorted(all_dates["Date"], reverse=True):
    row = all_dates[all_dates["Date"] == date].iloc[0]

    if row["IsTradingDay"]:
        # Trading day - apply pending sentiments
        if pending_sentiments:
            avg_pending = sum(pending_sentiments) / len(pending_sentiments)
            # Add to existing sentiment
            mask = df_final["Date"] == date
            current = df_final.loc[mask, "News_Sentiment"].values[0]
            df_final.loc[mask, "News_Sentiment"] = current + avg_pending
            pending_sentiments = []

        # Add current day's sentiment if exists
        if pd.notna(row["News_Sentiment"]):
            mask = df_final["Date"] == date
            current = df_final.loc[mask, "News_Sentiment"].values[0]
            df_final.loc[mask, "News_Sentiment"] = current + row["News_Sentiment"]
    else:
        # Non-trading day - accumulate sentiment
        if pd.notna(row["News_Sentiment"]):
            pending_sentiments.append(row["News_Sentiment"])

# Rename columns for clarity
#df_twitter.rename(columns={"Sentiment_Score": "Twitter_Sentiment"}, inplace=True)
#df_news.rename(columns={"Sentiment_Score": "News_Sentiment"}, inplace=True)

# Merge sentiment data with stock prices
#df_final = df_stock.merge(df_news_aggregated, on="Date", how="left")
#df_final = df_stock.merge(df_twitter, on="Date", how="left")
#df_final = df_final.merge(df_news, on="Date", how="left")

# Fill NaN values (if sentiment data is missing for some days)
#df_final=df_stock


# Save merged dataset
final_file_path = "final_dataset.csv"
if os.path.exists(final_file_path):
    os.remove(final_file_path)  # Delete old file if exists

df_final.to_csv(final_file_path, index=False)

print("Final dataset saved as final_stock_dataset.csv")
print(df_final.tail())  # Preview the dataset
