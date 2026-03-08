"""Export the final daily-level dataset for proposal submission."""
import pandas as pd
import numpy as np
import os

TICKERS = ['NVDA', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'LMT', 'NEM', 'AAPL', 'META', 'JPM']
cache = pd.read_csv('dataset/sentiment_cache.csv')
cache['Date'] = pd.to_datetime(cache['Date'])

all_daily = []
for ticker in TICKERS:
    market = pd.read_csv(f'dataset/{ticker}_market.csv')
    market['Date'] = pd.to_datetime(market['Date'])
    market = market.sort_values('Date')
    market['Next_Close'] = market['Close'].shift(-1)
    market['Price_Label'] = (market['Next_Close'] > market['Close']).astype(int)
    market = market.dropna(subset=['Price_Label'])
    market['Price_Label'] = market['Price_Label'].astype(int)
    market['volume_pct_chg'] = market['Volume'].pct_change()
    market['pe_chg'] = market['PE_Ratio'].diff()
    market['MA5'] = market['Close'].rolling(5).mean()
    market['MA10'] = market['Close'].rolling(10).mean()
    market['ma5_ratio'] = market['Close'] / market['MA5']
    market['ma10_ratio'] = market['Close'] / market['MA10']
    market['momentum_5d'] = market['Close'].pct_change(5)
    market['volatility_5d'] = market['Close'].pct_change().rolling(5).std()

    ticker_news = cache[cache['Ticker'] == ticker].copy()
    daily_sent = ticker_news.groupby('Date')['Sentiment_Score'].agg(
        sent_mean='mean', sent_std='std', sent_max='max', sent_min='min', news_count='count'
    ).reset_index()
    daily_sent['sent_std'] = daily_sent['sent_std'].fillna(0)

    for pub in ticker_news['Publisher'].unique():
        pub_col = f'pub_{pub}_sent'
        pub_daily = ticker_news[ticker_news['Publisher'] == pub].groupby('Date')['Sentiment_Score'].mean()
        pub_daily = pub_daily.reset_index().rename(columns={'Sentiment_Score': pub_col})
        daily_sent = pd.merge(daily_sent, pub_daily, on='Date', how='left')

    market_cols = ['Date', 'Ticker', 'PE_Ratio', 'Volume', 'Price_Label',
                   'volume_pct_chg', 'pe_chg', 'ma5_ratio', 'ma10_ratio',
                   'momentum_5d', 'volatility_5d']
    daily = pd.merge(market[market_cols], daily_sent, on='Date', how='inner')
    all_daily.append(daily)

df = pd.concat(all_daily, ignore_index=True)
df = df.dropna(subset=['Price_Label']).sort_values('Date').reset_index(drop=True)
pub_sent_cols = [c for c in df.columns if c.startswith('pub_') and c.endswith('_sent')]
df[pub_sent_cols] = df[pub_sent_cols].fillna(0)
df = df.dropna().reset_index(drop=True)

df.to_csv('dataset/final_daily_dataset.csv', index=False)
size_kb = os.path.getsize('dataset/final_daily_dataset.csv') / 1024
print(f"Rows: {len(df)}, Cols: {len(df.columns)}")
print(f"Columns: {list(df.columns)}")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
print(f"Tickers: {df['Ticker'].nunique()}")
print(f"Label balance: {df['Price_Label'].mean():.2%} positive")
print(f"File size: {size_kb:.0f} KB")
