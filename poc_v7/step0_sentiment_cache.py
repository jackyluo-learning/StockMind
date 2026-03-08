"""
ST545 POC v4 — Step 0: FinBERT Sentiment Cache
================================================
Run this ONCE to compute FinBERT sentiment for all articles across 10 tickers.
Saves to dataset/sentiment_cache.csv, shared by step3/4/5.

Publisher normalization: benzinga → Benzinga.
"""

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

PUBLISHER_NORM = {'benzinga': 'Benzinga'}
TICKERS = ['NVDA', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'LMT', 'NEM', 'AAPL', 'META', 'JPM']
CACHE_PATH = 'dataset/sentiment_cache.csv'

# ── 1. Load all news, normalize publishers ──
all_news = []
for ticker in TICKERS:
    news = pd.read_csv(f"dataset/{ticker}_news.csv")
    news['Date'] = pd.to_datetime(news['Date'])
    news['Publisher'] = news['Publisher'].replace(PUBLISHER_NORM)
    news['Summary'] = news['Summary'].fillna('')
    news['Text'] = news['Headline'].str.strip() + '. ' + news['Summary'].str.strip()
    all_news.append(news)

df = pd.concat(all_news, ignore_index=True).sort_values('Date').reset_index(drop=True)
print(f"--- Total articles: {len(df)} across {df['Ticker'].nunique()} tickers ---")
print(f"--- Publishers (after normalization):\n{df['Publisher'].value_counts().to_string()}\n")

# ── 2. FinBERT Sentiment Score ──
print("--- Loading FinBERT (ProsusAI/finbert)... ---")
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
finbert = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
finbert.eval()

def get_finbert_sentiment(texts, batch_size=64):
    """Return P(positive) - P(negative) for each text."""
    scores = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors='pt', padding=True,
                           truncation=True, max_length=128)
        with torch.no_grad():
            outputs = finbert(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).numpy()
        batch_scores = probs[:, 0] - probs[:, 1]
        scores.extend(batch_scores)
        if (i // batch_size) % 50 == 0:
            print(f"  FinBERT: {i + len(batch)}/{len(texts)}")
    return np.array(scores)

print("--- Running FinBERT on all articles... ---")
df['Sentiment_Score'] = get_finbert_sentiment(df['Text'].tolist())
print(f"--- FinBERT done. Mean sentiment: {df['Sentiment_Score'].mean():.4f}, "
      f"Std: {df['Sentiment_Score'].std():.4f} ---")

# ── 3. Save cache ──
cache_df = df[['Date', 'Ticker', 'Publisher', 'Headline', 'Summary', 'Sentiment_Score']].copy()
cache_df['Date'] = cache_df['Date'].dt.strftime('%Y-%m-%d')
cache_df.to_csv(CACHE_PATH, index=False)
print(f"\n[+] Sentiment cache saved: {CACHE_PATH}")
print(f"    {len(cache_df)} rows, columns: {list(cache_df.columns)}")
print(f"    Publisher distribution:")
for pub, cnt in cache_df['Publisher'].value_counts().items():
    mean_s = cache_df.loc[cache_df['Publisher'] == pub, 'Sentiment_Score'].mean()
    print(f"      {pub:20s}  {cnt:>6d} articles  mean_sentiment={mean_s:+.4f}")
print(f"    Per-ticker counts:")
for t in TICKERS:
    cnt = len(cache_df[cache_df['Ticker'] == t])
    print(f"      {t:6s}  {cnt:>6d} articles")
