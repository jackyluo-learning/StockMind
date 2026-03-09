"""
ST545 POC v9 (Unified) — Step 1 & 2: NLP Benchmarking (TF-IDF vs FinBERT)
===========================================================================
Refactored to compare NLP representations for EACH ticker in one place.
1. Global EDA (Publisher Distribution).
2. Ticker-Specific TF-IDF vs FinBERT Embedding performance comparison.
"""

import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import warnings
warnings.filterwarnings('ignore')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

PUBLISHER_NORM = {'benzinga': 'Benzinga'}
TICKERS = ['NVDA', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'LMT', 'NEM', 'AAPL', 'META', 'JPM']
CACHE_PATH = 'dataset/sentiment_cache.csv'
EMBED_CACHE_PATH = 'dataset/finbert_embeddings_768_v8.npy'

# 1. Load Data & Embeddings
print("--- Loading Global Cache & FinBERT Embeddings ---")
cache_df = pd.read_csv(CACHE_PATH)
valid_pubs = cache_df['Publisher'].value_counts()[lambda x: x >= 100].index.tolist()
cache_df = cache_df[cache_df['Publisher'].isin(valid_pubs)].copy()
cache_df['Date'] = pd.to_datetime(cache_df['Date'])

if os.path.exists(EMBED_CACHE_PATH):
    embeddings = np.load(EMBED_CACHE_PATH)
    cache_embed_cols = [f'emb_{i}' for i in range(768)]
    cache_df = pd.concat([cache_df.reset_index(drop=True), pd.DataFrame(embeddings, columns=cache_embed_cols)], axis=1)
else:
    print("WARNING: FinBERT cache not found. Only TF-IDF will be run.")
    cache_embed_cols = []

# 2. Text Preprocessing
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))
def preprocess_text(text):
    if not isinstance(text, str): return ""
    text = re.sub('[^a-zA-Z]', ' ', text).lower().split()
    return ' '.join([ps.stem(word) for word in text if word not in stop_words])

all_ticker_results = []
all_data_for_eda = []

print("\n--- Starting Ticker-Specific Benchmarking (TF-IDF vs FinBERT) ---")

for ticker in TICKERS:
    print(f"\n>>> Analyzing {ticker}...")
    market = pd.read_csv(f"dataset/{ticker}_market.csv")
    market['Date'] = pd.to_datetime(market['Date'])
    market['Next_Close'] = market['Close'].shift(-1)
    market['Price_Label'] = (market['Next_Close'] > market['Close']).astype(int)
    market = market.dropna(subset=['Price_Label'])

    ticker_news = cache_df[cache_df['Ticker'] == ticker].copy()
    df_t = pd.merge(ticker_news, market[['Date', 'Price_Label']], on='Date', how='inner').dropna()
    all_data_for_eda.append(df_t)
    
    if len(df_t) < 100:
        print(f"  Skipping {ticker} (insufficient data).")
        continue

    # A. TF-IDF Prep
    texts = (df_t['Headline'].fillna('') + " " + df_t['Summary'].fillna('')).tolist()
    X_tfidf = TfidfVectorizer(max_features=500).fit_transform([preprocess_text(t) for t in texts]).toarray()
    
    # B. FinBERT Prep (if exists)
    X_fin = df_t[cache_embed_cols].values if cache_embed_cols else None
    
    y = df_t['Price_Label'].values
    tscv = TimeSeriesSplit(n_splits=3)
    sc = StandardScaler()
    
    scores = {'TF-IDF': [], 'FinBERT': []}
    for tr, te in tscv.split(y):
        # Model TF-IDF
        m1 = LogisticRegression(max_iter=1000).fit(sc.fit_transform(X_tfidf[tr]), y[tr])
        scores['TF-IDF'].append(roc_auc_score(y[te], m1.predict_proba(sc.transform(X_tfidf[te]))[:, 1]))
        
        # Model FinBERT
        if X_fin is not None:
            m2 = LogisticRegression(max_iter=1000).fit(sc.fit_transform(X_fin[tr]), y[tr])
            scores['FinBERT'].append(roc_auc_score(y[te], m2.predict_proba(sc.transform(X_fin[te]))[:, 1]))

    res = {
        'Ticker': ticker,
        'TF-IDF_AUC': np.mean(scores['TF-IDF']),
        'FinBERT_AUC': np.mean(scores['FinBERT']) if cache_embed_cols else 0.0,
        'Samples': len(df_t)
    }
    print(f"  {ticker} Done. TF-IDF: {res['TF-IDF_AUC']:.4f} | FinBERT: {res['FinBERT_AUC']:.4f}")
    all_ticker_results.append(res)

# 3. Aggregation & Viz
res_df = pd.DataFrame(all_ticker_results)
res_df.to_csv('poc/result/step1_2/step1_2_ticker_summary.csv', index=False)

# Global EDA Plot
global_df = pd.concat(all_data_for_eda)
plt.figure(figsize=(12, 6))
top_p = global_df['Publisher'].value_counts().head(10)
sns.barplot(x=top_p.values, y=top_p.index, palette='viridis')
plt.title('Global Publisher Distribution')
plt.savefig('poc/result/step1_2/publisher_distribution.png', bbox_inches='tight'); plt.close()

# Comparison Plot
plt.figure(figsize=(12, 6))
x_axis = np.arange(len(res_df))
width = 0.35
plt.bar(x_axis - width/2, res_df['TF-IDF_AUC'], width, label='TF-IDF', color='lightgray')
plt.bar(x_axis + width/2, res_df['FinBERT_AUC'], width, label='FinBERT', color='skyblue')
plt.xticks(x_axis, res_df['Ticker'])
plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
plt.ylabel('ROC-AUC')
plt.title('Benchmarking: TF-IDF vs FinBERT (Ticker-Specific)')
plt.legend(); plt.savefig('poc/result/step1_2/finbert_tfidf_comparison.png', bbox_inches='tight'); plt.close()

with open('poc/result/step1_2/eda_tfidf_results.txt', 'w') as f:
    f.write("ST545 POC v9 Step 1 & 2 - Unified NLP Benchmarking\n" + "="*60 + "\n")
    f.write(res_df.to_string(index=False))
    if cache_embed_cols:
        gain = res_df['FinBERT_AUC'].mean() - res_df['TF-IDF_AUC'].mean()
        f.write(f"\n\nAverage FinBERT Gain over TF-IDF: {gain:+.4f}")

print(f"\n[+] Step 1 & 2 complete. Reports and plots generated in poc/result/step1_2/")
