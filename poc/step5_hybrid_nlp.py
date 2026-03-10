"""
ST545 POC v10 (Hybrid) — Step 5: Combined TF-IDF (Lasso) + FinBERT (PCA)
========================================================================
1. Apply Lasso (L1) to select top 20 keywords from TF-IDF.
2. Apply PCA to reduce 768-dim FinBERT to 16 dims.
3. Feature Fusion: Combined features + Market data.
4. Model: Tuned XGBoost (GridSearchCV) with TimeSeriesSplit.
"""

import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV
import xgboost as xgb
import torch
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re
import warnings
warnings.filterwarnings('ignore')

# ── 1. Load Data ──
TICKERS = ['NVDA', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'LMT', 'NEM', 'AAPL', 'META', 'JPM']
CACHE_PATH = 'dataset/sentiment_cache.csv'
EMBED_CACHE_PATH = 'dataset/finbert_embeddings_768_v8.npy'
RESULT_DIR = 'poc/result/step5/'
os.makedirs(RESULT_DIR, exist_ok=True)

cache_df = pd.read_csv(CACHE_PATH)
cache_df['Date'] = pd.to_datetime(cache_df['Date'])
embeddings = np.load(EMBED_CACHE_PATH)
cache_embed_cols = [f'emb_{i}' for i in range(768)]
cache_df = pd.concat([cache_df.reset_index(drop=True), pd.DataFrame(embeddings, columns=cache_embed_cols)], axis=1)

# Text Prep
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))
def preprocess_text(text):
    if not isinstance(text, str): return ""
    text = re.sub('[^a-zA-Z]', ' ', text).lower().split()
    return ' '.join([ps.stem(word) for word in text if word not in stop_words])

all_ticker_summary = []

# Hyperparameter Grid
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.1]
}

for ticker in TICKERS:
    print(f"\n>>> Hybrid Experiment: {ticker}...")
    market = pd.read_csv(f"dataset/{ticker}_market.csv")
    market['Date'] = pd.to_datetime(market['Date'])
    market = market.sort_values('Date')
    market['Next_Close'] = market['Close'].shift(-1)
    market['Price_Label'] = (market['Next_Close'] > market['Close']).astype(int)
    market = market.dropna(subset=['Price_Label'])
    
    # Technicals
    market['vol_pct_chg'] = market['Volume'].pct_change()
    market['pe_chg'] = market['PE_Ratio'].diff()
    market['ma10_ratio'] = market['Close'] / market['Close'].rolling(10).mean()
    market['volatility_5d'] = market['Close'].pct_change().rolling(5).std()

    ticker_news = cache_df[cache_df['Ticker'] == ticker].copy()
    if len(ticker_news) < 100: continue

    # A. Daily Aggregation
    ticker_news['Full_Text'] = ticker_news['Headline'].fillna('') + " " + ticker_news['Summary'].fillna('')
    daily_text = ticker_news.groupby('Date')['Full_Text'].apply(lambda x: " ".join(x)).reset_index()
    daily_embed = ticker_news.groupby('Date')[cache_embed_cols].mean().reset_index()

    # Merge
    df_t = pd.merge(market, daily_text, on='Date', how='inner')
    df_t = pd.merge(df_t, daily_embed, on='Date', how='inner').dropna().reset_index(drop=True)
    if len(df_t) < 50: continue

    # ── B. Feature 1: Top 20 Keywords via Lasso ──
    texts_processed = [preprocess_text(t) for t in df_t['Full_Text']]
    tfidf_vec = TfidfVectorizer(max_features=1000) # Start with 1000 candidate features
    tfidf_raw = tfidf_vec.fit_transform(texts_processed).toarray()
    words = tfidf_vec.get_feature_names_out()
    
    y = df_t['Price_Label'].values
    tscv = TimeSeriesSplit(n_splits=3)
    sc = StandardScaler()

    # Use Lasso (L1) to select the most predictive words
    print(f"  Selecting top 20 keywords via Lasso...")
    lasso_model = LogisticRegressionCV(penalty='l1', solver='saga', cv=tscv, max_iter=5000, random_state=42)
    lasso_model.fit(sc.fit_transform(tfidf_raw), y)
    
    coefs = lasso_model.coef_[0]
    coef_df = pd.DataFrame({'word': words, 'coef_abs': np.abs(coefs)})
    # Sort by absolute magnitude and pick top 20
    top_20_words_df = coef_df.sort_values('coef_abs', ascending=False).head(20)
    top_20_indices = top_20_words_df.index.tolist()
    top_20_features = tfidf_raw[:, top_20_indices]
    top_20_names = top_20_words_df['word'].tolist()
    print(f"  Top keywords: {', '.join(top_20_names[:5])}...")

    # ── C. Feature 2: PCA-FinBERT (16 dims) ──
    print(f"  Reducing FinBERT to 16 dims (PCA)...")
    pca_fin = PCA(n_components=16, random_state=42)
    fin_pca_data = pca_fin.fit_transform(df_t[cache_embed_cols].values)

    # ── D. Hybrid Feature Fusion ──
    MARKET_BASE = ['PE_Ratio', 'vol_pct_chg', 'pe_chg', 'ma10_ratio', 'volatility_5d']
    X_market = df_t[MARKET_BASE].values
    X_hybrid = np.hstack((X_market, top_20_features, fin_pca_data))

    # ── E. Train XGBoost ──
    print(f"  Tuning Hybrid XGBoost...")
    xgb_model = xgb.XGBClassifier(n_jobs=-1, random_state=42, eval_metric='logloss')
    grid = GridSearchCV(xgb_model, param_grid, cv=tscv, scoring='roc_auc', n_jobs=-1)
    grid.fit(sc.fit_transform(X_hybrid), y)
    
    best_auc = grid.best_score_
    print(f"  Hybrid AUC: {best_auc:.4f}")

    all_ticker_summary.append({
        'Ticker': ticker,
        'Hybrid_AUC': best_auc,
        'Samples': len(df_t),
        'Top_Keywords': ", ".join(top_20_names)
    })

# Final Report
res_df = pd.DataFrame(all_ticker_summary)
res_df.to_csv(f'{RESULT_DIR}/hybrid_results_v10.csv', index=False)

with open(f'{RESULT_DIR}/hybrid_report_v10.txt', 'w') as f:
    f.write("ST545 POC v10 Hybrid Experiment - TF-IDF (Lasso Top 20) + FinBERT (PCA 16)\n" + "="*80 + "\n")
    f.write(res_df[['Ticker', 'Hybrid_AUC', 'Samples']].to_string(index=False))
    f.write(f"\n\nMean Hybrid AUC: {res_df['Hybrid_AUC'].mean():.4f}\n")
    f.write("\nPer-Ticker Top Keywords (First 10):\n")
    for _, row in res_df.iterrows():
        f.write(f"  {row['Ticker']:6s}: {', '.join(row['Top_Keywords'].split(', ')[:10])}\n")

print(f"\n[+] Hybrid experiment complete. Results saved in {RESULT_DIR}")
