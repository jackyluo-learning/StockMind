"""
ST545 POC v9 (Enhanced) — Step 4: NLP Representation Battle (XGBoost)
========================================================================
1. Ticker-specific modeling for all 10 stocks.
2. Fair Comparison: XGBoost with 16-dim FinBERT PCA vs 16-dim TF-IDF PCA.
3. Quantifies if deep semantics beats statistical frequency in non-linear regimes.
"""

import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
import shap
import torch
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re
import warnings
warnings.filterwarnings('ignore')

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

TICKERS = ['NVDA', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'LMT', 'NEM', 'AAPL', 'META', 'JPM']
CACHE_PATH = 'dataset/sentiment_cache.csv'
EMBED_CACHE_PATH = 'dataset/finbert_embeddings_768_v8.npy'

# 1. Load Data & NLP Cache
cache_df = pd.read_csv(CACHE_PATH)
valid_pubs = cache_df['Publisher'].value_counts()[lambda x: x >= 100].index.tolist()
cache_df = cache_df[cache_df['Publisher'].isin(valid_pubs)].copy()
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

for ticker in TICKERS:
    print(f"\n>>> Battle: {ticker}...")
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

    # A. FinBERT Daily Aggregation
    agg_dict = {'Sentiment_Score': ['mean', 'std', 'max'], **{c: 'mean' for c in cache_embed_cols}}
    daily_sent = ticker_news.groupby('Date').agg(agg_dict).reset_index()
    daily_sent.columns = ['Date', 'sent_mean', 'sent_std', 'sent_max'] + cache_embed_cols
    for lag in [1, 2, 3]: daily_sent[f'sent_mean_lag{lag}'] = daily_sent['sent_mean'].shift(lag)

    # B. TF-IDF Daily Aggregation (Concatenate all news text for that day)
    ticker_news['Full_Text'] = ticker_news['Headline'].fillna('') + " " + ticker_news['Summary'].fillna('')
    daily_text = ticker_news.groupby('Date')['Full_Text'].apply(lambda x: " ".join(x)).reset_index()
    
    # Merge all
    df_t = pd.merge(market, daily_sent, on='Date', how='inner')
    df_t = pd.merge(df_t, daily_text, on='Date', how='inner').dropna().reset_index(drop=True)
    if len(df_t) < 50: continue

    # --- Feature Engineering ---
    MARKET_BASE = ['PE_Ratio', 'vol_pct_chg', 'pe_chg', 'ma10_ratio', 'volatility_5d']
    SENT_BASE = ['sent_mean', 'sent_std', 'sent_max', 'sent_mean_lag1', 'sent_mean_lag2', 'sent_mean_lag3']
    X_market_sent = df_t[MARKET_BASE + SENT_BASE].values
    
    # PCA FinBERT (16-dim)
    pca_fin = PCA(n_components=16, random_state=42)
    X_fin_pca = pca_fin.fit_transform(df_t[cache_embed_cols].values)
    
    # PCA TF-IDF (16-dim for fair fight)
    tfidf_raw = TfidfVectorizer(max_features=500).fit_transform([preprocess_text(t) for t in df_t['Full_Text']]).toarray()
    pca_tfidf = PCA(n_components=16, random_state=42)
    X_tfidf_pca = pca_tfidf.fit_transform(tfidf_raw)
    
    y = df_t['Price_Label'].values
    
    # --- XGBoost Cross-Validation ---
    tscv = TimeSeriesSplit(n_splits=3)
    sc = StandardScaler()
    
    def get_xgb_auc(X_addon):
        X_full = np.hstack((X_market_sent, X_addon))
        aucs = []
        for tr, te in tscv.split(X_full):
            model = xgb.XGBClassifier(n_estimators=100, max_depth=3, n_jobs=-1, random_state=42)
            model.fit(sc.fit_transform(X_full[tr]), y[tr])
            aucs.append(roc_auc_score(y[te], model.predict_proba(sc.transform(X_full[te]))[:, 1]))
        return np.mean(aucs)

    auc_fin = get_xgb_auc(X_fin_pca)
    auc_tfidf = get_xgb_auc(X_tfidf_pca)
    
    print(f"  {ticker} -> FinBERT XGB: {auc_fin:.4f} | TF-IDF XGB: {auc_tfidf:.4f}")
    all_ticker_summary.append({
        'Ticker': ticker, 'FinBERT_XGB_AUC': auc_fin, 
        'TF-IDF_XGB_AUC': auc_tfidf, 'Delta': auc_fin - auc_tfidf
    })

# Final Report & Viz
res_df = pd.DataFrame(all_ticker_summary)
plt.figure(figsize=(12, 6))
x = np.arange(len(res_df))
width = 0.35
plt.bar(x - width/2, res_df['TF-IDF_XGB_AUC'], width, label='TF-IDF + XGB', color='lightgray')
plt.bar(x + width/2, res_df['FinBERT_XGB_AUC'], width, label='FinBERT + XGB', color='skyblue')
plt.xticks(x, res_df['Ticker'])
plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
plt.title('NLP Representation Battle (XGBoost Classifier)')
plt.legend(); plt.savefig('poc/result/representation_comparison_xgb.png', bbox_inches='tight'); plt.close()

with open('poc/result/step4_results.txt', 'w') as f:
    f.write("ST545 POC v9 Step 4 - NLP Representation Comparison (XGBoost)\n" + "="*70 + "\n")
    f.write(res_df.to_string(index=False))
    f.write(f"\n\nMean FinBERT Advantage: {res_df['Delta'].mean():+.4f}")

print("\n[+] Step 4 complete. NLP battle results saved.")
