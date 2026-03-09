"""
ST545 POC v9 (Enhanced) — Step 4: Tuned NLP Representation Battle & SHAP Analysis
===================================================================================
1. Ticker-specific modeling for all 10 stocks with GridSearchCV tuning.
2. Fair Comparison: Tuned XGBoost with FinBERT PCA vs TF-IDF PCA.
3. SHAP Interpretation: Rank importance and Sentiment vs Market attribution percentages.
"""

import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import roc_auc_score, make_scorer
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

# Hyperparameter Grid
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.1]
}

for ticker in TICKERS:
    print(f"\n>>> Tuning & Battle: {ticker}...")
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

    # B. TF-IDF Daily Aggregation
    ticker_news['Full_Text'] = ticker_news['Headline'].fillna('') + " " + ticker_news['Summary'].fillna('')
    daily_text = ticker_news.groupby('Date')['Full_Text'].apply(lambda x: " ".join(x)).reset_index()
    
    # Merge all
    df_t = pd.merge(market, daily_sent, on='Date', how='inner')
    df_t = pd.merge(df_t, daily_text, on='Date', how='inner').dropna().reset_index(drop=True)
    if len(df_t) < 50: continue

    # --- Feature Engineering ---
    MARKET_BASE = ['PE_Ratio', 'vol_pct_chg', 'pe_chg', 'ma10_ratio', 'volatility_5d']
    SENT_BASE = ['sent_mean', 'sent_std', 'sent_max', 'sent_mean_lag1', 'sent_mean_lag2', 'sent_mean_lag3']
    
    # PCA FinBERT (16-dim) directly from the 768-dim embeddings
    # These are already aggregated by 'mean' in daily_sent[cache_embed_cols]
    pca_fin = PCA(n_components=16, random_state=42)
    fin_pca_data = pca_fin.fit_transform(df_t[cache_embed_cols].values)
    pca_cols = [f'finbert_pca_{i}' for i in range(16)]
    df_fin_pca = pd.DataFrame(fin_pca_data, columns=pca_cols)
    
    # PCA TF-IDF (16-dim)
    tfidf_raw = TfidfVectorizer(max_features=500).fit_transform([preprocess_text(t) for t in df_t['Full_Text']]).toarray()
    pca_tfidf = PCA(n_components=16, random_state=42)
    X_tfidf_pca = pca_tfidf.fit_transform(tfidf_raw)
    
    y = df_t['Price_Label'].values
    tscv = TimeSeriesSplit(n_splits=3)
    sc = StandardScaler()

    # --- GridSearchCV Tuning & Battle ---
    def tune_and_evaluate(X_base, X_addon):
        X_full = np.hstack((X_base, X_addon))
        X_scaled = sc.fit_transform(X_full)
        
        xgb_model = xgb.XGBClassifier(n_jobs=-1, random_state=42, eval_metric='logloss')
        grid = GridSearchCV(xgb_model, param_grid, cv=tscv, scoring='roc_auc', n_jobs=-1)
        grid.fit(X_scaled, y)
        
        return grid.best_score_, grid.best_estimator_

    X_market_sent = df_t[MARKET_BASE + SENT_BASE].values
    
    print(f"  Tuning FinBERT path...")
    auc_fin, best_model_fin = tune_and_evaluate(X_market_sent, fin_pca_data)
    
    print(f"  Tuning TF-IDF path...")
    auc_tfidf, _ = tune_and_evaluate(X_market_sent, X_tfidf_pca)

    # --- SHAP Interpretation (Using Best FinBERT Model) ---
    X_final_df = pd.concat([df_t[MARKET_BASE + SENT_BASE], df_fin_pca], axis=1)
    X_final_scaled = sc.fit_transform(X_final_df.values)
    
    explainer = shap.TreeExplainer(best_model_fin)
    shap_values = explainer.shap_values(X_final_scaled)

    # 1. Rank Importance Plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_final_df, show=False)
    plt.title(f'SHAP Importance (Tuned): {ticker}')
    plt.savefig(f'poc/result/step4/shap_summary_{ticker}.png', bbox_inches='tight'); plt.close()

    # 2. Contribution Percentage
    abs_shap = np.abs(shap_values).mean(axis=0)
    total_shap = abs_shap.sum()
    
    market_idx = [X_final_df.columns.get_loc(c) for c in MARKET_BASE]
    sent_idx = [X_final_df.columns.get_loc(c) for c in SENT_BASE + pca_cols]
    
    mkt_contrib = abs_shap[market_idx].sum() / total_shap if total_shap > 0 else 0
    sent_contrib = abs_shap[sent_idx].sum() / total_shap if total_shap > 0 else 0
    
    print(f"  {ticker} Result -> FinBERT AUC: {auc_fin:.4f} | Mkt: {mkt_contrib:.1%} vs Sent: {sent_contrib:.1%}")
    all_ticker_summary.append({
        'Ticker': ticker, 'FinBERT_AUC': auc_fin, 'TF-IDF_AUC': auc_tfidf,
        'Market_Contrib': mkt_contrib, 'Sent_Contrib': sent_contrib
    })

# Final Report & Viz
res_df = pd.DataFrame(all_ticker_summary)
plt.figure(figsize=(12, 6))
x = np.arange(len(res_df))
width = 0.35
plt.bar(x - width/2, res_df['TF-IDF_AUC'], width, label='Tuned TF-IDF', color='lightgray')
plt.bar(x + width/2, res_df['FinBERT_AUC'], width, label='Tuned FinBERT', color='skyblue')
plt.xticks(x, res_df['Ticker'])
plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
plt.title('Tuned NLP Representation Battle (XGBoost)')
plt.legend(); plt.savefig('poc/result/step4/representation_comparison_xgb.png', bbox_inches='tight'); plt.close()

with open('poc/result/step4/step4_results.txt', 'w') as f:
    f.write("ST545 POC v9 Step 4 - Tuned NLP Battle & SHAP Attribution\n" + "="*70 + "\n")
    f.write(res_df.to_string(index=False, formatters={
        'Market_Contrib': '{:,.2%}'.format, 'Sent_Contrib': '{:,.2%}'.format
    }))
    f.write(f"\n\nMean Tuned FinBERT Advantage: {(res_df['FinBERT_AUC'] - res_df['TF-IDF_AUC']).mean():+.4f}")
    f.write(f"\nMean Sentiment Attribution: {res_df['Sent_Contrib'].mean():.2%}")

print("\n[+] Step 4 complete. Tuned models, SHAP plots, and attribution results saved.")
