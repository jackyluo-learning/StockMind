"""
ST545 POC v13 (Gated Hybrid - Reduced) — Step 5: 8-dim PCA Gated Hybrid NLP
========================================================================
1. Global PCA: Reduce 768-dim FinBERT to 8-dim for all articles.
2. Media Weighting: Map publisher weights from Step 3.
3. DQS Logic: (Count/10) * (1 - Std) to assess daily signal quality.
4. Gating: Directly multiply PCA embeddings by DQS to dampen noise.
5. Feature Fusion: Market + DQS/Mean/Std + Gated PCA + Lasso Top 10 Keywords.
6. Feature Synergy Breakdown (XGBoost):
   - Market-only
   - Sentiment-only (PCA-8 Gated + DQS)
   - Keywords-only (Lasso Top 10)
7. Model: Tuned XGBoost (GridSearchCV) with TimeSeriesSplit.
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

# ── 1. Load Data & Global PCA ──
TICKERS = ['NVDA', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'LMT', 'NEM', 'AAPL', 'META', 'JPM']
CACHE_PATH = 'dataset/sentiment_cache.csv'
EMBED_CACHE_PATH = 'dataset/finbert_embeddings_768_v8.npy'
RESULT_DIR = 'poc/result/step5/'
os.makedirs(RESULT_DIR, exist_ok=True)

print("--- Loading Global Cache & Performing Global 8-dim PCA ---")
cache_df = pd.read_csv(CACHE_PATH)
embeddings = np.load(EMBED_CACHE_PATH)

# Global PCA reduction for consistency (8-dim)
pca_global = PCA(n_components=8, random_state=42)
pca_8_data = pca_global.fit_transform(embeddings)
pca_cols = [f'pca_{i}' for i in range(8)]
cache_df = pd.concat([cache_df.reset_index(drop=True), pd.DataFrame(pca_8_data, columns=pca_cols)], axis=1)
cache_df['Date'] = pd.to_datetime(cache_df['Date'])

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

def tune_and_evaluate(X_in, y_in, tscv):
    sc = StandardScaler()
    X_scaled = sc.fit_transform(X_in)
    xgb_model = xgb.XGBClassifier(n_jobs=-1, random_state=42, eval_metric='logloss')
    grid = GridSearchCV(xgb_model, param_grid, cv=tscv, scoring='roc_auc', n_jobs=-1)
    grid.fit(X_scaled, y_in)
    return grid.best_score_

for ticker in TICKERS:
    print(f"\n>>> Gated Hybrid Experiment (8-dim PCA): {ticker}...")
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

    # --- 1. 映射媒体权重 ---
    weight_path = f'poc/result/step3/weights_{ticker}.csv'
    if os.path.exists(weight_path):
        weights_df = pd.read_csv(weight_path)
        weight_map = dict(zip(weights_df['Publisher'], weights_df['Coef']))
        ticker_news['Pub_Weight'] = ticker_news['Publisher'].map(weight_map).fillna(1.0)
    else:
        ticker_news['Pub_Weight'] = 1.0
    
    ticker_news['Weighted_Sent'] = ticker_news['Sentiment_Score'] * ticker_news['Pub_Weight']

    # --- 2. 统一聚合 ---
    ticker_news['Full_Text'] = ticker_news['Headline'].fillna('') + " " + ticker_news['Summary'].fillna('')
    agg_logic = {
        'Full_Text': lambda x: " ".join(x),
        'Sentiment_Score': ['std', 'count'], 
        'Weighted_Sent': 'sum',
        'Pub_Weight': 'sum',
        **{c: 'mean' for c in pca_cols} 
    }

    daily_agg = ticker_news.groupby('Date').agg(agg_logic).reset_index()
    daily_agg.columns = ['Date', 'Full_Text', 'sent_std', 'news_count', 'sum_w_sent', 'sum_w_pub'] + pca_cols

    # --- 3. 计算 DQS 和加权均值 ---
    daily_agg['DQS'] = (daily_agg['news_count'] / 10).clip(upper=1.0) * (1.0 - daily_agg['sent_std'].fillna(0).clip(upper=1.0))
    daily_agg['sent_weighted_mean'] = np.where(daily_agg['sum_w_pub'] != 0, 
                                              daily_agg['sum_w_sent'] / daily_agg['sum_w_pub'], 0)

    # --- 4. 门控操作 ---
    for col in pca_cols:
        daily_agg[col] = daily_agg[col] * daily_agg['DQS']

    # --- 5. 最终清理与特征融合 ---
    df_t = pd.merge(market, daily_agg, on='Date', how='inner').dropna().reset_index(drop=True)
    if len(df_t) < 50: continue

    # Lasso for Keywords
    texts_processed = [preprocess_text(t) for t in df_t['Full_Text']]
    tfidf_vec = TfidfVectorizer(max_features=1000)
    tfidf_raw = tfidf_vec.fit_transform(texts_processed).toarray()
    
    y = df_t['Price_Label'].values
    tscv = TimeSeriesSplit(n_splits=3)
    sc = StandardScaler()

    lasso_model = LogisticRegressionCV(penalty='l1', solver='saga', cv=tscv, max_iter=5000, random_state=42)
    lasso_model.fit(sc.fit_transform(tfidf_raw), y)
    
    coef_df = pd.DataFrame({'word': tfidf_vec.get_feature_names_out(), 'coef_abs': np.abs(lasso_model.coef_[0])})
    top_10_words_df = coef_df.sort_values('coef_abs', ascending=False).head(10)
    top_10_features = tfidf_raw[:, top_10_words_df.index]
    top_10_names = top_10_words_df['word'].tolist()

    # Feature Groups
    MARKET_BASE = ['PE_Ratio', 'vol_pct_chg', 'pe_chg', 'ma10_ratio', 'volatility_5d']
    SENT_DQS = ['sent_weighted_mean', 'sent_std', 'DQS']
    
    X_market = df_t[MARKET_BASE].values
    X_sent = df_t[SENT_DQS + pca_cols].values
    X_keywords = top_10_features
    X_hybrid = np.hstack((X_market, X_sent, X_keywords))

    # --- 6. Synergy Breakdown ---
    print(f"  Running feature synergy breakdown...")
    auc_market = tune_and_evaluate(X_market, y, tscv)
    auc_sent = tune_and_evaluate(X_sent, y, tscv)
    auc_keywords = tune_and_evaluate(X_keywords, y, tscv)
    auc_hybrid = tune_and_evaluate(X_hybrid, y, tscv)
    
    print(f"  Hybrid AUC: {auc_hybrid:.4f}")

    all_ticker_summary.append({
        'Ticker': ticker,
        'Hybrid_AUC': auc_hybrid,
        'Market_only': auc_market,
        'Sentiment_only': auc_sent,
        'Keywords_only': auc_keywords,
        'Samples': len(df_t),
        'Top_Keywords': ", ".join(top_10_names)
    })

# Final Report
res_df = pd.DataFrame(all_ticker_summary)
res_df.to_csv(f'{RESULT_DIR}/hybrid_results_v13.csv', index=False)

with open(f'{RESULT_DIR}/hybrid_report_v13.txt', 'w') as f:
    f.write("ST545 POC v13 Gated Hybrid - DQS Gating + Media Weights + TF-IDF (10) + FinBERT (8)\n" + "="*90 + "\n")
    f.write(res_df[['Ticker', 'Hybrid_AUC', 'Market_only', 'Sentiment_only', 'Keywords_only', 'Samples']].to_string(index=False))
    
    # Aggregated Means
    f.write("\n\n" + "-"*30 + "\n")
    f.write("AGGREGATED PERFORMANCE MEANS:\n")
    f.write(f"  Mean Hybrid AUC    : {res_df['Hybrid_AUC'].mean():.4f}\n")
    f.write(f"  Mean Market-only   : {res_df['Market_only'].mean():.4f}\n")
    f.write(f"  Mean Sentiment-only: {res_df['Sentiment_only'].mean():.4f}\n")
    f.write(f"  Mean Keywords-only : {res_df['Keywords_only'].mean():.4f}\n")
    f.write("-"*30 + "\n")

    f.write("\nPer-Ticker Top Keywords (First 10):\n")
    for _, row in res_df.iterrows():
        f.write(f"  {row['Ticker']:6s}: {', '.join(row['Top_Keywords'].split(', ')[:10])}\n")

print(f"\n[+] Gated hybrid experiment complete. Results saved in {RESULT_DIR}")
