"""
ST545 POC v11 (Weighted Hybrid) — Step 5: v11 Synergy Analysis
========================================================================
1. Global PCA: Reduce 768-dim FinBERT to 16-dim.
2. Media Weighting: Map publisher weights from Step 3.
3. No DQS Gating.
4. Lasso Top 10 Keywords.
5. Synergy Breakdown: Market vs Sentiment vs Keywords vs Hybrid.
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV
import xgboost as xgb
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

print("--- v11: Loading Global Cache & Performing Global 16-dim PCA ---")
cache_df = pd.read_csv(CACHE_PATH)
embeddings = np.load(EMBED_CACHE_PATH)

pca_global = PCA(n_components=16, random_state=42)
pca_16_data = pca_global.fit_transform(embeddings)
pca_cols = [f'pca_{i}' for i in range(16)]
cache_df = pd.concat([cache_df.reset_index(drop=True), pd.DataFrame(pca_16_data, columns=pca_cols)], axis=1)
cache_df['Date'] = pd.to_datetime(cache_df['Date'])

# Text Prep
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))
def preprocess_text(text):
    if not isinstance(text, str): return ""
    text = re.sub('[^a-zA-Z]', ' ', text).lower().split()
    return ' '.join([ps.stem(word) for word in text if word not in stop_words])

all_ticker_summary = []
param_grid = {'n_estimators': [50, 100], 'max_depth': [3, 5], 'learning_rate': [0.05, 0.1]}

def tune_and_evaluate(X_in, y_in, tscv):
    sc = StandardScaler()
    X_scaled = sc.fit_transform(X_in)
    grid = GridSearchCV(xgb.XGBClassifier(n_jobs=-1, random_state=42, eval_metric='logloss'), 
                        param_grid, cv=tscv, scoring='roc_auc', n_jobs=-1)
    grid.fit(X_scaled, y_in)
    return grid.best_score_

for ticker in TICKERS:
    print(f"\n>>> v11 Synergy Analysis: {ticker}...")
    market = pd.read_csv(f"dataset/{ticker}_market.csv")
    market['Date'] = pd.to_datetime(market['Date'])
    market = market.sort_values('Date')
    market['Next_Close'] = market['Close'].shift(-1)
    market['Price_Label'] = (market['Next_Close'] > market['Close']).astype(int)
    market = market.dropna(subset=['Price_Label'])
    
    market['vol_pct_chg'] = market['Volume'].pct_change()
    market['pe_chg'] = market['PE_Ratio'].diff()
    market['ma10_ratio'] = market['Close'] / market['Close'].rolling(10).mean()
    market['volatility_5d'] = market['Close'].pct_change().rolling(5).std()

    ticker_news = cache_df[cache_df['Ticker'] == ticker].copy()
    if len(ticker_news) < 100: continue

    # Media Weighting
    weight_path = f'poc/result/step3/weights_{ticker}.csv'
    if os.path.exists(weight_path):
        w_df = pd.read_csv(weight_path)
        w_map = dict(zip(w_df['Publisher'], w_df['Coef']))
        ticker_news['Pub_Weight'] = ticker_news['Publisher'].map(w_map).fillna(1.0)
    else:
        ticker_news['Pub_Weight'] = 1.0
    ticker_news['Weighted_Sent'] = ticker_news['Sentiment_Score'] * ticker_news['Pub_Weight']
    ticker_news['Full_Text'] = ticker_news['Headline'].fillna('') + " " + ticker_news['Summary'].fillna('')

    # Aggregation
    agg_logic = {
        'Full_Text': lambda x: " ".join(x),
        'Weighted_Sent': 'sum',
        'Pub_Weight': 'sum',
        **{c: 'mean' for c in pca_cols}
    }
    daily_agg = ticker_news.groupby('Date').agg(agg_logic).reset_index()
    daily_agg['sent_weighted_mean'] = np.where(daily_agg['Pub_Weight'] != 0, 
                                              daily_agg['Weighted_Sent'] / daily_agg['Pub_Weight'], 0)

    df_t = pd.merge(market, daily_agg, on='Date', how='inner').dropna().reset_index(drop=True)
    if len(df_t) < 50: continue

    # Lasso Keywords
    texts_processed = [preprocess_text(t) for t in df_t['Full_Text']]
    tfidf_vec = TfidfVectorizer(max_features=1000)
    tfidf_raw = tfidf_vec.fit_transform(texts_processed).toarray()
    y = df_t['Price_Label'].values
    tscv = TimeSeriesSplit(n_splits=3)
    
    lasso = LogisticRegressionCV(penalty='l1', solver='saga', cv=tscv, max_iter=5000, random_state=42).fit(StandardScaler().fit_transform(tfidf_raw), y)
    top_10_idx = pd.DataFrame({'abs_coef': np.abs(lasso.coef_[0])}).sort_values('abs_coef', ascending=False).head(10).index
    X_keywords = tfidf_raw[:, top_10_idx]
    top_10_names = tfidf_vec.get_feature_names_out()[top_10_idx].tolist()

    # Feature Groups
    MARKET_BASE = ['PE_Ratio', 'vol_pct_chg', 'pe_chg', 'ma10_ratio', 'volatility_5d']
    X_market = df_t[MARKET_BASE].values
    X_sent = df_t[['sent_weighted_mean'] + pca_cols].values
    X_hybrid = np.hstack((X_market, X_sent, X_keywords))

    # Evaluate
    auc_market = tune_and_evaluate(X_market, y, tscv)
    auc_sent = tune_and_evaluate(X_sent, y, tscv)
    auc_keywords = tune_and_evaluate(X_keywords, y, tscv)
    auc_hybrid = tune_and_evaluate(X_hybrid, y, tscv)

    all_ticker_summary.append({
        'Ticker': ticker, 'Hybrid_AUC': auc_hybrid, 'Market_only': auc_market, 
        'Sentiment_only': auc_sent, 'Keywords_only': auc_keywords, 'Samples': len(df_t),
        'Top_Keywords': ", ".join(top_10_names)
    })

# Report
res_df = pd.DataFrame(all_ticker_summary)
res_df.to_csv(f'{RESULT_DIR}/hybrid_results_v11.csv', index=False)
with open(f'{RESULT_DIR}/hybrid_report_v11.txt', 'w') as f:
    f.write("ST545 POC v11 Synergy Report - Media Weight + PCA 16 (No Gating)\n" + "="*80 + "\n")
    f.write(res_df[['Ticker', 'Hybrid_AUC', 'Market_only', 'Sentiment_only', 'Keywords_only', 'Samples']].to_string(index=False))
    f.write(f"\n\nAGGREGATED MEANS:\n")
    f.write(f"  Mean Hybrid AUC    : {res_df['Hybrid_AUC'].mean():.4f}\n")
    f.write(f"  Mean Market-only   : {res_df['Market_only'].mean():.4f}\n")
    f.write(f"  Mean Sentiment-only: {res_df['Sentiment_only'].mean():.4f}\n")
    f.write(f"  Mean Keywords-only : {res_df['Keywords_only'].mean():.4f}\n")

print(f"\n[+] v11 complete. Results saved in {RESULT_DIR}")
