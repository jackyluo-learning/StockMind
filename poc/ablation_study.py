"""
ST545 POC v10 (Ablation) — Systematic Ablation Study: Model Comparison Focus
========================================================================
1. Model Comparison: Tuned RF, MLP vs. the pre-calculated v10 XGBoost (Step 5).
2. Target Ticker: LMT
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
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

# ── 1. Configuration ──
TICKER = 'LMT'
CACHE_PATH = 'dataset/sentiment_cache.csv'
EMBED_CACHE_PATH = 'dataset/finbert_embeddings_768_v8.npy'
STEP5_V10_PATH = 'poc/result/step5/hybrid_results_v10.csv'
RESULT_PATH = 'poc/result/ablation/ablation_results_v10.txt'

# Text Prep
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))
def preprocess_text(text):
    if not isinstance(text, str): return ""
    text = re.sub('[^a-zA-Z]', ' ', text).lower().split()
    return ' '.join([ps.stem(word) for word in text if word not in stop_words])

# ── 2. Load and Prepare Data (v10 Logic) ──
print(f"--- Preparing Ablation Data for {TICKER} (v10 Config) ---")
cache_df = pd.read_csv(CACHE_PATH)
embeddings = np.load(EMBED_CACHE_PATH)

# Global PCA reduction (8-dim) for consistency with v10
pca_8 = PCA(n_components=8, random_state=42)
pca_data = pca_8.fit_transform(embeddings)
pca_cols = [f'pca_{i}' for i in range(8)]
cache_df = pd.concat([cache_df.reset_index(drop=True), pd.DataFrame(pca_data, columns=pca_cols)], axis=1)
cache_df['Date'] = pd.to_datetime(cache_df['Date'])

market = pd.read_csv(f"dataset/{TICKER}_market.csv")
market['Date'] = pd.to_datetime(market['Date'])
market = market.sort_values('Date')
market['Next_Close'] = market['Close'].shift(-1)
market['Price_Label'] = (market['Next_Close'] > market['Close']).astype(int)
market = market.dropna(subset=['Price_Label'])
market['vol_pct_chg'] = market['Volume'].pct_change()
market['pe_chg'] = market['PE_Ratio'].diff()
market['ma10_ratio'] = market['Close'] / market['Close'].rolling(10).mean()
market['volatility_5d'] = market['Close'].pct_change().rolling(5).std()

ticker_news = cache_df[cache_df['Ticker'] == TICKER].copy()
ticker_news['Full_Text'] = ticker_news['Headline'].fillna('') + " " + ticker_news['Summary'].fillna('')

# Daily Aggregation (Simple Mean per v10)
agg_logic = {
    'Full_Text': lambda x: " ".join(x),
    'Sentiment_Score': 'mean',
    **{c: 'mean' for c in pca_cols}
}
daily_agg = ticker_news.groupby('Date').agg(agg_logic).reset_index()

df_t = pd.merge(market, daily_agg, on='Date', how='inner').dropna().reset_index(drop=True)

# ── 3. Feature Set Definitions ──
y = df_t['Price_Label'].values
tscv = TimeSeriesSplit(n_splits=3)
sc = StandardScaler()

# Full Combined Set
MARKET_BASE = ['PE_Ratio', 'vol_pct_chg', 'pe_chg', 'ma10_ratio', 'volatility_5d']
SENT_COLS = ['Sentiment_Score'] + pca_cols
texts_processed = [preprocess_text(t) for t in df_t['Full_Text']]
tfidf_vec = TfidfVectorizer(max_features=1000)
tfidf_raw = tfidf_vec.fit_transform(texts_processed).toarray()
lasso_model = LogisticRegressionCV(penalty='l1', solver='saga', cv=tscv, max_iter=5000, random_state=42).fit(sc.fit_transform(tfidf_raw), y)
coef_df = pd.DataFrame({'word': tfidf_vec.get_feature_names_out(), 'coef_abs': np.abs(lasso_model.coef_[0])})
top_10_idx = coef_df.sort_values('coef_abs', ascending=False).head(10).index
X_keywords = tfidf_raw[:, top_10_idx]

X_market = df_t[MARKET_BASE].values
X_sent = df_t[SENT_COLS].values
X_combined = np.hstack((X_market, X_sent, X_keywords))

# ── 4. Training and Comparisons ──
def tune_and_eval(X_in, model_type='rf'):
    X_scaled = sc.fit_transform(X_in)
    if model_type == 'rf':
        m = RandomForestClassifier(random_state=42)
        params = {'n_estimators': [50, 100], 'max_depth': [5, 10]}
    elif model_type == 'mlp':
        m = MLPClassifier(max_iter=500, random_state=42)
        params = {'hidden_layer_sizes': [(32,), (32, 16)], 'alpha': [0.001, 0.01]}
    
    grid = GridSearchCV(m, params, cv=tscv, scoring='roc_auc', n_jobs=-1)
    grid.fit(X_scaled, y)
    return grid.best_score_

# Retrieve the Pre-Calculated Combined AUC for LMT from Step 5
xgb_v10_auc = 0.5
if os.path.exists(STEP5_V10_PATH):
    res = pd.read_csv(STEP5_V10_PATH)
    lmt_row = res[res['Ticker'] == TICKER]
    if not lmt_row.empty:
        xgb_v10_auc = float(lmt_row['Hybrid_AUC'].values[0])
        print(f"--- Retrieved v10 XGBoost AUC for {TICKER}: {xgb_v10_auc:.4f} ---")

print("\n>>> Comparing Model Classes (Full Combined Set)...")
results_m = {
    'XGBoost (Step 5 v10)': xgb_v10_auc,
    'Random Forest': tune_and_eval(X_combined, 'rf'),
    'MLP (Deep Learning)': tune_and_eval(X_combined, 'mlp')
}

# ── 5. Save Results ──
with open(RESULT_PATH, 'w') as f:
    f.write(f"ST545 POC v10 Ablation Study Report (Ticker: {TICKER})\n" + "="*65 + "\n\n")
    f.write("EXP 1: Model Class Comparison (Full v10 Feature Set)\n")
    f.write("-" * 55 + "\n")
    for k, v in results_m.items(): f.write(f"  {k:25s}: AUC={v:.4f}\n")
    
    f.write("\nCONCLUSION:\n")
    best_m = max(results_m, key=results_m.get)
    f.write(f"  Best Model for {TICKER}: {best_m} (AUC={results_m[best_m]:.4f})\n")

print(f"\n[+] Ablation study complete. Results in {RESULT_PATH}")
