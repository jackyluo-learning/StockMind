"""
ST545 POC v9 (Fix) — Systematic Ablation Study: Focus LMT
=========================================================
1. Feature Groups: Sentiment-only vs. Market-only vs. Combined.
2. Model Classes: LogReg vs. XGBoost vs. Random Forest vs. MLP.
3. Logical consistency with Step 4 (Lags + PCA).
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

TICKER = 'LMT' # Focus on best ticker
CACHE_PATH = 'dataset/sentiment_cache.csv'
EMBED_CACHE_PATH = 'dataset/finbert_embeddings_768_v8.npy'

# 1. Load Data (Same logic as Step 4)
cache = pd.read_csv(CACHE_PATH)
cache['Date'] = pd.to_datetime(cache['Date'])
embeddings = np.load(EMBED_CACHE_PATH)
cache_embed_cols = [f'emb_{i}' for i in range(768)]
cache = pd.concat([cache.reset_index(drop=True), pd.DataFrame(embeddings, columns=cache_embed_cols)], axis=1)

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

ticker_news = cache[cache['Ticker'] == TICKER].copy()
daily_sent = ticker_news.groupby('Date').agg({
    'Sentiment_Score': ['mean', 'std', 'max'],
    **{c: 'mean' for c in cache_embed_cols}
}).reset_index()
daily_sent.columns = ['Date', 'sent_mean', 'sent_std', 'sent_max'] + cache_embed_cols
for lag in [1, 2, 3]: daily_sent[f'sent_mean_lag{lag}'] = daily_sent['sent_mean'].shift(lag)

df = pd.merge(market, daily_sent, on='Date', how='inner').dropna().reset_index(drop=True)

# PCA Reduction (Consistent with Step 4)
pca = PCA(n_components=16, random_state=42)
pca_feats = pca.fit_transform(df[cache_embed_cols].values)
MARKET_COLS = ['PE_Ratio', 'vol_pct_chg', 'pe_chg', 'ma10_ratio', 'volatility_5d']
SENT_BASE = ['sent_mean', 'sent_std', 'sent_max', 'sent_mean_lag1', 'sent_mean_lag2', 'sent_mean_lag3']
pca_cols = [f'pca_emb_{i}' for i in range(16)]

X_all = np.hstack((df[MARKET_COLS + SENT_BASE].values, pca_feats))
X_sent = np.hstack((df[SENT_BASE].values, pca_feats))
X_market = df[MARKET_COLS].values
y = df['Price_Label'].values

def evaluate(X, y, model_type='xgb'):
    tscv = TimeSeriesSplit(n_splits=3)
    sc = StandardScaler()
    aucs = []
    for tr, te in tscv.split(X):
        X_tr, X_te = sc.fit_transform(X[tr]), sc.transform(X[te])
        if model_type == 'lr': m = LogisticRegression()
        elif model_type == 'xgb': m = xgb.XGBClassifier(n_estimators=100, max_depth=3, random_state=42)
        elif model_type == 'rf': m = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        elif model_type == 'mlp': m = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)
        m.fit(X_tr, y[tr])
        aucs.append(roc_auc_score(y[te], m.predict_proba(X_te)[:, 1]))
    return np.mean(aucs)

# --- EXP 1: Model Classes (Combined Features) ---
results_m = {m: evaluate(X_all, y, m) for m in ['lr', 'xgb', 'rf', 'mlp']}

# --- EXP 2: Feature Groups (Best Model from EXP 1) ---
best_model_name = max(results_m, key=results_m.get)
res_f = {
    'Sentiment-only': evaluate(X_sent, y, best_model_name),
    'Market-only': evaluate(X_market, y, best_model_name),
    'Combined': evaluate(X_all, y, best_model_name)
}

# Save
with open('poc/result/ablation_results.txt', 'w') as f:
    f.write(f"ST545 POC v9 Ablation Report (Ticker: {TICKER})\n" + "="*50 + "\n\n")
    f.write(f"EXP 1: Model Comparison (Combined Features)\n")
    for k, v in results_m.items(): f.write(f"  {k:8s}: AUC={v:.4f}\n")
    
    f.write(f"\nEXP 2: Feature Groups (Best Model: {best_model_name})\n")
    for k, v in res_f.items(): f.write(f"  {k:15s}: AUC={v:.4f}\n")

print(f"\n[+] Ablation study complete for {TICKER}. Results in poc/result/ablation_results.txt")
