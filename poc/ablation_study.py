"""
ST545 POC v9 (Comprehensive) — Systematic Ablation Study: LMT Focus
====================================================================
1. Model Comparison: Tuned RF, MLP vs. the already-tuned XGBoost from Step 4.
2. Feature Ablation: Sentiment-only vs. Market-only vs. Combined.
3. All models tuned via GridSearchCV with TimeSeriesSplit.
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
import warnings
warnings.filterwarnings('ignore')

TICKER = 'LMT'
CACHE_PATH = 'dataset/sentiment_cache.csv'
EMBED_CACHE_PATH = 'dataset/finbert_embeddings_768_v8.npy'
STEP4_RESULT_PATH = 'poc/result/step4/step4_results.txt'

# 1. Load Data
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
df_pca = pd.DataFrame(pca_feats, columns=pca_cols)

y = df['Price_Label'].values
tscv = TimeSeriesSplit(n_splits=3)
sc = StandardScaler()

# 2. Extract Step 4 Tuned XGBoost result for LMT
xgb_auc = 0.5
if os.path.exists(STEP4_RESULT_PATH):
    try:
        import re
        with open(STEP4_RESULT_PATH, 'r') as f:
            content = f.read()
            # Match LMT line and extract the second numeric column (FinBERT_AUC)
            # Format: LMT 0.626873 ...
            match = re.search(r'LMT\s+([\d\.]+)', content)
            if match:
                xgb_auc = float(match.group(1))
                print(f"--- Retrieved Tuned XGBoost AUC for LMT from Step 4: {xgb_auc:.4f} ---")
    except Exception as e:
        print(f"--- Warning: Could not parse Step 4 results: {e}. Defaulting XGBoost to 0.5 ---")

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
    return float(grid.best_score_), grid.best_estimator_

# --- EXP 1: Model Comparison (Combined Features) ---
print("\n>>> Comparing Tuned Models (RF, MLP)...")
X_all = pd.concat([df[MARKET_COLS + SENT_BASE], df_pca], axis=1).values
rf_auc, best_rf = tune_and_eval(X_all, 'rf')
mlp_auc, best_mlp = tune_and_eval(X_all, 'mlp')

# Ensure all values are float for max()
results_m = {'XGBoost (Step 4)': float(xgb_auc), 'Random Forest': float(rf_auc), 'MLP': float(mlp_auc)}

# --- EXP 2: Feature Groups (Using Best Model from EXP 1) ---
best_model_name = max(results_m, key=results_m.get)
print(f"\n>>> Feature Ablation using {best_model_name}...")

# Define sets
X_sent = pd.concat([df[SENT_BASE], df_pca], axis=1).values
X_market = df[MARKET_COLS].values

def final_eval_fixed_model(X_in, model_name):
    # For RF and MLP, we use the tuning logic again for fairness on the subset
    if 'Random Forest' in model_name: auc, _ = tune_and_eval(X_in, 'rf')
    elif 'MLP' in model_name: auc, _ = tune_and_eval(X_in, 'mlp')
    else:
        # If XGBoost was best, we re-run its tuning logic on this feature subset
        import xgboost as xgb
        X_scaled = sc.fit_transform(X_in)
        grid = GridSearchCV(xgb.XGBClassifier(n_jobs=-1, random_state=42, eval_metric='logloss'), 
                            {'n_estimators': [50, 100], 'max_depth': [3, 5]}, 
                            cv=tscv, scoring='roc_auc', n_jobs=-1)
        grid.fit(X_scaled, y)
        auc = grid.best_score_
    return auc

res_f = {
    'Sentiment-only': final_eval_fixed_model(X_sent, best_model_name),
    'Market-only': final_eval_fixed_model(X_market, best_model_name),
    'Combined': results_m[best_model_name]
}

# Save
with open('poc/result/ablation/ablation_results.txt', 'w') as f:
    f.write(f"ST545 POC v9 Ablation Report (Ticker: {TICKER})\n" + "="*60 + "\n\n")
    f.write(f"EXP 1: Model Class Comparison (Combined Features)\n")
    f.write(f"--------------------------------------------------\n")
    for k, v in results_m.items(): f.write(f"  {k:20s}: AUC={v:.4f}\n")
    
    f.write(f"\nEXP 2: Feature Group Comparison (Best Model: {best_model_name})\n")
    f.write(f"--------------------------------------------------\n")
    for k, v in res_f.items(): f.write(f"  {k:20s}: AUC={v:.4f}\n")
    
    f.write(f"\nConclusion:\n")
    if res_f['Combined'] > res_f['Market-only'] and res_f['Combined'] > res_f['Sentiment-only']:
        f.write("  Synergy Confirmed: Combined features outperform individual groups.\n")
    else:
        f.write("  Weak Synergy: One feature group dominates or captures most of the signal.\n")

print(f"\n[+] Ablation study complete for {TICKER}. Results in poc/result/ablation/ablation_results.txt")
