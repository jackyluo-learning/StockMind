"""
ST545 POC v4 — Step 4: XGBoost + SHAP (Daily-Level)
=====================================================
Daily-level aggregation with 10 tickers (~2,460 rows).
Loads sentiment from step0 cache.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import shap

TICKERS = ['NVDA', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'LMT', 'NEM', 'AAPL', 'META', 'JPM']
CACHE_PATH = 'dataset/sentiment_cache.csv'

# ── 1. Load sentiment cache + market data ──
cache = pd.read_csv(CACHE_PATH)
cache['Date'] = pd.to_datetime(cache['Date'])
print(f"--- Loaded sentiment cache: {len(cache)} articles ---")

all_daily = []
for ticker in TICKERS:
    market = pd.read_csv(f"dataset/{ticker}_market.csv")
    market['Date'] = pd.to_datetime(market['Date'])
    market = market.sort_values('Date')

    market['Next_Close'] = market['Close'].shift(-1)
    market['Price_Label'] = (market['Next_Close'] > market['Close']).astype(int)
    market = market.dropna(subset=['Price_Label'])
    market['Price_Label'] = market['Price_Label'].astype(int)

    # Technical features
    market['volume_pct_chg'] = market['Volume'].pct_change()
    market['pe_chg'] = market['PE_Ratio'].diff()
    market['MA5'] = market['Close'].rolling(5).mean()
    market['MA10'] = market['Close'].rolling(10).mean()
    market['ma5_ratio'] = market['Close'] / market['MA5']
    market['ma10_ratio'] = market['Close'] / market['MA10']
    market['momentum_5d'] = market['Close'].pct_change(5)
    market['volatility_5d'] = market['Close'].pct_change().rolling(5).std()

    # Aggregate articles → daily sentiment features
    ticker_news = cache[cache['Ticker'] == ticker].copy()

    daily_sent = ticker_news.groupby('Date')['Sentiment_Score'].agg(
        sent_mean='mean', sent_std='std', sent_max='max', sent_min='min',
        news_count='count'
    ).reset_index()
    daily_sent['sent_std'] = daily_sent['sent_std'].fillna(0)

    # Per-publisher sentiment means
    publishers = ticker_news['Publisher'].unique()
    for pub in publishers:
        pub_col = f"pub_{pub}_sent"
        pub_daily = ticker_news[ticker_news['Publisher'] == pub].groupby('Date')['Sentiment_Score'].mean()
        pub_daily = pub_daily.reset_index().rename(columns={'Sentiment_Score': pub_col})
        daily_sent = pd.merge(daily_sent, pub_daily, on='Date', how='left')

    market_cols = ['Date', 'Ticker', 'PE_Ratio', 'Volume', 'Price_Label',
                   'volume_pct_chg', 'pe_chg', 'ma5_ratio', 'ma10_ratio',
                   'momentum_5d', 'volatility_5d']
    daily = pd.merge(market[market_cols], daily_sent, on='Date', how='inner')
    all_daily.append(daily)

df = pd.concat(all_daily, ignore_index=True)
df = df.dropna(subset=['Price_Label']).sort_values('Date').reset_index(drop=True)

pub_sent_cols = [c for c in df.columns if c.startswith('pub_') and c.endswith('_sent')]
df[pub_sent_cols] = df[pub_sent_cols].fillna(0)
df = df.dropna().reset_index(drop=True)

print(f"--- Daily-level dataset: {len(df)} rows ({df['Ticker'].nunique()} tickers) ---")
print(f"--- Date range: {df['Date'].min().date()} to {df['Date'].max().date()} ---")

# ── 2. Feature matrix ──
MARKET_COLS = ['PE_Ratio', 'Volume', 'volume_pct_chg', 'pe_chg',
               'ma5_ratio', 'ma10_ratio', 'momentum_5d', 'volatility_5d']
SENT_AGG_COLS = ['sent_mean', 'sent_std', 'sent_max', 'sent_min', 'news_count']
FEATURE_COLS = SENT_AGG_COLS + pub_sent_cols + MARKET_COLS

scaler = StandardScaler()
X = scaler.fit_transform(df[FEATURE_COLS])
y = df['Price_Label'].values

print(f"\n--- Features ({len(FEATURE_COLS)}): ---")
print(f"  Sentiment aggregated: {SENT_AGG_COLS}")
print(f"  Publisher sentiments: {pub_sent_cols}")
print(f"  Market technical:     {MARKET_COLS}")
print(f"--- Samples: {len(X)}, Label balance: {y.mean():.2%} positive ---")

# ── 3. TimeSeriesSplit: LogReg vs XGBoost ──
print("\n--- LogReg vs XGBoost (daily-level, TimeSeriesSplit 5-fold) ---")
tscv = TimeSeriesSplit(n_splits=5)

logreg_results = {'acc': [], 'auc': [], 'f1': []}
xgb_results = {'acc': [], 'auc': [], 'f1': []}
majority_accs = []

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    maj = np.bincount(y_train).argmax()
    majority_accs.append(accuracy_score(y_test, np.full(len(y_test), maj)))

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_prob = lr.predict_proba(X_test)[:, 1]
    logreg_results['acc'].append(accuracy_score(y_test, lr_pred))
    logreg_results['auc'].append(roc_auc_score(y_test, lr_prob))
    logreg_results['f1'].append(f1_score(y_test, lr_pred))

    xgb_model = xgb.XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        random_state=42, eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train, verbose=False)
    xgb_pred = xgb_model.predict(X_test)
    xgb_prob = xgb_model.predict_proba(X_test)[:, 1]
    xgb_results['acc'].append(accuracy_score(y_test, xgb_pred))
    xgb_results['auc'].append(roc_auc_score(y_test, xgb_prob))
    xgb_results['f1'].append(f1_score(y_test, xgb_pred))

    print(f"  Fold {fold+1}: LogReg Acc={logreg_results['acc'][-1]:.4f} AUC={logreg_results['auc'][-1]:.4f} | "
          f"XGBoost Acc={xgb_results['acc'][-1]:.4f} AUC={xgb_results['auc'][-1]:.4f}")

lr_avg = {k: np.mean(v) for k, v in logreg_results.items()}
xgb_avg = {k: np.mean(v) for k, v in xgb_results.items()}
maj_avg = np.mean(majority_accs)

print(f"\n  LogReg  avg: Acc={lr_avg['acc']:.4f}, AUC={lr_avg['auc']:.4f}, F1={lr_avg['f1']:.4f}")
print(f"  XGBoost avg: Acc={xgb_avg['acc']:.4f}, AUC={xgb_avg['auc']:.4f}, F1={xgb_avg['f1']:.4f}")
print(f"  Majority baseline: {maj_avg:.4f}")

# ── 4. SHAP Analysis ──
print("\n--- SHAP analysis (XGBoost, full data)... ---")
final_model = xgb.XGBClassifier(
    n_estimators=100, max_depth=3, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
    random_state=42, eval_metric='logloss'
)
final_model.fit(X, y, verbose=False)

explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X)

plt.figure(figsize=(14, 10))
shap.summary_plot(shap_values, X, feature_names=FEATURE_COLS, show=False)
plt.title('SHAP Feature Importance (XGBoost, Daily-Level, 10 Tickers)')
plt.tight_layout()
plt.savefig('poc/result/shap_summary.png', dpi=150, bbox_inches='tight')
plt.close()

plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X, feature_names=FEATURE_COLS, plot_type='bar', show=False)
plt.title('Mean |SHAP| Feature Importance (Daily-Level, 10 Tickers)')
plt.tight_layout()
plt.savefig('poc/result/shap_bar.png', dpi=150, bbox_inches='tight')
plt.close()

mean_abs_shap = np.abs(shap_values).mean(axis=0)

top3_idx = np.argsort(mean_abs_shap)[-3:][::-1]
for rank, idx in enumerate(top3_idx):
    plt.figure(figsize=(8, 5))
    shap.dependence_plot(idx, shap_values, X, feature_names=FEATURE_COLS, show=False)
    plt.title(f'SHAP Dependence: {FEATURE_COLS[idx]}')
    plt.tight_layout()
    plt.savefig(f'poc/result/shap_dependence_top{rank+1}.png', dpi=150)
    plt.close()

# ── 5. Feature group importance ──
sent_features = SENT_AGG_COLS + pub_sent_cols
market_features = MARKET_COLS
sent_shap = sum(mean_abs_shap[FEATURE_COLS.index(f)] for f in sent_features)
market_shap = sum(mean_abs_shap[FEATURE_COLS.index(f)] for f in market_features)
total_shap = sent_shap + market_shap

print(f"\n--- Feature group importance (mean |SHAP|) ---")
print(f"  Sentiment features: {sent_shap:.4f} ({sent_shap/total_shap*100:.1f}%)")
print(f"  Market features:    {market_shap:.4f} ({market_shap/total_shap*100:.1f}%)")

# ── 6. Save Results ──
with open('poc/result/xgboost_shap_results.txt', 'w') as f:
    f.write("ST545 POC v4 Step 4 Results: LogReg vs XGBoost (Daily-Level)\n")
    f.write("=============================================================\n")
    f.write(f"Tickers: {TICKERS}\n")
    f.write(f"Dataset: {len(df)} daily-level samples, {df['Ticker'].nunique()} tickers\n")
    f.write(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}\n")
    f.write(f"Granularity: daily (articles aggregated per day per ticker)\n")
    f.write(f"Validation: TimeSeriesSplit (5 folds)\n\n")
    f.write(f"Features ({len(FEATURE_COLS)}):\n")
    f.write(f"  Sentiment aggregated: {SENT_AGG_COLS}\n")
    f.write(f"  Publisher sentiments: {pub_sent_cols}\n")
    f.write(f"  Market technical:     {MARKET_COLS}\n\n")
    f.write(f"{'Model':<20} {'Acc':>8} {'AUC':>8} {'F1':>8}\n")
    f.write("-" * 48 + "\n")
    f.write(f"{'LogReg':<20} {lr_avg['acc']:>8.4f} {lr_avg['auc']:>8.4f} {lr_avg['f1']:>8.4f}\n")
    f.write(f"{'XGBoost':<20} {xgb_avg['acc']:>8.4f} {xgb_avg['auc']:>8.4f} {xgb_avg['f1']:>8.4f}\n")
    f.write(f"{'Majority Vote':<20} {maj_avg:>8.4f} {'N/A':>8} {'N/A':>8}\n\n")
    f.write(f"Per-fold results:\n")
    for i in range(5):
        f.write(f"  Fold {i+1}: LogReg Acc={logreg_results['acc'][i]:.4f} AUC={logreg_results['auc'][i]:.4f} | "
                f"XGBoost Acc={xgb_results['acc'][i]:.4f} AUC={xgb_results['auc'][i]:.4f}\n")
    f.write(f"\nFeature group importance (mean |SHAP|):\n")
    f.write(f"  Sentiment features: {sent_shap:.4f} ({sent_shap/total_shap*100:.1f}%)\n")
    f.write(f"  Market features:    {market_shap:.4f} ({market_shap/total_shap*100:.1f}%)\n\n")
    f.write(f"Top SHAP features (mean |SHAP|):\n")
    sorted_idx = np.argsort(mean_abs_shap)[::-1]
    for idx in sorted_idx:
        f.write(f"  {FEATURE_COLS[idx]:30s}  {mean_abs_shap[idx]:.6f}\n")

print("\n[+] POC v4 Step 4 complete. Results in poc/result/")
