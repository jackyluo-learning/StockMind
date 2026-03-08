"""
ST545 POC v4 — Ablation Study
===============================
Three experiments:
  Q1: RandomForest vs XGBoost vs LogReg (same data)
  Q2: Tech-only (7 tickers) vs All (10 tickers)
  Q3: Sentiment-only vs Market-only vs Combined
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

ALL_TICKERS = ['NVDA', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'LMT', 'NEM', 'AAPL', 'META', 'JPM']
TECH_TICKERS = ['NVDA', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'AAPL', 'META']
CACHE_PATH = 'dataset/sentiment_cache.csv'

MARKET_COLS = ['PE_Ratio', 'Volume', 'volume_pct_chg', 'pe_chg',
               'ma5_ratio', 'ma10_ratio', 'momentum_5d', 'volatility_5d']

cache = pd.read_csv(CACHE_PATH)
cache['Date'] = pd.to_datetime(cache['Date'])

# ── Build daily dataset for arbitrary ticker list ──
def build_daily(tickers):
    all_daily = []
    for ticker in tickers:
        market = pd.read_csv(f"dataset/{ticker}_market.csv")
        market['Date'] = pd.to_datetime(market['Date'])
        market = market.sort_values('Date')
        market['Next_Close'] = market['Close'].shift(-1)
        market['Price_Label'] = (market['Next_Close'] > market['Close']).astype(int)
        market = market.dropna(subset=['Price_Label'])
        market['Price_Label'] = market['Price_Label'].astype(int)
        market['volume_pct_chg'] = market['Volume'].pct_change()
        market['pe_chg'] = market['PE_Ratio'].diff()
        market['MA5'] = market['Close'].rolling(5).mean()
        market['MA10'] = market['Close'].rolling(10).mean()
        market['ma5_ratio'] = market['Close'] / market['MA5']
        market['ma10_ratio'] = market['Close'] / market['MA10']
        market['momentum_5d'] = market['Close'].pct_change(5)
        market['volatility_5d'] = market['Close'].pct_change().rolling(5).std()

        ticker_news = cache[cache['Ticker'] == ticker].copy()
        daily_sent = ticker_news.groupby('Date')['Sentiment_Score'].agg(
            sent_mean='mean', sent_std='std', sent_max='max', sent_min='min',
            news_count='count'
        ).reset_index()
        daily_sent['sent_std'] = daily_sent['sent_std'].fillna(0)

        for pub in ticker_news['Publisher'].unique():
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
    return df, pub_sent_cols

# ── Evaluate model with TimeSeriesSplit ──
def evaluate(X, y, model_fn, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scaler = StandardScaler()
    accs, aucs, maj_accs = [], [], []
    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)
        model = model_fn()
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        prob = model.predict_proba(X_te)[:, 1]
        accs.append(accuracy_score(y_te, pred))
        aucs.append(roc_auc_score(y_te, prob))
        maj = np.bincount(y_tr).argmax()
        maj_accs.append(accuracy_score(y_te, np.full(len(y_te), maj)))
    return np.mean(accs), np.mean(aucs), np.mean(maj_accs), accs, aucs

SENT_AGG = ['sent_mean', 'sent_std', 'sent_max', 'sent_min', 'news_count']

# ══════════════════════════════════════════════════════
#  Q1: RandomForest vs XGBoost vs LogReg (10 tickers)
# ══════════════════════════════════════════════════════
print("=" * 70)
print("Q1: RandomForest vs XGBoost vs LogReg (10 tickers, all features)")
print("=" * 70)

df_all, pub_cols_all = build_daily(ALL_TICKERS)
FEAT_ALL = SENT_AGG + pub_cols_all + MARKET_COLS
X_all = df_all[FEAT_ALL].values
y_all = df_all['Price_Label'].values

models_q1 = {
    'LogReg': lambda: LogisticRegression(max_iter=1000, random_state=42),
    'XGBoost': lambda: xgb.XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        random_state=42, eval_metric='logloss'),
    'RandomForest-100': lambda: RandomForestClassifier(
        n_estimators=100, max_depth=5, min_samples_leaf=10,
        random_state=42),
    'RandomForest-200': lambda: RandomForestClassifier(
        n_estimators=200, max_depth=4, min_samples_leaf=15,
        random_state=42),
    'RandomForest-500': lambda: RandomForestClassifier(
        n_estimators=500, max_depth=3, min_samples_leaf=20,
        random_state=42),
}

print(f"Dataset: {len(df_all)} rows, {len(FEAT_ALL)} features\n")
q1_results = {}
for name, fn in models_q1.items():
    acc, auc, maj, accs, aucs = evaluate(X_all, y_all, fn)
    q1_results[name] = (acc, auc, maj)
    print(f"  {name:20s}  Acc={acc:.4f}  AUC={auc:.4f}  Majority={maj:.4f}  "
          f"per-fold AUC=[{', '.join(f'{a:.3f}' for a in aucs)}]")

# ══════════════════════════════════════════════════════
#  Q2: Tech-only (7) vs All (10)
# ══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Q2: Tech-only (7 tickers) vs All (10 tickers)")
print("=" * 70)

df_tech, pub_cols_tech = build_daily(TECH_TICKERS)
FEAT_TECH = SENT_AGG + pub_cols_tech + MARKET_COLS
X_tech = df_tech[FEAT_TECH].values
y_tech = df_tech['Price_Label'].values

print(f"Tech-only: {len(df_tech)} rows, {len(FEAT_TECH)} features")
print(f"All:       {len(df_all)} rows, {len(FEAT_ALL)} features\n")

for name, fn in [
    ('LogReg', lambda: LogisticRegression(max_iter=1000, random_state=42)),
    ('XGBoost', lambda: xgb.XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        random_state=42, eval_metric='logloss')),
    ('RandomForest', lambda: RandomForestClassifier(
        n_estimators=200, max_depth=4, min_samples_leaf=15,
        random_state=42)),
]:
    acc_t, auc_t, maj_t, _, _ = evaluate(X_tech, y_tech, fn)
    acc_a, auc_a, maj_a, _, _ = evaluate(X_all, y_all, fn)
    print(f"  {name:15s}  Tech: Acc={acc_t:.4f} AUC={auc_t:.4f} Maj={maj_t:.4f}  |  "
          f"All: Acc={acc_a:.4f} AUC={auc_a:.4f} Maj={maj_a:.4f}")

# ══════════════════════════════════════════════════════
#  Q3: Sentiment-only vs Market-only vs Combined
# ══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Q3: Sentiment-only vs Market-only vs Combined (10 tickers)")
print("=" * 70)

FEAT_SENT = SENT_AGG + pub_cols_all
FEAT_MARKET = MARKET_COLS

feature_sets = {
    'Sentiment-only': FEAT_SENT,
    'Market-only': FEAT_MARKET,
    'Combined': FEAT_ALL,
}

for feat_name, feat_cols in feature_sets.items():
    X_sub = df_all[feat_cols].values
    print(f"\n  [{feat_name}] ({len(feat_cols)} features: {feat_cols})")
    for model_name, fn in [
        ('LogReg', lambda: LogisticRegression(max_iter=1000, random_state=42)),
        ('XGBoost', lambda: xgb.XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            random_state=42, eval_metric='logloss')),
        ('RandomForest', lambda: RandomForestClassifier(
            n_estimators=200, max_depth=4, min_samples_leaf=15,
            random_state=42)),
    ]:
        acc, auc, maj, _, _ = evaluate(X_sub, y_all, fn)
        print(f"    {model_name:15s}  Acc={acc:.4f}  AUC={auc:.4f}  Majority={maj:.4f}")

# ── Save ──
print("\n" + "=" * 70)
print("Done. All ablation experiments complete.")
print("=" * 70)
