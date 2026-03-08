"""
ST545 POC v7 — Ablation Study
===============================
Three experiments:
  Q1: RandomForest vs XGBoost vs LogReg (Direct, no Calibration)
  Q2: Tech-only (7 tickers) vs All (10 tickers)
  Q3: Sentiment-only vs Market-only vs Combined (incl Sector Driver)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

ALL_TICKERS = ['NVDA', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'LMT', 'NEM', 'AAPL', 'META', 'JPM']
GROWTH_TICKERS = ['NVDA', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'AAPL', 'META']
TECH_TICKERS = GROWTH_TICKERS
CACHE_PATH = 'dataset/sentiment_cache.csv'

MARKET_COLS_BASE = ['PE_Ratio', 'Volume', 'volume_pct_chg', 'pe_chg',
                    'ma5_ratio', 'ma10_ratio', 'momentum_5d', 'volatility_5d']

cache = pd.read_csv(CACHE_PATH)
cache['Date'] = pd.to_datetime(cache['Date'])

# ── Build daily dataset for arbitrary ticker list (v7: Sector Driver) ──
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

        # 🚀 v7 Sector Driver
        market['Sector_Driver'] = 1 if ticker in GROWTH_TICKERS else 0

        ticker_news = cache[cache['Ticker'] == ticker].copy()
        daily_sent = ticker_news.groupby('Date')['Sentiment_Score'].agg(
            sent_mean='mean', sent_std='std', sent_max='max', sent_min='min',
            news_count='count'
        ).reset_index()
        daily_sent['sent_std'] = daily_sent['sent_std'].fillna(0)

        # v6-v7: Add 1-3 day lags
        for lag in [1, 2, 3]:
            daily_sent[f'sent_mean_lag{lag}'] = daily_sent['sent_mean'].shift(lag)
            daily_sent[f'news_count_lag{lag}'] = daily_sent['news_count'].shift(lag)

        for pub in ticker_news['Publisher'].unique():
            pub_col = f"pub_{pub}_sent"
            pub_daily = ticker_news[ticker_news['Publisher'] == pub].groupby('Date')['Sentiment_Score'].mean()
            pub_daily = pub_daily.reset_index().rename(columns={'Sentiment_Score': pub_col})
            daily_sent = pd.merge(daily_sent, pub_daily, on='Date', how='left')

        market_cols = ['Date', 'Ticker', 'PE_Ratio', 'Volume', 'Price_Label', 'Sector_Driver',
                       'volume_pct_chg', 'pe_chg', 'ma5_ratio', 'ma10_ratio',
                       'momentum_5d', 'volatility_5d']
        daily = pd.merge(market[market_cols], daily_sent, on='Date', how='inner')
        all_daily.append(daily)

    df = pd.concat(all_daily, ignore_index=True)
    df = df.dropna(subset=['Price_Label']).sort_values('Date').reset_index(drop=True)
    pub_sent_cols = [c for c in df.columns if c.startswith('pub_') and c.endswith('_sent')]
    df[pub_sent_cols] = df[pub_sent_cols].fillna(0)
    
    # Fill lag NaNs
    lag_cols = [c for c in df.columns if '_lag' in c]
    df[lag_cols] = df[lag_cols].fillna(0)
    
    df = df.dropna().reset_index(drop=True)
    return df, pub_sent_cols, lag_cols

# ── Evaluate model (v7: Direct) ──
def evaluate(X, y, model_type='xgb', n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scaler = StandardScaler()
    accs, aucs, maj_accs = [], [], []
    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)
        
        if model_type == 'lr':
            model = LogisticRegression(max_iter=1000, random_state=42)
        elif model_type == 'xgb':
            model = xgb.XGBClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                random_state=42, eval_metric='logloss')
        elif model_type == 'rf100':
            model = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=10, random_state=42)
        elif model_type == 'rf200':
            model = RandomForestClassifier(n_estimators=200, max_depth=4, min_samples_leaf=15, random_state=42)
        elif model_type == 'rf500':
            model = RandomForestClassifier(n_estimators=500, max_depth=3, min_samples_leaf=20, random_state=42)
        else:
            model = LogisticRegression(max_iter=1000, random_state=42)

        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        prob = model.predict_proba(X_te)[:, 1]
        accs.append(accuracy_score(y_te, pred))
        aucs.append(roc_auc_score(y_te, prob))
        maj = np.bincount(y_tr).argmax()
        maj_accs.append(accuracy_score(y_te, np.full(len(y_te), maj)))
    return np.mean(accs), np.mean(aucs), np.mean(maj_accs), accs, aucs

SENT_AGG = ['sent_mean', 'sent_std', 'sent_max', 'sent_min', 'news_count']
MARKET_COLS_V7 = MARKET_COLS_BASE + ['Sector_Driver']

# ══════════════════════════════════════════════════════
#  Q1: RandomForest vs XGBoost vs LogReg
# ══════════════════════════════════════════════════════
print("=" * 70)
print("Q1: RandomForest vs XGBoost vs LogReg (10 tickers, Sector Driver, Direct)")
print("=" * 70)

df_all, pub_cols_all, lags_all = build_daily(ALL_TICKERS)
FEAT_ALL = SENT_AGG + lags_all + pub_cols_all + MARKET_COLS_V7
X_all = df_all[FEAT_ALL].values
y_all = df_all['Price_Label'].values

models_q1 = {
    'LogReg': 'lr',
    'XGBoost': 'xgb',
    'RandomForest-100': 'rf100',
    'RandomForest-200': 'rf200',
    'RandomForest-500': 'rf500',
}

print(f"Dataset: {len(df_all)} rows, {len(FEAT_ALL)} features\n")
for name, mtype in models_q1.items():
    acc, auc, maj, accs, aucs = evaluate(X_all, y_all, mtype)
    print(f"  {name:20s}  Acc={acc:.4f}  AUC={auc:.4f}  Majority={maj:.4f}  "
          f"per-fold AUC=[{', '.join(f'{a:.3f}' for a in aucs)}]")

# ══════════════════════════════════════════════════════
#  Q2: Tech-only (7) vs All (10)
# ══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Q2: Tech-only (7 tickers) vs All (10 tickers) - Direct")
print("=" * 70)

df_tech, pub_cols_tech, lags_tech = build_daily(TECH_TICKERS)
FEAT_TECH = SENT_AGG + lags_tech + pub_cols_tech + MARKET_COLS_V7
X_tech = df_tech[FEAT_TECH].values
y_tech = df_tech['Price_Label'].values

print(f"Tech-only: {len(df_tech)} rows, {len(FEAT_TECH)} features")
print(f"All:       {len(df_all)} rows, {len(FEAT_ALL)} features\n")

for name, mtype in [('LogReg', 'lr'), ('XGBoost', 'xgb'), ('RandomForest', 'rf200')]:
    acc_t, auc_t, maj_t, _, _ = evaluate(X_tech, y_tech, mtype)
    acc_a, auc_a, maj_a, _, _ = evaluate(X_all, y_all, mtype)
    print(f"  {name:15s}  Tech: Acc={acc_t:.4f} AUC={auc_t:.4f} Maj={maj_t:.4f}  |  "
          f"All: Acc={acc_a:.4f} AUC={auc_a:.4f} Maj={maj_a:.4f}")

# ══════════════════════════════════════════════════════
#  Q3: Sentiment-only vs Market-only vs Combined
# ══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Q3: Sentiment-only vs Market-only vs Combined (v7 Features)")
print("=" * 70)

feature_sets = {
    'Sentiment-only': SENT_AGG + lags_all + pub_cols_all,
    'Market-only': MARKET_COLS_V7,
    'Combined': FEAT_ALL,
}

output_summary = []

for feat_name, feat_cols in feature_sets.items():
    X_sub = df_all[feat_cols].values
    print(f"\n  [{feat_name}] ({len(feat_cols)} features)")
    output_summary.append(f"\n[{feat_name}]")
    for model_name, mtype in [('LogReg', 'lr'), ('XGBoost', 'xgb'), ('RandomForest', 'rf200')]:
        acc, auc, maj, _, _ = evaluate(X_sub, y_all, mtype)
        print(f"    {model_name:15s}  Acc={acc:.4f}  AUC={auc:.4f}  Majority={maj:.4f}")
        output_summary.append(f"  {model_name:15s}  Acc={acc:.4f}  AUC={auc:.4f}  Majority={maj:.4f}")

with open('poc/result/ablation_results.txt', 'w') as f:
    f.write("ST545 POC v7 Ablation Results\n" + "=" * 40 + "\n" + "".join(output_summary))

print("\n" + "=" * 70 + "\nDone. All ablation experiments complete.\n" + "=" * 70)
