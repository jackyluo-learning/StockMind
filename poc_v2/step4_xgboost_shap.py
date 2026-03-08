"""
ST545 POC v2 — Step 4: XGBoost + SHAP (Article-Level)
======================================================
Fix: article-level training (17K rows) instead of daily-aggregated (708 rows)
so XGBoost and LogReg are compared at the same granularity.

Each article gets: FinBERT sentiment + market features for that day.
Both LogReg and XGBoost trained on identical features for fair comparison.
"""

import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import shap

# ── 1. Load Market + News ──
TICKERS = ['NVDA', 'GOOGL', 'MSFT']
all_data = []

for ticker in TICKERS:
    market = pd.read_csv(f"dataset/{ticker}_market.csv")
    news = pd.read_csv(f"dataset/{ticker}_news.csv")
    market['Date'] = pd.to_datetime(market['Date'])
    news['Date'] = pd.to_datetime(news['Date'])

    market = market.sort_values('Date')
    market['Next_Close'] = market['Close'].shift(-1)
    market['Price_Label'] = (market['Next_Close'] > market['Close']).astype(int)
    market = market.dropna(subset=['Price_Label'])
    market['Price_Label'] = market['Price_Label'].astype(int)

    # Technical features on market data
    market['volume_pct_chg'] = market['Volume'].pct_change()
    market['pe_chg'] = market['PE_Ratio'].diff()
    market['MA5'] = market['Close'].rolling(5).mean()
    market['MA10'] = market['Close'].rolling(10).mean()
    market['ma5_ratio'] = market['Close'] / market['MA5']
    market['ma10_ratio'] = market['Close'] / market['MA10']
    market['momentum_5d'] = market['Close'].pct_change(5)
    market['volatility_5d'] = market['Close'].pct_change().rolling(5).std()

    news['Summary'] = news['Summary'].fillna('')
    news['Text'] = news['Headline'].str.strip() + '. ' + news['Summary'].str.strip()

    # Merge: each article gets its day's market features + label
    market_cols = ['Date', 'Ticker', 'PE_Ratio', 'Volume', 'Price_Label',
                   'volume_pct_chg', 'pe_chg', 'ma5_ratio', 'ma10_ratio',
                   'momentum_5d', 'volatility_5d']
    merged = pd.merge(news, market[market_cols], on=['Date', 'Ticker'], how='inner')
    all_data.append(merged)

df = pd.concat(all_data, ignore_index=True)
df = df.dropna().sort_values('Date').reset_index(drop=True)
print(f"--- Loaded {len(df)} article-level rows ---")

# ── 2. FinBERT Sentiment ──
print("--- Loading FinBERT... ---")
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
model.eval()

def get_finbert_sentiment(texts, batch_size=64):
    scores = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors='pt', padding=True,
                          truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).numpy()
        batch_scores = probs[:, 0] - probs[:, 1]
        scores.extend(batch_scores)
        if (i // batch_size) % 20 == 0:
            print(f"  FinBERT: {i+len(batch)}/{len(texts)}")
    return np.array(scores)

print("--- Running FinBERT on all articles... ---")
df['Sentiment_Score'] = get_finbert_sentiment(df['Text'].tolist())
print(f"--- FinBERT done. Mean sentiment: {df['Sentiment_Score'].mean():.4f} ---")

# ── 3. Feature Matrix ──
FEATURE_COLS = [
    'Sentiment_Score', 'PE_Ratio', 'Volume',
    'volume_pct_chg', 'pe_chg', 'ma5_ratio', 'ma10_ratio',
    'momentum_5d', 'volatility_5d'
]

scaler = StandardScaler()
X = scaler.fit_transform(df[FEATURE_COLS])
y = df['Price_Label'].values
print(f"\n--- Features ({len(FEATURE_COLS)}): {FEATURE_COLS} ---")
print(f"--- Samples: {len(X)}, Label balance: {y.mean():.2%} positive ---")

# ── 4. TimeSeriesSplit: LogReg vs XGBoost Head-to-Head ──
print("\n--- LogReg vs XGBoost (same features, TimeSeriesSplit 5-fold) ---")
tscv = TimeSeriesSplit(n_splits=5)

logreg_results = {'acc': [], 'auc': [], 'f1': []}
xgb_results = {'acc': [], 'auc': [], 'f1': []}
majority_accs = []

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Majority baseline
    maj = np.bincount(y_train).argmax()
    majority_accs.append(accuracy_score(y_test, np.full(len(y_test), maj)))

    # LogReg
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_prob = lr.predict_proba(X_test)[:, 1]
    logreg_results['acc'].append(accuracy_score(y_test, lr_pred))
    logreg_results['auc'].append(roc_auc_score(y_test, lr_prob))
    logreg_results['f1'].append(f1_score(y_test, lr_pred))

    # XGBoost
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

# ── 5. SHAP Analysis (XGBoost on full data) ──
print("\n--- SHAP analysis (XGBoost, full data)... ---")
final_model = xgb.XGBClassifier(
    n_estimators=100, max_depth=3, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
    random_state=42, eval_metric='logloss'
)
final_model.fit(X, y, verbose=False)

explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X)

# SHAP summary plot
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X, feature_names=FEATURE_COLS, show=False)
plt.title('SHAP Feature Importance (XGBoost, All Tickers)')
plt.tight_layout()
plt.savefig('poc/result/shap_summary.png', dpi=150, bbox_inches='tight')
plt.close()

# SHAP bar plot (mean |SHAP|)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X, feature_names=FEATURE_COLS, plot_type='bar', show=False)
plt.title('Mean |SHAP| Feature Importance')
plt.tight_layout()
plt.savefig('poc/result/shap_bar.png', dpi=150, bbox_inches='tight')
plt.close()

# Top dependence plots (top 3 features by mean |SHAP|)
mean_abs_shap = np.abs(shap_values).mean(axis=0)
top3_idx = np.argsort(mean_abs_shap)[-3:][::-1]
for rank, idx in enumerate(top3_idx):
    plt.figure(figsize=(8, 5))
    shap.dependence_plot(idx, shap_values, X, feature_names=FEATURE_COLS, show=False)
    plt.title(f'SHAP Dependence: {FEATURE_COLS[idx]}')
    plt.tight_layout()
    plt.savefig(f'poc/result/shap_dependence_top{rank+1}.png', dpi=150)
    plt.close()

# ── 7. Save Results ──
with open('poc/result/xgboost_shap_results.txt', 'w') as f:
    f.write("ST545 POC v2 Step 4 Results: LogReg vs XGBoost (Article-Level)\n")
    f.write("===============================================================\n")
    f.write(f"Dataset: {len(df)} article-level samples, {df['Ticker'].nunique()} tickers\n")
    f.write(f"Features ({len(FEATURE_COLS)}): {FEATURE_COLS}\n")
    f.write(f"Text Input: Headline + Summary → FinBERT sentiment score\n")
    f.write(f"Validation: TimeSeriesSplit (5 folds)\n\n")
    f.write(f"{'Model':<20} {'Acc':>8} {'AUC':>8} {'F1':>8}\n")
    f.write("-" * 48 + "\n")
    f.write(f"{'LogReg':<20} {lr_avg['acc']:>8.4f} {lr_avg['auc']:>8.4f} {lr_avg['f1']:>8.4f}\n")
    f.write(f"{'XGBoost':<20} {xgb_avg['acc']:>8.4f} {xgb_avg['auc']:>8.4f} {xgb_avg['f1']:>8.4f}\n")
    f.write(f"{'Majority Vote':<20} {maj_avg:>8.4f} {'N/A':>8} {'N/A':>8}\n\n")
    f.write(f"Per-fold results:\n")
    for i in range(5):
        f.write(f"  Fold {i+1}: LogReg Acc={logreg_results['acc'][i]:.4f} AUC={logreg_results['auc'][i]:.4f} | "
                f"XGBoost Acc={xgb_results['acc'][i]:.4f} AUC={xgb_results['auc'][i]:.4f}\n")
    f.write(f"\nTop SHAP features (mean |SHAP|):\n")
    sorted_idx = np.argsort(mean_abs_shap)[::-1]
    for idx in sorted_idx:
        f.write(f"  {FEATURE_COLS[idx]:25s}  {mean_abs_shap[idx]:.6f}\n")

print("\n[+] POC v2 Step 4 complete. Results in poc/result/")
