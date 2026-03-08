"""
ST545 POC v4 — Step 3: Media-Source Weighting (Lasso / L1)
==========================================================
Article-level: discovers per-publisher sentiment differences.
Loads sentiment from step0 cache. 10 tickers.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler

TICKERS = ['NVDA', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'LMT', 'NEM', 'AAPL', 'META', 'JPM']
CACHE_PATH = 'dataset/sentiment_cache.csv'

# ── 1. Load sentiment cache + market data ──
cache = pd.read_csv(CACHE_PATH)
cache['Date'] = pd.to_datetime(cache['Date'])
print(f"--- Loaded sentiment cache: {len(cache)} articles ---")

all_data = []
for ticker in TICKERS:
    market = pd.read_csv(f"dataset/{ticker}_market.csv")
    market['Date'] = pd.to_datetime(market['Date'])
    market = market.sort_values('Date')
    market['Next_Close'] = market['Close'].shift(-1)
    market['Price_Label'] = (market['Next_Close'] > market['Close']).astype(int)
    market = market.dropna(subset=['Price_Label'])
    market['Price_Label'] = market['Price_Label'].astype(int)

    ticker_news = cache[cache['Ticker'] == ticker].copy()
    merged = pd.merge(ticker_news, market[['Date', 'Ticker', 'Close', 'Volume', 'PE_Ratio', 'Price_Label']],
                       on=['Date', 'Ticker'], how='inner')
    all_data.append(merged)

df = pd.concat(all_data, ignore_index=True).sort_values('Date').reset_index(drop=True)
print(f"--- Merged: {len(df)} articles from {df['Publisher'].nunique()} publishers, {df['Ticker'].nunique()} tickers ---")
print(f"--- Publisher counts:\n{df['Publisher'].value_counts().to_string()}\n")

# ── 2. Publisher One-Hot ──
publisher_dummies = pd.get_dummies(df['Publisher'], prefix='pub')
publisher_names = publisher_dummies.columns.tolist()
X_publisher = publisher_dummies.values
print(f"--- Publishers ({len(publisher_names)}): {publisher_names} ---")

# ── 3. Interaction: Publisher × Sentiment_Score ──
sentiment = df['Sentiment_Score'].values.reshape(-1, 1)
X_interaction = X_publisher * sentiment
interaction_names = [f"{p}×Sentiment" for p in publisher_names]

# ── 4. Combine features ──
scaler = StandardScaler()
X_numeric = scaler.fit_transform(df[['PE_Ratio', 'Volume']])
X = np.hstack((X_publisher, sentiment, X_interaction, X_numeric))
y = df['Price_Label'].values
feature_names = publisher_names + ['Sentiment_Score'] + interaction_names + ['PE_Ratio', 'Volume']
print(f"--- Total features: {X.shape[1]} ---")

# ── 5. Lasso (L1) with TimeSeriesSplit ──
print("\n--- Training LassoCV with TimeSeriesSplit... ---")
tscv = TimeSeriesSplit(n_splits=5)
train_idx, test_idx = list(tscv.split(X))[-1]
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

lasso_model = LogisticRegressionCV(
    penalty='l1', solver='saga', cv=3, Cs=10,
    max_iter=5000, tol=1e-3, random_state=42
)
lasso_model.fit(X_train, y_train)

y_pred = lasso_model.predict(X_test)
y_prob = lasso_model.predict_proba(X_test)[:, 1]
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
f1 = f1_score(y_test, y_pred)

majority_class = np.bincount(y_train).argmax()
majority_acc = accuracy_score(y_test, np.full(len(y_test), majority_class))

print(f"  Lasso Accuracy : {acc:.4f}")
print(f"  Lasso ROC-AUC  : {auc:.4f}")
print(f"  Lasso F1       : {f1:.4f}")
print(f"  Majority Acc   : {majority_acc:.4f}")

# ── 6. Coefficient Analysis ──
coefs = lasso_model.coef_[0]
n_nonzero = (coefs != 0).sum()
print(f"\n--- Sparsity: {n_nonzero}/{len(coefs)} features survived L1 ({n_nonzero/len(coefs)*100:.1f}%) ---")

coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefs})
coef_df['AbsCoef'] = coef_df['Coefficient'].abs()
coef_df = coef_df[coef_df['AbsCoef'] > 0].sort_values('AbsCoef', ascending=False)

top_n = min(30, len(coef_df))
top_features = coef_df.head(top_n)

plt.figure(figsize=(12, 10))
colors = ['green' if c > 0 else 'red' for c in top_features['Coefficient']]
plt.barh(range(top_n), top_features['Coefficient'].values, color=colors)
plt.yticks(range(top_n), top_features['Feature'].values, fontsize=8)
plt.xlabel('Lasso Coefficient')
plt.title(f'Top {top_n} Non-Zero Lasso Coefficients\n(10 Tickers, Publisher × Sentiment)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('poc/result/lasso_coefficients.png', dpi=150)
plt.close()

# ── 7. Publisher Importance ──
pub_importance = {}
for pname in publisher_names:
    mask = coef_df['Feature'].str.startswith(pname + '×')
    pub_importance[pname] = coef_df.loc[mask, 'AbsCoef'].mean() if mask.sum() > 0 else 0.0

pub_imp_df = pd.DataFrame.from_dict(pub_importance, orient='index', columns=['Mean_AbsCoef'])
pub_imp_df = pub_imp_df.sort_values('Mean_AbsCoef', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=pub_imp_df['Mean_AbsCoef'], y=pub_imp_df.index, palette='coolwarm')
plt.title('Publisher Importance (10 Tickers)')
plt.xlabel('Mean |Coefficient|')
plt.tight_layout()
plt.savefig('poc/result/publisher_importance.png', dpi=150)
plt.close()

# ── 8. Sentiment by Publisher ──
plt.figure(figsize=(12, 6))
pub_order = df['Publisher'].value_counts().index.tolist()
sns.boxplot(data=df, x='Publisher', y='Sentiment_Score', order=pub_order)
plt.xticks(rotation=45, ha='right')
plt.title('FinBERT Sentiment Distribution by Publisher (10 Tickers)')
plt.tight_layout()
plt.savefig('poc/result/sentiment_by_publisher.png', dpi=150)
plt.close()

# ── 9. Save Results ──
with open('poc/result/media_weighting_results.txt', 'w') as f:
    f.write("ST545 POC v4 Step 3 Results: Media-Source Weighting\n")
    f.write("====================================================\n")
    f.write(f"Tickers: {TICKERS}\n")
    f.write(f"Publishers: {df['Publisher'].nunique()} unique (benzinga normalized)\n")
    f.write(f"Articles: {len(df)}\n")
    f.write(f"Interaction features: {len(interaction_names)} (Publisher × Sentiment)\n")
    f.write(f"Total features: {X.shape[1]}\n")
    f.write(f"Validation: TimeSeriesSplit (last fold)\n\n")
    f.write(f"Lasso Accuracy : {acc:.4f}\n")
    f.write(f"Lasso ROC-AUC  : {auc:.4f}\n")
    f.write(f"Lasso F1       : {f1:.4f}\n")
    f.write(f"Majority Acc   : {majority_acc:.4f}\n\n")
    f.write(f"Sparsity: {n_nonzero}/{len(coefs)} features survived L1\n\n")
    f.write(f"Top 20 Features by |Coefficient|:\n")
    for _, row in top_features.head(20).iterrows():
        f.write(f"  {row['Feature']:50s}  {row['Coefficient']:+.6f}\n")
    f.write(f"\nPublisher Importance:\n")
    for pub, row in pub_imp_df.iterrows():
        f.write(f"  {pub:30s}  {row['Mean_AbsCoef']:.6f}\n")

print("\n[+] POC v4 Step 3 complete. Results in poc/result/")
