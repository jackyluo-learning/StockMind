"""
ST545 POC v2 — Step 3: Media-Source Weighting (Lasso / L1)
==========================================================
 - Uses new split dataset format (_market.csv + _news.csv)
 - Text = Headline + Summary → FinBERT sentiment score
 - 8+ real publishers, no grouping
 - Interaction: Publisher × Sentiment_Score (each publisher gets its own
   sentiment coefficient, so Lasso can decide which sources matter)
 - TimeSeriesSplit for evaluation
"""

import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── 1. Load Data ──
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

    news['Summary'] = news['Summary'].fillna('')
    news['Text'] = news['Headline'].str.strip() + '. ' + news['Summary'].str.strip()

    merged = pd.merge(news, market[['Date', 'Ticker', 'Close', 'Volume', 'PE_Ratio', 'Price_Label']],
                       on=['Date', 'Ticker'], how='inner')
    all_data.append(merged)

df = pd.concat(all_data, ignore_index=True).sort_values('Date').reset_index(drop=True)
print(f"--- Loaded {len(df)} articles from {df['Publisher'].nunique()} publishers ---")
print(f"--- Publisher counts:\n{df['Publisher'].value_counts().to_string()}\n")

# ── 2. FinBERT Sentiment Score ──
print("--- Loading FinBERT... ---")
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
finbert = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
finbert.eval()

def get_finbert_sentiment(texts, batch_size=64):
    scores = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors='pt', padding=True,
                          truncation=True, max_length=128)
        with torch.no_grad():
            outputs = finbert(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).numpy()
        batch_scores = probs[:, 0] - probs[:, 1]  # P(pos) - P(neg)
        scores.extend(batch_scores)
        if (i // batch_size) % 20 == 0:
            print(f"  FinBERT: {i+len(batch)}/{len(texts)}")
    return np.array(scores)

print("--- Running FinBERT on all articles... ---")
df['Sentiment_Score'] = get_finbert_sentiment(df['Text'].tolist())
print(f"--- FinBERT done. Mean sentiment: {df['Sentiment_Score'].mean():.4f} ---")

# ── 3. Publisher One-Hot (all publishers, no grouping) ──
publisher_dummies = pd.get_dummies(df['Publisher'], prefix='pub')
publisher_names = publisher_dummies.columns.tolist()
X_publisher = publisher_dummies.values
print(f"--- Publishers ({len(publisher_names)}): {publisher_names} ---")

# ── 4. Interaction: Publisher × Sentiment_Score ──
# Each publisher gets its own sentiment coefficient
sentiment = df['Sentiment_Score'].values.reshape(-1, 1)
X_interaction = X_publisher * sentiment  # broadcast: (N, n_pub) * (N, 1)
interaction_names = [f"{p}×Sentiment" for p in publisher_names]
print(f"--- Interaction features: {len(interaction_names)} (1 per publisher) ---")

# ── 5. Combine all features ──
scaler = StandardScaler()
X_numeric = scaler.fit_transform(df[['PE_Ratio', 'Volume']])
X_sentiment = sentiment  # standalone sentiment as well
X = np.hstack((X_publisher, X_sentiment, X_interaction, X_numeric))
y = df['Price_Label'].values

feature_names = publisher_names + ['Sentiment_Score'] + interaction_names + ['PE_Ratio', 'Volume']
print(f"--- Total features: {X.shape[1]} ---")

# ── 6. Lasso (L1) with TimeSeriesSplit ──
print("\n--- Training LassoCV with TimeSeriesSplit... ---")
tscv = TimeSeriesSplit(n_splits=5)

# Use the last fold for coefficient analysis (most data for training)
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

# Majority baseline
majority_class = np.bincount(y_train).argmax()
majority_acc = accuracy_score(y_test, np.full(len(y_test), majority_class))

print(f"  Lasso Accuracy : {acc:.4f}")
print(f"  Lasso ROC-AUC  : {auc:.4f}")
print(f"  Lasso F1       : {f1:.4f}")
print(f"  Majority Acc   : {majority_acc:.4f}")

# ── 7. Coefficient Analysis ──
coefs = lasso_model.coef_[0]
nonzero_mask = coefs != 0
n_nonzero = nonzero_mask.sum()
print(f"\n--- Sparsity: {n_nonzero}/{len(coefs)} features survived L1 ({n_nonzero/len(coefs)*100:.1f}%) ---")

coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefs})
coef_df['AbsCoef'] = coef_df['Coefficient'].abs()
coef_df = coef_df[coef_df['AbsCoef'] > 0].sort_values('AbsCoef', ascending=False)

# Top features
top_n = min(30, len(coef_df))
top_features = coef_df.head(top_n)

plt.figure(figsize=(12, 10))
colors = ['green' if c > 0 else 'red' for c in top_features['Coefficient']]
plt.barh(range(top_n), top_features['Coefficient'].values, color=colors)
plt.yticks(range(top_n), top_features['Feature'].values, fontsize=8)
plt.xlabel('Lasso Coefficient')
plt.title(f'Top {top_n} Non-Zero Lasso Coefficients\n(Publisher × Sentiment Interactions)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('poc/result/lasso_coefficients.png', dpi=150)
plt.close()

# ── 8. Publisher-Level Summary ──
# Average absolute coefficient per publisher (from interaction terms)
pub_importance = {}
for pname in publisher_names:
    mask = coef_df['Feature'].str.startswith(pname + '×')
    if mask.sum() > 0:
        pub_importance[pname] = coef_df.loc[mask, 'AbsCoef'].mean()
    else:
        pub_importance[pname] = 0.0

pub_imp_df = pd.DataFrame.from_dict(pub_importance, orient='index', columns=['Mean_AbsCoef'])
pub_imp_df = pub_imp_df.sort_values('Mean_AbsCoef', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=pub_imp_df['Mean_AbsCoef'], y=pub_imp_df.index, palette='coolwarm')
plt.title('Publisher Importance (Mean |Coefficient| of Interaction Terms)')
plt.xlabel('Mean |Coefficient|')
plt.tight_layout()
plt.savefig('poc/result/publisher_importance.png', dpi=150)
plt.close()

# ── 9. Save Results ──
with open('poc/result/media_weighting_results.txt', 'w') as f:
    f.write("ST545 POC v2 Step 3 Results: Media-Source Weighting\n")
    f.write("====================================================\n")
    f.write(f"Publishers: {df['Publisher'].nunique()} unique\n")
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
    f.write(f"\nPublisher Importance (Mean |Coef| of interactions):\n")
    for pub, row in pub_imp_df.iterrows():
        f.write(f"  {pub:30s}  {row['Mean_AbsCoef']:.6f}\n")

print("\n[+] POC v2 Step 3 complete. Results in poc/result/")
