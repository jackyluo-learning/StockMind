"""
ST545 POC v4 — Step 1 & 2: EDA + TF-IDF Baseline
===================================================
10 tickers: NVDA, GOOGL, MSFT, AMZN, TSLA, LMT, NEM, AAPL, META, JPM
Publisher normalization (benzinga → Benzinga)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

PUBLISHER_NORM = {'benzinga': 'Benzinga'}
TICKERS = ['NVDA', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'LMT', 'NEM', 'AAPL', 'META', 'JPM']

# ── 1. Load Data ──
all_daily = []
for ticker in TICKERS:
    market = pd.read_csv(f"dataset/{ticker}_market.csv")
    news = pd.read_csv(f"dataset/{ticker}_news.csv")
    market['Date'] = pd.to_datetime(market['Date'])
    news['Date'] = pd.to_datetime(news['Date'])
    news['Publisher'] = news['Publisher'].replace(PUBLISHER_NORM)

    market = market.sort_values('Date')
    market['Next_Close'] = market['Close'].shift(-1)
    market['Price_Label'] = (market['Next_Close'] > market['Close']).astype(int)
    market = market.dropna(subset=['Price_Label'])
    market['Price_Label'] = market['Price_Label'].astype(int)

    news['Summary'] = news['Summary'].fillna('')
    news['Text'] = news['Headline'].str.strip() + '. ' + news['Summary'].str.strip()
    news['Text'] = news['Text'].str.strip('. ')

    merged = pd.merge(news, market[['Date', 'Ticker', 'Close', 'Volume', 'PE_Ratio', 'Price_Label']],
                       on=['Date', 'Ticker'], how='inner')
    all_daily.append(merged)

df = pd.concat(all_daily, ignore_index=True).sort_values('Date').reset_index(drop=True)
print(f"--- Data Loaded: {len(df)} rows, {df['Date'].nunique()} unique dates, {df['Ticker'].nunique()} tickers ---")
print(f"--- Label Balance: {df['Price_Label'].value_counts(normalize=True).to_dict()} ---")

# ── 2. EDA: Publisher Distribution ──
plt.figure(figsize=(12, 6))
top_publishers = df['Publisher'].value_counts().head(15)
sns.barplot(x=top_publishers.values, y=top_publishers.index, palette='viridis')
plt.title('Top 15 News Publishers (10 Tickers, Normalized)')
plt.xlabel('Number of Articles')
plt.tight_layout()
plt.savefig('poc/result/publisher_distribution.png', dpi=150)
plt.close()

# ── 3. Text Preprocessing ──
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [ps.stem(word) for word in text if word not in stop_words]
    return ' '.join(text)

print("--- Preprocessing text... ---")
df['Processed_Text'] = df['Text'].apply(preprocess_text)

# ── 4. TF-IDF Features ──
tfidf = TfidfVectorizer(max_features=1000)
X_text = tfidf.fit_transform(df['Processed_Text']).toarray()

scaler = StandardScaler()
X_numeric = scaler.fit_transform(df[['PE_Ratio', 'Volume']])
X = np.hstack((X_text, X_numeric))
y = df['Price_Label'].values

# ── 5. TimeSeriesSplit Evaluation ──
print("--- Training with TimeSeriesSplit (5 folds)... ---")
tscv = TimeSeriesSplit(n_splits=5)

results = {'acc': [], 'auc': [], 'f1': []}
baseline_results = {'majority_acc': []}

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    majority_class = np.bincount(y_train).argmax()
    baseline_results['majority_acc'].append(accuracy_score(y_test, np.full(len(y_test), majority_class)))

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results['acc'].append(accuracy_score(y_test, y_pred))
    results['auc'].append(roc_auc_score(y_test, y_prob))
    results['f1'].append(f1_score(y_test, y_pred))

    print(f"  Fold {fold+1}: Acc={results['acc'][-1]:.4f}, AUC={results['auc'][-1]:.4f}, "
          f"F1={results['f1'][-1]:.4f} | Majority Acc={baseline_results['majority_acc'][-1]:.4f}")

# ── 6. Summary ──
avg_acc = np.mean(results['acc'])
avg_auc = np.mean(results['auc'])
avg_f1 = np.mean(results['f1'])
avg_majority = np.mean(baseline_results['majority_acc'])

print(f"\n{'='*50}")
print(f"  POC v4 Step 1 & 2 Results (TimeSeriesSplit)")
print(f"{'='*50}")
print(f"  TF-IDF + LogReg (5-fold avg):")
print(f"    Accuracy : {avg_acc:.4f}")
print(f"    ROC-AUC  : {avg_auc:.4f}")
print(f"    F1       : {avg_f1:.4f}")
print(f"  Baselines:")
print(f"    Majority Vote Acc : {avg_majority:.4f}")
print(f"  Tickers: {len(TICKERS)}, Total articles: {len(df)}")

with open('poc/result/eda_tfidf_results.txt', 'w') as f:
    f.write("ST545 POC v4 Step 1 & 2 Results\n")
    f.write("================================\n")
    f.write(f"Dataset: {len(df)} articles, {df['Date'].nunique()} dates, {len(TICKERS)} tickers\n")
    f.write(f"Tickers: {TICKERS}\n")
    f.write(f"Publisher normalization: benzinga → Benzinga\n")
    f.write(f"Validation: TimeSeriesSplit (5 folds)\n")
    f.write(f"Label Balance (Up): {df['Price_Label'].mean():.2%}\n\n")
    f.write(f"TF-IDF + LogReg (5-fold avg):\n")
    f.write(f"  Accuracy : {avg_acc:.4f}\n")
    f.write(f"  ROC-AUC  : {avg_auc:.4f}\n")
    f.write(f"  F1       : {avg_f1:.4f}\n\n")
    f.write(f"Baselines:\n")
    f.write(f"  Majority Vote Accuracy : {avg_majority:.4f}\n\n")
    f.write(f"Per-fold results:\n")
    for i in range(5):
        f.write(f"  Fold {i+1}: Acc={results['acc'][i]:.4f}, AUC={results['auc'][i]:.4f}, F1={results['f1'][i]:.4f}\n")

print("\n[+] POC v4 Step 1 & 2 complete. Results in poc/result/")
