"""
ST545 POC v4 — Step 5: FinBERT vs TF-IDF Fair Benchmarking
============================================================
Same LogReg classifier on 3 text representations:
  1. TF-IDF (500-dim)
  2. FinBERT CLS embedding (768-dim)
  3. FinBERT 3-class softmax (3-dim, POC v1 style)
10 tickers, publisher normalization.
"""

import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
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

# ── 1. Load Data ──
TICKERS = ['NVDA', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'LMT', 'NEM', 'AAPL', 'META', 'JPM']
all_data = []

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

    news['Summary'] = news['Summary'].fillna('')
    news['Text'] = news['Headline'].str.strip() + '. ' + news['Summary'].str.strip()

    merged = pd.merge(news, market[['Date', 'Ticker', 'Close', 'Volume', 'PE_Ratio', 'Price_Label']],
                       on=['Date', 'Ticker'], how='inner')
    all_data.append(merged)

df = pd.concat(all_data, ignore_index=True).sort_values('Date').reset_index(drop=True)

# Sample for tractability (FinBERT 768-dim embeddings on ~46K texts is very slow)
MAX_SAMPLES = 8000
if len(df) > MAX_SAMPLES:
    df = df.tail(MAX_SAMPLES).reset_index(drop=True)
    print(f"--- Using last {MAX_SAMPLES} articles for benchmarking ---")

y = df['Price_Label'].astype(int).values
texts = df['Text'].tolist()
print(f"--- {len(df)} articles, label balance: {np.mean(y):.2%} positive ---")

# ── 2. TF-IDF Representation ──
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [ps.stem(word) for word in text if word not in stop_words]
    return ' '.join(text)

print("--- Building TF-IDF (500-dim)... ---")
processed = [preprocess_text(t) for t in texts]
tfidf = TfidfVectorizer(max_features=500)
X_tfidf = tfidf.fit_transform(processed).toarray()
print(f"  TF-IDF shape: {X_tfidf.shape}")

# ── 3. FinBERT 768-dim Embedding ──
print("--- Extracting FinBERT 768-dim embeddings ([CLS] token)... ---")
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
bert_model = AutoModel.from_pretrained("ProsusAI/finbert")
bert_model.eval()

def get_finbert_embeddings(texts, batch_size=64):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors='pt', padding=True,
                          truncation=True, max_length=128)
        with torch.no_grad():
            outputs = bert_model(**inputs)
        cls_emb = outputs.last_hidden_state[:, 0, :].numpy()
        embeddings.append(cls_emb)
        if (i // batch_size) % 10 == 0:
            print(f"  FinBERT embed: {i+len(batch)}/{len(texts)}")
    return np.vstack(embeddings)

X_finbert = get_finbert_embeddings(texts)
print(f"  FinBERT embedding shape: {X_finbert.shape}")

# ── 4. FinBERT 3-class softmax ──
print("--- Getting FinBERT 3-class probs... ---")
cls_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
cls_model.eval()

def get_finbert_probs(texts, batch_size=64):
    probs_all = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors='pt', padding=True,
                          truncation=True, max_length=128)
        with torch.no_grad():
            outputs = cls_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).numpy()
        probs_all.append(probs)
        if (i // batch_size) % 10 == 0:
            print(f"  FinBERT probs: {i+len(batch)}/{len(texts)}")
    return np.vstack(probs_all)

X_finbert_3d = get_finbert_probs(texts)
print(f"  FinBERT 3-class shape: {X_finbert_3d.shape}")

# ── 5. TimeSeriesSplit Evaluation ──
print("\n--- Fair Comparison: same LogisticRegression on 3 representations ---")
tscv = TimeSeriesSplit(n_splits=5)

representations = {
    'TF-IDF (500-dim)': X_tfidf,
    'FinBERT Embedding (768-dim)': X_finbert,
    'FinBERT 3-class (3-dim, v1 style)': X_finbert_3d,
}

all_results = {}
for name, X in representations.items():
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    fold_results = {'acc': [], 'auc': [], 'f1': []}
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_scaled)):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        fold_results['acc'].append(accuracy_score(y_test, y_pred))
        fold_results['auc'].append(roc_auc_score(y_test, y_prob))
        fold_results['f1'].append(f1_score(y_test, y_pred))

    all_results[name] = {k: np.mean(v) for k, v in fold_results.items()}
    all_results[name]['per_fold'] = fold_results
    print(f"  {name}: Acc={all_results[name]['acc']:.4f}, "
          f"AUC={all_results[name]['auc']:.4f}, F1={all_results[name]['f1']:.4f}")

# Majority baseline
majority_accs = []
for train_idx, test_idx in tscv.split(X_tfidf):
    y_train, y_test = y[train_idx], y[test_idx]
    maj = np.bincount(y_train).argmax()
    majority_accs.append(accuracy_score(y_test, np.full(len(y_test), maj)))
majority_avg = np.mean(majority_accs)
print(f"  Majority Vote Baseline: Acc={majority_avg:.4f}")

# ── 6. Visualization ──
methods = list(all_results.keys()) + ['Majority Vote']
accs = [all_results[m]['acc'] for m in all_results] + [majority_avg]
aucs = [all_results[m]['auc'] for m in all_results] + [0.5]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

colors = ['steelblue', 'darkorange', 'gray', 'lightgray']
axes[0].barh(methods, accs, color=colors)
axes[0].set_xlabel('Accuracy')
axes[0].set_title('Accuracy Comparison (TimeSeriesSplit 5-fold avg)')
axes[0].axvline(x=majority_avg, color='red', linestyle='--', alpha=0.7, label='Majority')
for i, v in enumerate(accs):
    axes[0].text(v + 0.005, i, f'{v:.4f}', va='center')

axes[1].barh(methods, aucs, color=colors)
axes[1].set_xlabel('ROC-AUC')
axes[1].set_title('ROC-AUC Comparison (TimeSeriesSplit 5-fold avg)')
axes[1].axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Random')
for i, v in enumerate(aucs):
    axes[1].text(v + 0.005, i, f'{v:.4f}', va='center')

plt.tight_layout()
plt.savefig('poc/result/finbert_tfidf_comparison.png', dpi=150)
plt.close()
print("\n--- Comparison chart saved ---")

# ── 7. Save Results ──
with open('poc/result/finbert_benchmark_results.txt', 'w') as f:
    f.write("ST545 POC v4 Step 5 Results: FinBERT vs TF-IDF (Fair Comparison)\n")
    f.write("=================================================================\n")
    f.write(f"Tickers: {TICKERS}\n")
    f.write(f"Dataset: {len(df)} articles (last {MAX_SAMPLES})\n")
    f.write(f"Text Input: Headline + Summary\n")
    f.write(f"Publisher normalization: benzinga → Benzinga\n")
    f.write(f"Validation: TimeSeriesSplit (5 folds)\n")
    f.write(f"Classifier: LogisticRegression (same for all)\n\n")
    f.write(f"{'Method':<40} {'Acc':>8} {'AUC':>8} {'F1':>8}\n")
    f.write("-" * 68 + "\n")
    for name, res in all_results.items():
        f.write(f"{name:<40} {res['acc']:>8.4f} {res['auc']:>8.4f} {res['f1']:>8.4f}\n")
    f.write(f"{'Majority Vote Baseline':<40} {majority_avg:>8.4f} {'N/A':>8} {'N/A':>8}\n")
    f.write(f"{'Random Baseline':<40} {'N/A':>8} {'0.5000':>8} {'N/A':>8}\n")

print("\n[+] POC v4 Step 5 complete. Results in poc/result/")
