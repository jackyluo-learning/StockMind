import pandas as pd
import numpy as np
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from tqdm import tqdm
import re

# 1. Load Combined Data
tickers = ['nvda', 'googl', 'msft']
dfs = []
for t in tickers:
    path = f"dataset/real_{t}_dataset.csv"
    if os.path.exists(path):
        temp_df = pd.read_csv(path)
        temp_df['Date'] = pd.to_datetime(temp_df['Date'])
        temp_df = temp_df.sort_values('Date')
        daily = temp_df[['Date', 'Close']].drop_duplicates().sort_values('Date')
        daily['Next_Close'] = daily['Close'].shift(-1)
        daily['Price_Label'] = (daily['Next_Close'] > daily['Close']).astype(int)
        temp_df = pd.merge(temp_df, daily[['Date', 'Price_Label']], on='Date', how='inner')
        dfs.append(temp_df)

df = pd.concat(dfs, ignore_index=True).dropna(subset=['Price_Label'])

# For POC Benchmarking, use a subset to save time (FinBERT on CPU is slow)
df_subset = df.sample(n=1000, random_state=42).copy()
print(f"--- Benchmarking on {len(df_subset)} rows ---")

# 2. TF-IDF Baseline
tfidf = TfidfVectorizer(max_features=500)
X_tfidf = tfidf.fit_transform(df_subset['Headline']).toarray()
y = df_subset['Price_Label']

X_train_tf, X_test_tf, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)
model_tf = LogisticRegression(max_iter=1000)
model_tf.fit(X_train_tf, y_train)
y_prob_tf = model_tf.predict_proba(X_test_tf)[:, 1]

# 3. FinBERT Sentiment Extraction
print("--- Loading FinBERT and processing headlines... ---")
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model_fb = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

def get_finbert_sentiment(headlines):
    inputs = tokenizer(headlines, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model_fb(**inputs)
    # FinBERT outputs 3 classes: positive, negative, neutral
    return torch.nn.functional.softmax(outputs.logits, dim=-1).numpy()

# Process in batches for efficiency
batch_size = 32
fb_features = []
headlines = df_subset['Headline'].tolist()

for i in tqdm(range(0, len(headlines), batch_size)):
    batch = headlines[i:i+batch_size]
    fb_features.append(get_finbert_sentiment(batch))

X_finbert = np.vstack(fb_features)

X_train_fb, X_test_fb, _, _ = train_test_split(X_finbert, y, test_size=0.2, random_state=42)
model_fb_log = LogisticRegression(max_iter=1000)
model_fb_log.fit(X_train_fb, y_train)
y_prob_fb = model_fb_log.predict_proba(X_test_fb)[:, 1]

# 4. Results Comparison
res_tfidf = {
    "Acc": accuracy_score(y_test, (y_prob_tf > 0.5).astype(int)),
    "AUC": roc_auc_score(y_test, y_prob_tf),
    "F1": f1_score(y_test, (y_prob_tf > 0.5).astype(int))
}

res_finbert = {
    "Acc": accuracy_score(y_test, (y_prob_fb > 0.5).astype(int)),
    "AUC": roc_auc_score(y_test, y_prob_fb),
    "F1": f1_score(y_test, (y_prob_fb > 0.5).astype(int))
}

print("\n--- NLP Benchmarking Results ---")
print(f"{'Metric':<10} | {'TF-IDF':<10} | {'FinBERT':<10}")
print("-" * 35)
for m in ["Acc", "AUC", "F1"]:
    print(f"{m:<10} | {res_tfidf[m]:<10.4f} | {res_finbert[m]:<10.4f}")

with open('poc/nlp_benchmarking_results.txt', 'w') as f:
    f.write("ST545 POC Step 5: NLP Benchmarking (TF-IDF vs FinBERT)\n")
    f.write("====================================================\n")
    f.write(f"Subset Size: {len(df_subset)}\n\n")
    f.write(f"{'Metric':<10} | {'TF-IDF':<10} | {'FinBERT':<10}\n")
    f.write("-" * 35 + "\n")
    for m in ["Acc", "AUC", "F1"]:
        f.write(f"{m:<10} | {res_tfidf[m]:<10.4f} | {res_finbert[m]:<10.4f}\n")

print("\n[+] POC Step 5 complete. Results saved to 'poc/nlp_benchmarking_results.txt'.")
