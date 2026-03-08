import pandas as pd
import numpy as np
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from tqdm import tqdm
import re

# 1. Load Combined Data (NVDA, GOOGL, MSFT)
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
# For efficiency in POC, use a representative sample
df_sample = df.sample(n=2000, random_state=42).copy()
print(f"--- Dataset Size for Combined XGBoost & SHAP: {len(df_sample)} rows ---")

# 2. Generate Continuous Sentiment_Score using FinBERT
print("--- Loading FinBERT for Sentiment Extraction... ---")
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model_fb = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

def get_sentiment_score(headlines):
    inputs = tokenizer(headlines, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model_fb(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1).numpy()
    # Score = Prob(Positive) - Prob(Negative)
    return probs[:, 0] - probs[:, 1]

batch_size = 32
headlines = df_sample['Headline'].tolist()
scores = []

print("--- Calculating Sentiment Scores... ---")
for i in tqdm(range(0, len(headlines), batch_size)):
    batch = headlines[i:i+batch_size]
    scores.extend(get_sentiment_score(batch))

df_sample['Sentiment_Score'] = scores

# 3. Model Training
# Features: PE_Ratio, Volume, Sentiment_Score
X = df_sample[['PE_Ratio', 'Volume', 'Sentiment_Score']]
y = df_sample['Price_Label']

# Standardize features for cleaner SHAP visualization
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

print("--- Training XGBoost Model... ---")
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.05,
    random_state=42,
    eval_metric='logloss'
)
model.fit(X_train, y_train)

# 4. Evaluation
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
print(f"\nXGBoost Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"XGBoost ROC-AUC:  {roc_auc_score(y_test, y_prob):.4f}")

# 5. SHAP Interaction Analysis
print("\n--- Generating SHAP Interaction Plots... ---")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

def save_interaction(feat_main, feat_interact, filename):
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(
        feat_main, 
        shap_values, 
        X_test, 
        interaction_index=feat_interact, 
        show=False
    )
    plt.title(f"Nonlinear Interaction: {feat_main} vs. {feat_interact}")
    plt.savefig(f"poc/{filename}")
    plt.close()

# Figure 1: PE_Ratio vs Sentiment_Score (The "Valuation Trap" hypothesis)
save_interaction('PE_Ratio', 'Sentiment_Score', 'interact_pe_sentiment.png')

# Figure 2: PE_Ratio vs Volume
save_interaction('PE_Ratio', 'Volume', 'interact_pe_volume.png')

# Figure 3: Volume vs Sentiment_Score
save_interaction('Volume', 'Sentiment_Score', 'interact_volume_sentiment.png')

# Summary Plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, show=False)
plt.title("XGBoost + SHAP POC Summary")
plt.savefig('poc/xgboost_shap_summary.png')
plt.close()

with open('poc/xgboost_shap_results.txt', 'w') as f:
    f.write("ST545 POC Step 4 (Consolidated): XGBoost & SHAP Interaction Results\n")
    f.write("==================================================================\n")
    f.write(f"XGBoost Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
    f.write(f"XGBoost ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}\n")
    f.write("\nInteraction figures generated in poc/:\n")
    f.write("- interact_pe_sentiment.png\n")
    f.write("- interact_pe_volume.png\n")
    f.write("- interact_volume_sentiment.png\n")

print("\n[+] SUCCESS! Consolidated POC Step 4 complete. All figures and results are in 'poc/'.")
