import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

# Ensure NLTK resources are available
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# 1. Load Data
DATA_PATH = "dataset/real_nvda_dataset.csv"
if not os.path.exists(DATA_PATH):
    print(f"Error: {DATA_PATH} not found.")
    exit(1)

df = pd.read_csv(DATA_PATH)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

print(f"--- Data Loaded: {len(df)} rows ---")

# 2. Target Generation (Price_Label)
# Label = 1 if Close(t+1) > Close(t), else 0
# Since the dataset has multiple news entries per day, we first get unique daily closes
daily_prices = df[['Date', 'Close']].drop_duplicates().sort_values('Date')
daily_prices['Next_Close'] = daily_prices['Close'].shift(-1)
daily_prices['Price_Label'] = (daily_prices['Next_Close'] > daily_prices['Close']).astype(int)

# Merge back to the main dataframe
df = pd.merge(df, daily_prices[['Date', 'Price_Label']], on='Date', how='inner')

# Drop the last day since we don't have t+1 data for it
df = df.dropna(subset=['Price_Label'])
print(f"--- Price Labels Generated (Target Balance): ---\n{df['Price_Label'].value_counts(normalize=True)}")

# 3. EDA: Publisher Distribution
plt.figure(figsize=(10, 6))
top_publishers = df['Publisher'].value_counts().head(10)
sns.barplot(x=top_publishers.values, y=top_publishers.index, palette='viridis')
plt.title('Top 10 News Publishers for NVDA')
plt.xlabel('Number of Articles')
plt.savefig('poc/publisher_distribution.png')
print(f"--- Top 5 Publishers: ---\n{top_publishers.head(5)}")

# 4. Text Preprocessing (ST545 convention)
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    # Remove non-alphabetic chars
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    # Stem and remove stopwords
    text = [ps.stem(word) for word in text if word not in stop_words]
    return ' '.join(text)

print("--- Preprocessing Headlines... ---")
df['Processed_Headline'] = df['Headline'].apply(preprocess_text)

# 5. TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=1000)
X_text = tfidf.fit_transform(df['Processed_Headline']).toarray()

# Combine with PE_Ratio and Volume (Standardized)
scaler = StandardScaler()
X_numeric = scaler.fit_transform(df[['PE_Ratio', 'Volume']])
X = np.hstack((X_text, X_numeric))
y = df['Price_Label']

# 6. Model Training: Logistic Regression (ST545 Baseline)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 7. Evaluation
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print(f"\n--- POC Baseline Results (TF-IDF + LogReg) ---")
print(f"Accuracy: {acc:.4f}")
print(f"ROC-AUC:  {auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save Results Summary
with open('poc/eda_tfidf_results.txt', 'w') as f:
    f.write(f"ST545 POC Step 1 & 2 Results\n")
    f.write(f"============================\n")
    f.write(f"Dataset Rows: {len(df)}\n")
    f.write(f"Price Label Balance (Up/Down): {df['Price_Label'].mean():.2%}\n")
    f.write(f"Baseline Accuracy: {acc:.4f}\n")
    f.write(f"Baseline AUC: {auc:.4f}\n")
    f.write(f"\nClassification Report:\n{classification_report(y_test, y_pred)}\n")

print("\n[+] POC Step 1 & 2 complete. Outputs saved in 'poc/' directory.")
