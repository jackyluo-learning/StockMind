import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

# Ensure NLTK resources
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def preprocess_text(text):
    if not isinstance(text, str): return ""
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    text = re.sub('[^a-zA-Z]', ' ', text).lower().split()
    text = [ps.stem(word) for word in text if word not in stop_words]
    return ' '.join(text)

# 1. Load and Combine Data
tickers = ['nvda', 'googl', 'msft']
dfs = []

for t in tickers:
    path = f"dataset/real_{t}_dataset.csv"
    if os.path.exists(path):
        temp_df = pd.read_csv(path)
        temp_df['Ticker'] = t.upper()
        # Generate target for each ticker separately
        temp_df['Date'] = pd.to_datetime(temp_df['Date'])
        temp_df = temp_df.sort_values('Date')
        daily = temp_df[['Date', 'Close']].drop_duplicates().sort_values('Date')
        daily['Next_Close'] = daily['Close'].shift(-1)
        daily['Price_Label'] = (daily['Next_Close'] > daily['Close']).astype(int)
        temp_df = pd.merge(temp_df, daily[['Date', 'Price_Label']], on='Date', how='inner')
        dfs.append(temp_df)

df = pd.concat(dfs, ignore_index=True).dropna(subset=['Price_Label'])
print(f"--- Combined Dataset: {len(df)} rows across {len(tickers)} tickers ---")

# 2. Feature Engineering: Interaction Terms
# We want: Price_Label ~ Publisher * Sentiment_Score
# For POC, we'll use TF-IDF top terms as "sentiment proxies" + One-Hot for Publishers

df['Processed_Headline'] = df['Headline'].apply(preprocess_text)
tfidf = TfidfVectorizer(max_features=100) # Keep it small for POC interactions
tfidf_matrix = tfidf.fit_transform(df['Processed_Headline']).toarray()
tfidf_cols = tfidf.get_feature_names_out()

# One-hot encode Publishers
publisher_dummies = pd.get_dummies(df['Publisher'], prefix='Pub')
pub_cols = publisher_dummies.columns

# Interaction Terms: tfidf_word * Pub_name
interaction_data = {}
for pub in pub_cols:
    for word in tfidf_cols:
        col_name = f"{pub}_{word}"
        interaction_data[col_name] = tfidf_matrix[:, list(tfidf_cols).index(word)] * publisher_dummies[pub].values

interaction_df = pd.DataFrame(interaction_data)

# Combine all features
X_numeric = StandardScaler().fit_transform(df[['PE_Ratio', 'Volume']])
X = np.hstack((X_numeric, tfidf_matrix, publisher_dummies.values, interaction_df.values))
y = df['Price_Label']

feature_names = ['PE_Ratio', 'Volume'] + list(tfidf_cols) + list(pub_cols) + list(interaction_df.columns)

# 3. Lasso (L1) Logistic Regression
print("--- Training Lasso (L1) model to identify key features... ---")
# Using liblinear solver for L1 penalty
model = LogisticRegressionCV(
    cv=5, 
    penalty='l1', 
    solver='liblinear', 
    max_iter=1000, 
    random_state=42,
    scoring='roc_auc'
)
model.fit(X, y)

# 4. Identify Sparsity (Zeroed out coefficients)
coefs = model.coef_[0]
important_features = []
for name, coef in zip(feature_names, coefs):
    if abs(coef) > 1e-5:
        important_features.append((name, coef))

important_features.sort(key=lambda x: abs(x[1]), reverse=True)

# 5. Output Results
print(f"\n--- POC Step 3: Lasso Media Weighting Results ---")
print(f"Total features: {len(feature_names)}")
print(f"Features with non-zero weights: {len(important_features)}")
print(f"Top 10 most predictive features/interactions:")
for name, coef in important_features[:10]:
    print(f"  {name:40} : {coef:.4f}")

with open('poc/media_weighting_results.txt', 'w') as f:
    f.write("ST545 POC Step 3: Media Weighting Results\n")
    f.write("========================================\n")
    f.write(f"Combined Dataset Size: {len(df)}\n")
    f.write(f"Total Initial Features: {len(feature_names)}\n")
    f.write(f"Features Retained by Lasso: {len(important_features)}\n")
    f.write("\nTop 20 Features (Non-zero Weights):\n")
    for name, coef in important_features[:20]:
        f.write(f"{name}: {coef:.6f}\n")

print("\n[+] POC Step 3 complete. Results saved to 'poc/media_weighting_results.txt'.")
