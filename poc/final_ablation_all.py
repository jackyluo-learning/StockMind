import pandas as pd
import numpy as np
import os
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV
import xgboost as xgb
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re
import warnings
warnings.filterwarnings('ignore')

TICKERS = ['NVDA', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'LMT', 'NEM', 'AAPL', 'META', 'JPM']
CACHE_PATH = 'dataset/sentiment_cache.csv'
EMBED_CACHE_PATH = 'dataset/finbert_embeddings_768_v8.npy'
RESULT_PATH = 'poc/result/ablation/final_ablation_all_tickers.txt'

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))
def preprocess_text(text):
    if not isinstance(text, str): return ""
    text = re.sub('[^a-zA-Z]', ' ', text).lower().split()
    return ' '.join([ps.stem(word) for word in text if word not in stop_words])

print("--- Loading Global Data ---")
cache_df = pd.read_csv(CACHE_PATH)
embeddings = np.load(EMBED_CACHE_PATH)
pca_8 = PCA(n_components=8, random_state=42)
pca_data = pca_8.fit_transform(embeddings)
pca_cols = [f'pca_{i}' for i in range(8)]
cache_df = pd.concat([cache_df.reset_index(drop=True), pd.DataFrame(pca_data, columns=pca_cols)], axis=1)
cache_df['Date'] = pd.to_datetime(cache_df['Date'])

results = []

for ticker in TICKERS:
    print(f"\n--- Processing {ticker} ---")
    market = pd.read_csv(f"dataset/{ticker}_market.csv")
    market['Date'] = pd.to_datetime(market['Date'])
    market = market.sort_values('Date')
    market['Next_Close'] = market['Close'].shift(-1)
    market['Price_Label'] = (market['Next_Close'] > market['Close']).astype(int)
    market = market.dropna(subset=['Price_Label'])
    market['vol_pct_chg'] = market['Volume'].pct_change()
    market['pe_chg'] = market['PE_Ratio'].diff()
    market['ma10_ratio'] = market['Close'] / market['Close'].rolling(10).mean()
    market['volatility_5d'] = market['Close'].pct_change().rolling(5).std()

    ticker_news = cache_df[cache_df['Ticker'] == ticker].copy()
    ticker_news['Full_Text'] = ticker_news['Headline'].fillna('') + " " + ticker_news['Summary'].fillna('')
    daily_agg = ticker_news.groupby('Date').agg({
        'Full_Text': lambda x: " ".join(x),
        'Sentiment_Score': 'mean',
        **{c: 'mean' for c in pca_cols}
    }).reset_index()

    df_t = pd.merge(market, daily_agg, on='Date', how='inner').dropna().reset_index(drop=True)
    if len(df_t) < 50:
        print(f"Skipping {ticker} due to low samples: {len(df_t)}")
        continue

    y = df_t['Price_Label'].values
    tscv = TimeSeriesSplit(n_splits=3)
    sc = StandardScaler()

    MARKET_BASE = ['PE_Ratio', 'vol_pct_chg', 'pe_chg', 'ma10_ratio', 'volatility_5d']
    SENT_COLS = ['Sentiment_Score'] + pca_cols
    texts_processed = [preprocess_text(t) for t in df_t['Full_Text']]
    tfidf_vec = TfidfVectorizer(max_features=1000)
    tfidf_raw = tfidf_vec.fit_transform(texts_processed).toarray()
    
    lasso_model = LogisticRegressionCV(penalty='l1', solver='saga', cv=tscv, max_iter=2000, random_state=42).fit(sc.fit_transform(tfidf_raw), y)
    coef_df = pd.DataFrame({'word': tfidf_vec.get_feature_names_out(), 'coef_abs': np.abs(lasso_model.coef_[0])})
    top_10_idx = coef_df.sort_values('coef_abs', ascending=False).head(10).index
    X_keywords = tfidf_raw[:, top_10_idx]

    X_market = df_t[MARKET_BASE].values
    X_sent = df_t[SENT_COLS].values
    X_combined = np.hstack((X_market, X_sent, X_keywords))
    X_scaled = sc.fit_transform(X_combined)

    # Models
    models = {
        'XGB': xgb.XGBClassifier(eval_metric='logloss', random_state=42),
        'RF': RandomForestClassifier(random_state=42),
        'MLP': MLPClassifier(max_iter=500, random_state=42)
    }
    
    ticker_res = {'Ticker': ticker}
    for name, m in models.items():
        if name == 'MLP':
            params = {'hidden_layer_sizes': [(32,), (32, 16)], 'alpha': [0.001, 0.01]}
        elif name == 'RF':
            params = {'n_estimators': [50, 100], 'max_depth': [5, 10]}
        else: # XGB
            params = {'n_estimators': [50, 100], 'max_depth': [3, 5], 'learning_rate': [0.05, 0.1]}
        
        grid = GridSearchCV(m, params, cv=tscv, scoring='roc_auc', n_jobs=-1)
        grid.fit(X_scaled, y)
        ticker_res[name] = grid.best_score_
        print(f"  {name}: {grid.best_score_:.4f}")
    
    results.append(ticker_res)

df_res = pd.DataFrame(results)
with open(RESULT_PATH, 'w') as f:
    f.write("ST545 Final Ablation Study: Model Comparison (v10 Feature Set)\n")
    f.write("="*65 + "\n")
    f.write(df_res.to_string(index=False) + "\n\n")
    f.write("MEAN PERFORMANCE:\n")
    f.write(df_res[['XGB', 'RF', 'MLP']].mean().to_string() + "\n")

print(f"\n[+] Final ablation complete. Results in {RESULT_PATH}")
