"""
ST545 POC v9 (Comprehensive) — Step 3: Per-Ticker Media Weighting
==================================================================
1. Ticker-specific Lasso analysis.
2. Individual plots for EACH ticker (lasso coefficients, publisher importance).
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

TICKERS = ['NVDA', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'LMT', 'NEM', 'AAPL', 'META', 'JPM']
CACHE_PATH = 'dataset/sentiment_cache.csv'

cache = pd.read_csv(CACHE_PATH)
cache['Date'] = pd.to_datetime(cache['Date'])

for ticker in TICKERS:
    print(f"\n>>> Analyzing Media: {ticker}...")
    market = pd.read_csv(f"dataset/{ticker}_market.csv")
    market['Date'] = pd.to_datetime(market['Date'])
    market = market.sort_values('Date')
    market['Next_Close'] = market['Close'].shift(-1)
    market['Price_Label'] = (market['Next_Close'] > market['Close']).astype(int)
    market = market.dropna(subset=['Price_Label'])

    ticker_news = cache[cache['Ticker'] == ticker].copy()
    df_t = pd.merge(ticker_news, market[['Date', 'PE_Ratio', 'Volume', 'Price_Label']], on='Date', how='inner')
    
    if len(df_t) < 50 or df_t['Publisher'].nunique() < 2: continue

    pub_dummies = pd.get_dummies(df_t['Publisher'], prefix='pub')
    pub_names = pub_dummies.columns.tolist()
    sentiment = df_t['Sentiment_Score'].values.reshape(-1, 1)
    X_inter = pub_dummies.values * sentiment
    X_num = StandardScaler().fit_transform(df_t[['PE_Ratio', 'Volume']])
    X = np.hstack((pub_dummies.values, sentiment, X_inter, X_num))
    y = df_t['Price_Label'].values
    feat_names = pub_names + ['Sentiment'] + [f"{p}×Sent" for p in pub_names] + ['PE', 'Vol']

    lasso = LogisticRegressionCV(penalty='l1', solver='saga', cv=3, max_iter=5000, random_state=42).fit(X, y)
    coefs = lasso.coef_[0]

    # Plot 1: Coefficients
    coef_df = pd.DataFrame({'Feature': feat_names, 'Coef': coefs}).loc[lambda x: x['Coef'] != 0].sort_values('Coef')
    if not coef_df.empty:
        plt.figure(figsize=(12, 8))
        colors = ['green' if c > 0 else 'red' for c in coef_df['Coef']]
        plt.barh(coef_df['Feature'], coef_df['Coef'], color=colors)
        plt.title(f'Lasso Coefficients: {ticker}')
        plt.savefig(f'poc/result/lasso_coef_{ticker}.png', bbox_inches='tight'); plt.close()

        # Plot 2: Importance
        pub_imp = {p: np.abs(coefs[feat_names.index(f"{p}×Sent")]) if f"{p}×Sent" in feat_names else 0 for p in pub_names}
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(pub_imp.values()), y=list(pub_imp.keys()), palette='coolwarm')
        plt.title(f'Publisher Importance: {ticker}')
        plt.savefig(f'poc/result/pub_imp_{ticker}.png', bbox_inches='tight'); plt.close()

print("\n[+] Step 3 complete. Individual plots saved.")
