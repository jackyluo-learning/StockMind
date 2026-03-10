"""
ST545 POC v9 (Comprehensive) — Step 3: Per-Ticker Media Weighting
==================================================================
1. Ticker-specific Media Weighting analysis.
2. Strategy: Try Lasso (L1) first for sparse signals.
3. Fallback: If Lasso zeros all coefficients, use Ridge (L2) to show relative weights.
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
    
    # Pure Media Weighting: Only Interaction Terms (Publisher_i * Sentiment)
    X = pub_dummies.values * sentiment
    y = df_t['Price_Label'].values
    feat_names = [f"{p}×Sent" for p in pub_names]

    # --- STEP A: Try Lasso (L1) ---
    model = LogisticRegressionCV(penalty='l1', solver='saga', cv=3, max_iter=5000, random_state=42).fit(X, y)
    coefs = model.coef_[0]
    method = "Lasso (L1)"

    # --- STEP B: Fallback to Ridge (L2) if Lasso is empty ---
    if np.sum(coefs != 0) == 0:
        print(f"  {ticker}: Lasso zeroed all coefficients. Falling back to Ridge (L2)...")
        model = LogisticRegressionCV(penalty='l2', cv=3, max_iter=5000, random_state=42).fit(X, y)
        coefs = model.coef_[0]
        method = "Ridge (L2)"
    
    non_zero = np.sum(coefs != 0)
    print(f"  {ticker} Done ({method}). Samples: {len(df_t)}, Non-zero Coefficients: {non_zero}")

    # Plot 1: Coefficients (Media Weights)
    coef_df = pd.DataFrame({'Feature': feat_names, 'Coef': coefs})
    # For Lasso, we only show non-zero. For Ridge, we show all since they are all non-zero.
    if method == "Lasso (L1)":
        coef_df = coef_df.loc[lambda x: x['Coef'] != 0]
    
    coef_df = coef_df.sort_values('Coef')

    if not coef_df.empty:
        plt.figure(figsize=(12, 8))
        colors = ['green' if c > 0 else 'red' for c in coef_df['Coef']]
        plt.barh(coef_df['Feature'], coef_df['Coef'], color=colors)
        plt.title(f'Media Weighting ({method}): {ticker}')
        plt.xlabel('Coefficient Weight')
        plt.savefig(f'poc/result/step3/lasso_coef_{ticker}.png', bbox_inches='tight'); plt.close()

    # Plot 2: Importance (Absolute Magnitude)
    plt.figure(figsize=(10, 6))
    coef_df['Abs_Coef'] = coef_df['Coef'].abs()
    coef_df = coef_df.sort_values('Abs_Coef', ascending=False)
    sns.barplot(data=coef_df, x='Abs_Coef', y='Feature', palette='coolwarm')
    plt.title(f'Publisher Importance ({method}): {ticker}')
    plt.savefig(f'poc/result/step3/pub_imp_{ticker}.png', bbox_inches='tight'); plt.close()

    # Save weights for Step 5
    # Feature names were f"{p}×Sent" where p is pub_pubname
    # We want to extract 'pubname'
    coef_df['Publisher'] = coef_df['Feature'].str.replace('×Sent', '').str.replace('pub_', '')
    coef_df[['Publisher', 'Coef']].to_csv(f'poc/result/step3/weights_{ticker}.csv', index=False)

print("\n[+] Step 3 complete. Individual plots saved in poc/result/step3/")
