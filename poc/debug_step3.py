
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

TICKERS = ['NVDA', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'LMT', 'NEM', 'AAPL', 'META', 'JPM']
CACHE_PATH = 'dataset/sentiment_cache.csv'

cache = pd.read_csv(CACHE_PATH)
cache['Date'] = pd.to_datetime(cache['Date'])

results = []

for ticker in TICKERS:
    try:
        market = pd.read_csv(f"dataset/{ticker}_market.csv")
        market['Date'] = pd.to_datetime(market['Date'])
        market = market.sort_values('Date')
        market['Next_Close'] = market['Close'].shift(-1)
        market['Price_Label'] = (market['Next_Close'] > market['Close']).astype(int)
        market = market.dropna(subset=['Price_Label'])

        ticker_news = cache[cache['Ticker'] == ticker].copy()
        df_t = pd.merge(ticker_news, market[['Date', 'PE_Ratio', 'Volume', 'Price_Label']], on='Date', how='inner')
        
        sample_count = len(df_t)
        unique_pubs = df_t['Publisher'].nunique()
        
        skipped_at_start = (sample_count < 50 or unique_pubs < 2)
        coef_empty = "N/A"
        
        if not skipped_at_start:
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
            coef_df = pd.DataFrame({'Feature': feat_names, 'Coef': coefs}).loc[lambda x: x['Coef'] != 0]
            coef_empty = coef_df.empty

        results.append({
            'Ticker': ticker,
            'Samples': sample_count,
            'Pubs': unique_pubs,
            'Skipped_Initial': skipped_at_start,
            'Coef_Empty': coef_empty
        })
    except Exception as e:
        results.append({
            'Ticker': ticker,
            'Samples': 0,
            'Pubs': 0,
            'Skipped_Initial': 'Error',
            'Coef_Empty': str(e)
        })

print(pd.DataFrame(results).to_string(index=False))
