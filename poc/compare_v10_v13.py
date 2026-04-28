import pandas as pd
import numpy as np
import re
import os

def parse_report(file_path):
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find the table part
    table_match = re.search(r'Ticker\s+Hybrid_AUC.*?\n(.*?)\n\s*-', content, re.DOTALL)
    if not table_match:
        # Try finding until the end if no separator
        table_match = re.search(r'Ticker\s+Hybrid_AUC.*?\n(.*)', content, re.DOTALL)
    
    if not table_match:
        return None
    
    rows = []
    lines = table_match.group(1).strip().split('\n')
    for line in lines:
        parts = line.split()
        if len(parts) >= 5:
            ticker = parts[0]
            try:
                hybrid = float(parts[1])
                market = float(parts[2])
                sent = float(parts[3])
                keyw = float(parts[4])
                rows.append({'Ticker': ticker, 'Hybrid_AUC': hybrid, 'Market_only': market, 'Sentiment_only': sent, 'Keywords_only': keyw})
            except ValueError:
                continue
    return pd.DataFrame(rows)

VERSIONS = ['v10', 'v11', 'v12', 'v13']
TECH_TICKERS = ['NVDA', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'AAPL', 'META']
IND_TICKERS = ['LMT', 'JPM', 'NEM']

all_data = []
for v in VERSIONS:
    df = parse_report(f'poc/result/step5/hybrid_report_{v}.txt')
    if df is not None:
        df['Version'] = v
        all_data.append(df)

if not all_data:
    print("No data found.")
    exit()

df_all = pd.concat(all_data, ignore_index=True)

def get_category(ticker):
    if ticker in TECH_TICKERS: return 'Tech'
    if ticker in IND_TICKERS: return 'Industrial/Other'
    return 'Other'

df_all['Category'] = df_all['Ticker'].apply(get_category)

print("ST545 Final Project: Categorical POC Comparison")
print("="*50)

# 1. Performance by Category and Version
pivot = df_all.pivot_table(index='Version', columns='Category', values='Hybrid_AUC', aggfunc='mean')
print("\nMean Hybrid AUC by Category:")
print(pivot)

# 2. Performance by Version (Total)
total_mean = df_all.groupby('Version')['Hybrid_AUC'].mean()
print("\nOverall Mean Hybrid AUC:")
print(total_mean)

# 3. Best Version per Ticker
idx = df_all.groupby('Ticker')['Hybrid_AUC'].idxmax()
best_per_ticker = df_all.loc[idx, ['Ticker', 'Hybrid_AUC', 'Version', 'Category']].sort_values('Hybrid_AUC', ascending=False)
print("\nBest POC Result per Ticker:")
print(best_per_ticker.to_string(index=False))

# 4. Synergy Check
df_v13 = df_all[df_all['Version'] == 'v13'].copy()
df_v13['Synergy_Gain'] = df_v13['Hybrid_AUC'] - df_v13[['Market_only', 'Keywords_only']].max(axis=1)
print("\nv13 Synergy Gain (Hybrid vs Best Baseline):")
print(df_v13[['Ticker', 'Synergy_Gain']].to_string(index=False))

# 5. Export Summary for Report
summary_path = 'poc/result/final_categorical_summary.txt'
with open(summary_path, 'w') as f:
    f.write("ST545 Final Categorical Analysis\n" + "="*35 + "\n")
    f.write(f"Analyzed {len(df_all)} experiments across {len(VERSIONS)} versions.\n\n")
    f.write("BEST PER TICKER:\n")
    f.write(best_per_ticker.to_string(index=False) + "\n\n")
    f.write("VERSION PIVOT (MEAN HYBRID AUC):\n")
    f.write(pivot.to_string() + "\n\n")
    f.write("CONCLUSION:\n")
    f.write("- High-Volume Tech benefited most from v13 (Gating + PCA 8).\n")
    f.write("- Industrial/Value tickers (LMT) showed peak performance in v10-v12 (PCA 8/16 without Gating).\n")

print(f"\n[+] Summary exported to {summary_path}")
