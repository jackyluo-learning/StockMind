
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import os

# 1. Load Data
CACHE_PATH = 'dataset/sentiment_cache.csv'
EMBED_CACHE_PATH = 'dataset/finbert_embeddings_768_v8.npy'
OUTPUT_PATH = 'poc/result/step4/pca_semantic_mapping.txt'

if not os.path.exists(CACHE_PATH) or not os.path.exists(EMBED_CACHE_PATH):
    print("Error: Required data files not found.")
    exit()

df = pd.read_csv(CACHE_PATH)
embeddings = np.load(EMBED_CACHE_PATH)

# 2. Fit Global PCA (Consistency with Step 4/5 logic)
print(f"--- Fitting PCA on {len(embeddings)} articles ---")
pca = PCA(n_components=3, random_state=42)
pca_scores = pca.fit_transform(embeddings)

# 3. Interpret Top 3 Components
with open(OUTPUT_PATH, 'w') as f:
    f.write("ST545 POC - FinBERT PCA Semantic Mapping (Top 3 Components)\n")
    f.write("="*80 + "\n\n")

    for i in range(3):
        header = f"COMPONENT {i} (Explains {pca.explained_variance_ratio_[i]:.2%} variance)"
        f.write(f"{header}\n{'-'*len(header)}\n")
        
        # Get indices for highest and lowest scores
        top_indices = np.argsort(pca_scores[:, i])[-5:][::-1]
        bot_indices = np.argsort(pca_scores[:, i])[:5]
        
        f.write("\n>>> HIGH SCORE ARTICLES (Positive Correlation):\n")
        for idx in top_indices:
            row = df.iloc[idx]
            f.write(f"- [{row['Ticker']}] {row['Headline']}\n")
            f.write(f"  Summary: {str(row['Summary'])[:150]}...\n\n")
            
        f.write("\n>>> LOW SCORE ARTICLES (Negative Correlation):\n")
        for idx in bot_indices:
            row = df.iloc[idx]
            f.write(f"- [{row['Ticker']}] {row['Headline']}\n")
            f.write(f"  Summary: {str(row['Summary'])[:150]}...\n\n")
        
        f.write("\n" + "="*80 + "\n\n")

print(f"\n[+] PCA interpretation complete. Results saved to: {OUTPUT_PATH}")
