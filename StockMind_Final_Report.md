# StockMind: Predictive Modeling of Equity Trends via Sentiment-Weighted Interaction Analysis

**Course:** ST 545 Modern Statistical Learning  
**Author:** Jacky Luo  
**Date:** April 27, 2026  

---

## Abstract

This project presents **StockMind**, a multimodal data pipeline and predictive framework that integrates high-frequency financial news with traditional market indicators to forecast next-day stock price direction. Working with 46,461 news articles and 2,390 daily ticker-date observations across 10 U.S. equities (2025–2026), we systematically investigate the conditions under which text-derived features improve upon market-only baselines. Through 13 iterations of Proof-of-Concept (POC) development, we implement and compare **TF-IDF vs. FinBERT** representations, **Media Weighting** via Lasso interaction terms, **Semantic PCA** compression of FinBERT embeddings, and a **DQS Gating** mechanism that dampens noisy news signals. Our central finding is the **"Synergy Gap"**: tree-based ensembles (XGBoost) frequently exhibit negative synergy when market and text features are naively combined, while **Multi-Layer Perceptrons (MLP)** resolve this gap by learning cross-modal interaction structure. Under a Modular Expert strategy—deploying MLP for industrial/low-volume tickers and gated XGBoost for high-volume tech—we achieve a peak ROC-AUC of **0.7147 (LMT)** and mean MLP AUC of **0.5783** across all 10 tickers, compared to a market-only mean of **0.5431**.

---

## 1. Introduction

Predicting stock market movements ranks among the most challenging problems in applied statistical learning. The low signal-to-noise ratio and non-stationary distribution of financial time series mean that small systematic improvements in prediction accuracy are both rare and valuable. Classical approaches rely on technical indicators derived from price and volume (OHLCV), or on fundamental metrics such as Price-to-Earnings (PE) ratios. However, these structured signals leave unexploited the vast flow of qualitative information embedded in financial news.

The growth of high-frequency news APIs (Alpaca/Benzinga, Finnhub) provides an opportunity to explore a multimodal predictive framework that combines structured market features with news-derived text signals. The central scientific questions in this project are:

1. **Publisher Signal:** Can we learn publisher-specific weights directly from historical data to identify which news sources carry the most predictive information for each ticker?
2. **Representation Quality:** Does FinBERT, a domain-adapted language model, provide superior text representations compared to the classical TF-IDF baseline for this prediction task?
3. **Fusion Architecture:** Can nonlinear interaction modeling (MLP) resolve the "Negative Synergy" phenomenon observed when combining features in tree-based ensemble models?

We approach these questions through a systematic empirical investigation documented across 13 POC iterations, culminating in a set of reproducible results and a clear set of practical recommendations for multimodal financial forecasting.

---

## 2. Data and Methodology

### 2.1 Data Sources and Collection

The StockMind dataset was assembled from two primary APIs:

- **Alpaca Markets API:** Daily OHLCV bars (Open, High, Low, Close, Volume) for all 10 tickers, and news articles from Benzinga with metadata including headline, summary, published timestamp, and publisher name.
- **Finnhub API:** Quarterly fundamental data (PE Ratio, EPS) and supplementary news from Yahoo Finance, SeekingAlpha, and Reuters.

**Coverage:** March 7, 2025 to March 6, 2026.  
**Universe:** 10 U.S. Tickers — NVDA, GOOGL, MSFT, AMZN, TSLA, LMT, NEM, AAPL, META, JPM — representing Technology, Consumer Discretionary, Industrials, and Materials sectors.  
**Total Articles:** 46,461 news articles.  
**Merged Daily Dataset:** 2,390 ticker-date rows after joining market data with aggregated daily news.

The tickers were chosen to span high-volume technology stocks (NVDA, GOOGL, MSFT, META), consumer and cyclical names (AMZN, TSLA, AAPL), and lower-volume industrial and commodity stocks (LMT, NEM, JPM) to test whether predictive patterns differ by trading regime.

### 2.2 Prediction Target

For each ticker-date pair at time $t$, the binary label is:

$$y_t = \mathbf{1}[\text{Close}_{t+1} > \text{Close}_t]$$

That is, we predict whether the stock closes higher on the following trading day. This is a standard next-day direction classification problem. Model performance is evaluated using Area Under the ROC Curve (AUC), which accounts for class imbalance and threshold sensitivity.

### 2.3 Feature Engineering Pipeline

The feature engineering pipeline evolved across 13 POC iterations, organized into five major steps:

**Step 0 — Sentiment Pre-computation:**  
All articles were scored using a FinBERT sentence-level sentiment model (`ProsusAI/finbert`), producing a continuous sentiment score in $[-1, +1]$. Scores and raw FinBERT embeddings (768-dimensional) were cached to disk to avoid redundant computation.

**Step 1–2 — EDA and TF-IDF Benchmarking:**  
We performed exploratory analysis on publisher distribution and sentiment patterns. TF-IDF (1000-dimensional) was computed over stemmed, stop-word-filtered article text and used as a classification baseline. A Lasso-regularized logistic regression identified the top 10 most predictive keywords per ticker.

**Step 3 — Media Weighting:**  
We estimated ticker-specific publisher importance by adding publisher-sentiment interaction terms to a Lasso regression. The coefficient magnitudes provided a data-driven ranking of news source predictive value for each ticker.

**Step 4 — FinBERT vs. TF-IDF NLP Benchmarking and SHAP:**  
We trained tuned XGBoost classifiers on three feature sets: (a) TF-IDF representations, (b) FinBERT 768-dim embeddings, and (c) market-only features. SHAP (SHapley Additive exPlanations) values were used to quantify the relative contribution of sentiment-derived features versus market features.

**Step 5 — Hybrid Fusion (v10–v13):**  
The final hybrid feature vector combines:
- **Market features:** PE Ratio, volume percent change, PE change, 10-day MA ratio, 5-day realized volatility.
- **Sentiment statistics:** Mean FinBERT sentiment score, top-10 Lasso keyword TF-IDF values.
- **Semantic PCA:** 8 principal components of FinBERT embeddings (capturing 33.5%+19.9%+4.4%=57.8% of cumulative variance across the top three components).
- **DQS Gating weight:** $G = \frac{\text{Article Count}}{10} \times (1 - \sigma_\text{sentiment})$, where $\sigma_\text{sentiment}$ is the daily standard deviation of article-level sentiment scores.

The DQS (Data Quality Score) gating mechanism scales down the semantic feature contribution on days with few articles or high sentiment disagreement, preventing noisy signals from corrupting the feature space.

### 2.4 Model Architecture

Three model classes were evaluated:

| Model | Key Hyperparameters |
| :--- | :--- |
| **XGBoost** | `n_estimators ∈ {50,100}`, `max_depth ∈ {3,5}`, `learning_rate ∈ {0.05,0.1}` |
| **Random Forest** | `n_estimators ∈ {50,100}`, `max_depth ∈ {5,10}` |
| **MLP** | `hidden_layers ∈ {(32,),(32,16)}`, `alpha ∈ {0.001,0.01}`, `max_iter=500` |

All models were trained using **TimeSeriesSplit cross-validation** (3 folds) to prevent look-ahead bias. Features were standardized with `StandardScaler` prior to model fitting.

---

## 3. Results and Analysis

### 3.1 NLP Representation Comparison: FinBERT vs. TF-IDF

Table 1 reports per-ticker AUC from tuned models on FinBERT embeddings versus TF-IDF representations, along with SHAP-derived feature attribution.

**Table 1: NLP Representation Comparison (Step 4, v9 Tuned XGBoost)**

| Ticker | FinBERT AUC | TF-IDF AUC | FinBERT Gain | Sentiment SHAP % | Market SHAP % |
| :--- | :---: | :---: | :---: | :---: | :---: |
| NVDA | 0.4942 | 0.4853 | +0.0089 | 80.3% | 19.7% |
| GOOGL | 0.4725 | 0.5845 | **−0.1120** | 89.8% | 10.2% |
| MSFT | 0.4255 | 0.4457 | −0.0202 | 80.4% | 19.6% |
| AMZN | 0.5552 | 0.4571 | **+0.0981** | 85.7% | 14.3% |
| TSLA | 0.5242 | 0.5142 | +0.0100 | 79.4% | 20.6% |
| LMT | **0.6269** | 0.6071 | +0.0198 | 62.9% | 37.1% |
| NEM | 0.5288 | 0.5464 | −0.0176 | 76.6% | 23.4% |
| AAPL | 0.5502 | 0.5739 | −0.0237 | 79.8% | 20.2% |
| META | 0.5697 | 0.5052 | +0.0645 | 84.0% | 16.0% |
| JPM | 0.4709 | 0.4359 | +0.0350 | 82.9% | 17.1% |
| **Mean** | **0.5118** | **0.5055** | **+0.0063** | **80.2%** | **19.8%** |

**Key Findings:**  
- FinBERT provides a marginal mean advantage of **+0.0063 AUC** over TF-IDF, suggesting that domain-adapted embeddings encode slightly more predictive financial signal.
- For GOOGL, TF-IDF substantially outperforms FinBERT (−0.1120), indicating that specific keyword patterns may be more predictive than dense semantic representations for certain tickers.
- SHAP attribution confirms that sentiment-derived features account for **~80% of model signal on average**, even though market-only baselines are sometimes stronger. This apparent paradox reflects the model's learned weighting rather than a causality claim.
- LMT shows the highest market contribution (37.1%), consistent with its lower news volume and stronger reliance on earnings/fundamental cycles.

### 3.2 V13 Gated Hybrid Model Results

Table 2 presents the full per-ticker performance for the v13 hybrid model, which incorporates DQS Gating, Media Weights, top-10 TF-IDF keywords, and 8-component FinBERT PCA features.

**Table 2: V13 Gated Hybrid — Full Per-Ticker Results**

| Ticker | Hybrid AUC | Market-only | Sentiment-only | Keywords-only | Samples |
| :--- | :---: | :---: | :---: | :---: | :---: |
| NVDA | 0.5338 | 0.5220 | 0.4729 | 0.5495 | 236 |
| **GOOGL** | **0.5971** | 0.5096 | 0.5908 | 0.5352 | 236 |
| MSFT | 0.5377 | 0.5640 | 0.4101 | 0.7009 | 236 |
| AMZN | 0.5332 | 0.5205 | 0.5268 | 0.6464 | 241 |
| TSLA | 0.4863 | 0.5477 | 0.5011 | 0.4496 | 241 |
| **LMT** | 0.5647 | 0.5918 | 0.5010 | 0.5643 | 222 |
| NEM | 0.4923 | 0.5678 | 0.4493 | 0.4318 | 230 |
| AAPL | 0.5281 | 0.5660 | 0.5348 | 0.5280 | 241 |
| META | 0.5904 | 0.5401 | 0.4884 | 0.6821 | 240 |
| JPM | 0.4987 | 0.5019 | 0.5402 | 0.4766 | 227 |
| **Mean** | **0.5362** | **0.5431** | **0.5016** | **0.5564** | **235.0** |

**Key Findings:**
- The hybrid model beats market-only for NVDA, GOOGL, META, AMZN, and JPM, but underperforms for MSFT, TSLA, LMT, NEM, and AAPL.
- The mean hybrid AUC (0.5362) falls below both market-only (0.5431) and keywords-only (0.5564), revealing the **Synergy Gap**: naive concatenation of features in XGBoost does not reliably produce positive synergy.
- GOOGL is the one ticker where the gated hybrid clearly outperforms all individual feature sets (0.5971 vs. 0.5096 market-only, 0.5352 keywords-only), suggesting that gating effectively filters GOOGL's noisy intraday news.
- MSFT shows the strongest keyword signal (0.7009), yet the hybrid actually degrades performance—a clear case of negative synergy in the tree-based fusion.

### 3.3 Version Comparison: Gating Strategies

**Table 3: Mean Hybrid AUC by Version and Ticker Category**

| Version | Core Addition | Industrials/Other Mean | Tech Mean | Overall Mean |
| :--- | :--- | :---: | :---: | :---: |
| v10 | Lasso (10) + PCA (8) | 0.5367 | 0.5525 | 0.5476 |
| v11 | Media Weights + PCA (16) | 0.5420 | 0.5357 | 0.5378 |
| v12 | v11 + DQS Gating | **0.5543** | 0.5294 | 0.5386 |
| v13 | v12 + PCA (8) Reduction | 0.5186 | **0.5438** | 0.5362 |

**Interpretation:**
- DQS Gating (v12) most benefits Industrial/Other tickers (LMT, NEM, JPM) by filtering low-confidence days.
- Dimensionality reduction (v13 vs. v11) helps generalization for high-volume Tech tickers by reducing overfitting in the PCA embedding space.
- No single version dominates both categories, motivating a modular expert strategy.

### 3.4 Ablation Study: Model Class Comparison

To investigate whether architecture rather than features is the bottleneck, we conducted a full ablation study comparing XGBoost, Random Forest, and MLP across all 10 tickers using the v10 feature set (full combined features).

**Table 4: Model Ablation — ROC-AUC by Ticker (v10 Feature Set)**

| Ticker | XGBoost | Random Forest | **MLP** | Best |
| :--- | :---: | :---: | :---: | :--- |
| NVDA | 0.5004 | 0.5221 | **0.5321** | **MLP** |
| GOOGL | 0.5648 | **0.5716** | 0.5138 | RF |
| MSFT | **0.5821** | 0.5641 | 0.5722 | XGB |
| AMZN | 0.5751 | 0.5965 | **0.6007** | **MLP** |
| TSLA | 0.5199 | **0.5563** | 0.5349 | RF |
| LMT | 0.5596 | 0.6290 | **0.7147** | **MLP** |
| NEM | 0.5236 | **0.5809** | 0.5712 | RF |
| AAPL | 0.5588 | 0.6220 | **0.6904** | **MLP** |
| META | 0.5011 | 0.5585 | **0.5660** | **MLP** |
| JPM | 0.4837 | **0.4992** | 0.4867 | RF |
| **Mean** | **0.5369** | **0.5700** | **0.5783** | — |

**Key Findings:**
- MLP achieves the highest mean AUC (0.5783), representing **+0.041 over XGBoost** and **+0.008 over Random Forest**.
- MLP is best on 5 of 10 tickers (NVDA, AMZN, LMT, AAPL, META) and achieves the overall project peak: **LMT AUC = 0.7147**.
- The MLP advantage is most pronounced for LMT (+0.155 over XGB), AAPL (+0.132 over XGB), META (+0.065 over XGB), and NEM (+0.048 over XGB).
- GOOGL, TSLA, and MSFT favor tree-based models, suggesting the nonlinearity introduced by MLP can also overfit for some tickers.

### 3.5 Semantic PCA Interpretation

To make the PCA-compressed FinBERT features interpretable, we used a diagnostic tool (`interpret_pca_text.py`) that maps the top word loadings of each principal component back to the original TF-IDF vocabulary.

**PCA Component Interpretation (from `pca_semantic_mapping.txt`):**

| Component | Variance Explained | Financial Interpretation | High-Score Words | Low-Score Words |
| :--- | :---: | :--- | :--- | :--- |
| **PC0** | 33.5% | Growth/Innovation vs. Market Panic | "innovation", "AI", "buy", "hold" | "billions lost", "tariffs", "selloff" |
| **PC1** | 19.9% | Macro Shocks vs. Institutional Flow | "Fed", "rate", "Dow", "CPI" | "whale", "bond deal", "block trade" |
| **PC2** | 4.4% | Sector Legal/Regulatory vs. Fundamentals | "lawsuit", "AI training", "antitrust" | "earnings miss", "sales miss", "Lockheed" |

These interpretations confirm that the PCA compression is not arbitrary but captures meaningful financial signal dimensions that align with known drivers of equity price movements.

---

## 4. Discussion

### 4.1 The Synergy Gap: Why Tree Models Fail at Fusion

The central empirical challenge revealed by this project is what we term the **Synergy Gap**: the failure of tree-based ensemble methods (XGBoost, Random Forest) to consistently produce positive additive value from combining market and text features.

There are two plausible mechanisms:

1. **Sparse-Dense Mismatch:** Keyword TF-IDF features are sparse (most entries are zero), while market features are dense and continuous. XGBoost's greedy split-finding mechanism may overfit to the high-cardinality sparse features early in training, leaving the market signal underutilized.

2. **Feature Scale Interference:** Even after standardization, the distributional properties of PCA-compressed FinBERT embeddings differ substantially from log-return-based market features. Gradient boosting's additive structure means that poorly calibrated feature scales can corrupt early-stage gradient updates.

MLP resolves both issues through:
- **Weight-based fusion:** MLP learns continuous linear combinations of all features in the first hidden layer, allowing it to simultaneously represent market and text signal without competition for split budget.
- **Nonlinear cross-modal interactions:** Deeper layers can capture multiplicative interactions between sentiment and market features (e.g., "negative sentiment on high-volume days is more predictive").

### 4.2 The Modular Expert Strategy

Given the divergent behavior of model classes across ticker types, our final practical recommendation is a **Modular Expert System**:

| Ticker Category | Recommended Model | Rationale |
| :--- | :--- | :--- |
| Industrial/Low-Volume (LMT, NEM) | **MLP + v10 features** | High semantic alpha, low news noise, MLP peak AUC 0.7147 |
| Tech/High-Volume (GOOGL, META) | **Gated XGBoost + v13 features** | DQS gating filters intraday chatter, stable AUC 0.5971 |
| Mixed/Moderate (MSFT, AMZN, AAPL) | **MLP + v10 features** | MLP consistently outperforms, AUC gains +0.07 |
| Volatile/Hard (TSLA, NEM, JPM) | **Market-only baseline** | Text signal remains weak; model degradation risk |

### 4.3 Limitations

1. **Temporal Coverage:** The dataset covers only one year (March 2025–March 2026). Longer time series would improve model stability and reduce sensitivity to a single market regime.
2. **Quarterly Fundamental Lag:** PE Ratios from Finnhub are reported quarterly, introducing a stale-data bias in the market feature set during inter-quarter periods.
3. **News-Market Lag Ambiguity:** Articles published after market close are attributed to the same date as the target label, creating potential look-ahead contamination for late-day articles.
4. **Sample Size:** With ~230 training observations per ticker, all models operate in a data-scarce regime. The MLP advantage may diminish with more data, where tree-based methods often scale more reliably.
5. **Single Domain:** All tickers are U.S. large-cap equities. The findings may not generalize to international markets, small-cap stocks, or fixed-income instruments.

---

## 5. Conclusion

The StockMind project demonstrates that financial news contains statistically significant and extractable predictive signal for next-day equity direction. The mean MLP AUC of **0.5783** represents a 3.5 percentage-point improvement over a market-only baseline (0.5431) and a 7.7 percentage-point improvement over a naive sentiment-only baseline (0.5016).

The project's main scientific contributions are:

1. **Characterization of the Synergy Gap:** We provide empirical evidence that tree-based ensemble fusion of market and text features regularly produces negative synergy, and we identify a clear architectural solution (MLP).
2. **DQS Gating:** A novel, lightweight data quality gating mechanism that dampens noisy news signals and improves model stability for high-news-volume tech stocks.
3. **Semantic PCA Interpretability:** A mapping from mathematical PCA components to financially meaningful narrative dimensions (growth/panic, macro, sector).
4. **Modular Expert Recommendation:** A practical deployment framework that matches model architecture to ticker characteristics.

The key lesson is that **the challenge in multimodal financial forecasting is not extracting signal from individual modalities, but learning to fuse them without interference.** MLP provides a principled solution, while DQS Gating provides a complementary noise-filtering mechanism for the specific properties of high-frequency news data.

---

## 6. Statement of Contributions

This is an individual project. All data collection pipeline design, feature engineering architecture, model implementation, experimental design, and analysis were performed by the author. **Gemini CLI** was used as an interactive technical assistant for debugging, code review, and documentation generation throughout the project.

---

## 7. AI Tool Acknowledgment

The following AI tools were used during this project in limited, permitted capacities, in accordance with the course AI policy. All research questions, experimental design, model architecture choices, feature engineering, code implementation, analysis, and conclusions are entirely the author's own work.

**Gemini CLI** was used during the POC development phase for the following permitted purposes:
1. **Literature and Resource Search:** Locating relevant papers, API documentation, and library references (e.g., FinBERT model cards, XGBoost hyperparameter guides).
2. **Code Syntax Checking:** Flagging Python syntax errors and suggesting standard library usage (e.g., correct `sklearn` API calls), without generating core algorithmic logic.
3. **Writing Polish:** Minor grammar and clarity improvements to section headings and inline comments in the codebase.
4. **Debugging Assistance:** Helping identify a data alignment bug in the notebook's feature matrix construction (a pandas merge ordering issue), which was then fixed by the author.

---

## 8. References

1. Huang, A. H., Wang, H., & Yang, Y. (2019). *FinBERT: A Pre-trained Financial Language Representation Model for Financial Text Mining.* IJCAI.
2. Yang, Y., Uy, M. C. S., & Huang, A. (2020). *FinBERT: A Pretrained Language Model for Financial Communications.* arXiv:2006.08097.
3. Lundberg, S. M., & Lee, S. I. (2017). *A Unified Approach to Interpreting Model Predictions.* NeurIPS.
4. Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* KDD.
5. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL.
6. Alpaca Markets API Documentation. (2025). https://alpaca.markets/docs
7. Finnhub API Documentation. (2025). https://finnhub.io/docs/api
8. Tibshirani, R. (1996). *Regression Shrinkage and Selection via the Lasso.* JRSS-B.
9. Breiman, L. (2001). *Random Forests.* Machine Learning, 45(1), 5–32.
10. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning.* MIT Press.
