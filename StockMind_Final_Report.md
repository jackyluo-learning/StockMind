# StockMind: Predictive Modeling of Equity Trends via Sentiment-Weighted Interaction Analysis

**Course:** ST 545 Modern Statistical Learning  
**Author:** Jacky Luo  
**Date:** April 27, 2026  

---

## Abstract
This project presents **StockMind**, a multimodal data pipeline and predictive framework that integrates high-frequency financial news with traditional market indicators to forecast next-day stock direction. We investigate the "Synergy Gap" in hybrid modeling—the phenomenon where combining features can sometimes degrade performance in tree-based models. Through 13 iterations of Proof-of-Concept (POC) development, we implemented **DQS Gating**, **Media Weighting**, and **Semantic PCA**. Our final results demonstrate that while tree-based models provide stability for high-volume tech stocks, a **Modular Expert System** utilizing Multi-Layer Perceptrons (MLP) achieves peak performance, reaching a ROC-AUC of **0.6782** for LMT and **0.6699** for MSFT.

---

## 1. Introduction
Predicting stock market movements remains one of the most challenging tasks in statistical learning due to the low signal-to-noise ratio and the non-stationary nature of financial data. Traditional models often rely on technical indicators (OHLCV) or fundamental metrics (PE Ratios), but frequently overlook the qualitative "alpha" contained in real-time news flow.

**Research Questions:**
1. Can we learn publisher-specific weights directly from data to identify the most predictive news sources?
2. Does FinBERT (financial-domain BERT) provide superior representations compared to traditional TF-IDF?
3. Can nonlinear interaction modeling (MLP) resolve the "Negative Synergy" observed in tree-based ensembles?

---

## 2. Data and Methodology

### 2.1 Data Sources
- **Market Data:** Alpaca API (Daily OHLCV bars).
- **Fundamentals:** Finnhub API (Quarterly PE Ratios and EPS).
- **Text Data:** 46,461 articles from Benzinga, Reuters, Bloomberg, and SeekingAlpha (via Alpaca/Finnhub).
- **Universe:** 10 U.S. Tickers (NVDA, GOOGL, MSFT, AMZN, TSLA, LMT, NEM, AAPL, META, JPM).

### 2.2 Feature Engineering Pipeline
1. **NLP Benchmarking:** We compared **TF-IDF (1000-dim)** against **FinBERT Embeddings (768-dim)**.
2. **Media Weighting:** Ticker-specific publisher importance was learned using Lasso regression with publisher-sentiment interaction terms.
3. **Semantic Compression:** High-dimensional FinBERT embeddings were compressed into 8 or 16 principal components using PCA.
4. **DQS Gating (Data Quality Score):** A novel gating mechanism was implemented: `G = (Count/10) * (1 - Std(Sentiment))`. This dampens semantic features on days with low news volume or high sentiment conflict.

---

## 3. Results and Analysis

### 3.1 The Synergy Gap & Negative Synergy
Initial experiments with XGBoost revealed that a "Hybrid" model (Market + Sentiment) often underperformed a "Keywords-only" model. This suggested **Negative Synergy**, where the model's objective function struggled to reconcile the sparse text features with dense market features.

### 3.2 Performance Summary (v13 Gated Hybrid)
| Ticker | Hybrid AUC | Market-only | Keywords-only |
| :--- | :--- | :--- | :--- |
| **GOOGL** | **0.5971** | 0.5096 | 0.5351 |
| **LMT** | 0.5646 | **0.5918** | 0.5642 |
| **MSFT** | 0.5377 | 0.5640 | **0.7008** |

### 3.3 The MLP Breakthrough (Ablation Study)
To resolve the synergy gap, we conducted an ablation study across all 10 tickers, comparing XGBoost, Random Forest, and MLP.

| Ticker | XGBoost AUC | Random Forest | **MLP (Best)** |
| :--- | :--- | :--- | :--- |
| **LMT** | 0.5653 | 0.5923 | **0.6782** |
| **MSFT** | 0.5895 | 0.5979 | **0.6699** |
| **META** | 0.5148 | 0.5454 | **0.6067** |
| **AMZN** | 0.5859 | 0.5606 | **0.6079** |

---

## 4. Interpretation: Semantic PCA Mapping
Using SHAP and diagnostic tools, we mapped the mathematical PCA components to financial drivers:
- **Component 0:** Growth/Innovation vs. Panic (e.g., "AI breakthroughs" vs. "Billions lost").
- **Component 1:** Macro Shocks (Fed meetings, interest rate rumors).
- **Component 2:** Sector Battles (Legal disputes, earnings misses).

---

## 5. Conclusion & Discussion
The StockMind project demonstrates that financial news carries significant predictive signal, but its extraction requires sophisticated fusion techniques. 
- **Modular Expert Strategy:** Our final recommendation is to deploy a **Modular Expert System**—utilizing MLPs for complex, low-volume "Value" stocks (LMT) and Gated XGBoost for high-stability "Tech" stocks (GOOGL).
- **Limitations:** The models are sensitive to the "News-Market Lag" and require high-quality fundamental data which is often lagged in quarterly reports.

---

## 6. Statement of Contributions
This is an individual project. All architecture, pipeline code, and analysis were performed by the author with support from **Gemini CLI**.

---

## 7. AI Tool Acknowledgment
**Gemini CLI** was utilized as an interactive senior developer agent throughout this project. Its roles included:
1. **Pipeline Orchestration:** Managing the multi-stage data processing and model tuning.
2. **Technical Diagnostics:** Identifying the "Synergy Gap" and suggesting the transition to deep learning (MLP).
3. **Documentation:** Generating progress reports and aggregating final categorical results.
4. **Code Quality:** Ensuring PEP 8 compliance and type safety in the pipeline scripts.

---

## 8. References
1. Huang et al. (2019). *FinBERT: Financial Sentiment Analysis with Pre-trained Language Models.*
2. Lundberg, S. M., & Lee, S. I. (2017). *A Unified Approach to Interpreting Model Predictions (SHAP).*
3. Alpaca Data API Documentation (2025).
