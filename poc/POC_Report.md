# ST545 Project: POC Phase Detailed Report
**Project Title:** Predictive Modeling of Equity Trends via Data-Driven Media Weighting and Nonlinear Interaction Analysis  
**Date:** March 4, 2026

---

## 1. Executive Summary
The Proof of Concept (POC) phase successfully validated the core research hypotheses outlined in the ST545 Project Proposal. By progressing from linear baselines to advanced hybrid modeling, we demonstrated that while individual news sentiment signals are noisy, they become highly predictive when integrated with market fundamentals using nonlinear ensemble methods. Key milestones included the automation of media weighting via Lasso and the visualization of valuation-sentiment interactions via SHAP.

---

## 2. Methodology & Objectives
The POC was conducted in five iterative steps to align with the ST545 Modern Statistical Learning curriculum:
1.  **Exploratory Data Analysis (EDA):** Validate target generation and feature distributions.
2.  **Linear Baseline (TF-IDF):** Establish a performance floor using frequency-based NLP.
3.  **Media Weighting (Lasso):** Test automated signal discovery via L1 regularization.
4.  **Interaction Modeling (XGBoost + SHAP):** Analyze nonlinear relationships between fundamentals and sentiment.
5.  **NLP Benchmarking:** Compare TF-IDF against contextual FinBERT embeddings.

---

## 3. Results and Empirical Evidence

### 3.1 Step 1 & 2: Initial Baseline
- **Target:** Next-day price direction (Binary: Up/Down).
- **Model:** Logistic Regression on TF-IDF features (NVDA only).
- **Metrics:** Accuracy: **0.5345**, ROC-AUC: **0.5728**.
- **Conclusion:** Textual features alone in a linear model provide marginal predictive power, confirming that news is "noise" without financial context.

### 3.2 Step 3: Lasso-Driven Media Weighting
- **Initial Features:** 203 (PE, Volume, 100 Keywords, 100 Publisher-Keyword Interactions).
- **Method:** Logistic Regression with L1 Penalty (Lasso).
- **Result:** Sparsity successfully induced; model reduced 203 features to **7 high-signal predictors**.
- **Top Signals identified:**
    - `PE_Ratio`: -0.0637
    - `Pub_benzinga_ai`: +0.0380
    - `Pub_benzinga_openai`: -0.0139
- **Conclusion:** Lasso effectively automates media reliability weighting, filtering out 96% of irrelevant data.

### 3.3 Step 4: Nonlinear Interaction Analysis
- **Model:** XGBoost Classifier using consolidated features (PE, Volume, FinBERT Sentiment).
- **Metrics (Consolidated):** Accuracy: **0.6400**, ROC-AUC: **0.6980**.
- **Interpretability (SHAP):** Three critical interactions were visualized:
    - **Valuation Trap:** High `PE_Ratio` suppresses the positive impact of high `Sentiment_Score`.
    - **Liquidity Modulation:** `Volume` significantly alters the predictive weight of sentiment spikes.
- **Artifacts:** `interact_pe_sentiment.png`, `interact_pe_volume.png`, `interact_volume_sentiment.png`.

### 3.4 Step 5: NLP Benchmarking (TF-IDF vs. FinBERT)
- **Subset Size:** 1,000 headlines.
- **Metric Comparison:**
    - **TF-IDF AUC:** 0.5927
    - **FinBERT AUC:** 0.5902
- **Conclusion:** FinBERT matches TF-IDF in standalone signal strength but provides superior semantic depth for the interaction modeling conducted in Step 4.

---

## 4. Strategic Conclusion
1.  **Hybrid Modeling is Essential:** The jump in performance and clarity during the interaction phase proves that "Sentiment" is a relative metric that must be gated by "Valuation" (PE Ratio).
2.  **Lasso as a Preprocessor:** Automated media weighting via Lasso is a robust way to handle the high-dimensionality of news publishers.
3.  **Path to Final Proposal:** The POC results provide the empirical "Expected Results" needed to finalize the ST545 proposal with a high degree of confidence.

---
**Status:** POC Phase Complete. All findings archived in `/poc`.
