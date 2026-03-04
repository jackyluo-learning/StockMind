# Project Proposal: Predictive Modeling of Equity Trends via Data-Driven Media Weighting and Nonlinear Interaction Analysis

**Course:** ST 545 Modern Statistical Learning (Spring 2026)  
**Proposed by:** Jacky Luo  
**Target Submission Date:** March 12, 2026

## 1. Central Research Questions
This project investigates the synergy between unstructured news sentiment and structured financial metrics for predicting stock price movements. Key questions include:
- **Data-Driven Media Weighting:** Can I utilize Lasso (L1 Regularization) to automatically identify "high-signal" publishers and determine their predictive weights, rather than relying on subjective manual assignment?
- **NLP Benchmarking:** Does the contextual depth of FinBERT provide a statistically significant improvement in prediction accuracy over traditional frequency-based methods like TF-IDF?
- **Nonlinear Interactions:** To what extent do nonlinear interactions between news sentiment and market fundamentals (e.g., P/E Ratio) modulate price trends, and can these be accurately captured by ensemble methods?
- **Calibrated Confidence:** How can I translate model probabilities into a reliable "AI Confidence Score" for real-world decision support?

## 2. Data Description
The project utilizes a dynamic, multi-modal dataset specifically engineered via Alpaca and Finnhub APIs.
- **Market & News Feed (Alpaca):** I extract synchronized daily historical bars (Close, Volume) and the corresponding news headlines for stocks.
- **Fundamental Metrics (Finnhub):** Trailing P/E ratios are integrated to provide a valuation context for the sentiment signals.
- **Dataset Characteristics:** Unlike static datasets, our pipeline supports incremental updates, ensuring the model remains responsive to the latest market shifts and headlines.

## 3. Methodology
I propose a robust analytic pipeline centered on the techniques explored in class:

### 3.1 NLP Comparative Study (TF-IDF vs. FinBERT)
I will conduct a controlled experiment to compare two textual embedding strategies:
- **Baseline (TF-IDF):** A frequency-based bag-of-words approach, consistent with the SMS Spam lab methodology.
- **Advanced (FinBERT):** A pre-trained Transformer model optimized for financial semantics.
This comparison will quantify the value-added by deep learning in capturing market sentiment nuances.

### 3.2 Automatic Media Weighting via Lasso (L1 Regularization)
To handle the high-dimensional space of diverse news publishers, I will implement Lasso Regression/Logistic Regression (L1 penalty). By constructing interaction terms between Sentiment_Score and Publisher_ID, the model will automatically:
- Assign coefficients (weights) to individual media sources based on historical predictive power.
- Induce Sparsity, driving the coefficients of "noisy" or irrelevant publishers to zero, thus identifying the true market-moving sources.

### 3.3 Nonlinear Interaction Modeling (XGBoost)
While linear models establish a baseline, I will employ XGBoost (Extreme Gradient Boosting) to capture complex, nonlinear interactions. Specifically, I aim to model how high P/E ratios (valuation pressure) might dampen the positive impact of bullish news sentiment—a "valuation trap" scenario that simple linear models often fail to detect.

### 3.4 Interpretability and Visualization (SHAP)
To move beyond "black-box" predictions, I will use SHAP (SHapley Additive exPlanations).
- **Global Importance:** Identify the most influential drivers of NVDA’s price.
- **Interaction Plots:** Specifically visualize the nonlinear relationship between Sentiment and P/E Ratios to validate our research hypotheses.

### 3.5 Probability Calibration
We will apply Platt Scaling or Isotonic Regression to calibrate the model’s output probabilities, ensuring the predicted "AI Confidence Score" reflects the empirical frequency of correct predictions.

## 4. Evaluation Metrics
- **Prediction Accuracy:** ROC-AUC and F1-Score for trend direction (Up/Down).
- **Sparsity Metrics:** Number of publishers retained vs. discarded by Lasso.
- **Calibration Quality:** Brier Score and Reliability Diagrams for the Confidence Score.
- **Model Comparison:** Performance delta between TF-IDF + Logistic vs. FinBERT + XGBoost.

## 5. Societal Impact
This project provides an automated, objective framework for filtering financial noise. By quantifying media reliability and providing a calibrated confidence layer, it assists investors in making data-driven decisions while mitigating the influence of market-distorting misinformation.

## 6. Project Timeline
- **March 12:** Proposal Submission.
- **March 13 – April 1:** POC Phase: Implementing Lasso for media weighting and TF-IDF baseline.
- **April 1:** Mid-checkpoint Presentation (Showcasing initial Lasso weights).
- **April 2 – April 17:** Advanced Modeling: FinBERT integration, XGBoost training, and SHAP analysis.
- **April 29:** Final Project Report Submission.
