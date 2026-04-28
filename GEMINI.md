# 📈 StockMind - AI-Driven Stock Analysis

## 🎯 Project Overview
**StockMind** is an advanced data pipeline and analysis platform designed for stock market prediction using hybrid data sources. It integrates market data (OHLCV), fundamental metrics (PE Ratios), and financial news for sentiment-weighted interaction analysis.

### 🛠️ Core Technologies
- **Language**: Python 3.14+ (Conda environment: `stock_mind`)
- **APIs**:
  - **Alpaca**: Stable historical daily bars and comprehensive news (Benzinga source).
  - **Finnhub**: Historical and current fundamental metrics (Quarterly EPS/PE).
- **Data Handling**: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `shap`, `transformers` (FinBERT).

## 🚀 Building and Running

### Environment Setup
The project uses a Conda environment named `stock_mind`.
```bash
conda activate stock_mind
```

### Key Commands
- **Run Hybrid Pipeline**: Fetches historical data and news.
  ```bash
  python dataset/alpaca_finnhub_pipeline.py
  ```
- **Execute Synergy Analysis (v13)**: Run the gated hybrid NLP experiment.
  ```bash
  python poc/step5_hybrid_nlp_v13.py
  ```
- **Execute Model Comparison**: Systematic model class comparison (MLP focus).
  ```bash
  python poc/ablation_study.py
  ```
- **Generate Comprehensive Report**: Aggregate all POC results into categorical summaries.
  ```bash
  python poc/compare_v10_v13.py
  ```
- **Export Final Dataset**: Generate the daily-level consolidated dataset for submission.
  ```bash
  python export_final_dataset.py
  ```

## 📂 Directory Structure

- **`/dataset`**: Core pipeline logic and consolidated data storage.
    - `sentiment_cache.csv`: Global news cache with FinBERT scores and embeddings.
    - `finbert_embeddings_768_v8.npy`: High-dimensional semantic embeddings.
- **`/poc`**: Proof of Concept iterations (v1 - v13).
    - `step3_media_weighting.py`: Ticker-specific weighting (Lasso with Ridge fallback).
    - `step5_hybrid_nlp_v10..v13.py`: Versioned synergy experiments (Lasso Keywords, PCA, Gating).
    - `interpret_pca_text.py`: Diagnostic tool for mapping PCA components to financial drivers.
    - `compare_v10_v13.py`: Categorical performance evaluator (Sector, Volume).
    - **`/result`**: Artifacts and reports.
        - **/step3**: Publisher importance charts and weights.
        - **/step5**: Detailed synergy breakdowns and aggregated means for v10-v13.
        - **/ablation**: Model class comparison (XGB vs RF vs MLP) for LMT.
- **`ST545_StockMind_Progress_Report_V13.md`**: Current definitive status report.

## 🧪 Development Conventions

1.  **DQS Gating**: Implement Data Quality Score gating: `(Count/10) * (1 - Std)`. Dampen semantic features on low-confidence trading days.
2.  **Lasso Feature Selection**: Prioritize **Top 10 Keywords** specifically for each ticker to capture sparse "alpha tokens."
3.  **Semantic Dimensionality**: 
    - Use **PCA-16** for complex industrials (Precision focus).
    - Use **PCA-8** for high-volume tech (Generalization focus).
4.  **Deep Learning Shift**: Prioritize **MLP (Multi-Layer Perceptron)** for model fusion when XGBoost synergy gaps occur.

---
*Note: This project is optimized for the Alpaca and Finnhub APIs. Ensure valid API keys are configured in the pipeline scripts.*

## 📈 Project Status

### **[FINAL POC PHASE COMPLETE]** (April 2026)
Successfully completed 13 POC iterations. Now transitioning from feature engineering to **Model Fusion**.

### Latest Breakthroughs
- **MLP Peak Performance**: Achieved **AUC 0.6782** for **LMT** using the MLP architecture, significantly outperforming tree-based models (+13% gain).
- **Keyword Alpha Confirmation**: Lasso-selected keywords identified as the strongest standalone predictor (**Mean AUC 0.5564**).
- **Synergy Gap Discovery**: Identified **Negative Synergy** in tree-based models (XGBoost) where combining features sometimes underperforms standalone keywords.
- **Gated Generalization**: Implementation of **DQS Gating** in v12/v13 provided **+6% AUC gains** for high-volume tech stocks (AAPL, GOOGL).
- **Semantic Mapping**: Successfully mapped PCA Component 0 to **Growth/Innovation**, Component 1 to **Macro Shocks**, and Component 2 to **Sector Battles**.
- **Categorical Strategy**: Transitioned to a **Modular Expert System**:
    - **MLP** for Industrials/Low-Volume (High Alpha capture).
    - **Gated XGBoost** for Tech/High-Volume (Stability & Noise Filtering).
---
