# 📈 StockMind - AI-Driven Stock Analysis

## 🎯 Project Overview
**StockMind** is an advanced data pipeline and analysis platform designed for stock market prediction using hybrid data sources. It integrates market data (OHLCV), fundamental metrics (PE Ratios), and financial news for sentiment-weighted analysis.

### 🛠️ Core Technologies
- **Language**: Python 3.14+ (Conda environment: `stock_mind`)
- **APIs**:
  - **Alpaca**: Used for stable historical daily bars and comprehensive news (Benzinga source).
  - **Finnhub**: Used for historical and current fundamental metrics (Quarterly EPS/PE).
  - **yfinance**: (Legacy/Secondary) used for additional market metadata.
- **Data Handling**: `pandas`, `numpy`, `requests`, `scikit-learn`, `xgboost`, `shap`, `transformers`.

## 🚀 Building and Running

### Environment Setup
The project uses a Conda environment named `stock_mind`.
```bash
conda activate stock_mind
```

### Key Commands
- **Run the Hybrid Pipeline**: Fetches historical data and news, saving to the `dataset/` directory.
  ```bash
  python dataset/alpaca_finnhub_pipeline.py
  ```
- **Execute Tuned ST545 POC**: Run the consolidated modeling, tuning, and interpretability analysis.
  ```bash
  python poc/step4_xgboost_shap.py
  ```
- **Execute Ablation Study**: Perform systematic feature and model comparison for LMT.
  ```bash
  python poc/ablation_study.py
  ```
- **Export Final Dataset**: Consolidate all ticker data into a single master feature matrix.
  ```bash
  python export_final_dataset.py
  ```

## 📂 Directory Structure

- **`/dataset`**: Core pipeline logic and consolidated data storage.
    - `alpaca_finnhub_pipeline.py`: Main hybrid data ingestion engine.
    - `real_*_dataset.csv`: Final fused feature matrices for modeling (NVDA, MSFT, GOOGL).
    - `sentiment_cache.csv`: Global news cache with FinBERT scores.
    - `finbert_embeddings_768_v8.npy`: High-dimensional semantic embeddings.
- **`/poc`**: Proof of Concept for ST545 project modeling.
    - `step0_sentiment_cache.py`: Pre-calculation and caching of FinBERT sentiment scores.
    - `step1_2_eda_tfidf.py`: Unified NLP benchmarking (TF-IDF vs FinBERT) and global EDA.
    - `step3_media_weighting.py`: Ticker-specific media weighting (Lasso with Ridge fallback).
    - `step4_xgboost_shap.py`: Tuned NLP representation battle (XGBoost + GridSearchCV) and SHAP attribution.
    - `ablation_study.py`: Systematic model and feature group synergy analysis (LMT focus).
    - **`/result`**: Organized artifacts and reports from the POC phase.
        - **/step0**: Sentiment cache summaries and distribution metrics.
        - **/step1_2**: NLP benchmarking reports and publisher distribution plots.
        - **/step3**: Media weighting coefficients and publisher importance charts for 10 tickers.
        - **/step4**: Tuned XGBoost AUC comparison and SHAP summary plots.
        - **/ablation**: Detailed synergy reports and model class comparisons for LMT.
- **`/poc_v1` to `/poc_v7`**: Experimental archives and versioned iterations of the POC phase.
- **`export_final_dataset.py`**: Script to generate the unified `final_daily_dataset.csv` from cached data.
- **`ST545_StockMind_Progress_Report_V1_to_V5.md`**: Comprehensive review of project evolution.

## 🧪 Development Conventions

1.  **Hybrid Data Strategy**: Prefer Alpaca for news and bars to avoid the aggressive rate-limiting often encountered with yfinance.
2.  **Robust Error Handling**:
    - Always wrap API calls in try-except blocks.
    - Implement exponential backoff for `429 Too Many Requests`.
3.  **Local Caching**: Implement aggressive local caching for raw data. **All dataset outputs must be saved to the `/dataset` folder.**
4.  **Result Organization**: All POC artifacts must be saved into step-specific subdirectories within `poc/result/` to maintain clarity.
5.  **Nonlinear Modeling**: Prioritize **XGBoost** with **GridSearchCV** tuning and **SHAP** interpretability. Utilize full **768-dim FinBERT embeddings** for semantic feature capture.

---
*Note: This project is optimized for the Alpaca and Finnhub APIs. Ensure valid API keys are configured in the pipeline scripts.*

## 📈 Project Status

### Recent Improvements (March 2026)
- **Result Reorganization**: Restructured the `poc/result/` folder into a modular hierarchy (step0-step4, ablation) for better scalability and navigation.
- **Deep Semantic Integration**: Updated Step 4 to perform PCA reduction directly on full **768-dimensional FinBERT embeddings**, capturing richer semantic signals than the previous 3-class scores.
- **Hyperparameter Optimization**: Integrated `GridSearchCV` with `TimeSeriesSplit` across the pipeline, ensuring performance metrics reflect optimized model states.
- **Robust Attribution (SHAP)**: Decomposed model logic into Sentiment vs. Market attribution. Findings reveal a dominant **~80% sentiment contribution** across the 10 core tickers.
- **Media Weighting Fallback**: Implemented a **Lasso-to-Ridge fallback** mechanism in Step 3, ensuring valid media weighting profiles for all tickers even in low-signal regimes.
- **Ablation Synergy**: Confirmed feature synergy for **LMT** (AUC 0.6269), proving that the combination of sentiment and market fundamentals significantly outperforms individual groups.
---
