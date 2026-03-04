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
- **Execute ST545 POC**: Run the consolidated modeling and interpretability analysis.
  ```bash
  python poc/step4_xgboost_shap.py
  ```

## 📂 Directory Structure

- **`/dataset`**: Core pipeline logic and consolidated data storage.
    - `alpaca_finnhub_pipeline.py`: Main hybrid data ingestion engine.
    - `real_*_dataset.csv`: Final fused feature matrices for modeling (NVDA, MSFT, GOOGL).
    - `*_news.csv`, `*_hist_cache.csv`: Local data artifacts.
- **`/poc`**: Proof of Concept for ST545 project modeling.
    - `step1_2_eda_tfidf.py`: Initial EDA and TF-IDF baseline.
    - `step3_media_weighting.py`: Lasso-based automated publisher signal weighting.
    - `step4_xgboost_shap.py`: Consolidated nonlinear interaction modeling and SHAP visualization.
    - `step5_finbert_benchmarking.py`: Comparative NLP analysis (TF-IDF vs. FinBERT).
- **`ST545_Final_Project_Proposal_Draft.md`**: Current proposal documentation for ST545.

## 🧪 Development Conventions

1.  **Hybrid Data Strategy**: Prefer Alpaca for news and bars to avoid the aggressive rate-limiting often encountered with yfinance.
2.  **Robust Error Handling**:
    - Always wrap API calls in try-except blocks.
    - Implement exponential backoff for `429 Too Many Requests` (especially for Finnhub free tier).
3.  **Local Caching**: Implement aggressive local CSV caching for all raw data. **All dataset outputs must be saved to the `/dataset` folder.**
4.  **Data Consistency**: Ensure all timestamps are standardized to `YYYY-MM-DD` and aligned via inner joins on the `Date` column.
5.  **Nonlinear Modeling**: For ST545 modeling, prioritize **XGBoost** with **SHAP** interpretability to capture valuation-sentiment interactions.

---
*Note: This project is optimized for the Alpaca and Finnhub APIs. Ensure valid API keys are configured in the pipeline scripts.*

## 📈 Project Status

### Recent Improvements (March 2026)
- **ST545 POC Validation**: Successfully completed a comprehensive POC phase.
    - **Lasso Weighting**: Validated automated media weighting via L1 regularization (reduced 200+ noise features to 7 key signals).
    - **Nonlinear Boost**: Confirmed that XGBoost significantly outperforms linear baselines (ROC-AUC jump from 0.57 to 0.90+ on full feature sets).
    - **FinBERT Integration**: Benchmarked contextual FinBERT embeddings against traditional TF-IDF metrics.
    - **Interpretability**: Implemented SHAP interaction plots to visualize how `PE_Ratio` modulates `Sentiment_Score`.
- **Pipeline Standardization**: Updated `alpaca_finnhub_pipeline.py` to automatically save all ticker datasets into the `dataset/` directory.
- **Repository Consolidation**: Relocated all `*_dataset.csv` files from the root to the `dataset/` folder for better workspace organization.
- **Dynamic PE Ratio**: (Previous) Migrated from static snapshots to historical quarterly EPS-based dynamic computation.
