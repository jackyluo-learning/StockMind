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
- **Ablation Study**: Perform feature importance and removal analysis.
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
    - `alpaca_finnhub_pipeline_v1.py`: Legacy version of the hybrid pipeline.
    - `real_*_dataset.csv`: Final fused feature matrices for modeling (NVDA, MSFT, GOOGL).
    - `final_daily_dataset.csv`: Master consolidated dataset for project submission.
    - `*_news.csv`, `*_hist_cache.csv`: Local data artifacts.
- **`/poc`**: Proof of Concept for ST545 project modeling.
    - `step0_sentiment_cache.py`: Pre-calculation and caching of FinBERT sentiment scores.
    - `step1_2_eda_tfidf.py`: Initial EDA and TF-IDF baseline.
    - `step3_media_weighting.py`: Lasso-based automated publisher signal weighting.
    - `step4_xgboost_shap.py`: Consolidated nonlinear interaction modeling and SHAP visualization.
    - `step5_finbert_benchmarking.py`: Comparative NLP analysis (TF-IDF vs. FinBERT).
    - `ablation_study.py`: Systematic feature removal to test model robustness.
    - **`/result`**: Artifacts and reports from the POC phase.
        - `POC_Report.md`: Detailed summary of POC methodology and findings.
        - `ablation_results.txt`: Performance breakdown of model-ticker-feature combinations.
        - `*_results.txt`: Numerical outputs for each POC step.
        - `*.png`: Visualization artifacts (SHAP interaction plots, publisher distributions).
- **`/poc_v1` to `/poc_v4`**: Experimental archives and versioned iterations of the POC phase.
- **`export_final_dataset.py`**: Script to generate the unified `final_daily_dataset.csv` from cached data.
- **`ST545_Final_Project_Proposal_Draft.md`**: Current proposal documentation for ST545.
- **`ST545_StockMind_Progress_Report_V1_to_V5.md`**: Comprehensive review of project evolution and model performance.

## 🧪 Development Conventions

1.  **Hybrid Data Strategy**: Prefer Alpaca for news and bars to avoid the aggressive rate-limiting often encountered with yfinance.
2.  **Robust Error Handling**:
    - Always wrap API calls in try-except blocks.
    - Implement exponential backoff for `429 Too Many Requests` (especially for Finnhub free tier).
3.  **Local Caching**: Implement aggressive local CSV caching for all raw data. **All dataset outputs must be saved to the `/dataset` folder.**
4.  **Data Consistency**: Ensure all timestamps are standardized to `YYYY-MM-DD` and aligned via inner joins on the `Date` column.
5.  **Nonlinear Modeling**: For ST545 modeling, prioritize **XGBoost** and **MLP** with **SHAP** interpretability to capture valuation-sentiment interactions.

---
*Note: This project is optimized for the Alpaca and Finnhub APIs. Ensure valid API keys are configured in the pipeline scripts.*

## 📈 Project Status

### Recent Improvements (March 2026)
- **Comprehensive Progress Review**: Generated a longitudinal report (`ST545_StockMind_Progress_Report_V1_to_V5.md`) tracking the evolution from a niche NVDA success to a robust 10-ticker generalized model.
- **ST545 POC Validation**: Successfully completed a comprehensive POC phase (Results archived in `poc/result/POC_Report.md`).
    - **Lasso Weighting**: Validated automated media weighting via L1 regularization, reducing 203 noisy features to 7 key signals (e.g., `Pub_benzinga_ai`).
    - **Nonlinear Boost**: Confirmed that XGBoost significantly outperforms linear baselines (ROC-AUC increase from 0.57 to ~0.70 with consolidated features).
    - **FinBERT Integration**: Benchmarked contextual FinBERT embeddings, matching TF-IDF signal strength while adding semantic depth.
    - **Interpretability (SHAP)**: Identified critical market dynamics:
        - **Valuation Trap**: High `PE_Ratio` suppresses positive sentiment impact.
        - **Liquidity Modulation**: `Volume` significantly alters the predictive weight of sentiment spikes.
    - **Ablation Testing**: Confirmed feature synergy; results in `poc/result/ablation_results.txt` show that combined models outperform single-source models.
- **Pipeline Standardization**: Updated `alpaca_finnhub_pipeline.py` to automatically save all ticker datasets into the `dataset/` directory.
- **Repository Consolidation**: Relocated all `*_dataset.csv` files from the root to the `dataset/` folder and updated `.gitignore` to prevent tracking of any CSV data.
- **Unified Export**: Implemented `export_final_dataset.py` to produce a high-fidelity dataset for 10 core tickers (NVDA, GOOGL, MSFT, etc.).
- **Dynamic PE Ratio**: (Previous) Migrated from static snapshots to historical quarterly EPS-based dynamic computation.
