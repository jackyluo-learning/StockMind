# 📈 StockMind - AI-Driven Stock Analysis

## 🎯 Project Overview
**StockMind** is an advanced data pipeline and analysis platform designed for stock market prediction using hybrid data sources. It integrates market data (OHLCV), fundamental metrics (PE Ratios), and financial news for sentiment-weighted analysis.

### 🛠️ Core Technologies
- **Language**: Python 3.14+ (Conda environment: `stock_mind`)
- **APIs**:
  - **Alpaca**: Used for stable historical daily bars and comprehensive news (Benzinga source).
  - **Finnhub**: Used for historical and current fundamental metrics (Quarterly EPS/PE).
  - **yfinance**: (Legacy/Secondary) used for additional market metadata.
- **Data Handling**: `pandas`, `numpy`, `requests`.

## 🚀 Building and Running

### Environment Setup
The project uses a Conda environment named `stock_mind`.
```bash
conda activate stock_mind
```

### Key Commands
- **Run the Hybrid Pipeline**: Fetches one year of historical data and news, then performs incremental updates.
  ```bash
  python dataset/alpaca_finnhub_pipeline.py
  ```

## 📂 Directory Structure

- **`/dataset`**: Contains core pipeline logic and data caches.
    - `alpaca_finnhub_pipeline.py`: The main hybrid data ingestion engine.
    - `*_news.csv`: Cached news articles from various sources.
    - `*_hist_cache.csv`: Cached historical OHLCV data.
- **`real_nvda_dataset.csv`**: The final fused feature matrix (Date, Ticker, Close, Volume, PE_Ratio, Publisher, Headline) used for modeling.
- **`GEMINI.md`**: Foundational context and project mandates.

## 🧪 Development Conventions

1.  **Hybrid Data Strategy**: Prefer Alpaca for news and bars to avoid the aggressive rate-limiting often encountered with yfinance.
2.  **Robust Error Handling**:
    - Always wrap API calls in try-except blocks.
    - Implement exponential backoff for `429 Too Many Requests` errors (especially for Finnhub free tier).
    - Provide fallback mechanisms (e.g., using current PE snapshot if historical EPS is unavailable).
3.  **Local Caching**: Implement aggressive local CSV caching for all raw data to minimize API calls and speed up the "Seed Building" process.
4.  **Data Consistency**: Ensure all timestamps are standardized to `YYYY-MM-DD` and aligned via inner joins on the `Date` column for the final feature matrix.
5.  **Security**: Never commit API keys. Use environment variables or local configuration files (ensure `alpaca_key`, `finnhub_key`, etc., are protected).

---
*Note: This project is optimized for the Alpaca and Finnhub APIs. Ensure valid API keys are configured in the pipeline scripts.*
