"""
Alpaca + Finnhub Hybrid Data Pipeline
 - Alpaca: Historical daily OHLCV (bars), News (news)  ← Stable, no 429 issues
 - Finnhub: PE fundamentals (/stock/metric)
Output format: Date, Ticker, Close, Volume, PE_Ratio, Publisher, Headline
"""

import pandas as pd
import numpy as np
import requests
import time
import os
import sys
from datetime import datetime, timedelta

# ── Alpaca SDK imports ──
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame


class AlpacaFinnhubPipeline:
    FINNHUB_URL = "https://finnhub.io/api/v1"

    def __init__(
        self,
        alpaca_key: str,
        alpaca_secret: str,
        finnhub_key: str,
        ticker: str = "NVDA",
        cache_file: str = "dataset/real_nvda_dataset.csv",
    ):
        self.ticker = ticker
        self.cache_file = cache_file
        self.hist_cache_file = f"dataset/{ticker}_hist_cache.csv"
        self.news_cache_file = f"dataset/{ticker}_alpaca_news.csv"

        # ── Alpaca client (paper-trading key is sufficient for market data) ──
        self.alpaca = StockHistoricalDataClient(alpaca_key, alpaca_secret)
        self.alpaca_key = alpaca_key
        self.alpaca_secret = alpaca_secret

        # ── Finnhub session (used only for PE) ──
        self.fh_session = requests.Session()
        self.fh_session.params = {"token": finnhub_key}

    # ==========================================
    # Utilities
    # ==========================================
    def _finnhub_get(self, endpoint, params=None, description="data"):
        """Finnhub GET with retry (free tier: 60 req/min)"""
        url = f"{self.FINNHUB_URL}{endpoint}"
        for attempt in range(1, 4):
            resp = self.fh_session.get(url, params=params, timeout=15)
            if resp.status_code == 429:
                delay = 5 * (2 ** (attempt - 1))
                print(f"[Retry {attempt}/3] Finnhub {description} rate-limited, waiting {delay}s...")
                time.sleep(delay)
                continue
            resp.raise_for_status()
            return resp.json()
        raise RuntimeError(f"Finnhub failed to fetch {description}: still rate-limited after 3 retries")

    # ==========================================
    # 1. Historical OHLCV — Alpaca Bars API
    # ==========================================
    def fetch_history(self, start_date: str = None, end_date: str = None):
        """
        Fetch daily historical OHLCV via Alpaca StockBarsRequest.
        Returns DataFrame: Date, Close, Volume
        """
        # Read from local cache first
        if os.path.exists(self.hist_cache_file):
            print(f"[+] Found local OHLCV cache: {self.hist_cache_file}")
            hist = pd.read_csv(self.hist_cache_file)
            if not hist.empty:
                print(f"    Cached {len(hist)} rows ({hist['Date'].min()} ~ {hist['Date'].max()})")
                return hist

        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        print(f"[-] Fetching {self.ticker} daily bars from Alpaca ({start_date} ~ {end_date})...")

        try:
            request = StockBarsRequest(
                symbol_or_symbols=self.ticker,
                timeframe=TimeFrame.Day,
                start=datetime.strptime(start_date, "%Y-%m-%d"),
                end=datetime.strptime(end_date, "%Y-%m-%d"),
            )
            bars = self.alpaca.get_stock_bars(request)
            bars_df = bars.df  # MultiIndex: (symbol, timestamp)

            if bars_df.empty:
                print("[-] Alpaca returned empty data")
                return pd.DataFrame(columns=["Date", "Close", "Volume"])

            # Reset index, extract Date
            bars_df = bars_df.reset_index()
            hist = pd.DataFrame({
                "Date": pd.to_datetime(bars_df["timestamp"]).dt.strftime("%Y-%m-%d"),
                "Close": bars_df["close"],
                "Volume": bars_df["volume"],
            }).drop_duplicates(subset=["Date"]).sort_values("Date")

            hist.to_csv(self.hist_cache_file, index=False)
            print(f"[+] Historical OHLCV: {len(hist)} rows ({hist['Date'].min()} ~ {hist['Date'].max()}), cached")
            return hist

        except Exception as e:
            print(f"[-] Alpaca Bars API error: {e}")
            return pd.DataFrame(columns=["Date", "Close", "Volume"])

    # ==========================================
    # 2. PE Fundamentals — Finnhub /stock/metric (historical + current)
    # ==========================================
    def fetch_pe_ratio(self):
        """Fetch current PE (TTM) snapshot — used as fallback for incremental updates"""
        print(f"[-] Fetching {self.ticker} current PE from Finnhub...")
        data = self._finnhub_get("/stock/metric", params={
            "symbol": self.ticker,
            "metric": "all"
        }, description="PE fundamentals")

        metric = data.get("metric", {})
        pe = metric.get("peBasicExclExtraTTM")
        if pe is None:
            pe = metric.get("peTTM")
        if pe is None:
            pe = metric.get("peExclExtraTTM", np.nan)

        print(f"[+] PE_Ratio (TTM) = {pe}")
        return float(pe) if pe is not None else np.nan

    def fetch_historical_pe(self, hist_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute daily historical PE:
        1. Fetch quarterly EPS from Finnhub /stock/metric series (115+ quarters)
        2. Rolling 4-quarter sum -> TTM EPS
        3. Forward-fill to each trading day
        4. PE_daily = Close / EPS_TTM

        Args: hist_df must contain Date, Close columns
        Returns: hist_df with PE_Ratio column appended
        """
        print(f"[-] Fetching {self.ticker} quarterly EPS history from Finnhub...")
        data = self._finnhub_get("/stock/metric", params={
            "symbol": self.ticker,
            "metric": "all"
        }, description="quarterly EPS series")

        series = data.get("series", {})
        quarterly = series.get("quarterly", {})
        eps_list = quarterly.get("eps", [])

        if not eps_list:
            print("[-] No quarterly EPS data available, falling back to current PE snapshot")
            snapshot_pe = self.fetch_pe_ratio()
            hist_df = hist_df.copy()
            hist_df["PE_Ratio"] = snapshot_pe
            return hist_df

        # Build quarterly EPS DataFrame, sorted by date ascending
        eps_df = pd.DataFrame(eps_list)  # columns: period, v
        eps_df = eps_df.rename(columns={"period": "Quarter_End", "v": "EPS"})
        eps_df["Quarter_End"] = pd.to_datetime(eps_df["Quarter_End"])
        eps_df = eps_df.sort_values("Quarter_End").reset_index(drop=True)

        # Rolling 4-quarter sum -> TTM EPS
        eps_df["EPS_TTM"] = eps_df["EPS"].rolling(window=4, min_periods=4).sum()
        eps_df = eps_df.dropna(subset=["EPS_TTM"])
        eps_df["Quarter_End_str"] = eps_df["Quarter_End"].dt.strftime("%Y-%m-%d")

        print(f"[+] Quarterly EPS: {len(eps_list)} quarters, TTM EPS available: {len(eps_df)}")
        print(f"    Latest TTM EPS = {eps_df['EPS_TTM'].iloc[-1]:.4f} (as of {eps_df['Quarter_End_str'].iloc[-1]})")

        # Forward-fill TTM EPS to each trading day
        hist_df = hist_df.copy()
        hist_df["Date_dt"] = pd.to_datetime(hist_df["Date"])

        def get_ttm_eps(date_val):
            """Find the most recent quarterly TTM EPS <= date_val"""
            mask = eps_df["Quarter_End"] <= date_val
            if mask.any():
                return eps_df.loc[mask, "EPS_TTM"].iloc[-1]
            return np.nan

        hist_df["EPS_TTM"] = hist_df["Date_dt"].apply(get_ttm_eps)
        hist_df["PE_Ratio"] = np.where(
            hist_df["EPS_TTM"] > 0,
            hist_df["Close"] / hist_df["EPS_TTM"],
            np.nan
        )
        hist_df["PE_Ratio"] = hist_df["PE_Ratio"].round(4)

        # Drop temporary columns
        hist_df = hist_df.drop(columns=["Date_dt", "EPS_TTM"])

        pe_min = hist_df["PE_Ratio"].min()
        pe_max = hist_df["PE_Ratio"].max()
        print(f"[+] Daily historical PE computed: range {pe_min:.2f} ~ {pe_max:.2f}")
        return hist_df

    # ==========================================
    # 3. News — Alpaca News API (v1beta1, auto-pagination)
    # ==========================================
    def fetch_news(self, start_date: str = None, end_date: str = None):
        """
        Fetch company news via Alpaca v1beta1/news (Benzinga data source).
        Free tier: max 50 per request, supports page_token pagination.
        Returns DataFrame: Date, Publisher, Headline
        """
        # Read from local cache first
        if os.path.exists(self.news_cache_file):
            print(f"[+] Found local news cache: {self.news_cache_file}")
            news_df = pd.read_csv(self.news_cache_file)
            if not news_df.empty:
                print(f"    Cached {len(news_df)} articles")
                return news_df

        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        print(f"[-] Fetching {self.ticker} news from Alpaca News API ({start_date} ~ {end_date})...")

        news_url = "https://data.alpaca.markets/v1beta1/news"
        headers = {
            "APCA-API-KEY-ID": self.alpaca_key,
            "APCA-API-SECRET-KEY": self.alpaca_secret,
        }

        all_news = []
        page_token = None
        page = 0
        MAX_PAGES = 200  # Safety cap: 200 * 50 = 10,000 articles

        while page < MAX_PAGES:
            params = {
                "symbols": self.ticker,
                "start": f"{start_date}T00:00:00Z",
                "end": f"{end_date}T23:59:59Z",
                "limit": 50,
                "sort": "desc",
            }
            if page_token:
                params["page_token"] = page_token

            try:
                resp = requests.get(news_url, headers=headers, params=params, timeout=15)
                if resp.status_code == 429:
                    print(f"  Rate-limited, waiting 5s...")
                    time.sleep(5)
                    continue
                resp.raise_for_status()
                data = resp.json()

                articles = data.get("news", [])
                if not articles:
                    break

                for a in articles:
                    created = a.get("created_at", "")[:10]  # "2025-03-15T..."
                    all_news.append({
                        "Date": created,
                        "Publisher": a.get("source", "Unknown"),
                        "Headline": a.get("headline", ""),
                    })

                page_token = data.get("next_page_token")
                if not page_token:
                    break

                page += 1
                if page % 10 == 0:
                    print(f"  Fetched {len(all_news)} articles (page {page})...")
                time.sleep(0.2)  # Polite delay

            except Exception as e:
                print(f"  News API error: {e}")
                break

        if not all_news:
            print("[-] No news articles retrieved")
            return pd.DataFrame(columns=["Date", "Publisher", "Headline"])

        news_df = pd.DataFrame(all_news)
        news_df = news_df.drop_duplicates(subset=["Date", "Publisher", "Headline"])
        news_df = news_df.sort_values("Date", ascending=False)

        news_df.to_csv(self.news_cache_file, index=False)
        print(f"[+] News fetched: {len(news_df)} articles, cached to {self.news_cache_file}")
        return news_df

    # ==========================================
    # 4. Main Pipeline: Data Fusion + Output
    # ==========================================
    def build_dataset(self, news_start: str = None):
        """
        Build complete feature matrix:
        1. Historical OHLCV (Alpaca Bars)
        2. PE (Finnhub quarterly EPS)
        3. News (Alpaca News)
        4. Inner join on Date
        5. Output CSV
        """
        one_year_ago = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        if news_start is None:
            news_start = one_year_ago

        print(f"\n{'='*60}")
        print(f"  Alpaca + Finnhub Pipeline — {self.ticker}")
        print(f"{'='*60}")

        # Step 1: Historical OHLCV
        hist = self.fetch_history(start_date=one_year_ago)
        if hist.empty:
            print("[FATAL] Failed to fetch historical OHLCV, exiting")
            sys.exit(1)

        # Step 2: Compute daily historical PE (from quarterly EPS)
        hist = self.fetch_historical_pe(hist)

        # Step 3: News
        news_df = self.fetch_news(start_date=news_start)
        if news_df.empty:
            print("[WARNING] No news data, saving OHLCV+PE only")
            result = hist.copy()
            result["Ticker"] = self.ticker
            result["Publisher"] = ""
            result["Headline"] = ""
        else:
            # Step 4: inner join (hist already contains PE_Ratio)
            result = pd.merge(news_df, hist[["Date", "Close", "Volume", "PE_Ratio"]], on="Date", how="inner")
            result["Ticker"] = self.ticker

        # Standardize column order
        result = result[["Date", "Ticker", "Close", "Volume", "PE_Ratio", "Publisher", "Headline"]]
        result = result.sort_values("Date", ascending=False)

        # Step 5: Save
        result.to_csv(self.cache_file, index=False)
        print(f"\n[+] Dataset saved: {self.cache_file} ({len(result)} rows)")
        print(f"    Date range: {result['Date'].min()} ~ {result['Date'].max()}")

        return result

    # ==========================================
    # 5. Incremental Update (last 7 days)
    # ==========================================
    def update(self):
        """Incremental update: fetch last 7 days of OHLCV and news, append to existing cache"""
        if not os.path.exists(self.cache_file):
            print("[-] No local cache found, running full build...")
            return self.build_dataset()

        print(f"\n--- Incremental sync {self.ticker} (Alpaca + Finnhub) ---")

        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

        # Last 7 days OHLCV (Alpaca)
        try:
            request = StockBarsRequest(
                symbol_or_symbols=self.ticker,
                timeframe=TimeFrame.Day,
                start=datetime.strptime(start, "%Y-%m-%d"),
                end=datetime.strptime(end, "%Y-%m-%d"),
            )
            bars = self.alpaca.get_stock_bars(request)
            bars_df = bars.df.reset_index()
            hist = pd.DataFrame({
                "Date": pd.to_datetime(bars_df["timestamp"]).dt.strftime("%Y-%m-%d"),
                "Close": bars_df["close"],
                "Volume": bars_df["volume"],
            }).drop_duplicates(subset=["Date"])
        except Exception as e:
            print(f"[-] Alpaca incremental OHLCV failed: {e}")
            hist = pd.DataFrame()

        # Last 7 days news (Alpaca)
        news_url = "https://data.alpaca.markets/v1beta1/news"
        headers = {
            "APCA-API-KEY-ID": self.alpaca_key,
            "APCA-API-SECRET-KEY": self.alpaca_secret,
        }
        params = {
            "symbols": self.ticker,
            "start": f"{start}T00:00:00Z",
            "end": f"{end}T23:59:59Z",
            "limit": 50,
            "sort": "desc",
        }
        news_records = []
        try:
            resp = requests.get(news_url, headers=headers, params=params, timeout=15)
            resp.raise_for_status()
            for a in resp.json().get("news", []):
                news_records.append({
                    "Date": a.get("created_at", "")[:10],
                    "Publisher": a.get("source", "Unknown"),
                    "Headline": a.get("headline", ""),
                })
        except Exception as e:
            print(f"[-] Alpaca incremental news failed: {e}")

        news_df = pd.DataFrame(news_records) if news_records else pd.DataFrame()

        if news_df.empty:
            print("[-] No new news data")
            return pd.read_csv(self.cache_file)

        # PE: compute historical PE (same method for consistency)
        if not hist.empty:
            hist = self.fetch_historical_pe(hist)

        # Merge
        if not hist.empty:
            new_data = pd.merge(news_df, hist[["Date", "Close", "Volume", "PE_Ratio"]], on="Date", how="inner")
        else:
            new_data = news_df.copy()
            new_data["Close"] = 0.0
            new_data["Volume"] = 0
            new_data["PE_Ratio"] = self.fetch_pe_ratio()  # fallback: snapshot

        new_data["Ticker"] = self.ticker
        new_data = new_data[["Date", "Ticker", "Close", "Volume", "PE_Ratio", "Publisher", "Headline"]]

        # Append and deduplicate
        old_data = pd.read_csv(self.cache_file)
        combined = pd.concat([old_data, new_data], ignore_index=True)
        combined = combined.drop_duplicates(subset=["Date", "Ticker", "Publisher", "Headline"])
        combined.to_csv(self.cache_file, index=False)
        print(f"[+] Incremental update complete, total dataset size: {len(combined)} rows")
        return combined


import argparse

# ... (rest of the imports)

# ==========================================
# Entry Point
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alpaca + Finnhub Hybrid Data Pipeline")
    parser.add_argument(
        "tickers",
        nargs="*",
        help="Stock tickers to process (e.g., NVDA GOOGL). If omitted, you will be prompted.",
    )
    args = parser.parse_args()

    # API Keys (Ideally these should be in environment variables)
    ALPACA_KEY    = "PK6FYKALELA4WL7K47AD2UC626"
    ALPACA_SECRET = "A5mWQGEDL2uwEtWGVkuRkVCAAnp1qGKnQJ5e5osoRQ64"
    FINNHUB_KEY   = "d6hksupr01qr5k4ccku0d6hksupr01qr5k4cckug"

    # Determine tickers to process
    tickers = args.tickers
    if not tickers:
        user_input = input("Enter ticker symbols separated by spaces (e.g., NVDA GOOGL): ").strip()
        if user_input:
            tickers = [t.strip().upper() for t in user_input.split()]
        else:
            print("[-] No tickers provided. Exiting.")
            sys.exit(0)

    for ticker in tickers:
        print(f"\n{'#'*60}")
        print(f"  Processing Ticker: {ticker}")
        print(f"{'#'*60}")

        cache_file = f"dataset/real_{ticker.lower()}_dataset.csv"
        
        pipeline = AlpacaFinnhubPipeline(
            alpaca_key=ALPACA_KEY,
            alpaca_secret=ALPACA_SECRET,
            finnhub_key=FINNHUB_KEY,
            ticker=ticker,
            cache_file=cache_file,
        )

        # First run: build last 1 year dataset
        # Subsequent runs: incremental update (last 7 days)
        if os.path.exists(pipeline.cache_file):
            df = pipeline.update()
        else:
            df = pipeline.build_dataset()

        print(f"\n--- Feature Matrix (Top 5 Rows for {ticker}) ---")
        print(df.head(5).to_string(index=False))
