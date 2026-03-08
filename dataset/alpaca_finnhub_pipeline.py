"""
Alpaca + Finnhub Hybrid Data Pipeline v2
=========================================
Data Sources:
 - Alpaca:   Historical daily OHLCV (bars) + News (Benzinga)
 - Finnhub:  PE fundamentals (/stock/metric) + Company News (Yahoo, SeekingAlpha, CNBC, etc.)

Output (split datasets, date-aligned):
 - {TICKER}_market.csv : Date, Ticker, Close, Volume, PE_Ratio
 - {TICKER}_news.csv   : Date, Ticker, Publisher, Headline, Summary
Both share the same date range (inner join on Date).
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
    ):
        self.ticker = ticker

        # ── Output paths (split datasets) ──
        self.market_file = f"dataset/{ticker}_market.csv"
        self.news_file = f"dataset/{ticker}_news.csv"

        # ── Cache paths (raw API responses) ──
        self.hist_cache_file = f"dataset/{ticker}_hist_cache.csv"
        self.alpaca_news_cache = f"dataset/{ticker}_alpaca_news.csv"
        self.finnhub_news_cache = f"dataset/{ticker}_finnhub_news.csv"

        # ── Alpaca client ──
        self.alpaca = StockHistoricalDataClient(alpaca_key, alpaca_secret)
        self.alpaca_key = alpaca_key
        self.alpaca_secret = alpaca_secret

        # ── Finnhub session ──
        self.fh_session = requests.Session()
        self.fh_session.params = {"token": finnhub_key}
        self.finnhub_key = finnhub_key

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
                print(f"  [Retry {attempt}/3] Finnhub {description} rate-limited, waiting {delay}s...")
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
        if os.path.exists(self.hist_cache_file):
            print(f"  [cache] OHLCV: {self.hist_cache_file}")
            hist = pd.read_csv(self.hist_cache_file)
            if not hist.empty:
                print(f"    {len(hist)} rows ({hist['Date'].min()} ~ {hist['Date'].max()})")
                return hist

        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        print(f"  [fetch] Alpaca bars ({start_date} ~ {end_date})...")

        try:
            request = StockBarsRequest(
                symbol_or_symbols=self.ticker,
                timeframe=TimeFrame.Day,
                start=datetime.strptime(start_date, "%Y-%m-%d"),
                end=datetime.strptime(end_date, "%Y-%m-%d"),
            )
            bars = self.alpaca.get_stock_bars(request)
            bars_df = bars.df.reset_index()

            if bars_df.empty:
                print("  [warn] Alpaca returned empty data")
                return pd.DataFrame(columns=["Date", "Close", "Volume"])

            hist = pd.DataFrame({
                "Date": pd.to_datetime(bars_df["timestamp"]).dt.strftime("%Y-%m-%d"),
                "Close": bars_df["close"],
                "Volume": bars_df["volume"],
            }).drop_duplicates(subset=["Date"]).sort_values("Date")

            hist.to_csv(self.hist_cache_file, index=False)
            print(f"  [done] OHLCV: {len(hist)} rows, cached")
            return hist

        except Exception as e:
            print(f"  [error] Alpaca Bars API: {e}")
            return pd.DataFrame(columns=["Date", "Close", "Volume"])

    # ==========================================
    # 2. PE Fundamentals — Finnhub
    # ==========================================
    def fetch_pe_ratio(self):
        """Fetch current PE (TTM) snapshot — fallback only"""
        data = self._finnhub_get("/stock/metric", params={
            "symbol": self.ticker,
            "metric": "all"
        }, description="PE snapshot")
        metric = data.get("metric", {})
        pe = metric.get("peBasicExclExtraTTM") or metric.get("peTTM") or metric.get("peExclExtraTTM", np.nan)
        return float(pe) if pe is not None else np.nan

    def fetch_historical_pe(self, hist_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute daily historical PE from quarterly EPS:
        1. Fetch quarterly EPS from Finnhub
        2. Rolling 4-quarter sum -> TTM EPS
        3. Forward-fill to each trading day
        4. PE = Close / EPS_TTM
        """
        print(f"  [fetch] Finnhub quarterly EPS...")
        data = self._finnhub_get("/stock/metric", params={
            "symbol": self.ticker,
            "metric": "all"
        }, description="quarterly EPS")

        series = data.get("series", {})
        eps_list = series.get("quarterly", {}).get("eps", [])

        if not eps_list:
            print("  [warn] No quarterly EPS data, using snapshot PE")
            hist_df = hist_df.copy()
            hist_df["PE_Ratio"] = self.fetch_pe_ratio()
            return hist_df

        eps_df = pd.DataFrame(eps_list).rename(columns={"period": "Quarter_End", "v": "EPS"})
        eps_df["Quarter_End"] = pd.to_datetime(eps_df["Quarter_End"])
        eps_df = eps_df.sort_values("Quarter_End").reset_index(drop=True)
        eps_df["EPS_TTM"] = eps_df["EPS"].rolling(window=4, min_periods=4).sum()
        eps_df = eps_df.dropna(subset=["EPS_TTM"])

        print(f"  [done] {len(eps_list)} quarters, TTM EPS = {eps_df['EPS_TTM'].iloc[-1]:.4f}")

        hist_df = hist_df.copy()
        hist_df["Date_dt"] = pd.to_datetime(hist_df["Date"])

        def get_ttm_eps(date_val):
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
        hist_df = hist_df.drop(columns=["Date_dt", "EPS_TTM"])

        print(f"  [done] PE range: {hist_df['PE_Ratio'].min():.2f} ~ {hist_df['PE_Ratio'].max():.2f}")
        return hist_df

    # ==========================================
    # 3a. News — Alpaca News API (Benzinga source)
    # ==========================================
    def fetch_alpaca_news(self, start_date: str = None, end_date: str = None):
        """
        Fetch news via Alpaca v1beta1/news (Benzinga).
        Returns DataFrame: Date, Publisher, Headline, Summary
        """
        if os.path.exists(self.alpaca_news_cache):
            print(f"  [cache] Alpaca news: {self.alpaca_news_cache}")
            news_df = pd.read_csv(self.alpaca_news_cache)
            if not news_df.empty:
                print(f"    {len(news_df)} articles")
                return news_df

        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        print(f"  [fetch] Alpaca news ({start_date} ~ {end_date})...")

        news_url = "https://data.alpaca.markets/v1beta1/news"
        headers = {
            "APCA-API-KEY-ID": self.alpaca_key,
            "APCA-API-SECRET-KEY": self.alpaca_secret,
        }

        all_news = []
        page_token = None
        page = 0
        MAX_PAGES = 200

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
                    print(f"    Rate-limited, waiting 5s...")
                    time.sleep(5)
                    continue
                resp.raise_for_status()
                data = resp.json()

                articles = data.get("news", [])
                if not articles:
                    break

                for a in articles:
                    all_news.append({
                        "Date": a.get("created_at", "")[:10],
                        "Publisher": a.get("source", "Unknown"),
                        "Headline": a.get("headline", ""),
                        "Summary": a.get("summary", ""),
                    })

                page_token = data.get("next_page_token")
                if not page_token:
                    break

                page += 1
                if page % 10 == 0:
                    print(f"    {len(all_news)} articles (page {page})...")
                time.sleep(0.2)

            except Exception as e:
                print(f"    Alpaca news error: {e}")
                break

        if not all_news:
            return pd.DataFrame(columns=["Date", "Publisher", "Headline", "Summary"])

        news_df = pd.DataFrame(all_news)
        news_df = news_df.drop_duplicates(subset=["Date", "Publisher", "Headline"])
        news_df = news_df.sort_values("Date", ascending=False)

        news_df.to_csv(self.alpaca_news_cache, index=False)
        print(f"  [done] Alpaca news: {len(news_df)} articles, cached")
        return news_df

    # ==========================================
    # 3b. News — Finnhub /company-news (multi-publisher)
    # ==========================================
    def fetch_finnhub_news(self, start_date: str = None, end_date: str = None):
        """
        Fetch news via Finnhub /company-news endpoint.
        Returns diverse publishers: Yahoo, SeekingAlpha, MarketWatch, CNBC, etc.
        Finnhub free tier: 1 year of news, paginated by date chunks.
        """
        if os.path.exists(self.finnhub_news_cache):
            print(f"  [cache] Finnhub news: {self.finnhub_news_cache}")
            news_df = pd.read_csv(self.finnhub_news_cache)
            if not news_df.empty:
                print(f"    {len(news_df)} articles")
                return news_df

        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        print(f"  [fetch] Finnhub company news ({start_date} ~ {end_date})...")

        # Finnhub /company-news returns max ~100 articles per call.
        # To get full coverage, we chunk by month.
        all_news = []
        chunk_start = datetime.strptime(start_date, "%Y-%m-%d")
        chunk_end_limit = datetime.strptime(end_date, "%Y-%m-%d")

        while chunk_start < chunk_end_limit:
            chunk_end = min(chunk_start + timedelta(days=30), chunk_end_limit)
            cs = chunk_start.strftime("%Y-%m-%d")
            ce = chunk_end.strftime("%Y-%m-%d")

            try:
                articles = self._finnhub_get(
                    "/company-news",
                    params={"symbol": self.ticker, "from": cs, "to": ce},
                    description=f"news {cs}~{ce}"
                )

                if isinstance(articles, list):
                    for a in articles:
                        dt = a.get("datetime")
                        if dt:
                            date_str = datetime.fromtimestamp(dt).strftime("%Y-%m-%d")
                        else:
                            continue

                        headline = a.get("headline", "").strip()
                        source = a.get("source", "Unknown").strip()

                        if headline:
                            all_news.append({
                                "Date": date_str,
                                "Publisher": source,
                                "Headline": headline,
                                "Summary": a.get("summary", "").strip(),
                            })

                count = len(articles) if isinstance(articles, list) else 0
                print(f"    {cs} ~ {ce}: {count} articles")

            except Exception as e:
                print(f"    Finnhub news error ({cs}~{ce}): {e}")

            chunk_start = chunk_end + timedelta(days=1)
            time.sleep(1)  # Finnhub free tier: 60 req/min

        if not all_news:
            return pd.DataFrame(columns=["Date", "Publisher", "Headline", "Summary"])

        news_df = pd.DataFrame(all_news)
        news_df = news_df.drop_duplicates(subset=["Date", "Publisher", "Headline"])
        news_df = news_df.sort_values("Date", ascending=False)

        news_df.to_csv(self.finnhub_news_cache, index=False)
        print(f"  [done] Finnhub news: {len(news_df)} articles, cached")
        return news_df

    # ==========================================
    # 3c. Merge News from Both Sources
    # ==========================================
    def fetch_all_news(self, start_date: str = None, end_date: str = None):
        """
        Combine Alpaca (Benzinga) + Finnhub (Yahoo, SeekingAlpha, CNBC, etc.)
        Deduplicate by (Date, Headline) to avoid exact duplicates across sources.
        """
        alpaca_news = self.fetch_alpaca_news(start_date, end_date)
        finnhub_news = self.fetch_finnhub_news(start_date, end_date)

        combined = pd.concat([alpaca_news, finnhub_news], ignore_index=True)

        # Deduplicate: same headline on same day (even from different publishers)
        combined = combined.drop_duplicates(subset=["Date", "Headline"])
        combined = combined.sort_values("Date", ascending=False).reset_index(drop=True)

        publishers = combined["Publisher"].value_counts()
        print(f"\n  [summary] Combined news: {len(combined)} unique articles")
        print(f"  Publisher distribution:")
        for pub, count in publishers.head(10).items():
            print(f"    {pub:20s} : {count}")

        return combined

    # ==========================================
    # 4. Main Pipeline: Build Split Datasets
    # ==========================================
    def build_dataset(self, news_start: str = None):
        """
        Build two date-aligned datasets:
         1. {TICKER}_market.csv : Date, Ticker, Close, Volume, PE_Ratio
         2. {TICKER}_news.csv   : Date, Ticker, Publisher, Headline, Summary

        Both share the same set of dates (inner join).
        """
        one_year_ago = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        if news_start is None:
            news_start = one_year_ago

        print(f"\n{'='*60}")
        print(f"  Pipeline v2 — {self.ticker}")
        print(f"{'='*60}")

        # ── Step 1: Historical OHLCV ──
        print("\n[1/4] Historical OHLCV (Alpaca)")
        hist = self.fetch_history(start_date=one_year_ago)
        if hist.empty:
            print("[FATAL] Failed to fetch OHLCV, exiting")
            sys.exit(1)

        # ── Step 2: Historical PE ──
        print("\n[2/4] PE Ratio (Finnhub)")
        hist = self.fetch_historical_pe(hist)

        # ── Step 3: News (Alpaca + Finnhub combined) ──
        print("\n[3/4] News (Alpaca + Finnhub)")
        news_df = self.fetch_all_news(start_date=news_start)

        if news_df.empty:
            print("[WARNING] No news data available")
            hist["Ticker"] = self.ticker
            market = hist[["Date", "Ticker", "Close", "Volume", "PE_Ratio"]].copy()
            market = market.sort_values("Date", ascending=False)
            market.to_csv(self.market_file, index=False)
            print(f"\n[+] Market data saved: {self.market_file} ({len(market)} rows)")
            print("[!] No news file generated (no news data)")
            return market, pd.DataFrame()

        # ── Step 4: Date alignment (inner join) ──
        print("\n[4/4] Date alignment & output")

        market_dates = set(hist["Date"].unique())
        news_dates = set(news_df["Date"].unique())
        common_dates = market_dates & news_dates

        print(f"  Market trading days : {len(market_dates)}")
        print(f"  News coverage days  : {len(news_dates)}")
        print(f"  Aligned dates       : {len(common_dates)}")

        # Filter both datasets to common dates only
        market = hist[hist["Date"].isin(common_dates)].copy()
        market["Ticker"] = self.ticker
        market = market[["Date", "Ticker", "Close", "Volume", "PE_Ratio"]]
        market = market.drop_duplicates(subset=["Date"]).sort_values("Date", ascending=False)

        news = news_df[news_df["Date"].isin(common_dates)].copy()
        news["Ticker"] = self.ticker
        news = news[["Date", "Ticker", "Publisher", "Headline", "Summary"]]
        news = news.sort_values("Date", ascending=False).reset_index(drop=True)

        # Save
        market.to_csv(self.market_file, index=False)
        news.to_csv(self.news_file, index=False)

        print(f"\n  Market data : {self.market_file} ({len(market)} rows)")
        print(f"  News data   : {self.news_file} ({len(news)} rows)")
        print(f"  Date range  : {market['Date'].min()} ~ {market['Date'].max()}")

        return market, news

    # ==========================================
    # 5. Incremental Update (last 7 days)
    # ==========================================
    def update(self):
        """Incremental update: fetch last 7 days, append to existing datasets"""
        if not os.path.exists(self.market_file):
            print("[-] No existing dataset found, running full build...")
            return self.build_dataset()

        print(f"\n--- Incremental sync {self.ticker} ---")

        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

        # ── Fetch last 7 days OHLCV ──
        try:
            request = StockBarsRequest(
                symbol_or_symbols=self.ticker,
                timeframe=TimeFrame.Day,
                start=datetime.strptime(start, "%Y-%m-%d"),
                end=datetime.strptime(end, "%Y-%m-%d"),
            )
            bars = self.alpaca.get_stock_bars(request)
            bars_df = bars.df.reset_index()
            new_hist = pd.DataFrame({
                "Date": pd.to_datetime(bars_df["timestamp"]).dt.strftime("%Y-%m-%d"),
                "Close": bars_df["close"],
                "Volume": bars_df["volume"],
            }).drop_duplicates(subset=["Date"])
        except Exception as e:
            print(f"  [error] Alpaca incremental OHLCV: {e}")
            new_hist = pd.DataFrame()

        # ── Fetch last 7 days news (both sources) ──
        # Alpaca news (fresh, no cache)
        alpaca_records = []
        news_url = "https://data.alpaca.markets/v1beta1/news"
        headers = {
            "APCA-API-KEY-ID": self.alpaca_key,
            "APCA-API-SECRET-KEY": self.alpaca_secret,
        }
        try:
            resp = requests.get(news_url, headers=headers, params={
                "symbols": self.ticker,
                "start": f"{start}T00:00:00Z",
                "end": f"{end}T23:59:59Z",
                "limit": 50, "sort": "desc",
            }, timeout=15)
            resp.raise_for_status()
            for a in resp.json().get("news", []):
                alpaca_records.append({
                    "Date": a.get("created_at", "")[:10],
                    "Publisher": a.get("source", "Unknown"),
                    "Headline": a.get("headline", ""),
                    "Summary": a.get("summary", ""),
                })
        except Exception as e:
            print(f"  [error] Alpaca incremental news: {e}")

        # Finnhub news (last 7 days)
        finnhub_records = []
        try:
            articles = self._finnhub_get(
                "/company-news",
                params={"symbol": self.ticker, "from": start, "to": end},
                description=f"incremental news {start}~{end}"
            )
            if isinstance(articles, list):
                for a in articles:
                    dt = a.get("datetime")
                    if dt:
                        date_str = datetime.fromtimestamp(dt).strftime("%Y-%m-%d")
                        headline = a.get("headline", "").strip()
                        source = a.get("source", "Unknown").strip()
                        if headline:
                            finnhub_records.append({
                                "Date": date_str,
                                "Publisher": source,
                                "Headline": headline,
                                "Summary": a.get("summary", "").strip(),
                            })
        except Exception as e:
            print(f"  [error] Finnhub incremental news: {e}")

        new_news = pd.DataFrame(alpaca_records + finnhub_records)
        if not new_news.empty:
            new_news = new_news.drop_duplicates(subset=["Date", "Headline"])

        if new_news.empty and new_hist.empty:
            print("  No new data to add")
            return pd.read_csv(self.market_file), pd.read_csv(self.news_file)

        # ── PE for new hist rows ──
        if not new_hist.empty:
            new_hist = self.fetch_historical_pe(new_hist)

        # ── Merge incremental data ──
        old_market = pd.read_csv(self.market_file)
        old_news = pd.read_csv(self.news_file)

        if not new_hist.empty:
            new_hist["Ticker"] = self.ticker
            new_market_rows = new_hist[["Date", "Ticker", "Close", "Volume", "PE_Ratio"]]
            combined_market = pd.concat([old_market, new_market_rows], ignore_index=True)
            combined_market = combined_market.drop_duplicates(subset=["Date"])
        else:
            combined_market = old_market

        if not new_news.empty:
            new_news["Ticker"] = self.ticker
            new_news_rows = new_news[["Date", "Ticker", "Publisher", "Headline", "Summary"]]
            combined_news = pd.concat([old_news, new_news_rows], ignore_index=True)
            combined_news = combined_news.drop_duplicates(subset=["Date", "Headline"])
        else:
            combined_news = old_news

        # Re-align dates
        common_dates = set(combined_market["Date"].unique()) & set(combined_news["Date"].unique())
        combined_market = combined_market[combined_market["Date"].isin(common_dates)]
        combined_news = combined_news[combined_news["Date"].isin(common_dates)]

        combined_market = combined_market.sort_values("Date", ascending=False)
        combined_news = combined_news.sort_values("Date", ascending=False)

        combined_market.to_csv(self.market_file, index=False)
        combined_news.to_csv(self.news_file, index=False)

        print(f"  [done] Market: {len(combined_market)} rows, News: {len(combined_news)} rows")
        return combined_market, combined_news


# ==========================================
# Entry Point
# ==========================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Alpaca + Finnhub Hybrid Data Pipeline v2")
    parser.add_argument(
        "tickers",
        nargs="*",
        help="Stock tickers to process (e.g., NVDA GOOGL MSFT)",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force full rebuild even if datasets already exist",
    )
    args = parser.parse_args()

    # API Keys
    ALPACA_KEY    = "PK6FYKALELA4WL7K47AD2UC626"
    ALPACA_SECRET = "A5mWQGEDL2uwEtWGVkuRkVCAAnp1qGKnQJ5e5osoRQ64"
    FINNHUB_KEY   = "d6hksupr01qr5k4ccku0d6hksupr01qr5k4cckug"

    # Determine tickers
    tickers = args.tickers
    if not tickers:
        user_input = input("Enter ticker symbols separated by spaces (e.g., NVDA GOOGL MSFT): ").strip()
        if user_input:
            tickers = [t.strip().upper() for t in user_input.split()]
        else:
            print("[-] No tickers provided. Exiting.")
            sys.exit(0)

    for ticker in tickers:
        print(f"\n{'#'*60}")
        print(f"  Processing: {ticker}")
        print(f"{'#'*60}")

        pipeline = AlpacaFinnhubPipeline(
            alpaca_key=ALPACA_KEY,
            alpaca_secret=ALPACA_SECRET,
            finnhub_key=FINNHUB_KEY,
            ticker=ticker,
        )

        if args.force_rebuild or not os.path.exists(pipeline.market_file):
            market, news = pipeline.build_dataset()
        else:
            market, news = pipeline.update()

        print(f"\n--- {ticker} Market Data (Top 5) ---")
        if isinstance(market, pd.DataFrame) and not market.empty:
            print(market.head(5).to_string(index=False))

        print(f"\n--- {ticker} News Data (Top 5) ---")
        if isinstance(news, pd.DataFrame) and not news.empty:
            print(news.head(5).to_string(index=False))
