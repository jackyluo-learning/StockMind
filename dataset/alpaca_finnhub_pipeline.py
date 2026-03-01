"""
Alpaca + Finnhub 混合数据管道
 - Alpaca: 历史日级别量价 (bars)、新闻 (news)  ← 稳定，无 429 问题
 - Finnhub: PE 基本面 (/stock/metric)
输出格式: Date, Ticker, Close, Volume, PE_Ratio, Publisher, Headline
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
        cache_file: str = "real_nvda_dataset.csv",
    ):
        self.ticker = ticker
        self.cache_file = cache_file
        self.hist_cache_file = f"dataset/{ticker}_hist_cache.csv"
        self.news_cache_file = f"dataset/{ticker}_alpaca_news.csv"

        # ── Alpaca client (paper-trading key 即可获取行情) ──
        self.alpaca = StockHistoricalDataClient(alpaca_key, alpaca_secret)
        self.alpaca_key = alpaca_key
        self.alpaca_secret = alpaca_secret

        # ── Finnhub session (仅用于 PE) ──
        self.fh_session = requests.Session()
        self.fh_session.params = {"token": finnhub_key}

    # ==========================================
    # 通用工具
    # ==========================================
    def _finnhub_get(self, endpoint, params=None, description="数据"):
        """Finnhub GET with retry (免费版 60 req/min)"""
        url = f"{self.FINNHUB_URL}{endpoint}"
        for attempt in range(1, 4):
            resp = self.fh_session.get(url, params=params, timeout=15)
            if resp.status_code == 429:
                delay = 5 * (2 ** (attempt - 1))
                print(f"[Retry {attempt}/3] Finnhub 获取{description}被限流，等待 {delay}s...")
                time.sleep(delay)
                continue
            resp.raise_for_status()
            return resp.json()
        raise RuntimeError(f"Finnhub 获取{description}失败：重试 3 次后仍被限流")

    # ==========================================
    # 1. 历史量价 — Alpaca Bars API
    # ==========================================
    def fetch_history(self, start_date: str = None, end_date: str = None):
        """
        通过 Alpaca StockBarsRequest 获取日线历史量价
        返回 DataFrame: Date, Close, Volume
        """
        # 优先读缓存
        if os.path.exists(self.hist_cache_file):
            print(f"[+] 发现本地历史量价缓存: {self.hist_cache_file}")
            hist = pd.read_csv(self.hist_cache_file)
            if not hist.empty:
                print(f"    已缓存 {len(hist)} 条 ({hist['Date'].min()} ~ {hist['Date'].max()})")
                return hist

        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        print(f"[-] 正在从 Alpaca Bars API 获取 {self.ticker} 日线量价 ({start_date} ~ {end_date})...")

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
                print("[-] Alpaca 返回空数据")
                return pd.DataFrame(columns=["Date", "Close", "Volume"])

            # 重置索引，提取 Date
            bars_df = bars_df.reset_index()
            hist = pd.DataFrame({
                "Date": pd.to_datetime(bars_df["timestamp"]).dt.strftime("%Y-%m-%d"),
                "Close": bars_df["close"],
                "Volume": bars_df["volume"],
            }).drop_duplicates(subset=["Date"]).sort_values("Date")

            hist.to_csv(self.hist_cache_file, index=False)
            print(f"[+] 历史量价: {len(hist)} 条 ({hist['Date'].min()} ~ {hist['Date'].max()})，已缓存")
            return hist

        except Exception as e:
            print(f"[-] Alpaca Bars API 错误: {e}")
            return pd.DataFrame(columns=["Date", "Close", "Volume"])

    # ==========================================
    # 2. PE 基本面 — Finnhub /stock/metric (历史 + 当前)
    # ==========================================
    def fetch_pe_ratio(self):
        """获取当前 PE (TTM) — 快照，用于增量更新"""
        print(f"[-] 正在从 Finnhub 获取 {self.ticker} 当前 PE 指标...")
        data = self._finnhub_get("/stock/metric", params={
            "symbol": self.ticker,
            "metric": "all"
        }, description="PE 基本面")

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
        计算每日历史 PE:
        1. 从 Finnhub /stock/metric series 获取季度 EPS (115+ 个季度)
        2. 滚动 4 季求和 → TTM EPS
        3. Forward-fill 到每个交易日
        4. PE_daily = Close / EPS_TTM

        参数: hist_df 需包含 Date, Close 列
        返回: hist_df 附加 PE_Ratio 列
        """
        print(f"[-] 正在从 Finnhub 获取 {self.ticker} 季度 EPS 历史序列...")
        data = self._finnhub_get("/stock/metric", params={
            "symbol": self.ticker,
            "metric": "all"
        }, description="季度 EPS 序列")

        series = data.get("series", {})
        quarterly = series.get("quarterly", {})
        eps_list = quarterly.get("eps", [])

        if not eps_list:
            print("[-] 无季度 EPS 数据，降级使用当前快照 PE")
            snapshot_pe = self.fetch_pe_ratio()
            hist_df = hist_df.copy()
            hist_df["PE_Ratio"] = snapshot_pe
            return hist_df

        # 构建季度 EPS DataFrame，按日期升序
        eps_df = pd.DataFrame(eps_list)  # columns: period, v
        eps_df = eps_df.rename(columns={"period": "Quarter_End", "v": "EPS"})
        eps_df["Quarter_End"] = pd.to_datetime(eps_df["Quarter_End"])
        eps_df = eps_df.sort_values("Quarter_End").reset_index(drop=True)

        # 滚动 4 季求和 → TTM EPS
        eps_df["EPS_TTM"] = eps_df["EPS"].rolling(window=4, min_periods=4).sum()
        eps_df = eps_df.dropna(subset=["EPS_TTM"])
        eps_df["Quarter_End_str"] = eps_df["Quarter_End"].dt.strftime("%Y-%m-%d")

        print(f"[+] 季度 EPS: {len(eps_list)} 个季度，TTM EPS 可用: {len(eps_df)} 个")
        print(f"    最近 TTM EPS = {eps_df['EPS_TTM'].iloc[-1]:.4f} (截至 {eps_df['Quarter_End_str'].iloc[-1]})")

        # 将 TTM EPS forward-fill 到每个交易日
        hist_df = hist_df.copy()
        hist_df["Date_dt"] = pd.to_datetime(hist_df["Date"])

        def get_ttm_eps(date_val):
            """找到 ≤ date_val 的最近一个季度 TTM EPS"""
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

        # 清理临时列
        hist_df = hist_df.drop(columns=["Date_dt", "EPS_TTM"])

        pe_min = hist_df["PE_Ratio"].min()
        pe_max = hist_df["PE_Ratio"].max()
        print(f"[+] 每日历史 PE 计算完成: 范围 {pe_min:.2f} ~ {pe_max:.2f}")
        return hist_df

    # ==========================================
    # 3. 新闻 — Alpaca News API (v1beta1，自动分页)
    # ==========================================
    def fetch_news(self, start_date: str = None, end_date: str = None):
        """
        通过 Alpaca v1beta1/news 获取公司新闻 (Benzinga 数据源)
        免费版每次最多 50 条，支持 page_token 翻页
        返回 DataFrame: Date, Publisher, Headline
        """
        # 优先读缓存
        if os.path.exists(self.news_cache_file):
            print(f"[+] 发现本地新闻缓存: {self.news_cache_file}")
            news_df = pd.read_csv(self.news_cache_file)
            if not news_df.empty:
                print(f"    已缓存 {len(news_df)} 条")
                return news_df

        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        print(f"[-] 正在从 Alpaca News API 获取 {self.ticker} 新闻 ({start_date} ~ {end_date})...")

        news_url = "https://data.alpaca.markets/v1beta1/news"
        headers = {
            "APCA-API-KEY-ID": self.alpaca_key,
            "APCA-API-SECRET-KEY": self.alpaca_secret,
        }

        all_news = []
        page_token = None
        page = 0
        MAX_PAGES = 200  # 安全上限 200 * 50 = 10,000 条

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
                    print(f"  限流，等待 5s...")
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
                    print(f"  已拉取 {len(all_news)} 条新闻 (第 {page} 页)...")
                time.sleep(0.2)  # 礼貌间隔

            except Exception as e:
                print(f"  新闻 API 错误: {e}")
                break

        if not all_news:
            print("[-] 未获取到任何新闻")
            return pd.DataFrame(columns=["Date", "Publisher", "Headline"])

        news_df = pd.DataFrame(all_news)
        news_df = news_df.drop_duplicates(subset=["Date", "Publisher", "Headline"])
        news_df = news_df.sort_values("Date", ascending=False)

        news_df.to_csv(self.news_cache_file, index=False)
        print(f"[+] 新闻获取完成: {len(news_df)} 条，已缓存到 {self.news_cache_file}")
        return news_df

    # ==========================================
    # 4. 主流程: 数据融合 + 输出
    # ==========================================
    def build_dataset(self, news_start: str = None):
        """
        构建完整特征矩阵:
        1. 历史量价 (Alpaca Bars)
        2. PE (Finnhub)
        3. 新闻 (Alpaca News)
        4. 按 Date inner join
        5. 输出 CSV
        """
        one_year_ago = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        if news_start is None:
            news_start = one_year_ago

        print(f"\n{'='*60}")
        print(f"  Alpaca + Finnhub Pipeline — {self.ticker}")
        print(f"{'='*60}")

        # Step 1: 历史量价
        hist = self.fetch_history(start_date=one_year_ago)
        if hist.empty:
            print("[FATAL] 无法获取历史量价，退出")
            sys.exit(1)

        # Step 2: 计算每日历史 PE (基于季度 EPS)
        hist = self.fetch_historical_pe(hist)

        # Step 3: 新闻
        news_df = self.fetch_news(start_date=news_start)
        if news_df.empty:
            print("[WARNING] 无新闻数据，仅保存量价+PE")
            result = hist.copy()
            result["Ticker"] = self.ticker
            result["Publisher"] = ""
            result["Headline"] = ""
        else:
            # Step 4: inner join (hist 已含 PE_Ratio)
            result = pd.merge(news_df, hist[["Date", "Close", "Volume", "PE_Ratio"]], on="Date", how="inner")
            result["Ticker"] = self.ticker

        # 统一列顺序
        result = result[["Date", "Ticker", "Close", "Volume", "PE_Ratio", "Publisher", "Headline"]]
        result = result.sort_values("Date", ascending=False)

        # Step 5: 保存
        result.to_csv(self.cache_file, index=False)
        print(f"\n[+] 数据集已保存: {self.cache_file} (共 {len(result)} 条)")
        print(f"    日期范围: {result['Date'].min()} ~ {result['Date'].max()}")

        return result

    # ==========================================
    # 5. 增量更新 (最近 7 天)
    # ==========================================
    def update(self):
        """增量更新: 拉取最近 7 天的量价和新闻，追加到已有缓存"""
        if not os.path.exists(self.cache_file):
            print("[-] 本地无缓存，执行完整构建...")
            return self.build_dataset()

        print(f"\n--- 增量同步 {self.ticker} (Alpaca + Finnhub) ---")

        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

        # 近 7 天量价 (Alpaca)
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
            print(f"[-] Alpaca 增量量价失败: {e}")
            hist = pd.DataFrame()

        # 近 7 天新闻 (Alpaca)
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
            print(f"[-] Alpaca 增量新闻失败: {e}")

        news_df = pd.DataFrame(news_records) if news_records else pd.DataFrame()

        if news_df.empty:
            print("[-] 无新的新闻数据")
            return pd.read_csv(self.cache_file)

        # PE: 用历史 PE 计算（增量也用同一方法，保持一致性）
        if not hist.empty:
            hist = self.fetch_historical_pe(hist)

        # 合并
        if not hist.empty:
            new_data = pd.merge(news_df, hist[["Date", "Close", "Volume", "PE_Ratio"]], on="Date", how="inner")
        else:
            new_data = news_df.copy()
            new_data["Close"] = 0.0
            new_data["Volume"] = 0
            new_data["PE_Ratio"] = self.fetch_pe_ratio()  # fallback: 快照

        new_data["Ticker"] = self.ticker
        new_data = new_data[["Date", "Ticker", "Close", "Volume", "PE_Ratio", "Publisher", "Headline"]]

        # 追加去重
        old_data = pd.read_csv(self.cache_file)
        combined = pd.concat([old_data, new_data], ignore_index=True)
        combined = combined.drop_duplicates(subset=["Date", "Ticker", "Publisher", "Headline"])
        combined.to_csv(self.cache_file, index=False)
        print(f"[+] 增量更新完成，数据集总规模: {len(combined)} 行")
        return combined


# ==========================================
# 入口
# ==========================================
if __name__ == "__main__":
    ALPACA_KEY    = "PK6FYKALELA4WL7K47AD2UC626"
    ALPACA_SECRET = "A5mWQGEDL2uwEtWGVkuRkVCAAnp1qGKnQJ5e5osoRQ64"
    FINNHUB_KEY   = "d6hksupr01qr5k4ccku0d6hksupr01qr5k4cckug"

    pipeline = AlpacaFinnhubPipeline(
        alpaca_key=ALPACA_KEY,
        alpaca_secret=ALPACA_SECRET,
        finnhub_key=FINNHUB_KEY,
        ticker="NVDA",
        cache_file="real_nvda_dataset.csv",
    )

    # 首次运行: 构建最近 1 年数据集
    # 后续运行: 增量更新最近 7 天
    if os.path.exists(pipeline.cache_file):
        df = pipeline.update()
    else:
        df = pipeline.build_dataset()

    print(f"\n--- 特征矩阵 (Feature Matrix) 前 10 行 ---")
    print(df.head(10).to_string(index=False))
