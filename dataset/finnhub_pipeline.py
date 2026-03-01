"""
Finnhub + Yahoo 混合数据管道
 - Finnhub: 新闻 (/company-news)、PE (/stock/metric)、实时报价 (/quote)
 - Yahoo Direct API: 历史日级别量价 (/v8/finance/chart)  ← Finnhub 免费版不开放 /stock/candle
输出格式: Date, Ticker, Close, Volume, PE_Ratio, Publisher, Headline
"""

import pandas as pd
import numpy as np
import requests
import time
import os
import sys
from datetime import datetime, timedelta

class FinnhubPipeline:
    FINNHUB_URL = "https://finnhub.io/api/v1"
    YAHOO_CHART_URL = "https://query2.finance.yahoo.com/v8/finance/chart"

    def __init__(self, api_key, ticker="NVDA", cache_file="real_nvda_dataset.csv"):
        self.api_key = api_key
        self.ticker = ticker
        self.cache_file = cache_file
        self.hist_cache_file = f"dataset/{ticker}_hist_cache.csv"
        self.news_cache_file = f"dataset/{ticker}_finnhub_news.csv"

        # Finnhub session
        self.fh_session = requests.Session()
        self.fh_session.params = {"token": self.api_key}

        # Yahoo session (需要浏览器 UA)
        self.yh_session = requests.Session()
        self.yh_session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        })

    # ==========================================
    # 通用工具
    # ==========================================
    def _finnhub_get(self, endpoint, params=None, description="数据"):
        """Finnhub GET 请求，带重试 (免费版 60 次/分钟)"""
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
    # 1. 历史量价 — Yahoo v8/chart  (Finnhub 免费版不支持 /stock/candle)
    # ==========================================
    def fetch_history(self, start_date="2010-01-01", end_date=None):
        """
        从 Yahoo Chart API 获取日级别历史量价
        优先读本地缓存；短期(<2年)用 range 参数一次拉取，长期分段拉取
        返回 DataFrame: Date, Close, Volume
        """
        # 优先读本地缓存
        if os.path.exists(self.hist_cache_file):
            print(f"[+] 发现本地历史量价缓存: {self.hist_cache_file}")
            hist = pd.read_csv(self.hist_cache_file)
            if not hist.empty:
                print(f"    已缓存 {len(hist)} 条 ({hist['Date'].min()} ~ {hist['Date'].max()})")
                return hist

        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        # 计算时间跨度
        dt_start = datetime.strptime(start_date, "%Y-%m-%d")
        dt_end = datetime.strptime(end_date, "%Y-%m-%d")
        span_days = (dt_end - dt_start).days

        # 根据跨度选择策略
        if span_days <= 730:  # ≤2 年：一次性 range 请求（更稳定，不易被限流）
            range_map = {365: "1y", 180: "6mo", 90: "3mo", 30: "1mo", 7: "5d"}
            yf_range = "2y"
            for threshold, r in sorted(range_map.items()):
                if span_days <= threshold:
                    yf_range = r
                    break
            return self._fetch_history_by_range(yf_range)
        else:
            return self._fetch_history_by_chunks(start_date, end_date)

    def _fetch_history_by_range(self, yf_range):
        """用 range 参数一次性拉取（适合 ≤2 年的请求）"""
        url = f"{self.YAHOO_CHART_URL}/{self.ticker}"
        params = {"range": yf_range, "interval": "1d", "includePrePost": "false"}

        print(f"[-] 正在从 Yahoo Chart API 获取 {self.ticker} 量价 (range={yf_range})...")
        for attempt in range(1, 4):
            try:
                resp = self.yh_session.get(url, params=params, timeout=15)
                if resp.status_code == 429:
                    delay = 10 * attempt
                    print(f"  [Retry {attempt}/3] 限流，等待 {delay}s...")
                    time.sleep(delay)
                    continue
                resp.raise_for_status()
                data = resp.json()
                result = data.get("chart", {}).get("result", [])
                if result and result[0].get("timestamp"):
                    timestamps = result[0]["timestamp"]
                    indicators = result[0].get("indicators", {}).get("quote", [{}])[0]
                    hist = pd.DataFrame({
                        "Date": [datetime.fromtimestamp(ts).strftime("%Y-%m-%d") for ts in timestamps],
                        "Close": indicators.get("close", []),
                        "Volume": indicators.get("volume", [])
                    }).dropna(subset=["Close"])
                    hist.to_csv(self.hist_cache_file, index=False)
                    print(f"[+] 历史量价: {len(hist)} 条 ({hist['Date'].min()} ~ {hist['Date'].max()})，已缓存")
                    return hist
            except Exception as e:
                if attempt == 3:
                    print(f"[-] Yahoo Direct API 请求失败: {e}")

        # 降级2: 尝试 yfinance 库（内置 cookie/crumb 机制，有时能绕过直接请求的限流）
        print("[Fallback] 尝试通过 yfinance 库获取量价...")
        try:
            import yfinance as yf
            stock = yf.Ticker(self.ticker)
            yf_hist = stock.history(period=yf_range)
            if yf_hist is not None and not yf_hist.empty:
                yf_hist = yf_hist.reset_index()
                hist = pd.DataFrame({
                    "Date": pd.to_datetime(yf_hist["Date"]).dt.strftime("%Y-%m-%d"),
                    "Close": yf_hist["Close"],
                    "Volume": yf_hist["Volume"]
                })
                hist.to_csv(self.hist_cache_file, index=False)
                print(f"[+] yfinance 历史量价: {len(hist)} 条 ({hist['Date'].min()} ~ {hist['Date'].max()})，已缓存")
                return hist
        except Exception as e:
            print(f"[-] yfinance 也失败: {e}")

        print("[-] 所有量价源均失败，降级使用 Finnhub /quote 获取当日数据")
        return self._fetch_today_quote()

    def _fetch_history_by_chunks(self, start_date, end_date):
        """分段拉取 (适合 >2 年的长期请求)"""
        print(f"[-] 正在从 Yahoo Chart API 分段拉取 {self.ticker} 日级别量价...")
        all_frames = []
        start_year = int(start_date[:4])
        end_year = datetime.now().year
        now_ts = int(time.time())
        chunk_years = 2

        for yr in range(start_year, end_year + 1, chunk_years):
            p1 = int(datetime(yr, 1, 1).timestamp())
            p2 = min(int(datetime(yr + chunk_years, 1, 1).timestamp()), now_ts)
            url = f"{self.YAHOO_CHART_URL}/{self.ticker}"
            params = {"period1": p1, "period2": p2, "interval": "1d", "includePrePost": "false"}
            try:
                resp = self.yh_session.get(url, params=params, timeout=15)
                if resp.status_code == 429:
                    print(f"  [{yr}~{yr+chunk_years}] 限流，等待 15s...")
                    time.sleep(15)
                    resp = self.yh_session.get(url, params=params, timeout=15)
                if resp.status_code != 200:
                    print(f"  [{yr}~{yr+chunk_years}] HTTP {resp.status_code}，跳过")
                    continue
                data = resp.json()
                result = data.get("chart", {}).get("result", [])
                if not result or not result[0].get("timestamp"):
                    continue
                timestamps = result[0]["timestamp"]
                indicators = result[0].get("indicators", {}).get("quote", [{}])[0]
                chunk_df = pd.DataFrame({
                    "Date": [datetime.fromtimestamp(ts).strftime("%Y-%m-%d") for ts in timestamps],
                    "Close": indicators.get("close", []),
                    "Volume": indicators.get("volume", [])
                }).dropna(subset=["Close"])
                all_frames.append(chunk_df)
                print(f"  [{yr}~{yr+chunk_years}] 获取 {len(chunk_df)} 条")
            except Exception as e:
                print(f"  [{yr}~{yr+chunk_years}] 失败: {e}")
            time.sleep(0.5)

        if not all_frames:
            print("[-] Yahoo API 全部失败，降级使用 Finnhub /quote 获取当日数据")
            return self._fetch_today_quote()

        hist = pd.concat(all_frames, ignore_index=True).drop_duplicates(subset=["Date"]).sort_values("Date")
        hist.to_csv(self.hist_cache_file, index=False)
        print(f"[+] 历史量价: {len(hist)} 条 ({hist['Date'].min()} ~ {hist['Date'].max()})，已缓存")
        return hist

    def _fetch_today_quote(self):
        """Finnhub /quote — 获取当日实时报价(降级兜底)"""
        data = self._finnhub_get("/quote", params={"symbol": self.ticker}, description="实时报价")
        today = datetime.now().strftime("%Y-%m-%d")
        return pd.DataFrame([{
            "Date": today,
            "Close": data.get("c", 0),
            "Volume": 0  # /quote 不返回成交量
        }])

    # ==========================================
    # 2. PE 基本面 — /stock/metric
    # ==========================================
    def fetch_pe_ratio(self):
        """获取当前 PE (TTM)"""
        print(f"[-] 正在从 Finnhub 获取 {self.ticker} 基本面指标...")
        data = self._finnhub_get("/stock/metric", params={
            "symbol": self.ticker,
            "metric": "all"
        }, description="PE 基本面")

        metric = data.get("metric", {})
        pe = metric.get("peBasicExclExtraTTM")       # 基本 PE (TTM)
        if pe is None:
            pe = metric.get("peTTM")                  # 备选
        if pe is None:
            pe = metric.get("peExclExtraTTM", np.nan) # 再备选

        print(f"[+] PE_Ratio (TTM) = {pe}")
        return float(pe) if pe is not None else np.nan

    # ==========================================
    # 3. 新闻 — /company-news
    # ==========================================
    def fetch_news(self, start_date="2020-01-01", end_date=None):
        """
        获取公司新闻
        Finnhub 免费版单次最多返回约 250 条，按日期区间查询
        分段拉取以尽量覆盖完整时间范围
        返回 DataFrame: Date, Publisher, Headline
        """
        # 优先读缓存
        if os.path.exists(self.news_cache_file):
            print(f"[+] 发现本地新闻缓存: {self.news_cache_file}")
            news_df = pd.read_csv(self.news_cache_file)
            if not news_df.empty:
                print(f"    已缓存 {len(news_df)} 条")
                return news_df

        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        print(f"[-] 正在从 Finnhub 分段拉取 {self.ticker} 新闻 ({start_date} ~ {end_date})...")

        all_news = []
        dt_start = datetime.strptime(start_date, "%Y-%m-%d")
        dt_end = datetime.strptime(end_date, "%Y-%m-%d")
        chunk_days = 90  # 每次查 90 天，避免单次返回量被截断

        current = dt_start
        while current < dt_end:
            chunk_end = min(current + timedelta(days=chunk_days), dt_end)
            s = current.strftime("%Y-%m-%d")
            e = chunk_end.strftime("%Y-%m-%d")

            try:
                data = self._finnhub_get("/company-news", params={
                    "symbol": self.ticker,
                    "from": s,
                    "to": e
                }, description=f"新闻({s}~{e})")

                if isinstance(data, list) and len(data) > 0:
                    for article in data:
                        pub_ts = article.get("datetime", 0)
                        all_news.append({
                            "Date": datetime.fromtimestamp(pub_ts).strftime("%Y-%m-%d") if pub_ts else s,
                            "Publisher": article.get("source", "Unknown"),
                            "Headline": article.get("headline", "")
                        })
                    print(f"  [{s} ~ {e}] 获取 {len(data)} 条")
                else:
                    print(f"  [{s} ~ {e}] 无新闻")
            except Exception as ex:
                print(f"  [{s} ~ {e}] 请求失败: {ex}")

            current = chunk_end + timedelta(days=1)
            time.sleep(0.3)  # 免费版限流: 60 次/分钟 → 每次间隔 0.3s 安全

        if not all_news:
            print("[-] 未获取到任何新闻数据")
            return pd.DataFrame(columns=["Date", "Publisher", "Headline"])

        news_df = pd.DataFrame(all_news)
        news_df = news_df.drop_duplicates(subset=["Date", "Publisher", "Headline"])
        news_df = news_df.sort_values("Date", ascending=False)

        # 缓存
        news_df.to_csv(self.news_cache_file, index=False)
        print(f"[+] 新闻获取完成: {len(news_df)} 条，已缓存到 {self.news_cache_file}")
        return news_df

    # ==========================================
    # 4. 主流程: 数据融合 + 输出
    # ==========================================
    def build_dataset(self, news_start=None):
        """
        构建完整的特征矩阵:
        1. 拉取历史量价 (日级别)
        2. 拉取 PE
        3. 拉取新闻
        4. 按 Date 对齐合并
        5. 输出 CSV
        """
        one_year_ago = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        if news_start is None:
            news_start = one_year_ago

        print(f"\n{'='*60}")
        print(f"  Finnhub Pipeline — {self.ticker}")
        print(f"{'='*60}")

        # Step 1: 历史量价 (最近 1 年)
        hist = self.fetch_history(start_date=one_year_ago)
        if hist.empty:
            print("[FATAL] 无法获取历史量价，退出")
            sys.exit(1)

        # Step 2: PE
        pe_ratio = self.fetch_pe_ratio()

        # Step 3: 新闻
        news_df = self.fetch_news(start_date=news_start)
        if news_df.empty:
            print("[WARNING] 无新闻数据，仅保存量价+PE")
            result = hist.copy()
            result["Ticker"] = self.ticker
            result["PE_Ratio"] = pe_ratio
            result["Publisher"] = ""
            result["Headline"] = ""
        else:
            # Step 4: 按 Date 合并 (inner join — 只保留有量价又有新闻的日期)
            result = pd.merge(news_df, hist[["Date", "Close", "Volume"]], on="Date", how="inner")
            result["Ticker"] = self.ticker
            result["PE_Ratio"] = pe_ratio

        # 统一列顺序
        result = result[["Date", "Ticker", "Close", "Volume", "PE_Ratio", "Publisher", "Headline"]]
        result = result.sort_values("Date", ascending=False)

        # Step 5: 保存
        result.to_csv(self.cache_file, index=False)
        print(f"\n[+] 数据集已保存: {self.cache_file} (共 {len(result)} 条)")
        print(f"    日期范围: {result['Date'].min()} ~ {result['Date'].max()}")

        return result

    # ==========================================
    # 5. 增量更新 (仅拉取最近数据追加)
    # ==========================================
    def update(self):
        """增量更新: 拉取最近 7 天的量价和新闻，追加到本地缓存"""
        if not os.path.exists(self.cache_file):
            print("[-] 本地无缓存，执行完整构建...")
            return self.build_dataset()

        print(f"\n--- 增量同步 {self.ticker} (Finnhub) ---")

        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

        # 近 7 天量价 (Yahoo Chart API)
        start_ts = int((datetime.now() - timedelta(days=7)).timestamp())
        end_ts = int(datetime.now().timestamp())
        url = f"{self.YAHOO_CHART_URL}/{self.ticker}"
        params = {"period1": start_ts, "period2": end_ts, "interval": "1d", "includePrePost": "false"}
        try:
            resp = self.yh_session.get(url, params=params, timeout=15)
            resp.raise_for_status()
            chart_data = resp.json().get("chart", {}).get("result", [])
            if chart_data and chart_data[0].get("timestamp"):
                timestamps = chart_data[0]["timestamp"]
                indicators = chart_data[0].get("indicators", {}).get("quote", [{}])[0]
                hist = pd.DataFrame({
                    "Date": [datetime.fromtimestamp(ts).strftime("%Y-%m-%d") for ts in timestamps],
                    "Close": indicators.get("close", []),
                    "Volume": indicators.get("volume", [])
                }).dropna(subset=["Close"])
            else:
                hist = pd.DataFrame()
        except Exception:
            hist = pd.DataFrame()

        # 近 7 天新闻
        news_data = self._finnhub_get("/company-news", params={
            "symbol": self.ticker, "from": start, "to": end
        }, description="近期新闻")

        news_records = []
        if isinstance(news_data, list):
            for a in news_data:
                ts = a.get("datetime", 0)
                news_records.append({
                    "Date": datetime.fromtimestamp(ts).strftime("%Y-%m-%d") if ts else end,
                    "Publisher": a.get("source", "Unknown"),
                    "Headline": a.get("headline", "")
                })
        news_df = pd.DataFrame(news_records) if news_records else pd.DataFrame()

        if news_df.empty:
            print("[-] 无新的新闻数据")
            return pd.read_csv(self.cache_file)

        # PE
        pe_ratio = self.fetch_pe_ratio()

        # 合并新数据
        if not hist.empty:
            new_data = pd.merge(news_df, hist[["Date", "Close", "Volume"]], on="Date", how="inner")
        else:
            new_data = news_df.copy()
            new_data["Close"] = 0.0
            new_data["Volume"] = 0

        new_data["Ticker"] = self.ticker
        new_data["PE_Ratio"] = pe_ratio
        new_data = new_data[["Date", "Ticker", "Close", "Volume", "PE_Ratio", "Publisher", "Headline"]]

        # 追加并去重
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
    API_KEY = "d6hksupr01qr5k4ccku0d6hksupr01qr5k4cckug"

    pipeline = FinnhubPipeline(
        api_key=API_KEY,
        ticker="NVDA",
        cache_file="real_nvda_dataset.csv"
    )

    # 首次运行: 构建最近 1 年的数据集
    # 后续运行: 增量更新最近 7 天
    if os.path.exists(pipeline.cache_file):
        df = pipeline.update()
    else:
        df = pipeline.build_dataset()

    print(f"\n--- 特征矩阵 (Feature Matrix) 前 10 行 ---")
    print(df.head(10).to_string(index=False))
