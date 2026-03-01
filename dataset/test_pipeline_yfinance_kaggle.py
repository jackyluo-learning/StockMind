import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import os
import requests
import sys
import time
import warnings

# 忽略 yfinance 内部产生的 Pandas4Warning 警告
warnings.filterwarnings("ignore", category=pd.errors.Pandas4Warning if hasattr(pd.errors, 'Pandas4Warning') else DeprecationWarning)
warnings.filterwarnings("ignore", message="Timestamp.utcnow is deprecated")

class RealStockDataPipeline:
    def __init__(self, ticker="NVDA", cache_file="real_nvda_dataset.csv", kaggle_raw_file="dataset/raw_analyst_ratings.csv"):
        self.ticker = ticker
        self.cache_file = cache_file
        self.kaggle_raw_file = kaggle_raw_file
        self.hist_cache_file = f"dataset/{ticker}_hist_cache.csv"  # yfinance 历史量价本地缓存
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'})

    def _fetch_with_retry(self, fetch_fn, description="数据", max_retries=3, base_delay=5):
        """带指数退避重试的通用请求包装器"""
        for attempt in range(1, max_retries + 1):
            try:
                result = fetch_fn()
                return result
            except Exception as e:
                err_msg = str(e)
                if 'Rate' in err_msg or 'Too Many' in err_msg or '429' in err_msg:
                    delay = base_delay * (2 ** (attempt - 1))  # 5s, 10s, 20s
                    print(f"[Retry {attempt}/{max_retries}] 获取{description}被限流，等待 {delay} 秒后重试...")
                    time.sleep(delay)
                else:
                    raise  # 非限流错误直接抛出
        raise RuntimeError(f"获取{description}失败：重试 {max_retries} 次后仍被限流。")

    # ==========================================
    # 直接调用 Yahoo Finance API (绕开 yfinance)
    # ==========================================
    def _fetch_history_direct(self, period="max"):
        """直接调用 Yahoo Finance v8/chart API 获取日级历史量价，绕开 yfinance"""
        import calendar

        interval = "1d"

        if period in ("7d", "5d", "1mo"):
            # 短期查询，直接用 range 参数
            range_map = {"7d": "5d", "5d": "5d", "1mo": "1mo"}
            url = f"https://query2.finance.yahoo.com/v8/finance/chart/{self.ticker}"
            params = {"range": range_map[period], "interval": interval, "includePrePost": "false"}
            print(f"[Direct API] 正在从 Yahoo Chart API 获取 {self.ticker} 近期数据 (range={range_map[period]})...")
            resp = self.session.get(url, params=params, timeout=15)
            if resp.status_code == 429:
                raise RuntimeError("Too Many Requests (429) from Yahoo Chart API")
            resp.raise_for_status()
            data = resp.json()
            result = data.get('chart', {}).get('result', [])
            if not result or not result[0].get('timestamp'):
                raise ValueError("Chart API 返回空结果")
            timestamps = result[0]['timestamp']
            indicators = result[0].get('indicators', {}).get('quote', [{}])[0]
            hist = pd.DataFrame({
                'Date': [datetime.fromtimestamp(ts).strftime('%Y-%m-%d') for ts in timestamps],
                'Close': indicators.get('close', []),
                'Volume': indicators.get('volume', [])
            }).dropna(subset=['Close'])
            print(f"[Direct API] 成功获取 {len(hist)} 条近期数据")
            return hist

        # period="max": 日级数据最多只能查约 2 年，需要分段拉取
        print(f"[Direct API] 正在分段从 Yahoo Chart API 获取 {self.ticker} 全量日级历史数据...")
        all_frames = []
        now = int(time.time())
        # 从 2010-01-01 开始，每次拉 2 年，用 period1/period2 时间戳
        start_year = 2010
        current_year = datetime.now().year
        chunk_years = 2

        for yr in range(start_year, current_year + 1, chunk_years):
            p1 = int(datetime(yr, 1, 1).timestamp())
            p2 = min(int(datetime(yr + chunk_years, 1, 1).timestamp()), now)
            url = f"https://query2.finance.yahoo.com/v8/finance/chart/{self.ticker}"
            params = {"period1": p1, "period2": p2, "interval": interval, "includePrePost": "false"}
            try:
                resp = self.session.get(url, params=params, timeout=15)
                if resp.status_code == 429:
                    print(f"[Direct API] 被限流，等待 5 秒...")
                    time.sleep(5)
                    resp = self.session.get(url, params=params, timeout=15)
                if resp.status_code != 200:
                    continue
                data = resp.json()
                result = data.get('chart', {}).get('result', [])
                if not result or not result[0].get('timestamp'):
                    continue
                timestamps = result[0]['timestamp']
                indicators = result[0].get('indicators', {}).get('quote', [{}])[0]
                chunk_df = pd.DataFrame({
                    'Date': [datetime.fromtimestamp(ts).strftime('%Y-%m-%d') for ts in timestamps],
                    'Close': indicators.get('close', []),
                    'Volume': indicators.get('volume', [])
                }).dropna(subset=['Close'])
                all_frames.append(chunk_df)
                print(f"  [{yr}-{yr+chunk_years}] 获取 {len(chunk_df)} 条")
            except Exception as e:
                print(f"  [{yr}-{yr+chunk_years}] 失败: {e}")
            time.sleep(0.5)  # 每段间略停避免限流

        if not all_frames:
            raise ValueError("Direct API 分段拉取全部失败")

        hist = pd.concat(all_frames, ignore_index=True).drop_duplicates(subset=['Date']).sort_values('Date')
        print(f"[Direct API] 全量历史数据: {len(hist)} 条 ({hist['Date'].min()} ~ {hist['Date'].max()})")
        return hist

    def _fetch_quote_direct(self):
        """直接调用 Yahoo Finance v6/quote API 获取 PE 等基本面数据（无需认证）"""
        url = f"https://query2.finance.yahoo.com/v6/finance/quote"
        params = {"symbols": self.ticker, "fields": "trailingPE,forwardPE"}

        resp = self.session.get(url, params=params, timeout=10)
        if resp.status_code == 429:
            raise RuntimeError("Too Many Requests (429) from Yahoo Quote API")

        # v6 可能也需要认证，尝试 v7
        if resp.status_code in (401, 403):
            url = f"https://query1.finance.yahoo.com/v7/finance/quote"
            params = {"symbols": self.ticker}
            resp = self.session.get(url, params=params, timeout=10)
            if resp.status_code == 429:
                raise RuntimeError("Too Many Requests (429) from Yahoo Quote API")

        resp.raise_for_status()
        data = resp.json()

        results = data.get('quoteResponse', {}).get('result', [])
        if not results:
            return np.nan

        pe = results[0].get('trailingPE', results[0].get('forwardPE', np.nan))
        return float(pe) if pe is not None else np.nan

    def _get_pe_ratio(self, stock=None):
        """获取 PE 比率：先尝试 yfinance，失败后降级到直接 API"""
        # 1. 先尝试 yfinance
        try:
            info = self._fetch_with_retry(lambda: stock.info, description="PE数据", max_retries=2, base_delay=3)
            pe = info.get('trailingPE', np.nan)
            if pe is not None and not (isinstance(pe, float) and np.isnan(pe)):
                print(f"[+] 从 yfinance 获取 PE = {pe}")
                return pe
        except Exception:
            pass
        # 2. 降级：直接调 Yahoo API
        try:
            pe = self._fetch_with_retry(self._fetch_quote_direct, description="PE(Direct API)", max_retries=2, base_delay=3)
            print(f"[+] 从 Direct API 获取 PE = {pe}")
            return pe
        except Exception as e:
            print(f"[-] 获取 PE 数据均失败: {e}，使用 NaN 填充")
            return np.nan

    def _get_history_with_cache(self, stock, period="max"):
        """优先从本地缓存读取历史量价，缓存不存在时依次尝试 yfinance 和直接 API"""
        # 尝试读取本地缓存
        if os.path.exists(self.hist_cache_file):
            print(f"[+] 发现本地历史量价缓存: {self.hist_cache_file}，直接加载...")
            hist = pd.read_csv(self.hist_cache_file)
            if not hist.empty:
                return hist

        # === 策略 1: yfinance ===
        print(f"[-] 本地无历史量价缓存，尝试从 yfinance 拉取 (period={period})...")
        try:
            hist = self._fetch_with_retry(
                lambda: stock.history(period=period),
                description="历史量价(yfinance)",
                max_retries=2, base_delay=3
            )
            if hist is not None and not hist.empty:
                hist = hist.reset_index()
                hist['Date'] = pd.to_datetime(hist['Date']).dt.strftime('%Y-%m-%d')
                hist[['Date', 'Close', 'Volume']].to_csv(self.hist_cache_file, index=False)
                print(f"[+] 历史量价已缓存到: {self.hist_cache_file} ({len(hist)} 条)")
                return hist
        except Exception as e:
            print(f"[-] yfinance 获取失败: {e}")

        # === 策略 2: 直接调用 Yahoo Finance API ===
        print(f"[Fallback] 尝试直接调用 Yahoo Finance Chart API...")
        try:
            hist = self._fetch_with_retry(
                lambda: self._fetch_history_direct(period=period),
                description="历史量价(Direct API)",
                max_retries=2, base_delay=5
            )
            if hist is not None and not hist.empty:
                hist[['Date', 'Close', 'Volume']].to_csv(self.hist_cache_file, index=False)
                print(f"[+] 历史量价已缓存到: {self.hist_cache_file} ({len(hist)} 条)")
                return hist
        except Exception as e:
            print(f"[-] Direct API 也失败: {e}")

        return pd.DataFrame()

    # ==========================================
    # 阶段 1: 真实数据集冷启动 (Seed Building)
    # ==========================================
    def build_seed_from_kaggle(self):
        """从 Kaggle 原始真实数据集中提取目标股票新闻，并补全量价数据"""
        print(f"\n[Seed Builder] 检测到缺少本地缓存，正在从 {self.kaggle_raw_file} 构建真实种子数据集...")
        
        if not os.path.exists(self.kaggle_raw_file):
            print("\n" + "="*60)
            print("[CRITICAL ERROR] 缺少 Kaggle 原始数据集！")
            print("="*60)
            print("请前往 Kaggle 下载 'Massive Stock News Analysis DB for NLP Backtests'")
            print(f"并将解压后的 '{self.kaggle_raw_file}' 放入当前目录。")
            print("="*60)
            sys.exit(1)

        # 1. 读取 Kaggle 数据并过滤
        print(f"[-] 正在读取庞大的新闻数据库 (这可能需要几秒钟)...")
        # Kaggle 数据集列名通常为: Unnamed: 0, headline, url, publisher, date, stock
        df_raw = pd.read_csv(self.kaggle_raw_file, usecols=['headline', 'publisher', 'date', 'stock'])
        df_target_news = df_raw[df_raw['stock'] == self.ticker].copy()
        
        if df_target_news.empty:
            print(f"[-] 数据集中未找到 {self.ticker} 的新闻。请更换 Ticker。")
            sys.exit(1)
        else:
            print(f"[+] 从 Kaggle 数据集中提取到 {len(df_target_news)} 条 {self.ticker} 相关的新闻记录。")
            
        # 格式化日期，统一为 YYYY-MM-DD
        # 先去掉时区后缀（如 -04:00），避免 pandas 3.x 中混合时区导致大量解析失败
        df_target_news['date'] = df_target_news['date'].str.replace(r'[+-]\d{2}:\d{2}$', '', regex=True)
        df_target_news['Date'] = pd.to_datetime(df_target_news['date'], errors='coerce').dt.strftime('%Y-%m-%d')
        df_target_news = df_target_news.dropna(subset=['Date'])
        df_target_news = df_target_news.rename(columns={'headline': 'Headline', 'publisher': 'Publisher', 'stock': 'Ticker'})
        df_target_news = df_target_news.sort_values('Date', ascending=False)
        print(f"预处理后，数据量为 {len(df_target_news)} 条。日期范围: {df_target_news['Date'].min()} 到 {df_target_news['Date'].max()}")
        
        # 2. 从 yfinance 补全这些日期的历史真实量价（优先使用本地缓存）
        print(f"[-] 正在补全对应的历史真实量价数据...")
        stock = yf.Ticker(self.ticker, session=self.session)
        
        hist = pd.DataFrame()
        try:
            hist = self._get_history_with_cache(stock, period="max")
            if hist is None or hist.empty:
                print("[-] 未能获取到历史数据。")
                hist = pd.DataFrame()
        except Exception as e:
            print(f"[-] 获取历史量价失败: {e}")

        if not hist.empty:
            print(f"[Debug] 历史数据日期范围: {hist['Date'].min()} 到 {hist['Date'].max()}")
            print(f"[Debug] 新闻数据日期范围: {df_target_news['Date'].min()} 到 {df_target_news['Date'].max()}")
            
            # 3. 数据融合：完美的时间对齐
            # 注意：这里我们只取最近的 500 条进行对齐，以防合并后过于庞大
            seed_df = pd.merge(df_target_news.head(500), hist[['Date', 'Close', 'Volume']], on='Date', how='inner')
            if seed_df.empty:
                 print("[Debug] 警告：合并后数据为空。检查日期是否重叠。")
        else:
            print("[-] 降级：使用模拟量价数据进行初始化 (保留所有新闻记录)...")
            seed_df = df_target_news.copy()
            seed_df['Close'] = 0.0
            seed_df['Volume'] = 0
        
        # 补充当前静态 PE（先 yfinance，再直接 API）
        pe_ratio = self._get_pe_ratio(stock)
        seed_df['PE_Ratio'] = pe_ratio
        
        seed_df = seed_df[['Date', 'Ticker', 'Close', 'Volume', 'PE_Ratio', 'Publisher', 'Headline']]
        
        # 4. 保存为本地真实缓存
        seed_df.to_csv(self.cache_file, index=False)
        print(f"[+] 真实种子数据集构建完成！已保存为: {self.cache_file} (共 {len(seed_df)} 条对齐数据)")
        return seed_df

    # ==========================================
    # 阶段 2: 实时增量更新 (Real-time Fetching)
    # ==========================================
    def fetch_real_time_data(self):
        """尝试获取今日实时新闻与量价"""
        stock = yf.Ticker(self.ticker, session=self.session)
        
        pe_ratio = self._get_pe_ratio(stock)
            
        hist = pd.DataFrame()
        # 先尝试 yfinance，失败后降级到直接 API
        try:
            hist = self._fetch_with_retry(
                lambda: stock.history(period="7d"),
                description="近7日量价(yfinance)",
                max_retries=2, base_delay=3
            )
            if hist is not None and not hist.empty:
                hist = hist.reset_index()
                hist['Date'] = pd.to_datetime(hist['Date']).dt.strftime('%Y-%m-%d')
        except:
            pass
        if hist is None or hist.empty:
            try:
                print("[Fallback] 尝试直接调用 Yahoo Chart API 获取近期量价...")
                hist = self._fetch_history_direct(period="7d")
            except Exception as e:
                print(f"[-] Direct API 近期量价也失败: {e}")
                hist = pd.DataFrame()
        
        news = []
        try:
            news = stock.news
            if news and len(news) > 0:
                print(f"[Debug] 第一条新闻的键: {news[0].keys()}")
        except Exception as ne:
            print(f"[Debug] 获取 stock.news 抛出异常: {ne}")

        if not news:
            raise ValueError("未能获取到实时新闻数据。")
            
        news_records = []
        for article in news:
            # 安全地获取时间戳，如果缺失则跳过或使用当前时间
            pt = article.get('providerPublishTime')
            if pt is None:
                # 尝试获取其他可能的时间键，比如 'pubDate'
                pt = article.get('pubDate', time.time())
            
            try:
                pub_date = datetime.fromtimestamp(pt).strftime('%Y-%m-%d')
            except Exception as te:
                print(f"[Debug] 时间转换失败 ({te}), pt={pt}")
                pub_date = datetime.now().strftime('%Y-%m-%d')

            news_records.append({
                'Date': pub_date,
                'Publisher': article.get('publisher', 'Unknown'),
                'Headline': article.get('title', '')
            })
        news_df = pd.DataFrame(news_records)
        
        if not hist.empty:
            merged_df = pd.merge(news_df, hist[['Date', 'Close', 'Volume']], on='Date', how='inner')
        else:
            merged_df = news_df.copy()
            merged_df['Close'] = 0.0
            merged_df['Volume'] = 0

        merged_df['PE_Ratio'] = pe_ratio
        merged_df['Ticker'] = self.ticker
        
        return merged_df[['Date', 'Ticker', 'Close', 'Volume', 'PE_Ratio', 'Publisher', 'Headline']]

    def update_local_cache(self, new_data):
        """将实时数据追加到本地，实现增量更新"""
        old_data = pd.read_csv(self.cache_file)
        combined_data = pd.concat([old_data, new_data], ignore_index=True)
        # 去重逻辑
        combined_data = combined_data.drop_duplicates(subset=['Date', 'Ticker', 'Publisher', 'Headline'])
        combined_data.to_csv(self.cache_file, index=False)
        print(f"[System] 实时数据已追加到缓存。当前数据集总规模: {len(combined_data)} 行")

    # ==========================================
    # 阶段 3: 主调度引擎
    # ==========================================
    def get_data(self):
        # 检查冷启动
        if not os.path.exists(self.cache_file):
            self.build_seed_from_kaggle()
            
        print(f"\n--- 开始同步 {self.ticker} 数据 ---")
        try:
            print("[1] 尝试获取今日实时增量数据...")
            real_df = self.fetch_real_time_data()
            print("[+] 实时数据获取成功！")
            self.update_local_cache(real_df)
            return pd.read_csv(self.cache_file)
        except Exception as e:
            print(f"[-] 实时 API 受限 ({e})。")
            print("[2] 触发降级机制：直接加载本地真实数据集...")
            return pd.read_csv(self.cache_file)

if __name__ == "__main__":
    # 初始化数据流，目标: 英伟达
    pipeline = RealStockDataPipeline(ticker="NVDA")
    
    # 提取最终用于机器学习的特征矩阵
    feature_matrix = pipeline.get_data()
    
    print("\n--- 最终输入模型的特征矩阵 (Feature Matrix) 头 5 行 ---")
    print(feature_matrix.head())