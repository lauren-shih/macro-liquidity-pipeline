"""
fred_loader.py
==============
Production-grade FRED API loader with local Parquet caching.

設計重點
--------
1. **快取優先 (cache-first)**: 先讀本地 parquet,只取增量
2. **錯誤隔離**: 單一指標失敗不會中斷整個 pipeline
3. **批次效率**: 並行抓取 (ThreadPoolExecutor)
4. **可審計 (auditable)**: 紀錄每次抓取的時間、行數、來源

使用方式
--------
>>> loader = FredLoader(api_key=os.getenv('FRED_API_KEY'))
>>> df = loader.fetch_one('SOFR', start='1980-01-01')
>>> panel = loader.fetch_many(['SOFR', 'EFFR', 'IORB'])

Why this matters
----------------
QR analysts need data pipelines that:
  (a) don't hit rate limits,
  (b) handle vendor outages gracefully,
  (c) produce reproducible point-in-time snapshots.
This loader does all three.
"""

# from __future__ import annotations: type hint 延後評估 → 新式 union 語法（str | Path）在舊版 Python 相容。

from __future__ import annotations
import os
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

try:
    from fredapi import Fred
except ImportError:
    Fred = None  # 允許在沒裝fredapi時import config等檔案


# ====================================================================================================================================
# logging = 「可審計」的基礎設施: 每個 cache HIT / STALE / full fetch / FAIL 都留時間戳紀錄,
# 事後可追「哪個 series 何時抓、走哪條 cache 路徑」（production 用 logging, 不用 print）。
# ====================================================================================================================================

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ── cache 累積策略（accumulate-and-merge）─────────────────────────────────────────────────────
# fetch_one 的全量路徑 = 「全量抓取 + combine_first 累積合併」, 對所有 series 一律適用:
#   fetch 有值處優先（抓任何深度的修訂）; fetch 缺處保留舊 cache（深史 / 被限縮 / 缺口永不丟）。
# → 任何 series 哪天被供應端限成滾動短窗（如 ICE OAS 家族 ~3 年窗）, 深史都自動保留 —
#   不需 hardcode 保護清單、不需改 code（self-healing）。


# ====================================================================================================================================
# FRED Loader
# ====================================================================================================================================
class FredLoader:
    """
    Cache-first FRED data loader.

    Parameters
    ----------
    api_key : str
        FRED API key, get one at https://fred.stlouisfed.org/docs/api/api_key.html
    cache_dir : str | Path
        Directory to store Parquet cache files.
    cache_ttl_hours : int
        How fresh the cache must be (default 24h; production passes 12h via main.py CACHE_TTL_HOURS). Older cache triggers refresh.
    """

    # ================================================================================================================================
    # 初始化設計:
    #   ● mkdir(parents=True, exist_ok=True) → 冪等（重複執行不炸）, 缺父層一併建。
    #   ● cache_ttl = 新鮮度「節流閥」（非即時性保證）: TTL 內直接用 cache, 過期走增量;
    #     固定 TTL 有「剛抓完隨即更新」的盲區, 但月頻研究 + 趨勢監控不受影響, 且有 force_refresh 後門。
    # ================================================================================================================================

    def __init__(
        self,
        api_key: str | None = None,
        cache_dir: str | Path = "./cache",
        cache_ttl_hours: int = 24,
    ):

        if Fred is None:
            raise ImportError("fredapi 未安裝, 請: pip install fredapi")

        self.api_key = api_key or os.getenv("FRED_API_KEY")
        if not self.api_key:
            raise ValueError(
                "FRED_API_KEY 未提供。請在 .env 設定, 或:\n"
                '  export FRED_API_KEY="your_key_here"\n'
                "免費申請: https://fred.stlouisfed.org/docs/api/api_key.html"
            )

        self.fred = Fred(api_key=self.api_key)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl = timedelta(hours=cache_ttl_hours)

    # ================================================================================================================================
    # Cache helpers: 一個 series 一個 parquet（<cache_dir>/<series_id>.parquet）;
    # _is_cache_fresh = 檔案存在且「年齡 < TTL」。
    # ================================================================================================================================
    def _cache_path(self, series_id: str) -> Path:
        return self.cache_dir / f"{series_id}.parquet"

    def _is_cache_fresh(self, path: Path) -> bool:
        if not path.exists():
            return False
        age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
        return age < self.cache_ttl

    # ================================================================================================================================
    # Public fetch methods — force_refresh 預設 False: 預設走 cache-first（最常用、最安全）,
    # 要保證最新時才明確開 True（TTL 盲區的解藥）。
    # ================================================================================================================================
    def fetch_one(
        self,
        series_id: str,
        start: str = "1980-01-01",
        end: str | None = None,
        force_refresh: bool = False,  # True → 全量重抓 + 累積合併（不覆蓋深史）
    ) -> pd.Series:
        """
        Fetch a single FRED series. Returns a pd.Series indexed by date.

        快取邏輯:
            1. 若 cache 新鮮且非 force_refresh → 直接讀本地
            2. 若 cache 過期且非 force_refresh → 讀本地最後日期, 增量(尾段)抓取後合併
            3. 若 cache 不存在 → 全量抓取
            4. 若 force_refresh → 全量抓取 + 與舊 cache 累積合併(combine_first:新值優先抓修訂,
               舊值在新 fetch 缺處保留 → 深史/被限滾動短窗永不洗刷,不需 hardcode 清單)
        """
        cache_path = self._cache_path(series_id)

        # ── 累積式 combine_first 合併 ─────────────────────────────────────────────────────────────
        # force_refresh 一律「全量抓取 + 與舊 cache 合併」（而非全量覆蓋）：
        #   • fetch 有值處優先 → 抓到任何深度的修訂（如 M2 回溯重估）。
        #   • fetch 缺的地方保留舊 cache → 深史/被限滾動短窗/缺口,永不丟。
        # 任何 series 哪天被 FRED 限成短窗（如 ICE OAS 滾動 3 年窗）→ 深史自動保留,不需清單、不需改 code。

        # Path 1: fresh cache → 「沒強制重抓」且「cache新鮮」 → 直接讀本地(秒回,不碰網路)
        if not force_refresh and self._is_cache_fresh(cache_path):
            logger.info(f"  [cache HIT ] {series_id}")
            return pd.read_parquet(cache_path)["value"]

        # Path 2 & 3: need API call
        try:
            cache_exists = cache_path.exists()
            if cache_exists and not force_refresh:
                # Path 2:增量(尾段)刷新 → cache 過期但沒強制重抓 → 只抓尾段、合併(快)
                cached = pd.read_parquet(cache_path)["value"]  # 讀舊cache
                last_date = cached.index.max()  # 「舊cache抓到的最後一天」
                fetch_start = (last_date - timedelta(days=7)).strftime("%Y-%m-%d")  # 往前挪7天(重抓近期可能被修訂的)
                logger.info(f"  [cache STALE] {series_id}: incremental from {fetch_start}")
                new_data = self.fred.get_series(  # 只抓尾段
                    series_id,
                    observation_start=fetch_start,
                    observation_end=end,
                )
                merged = pd.concat([cached, new_data]).groupby(level=0).last().sort_index()  # 舊+新合併(new 在重疊區優先)
            else:
                # Path 3:force_refresh 全量抓取（或首抓）→ 抓深史起點起的全段
                logger.info(f"  [full fetch] {series_id}: from {start}")
                new_full = self.fred.get_series(
                    series_id,
                    observation_start=start,
                    observation_end=end,
                )
                if cache_exists:
                    # ★ 累積合併:new 在重疊區優先(抓修訂),舊 cache 在 new 缺處保留(深史/被限永不洗)
                    cached = pd.read_parquet(cache_path)["value"]
                    merged = pd.concat([cached, new_full]).groupby(level=0).last().sort_index()
                    logger.info(f"  [accumulate] {series_id}: 全量+合併(深史保留,被限序列零洗刷)")
                else:
                    merged = new_full  # 首抓,無舊 cache 可合併

            # Persist to cache → 把整理好的 merged 寫進 cache + 回傳
            # index 明確命名為 date,存進 parquet 後讀回來 index 就叫 date,清楚。
            df_out = merged.to_frame("value")
            df_out.index.name = "date"
            df_out.to_parquet(cache_path)  # ⭐真正寫檔
            return merged

        # 容錯: 抓取失敗但有舊 cache → fallback 舊 cache（過期資料勝過沒資料）;
        # 連舊 cache 都沒有 → raise, 讓呼叫端知道此 series 徹底失敗。
        except Exception as e:
            logger.error(f"  [FAIL ] {series_id}: {e}")
            if cache_path.exists():
                logger.warning(f"  [fallback] 使用過期 cache: {series_id}")
                return pd.read_parquet(cache_path)["value"]
            raise

    # ================================================================================================================================
    # fetch_many — 並行抓取設計:
    #   ● max_workers=5: 並行度折衷（全部同時抓會打爆 rate limit, 逐一抓又太慢）。
    #   ● as_completed: 按「完成順序」處理, 不等最慢的。
    #   ● 錯誤隔離: 單一 series 失敗 → log + 跳過, 不中斷其他; 全數失敗 → 回空 DataFrame（防呆, 不讓下游 concat 崩）。
    #   ● concat(axis=1) 以日期 index 自動對齊成寬表, 缺日補 NaN; sort_index 保時序。
    # ================================================================================================================================
    def fetch_many(
        self,
        series_ids: list[str],
        start: str = "1980-01-01",
        end: str | None = None,
        max_workers: int = 5,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        平行抓取多個指標, 回傳寬格式 DataFrame (columns = series_ids)。

        失敗的 series 會被 log 出來, 但不會中斷其他抓取。
        """
        results: dict[str, pd.Series] = {}

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(self.fetch_one, sid, start, end, force_refresh): sid for sid in series_ids}
            for fut in as_completed(futures):
                sid = futures[fut]
                try:
                    results[sid] = fut.result()
                except Exception as e:
                    logger.error(f"Skipping {sid}: {e}")

        if not results:
            return pd.DataFrame()

        panel = pd.concat(results, axis=1)
        panel.columns = list(results.keys())
        panel.index.name = "date"
        return panel.sort_index()

    # ================================================================================================================================
    # metadata: series 的描述資訊（title / units / freq）。資料是核心（fetch_one 失敗會 fallback / raise）,
    # metadata 只是附屬說明 → 失敗僅 log + 回空 dict, 不影響主流程。
    # ================================================================================================================================
    def metadata(self, series_id: str) -> dict:
        """Pull series metadata (title, units, freq, etc.) from FRED."""
        try:
            info = self.fred.get_series_info(series_id)
            return info.to_dict()
        except Exception as e:
            logger.error(f"metadata fail for {series_id}: {e}")
            return {}


if __name__ == "__main__":
    # Smoke test (要 FRED_API_KEY 才能跑)
    from dotenv import load_dotenv

    load_dotenv()

    loader = FredLoader(cache_dir="./cache")
    sofr = loader.fetch_one("SOFR", start="1980-01-01")
    print(f"SOFR: {len(sofr)} obs, last value = {sofr.iloc[-1]:.4f}% on {sofr.index[-1]:%Y-%m-%d}")
