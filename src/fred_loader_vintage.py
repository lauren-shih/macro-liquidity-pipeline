"""
fred_loader_vintage.py
======================
Production-grade ALFRED (Archival FRED) vintage data loader.

Scope-aligned design (Selective ALFRED Upgrade)
-----------------------------------------------
**Not all series benefit from ALFRED upgrade.** Empirical sweep on 37 FRED series
revealed that the standard path's publication_lag config is
already aligned to true publication schedule for daily / weekly indicators:
  - BAML* OAS series:       standard lag=1, real median=0  → gap=0
  - SOFR / EFFR / IORB:     standard lag=1, real median=1  → gap=0
  - WALCL / WDTGAL:        standard lag=5, real median=1-2 → gap=3-4 (over-conservative,
                                                          but no material PIT impact)

Monthly macro indicators with material calibration gap (revision-prone level):
  - CPIAUCSL:  standard lag=14, real median=50  → gap=36 days
  - M2SL:      standard lag=22, real median=52  → gap=30 days
  - PCEPILFE:  standard lag=14, BEA release ~30 days post-month-end → gap material
               (Fed reaction variable for Core PCE)

Plus the NFCI family x5 (NFCI / NFCILEVERAGE / NFCIRISK / NFCICREDIT / ANFCI):
weekly, Chicago Fed retroactively revises history (measured: ANFCI dlt_vs_std
=0.468 = material). One-cut rule: anything entering FM selection takes the
first-release vintage path; pure MON-only stays on standard fixed-lag.

This loader targets the vintage-worthy series listed in VINTAGE_SERIES (config is
the SSoT; currently 8). Other series stay on the standard fred_loader.py path.


Architecture
------------
1. **Cache-first**:  讀本地 cache_vintage/ parquet, TTL 過期才 refetch（default 24h; production 由 main.py CACHE_TTL_HOURS 統一傳 12h）
2. **Initial-release via output_type=4**:  用 FRED 原生 output_type=4 (Initial Release
                          Only) 直接取每個 date 的初值 — 一日一列, 每列附發佈日。避開
                          get_series_all_releases 下載「全部 vintage 修訂版」: 後者對週頻
                          全史重述的序列 (NFCI 家族) 會撞 ALFRED 100k row cap → silent
                          截斷成最早的 ~300 個古早日期 (1971-1976)、近期初值整個遺失 (R17b)。
                          realtime 範圍須開全幅 (1776-07-04 .. 9999-12-31), 否則預設
                          realtime=今天 → 該期間無 vintage date → FRED 回 400。另用
                          observation_start (~2017) 把資料日 bound 在分析窗: 否則 NFCI 週頻
                          全史會讓 FRED 算到 504 Gateway Time-out (R17b);bound 後秒回, 且
                          FM sample 初值不變。
3. **Dtype-normalized**:  FRED 回的 value/realtime_start 全是 str, 一律 cast 成 datetime64 + float64
4. **Error-isolated**:    單一 series 失敗不中斷, fallback 用過期 cache

Usage
-----
>>> loader = FredVintageLoader(api_key=os.getenv('FRED_API_KEY'))
>>> df = loader.fetch_vintage('CPIAUCSL')                     # 單一 series
>>> panel = loader.fetch_vintage_many()                       # 預設 VINTAGE_SERIES
>>> panel = loader.fetch_vintage_many(['CPIAUCSL'])           # 自訂 subset

Why selective vintage scope matters
-----------------------------------
Demonstrates scope-aligned infrastructure thinking:
  "We measured real publication lag / revision magnitude via ALFRED, found that
  daily/weekly market-priced indicators' standard calibration was already correct, and
  selectively upgraded the revision-prone series: CPI, M2, Core PCE (monthly,
  material PIT gap) plus the NFCI family (weekly, retroactively revised, and
  entering FM selection). Production-grade cost-benefit decisions driven by
  measured revision data, not blanket adoption."
"""

from __future__ import annotations
import os
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests  # vintage fetch 直接打 FRED observations endpoint (output_type=4)

from config import VINTAGE_SERIES as _VINTAGE_FROM_CONFIG  # SSoT：vintage 成員清單定義在 config（R24），此處不自行定義


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ============================================================================
# FRED Vintage Loader (ALFRED)
# ============================================================================


class FredVintageLoader:
    """
    Cache-first ALFRED vintage data loader.

    Scope: Only fetches vintage for series with material standard → real lag gap.
    See VINTAGE_SERIES class constant.

    Parameters
    ----------
    api_key : str | None
        FRED API key (same key works for ALFRED). Defaults to env FRED_API_KEY.
    cache_dir : str | Path
        Cache subdirectory. Default './cache_vintage' (separate from standard ./cache).
    cache_ttl_hours : int
        Cache freshness. Default 24h; production passes 12h via main.py CACHE_TTL_HOURS
        (a fixed daily schedule then always triggers the incremental fetch).
    realtime_start, realtime_end : str
        Real-time window passed to FRED output_type=4. MUST span the series'
        vintage dates, so it defaults to the full range '1776-07-04' ..
        '9999-12-31'. (With the default real-time period = today, FRED returns
        400 'no vintage dates exist for the specified real-time period'.)
    observation_start : str
        Lower bound on the DATA-DATE range. Defaults to '2017-01-01' (a buffer
        before the 2018 analysis grid). Required for the weekly NFCI family:
        unbounded output_type=4 over their full 1971-now history makes FRED time
        out (504). Does not change any first-release value inside the grid.
    """

    # VINTAGE_SERIES：SSoT 在 config.py（vintage 成員身分屬 indicator metadata，含 NFCI×5）。
    # 此處用 class-level alias 指向 config，保留既有 self.VINTAGE_SERIES / FredVintageLoader.VINTAGE_SERIES 引用不變。
    VINTAGE_SERIES: set = _VINTAGE_FROM_CONFIG

    # FRED ALFRED observations endpoint（raw）。fredapi 沒有同時暴露 output_type + realtime
    # range 的入口，所以 vintage 的「初值」抓取直接打這個 endpoint（見 _fetch_first_release_raw）。
    FRED_OBS_URL = "https://api.stlouisfed.org/fred/series/observations"

    def __init__(
        self,
        api_key: str | None = None,
        cache_dir: str | Path = "./cache_vintage",
        cache_ttl_hours: int = 24,
        realtime_start: str = "1776-07-04",
        realtime_end: str = "9999-12-31",
        observation_start: str = "2017-01-01",
    ):
        self.api_key = api_key or os.getenv("FRED_API_KEY")
        if not self.api_key:
            raise ValueError(
                "FRED_API_KEY 未提供。請在 .env 設定, 或:\n"
                '  export FRED_API_KEY="your_key_here"\n'
                "免費申請: https://fred.stlouisfed.org/docs/api/api_key.html"
            )

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        # output_type=4 必須給「全幅 realtime」: 預設 realtime=今天 → 該期間無 vintage date → 400。
        # observation_start 把「資料日範圍」bound 在分析窗 (~2017, grid 起點 2018 前一年當 buffer)：
        # NFCI 家族週頻 + 全史重述, 若叫 FRED 用 output_type=4 掃 1971-now 算每個 date 初值 → 後端
        # 60s 算不完 → 504 Gateway Time-out（實測）。bound 到近期 → 只算 ~470 列 → 秒回。對 FM
        # sample (2021+) 初值不變 (近期初值跟抓全史完全相同), 僅排除無關的 pre-2017 古早日期。
        self.realtime_start = realtime_start
        self.realtime_end = realtime_end
        self.observation_start = observation_start

    # ============================================================================
    # Cache helpers (mirror standard fred_loader.py idiom)
    # ============================================================================
    def _cache_path(self, series_id: str) -> Path:
        return self.cache_dir / f"{series_id}.parquet"

    def _is_cache_fresh(self, path: Path) -> bool:
        if not path.exists():
            return False
        age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
        return age < self.cache_ttl

    # ===============================================================================================================================
    # Internal: dtype normalization（@staticmethod: 純資料轉換, 不依賴物件狀態）
    # ALFRED long-format 三欄語意: realtime_start = 該 (date, value) 何時起可見; date = 資料所屬期間; value = 該次發佈值。
    # 「每個 date 取 realtime_start 最小的那筆」= first release — 由下游 pit_safe_vintage 萃取。
    # ===============================================================================================================================
    @staticmethod
    def _normalize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
        """
        Cast the raw FRED return DataFrame to proper dtypes.
        FRED's JSON observations come back with realtime_start / date / value all
        as strings — wasteful for parquet storage and breaks cross-language
        round-trips. (value uses '.' for missing → coerced to NaN.)
        """
        df = df.copy()
        df["realtime_start"] = pd.to_datetime(df["realtime_start"])
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")  # 強制轉換成數字，轉不動的「降級」成"NaN"，絕不報錯中斷crash
        return df

    # ===============================================================================================================================
    # Internal: raw FRED fetch (output_type=4 = "Observations, Initial Release Only")
    # ===============================================================================================================================
    def _fetch_first_release_raw(self, series_id: str) -> pd.DataFrame:
        """
        Fetch initial-release-only observations via FRED ``output_type=4``.

        Returns long-format ``[realtime_start, date, value]``, ONE row per data
        date, where ``realtime_start`` is the date that initial release was first
        published on FRED — i.e. exactly the effective_date the downstream
        first-release extractor / PIT-align needs.

        Why a raw request (not fredapi):
          - fredapi's ``get_series_all_releases`` downloads ALL vintage revisions
            (output_type=2). For weekly, full-history-restated series (the NFCI
            family) that exceeds ALFRED's 100,000-row cap and silently truncates
            to the EARLIEST dates → recent first-releases lost (R17b).
          - ``output_type=4`` returns one row per date → no cap, correct first
            release straight from the API.
          - fredapi does not expose ``output_type`` together with a realtime
            range, so we call the observations endpoint directly.
          - The full realtime span is REQUIRED; with the default realtime period
            (today) FRED returns 400 'no vintage dates exist for the specified
            real-time period'.
          - observation_start bounds the DATA-DATE range to the analysis window.
            Without it, output_type=4 over a weekly, full-history-restated series
            (NFCI family, 1971-now) makes FRED scan the entire vintage history to
            compute each date's initial release → ~60s backend cost → 504 Gateway
            Time-out (R17b). Bounding to ~2017 keeps it to a few hundred rows and
            does not change any first-release value inside the analysis window.
        """
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "output_type": 4,  # Initial Release Only
            "realtime_start": self.realtime_start,
            "realtime_end": self.realtime_end,
            "observation_start": self.observation_start,
        }
        resp = requests.get(self.FRED_OBS_URL, params=params, timeout=90)
        resp.raise_for_status()
        observations = resp.json().get("observations", [])
        df = pd.DataFrame(observations)
        if df.empty:
            raise ValueError(f"{series_id}: FRED returned 0 observations (output_type=4)")
        # output_type=4 也回 realtime_end，PIT 對齊用不到 → 只留下游萃取器要的三欄。
        return df[["realtime_start", "date", "value"]]

    # ===============================================================================================================================
    # Public fetch methods — 與 fred_loader.py 的三個差異:
    # (1) 取「初值（first release）」而非最終修訂值:
    #       ● get_series() 只給「現在回頭看的最終修訂值」— 市場當下沒看過這個數字, 當 factor 會 look-ahead。
    #       ● 本 loader 用 FRED 原生 output_type=4（Initial Release Only）→ 每個 date 一列初值,
    #         帶 realtime_start 標明「該初值何時發佈」（= effective_date）→ 下游可還原任一時點市場真正看到的值。
    #       ● 不用 get_series_all_releases: 它抓回所有修訂版本, 對週頻 + 全史重述的序列（NFCI 家族）
    #         回傳 >100k 列、撞 ALFRED 100k row cap、被 silent 截斷成最早的古早日期 → 近期初值整個遺失（實證）。
    #         output_type=4 一日一列、不撞 cap, 從源頭就拿到正確初值。
    # (2) 範圍: 只抓 VINTAGE_SERIES（config SSoT, 8 個: CPI / M2 / PCE + NFCI 家族 ×5）;
    #       fetch_vintage 開頭有守門員 — 傳非 VINTAGE_SERIES 進來直接 raise, 導向 fred_loader。
    # (3) 回傳形狀是 dict 而非寬表: vintage 是 long-format（date + realtime_start 兩個維度）,
    #       無法「每 series 一欄」併寬表 → 保持 dict, 組裝交給下游 pit_safe_vintage（long → first-release 萃取）。
    # ===============================================================================================================================
    def fetch_vintage(
        self,
        series_id: str,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch full vintage history for one series.

        Returns
        -------
        pd.DataFrame
            Columns: [realtime_start, date, value]
            - realtime_start (datetime64): when this value was first/re-published
            - date (datetime64): the period the value pertains to (e.g. 2024-09-01)
            - value (float64): the numeric value at that vintage

        Cache logic
        -----------
        1. Cache fresh (<24h)   → read parquet, return
        2. Cache stale or miss  → ALFRED API call, cast dtypes, persist
        3. API fails             → fallback to stale cache if exists

        Raises
        ------
        ValueError if series_id not in VINTAGE_SERIES (guards against accidental
        misuse on non-vintage-worthy series — they should go via standard fred_loader).
        """
        if series_id not in self.VINTAGE_SERIES:
            raise ValueError(
                f"'{series_id}' not in VINTAGE_SERIES. Use standard fred_loader.py "
                f"for non-vintage-worthy series. Current VINTAGE_SERIES: {self.VINTAGE_SERIES}"
            )

        cache_path = self._cache_path(series_id)

        # Path 1: fresh cache
        if not force_refresh and self._is_cache_fresh(cache_path):
            logger.info(f"  [cache HIT ] {series_id} (vintage)")
            return pd.read_parquet(cache_path)

        # Path 2 & 3: API call（FRED output_type=4 = Initial Release Only）
        try:
            logger.info(f"  [cache MISS] {series_id} (vintage): output_type=4 first-release")
            df = self._fetch_first_release_raw(series_id)
            df = self._normalize_dtypes(df)

            # Sort for consistency (date asc, realtime_start asc within same date)
            df = df.sort_values(["date", "realtime_start"]).reset_index(drop=True)

            # Persist
            df.to_parquet(cache_path, compression="snappy", index=False)
            logger.info(f"  [cached    ] {series_id}: {len(df):,} rows, " f'{df["date"].nunique()} unique dates')
            return df

        except Exception as e:
            logger.error(f"  [FAIL     ] {series_id}: {e}")
            # Fallback to stale cache if available
            if cache_path.exists():
                logger.warning(f"  [fallback ] using stale vintage cache: {series_id}")
                return pd.read_parquet(cache_path)
            raise

    def fetch_vintage_many(
        self,
        series_ids: Iterable[str] | None = None,
        max_workers: int = 2,
        force_refresh: bool = False,
    ) -> dict[str, pd.DataFrame]:
        """
        Parallel fetch multiple vintage series.

        Parameters
        ----------
        series_ids : iterable of str, optional
            If None, defaults to VINTAGE_SERIES (the canonical scope).
        max_workers : int
            ThreadPool size. Default 2 — vintage scope is small (1-2 series),
            higher concurrency offers no benefit and risks API rate-limit.
        force_refresh : bool
            Bypass cache freshness check.

        Returns
        -------
        dict {series_id: DataFrame}
            Failed series are logged and excluded from the result dict.

        Note
        ----
        Returns a dict (not a wide panel) because vintage data is inherently
        long-format with two indices (data_date + realtime_start). Caller
        downstream (pit_safe_vintage.py) handles the long → first-release
        extraction.
        """
        targets = list(series_ids) if series_ids is not None else list(self.VINTAGE_SERIES)

        # Guard: warn if caller passes non-vintage series
        out_of_scope = [s for s in targets if s not in self.VINTAGE_SERIES]
        if out_of_scope:
            logger.warning(
                f"Series not in VINTAGE_SERIES (will skip): {out_of_scope}. " f"Route them through fred_loader.py instead."
            )
            targets = [s for s in targets if s in self.VINTAGE_SERIES]

        results: dict[str, pd.DataFrame] = {}

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(self.fetch_vintage, sid, force_refresh): sid for sid in targets}
            for fut in as_completed(futures):
                sid = futures[fut]
                try:
                    results[sid] = fut.result()
                except Exception as e:
                    logger.error(f"Skipping {sid}: {e}")

        logger.info(f"Vintage fetch complete: {len(results)}/{len(targets)} succeeded")
        return results


# ============================================================================
# Convenience helpers
# ============================================================================
def is_vintage_series(series_id: str) -> bool:
    """
    Check whether a series should be loaded via vintage path.
    Convenience wrapper around FredVintageLoader.VINTAGE_SERIES.
    """
    return series_id in FredVintageLoader.VINTAGE_SERIES


# ============================================================================
# Smoke test (要 FRED_API_KEY 才能跑)
# ============================================================================
if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    loader = FredVintageLoader(cache_dir="./cache_vintage")

    print("=" * 60)
    print('Smoke test 1: fetch_vintage("CPIAUCSL")')
    print("=" * 60)
    cpi = loader.fetch_vintage("CPIAUCSL")
    print(f"Shape: {cpi.shape}")
    print(f"Dtypes:\n{cpi.dtypes}")
    print(f"\nFirst 3 rows:\n{cpi.head(3)}")
    print(f"\nLast 3 rows:\n{cpi.tail(3)}")

    print("\n" + "=" * 60)
    print("Smoke test 2: fetch_vintage_many() default scope")
    print("=" * 60)
    panel_dict = loader.fetch_vintage_many()
    for sid, df in panel_dict.items():
        print(f'  {sid}: {df.shape[0]:>6} rows, {df["date"].nunique()} unique dates')

    print("\n" + "=" * 60)
    print("Smoke test 3: is_vintage_series() helper")
    print("=" * 60)
    print(f'  is_vintage_series("CPIAUCSL")     = {is_vintage_series("CPIAUCSL")}')
    print(f'  is_vintage_series("M2SL")          = {is_vintage_series("M2SL")}')
    print(f'  is_vintage_series("PCEPILFE")      = {is_vintage_series("PCEPILFE")}')
    print(f'  is_vintage_series("SOFR")          = {is_vintage_series("SOFR")}')
    print(f'  is_vintage_series("BAMLH0A0HYM2")  = {is_vintage_series("BAMLH0A0HYM2")}')
    print(f'  is_vintage_series("DTWEXEMEGS")    = {is_vintage_series("DTWEXEMEGS")}')
