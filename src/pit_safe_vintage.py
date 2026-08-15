"""
pit_safe_vintage.py
====================

PIT-safe feature engineering for ALFRED vintage data (selective vintage scope).

Mission
-------
Convert ALFRED 3D vintage panel `(realtime_start, date, value)` into 2D
PIT-correct feature panel `(query_date, feature_value)` ready for downstream
Fama-MacBeth cross-sectional regression.

Three-layer architecture
------------------------
Layer 1: `extract_first_release(panel, series_id)`
    對每個 data_date 抽出 first release (min realtime_start).
    Rationale: Backtesting must mimic policy-maker's reality
    (Croushore 2011; Diebold-Rudebusch real-time literature).

Layer 2: `pit_align_vintage(first_release_panel, query_dates, lag_days=0)`
    對齊 first-release panel 到 query_dates,
    via `pd.merge_asof(direction='backward')`.
    Result: per query_date, the latest knowable first-release value.

Layer 3: `build_vintage_features(vintage_panel, query_dates)`
    Orchestrator: panel → first_release → pit_align → daily first-release LEVELS (wide).
    Output = first-release levels only (NO transforms). main.py joins these into
    pit_panel, then `transformations.apply_transforms(mode='fm')` computes the FM
    change/innovation uniformly — single transform source, mirror the standard path.

Design rationale notes
----------------------
1. **First release, not latest available**
   Latest-available is for current-day dashboards. Backtesting requires
   first-release to avoid revision-injected look-ahead bias.

2. **Explicit sort + drop_duplicates over groupby+agg**
   `groupby('date').agg(value=('value', 'first'))` is silent-bug-prone
   if upstream panel isn't sorted by realtime_start. We sort explicitly
   then drop_duplicates with stable kind — no implicit assumptions.

3. **Transforms live downstream, not here**
   This module outputs first-release LEVELS only. The FM change/innovation
   (yoy/diff/diff_bps) is computed by `transformations.apply_transforms` in
   main.py's features path, AFTER these levels are joined into pit_panel — so
   the vintage path and the standard path share one transform implementation (no
   duplicated yoy/zscore formulas). apply_transforms is itself frequency-aware
   (resamples to native frequency before yoy/diff), so monthly CPI/M2/PCE and
   weekly NFCI are handled correctly on the daily grid.

4. **lag_days=0 default for vintage path**
   ALFRED vintage data is already PIT-correct by construction.
   `lag_days` parameter exists for opt-in extra safety buffer
   but is 0 by default.

5. **Out-of-scope is main.py's job, not this module's**
   Only handles vintage series. Non-vintage series go through
   `pit_safe.py` (standard fallback). Integration in `main.py`.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd


# ==================================================================================================================================
# module-level logger（production 慣例: 分等級 / 可導向 / 可關閉, 不用 print）。
# guard 防重複 handler: 模組被 import 多次時 handler 會疊加 → 同一行 log 印 N 次。
# ==================================================================================================================================
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# ==================================================================================================================================
# Layer 1: extract_first_release
# ==================================================================================================================================
def extract_first_release(
    panel: pd.DataFrame,
    series_id: str,
) -> pd.DataFrame:
    """
    Extract first-release per data_date from an ALFRED vintage panel.

    Parameters
    ----------
    panel : pd.DataFrame
        ALFRED vintage panel from `FredVintageLoader.fetch_vintage(series_id)`.
        Required columns:
            - realtime_start : datetime  (when value was first publish-time)
            - date           : datetime  (the data's calendar date)
            - value          : float
    series_id : str
        FRED series id, used for logging only.

    Returns
    -------
    pd.DataFrame
        First-release panel with columns:
            - effective_date : datetime  (= min realtime_start per data date)
            - date           : datetime
            - value          : float
        Sorted by date ascending. NaN values preserved
        (e.g. BLS suspension months like CPI 2025-10-01).

    Design rationale
    ----------------
    - Explicit stable sort by ['date', 'realtime_start'] before
      drop_duplicates ensures deterministic first-release pick,
      not relying on upstream sort order.
    - Rename `realtime_start` → `effective_date` makes semantic
      clearer: this is *when downstream can know this value*.
    """

    # 邊界驗證（panel 來自外部 → 函式邊界用 raise, 非 assert）: required 欄缺一即 fail loud; 多餘欄不影響。
    required_cols = {"realtime_start", "date", "value"}
    missing = required_cols - set(panel.columns)
    if missing:
        raise ValueError(f"[{series_id}] panel missing required columns: {missing}. " f"Got: {list(panel.columns)}")

    # 缺欄位 = 結構錯誤 → raise 硬停（上面）; 空 panel = 結構對但沒資料（可能合理）
    # → warning + 回傳「欄位對齊輸出 schema」的空表（graceful 降級, 下游仍能靠欄位結構運作）。
    if panel.empty:
        logger.warning(f"[{series_id}] empty panel passed to extract_first_release")
        return pd.DataFrame(columns=["effective_date", "date", "value"])

    # Step 1. Explicit stable sort — never trust upstream ordering for PIT logic。
    # 目標是「per date 取首發」(min realtime_start), 非「全表第一筆」;
    # kind="stable" 強制穩定排序: sort key 相等的 row 保持原順序。
    sorted_panel = panel.sort_values(
        ["date", "realtime_start"],
        kind="stable",
    )

    # Step 2. First row per data_date = min realtime_start due to stable sort。
    # ⚠️ Croushore trap: `groupby("date").agg(value=("value","first"))` 看似等價, 但 "first" 取的是
    #    「group 內第一個非 null」, 順序取決於上游 —— 未排序時可能抓到「修訂值」而非首發
    #    = silent revision leakage（不報錯、值錯）。explicit sort + drop_duplicates(keep="first")
    #    = 自己控制順序, "first" 才保證等於首發（降冪排序下 keep="first" 反而會拿到最新修訂）。
    first_release = sorted_panel.drop_duplicates(
        subset="date",
        keep="first",
    ).copy()

    # Semantic rename: realtime_start → effective_date（= 下游「可知道此值」的日期）
    first_release = first_release.rename(columns={"realtime_start": "effective_date"})

    # 選欄定序 + reset_index → 乾淨 schema 給下游。
    first_release = first_release[["effective_date", "date", "value"]]
    first_release = first_release.sort_values("date").reset_index(drop=True)

    # PIT invariant（防禦）: effective_date < date 不該存在 → raise（外部資料完整性, 不可 strip 的硬失敗）。
    # log 同印 n_rows 與 n_dates: drop_duplicates 後兩者應相等 → log 一眼即交叉 sanity check;
    # lag_median = 真實 publication lag（CPI~50 / M2~52 / PCE~58 天）, 與 ALFRED sweep 的量測互為印證。
    if (first_release["effective_date"] < first_release["date"]).any():
        raise ValueError(
            f"[{series_id}] PIT invariant violated: effective_date < date "
            f"found in first_release panel. This should never happen — "
            f"ALFRED data integrity issue."
        )

    n_dates = first_release["date"].nunique()
    n_rows = len(first_release)
    lag_median = (first_release["effective_date"] - first_release["date"]).dt.days.median()

    logger.info(
        f"[{series_id}] first_release extracted: " f"{n_rows} rows, {n_dates} unique dates, " f"median lag {lag_median:.0f} days"
    )

    return first_release


# ==================================================================================================================================
# Layer 2: pit_align_vintage
# ==================================================================================================================================
# 概念圖: query grid（分析期月底 / daily grid）× first-release 答案庫 → PIT-correct 特徵
#               ────────────────────────────────────────────────────────────────────────────────
#               |     你的分析期月底     first_release (資料/答案庫)                              |
#               |    （query_dates）              |                                             |
#               |       = 特徵矩陣      由extract_first_release產出                              |
#               |     每個row的日期               |                                              |
#               |           \                    |                                              |
#               |            \                  /                                               |
#               |     pit_align_vintage (對每個"query_date"找"effective_date" <= 它的最新值)      |
#               |                       |                                                       |
#               |     輸出：每個月底 → 那時你「真的知道」的macro值(PIT-correct特徵)                 |
#               |                       |                                                       |
#               |     ⭐下游 Fama-MacBeth 的特徵矩陣                                              |
#               ────────────────────────────────────────────────────────────────────────────────
# ==================================================================================================================================
def pit_align_vintage(
    first_release_panel: pd.DataFrame,
    query_dates: pd.DatetimeIndex,
    lag_days: int = 0,
    series_id: str = "<unknown>",
) -> pd.DataFrame:
    """
    PIT-align a first-release panel to a set of query_dates.

    For each query_date, find the row with the latest effective_date
    such that `effective_date + lag_days <= query_date`.

    Parameters
    ----------
    first_release_panel : pd.DataFrame
        Output of `extract_first_release`.
        Required columns: effective_date, date, value.
    query_dates : pd.DatetimeIndex
        Target dates for PIT alignment (e.g. month-end query schedule).
    lag_days : int, default 0
        Extra safety buffer in days. ALFRED vintage is already
        PIT-correct, so default 0. Bump only for opt-in over-conservatism.
    series_id : str
        For logging only.

    Returns
    -------
    pd.DataFrame
        PIT-aligned panel with columns:
            - query_date     : datetime  (from query_dates)
            - pit_value      : float     (latest knowable value at query_date)
            - data_date      : datetime  (underlying data_date that pit_value refers to)
            - effective_date : datetime  (when pit_value became knowable)
        For query_dates earlier than any effective_date, pit_value is NaN.

    Design rationale
    ----------------
    - `pd.merge_asof(direction='backward')` is pandas-native "as-of join"
      idiom, O(n+m) for sorted inputs.
    - `direction='backward'` = "find rows where right.effective_date
       <= left.query_date" → PIT-correct.
    - `direction='forward'/'nearest'` = look-ahead leak (forbidden).
    """
    if first_release_panel.empty:
        logger.warning(f"[{series_id}] empty first_release_panel; " f"returning all-NaN aligned panel")
        return pd.DataFrame(
            {
                "query_date": pd.to_datetime(query_dates),
                "pit_value": np.nan,
                "data_date": pd.NaT,  # NaT（非 NaN）→ 下游 .dt 操作才不會出錯
                "effective_date": pd.NaT,
            }
        )

    if lag_days < 0:  # 負 lag = look-ahead → raise
        raise ValueError(f"lag_days must be >= 0, got {lag_days}")

    # Sort by effective_date (required by merge_asof)
    panel = first_release_panel.sort_values("effective_date").copy()

    # lag_days 是 opt-in 的額外保守 buffer（default 0）, 不是修正首發日 —— effective_date 本來就是
    # ALFRED ground truth 的真實首發日。lag>0 時另建 with_lag 欄僅供 merge 配對（保守閘門）,
    # 輸出仍記「真實 effective_date」; right_on 用變數集中判斷, merge 呼叫保持單一寫法。
    if lag_days > 0:
        panel["effective_date_with_lag"] = panel["effective_date"] + pd.Timedelta(days=lag_days)
        right_on = "effective_date_with_lag"
    else:
        right_on = "effective_date"

    # merge_asof 硬性要求: 左右兩邊皆為 DataFrame、且皆按 join key「升序」排序（否則 raise）。
    # query_dates 是 DatetimeIndex → 包成單欄 DataFrame 當左邊; to_datetime 為防禦性正規化。
    target_df = (
        pd.DataFrame(
            {
                "query_date": pd.to_datetime(query_dates),
            }
        )
        .sort_values("query_date")
        .reset_index(drop=True)
    )

    # As-of merge: 對每個 query_date（左邊驅動、1 對 1 → 輸出形狀 = 左邊）,
    # 找右邊 effective_date <= query_date 的最近一筆 → PIT-correct;
    # direction='forward' / 'nearest' = 偷看未來, 禁用。
    # 暖身期（query 早於任何首發）→ NaN / NaT, 下游 dropna。
    aligned = pd.merge_asof(
        target_df,
        panel,
        left_on="query_date",
        right_on=right_on,
        direction="backward",  # PIT-correct（forward / nearest = leak）
    )

    # Build clean output schema
    result = pd.DataFrame(
        {
            "query_date": aligned["query_date"],
            "pit_value": aligned["value"],
            "data_date": aligned["date"],
            "effective_date": aligned["effective_date"],  # 真實首發日（非 lagged）
        }
    )

    # observability: 對齊覆蓋率 n_aligned / n_total —— 遠低於全額（如 2/40）即異常
    # （資料起始太晚 / lag 過激 / series 未涵蓋分析期）; 差幾筆 = 暖身期, 屬正常。
    n_aligned = result["pit_value"].notna().sum()
    n_total = len(result)
    logger.info(f"[{series_id}] pit_align complete: " f"{n_aligned}/{n_total} query_dates aligned " f"(lag_days={lag_days})")

    return result


# ==================================================================================================================================
# Layer 3: build_vintage_features (Orchestrator)
# ==================================================================================================================================
# 設計: vintage 路徑「只做 alignment」—— 輸出 daily first-release LEVELS 寬表（欄 = series_id、值 = pit_value）。
#   ● transform 不在此做: 統一交給 transformations.apply_transforms —— main.py 在 features 路徑把這些 level
#     join 進 pit_panel 後「一次」算 FM 變化量（mirror standard 結構、單一 transform 來源, 消滅公式重複）。
#   ● query_dates 由 main 傳「daily business-day grid」（= pit_panel.index）→ 與 standard pit_panel 同網格, join 乾淨。
#   ● extract_first_release（Layer 1）+ pit_align_vintage（Layer 2）= PIT 對齊核心。
# ==================================================================================================================================
def build_vintage_features(
    vintage_panel: dict[str, pd.DataFrame],
    query_dates: pd.DatetimeIndex,
    lag_days: int = 0,
) -> pd.DataFrame:
    """
    Build a wide panel of PIT-safe FIRST-RELEASE LEVELS from vintage series.

    Orchestrates: vintage panel -> first_release (Layer 1) -> pit_align (Layer 2)
    -> daily first-release levels (one column per series). NO transforms here:
    the FM change/innovation is applied downstream by
    `transformations.apply_transforms` in main.py's features path, so the vintage
    path and the standard path share a single transform implementation.

    Parameters
    ----------
    vintage_panel : dict[series_id -> DataFrame]
        Output of `FredVintageLoader.fetch_vintage_many()`.
    query_dates : pd.DatetimeIndex
        Target query grid. main.py passes pit_panel's business-day index so the
        output aligns to the same grid as the standard PIT panel (clean join).
    lag_days : int, default 0
        Passed to `pit_align_vintage` (opt-in extra safety buffer; ALFRED vintage
        is already PIT-correct so default 0).

    Returns
    -------
    pd.DataFrame
        Wide first-release LEVEL panel:
            - index   : query_dates (named 'date', matching pit_panel)
            - columns : one per vintage series, named by series_id, value = pit_value
        For query_dates earlier than a series' first effective_date, that cell is
        NaN (warm-up). Downstream apply_transforms handles NaN (dropna in yoy/diff).

    Examples
    --------
    >>> from fred_loader_vintage import FredVintageLoader
    >>> loader = FredVintageLoader()
    >>> vintage = loader.fetch_vintage_many()        # {'CPIAUCSL': df, 'M2SL': df, ...}
    >>> grid = pit_panel.index                        # business-day grid from standard pit_align
    >>> levels = build_vintage_features(vintage, grid)
    >>> levels.columns.tolist()
    ['CPIAUCSL', 'M2SL', 'PCEPILFE', 'NFCI', ...]     # first-release LEVELS only
    """
    level_cols: dict[str, pd.Series] = {}

    for series_id, panel in vintage_panel.items():
        logger.info(f"--- Aligning first-release levels for {series_id} ---")

        # Layer 1: first release per data_date（min realtime_start）
        first_release = extract_first_release(panel, series_id)
        if first_release.empty:
            logger.warning(f"[{series_id}] empty first_release; skipping")
            continue  # 空 series 不進 level_cols

        # Layer 2: PIT-align first-release 到 query grid（merge_asof backward as-of join）
        aligned = pit_align_vintage(first_release, query_dates, lag_days=lag_days, series_id=series_id)

        # 只取 level: query_date 升為 index, 抽出 pit_value Series, 欄名 rename 成 series_id。
        # 不做任何 transform —— 留給 main.py 的 apply_transforms(mode='fm') 統一處理。
        level_cols[series_id] = aligned.set_index("query_date")["pit_value"].rename(series_id)

    if not level_cols:
        logger.error("No vintage levels were built; returning empty DataFrame")
        return pd.DataFrame(index=pd.to_datetime(query_dates))

    # 各欄共用同一 query_date index（都來自同一份 query_dates）→ concat(axis=1) 對齊 index 組寬表, 無錯位。
    levels = pd.concat(level_cols.values(), axis=1)
    levels.index.name = "date"  # 對齊 pit_panel 的 index 名, 讓 main.py 的 pit_panel.join(levels) 乾淨
    
    logger.info(f"Vintage first-release levels built: shape={levels.shape}, columns={list(levels.columns)}")
    return levels
