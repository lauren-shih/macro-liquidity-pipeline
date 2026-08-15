"""
pit_safe.py
===========
Point-in-Time (PIT) safety helpers — handles publication lag.

問題背景 (Why this module exists)
----------------------------------
FRED API 給的是 "重述後" (revised) 的歷史值, 並且把資料放在它"指涉"的日期。
但實盤上, 我們在 "事件日" 是看不到這個值的, 必須等到資料被官方發佈。

例:
  FINRA margin debt 2024-09 值, 標記 index date = 2024-09-30(月底),
  但實際 ~25 天後(10 月下旬)才發布。
  => 若 ML model 在 2024-10-05 用了這個值當 feature, 就洩漏未來。

正確做法:
  把 FM-reach 的 standard 序列各自向後挪其 lag, 對齊到 "可知日"。

每個指標的 lag 從哪來
--------------------
SSoT = config.Indicator.lag（只有 FM-reach 序列有值; main.py 組 lag_map 傳入, 本模組不持有預設表）。
原則: lag 編碼「資訊何時存在」——
  · market-priced（收盤價/殖利率/即期匯率）: 收盤即知, config 帶 0-1 日 strictly-prior 保守日
  · computed / compiled（SOFR T+1、FINRA ~25 日、CFTC 週五、DTWEXEMEGS 週發）: 依發布時程
  · 易修訂統計發布（CPI / M2 / Core PCE / NFCI 家族）: 不走固定 lag —— 走 vintage 路徑
    (fred_loader_vintage / pit_safe_vintage, 不經本模組)

設計
----
- `apply_publication_lag(series, lag_days)`: 通用 lag 套用
- `pit_align(panel, lag_map)`: 一次套用整個 panel 的 lag map
"""

from __future__ import annotations
import pandas as pd

# ====================================================================================================================================
# Publication lag — 設計取捨與 SSoT
# ====================================================================================================================================
# 用 calendar days, 因為實際取得資料日是 wall-clock 時間;
# 對 daily series, T+1 發佈意指 "明天才能看到今天的值"。
# lag 的 SSoT 在 config.Indicator.lag — main.py 從 config 組 lag_map（只含 FM-reach 序列：lag≠None）
# 傳給 pit_align, 本模組不持有 module-level 預設表。
# ====================================================================================================================================


# ====================================================================================================================================
# Core functions
# ====================================================================================================================================
def apply_publication_lag(
    series: pd.Series,
    lag_days: int,
) -> pd.Series:
    """
    Shift a series forward (in time) by `lag_days` calendar days to reflect
    when the data was actually publishable.

    Parameters
    ----------
    series : pd.Series
        Time-indexed series with index date == "the date the value pertains to"
    lag_days : int
        Calendar days of publication delay.

    Returns
    -------
    pd.Series
        Series re-indexed so the value at date d corresponds to the
        REPORTING period (d - lag_days).

    Example
    -------
    >>> # CPI for 2024-09 published 2024-10-10 (10 day lag)
    >>> cpi = fetch_cpi()  # values dated 2024-09-01, etc.
    >>> cpi_pit = apply_publication_lag(cpi, lag_days=14)
    >>> # Now cpi_pit on 2024-09-15 returns NaN (we didn't have it yet)
    >>> # cpi_pit on 2024-10-15 returns the 2024-09-01 value (now published)
    """

    # ================================================================================================================================
    # PIT 推移: 推 index（可見日）, 不推 value
    #   ● 推 index = 同一筆資料延後「看得到的日期」(值不變) — publication lag 的正確語意;
    #     例: 2024-01-31 = 309, 推 +14 → 2024-02-14 = 309（仍是「1 月 CPI」這筆, 只是 2/14 才看得到）。
    #     用 .shift() 推 value 會把數字掛到錯的參考期 = 竄改資料的時間歸屬 → 錯。
    #   ● index 契約: fredapi.get_series() 回傳 DatetimeIndex（文件契約 + 實測驗證）;
    #     契約只是合理預期、不是保證 → isinstance 驗證, 非 DatetimeIndex 即 raise TypeError
    #     — fail loud, 而非 silent corrupt。
    #   ● index.name 補預設 "date": shift 可能讓 name 掉, 下游 parquet / merge 需要具名 index。
    # ================================================================================================================================
    if lag_days == 0:
        return series.copy()
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("Series must have DatetimeIndex")

    shifted = series.copy()
    shifted.index = shifted.index + pd.Timedelta(days=lag_days)
    shifted.index.name = series.index.name or "date"
    return shifted


def pit_align(
    panel: pd.DataFrame,
    lag_map: dict[str, int],
    align_freq: str = "B",
) -> pd.DataFrame:
    """
    Apply publication lag to every column in a panel, then re-align to a
    common business-day frequency with forward-fill.

    Parameters
    ----------
    panel : DataFrame
        Wide panel, columns = series_ids
    lag_map : dict
        必填。{series_id: publication_lag_in_business_days}。
        呼叫端（main）從 config.Indicator.lag 組（只含 FM-reach 序列）。
        panel 的每一欄都必須在 lag_map 中，否則 raise。
    align_freq : str
        Re-sample frequency, default 'B' (business day).

    Returns
    -------
    DataFrame
        Same shape as input but PIT-aligned. Each column shifted by its
        respective lag, then ffill'd onto a unified business-day index.

    Notes
    -----
    Use this for ML feature generation. For dashboard display, you typically
    want the un-lagged version (so today's chart shows today's SOFR).
    """

    # ================================================================================================================================
    # pit_align: sparse panel → PIT 對齊的 dense panel（features 路徑核心）
    #   (A) 每欄各推自己的 publication lag → (B) reindex 到統一 business-day 網格 + ffill 成 dense。
    #   lag_map = 唯一來源（SSoT 在 config.Indicator.lag, 呼叫端必填組好傳入, 無 default fallback）。
    #   features（走 pit_align）vs dashboard（un-lagged 不走）— 見 docstring Notes。
    # ================================================================================================================================
    lag_map_full = lag_map

    # 逐欄推 index, 暫存 dict: 各欄推不同天數後 index 互不對齊（如 CPI→2/14、M2→2/22、SOFR→2/01）,
    # 不能直接進 DataFrame（各欄須共用同一 index）→ 先各自推、後統一對齊。
    # 欄不在 lag_map → raise: 舊式 .get(col, 1) 的 silent 預設會掩蓋呼叫端漏設 lag 的錯誤 → 改 fail-loud。
    shifted_cols = {}
    for col in panel.columns:
        if col not in lag_map_full:
            raise KeyError(f"pit_align: 欄 {col!r} 不在 lag_map 中（呼叫端應只傳有 lag 的 FM-reach 序列；見 config.Indicator.lag）")
        lag = lag_map_full[col]
        shifted_cols[col] = apply_publication_lag(panel[col], lag)

    if not shifted_cols:
        return pd.DataFrame()

    # 統一網格 + densify:
    #   ● freq="B" = 泛用週一至週五網格（無台/美股假日認知）; cross-market 假日/時差對齊在下游 FM merge 層處理。
    #   ● dropna 必須在 reindex(ffill) 之前 — ffill 只補「缺 label」、不覆蓋既存值（NaN 也算存在的值）;
    #     先 dropna 把 sparse 的 NaN label 移除, 月中日期才補得到前值 → dense plateau。
    #   ● sort_index 先行, reindex 的 ffill 才正確。
    all_dates = pd.concat(shifted_cols.values()).index
    bday_idx = pd.date_range(all_dates.min(), all_dates.max(), freq=align_freq)

    out = pd.DataFrame(index=bday_idx)
    for col, s in shifted_cols.items():
        out[col] = s.dropna().sort_index().reindex(bday_idx, method="ffill")

    out.index.name = "date"
    return out


# ====================================================================================================================================
# Diagnostic: compare with-vs-without PIT alignment (manually sanity-check)
# ====================================================================================================================================
def diagnose_lookahead(
    panel_raw: pd.DataFrame,  # 沒做PIT的原始panel
    panel_pit: pd.DataFrame,  # 做了PIT對齊的panel (pit_align的輸出)
    sample_date: str | None = None,  # 要檢查的日期(可不給)
) -> pd.DataFrame:
    """
    Show side-by-side: what raw panel claimed at sample_date vs what
    PIT-aligned panel claimed.

    Useful for sanity-checking before plugging features into a model.
    """
    if sample_date is None:
        sample_date = panel_raw.index[-90].strftime("%Y-%m-%d")  # 預設挑倒數第 90 天當檢查點

    sample = pd.Timestamp(sample_date)
    out = pd.DataFrame(
        {
            "raw_value": panel_raw.asof(sample),  # asof = 截至該時點最近一筆已知值
            "pit_value": panel_pit.asof(sample),
        }
    )
    out["diff"] = out["raw_value"] - out["pit_value"]
    out["look_ahead?"] = (out["diff"].abs() > 1e-9) & out["raw_value"].notna()  # diff 實質非零且 raw 有值 → 疑似 look-ahead
    return out.sort_values("look_ahead?", ascending=False)


# ====================================================================================================================================
# smoke test — 定位: 通電檢查（跑得動 + 結構合理）, 非正確性驗證。
# PIT 正確性由 tests/test_no_lookahead.py 的 12 個 test 負責;
# 常數值 demo 可看出 NaN 階梯與 shape 邊界效應（lag 把日期推出年底 → 列數增加）。
# ====================================================================================================================================
if __name__ == "__main__":
    # Self-test
    idx = pd.date_range("2024-01-01", "2024-12-31", freq="B")
    test_panel = pd.DataFrame(
        {
            "SOFR": [5.30] * len(idx),
            "WALCL": [7_500_000] * len(idx),
            "CPIAUCSL": [310.0] * len(idx),
        },
        index=idx,
    )

    pit = pit_align(test_panel, lag_map={"SOFR": 1, "WALCL": 5, "CPIAUCSL": 14})  # demo-local lag_map（必填）
    print("Raw shape:", test_panel.shape)
    print("PIT shape:", pit.shape)
    print("\nFirst 30 rows of PIT-aligned panel:")
    print(pit.head(30))
    print("\nLook-ahead diagnostic on 2024-06-01:")
    print(diagnose_lookahead(test_panel, pit, "2024-06-01"))
