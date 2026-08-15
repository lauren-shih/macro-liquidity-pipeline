"""
test_no_lookahead.py
====================
Unit tests verifying NO look-ahead bias in transformations and PIT layer.

Test strategy: "tail-modification invariance"
---------------------------------------------
A function is PIT-safe if and only if:
  Modifying values at dates AFTER t does NOT change the function's output AT t.

Concretely:
  1. Compute f(series) → result_a
  2. Modify series[t+1 : end] to garbage values
  3. Compute f(modified_series) → result_b
  4. Assert result_a[:t] == result_b[:t]    # values at t and earlier unchanged

If this fails for any t, the function is leaking future info.

This is the standard no-look-ahead check used at multi-manager
quantitative funds to validate alpha pipelines before deployment.

Run: python -m pytest tests/test_no_lookahead.py -v
Or:  python tests/test_no_lookahead.py
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))  # Add src/ to path so we can import without installing

from transformations import (
    rolling_zscore,
    expanding_zscore,
    yoy,
    pct_change_1m,
    diff_bps,
)
from pit_safe import apply_publication_lag, pit_align

# ===================================================================================================================================
# Helpers — tail-modification invariance 核心邏輯:
#   若 transform「偷看未來」, 把未來的值改爛, 它「當下」的輸出就會跟著變。
#   → 改爛未來、檢查當下輸出: 沒變 = 沒偷看; 變了 = look-ahead。
#   _make_test_series（造可重現的假時序）/ _corrupt_future（把 cutoff 之後改爛）/ _check_no_lookahead（比對頭部）。
# ===================================================================================================================================

def _make_test_series(n: int = 500, seed: int = 42) -> pd.Series:
    """Synthetic random-walk series for testing."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    vals = rng.standard_normal(n).cumsum() + 100  # +100: 平移到價格量級
    return pd.Series(vals, index=idx, name="test")


def _corrupt_future(series: pd.Series, cutoff_idx: int) -> pd.Series:
    """Replace all values AFTER cutoff_idx with extreme garbage."""
    corrupted = series.copy()
    corrupted.iloc[cutoff_idx + 1 :] = -99999.0  # 故意用超大負數(極端值)，如果transform偷看了未來，輸出差異會非常明顯
    return corrupted


# ===================================================================================================================================
# _check_no_lookahead — 通用檢核器:
#   transform_fn 以「函式當參數」傳入 → 任何 transform 都能丟進來測, 不用逐函式重寫。
#   頭部比對取 [0, cutoff]（應只依賴過去的區段）; pandas 算術按 index 對齊（label alignment）,
#   兩條 head 來自同一 series、同一切法 → index 必然一致, 不存在錯位。
#   門檻 1e-9 = 「實質為零」(浮點誤差容忍), 非 == 0。
#   assert（測試內部不變量）而非 raise（外部輸入驗證）; assert 訊息的逗號須在括號外 —
#   放進括號 = 條件變 tuple = 恆真 = assert 永不失敗（silent bug）。
# ===================================================================================================================================
def _check_no_lookahead(transform_fn, series: pd.Series, cutoff_idx: int):
    """
    Assert that transform_fn output at dates ≤ cutoff_idx is unchanged
    when future values are corrupted.
    """
    out_clean = transform_fn(series)
    out_corrupt = transform_fn(_corrupt_future(series, cutoff_idx))

    clean_head = out_clean.iloc[: cutoff_idx + 1]
    corrupt_head = out_corrupt.iloc[: cutoff_idx + 1]

    # Compare ignoring NaN (NaN == NaN should pass)
    diff = (clean_head - corrupt_head).abs()
    diff = diff.dropna()  # 頭部可能有NaN，例如：yoy前期都NaN，這些無從比較，該算「相等通過」→ 直接dropna
    if len(diff) == 0:  # 如果去完NaN後空了(頭部全是NaN) → 無從比較 → 直接視為通過(trivially equal)
        return  # all NaN, trivially equal

    max_diff = diff.max()  # 找出頭部的最大差異
    assert max_diff < 1e-9, (
        f"LOOK-AHEAD DETECTED in {transform_fn.__name__}: "
        f"max diff = {max_diff:.6f} at dates ≤ {series.index[cutoff_idx].date()}"
    )


# ===================================================================================================================================
# Group A: Look-ahead invariance（核心性質）
#   每個 test = 造資料 → 選 cutoff → 丟進 _check_no_lookahead; 帶額外參數的 transform 以 lambda
#   當「簽名適配器」（先填好 window / lag, 只留 series 一個洞）。
#   rolling_zscore 測 lag=0 與 lag=1 兩種模式（兩者都不得偷看未來）;
#   expanding_zscore 只測一個代表 case — 其 lag 處理與 rolling 共用同一段 code, invariance 對任何 lag≥0 成立,
#   差異只在固定窗 vs 成長窗, 加測 lag=1 屬 redundant。
# ===================================================================================================================================
def test_rolling_zscore_no_lookahead_lag0():  # 測lag=0(監控模式，含當天)
    """rolling_zscore(lag=0) is OK for monitoring (window inclusive of t)
    but still must not depend on FUTURE values beyond t."""
    s = _make_test_series()
    cutoff = 250
    fn = lambda x: rolling_zscore(x, window=60, lag=0)
    _check_no_lookahead(fn, s, cutoff)


def test_rolling_zscore_no_lookahead_lag1():  # 測lag=1 (ML，嚴格PIT只用過去)
    """rolling_zscore(lag=1) for ML feature use — strict PIT."""
    s = _make_test_series()
    cutoff = 250
    fn = lambda x: rolling_zscore(x, window=60, lag=1)
    _check_no_lookahead(fn, s, cutoff)


def test_expanding_zscore_no_lookahead():
    s = _make_test_series()
    cutoff = 300
    fn = lambda x: expanding_zscore(x, min_periods=30)
    _check_no_lookahead(fn, s, cutoff)


# ===================================================================================================================================
# YoY test 用「月頻」fixture: invariance 本身與頻率無關（日頻也會 PASS）, 但 yoy 的 docstring 契約
# 規定必須以原生頻率呼叫（periods=12 = 12 個月）; 用日頻雖是綠燈, 卻製造「文件禁止 vs test 在做」的矛盾
# → fixture 對齊契約（narrative 一致性）。
# ===================================================================================================================================
def test_yoy_no_lookahead():
    rng = np.random.default_rng(42)
    idx = pd.date_range("2020-01-01", periods=60, freq="MS")  # 60個月 = 5年月頻
    s = pd.Series(rng.standard_normal(60).cumsum() + 100, index=idx)
    cutoff = 40  # warm-up後代表點
    fn = lambda x: yoy(x, periods=12)  # lambda適配periods
    _check_no_lookahead(fn, s, cutoff)


def test_pct_change_1m_no_lookahead():  # 直接傳(無額外參數)
    s = _make_test_series()
    cutoff = 200
    _check_no_lookahead(pct_change_1m, s, cutoff)


def test_diff_bps_no_lookahead():  # 直接傳(無額外參數)
    s = _make_test_series()
    cutoff = 100
    _check_no_lookahead(diff_bps, s, cutoff)


# ===================================================================================================================================
# Group B: PIT semantics — 測的不是「不變性」, 是「lag 參數的具體行為」
# （lag=1 必須與 lag=0 實質不同, 且 lag=1 的統計量嚴格取 [t-window, t-1]）。
# ===================================================================================================================================
def test_lag1_differs_from_lag0():  # 驗證lag=0跟lag=1結果「真的不同」(差異 > 0.01)
    """Verify lag=1 actually does something different from lag=0.
    If they were equal, the lag parameter would be a no-op."""
    s = _make_test_series()
    z0 = rolling_zscore(s, window=60, lag=0)
    z1 = rolling_zscore(s, window=60, lag=1)
    diff = (z0 - z1).dropna().abs()
    assert diff.max() > 0.01, "lag=0 and lag=1 should produce different results"


def test_lag1_uses_strictly_past():  # 手算 [t-window, t-1] 的 mean/std 當正確答案, 與函式輸出比對
    """For lag=1, the rolling mean at t should equal mean of [t-window, t-1]."""
    s = _make_test_series(n=200)
    window = 30
    z = rolling_zscore(s, window=window, lag=1, min_periods=window)

    t = 100  # any index after warm-up
    expected_mean = s.iloc[t - window : t].mean()  # exclusive of t
    expected_std = s.iloc[t - window : t].std()
    expected_z = (s.iloc[t] - expected_mean) / expected_std

    # 抽 t=100 單點與手算答案比對
    assert abs(z.iloc[t] - expected_z) < 1e-9, f"lag=1 z-score at t={t} = {z.iloc[t]:.6f}, " f"expected {expected_z:.6f}"


# ===================================================================================================================================
# Group C: Publication lag — 從「零件」到「整機」兩個子群:
#   C1（單元）: apply_publication_lag 推 index 對不對 — 手算「原 index + 14 天」比對; lag=0 應為恆等（identity）。
#   C2（整合）: pit_align 在月頻 + 14 天 lag 情境「窗內擋、窗後放」—
#     fixture 設計「值 = 月份序號」（查到值 1 = 看到 2 月值, 值 2 = 看到 3 月值）, 讓 asof 查詢結果自帶語意;
#     .asof(d) = 回傳 index ≤ d 的最新一筆（查詢日不必存在於 index 中）。
# ===================================================================================================================================
def test_publication_lag_shifts_index():
    s = _make_test_series(n=20)  # 造20天假資料
    s_lagged = apply_publication_lag(s, lag_days=14)  # 推14天
    expected_index = s.index + pd.Timedelta(days=14)  # 手算預期：原index + 14天
    assert (s_lagged.index == expected_index).all()  # 確認真的推了14天


def test_publication_lag_zero_is_identity():
    s = _make_test_series(n=20)
    s_lagged = apply_publication_lag(s, lag_days=0)
    pd.testing.assert_series_equal(s, s_lagged, check_names=False)  # identity: 值 + index + dtype 完全相等


# 這兩個 test 驗證「pit_align 的擋/放邏輯」（14 天 lag 的月頻序列: lag 窗內擋、窗後放）。
# pit_align 的 lag_map 為必填 → 用中性 fixture 名 TEST_MONTHLY + explicit lag_map={"TEST_MONTHLY": 14} 自帶 lag,
# 不依賴 config（CPI 等 vintage 序列 lag=None, 不適合當此處 fixture）。
def test_pit_align_blocks_future_info():
    """A monthly series with 14-day lag should NOT be visible during the
    14-day window after its index date."""
    idx = pd.date_range("2024-01-01", periods=12, freq="MS")  # 創一個「月頻、期間為一年的時間資料」
    s = pd.Series(range(12), index=idx, name="TEST_MONTHLY")  # 創一個「0~11的資料，index設定為idx(以上面的日期為index)」
    panel = s.to_frame()  # pit_align 吃 DataFrame → Series 轉單欄

    aligned = pit_align(panel, lag_map={"TEST_MONTHLY": 14})  # explicit lag_map（pit_align lag_map 必填）

    # The 2024-03-01 value (val=2) should not be available until ~2024-03-15
    queried_at = pd.Timestamp("2024-03-10")  # before publication
    val_at_q = aligned.asof(queried_at).get("TEST_MONTHLY", np.nan)  # 沒抓到值就NaN

    # We should see the FEB value (val=1) at this point, not MAR
    assert val_at_q == 1, f"PIT violation: at {queried_at}, saw {val_at_q}, expected 1 (Feb value)"


def test_pit_align_releases_after_publication():
    """Once the publication lag has elapsed, the value should be visible."""
    idx = pd.date_range("2024-01-01", periods=12, freq="MS")
    s = pd.Series(range(12), index=idx, name="TEST_MONTHLY")
    panel = s.to_frame()

    aligned = pit_align(panel, lag_map={"TEST_MONTHLY": 14})  # explicit lag_map（pit_align lag_map 必填）
    # 2024-03-01 + 14 days = 2024-03-15. By 2024-03-20 it should be visible.
    queried_at = pd.Timestamp("2024-03-20")  # after publication
    val_at_q = aligned.asof(queried_at).get("TEST_MONTHLY", np.nan)
    assert val_at_q == 2, f"Expected MAR value (2) by {queried_at}, but saw {val_at_q}"


# ===================================================================================================================================
# Run as script — tests 清單裝「函式本身」（無括號）, 迴圈內 t() 才逐一執行;
# 區分 AssertionError（預期型失敗: test 抓到問題）vs 其他 Exception（意外型失敗: crash / bug）。
# ===================================================================================================================================
if __name__ == "__main__":
    tests = [
        test_rolling_zscore_no_lookahead_lag0,
        test_rolling_zscore_no_lookahead_lag1,
        test_expanding_zscore_no_lookahead,
        test_yoy_no_lookahead,
        test_pct_change_1m_no_lookahead,
        test_diff_bps_no_lookahead,
        test_lag1_differs_from_lag0,
        test_lag1_uses_strictly_past,
        test_publication_lag_shifts_index,
        test_publication_lag_zero_is_identity,
        test_pit_align_blocks_future_info,
        test_pit_align_releases_after_publication,
    ]

    # 區分「test失敗(assert)」vs「test爆掉(error)」，診斷更清楚
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"  [PASS] {t.__name__}")
            passed += 1
        except AssertionError as e:  # 預期型失敗: test 抓到問題
            print(f"  [FAIL] {t.__name__}: {e}")
            failed += 1
        except Exception as e:  # 意外型失敗: crash / bug
            print(f"  [ERROR] {t.__name__}: {type(e).__name__}: {e}")
            failed += 1

    print(f'\n{"=" * 60}')
    print(f"Test results: {passed} passed, {failed} failed (of {len(tests)})")
    print("=" * 60)

    # 退出碼: 0 = 全過, 1 = 有失敗（CI 只看退出碼判綠 / 紅燈, 不讀 [PASS]/[FAIL] 文字）。
    # 只有「測試腳本」需要 sys.exit; 函式庫模組（被 import）沒有成敗要回報。
    sys.exit(0 if failed == 0 else 1)
