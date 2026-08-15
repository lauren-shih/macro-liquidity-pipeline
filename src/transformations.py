"""
transformations.py
==================
Time-series transformations used across macro/factor pipelines.

設計原則
--------
1. **Point-in-Time (PIT) safe**: 所有 rolling 計算只用過去資料, 絕不偷看未來
2. **Stateless functions**: 每個 transform 是 pure function, 方便單元測試與組合
3. **Vectorised**: 用 pandas/numpy, 不寫 for loop
4. **可在 ML / 因子研究階段直接 reuse**: 因子建構也會用到 Z-score / rank

為什麼 PIT 這麼重要 (Why PIT matters)
--------------------------------------
在 quant pipeline 裡, "未來函數" (look-ahead bias) 是最致命的 bug。
一個簡單例子:

    [錯誤 - 洩漏未來]
        z = (x - x.mean()) / x.std()    # 用了全期 mean, 未來資訊洩漏

    [半對 - rolling 但 inclusive of t]
        z = (x - x.rolling(252).mean()) / x.rolling(252).std()
        # 計算 t 時刻的 z 用了 [t-251, t] 的 mean, 包含 t 本身
        # 適用於 "當前訊號狀態", 但用來預測 t+1 報酬會偏樂觀

    [嚴格 PIT z-score - 視窗排除 t]
        rolling_zscore(x, 252, lag=1)
        # 計算 t 時刻的 z 用了 [t-252, t-1] 的 mean, 嚴格不含 t

lag 參數只作用在 rolling-window 類 transform (z-score / percentile), 控制 mean/std 視窗含不含 t:
- **訊號監控** (dashboard, MON): lag=0, 視窗含 t, 看當下 z 值
- **嚴格 PIT z-score** (MON 若需要): lag=1, 視窗排除 t

⚠️ FM 因子路徑不靠這個 lag: Carhart / CRR86 是 contemporaneous pricing (change_t 對 return_t, 同期不額外 lag),
   FM 用 change transforms (diff / yoy / pct, 都不吃 lag), 資料可得性 (publication lag) 由上游 pit_align 處理。
   → main.py FM 路徑一律 lag=0; lag>0 僅供嚴格 PIT z-score 的 opt-in 場景。
"""

# ==================================================================================================================================
# PIT 在本 pipeline 有兩層:
#   ● 演算法層: rolling / expanding 只用「該點及之前」的資料; lag 參數控制視窗含不含 t
#     （lag=0 = 監控描述現在; lag>0 = 預測 feature —— 用「過去的常態」衡量「今天的異常」）。
#   ● 資料層: 餵 vintage first-release（用當下發佈值, 不被事後修訂竄改）—— 見 pit_safe_vintage。
# ==================================================================================================================================

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional

# ==================================================================================================================================
# Frequency-aware YoY — 為什麼需要 PERIODS_PER_YEAR + RESAMPLE_RULE 兩個 map
# ==================================================================================================================================
# pct_change(n) 只數「列」、不看日期; 同一條 CPI 在 pipeline 會以三種密度出現:
#   native（一月一列）/ sparse（攤到每日, 僅 release 日有值）/ dense（ffill plateau）。
# apply_transforms 拿到的永遠是 sparse 或 dense → 必須先 resample 還原原生頻率, 「一格」才等於「一個原生週期」。
# 數字錨點（CPI 300 → 312, 真 YoY = 4%）:
#   ● native → pct_change(12)                          = 4.00%  ✅（12 格 = 12 個月）
#   ● dense  → pct_change(12)                          = 0.00%  ❌（12 格 = 12 天, 還在同月 plateau）
#   ● dense  → pct_change(252)                         = 3.65%  ❌（252 交易日 ≈ 11.6 個月, 且隨公布時點漂移）
#   ● dense  → dropna().resample("ME").last() 先還原   = 4.00%  ✅
# 設計原則: 頻率來自宣告式 metadata（config.py 為 SSoT）, 絕不從「資料儲存密度」推斷。
# ==================================================================================================================================
PERIODS_PER_YEAR = {"D": 252, "W": 52, "M": 12, "Q": 4}
RESAMPLE_RULE = {"D": None, "W": "W", "M": "ME", "Q": "QE"}


# ==================================================================================================================================
# Z-Score — 兩個獨立的設計維度
# ==================================================================================================================================
#   ● 維度 1 lag（用途）: lag=0 視窗含 t（dashboard 監控, 描述現在）; lag>0 視窗嚴格排除 t（PIT / 預測 feature）。
#     lag>0 時只 shift「mean & std」（歷史基準）, 分子的當前值永遠用原始 series ——
#     語意 = 用「過去的常態」衡量「t 的當前值」偏離幾個標準差（shift 分子會讓評價對象不再是 t）。
#   ● 維度 2 window 型（資料特性）: rolling（最近 252, 給非定態指標——基準結構性漂移, 相對「最近一年」才有意義）
#     vs expanding（全歷史, 給定態指標——回歸恆定中樞, 如 NFCI 家族）; 由 ADF 檢定原始 level 決定（見下方）。
#   兩維度獨立、自由組合。min_periods = window//2 是「資料利用率 vs std 穩定性」的權衡
#   （expanding 用 60 ≈ 一季交易日的啟動門檻）。
# ==================================================================================================================================
def rolling_zscore(
    series: pd.Series,
    window: int = 252,
    min_periods: int | None = None,
    lag: int = 0,
) -> pd.Series:
    """
    Trailing rolling Z-score.

    Z_t = (x_t - mean(x[t-window-lag+1 : t-lag])) / std(x[t-window-lag+1 : t-lag])

    Parameters
    ----------
    window : int
        Lookback window in observations (252 ≈ 1 trading year).
    min_periods : int
        Minimum obs required, default = window // 2.
    lag : int, default 0
        Number of periods to shift the rolling window backwards.
        - lag=0: current-state Z-score for monitoring (rolling window includes t)
        - lag=1: strict PIT for prediction (rolling window strictly < t)

    Examples
    --------
    >>> # For dashboard (current state):
    >>> z = rolling_zscore(hy_oas, 252, lag=0)
    >>>
    >>> # For ML feature (predicting next-day return):
    >>> z_feature = rolling_zscore(hy_oas, 252, lag=1)
    """

    if min_periods is None:
        min_periods = window // 2

    if lag > 0:
        s = series.shift(lag)
    else:
        s = series

    rolling_mean = s.rolling(window, min_periods=min_periods).mean()
    rolling_std = s.rolling(window, min_periods=min_periods).std()

    # Numerator: current value; denominator: shifted history
    return (series - rolling_mean) / rolling_std


def expanding_zscore(
    series: pd.Series,
    min_periods: int = 60,
    lag: int = 0,
) -> pd.Series:
    """
    Expanding-window Z-score（全歷史 [0, t] 基準）。
    適合長期結構性指標，例如：NFCI。
    Same `lag` semantics as rolling_zscore.
    """
    if lag > 0:
        s = series.shift(lag)
    else:
        s = series
    expanding_mean = s.expanding(min_periods=min_periods).mean()
    expanding_std = s.expanding(min_periods=min_periods).std()
    return (series - expanding_mean) / expanding_std


# ====================================================================================================================================
# rolling vs expanding 的選擇 → ADF 檢定「原始 level」:
#   拒絕單根（定態）→ expanding（相對全歷史）; 無法拒絕（非定態）→ rolling（相對最近一年）。
#   transform 後的 feature（z-score / yoy / diff）設計上已近似定態 → 進 FM 前理論上不需再檢;
#   對 transform 後 feature 再跑一次 ADF 屬低成本 double check（見 adf_check.py）。
#   ⭐ ADF 的核心用途是「window 選擇 methodology」, 不是「檢查 FM 的 feature」。
# ====================================================================================================================================


# ====================================================================================================================================
# Year-over-Year & Period Changes（pct_change / diff 皆只往回看, PIT-safe; 參數細節見各函式 docstring）
# ====================================================================================================================================
def yoy(series: pd.Series, periods: int) -> pd.Series:
    """
    Year-over-year % change. `periods` 必填,由呼叫方依 config 宣告的頻率傳入
    (PERIODS_PER_YEAR[frequency]);不再從 index 推斷頻率。

    ⚠️ 必須在「原生頻率」序列上呼叫(月頻傳 12,季頻傳 4)。
       若在 forward-filled 的 daily 網格上算:
         - periods=12  → 變成「12 個交易日」(≈ 2.4 週,值幾近 0)
         - periods=252 → 變成「11.6 個月」(252 交易日 ≠ 整年,且隨公布時點漂移)
       兩者都不是真 YoY。density 還原成原生頻率的責任在呼叫方(apply_transforms / compute_real_m2_yoy)。

    PIT note: pct_change(n) = (x_t - x_{t-n}) / x_{t-n}, 不涉及未來資訊。
    fill_method=None:不在 pct_change 內部補值(native 序列無內部 NaN);
                     同時相容 pandas 2.x(避免 'pad' 預設的 FutureWarning)與 3.0(預設已是 None)。
    """
    return series.pct_change(periods, fill_method=None) * 100


def pct_change_1m(series: pd.Series) -> pd.Series:
    """1-month % change (for daily series). PIT-safe by construction."""
    return series.pct_change(21) * 100


def diff_bps(series: pd.Series, periods: int = 1) -> pd.Series:
    """First difference, scaled to basis points. PIT-safe."""
    return series.diff(periods) * 100


def diff_raw(series: pd.Series, periods: int = 1) -> pd.Series:
    """
    First difference in raw units (NO bps scaling). PIT-safe.

    對映 diff_bps 的「不縮放」版本:
      - diff_bps : x_t − x_{t-n}, 再 ×100 → 把 % 轉 bps (OAS / spread / curve 等利率類)
      - diff_raw : x_t − x_{t-n}, 不縮放    → 保留原始單位 (index 無單位的 NFCILEVERAGE、
                   $bn 量級的 Net Liquidity 等;若誤用 diff_bps 會把 $bn 變動 ×100 = 錯單位)

    PIT note: diff(n) 只往回看,不涉及未來資訊。
    """
    return series.diff(periods)


# ====================================================================================================================================
# Frequency-aware rolling z-score 視窗換算 + fail-loud guard
# ====================================================================================================================================
# 為什麼需要這個 helper：
# rolling z-score 的「視窗長度」我們用「時間長度」命名 (zscore_rolling_1y / 3y / 5y)，而不是寫死點數。原因：
#   (1) SSoT - 頻率只宣告在config的frequency欄一處。視窗點數 = PERIODS_PER_YEAR[freq] × years，
#              不在transform名裡再編一次頻率(否則frequency="W"卻寫"_d"會自相矛盾)。
#   (2) 可讀性 - "_5y"直接讀出「正規化5年」；頻率命名"_m"會藏住「其實是5年」。
#   (3) 維運 - 將來要新窗口(例如：3y)只加一個named transform，跨頻率通用。
#   ⭐guard 動機（fail-loud）：
#     頻率 → 點數的換算「靠信」有風險(e.g. weekly序列被誤當daily稀疏排在日網格 → 視窗算錯)，
#     所以換算前先量「實測間隔中位數」是否吻合宣稱frequency，不符合即raise，不准silent算錯。
# ====================================================================================================================================
PERIODS_PER_YEAR_EXPECTED_GAP_DAYS = {"D": 1, "W": 7, "M": 30, "Q": 91}


def _resolve_zscore_window(series: pd.Series, frequency: str, years: float) -> int:
    """
    時間長度 (years) → rolling 視窗點數,並 fail-loud 驗證宣稱頻率與實測間隔相符。

    window = PERIODS_PER_YEAR[frequency] × years。
    但先驗:series 實測間隔中位數是否落在宣稱 frequency 的合理範圍 (±2×),
    不符即 raise — 防「頻率宣告錯誤 / 稀疏排在日網格」導致 silent 算錯視窗長度。

    Parameters
    ----------
    series : pd.Series
        要算 z-score 的序列 (帶 DatetimeIndex;內部會 dropna 後量間隔)。
    frequency : str
        config 宣告的頻率 ("D"/"W"/"M"/"Q")。
    years : float
        視窗時間長度 (年)。1 → rolling_1y;5 → rolling_5y。

    Returns
    -------
    int
        視窗點數 (= PERIODS_PER_YEAR[frequency] × years, 四捨五入)。
    """
    if frequency not in PERIODS_PER_YEAR:
        raise ValueError(
            f"{series.name}: 未知 frequency '{frequency}',無法換算 z-score 視窗 "
            f"(合法: {sorted(PERIODS_PER_YEAR)})"
        )
    clean = series.dropna()
    if len(clean) >= 3:
        med_gap = clean.index.to_series().diff().dt.days.median()
        expected = PERIODS_PER_YEAR_EXPECTED_GAP_DAYS[frequency]
        if not (0.5 * expected <= med_gap <= 2.0 * expected):
            raise ValueError(
                f"{series.name}: 宣稱 frequency='{frequency}' (預期間隔 ~{expected} 天) "
                f"但實測間隔中位數 {med_gap:.0f} 天 → 頻率宣告與資料不符,拒絕套窗 "
                f"(避免稀疏排在日網格等 silent 算錯視窗; fail-loud)"
            )
    return int(round(PERIODS_PER_YEAR[frequency] * years))


# ====================================================================================================================================
# Spreads & Composite indicators
# ====================================================================================================================================
# ------------------------------------------------------------------------------------------------------------------------------------
# 🔑 composite 函式群的「資料對齊」判準 — 兩個正交問題
# ------------------------------------------------------------------------------------------------------------------------------------
# 問題 A（選對齊方法）: 看「運算對頻率敏不敏感」——
#   ● 往回數 N 期（yoy / diff / pct_change）→ 超敏感（N 的意義依賴頻率）→ 先 dropna().resample(rule).last() 還原原生頻率再算。
#   ● 逐列對齊運算（相減 / 相除 / 組合）→ 不敏感（同列兩邊有值即可）→ .ffill() 把低頻補到高頻網格。
# 問題 B（要不要 .dropna(), 只在走 ffill 那條才問）: 看「NaN 的性質」, 不是頻率差多少——
#   ● 結構性低頻（weekly 值放進 daily 網格, 中間本來就空）→ ffill 填滿（「最新值持續有效」合理）。
#   ● 偶發缺漏（本該有值卻缺, 如 SOFR 某日缺報價）→ dropna（ffill 會造假值, 不誠實）。
#   ● 對齊後頭尾 NaN（兩序列起訖差很多）→ dropna 砍頭尾。
# 反例自查（為何不能互換）: YoY 用 ffill 硬算 = 「今天 vs 12 個交易日前」, 根本不是 YoY;
#   net_liquidity 用 resample = WALCL 週頻化, 白丟 80% 的 daily 觀測; repo_spread 用 ffill = 拿昨天 SOFR 造假今天的 spread。
# 各 composite 的套用: real_m2_yoy → resample 還原月頻; net_liquidity → ffill; repo_spread → dropna;
#   credit_quality_spread → 同源同頻同起點, 直接相減; margin_m2 / sp500_m2 → ffill + dropna（結構性低頻 + 頭尾 NaN）。
# ------------------------------------------------------------------------------------------------------------------------------------
def compute_net_liquidity(
    fed_assets: pd.Series,
    tga: pd.Series,
    on_rrp: pd.Series,
) -> pd.Series:
    """
    Net Liquidity = Fed Total Assets - TGA - ON RRP

    這是 Macro Compass / Andy Constan 等機構在用的核心流動性 proxy.

    單位 (重要)
    --------
    WALCL (fed_assets) 與 TGA (WDTGAL) 都是 "Millions of $"; ON RRP (RRPONTSYD)
    是 "Billions of $". 必須把 fed_assets / 1000 與 tga / 1000 轉成 billions 再相減, 否則差 1000x.
    (TGA 用 WDTGAL, Wed-level / millions → 與 WALCL 各自 /1000)

    PIT note
    --------
    Fed BS (WALCL) 與 TGA (WDTGAL) 都是 H.4.1 週四公布、cover 上週三; ON RRP (RRPONTSYD) 是 daily T+1.
    這個函數本身只做算術, 不處理 publication lag.
    若用作 ML feature, 請先用 pit_safe.apply_publication_lag() 處理輸入。
    """
    aligned = pd.DataFrame(
        {
            "fed_assets": fed_assets,
            "tga": tga,
            "on_rrp": on_rrp,
        }
    ).ffill()
    # WALCL + TGA(WDTGAL) millions → billions, 與 RRP(billions) 同單位 (單位統一 billions)
    return aligned["fed_assets"] / 1000.0 - aligned["tga"] / 1000.0 - aligned["on_rrp"]


def compute_repo_spread(sofr: pd.Series, effr: pd.Series) -> pd.Series:
    """SOFR - EFFR spread, 單位 bps."""
    aligned = pd.DataFrame({"sofr": sofr, "effr": effr}).dropna()
    return (aligned["sofr"] - aligned["effr"]) * 100


def compute_credit_quality_spread(
    ccc_oas: pd.Series,
    bb_oas: pd.Series,
) -> pd.Series:
    """CCC OAS - BB OAS = 信用品質溢酬."""
    return ccc_oas - bb_oas


def compute_real_m2_yoy(m2: pd.Series, cpi: pd.Series) -> pd.Series:
    """
    Real M2 YoY = M2 YoY - CPI YoY  (M2 與 CPI 皆為月頻)

    在原生月頻上計算:先 dropna().resample("MS").last() 還原月頻真值
    (對 sparse、ME 月底戳記、或已 ffill 成 daily 的輸入都成立),再各取 yoy(periods=12) 相減。
    不再對 daily 網格 ffill 後直接呼叫 yoy(那會誤推頻率,見 yoy docstring)。

    戳記慣例（月初 MS）
    --------
    值戳「月初」(MS) 而非月底 (ME):與 M2/CPI level 的 FRED data-date placement (月初)
    及 CPI YoY 的 resample("MS") 慣例一致 → 下游 reindex(daily, ffill) 後, 月中 hover
    顯示「當月」yoy, 不再與同列 level 差一個月。dashboard = latest-revised data-date
    placement (state lag=0), 非 as-known; as-known/PIT 由 features path 負責。
    ⚠️ scoped:只動本函式;全域 RESAMPLE_RULE["M"]="ME" 供 apply_transforms/FM path, 不可改。

    PIT note
    --------
    M2 與 CPI 都是月頻, 且 release 都有 publication lag (M2 ~22 day,
    CPI ~14 day). 用作 ML feature 時務必先 lag (上游 pit_align 處理)。
    """
    n = PERIODS_PER_YEAR["M"]
    rule = "MS"   # 月初戳記 (scoped; 勿用全域 RESAMPLE_RULE["M"]="ME" — 那是 FM path 的規則)
    m2_m = m2.dropna().resample(rule).last() # 把可能被ffill成daily的「假高頻」還原回原生月頻(取每月最後一筆), 值戳月初
    cpi_m = cpi.dropna().resample(rule).last()
    return yoy(m2_m, n) - yoy(cpi_m, n)


def compute_margin_m2(margin_debt: pd.Series, m2: pd.Series) -> pd.Series:
    """
    Margin Debt / M2 ratio = 槓桿相對貨幣供給 (核心槓桿泡沫指標).

    單看 Margin YoY 會被 M2 擴張 confound; 除以 M2 deflate 掉貨幣成長後,
    純槓桿位階才浮現 (MON 端再配 5yr z-score, bucket E)。

    單位
    ----
    margin_debt (FINRA_MARGIN_DEBT) 是 "Millions of $", 月頻(每月最後營業日);
    m2 (M2SL) 是 "Billions of $", 月頻。
    margin / 1000 轉 billions 後相除 → 乾淨無量綱比率(~6%:margin ~$1.3T / M2 ~$21T)。
    ffill 對齊 (兩者皆月頻;ffill 處理 release 日網格錯位)。

    PIT note: 兩者皆有 publication lag (FINRA ~25d, M2 ~22d), ML feature 先 lag。
    """
    aligned = pd.DataFrame({"margin": margin_debt, "m2": m2}).ffill().dropna()
    margin_bn = aligned["margin"] / 1000.0  # millions → billions
    return margin_bn / aligned["m2"]


def compute_sp500_m2(sp500: pd.Series, m2: pd.Series) -> pd.Series:
    """
    SP500 / M2 ratio = 資產估值相對貨幣供給 (asset-coverage).

    單位
    ----
    sp500 (index points) / m2 ("Billions of $") → index-per-billion;
    絕對值無直接意義, 看 trend / 5yr z-score (bucket E)。
    ffill 對齊 (daily SP500 與 monthly M2)。

    PIT note: M2 月頻有 ~22d lag; SP500 daily T+1. ML feature 先 lag。
    """
    aligned = pd.DataFrame({"sp500": sp500, "m2": m2}).ffill().dropna()
    return aligned["sp500"] / aligned["m2"]


def compute_margin_net_credit(
    fc_cash: pd.Series,
    fc_margin: pd.Series,
    margin_debt: pd.Series,
) -> pd.Series:
    """
    Net Credit = Free Credit Cash + Free Credit Margin − Debit Balances
                 (FINRA Customer Margin Balances 三欄的標準組合「Net Credit (2+3-1)」)。

    意義:現金緩衝脆弱度 / margin call 承受力 —— 與 Margin/M2(槓桿存量「高度」)互補的第二維度。
    Net Credit 越高 = 帳戶整體現金緩衝相對融資負債越厚;走低/轉負 = 槓桿吃掉緩衝(脆弱)。

    角色:MON-only(dashboard 顯示層)。FINRA 同家族的 FM universe 名額已給 Margin/M2
          (同源 parsimony + 構念更 US-specific)→ 此指標不進 FM、不進 features。

    單位:FINRA 三欄同為 "Millions of $" → 直接加減、無混算(顯示可 /1000 = $bn)。
    """
    aligned = (
        pd.DataFrame({"fc_cash": fc_cash, "fc_margin": fc_margin, "debit": margin_debt})
        .ffill()
        .dropna()
    )
    return aligned["fc_cash"] + aligned["fc_margin"] - aligned["debit"]


# ====================================================================================================================================
# Convenience: apply transforms based on config registry
# ====================================================================================================================================
def apply_transforms(
    panel: pd.DataFrame,
    indicators,
    mode: str,  # 必填: 'mon' 或 'fm'（無預設 → 需排在有預設的 lag 之前）
    lag: int = 0,
) -> pd.DataFrame:
    """
    Take raw panel + Indicator registry, apply all configured transforms.

    Parameters
    ----------
    lag : int, default 0
        Lag applied ONLY to rolling-window transforms (zscore / percentile);
        change transforms (diff / yoy / pct) 不吃 lag。lag=0 視窗含 t (MON);
        lag>0 視窗排除 t (嚴格 PIT zscore)。FM 路徑用 lag=0 (contemporaneous spine, CRR86);
        FM 的 PIT 由上游 pit_align 處理, 不靠這個 lag。
    mode : {'mon', 'fm'}  (必填)
        選擇讀哪份 transform 清單 (MON/FM 分流):
        - 'mon' → ind.mon_transforms (MON 監控, lag=0)
        - 'fm'  → ind.fm_transforms  (FM 因子,  lag=0; contemporaneous, change transforms 不吃 lag)
        mon/fm 為必填欄, 直接讀;
        其他值 (含 None) → fail-loud raise。
        典型用法: main.py 用 mode='mon', lag=0 產 transformed.parquet;
                  mode='fm', lag=0 產 features.parquet (contemporaneous)。

    Returns
    -------
    DataFrame with columns:
        - <series_id>            : raw level (always emitted)
        - <series_id>_zscore     : if 'zscore_rolling_1y' (rolling 1y, frequency-aware)
        - <series_id>_zscore_3y  : if 'zscore_rolling_3y' (rolling 3y;框架支援,目前無 consumer)
        - <series_id>_zscore_5y  : if 'zscore_rolling_5y' (rolling 5y,如 Margin/M2)
        - <series_id>_zscore_exp : if 'zscore_expanding'  (expanding,定態指標)
        - <series_id>_yoy        : if 'yoy'
        - <series_id>_pct1m      : if 'pct_change_1m'
        - <series_id>_diff       : if 'diff'      (純差分,原始單位;NFCILEVERAGE/$bn Net Liq)
        - <series_id>_diff_bps   : if 'diff_bps'  (差分 ×100 → bps;ΔOAS/Δspread/curve)
        - <series_id>_level      : if 'level'     (顯式 level 欄;list 多值語意統一用)
      rolling 視窗用時間長度命名 (1y/3y/5y),點數由 config frequency 換算 + fail-loud guard。
      未知 transform 名 → fail-loud raise（不 silent skip）。
    """
    out = pd.DataFrame(index=panel.index) # 輸出，跟panel同index
    # per-series 隔離: config 有、panel 沒抓到的欄（loader 失敗等）直接跳過, 一個壞掉不拖垮全部。
    # ⚠️ 輸入密度雙型態（本函式被呼叫兩次、餵不同 panel）:
    #   MON 路徑（fred_panel raw concat）= 稀疏（release 日才有值）; FM 路徑（pit_align 輸出）= 已被上游 ffill 成 dense。
    #   dropna().resample(rule).last() 對兩者皆 robust → 各 transform 一律先還原原生頻率再算,
    #   輸出端才 reindex(ffill) 攤回 daily 網格（本函式不對「輸入」ffill）。
    for ind in indicators:
        sid = ind.series_id
        if sid not in panel.columns:
            continue

        out[sid] = panel[sid] # 先放raw level

        # mode 選清單; mon/fm 為必填欄, 直接讀。
        if mode == "mon":
            tf_list = ind.mon_transforms
        elif mode == "fm":
            tf_list = ind.fm_transforms
        else:
            raise ValueError(f"apply_transforms: mode 必須是 'mon' 或 'fm',收到 {mode!r}")

        for tf in tf_list:
            # ---- z-score 系列:rolling 三窗 (1y/3y/5y) + expanding;全部 frequency-aware ----
            if tf in ("zscore_rolling_1y", "zscore_rolling_3y", "zscore_rolling_5y"):
                # 視窗用「時間長度」命名,點數由 config frequency 換算 (SSoT,不在名裡再編頻率)。
                # 在原生頻率上算 (沿用下方 yoy 模式) 再 ffill 回網格 → weekly/monthly 不會被日網格稀疏算錯。
                years = {"zscore_rolling_1y": 1, "zscore_rolling_3y": 3, "zscore_rolling_5y": 5}[tf]
                suffix = {"zscore_rolling_1y": "zscore",
                          "zscore_rolling_3y": "zscore_3y",
                          "zscore_rolling_5y": "zscore_5y"}[tf]
                freq = ind.frequency
                window = _resolve_zscore_window(panel[sid], freq, years)  # fail-loud:在 raw 上驗頻率
                rule = RESAMPLE_RULE.get(freq)
                native = panel[sid].dropna()
                if rule:
                    native = native.resample(rule).last().dropna()
                out[f"{sid}_{suffix}"] = (
                    rolling_zscore(native, window=window, lag=lag).reindex(out.index, method="ffill")
                )
            elif tf == "zscore_expanding":
                # 定態結構性指標用 expanding (全歷史基準);rolling vs expanding 由 ADF 定。
                # frequency-aware:原生頻率上算再 ffill 回網格;min_periods 也按頻率 = 1 年份
                # (daily 252 / weekly 52 / monthly 12),否則非日頻序列點數 < 252 會 silent 全 NaN。
                freq = ind.frequency
                rule = RESAMPLE_RULE.get(freq)
                native = panel[sid].dropna()
                if rule:
                    native = native.resample(rule).last().dropna()
                out[f"{sid}_zscore_exp"] = (
                    expanding_zscore(native, min_periods=PERIODS_PER_YEAR[freq], lag=lag)
                    .reindex(out.index, method="ffill")
                )
            elif tf == "diff":
                # 純一階差分 (原始單位,不 ×100);frequency-aware → weekly NFCILEVERAGE / $bn Net Liq 正確
                # (若在日網格直接 diff,weekly 序列相鄰列多為 NaN → 結果幾近全 NaN)。
                # lag 與 diff_bps 一致不在此套（level/diff/diff_bps/pct1m 的 FM lag 於下游統一處理）。
                freq = ind.frequency
                rule = RESAMPLE_RULE.get(freq)
                native = panel[sid].dropna()
                if rule:
                    native = native.resample(rule).last().dropna()
                out[f"{sid}_diff"] = diff_raw(native, periods=1).reindex(out.index, method="ffill")
            elif tf == "level":
                # 原樣輸出 level 欄,供 selection 層「list 每元素產一欄」語意統一 (一律讀 <sid>_<transform>);
                # 與 base <sid> 同值但顯式命名。lag 與 diff_bps 一致不在此套 (於下游統一處理)。
                out[f"{sid}_level"] = panel[sid]
            elif tf == "yoy":
                # YoY 在「原生頻率」上算(頻率由 config 宣告),算完再 ffill 回網格。
                #   1. dropna().resample(rule).last() → 還原原生頻率真值(對 sparse / ffilled 都成立)
                #   2. yoy(periods)            → 剛好 N 個原生週期(月頻 12 / 季頻 4),不受 panel 密度影響
                #   3. reindex(..., ffill)     → 把原生頻率 feature 攤回網格;native 無內部 NaN,ffill 才會真生效
                freq = ind.frequency
                rule = RESAMPLE_RULE.get(freq)
                native = panel[sid].dropna()
                if rule:
                    native = native.resample(rule).last().dropna()
                out[f"{sid}_yoy"] = yoy(native, PERIODS_PER_YEAR[freq]).reindex(out.index, method="ffill")
            elif tf == "pct_change_1m":
                out[f"{sid}_pct1m"] = pct_change_1m(panel[sid])
            elif tf == "diff_bps":
                # ΔOAS / Δspread 的 FM 變化量 (daily OAS / curve 用;dense daily 無需 resample)
                out[f"{sid}_diff_bps"] = diff_bps(panel[sid], periods=1)
            else:
                # 不認得的 transform 名 → fail-loud（防 config 打錯字默默無 transform）。
                raise ValueError(f"apply_transforms: 未知 transform {tf!r}(指標 {sid})")
    return out


# ====================================================================================================================================
# smoke test — 定位: 通電檢查（transform 跑得起來、lag 機制有運作）, 非正確性驗證;
# 完整 PIT 正確性由 tests/test_no_lookahead.py 的 12 個 test 負責。
# ====================================================================================================================================
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    idx = pd.date_range("2020-01-01", periods=1000, freq="B") # 造1000個2020-01-01起的daily日期索引，取工作日，跳過週末(freq="B")
    test = pd.Series(rng.standard_normal(1000).cumsum() + 100, index=idx) # 一條從100開始、隨機漫步的假價格序列，配上daily index

    # 對假資料test跑60天兩個版本的rolling z-score
    z_state = rolling_zscore(test, 60, lag=0) # dashboard用：含當期mean & std
    z_pred = rolling_zscore(test, 60, lag=1) # PIT prediction用：mean & std shift一期，不偷看t

    print("Z-score: lag=0 (dashboard) vs lag=1 (PIT prediction):")
    print(pd.DataFrame({"lag=0": z_state, "lag=1": z_pred}).tail())
    print(f"\nMax abs diff: {(z_state - z_pred).abs().max():.4f}")
