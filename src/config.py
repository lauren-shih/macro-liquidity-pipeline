# docstring宣告SSoT(Single source of truth)的設計理念
"""
config.py
========
Single source of truth for all macro/liquidity/leverage indicators.

每個指標都帶有:
- series_id: FRED 或 CFTC 的代碼
- source: 'FRED' / 'CFTC'
- frequency: 'D' / 'W' / 'M' / 'Q'
- category: 用於 dashboard 分類
- unit: 原始單位
- transform: 是否需要做 YoY/Z-score/level 轉換
- description: 簡短說明
- thesis: 為什麼追蹤這個指標

設計理念
--------
所有指標放在這一個檔案,讓 pipeline 完全 data-driven。
新增指標只要在 INDICATORS 加一筆,不需要改 pipeline 邏輯。
"""

from dataclasses import dataclass, field
from typing import Optional

# ==========================================================================================================================
# Indicator schema — 把 module docstring 的口頭規格升級成程式強制的 schema:
#   欄位打錯名 / 漏給必填欄, Python 當場報錯（口頭約定寫錯不會抗議, schema 會）; 新增指標照此 class 實例化。
#   ● @dataclass(frozen=True): config 是 SSoT — downstream（loader / transform / dashboard）只能讀不能改;
#     frozen 把這條紀律從「大家自律」變成「Python 強制」（任何地方手滑想改, 當場攔下; 附帶 hashable）。
#   ● mutable 預設值（list / dict / set）一律 field(default_factory=...) — 直接 "= []" 在 class 定義時只建一次、
#     所有 instance 共享（改 A 的 list, B/C/D 全跟著變）。
# ==========================================================================================================================


@dataclass(frozen=True)
class Indicator:
    series_id: str
    name: str  # 顯示名稱
    source: str  # 'FRED' / 'CFTC'
    frequency: str  # 'D' / 'W' / 'M' / 'Q'
    category: str  # 'rates' / 'credit' / 'liquidity' / 'leverage' / 'curve' / 'vol'
    unit: str
    # ==========================================================================================================================
    # 🔑 mon/fm「必填欄」設計 —— 為何必填、為何放 unit 後、為何不用 __post_init__
    # ==========================================================================================================================
    # 【角色分工】
    #   mon_transforms  → MON 監控專屬（dashboard，lag=0）
    #   fm_transforms   → FM 因子專屬（Fama-MacBeth feature；contemporaneous，用 change transforms 不吃 lag）
    #   （apply_transforms 直接讀 mon/fm; 未知 transform 名 → fail-loud raise, 不 silent skip。）
    #
    # 【為何 mon/fm 是「必填欄」（無預設），而非 __post_init__ 檢查】
    #   ⭐ 必填欄 enforce 的是「presence（必須明確指定）」，正是要防的 silent bug ──
    #     「新增 indicator 時忘了填 mon/fm → 默默拿到空 default → dashboard/FM 默默無 transform，不報錯但結果錯」。
    #     設必填欄後 Python 強制建物件時非寫 mon/fm 不可, 忘了當場 crash（loud）, 不可能 silent。
    #   ⭐ 合法的「空」用「明確寫 fm_transforms=[]」表達 ──
    #     MON-only（corridor/episodic）與 raw material（WALCL/TGA/RRP 餵 composite）本無 FM 角色,
    #     fm_transforms=[] 是正確編碼（「考慮過 FM, 結論是無」）。必填欄「允許明確空 []、禁止忘記寫」正是要的語意。
    #   ✗ 不用 __post_init__ raise-if-empty：(1) 它 enforce「non-empty」會誤殺上述合法空;
    #     (2) 有預設值的欄位建完物件後永遠有值（default []）, __post_init__ 分不出「明確 []」vs「忘填拿到 default []」
    #     → 無法 enforce presence。要 enforce「必須明確指定」只能用必填欄。
    #   ✓ 必填欄更簡單（Python 原生、零額外 code）＋ review surface 更小。
    #
    # 【dataclass 欄位順序】無預設值欄位不能排在有預設值欄位之後 → mon/fm（必填）放 unit 後、
    #   description / thesis / lag（有預設）之前。
    # ==========================================================================================================================
    mon_transforms: list[str]  # 必填：MON 監控 transform（lag=0）；無 MON 角色明確寫 []
    fm_transforms: list[str]   # 必填：FM feature transform（contemporaneous，change transforms）；無 FM 角色明確寫 []
    description: str = ""  # 選填(有預設)
    thesis: str = ""  # 選填(有預設)
    # ── lag：publication lag，單位＝營業日（business days），供 FM 因子做 PIT 對齊（pit_safe.pit_align 讀取）。
    #   ● 規則（FM-reach）：只有「會進 FM」的序列才設 lag —— 直接 FM（fm_transforms≠[]）或餵 FM composite 的原料。
    #     純 MON-only（fm=[]、不餵 composite）＝ None（預設）；vintage 序列 ＝ None（走 ALFRED，見 VINTAGE_SERIES）。
    #   ● 完整性由 validate_config() enforce（fm≠[] 且非 vintage 卻缺 lag → raise）。
    #   ● 消費端：main.py 從 config.lag 組 lag_map（只含 FM-reach：lag≠None）傳給 pit_align。
    lag: int | None = None  # 預設 None＝無固定 lag（MON-only / vintage）；FM-reach 才填 int



# =========================================================================================================================
# VINTAGE_SERIES —— ALFRED point-in-time 序列清單（SSoT）
# -------------------------------------------------------------------------------------------------------------------------
# 這些序列因「會被大幅修正（revision）」，PIT 對齊不能靠固定 publication lag，而要用 ALFRED vintage（每個觀測值
# 綁定其「當時真正可得的版本」）。其餘序列走 fred_loader.py 標準路徑 ＋ 固定 lag。
#   ● CPI / M2 / PCE：月頻 macro，release lag 大且常修正（sweep：CPI median≈42d、M2≈55d、PCE≈58d）。
#   ● NFCI 家族 ×5（NFCI/NFCILEVERAGE/NFCIRISK/NFCICREDIT/ANFCI）：週頻，Chicago Fed 回溯修正歷史值
#     （實測：ANFCI dlt_vs_std=0.468、NFCILEVERAGE dlt_corr=0.9644，修正幅度 material）→ 全數納入 vintage。
# SSoT：vintage 成員身分屬 config（indicator metadata）；fred_loader_vintage.py 從這裡 import，不自行定義。
#   vintage 路徑由 main 接線（vintage fetch → features 路徑 first-release levels）；此清單同時供 validate_config 使用。
# =========================================================================================================================
VINTAGE_SERIES: set[str] = {
    "CPIAUCSL", "M2SL", "PCEPILFE",
    "NFCI", "NFCILEVERAGE", "NFCIRISK", "NFCICREDIT", "ANFCI",
}


# ============================================================================
# 1. RATES & REPO PLUMBING (Daily)
# ============================================================================
RATES_REPO = [
    Indicator(
        series_id="SOFR",
        name="SOFR",
        source="FRED",
        frequency="D",
        category="rates",
        unit="%",
        description="Secured Overnight Financing Rate. 以美債作擔保的隔夜融資利率,反映 repo 市場真實成本",
        thesis="SOFR > EFFR 是 dealer balance sheet 緊繃的最強訊號。2019/9 重演要警惕",
        mon_transforms=[],
        fm_transforms=[],
        lag=1,  # composite 原料（repo_spread）→ 為 FM composite 設 lag（R24）
    ),
    Indicator(
        series_id="EFFR",
        name="EFFR",
        source="FRED",
        frequency="D",
        category="rates",
        unit="%",
        description="Effective Federal Funds Rate. 銀行間無擔保隔夜拆借利率",
        thesis="與 SOFR 對比可量化「擔保品溢價」,是回購壓力的純粹指標",
        mon_transforms=[],
        fm_transforms=[],
        lag=1,  # composite 原料（repo_spread）→ 為 FM composite 設 lag（R24）
    ),
    Indicator(
        series_id="IORB",
        name="IORB",
        source="FRED",
        frequency="D",
        category="rates",
        unit="%",
        description="Interest on Reserve Balances. Fed 付給銀行準備金的利息,通常是 rate corridor 的上沿",
        thesis="SOFR 觸及或超過 IORB 代表流動性即將從充裕轉為稀缺,是 NBFI 去槓桿前兆",
        mon_transforms=[],
        fm_transforms=[],
    ),
    Indicator(
        series_id="RRPONTSYAWARD",
        name="ON RRP Award Rate",
        source="FRED",
        frequency="D",
        category="rates",
        unit="%",
        description="Overnight Reverse Repo facility 的得標利率,是 rate corridor 下沿",
        mon_transforms=[],
        fm_transforms=[],
    ),
    Indicator(
        series_id="DFEDTARU",
        name="Fed Funds Target Upper",
        source="FRED",
        frequency="D",
        category="rates",
        unit="%",
        description="FOMC 設定的 federal funds rate 上限",
        mon_transforms=[],
        fm_transforms=[],
    ),
    Indicator(
        series_id="DFEDTARL",
        name="Fed Funds Target Lower",
        source="FRED",
        frequency="D",
        category="rates",
        unit="%",
        description="FOMC 設定的 federal funds rate 下限",
        mon_transforms=[],
        fm_transforms=[],
    ),
    Indicator(
        series_id="TGCRRATE",  # R7 定案:FRED 正確 ID（bare TGCR 不存在;FRED web-verify 2026-06 active）;BGCR 已 drop（FRED 無此 series）
        name="Tri-Party GC Repo Rate",
        source="FRED",
        frequency="D",
        category="rates",
        unit="%",
        description="Tri-Party General Collateral Rate. 比 SOFR 上游、純三方擔保品 GC repo",
        thesis="SOFR 是混合指標,TGCR 直接反映 cleared 市場壓力,2019 repo spike 領先 SOFR",
        mon_transforms=[],
        fm_transforms=[],
    ),
]


# ============================================================================
# 2. ON RRP & FED BALANCE SHEET LIQUIDITY (Daily / Weekly)
# ============================================================================
LIQUIDITY = [
    Indicator(
        series_id="RRPONTSYD",
        name="ON RRP Volume",
        source="FRED",
        frequency="D",
        category="liquidity",
        unit="Billions of $",
        description="Overnight Reverse Repo facility 隔夜參與量。MMF 把錢停在 Fed 而非市場",
        thesis="ON RRP 餘額減少 = 流動性流回市場,是 bullish risk asset 的訊號",
        mon_transforms=[],
        fm_transforms=[],
        lag=1,  # composite 原料（net_liquidity）→ 為 FM composite 設 lag（R24）
    ),
    Indicator(
        series_id="WALCL",
        name="Fed Total Assets",
        source="FRED",
        frequency="W",
        category="liquidity",
        unit="Millions of $",
        description="Fed 資產負債表總額。QT 縮表會反映在這裡",
        thesis="Net Liquidity = WALCL - TGA - ON RRP,是 risk asset 估值的核心驅動",
        mon_transforms=[],
        fm_transforms=[],
        lag=1,  # composite 原料（net_liquidity）→ 為 FM composite 設 lag（R24）
    ),
    Indicator(
        series_id="WDTGAL",
        name="Treasury General Account (TGA, Wed Level)",
        source="FRED",
        frequency="W",
        category="liquidity",
        unit="Millions of $",
        description="美國財政部在 Fed 的存款帳戶（H.4.1 週三水位, millions）。TGA 上升 = 政府從市場抽水",
        thesis="Debt ceiling 解除後 TGA 重建會狂抽流動性,要追 issuance 計畫",
        mon_transforms=[],
        fm_transforms=[],
        lag=1,  # composite 原料（net_liquidity）→ 為 FM composite 設 lag（R24）
    ),
    Indicator(
        series_id="M2SL",
        name="M2 Money Stock",
        source="FRED",
        frequency="M",
        category="liquidity",
        unit="Billions of $",
        description="M2 廣義貨幣供給",
        thesis="M2 YoY 與資產價格高度相關,Real M2 (扣 CPI) 才是真正的「實質流動性」",
        # M2 = 原料:fm=[] 自身不進 FM、mon=[] 顯示 level。
        #   理由:nominal M2 yoy 與 real_m2_yoy(M2 yoy − CPI yoy)近線性相依（多重共線）→ 成長判讀交給 real_m2_yoy（real＝購買力管道,thesis 自己說 Real M2 才是真正實質流動性）。
        #   MON 回 level（dashboard 顯示 M2 存量 level + real_m2_yoy 成長,互補不重複）→ MON/FM 兩側一致收斂為原料,不分裂。
        #   仍為 VINTAGE 序列（first-release）→ 進 pit_panel 當 raw level,餵 real_m2_yoy / Margin-M2 / SP500-M2 composite。
        #   lag 省略（None）＝ vintage 序列豁免固定 lag（走 ALFRED first-release,見 VINTAGE_SERIES）。
        mon_transforms=[],
        fm_transforms=[],
    ),
    Indicator(
        series_id="WRBWFRBL",
        name="Reserve Balances at Fed (Wed. Level)",
        source="FRED",
        frequency="W",
        category="liquidity",
        unit="Millions of $",
        description="銀行存放 Fed 的準備金（週三 level，非週平均）。Lorie Logan/Powell 都關注的「ample reserves」核心指標",
        thesis="準備金 / GDP 比率跌破 8-10% 是 NY Fed 認定的「scarcity」門檻",
        # WRESBAL（週平均）→ WRBWFRBL（週三 level）取最新 snapshot；FRED 單位皆 Millions of $（ground-truth verified，非 Billions）。
        # lag 省略（None）＝ 純監控、不進 FM（不在 fm_cols / pit_panel / B7）；dashboard /1e6 → Tn。
        mon_transforms=[],
        fm_transforms=[],
    ),
    Indicator(
        series_id="RPONTSYD",
        name="Standing Repo Facility (SRF) Use",
        source="FRED",
        frequency="D",
        category="liquidity",
        unit="Billions of $",
        description="Fed 的 SRF 使用量,是流動性緊縮時的最後防線",
        thesis="SRF 一旦被啟用代表市場吃緊,2024/9 quarter-end 已開始試水溫",
        mon_transforms=[],
        fm_transforms=[],
    ),
    Indicator(
        series_id="WLCFLPCL",                       # was DPCREDIT(=利率%, 抓錯); WLCFLPCL=H.4.1 primary credit 借款金額
        name="Discount Window Primary Credit",
        source="FRED",
        frequency="W",                              # H.4.1 Wednesday level, 週四發布
        category="liquidity",
        unit="Millions of $",                       # FRED native (原寫 Billions 也錯); dashboard /1000 → Bn
        description="Fed 貼現窗口主要信貸的借款餘額,銀行流動性壓力的最後管道",
        thesis="2023 SVB 危機曾飆破 $153B,是銀行 plumbing 壓力的真實 readings",
        mon_transforms=[],
        fm_transforms=[],
    ),
    Indicator(
        series_id="H41RESPPALDKNWW",
        name="BTFP Outstanding",
        source="FRED",
        frequency="W",
        category="liquidity",
        unit="Millions of $",
        description="BTFP（Bank Term Funding Program）餘額。Lifecycle：2023-03 SVB 危機後實施 → 2024-01 峰值 167,768 $mn(~$168bn) → 存量隨 1 年期貸款到期遞減 → 2025-03 末次非零、現 DISCONTINUED 歸零。MON-only、不進 FM；config+dashboard 保留並標 DISCONTINUED = for historical check",
        thesis="BTFP 退出時程影響中型銀行流動性,需追蹤其加權成本上升",
        # MON-only（mon=[]/fm=[]）:raw level 進 transformed_state 供 dashboard 歷史檢視;lag=None → 不在 pit_panel → 不進 features。
        # loader graceful:FredLoader per-series error isolation → 未來若 series delisted/discontinued 不 crash（loud-skip 該條、不影響其餘）。
        # 起訖月份由 H41 cache ground-truth（首非零 2023-03 / 峰 2024-01=167,768 $mn / 末非零 2025-03 / 現 0 @2026-05）;dashboard DISCONTINUED annotation 留 Cluster ⑥。
        mon_transforms=[],
        fm_transforms=[],
    ),
]


# ============================================================================
# 3. CREDIT SPREADS (Daily)
# ============================================================================
CREDIT = [
    Indicator(
        series_id="BAMLH0A0HYM2",
        name="HY OAS",
        source="FRED",
        frequency="D",
        category="credit",
        unit="%",
        description="ICE BofA US High Yield OAS。垃圾債相對國債的 option-adjusted spread",
        thesis="Z-score > 2 通常領先 SPX -10%,是信用循環反轉的最強訊號",
        # HY active;mon z=expanding 依 SPEC R21b (29yr splice 後 stationary)
        mon_transforms=["zscore_expanding"],
        fm_transforms=["diff_bps"],
        lag=1,  # ICE BofA 收盤後計算、次日發布（computed-after-close）
    ),
    Indicator(
        series_id="BAMLC0A0CM",
        name="IG Corp OAS",
        source="FRED",
        frequency="D",
        category="credit",
        unit="%",
        description="ICE BofA US Corporate Index OAS (Investment Grade)",
        thesis="IG 比 HY 更早出現信用緊縮跡象,因為 IG 流動性差(insurance/pension hold-to-maturity)",
        # IG: excluded from the FM set (usable history T<~60m); MON z-score uses rolling 1y (3yr window non-stationary)
        mon_transforms=["zscore_rolling_1y"],
        fm_transforms=[],
    ),
    Indicator(
        series_id="BAMLH0A1HYBB",
        name="BB HY OAS",
        source="FRED",
        frequency="D",
        category="credit",
        unit="%",
        description="ICE BofA BB US High Yield OAS。HY 內最高品質",
        mon_transforms=[],
        fm_transforms=[],
    ),
    Indicator(
        series_id="BAMLH0A3HYC",
        name="CCC HY OAS",
        source="FRED",
        frequency="D",
        category="credit",
        unit="%",
        description="ICE BofA CCC & Lower US High Yield OAS",
        thesis="CCC OAS / BB OAS 的「品質溢酬」比率是違約週期最敏感指標",
        mon_transforms=[],
        fm_transforms=[],
    ),
    Indicator(
        series_id="BAMLEMHYHYLCRPIUSOAS",
        name="EM HY OAS",
        source="FRED",
        frequency="D",
        category="credit",
        unit="%",
        description="ICE BofA EM HY USD OAS。新興市場高收益債",
        thesis="EM HY 是 global risk-off 的早期訊號,leads US HY by 5-10 days in stress",
        # EM: excluded from the FM set (same rationale as IG)
        mon_transforms=["zscore_rolling_1y"],
        fm_transforms=[],
    ),
]


# ============================================================================
# 4. YIELD CURVE & REAL RATES (Daily)
# ============================================================================
CURVE_RATES = [
    Indicator(
        series_id="DGS2",
        name="2Y Treasury",
        source="FRED",
        frequency="D",
        category="curve",
        unit="%",
        description="2-Year Treasury Constant Maturity Yield",
        mon_transforms=[],
        fm_transforms=[],
    ),
    Indicator(
        series_id="DGS10",
        name="10Y Treasury",
        source="FRED",
        frequency="D",
        category="curve",
        unit="%",
        description="10-Year Treasury Constant Maturity Yield",
        mon_transforms=[],
        fm_transforms=[],
    ),
    Indicator(
        series_id="T10Y2Y",
        name="10Y-2Y Spread",
        source="FRED",
        frequency="D",
        category="curve",
        unit="%",
        description="10Y - 2Y yield curve slope",
        thesis="倒掛 → un-invert 才是衰退訊號實際 trigger,2024/9 重新陡峭化已開始",
        mon_transforms=[],
        fm_transforms=["diff_bps"],
        lag=0,  # market-priced（收盤即時）→ pub lag=0；cross-market +1 在 FM notebook（SOP/R19）
    ),
    Indicator(
        series_id="T10Y3M",
        name="10Y-3M Spread",
        source="FRED",
        frequency="D",
        category="curve",
        unit="%",
        description="10Y - 3M Treasury spread。NY Fed 偏好的衰退領先指標",
        thesis="NY Fed Recession Probability 模型核心輸入;殖利率倒掛是市場最關注的領先衰退訊號之一",
        mon_transforms=[],
        fm_transforms=["diff_bps"],
        lag=0,  # market-priced（收盤即時）→ pub lag=0；cross-market +1 在 FM notebook（SOP/R19）
    ),
    Indicator(
        series_id="DFII10",
        name="10Y Real Yield",
        source="FRED",
        frequency="D",
        category="curve",
        unit="%",
        description="10-Year TIPS yield (real interest rate)",
        thesis="Real yield 是 long-duration risk asset 估值的核心折現率",
        mon_transforms=[],
        fm_transforms=["diff_bps"],
        lag=1,  # H.15 次日可得（ALFRED 實測 T+1）
    ),
    Indicator(
        series_id="T10YIE",
        name="10Y Breakeven",
        source="FRED",
        frequency="D",
        category="curve",
        unit="%",
        description="10Y nominal - 10Y TIPS = market-implied inflation expectation",
        mon_transforms=[],
        fm_transforms=["diff_bps"],
        lag=1,  # market-priced 收盤即知；+1 = strictly-prior 保守日（ALFRED 實測 T+0）
    ),
]


# ============================================================================
# 5. VOLATILITY (Daily)
# ============================================================================
VOL = [
    Indicator(
        series_id="VIXCLS",
        name="VIX",
        source="FRED",
        frequency="D",
        category="vol",
        unit="index",
        description="CBOE Volatility Index。S&P 500 30-day implied vol",
        thesis="VIX < 12 是極度 complacency,通常先於 vol shock",
        mon_transforms=["zscore_expanding"],
        fm_transforms=["diff"],
        lag=1,  # market-priced 收盤即知；+1 = strictly-prior 保守日（ALFRED 歸檔實測 T+0）
    ),
]


# ============================================================================
# 6. LEVERAGE & FINANCIAL CONDITIONS (Weekly / Monthly / Quarterly)
# ============================================================================
LEVERAGE = [
    Indicator(
        series_id="NFCILEVERAGE",
        name="NFCI Leverage Subindex",
        source="FRED",
        frequency="W",
        category="leverage",
        unit="index",
        description="Chicago Fed NFCI Leverage Subindex。整體金融體系槓桿",
        thesis="> 0 即高於歷史均值, > 1 為系統性壓力顯著",
        mon_transforms=[],
        fm_transforms=["level", "diff"],
    ),
    Indicator(
        series_id="NFCI",
        name="NFCI Headline",
        source="FRED",
        frequency="W",
        category="leverage",
        unit="index",
        description="Chicago Fed National Financial Conditions Index 主指標",
        thesis="最權威的 financial conditions 綜合指標,Fed 自己也看",
        mon_transforms=[],
        fm_transforms=["level", "diff"],
    ),
    Indicator(
        series_id="NFCIRISK",
        name="NFCI Risk Subindex",
        source="FRED",
        frequency="W",
        category="leverage",
        unit="index",
        description="NFCI 風險子指數,衡量市場 risk premia",
        mon_transforms=[],
        fm_transforms=["level", "diff"],
    ),
    Indicator(
        series_id="NFCICREDIT",
        name="NFCI Credit Subindex",
        source="FRED",
        frequency="W",
        category="leverage",
        unit="index",
        description="NFCI 信貸子指數,信貸標準寬鬆/緊縮",
        mon_transforms=[],
        fm_transforms=["level", "diff"],
    ),
    Indicator(
        series_id="ANFCI",
        name="Adjusted NFCI",
        source="FRED",
        frequency="W",
        category="leverage",
        unit="index",
        description="剔除商業循環影響的 NFCI,純粹的金融條件",
        thesis="ANFCI > 0 代表「在當前 GDP/inflation 條件下,金融條件異常緊」",
        mon_transforms=[],
        fm_transforms=["level", "diff"],
    ),
    Indicator(
        series_id="FINRA_MARGIN_DEBT",
        name="FINRA Margin Debt",
        source="FINRA",
        frequency="M",
        category="leverage",
        unit="Millions of $",
        description="FINRA Customer Debit Balances in securities margin accounts（每月最後營業日;月頻、M+第三週發布;無回溯修訂）",
        thesis="融資餘額 / M2 Z-score > 2 在 2000、2007、2021 都標記了泡沫頂部",
        # 取代已退役的 BOGZ1FL663067003Q（Z.1 broker receivables,季頻、落後 ~5 月、定義≠margin debt）。
        #      raw margin debt = 原料（mon=['yoy'] dashboard;fm=[] 自身不進 FM）→ 餵 Margin/M2（universe）+ margin_net_credit（MON-only）。
        #      lag=25:M+第三週（~15-21d）保守 round-up,月頻下對 merge 無差;非實測值,實測 pin 待後續。
        lag=25,
        mon_transforms=["yoy"],
        fm_transforms=[],
    ),
]


# ============================================================================
# 7. FX (Daily) — Cross-currency channel for Taiwan equity / RoW flows
# ============================================================================
# 設計理由 (Why FX deserves its own block):
#   FX 不是 macro reference 也不是 leverage / credit, 而是獨立的 cross-asset channel:
#   - 對 Taiwan equity 而言, USD/TWD 是 foreign capital inflow / outflow 的直接 proxy
#   - 對 macro RHS Fama-MacBeth test 而言, FX return 是與 SOFR / HY OAS 並列的
#     獨立 macro factor (與 US-side liquidity / credit 維度不重疊)
#   - 後續若擴展到 EUR/USD, DXY 等也都歸在這個 block
FX = [
    Indicator(
        series_id="DEXTAUS",
        name="USD/TWD",
        source="FRED",
        frequency="D",
        category="fx",
        unit="TWD per USD",
        description="Taiwan dollars per U.S. dollar. NY 12pm noon buying rate, FRED H.10 release",
        thesis="FX channel for foreign capital flows into Taiwan equity. Macro RHS factor "
        "for cross-sectional pricing test (Layer 1 Fama-MacBeth integration)",
        mon_transforms=["pct_change_1m"],
        fm_transforms=["pct_change_1m"],
        lag=1,  # market-priced：資訊時點＝市場日＋strictly-prior；ALFRED 實測 5＝H.10 週度歸檔節奏，非資訊時點
    ),
    Indicator(
        series_id="DTWEXEMEGS",
        name="Nominal EM USD Index",
        source="FRED",
        frequency="D",
        category="fx",
        unit="index",
        description="Nominal Broad U.S. Dollar Index against Emerging Market Economies. "
        "Trade-weighted USD vs EM basket, FRED H.10 release",
        thesis="Broader EM USD context, collinear FX alternative to DEXTAUS. "
        "FM role: C-univ — universe/sensitivity, NOT an active candidate "
        "(DEXTAUS is the active FX channel candidate; DTWEXEMEGS only appears in the "
        "40-indicator universe IC sweep, high correlation w/ DEXTAUS → VIF would cut). "
        "門檻 Y: FX influences liquidity via a channel, not a direct measure → not prior-promoted.",
        mon_transforms=["pct_change_1m"],
        fm_transforms=["pct_change_1m"],
        lag=5,  # compiled trade-weighted 指數，修正較慢（R15）
    ),
]


# ============================================================================
# 8. INFLATION & MACRO REFERENCE (Monthly)
# ============================================================================
MACRO = [
    Indicator(
        series_id="CPIAUCSL",
        name="CPI",
        source="FRED",
        frequency="M",
        category="macro",
        unit="index",
        description="Consumer Price Index, All Urban Consumers",
        mon_transforms=["yoy"],
        fm_transforms=["yoy"],
    ),
    Indicator(
        series_id="PCEPILFE",
        name="Core PCE",
        source="FRED",
        frequency="M",
        category="macro",
        unit="index",
        description="Personal Consumption Expenditures: Chain-type Price Index Excluding "
        "Food and Energy. BEA release, ~30 days after reference month",
        thesis="Fed policy reaction variable since 2000 (FOMC explicit) — supersedes CPI for "
        "rate-decision modeling, more stable than headline CPI (excludes volatile food + energy). "
        "FM role: C-active — one of the 8 active candidates, competitive (Layer 1 IC + Layer 2 VIF), "
        "NOT prior-promoted. 門檻 Y: inflation influences leverage via the real-rate channel, "
        "it is not a direct measure of liquidity/leverage/credit → competes, prior 不壓秤.",
        mon_transforms=["yoy"],
        fm_transforms=["yoy"],
    ),
    Indicator(
        series_id="SP500",
        name="S&P 500",
        source="FRED",
        frequency="D",
        category="macro",
        unit="index",
        description="S&P 500 Index",
        mon_transforms=["pct_change_1m"],
        fm_transforms=["pct_change_1m"],
        lag=0,  # market-priced（收盤即時）→ pub lag=0；cross-market +1 在 FM notebook（SOP/R19）
    ),
]


# ============================================================================
# 9. CFTC TFF Report (Bonus, non-FRED)
# ============================================================================
# Source: CFTC Commitments of Traders - Traders in Financial Futures (TFF)
# https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm
CFTC_TFF = [
    Indicator(
        series_id="TFF_10Y_LEVERAGED",
        name="Leveraged Funds Net Position - 10Y UST Futures",
        source="CFTC",
        frequency="W",
        category="cftc",
        unit="contracts",
        description="CFTC TFF report: Leveraged Funds 在 10Y 國債期貨的淨部位",
        thesis="淨空部位近年極端水位,反映 hedge fund basis trade 規模,是系統性風險源",
        # Δz(ADF) 待 R18 nested transform;expanding/rolling 待 R18 ADF on Δ
        mon_transforms=["diff"],
        fm_transforms=["diff"],
        lag=3,  # CFTC TFF：Tue as-of → Fri 發布（R18）
    ),
    Indicator(
        series_id="TFF_2Y_LEVERAGED",
        name="Leveraged Funds Net Position - 2Y UST Futures",
        source="CFTC",
        frequency="W",
        category="cftc",
        unit="contracts",
        description="CFTC TFF report: Leveraged Funds 在 2Y 國債期貨的淨部位",
        # Δz(ADF) 待 R18 nested transform;expanding/rolling 待 R18 ADF on Δ
        mon_transforms=["diff"],
        fm_transforms=["diff"],
        lag=3,  # CFTC TFF：Tue as-of → Fri 發布（R18）
    ),
]


# =========================================================================================================================
# Master list — 9 個 block 子清單首尾相接成總表（40 個 Indicator）; 全大寫 = 常數慣例（唯讀約定）。
# =========================================================================================================================
ALL_INDICATORS: list[Indicator] = RATES_REPO + LIQUIDITY + CREDIT + CURVE_RATES + VOL + LEVERAGE + FX + MACRO + CFTC_TFF


# =========================================================================================================================
# Query helpers — 「找一群」用 list comprehension（掃全表收集, 空結果回 []）;
# 「找唯一」（get）用 for + early return（series_id 唯一, 找到即停）, 回傳 Indicator | None 對應「找到 / 找不到」。
# =========================================================================================================================
def by_category(cat: str) -> list[Indicator]:
    """Filter indicators by category."""
    return [ind for ind in ALL_INDICATORS if ind.category == cat]


def by_source(src: str) -> list[Indicator]:
    """Filter indicators by data source."""
    return [ind for ind in ALL_INDICATORS if ind.source == src]


def by_frequency(freq: str) -> list[Indicator]:
    """Filter by frequency."""
    return [ind for ind in ALL_INDICATORS if ind.frequency == freq]


def get(series_id: str) -> Indicator | None:
    """Lookup single indicator by series_id."""
    for ind in ALL_INDICATORS:
        if ind.series_id == series_id:
            return ind
    return None


# =========================================================================================================================
# validate_config —— FM lag 完整性檢查（fail-loud）
# -------------------------------------------------------------------------------------------------------------------------
# 不變量：每個「會進 FM」（fm_transforms≠[]）且「非 vintage」的指標，都必須明確宣告 publication lag（lag≠None）。
#   ● 只查 fm≠[]：lag 是給 FM 因子做 PIT 對齊用的（MON 監控 lag=0，不需要）；純 MON-only（fm=[]）預設 lag=None 合法。
#   ● 豁免 vintage：vintage 序列（CPI/M2/PCE/NFCI×5）用 ALFRED point-in-time 解決 PIT，不靠固定 lag，故 lag=None 合法。
#   ● composite 原料（SOFR/EFFR/WALCL/WDTGAL/RRPONTSYD）：fm=[] 但有設 lag（餵 FM composite 用），此處不強制檢查
#     （它們的 FM-reach 是間接的；lag 值依各 series 的官方發布時程設定）。
#   ● 設計：lag「結構選填（預設 None）、語意上 FM-reach 必填」。用 validate_config 在「用 config 時」（__main__/main）
#     enforce presence —— 將來新增 FM 指標若忘了設 lag，當場 raise（loud），不會 silent 拿到 None 而錯誤對齊。
#   ● 為何不在 import 時跑：避免與 pytest 耦合（test import config 不該觸發驗證）；改在 python config.py / main.py 啟動時跑。
# =========================================================================================================================
def validate_config(verbose: bool = False) -> None:
    """Fail-loud：每個 FM-reaching（fm_transforms≠[]）非 vintage 指標都必須宣告 publication lag。
    在 module 載入時自動呼叫（fail at load）：成功靜默、失敗 raise。
    verbose=True（僅 __main__）才印健檢摘要，避免 import 時噴 print。"""
    missing = [
        ind.series_id
        for ind in ALL_INDICATORS
        if ind.fm_transforms and ind.lag is None and ind.series_id not in VINTAGE_SERIES
    ]
    if missing:
        raise ValueError(
            f"validate_config：{len(missing)} 個 FM 指標缺 publication lag "
            f"（請設 Indicator.lag；若為 ALFRED vintage 序列請加進 VINTAGE_SERIES）：{missing}"
        )
    if verbose:
        n_fm = sum(1 for ind in ALL_INDICATORS if ind.fm_transforms)
        n_vintage_fm = sum(1 for ind in ALL_INDICATORS if ind.fm_transforms and ind.series_id in VINTAGE_SERIES)
        print(f"  ✓ validate_config: {n_fm} 個 FM 指標 lag 完整（{n_vintage_fm} 個 vintage 豁免）")


# 在 module 載入時驗證（fail at load）：任何 import config 都會觸發。
# 成功靜默（不印，避免污染 import）；失敗 raise（漏設 lag 當場爆，不必等 main 跑）。
validate_config()


# =========================================================================================================================
# Quick sanity check — 只在 `python config.py` 直接執行時跑（被 import 不觸發, 避免污染 import）;
# validate_config 本身在 module 載入時已自動執行（成功靜默、失敗 raise）, 此處 verbose=True 額外印健檢摘要。
# =========================================================================================================================
if __name__ == "__main__":
    print(f"Total indicators: {len(ALL_INDICATORS)}")
    for src in ["FRED", "CFTC", "FINRA"]:
        n = len(by_source(src))
        print(f"  {src}: {n}")
    for cat in ["rates", "liquidity", "credit", "curve", "vol", "leverage", "fx", "macro", "cftc"]:
        n = len(by_category(cat))
        print(f"    {cat}: {n}")
    print()
    validate_config(verbose=True)
