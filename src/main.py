"""
main.py
=======
Pipeline orchestrator. Run end-to-end:
    1. Fetch all FRED + CFTC indicators
    2. Apply transformations (Z-scores, YoY, etc.)
    3. Persist to Parquet (panel + transforms)
    4. Generate Plotly dashboards
    5. (Optional) Update Excel template

Usage
-----
    # Full refresh
    python src/main.py --start 2018-01-01 --refresh

    # Incremental (default)
    python src/main.py

    # Just rebuild dashboards from cache
    python src/main.py --skip-fetch
"""

# ==================================================================================================================================
# CLI 旗標（改行為靠指令、不改 code）:
#   python main.py                      增量更新（cache-first, 只抓上次之後的新資料 — 最常用）
#   python main.py --refresh            全量重抓（無視 cache; force_refresh 走累積合併, 不覆蓋深史）
#   python main.py --skip-fetch         不連網: 只用現有 cache 重建 panel 與 dashboards 原料
#   python main.py --start 2010-01-01   改抓取起點（預設 2018-01-01）
#   python main.py --skip-cftc          跳過 CFTC, 只跑 FRED
# dotenv 責任分工: load_dotenv 由「entry point」灌一次; os.getenv 由任何被建構的 loader 讀 —
# 順序是「main 先灌、才建構會讀的 loader」。
# ==================================================================================================================================
from __future__ import annotations
import argparse
import logging
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

from config import ALL_INDICATORS, by_source, by_frequency
from fred_loader import FredLoader
from fred_loader_vintage import FredVintageLoader
from cftc_loader import CFTCLoader
from finra_loader import FinraLoader
from transformations import apply_transforms
from pit_safe import pit_align, diagnose_lookahead
from pit_safe_vintage import build_vintage_features
# Dashboards 由 rebuild_dashboards.py 統一產出（in-sync: scorecard + 8-panel; plot_* 在 dashboard.py）。
# main.py = 純 pipeline（fetch → transform → parquet → gate）, 不 build dashboard —
# 單一 dashboard builder, 杜絕兩處各自漂移的 drift。

# ── cache / output 路徑: __file__-based 絕對路徑（CWD-independent）─────────────────────────────
# 相對路徑（"./cache"）綁 CWD → 從不同目錄執行會打到不同 cache（split-brain, 深史 splice 可能被短序列覆蓋）。
# 修在「來源端」（main 算出絕對路徑再傳 loader）而非 loader 端 .resolve() —— 對相對路徑
# .resolve() 仍綁 CWD（只是正規化）, 治本是讓路徑在來源就是絕對的。
SRC_DIR = Path(__file__).resolve().parent              # = .../4.Macro_Pipeline/src
CACHE_DIR = SRC_DIR / "cache"                          # canonical cache（含 OAS 深史 splice）
CACHE_VINTAGE_DIR = SRC_DIR / "cache_vintage"          # ALFRED vintage cache
DEFAULT_OUTPUT_DIR = SRC_DIR / "output"                # canonical output（B7 / check_coverage 讀這）

# ── 日更 fetch 節流 TTL（單一來源）─────────────────────────────────────────────
# 兩個 FRED loader（FredLoader + FredVintageLoader）共用此值,勿在呼叫端各寫 literal（避免 magic number drift）。
# 12h：cache 保鮮期半天 → 每天固定跑一次一定判 STALE 觸發增量抓,執行時間不受 24h TTL 綁而被迫越拖越晚。
# 註：loader 各自預設仍 24h（library 保守值,docstring 為準）；此處為 pipeline override。調整只改這一行。
CACHE_TTL_HOURS = 12

logger = logging.getLogger(__name__)

# parse_args: 終端機旗標 → args 物件（下游讀 args.X 決定行為）
def parse_args():
    p = argparse.ArgumentParser(description="Macro Liquidity Pipeline")
    p.add_argument("--start", default="2018-01-01", help="History start date")
    p.add_argument("--refresh", action="store_true", help="Force full refresh from API")
    p.add_argument("--skip-fetch", action="store_true", help="Use cached data only")
    p.add_argument("--skip-cftc", action="store_true", help="Skip CFTC TFF fetch")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    return p.parse_args()


def run_pipeline(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ==================================================================================================================================
    # ---- Step 1: FRED ----
    # 空殼 fred_panel 宣告在分支外層 = 安全預設: 路徑 A 全失敗 / 路徑 B 無 cache → 維持空 → 下方 .empty 一次擋掉。
    # 路徑 A（連網）: config-driven — 平行抓 by_source("FRED") 全清單, 拼成寬表。
    # 路徑 B（--skip-fetch）: cache-directory-driven — glob cache/*.parquet 離線重建
    #   （逐檔 try/except 隔離; 跳過 cftc_tff_* — CFTC 由 Step 2 另建）。
    # ⚠️ 兩路徑 source of truth 不同: 「改了 config 卻沒重抓」時 --skip-fetch 的 panel 可能 ≠ fresh fetch
    #   （config 移除的指標舊檔仍在 → 仍載入; 新增指標未 fetch → 漏）。正常流程不 drift, 但這是要知道的邊界。
    # ==================================================================================================================================
    fred_panel = pd.DataFrame()  # 安全預設（外層空殼, 見上）
    if not args.skip_fetch:
        load_dotenv()
        loader = FredLoader(cache_dir=str(CACHE_DIR), cache_ttl_hours=CACHE_TTL_HOURS)  # TTL 見上方單一常數
        fred_ids = [i.series_id for i in by_source("FRED")]
        logger.info(f"Fetching {len(fred_ids)} FRED series...")
        fred_panel = loader.fetch_many(
            fred_ids,
            start=args.start,
            force_refresh=args.refresh,)
    else:
        # 路徑 B: 從 cache 離線重建
        cache_files = list(CACHE_DIR.glob("*.parquet"))
        logger.info(f"Loading {len(cache_files)} cached parquets...")
        series_dict = {}
        for f in cache_files:
            sid = f.stem
            if sid.startswith("cftc_tff_"):
                continue
            try:
                series_dict[sid] = pd.read_parquet(f)["value"]
            except Exception:
                logger.warning(f"Skip cache file: {f.name}")
        if series_dict:
            fred_panel = pd.concat(series_dict, axis=1)
            fred_panel.columns = list(series_dict.keys())
            fred_panel.index.name = "date"

    if fred_panel.empty:  # 路徑 A/B 共同守門
        logger.error("No FRED data available. Cannot proceed.")
        return None

    # Persist raw panel
    fred_panel.to_parquet(out_dir / "macro_panel_raw.parquet") # 存原始寬表到磁碟
    logger.info(f"Raw panel: {fred_panel.shape}")


    # ==================================================================================================================================
    # ────────────────────────────────────
    # ---- Step 1.5: Vintage (ALFRED first-release) fetch ----
    # ────────────────────────────────────
    # 抓 VINTAGE_SERIES（CPI/M2/PCE + NFCI×5）的 vintage panel,供 Step 3 features 路徑算 first-release levels。
    # 結構 mirror Step 1 的 path A/B：
    #   ● path A（連網）：FredVintageLoader cache-first 抓（load_dotenv 已在 Step 1 path A 灌過 key → 這裡能讀）。
    #   ● path B（--skip-fetch）：直接 glob cache_vintage/*.parquet 讀進 dict,不 instantiate loader
    #     （loader __init__ 強制要 API key,skip_fetch 不該碰 API → 鏡像 Step 1 path B 的 cache glob 做法）。
    # vintage_dict = {series_id: long-format DataFrame(realtime_start, date, value)};此處只 fetch,對齊在 Step 3。
    # vintage_dict 空（fetch 全失敗 / skip_fetch 無 cache）→ Step 3 的 if vintage_dict 守門 → 退回純 standard features,不崩。
    # ==================================================================================================================================
    vintage_dict = {}
    if not args.skip_fetch:
        vloader = FredVintageLoader(cache_dir=str(CACHE_VINTAGE_DIR), cache_ttl_hours=CACHE_TTL_HOURS)
        vintage_dict = vloader.fetch_vintage_many(force_refresh=args.refresh)
    else:
        for f in CACHE_VINTAGE_DIR.glob("*.parquet"):
            try:
                vintage_dict[f.stem] = pd.read_parquet(f)
            except Exception:
                logger.warning(f"Skip vintage cache file: {f.name}")
    logger.info(f"Vintage panels loaded: {len(vintage_dict)} series")


    # ==================================================================================================================================
    # ---- Step 2: CFTC TFF ----
    # 雙旗標 guard: 要連網（not skip_fetch）且沒單獨跳過（not skip_cftc）才抓。
    # 整段 try/except = bonus 源的 graceful degradation: CFTC 失敗僅 warning、pipeline 續走 —
    #   對比 Step 1 FRED 的 .empty → return None（硬停）。這個對比是設計訊號:
    #   FRED = pipeline 命脈（沒它整條死）; CFTC = bonus 監控源（掛掉不該拖垮 FRED → transforms → dashboard）。
    # 資料流: fetch_all（TFF_CONTRACTS 5 檔）→ build_summary_panel 寬表 → cftc_tff_panel.parquet。
    # ==================================================================================================================================
    cftc_panel = pd.DataFrame()  # 安全預設(同 fred_panel):CFTC skip/失敗 → 空 → 下方 R18 merge 守門跳過
    if not args.skip_fetch and not args.skip_cftc:
        try:
            cftc = CFTCLoader(cache_dir=str(CACHE_DIR))
            cftc_data = cftc.fetch_all(start_date=args.start)
            cftc_panel = cftc.build_summary_panel(cftc_data)
            cftc_panel.to_parquet(out_dir / "cftc_tff_panel.parquet")
            logger.info(f"CFTC TFF panel: {cftc_panel.shape}")
        except Exception as e:
            logger.warning(f"CFTC fetch skipped: {e}")
    elif args.skip_fetch and not args.skip_cftc:   # path B（--skip-fetch 離線重建, 鏡像 FRED/FINRA path B）: 讀 cached CFTC
        fp = out_dir / "cftc_tff_panel.parquet"
        if fp.exists():
            cftc_panel = pd.read_parquet(fp)
            logger.info(f"CFTC TFF panel (cached): {cftc_panel.shape}")

    # ── CFTC TFF（registered）接進 features 路徑 ─────────────────────────────────────────────────────────────
    # lag_map 提前於此建 → 供 CFTC merge 與 Step 3 的 fm_cols 共用單一來源。
    lag_map = {ind.series_id: ind.lag for ind in ALL_INDICATORS if ind.lag is not None}
    # CFTC 是 fixed-lag（lag=3、非 vintage）→ 併進 fred_panel（pit_align 之前）走同一 lag shift,
    #   與 WALCL/NFCI 等週頻 FRED 同路（對比 vintage 併 pit_panel/lag=None/PIT 來自 ALFRED archive）。
    # 只取 registered（在 lag_map）欄 = TFF_10Y/2Y_LEVERAGED → MON（mon=['diff']）+ FM（fm=['diff']）都吃到;
    #   unregistered（5Y/30Y/Ultra10Y）+ asset_mgr 不在 lag_map → 留 cftc_tff_panel.parquet 供 Cluster ⑥ dashboard。
    # apply_transforms 的 diff 是 frequency-aware（freq='W' → resample('W') 還原週頻再差分）→ native-weekly Δ、無日網格鋸齒。
    if not cftc_panel.empty:
        cftc_fm = cftc_panel[[c for c in cftc_panel.columns if c in lag_map]]
        if not cftc_fm.empty:
            obs_pre = {c: int(cftc_fm[c].notna().sum()) for c in cftc_fm.columns}   # CFTC 週頻觀測數
            fred_panel = fred_panel.join(cftc_fm)                                    # left-join on business-day index
            obs_post = {c: int(fred_panel[c].notna().sum()) for c in cftc_fm.columns}  # join 後 in-panel 數
            logger.info(f"CFTC TFF wired into panel: +{cftc_fm.shape[1]} cols {list(cftc_fm.columns)}")
            logger.info(f"  no-drop guardrail: weekly obs {obs_pre} → in-panel {obs_post}（應相等,否則 join 丟了週）")

    # ── FINRA margin debt 接進 pipeline（Cluster ⑤）─────────────────────────────────────────────
    # FINRA 月度 Excel（手動下載覆蓋,無公開 API → finra_loader 只讀+parse,不 fetch）→ margin debt + free credit 三欄。
    #   path A（not skip-fetch）：讀 Excel → 存 finra_panel.parquet;path B（--skip-fetch）：讀 cached parquet（離線重建,鏡像 FRED/vintage）。
    #   margin debt（registered,在 lag_map=25）join 進 fred_panel → MON yoy(dashboard) + FM 原料(pit_panel,餵 factor-model notebook 算 Margin/M2)。
    #   FC_cash/FC_margin 非 config Indicator → 不 join、留 finra_panel.parquet 供 Cluster ⑥ dashboard 算 margin_net_credit（CFTC unregistered 同模式）。
    #   FINRA index = business month-end（每月最後營業日 = reference date）⊂ business-day 網格 → .join() 對齊;no-drop guardrail 驗無月被丟。
    finra_panel = pd.DataFrame()  # 安全預設（讀檔失敗/skip 無 cache → 空 → 下方守門跳過）
    if not args.skip_fetch:
        try:
            finra_panel = FinraLoader().load()
            finra_panel.to_parquet(out_dir / "finra_panel.parquet")
            logger.info(f"FINRA panel: {finra_panel.shape}")
        except Exception as e:
            logger.warning(f"FINRA load skipped: {e}")
    else:
        fp = out_dir / "finra_panel.parquet"
        if fp.exists():
            finra_panel = pd.read_parquet(fp)
            logger.info(f"FINRA panel (cached): {finra_panel.shape}")

    if not finra_panel.empty:
        finra_fm = finra_panel[[c for c in finra_panel.columns if c in lag_map]]  # registered = FINRA_MARGIN_DEBT
        if not finra_fm.empty:
            # FINRA 月度史(1997+)比 fred_panel(2018+ = FRED fetch start)長 → join 只保留 overlap(2018+)。
            # pre-2018 月不屬 pipeline scope:M2 也僅 2018+ → Margin/M2 本就 2018+;dashboard 2018+;FM 窗 2021+;
            #   且 margin_net_credit 讀完整 finra_panel.parquet → 截斷不影響它 → pre-panel 截斷是預期、非 bug。
            # 故 no-drop guardrail 只驗「panel 日期範圍內的 FINRA 月有無被 weekend-misalign 丟」,不把 pre-range 截斷算丟。
            in_range = finra_fm[(finra_fm.index >= fred_panel.index.min())
                                & (finra_fm.index <= fred_panel.index.max())]
            obs_pre = {c: int(in_range[c].notna().sum()) for c in in_range.columns}        # panel 範圍內 FINRA 月數
            full = {c: int(finra_fm[c].notna().sum()) for c in finra_fm.columns}            # FINRA 全史月數(透明化)
            fred_panel = fred_panel.join(finra_fm)                                          # left-join on business-day index
            obs_post = {c: int(fred_panel[c].notna().sum()) for c in finra_fm.columns}      # join 後 in-panel 數
            logger.info(f"FINRA margin debt wired into panel: +{finra_fm.shape[1]} cols {list(finra_fm.columns)}")
            logger.info(f"  no-drop guardrail: 範圍內月 {obs_pre} → in-panel {obs_post}"
                        f"（應相等,否則 weekend-misalign 丟月;FINRA 全史 {full},pre-panel 史不在 scope 不計）")

    # ==================================================================================================================================
    # ────────────────────────────────────
    # ---- Step 3: Transforms ----
    # ────────────────────────────────────
    # Two outputs:
    #   (a) panel_transformed_state.parquet → for dashboards (lag=0)
    #   (b) panel_transformed_features.parquet → for ML (FM contemporaneous;PIT 由 pit_align)
    # ==================================================================================================================================
    transformed_state = apply_transforms(fred_panel, ALL_INDICATORS, lag=0, mode="mon")
    transformed_state.to_parquet(out_dir / "macro_panel_transformed.parquet")
    logger.info(f"Transformed (state, lag=0): {transformed_state.shape}")

    # Build PIT-aligned panel for ML feature use.
    # lag_map 已在 Step 2 後（CFTC merge 前）建好（共用單一來源）；先 filter fred_panel 只留 FM-reach 欄,再 pit_align
    # （pit_align 對未列欄會 raise,故先 filter）。fred_panel 此時已含 CFTC TFF（在 lag_map）→ fm_cols 自動納入。
    fm_cols = [c for c in fred_panel.columns if c in lag_map]
    pit_panel = pit_align(fred_panel[fm_cols], lag_map=lag_map)

    # ── vintage 序列接進 features 路徑 ──────────────────────────────────────────────────────────────────────
    # vintage 序列（CPI/M2/PCE + NFCI×5,全 lag=None → 不在上面的 fm_cols）走 ALFRED first-release path：
    #   build_vintage_features 對齊到 pit_panel「同一個 business-day 網格」（query_dates=pit_panel.index）
    #   → 輸出 daily first-release LEVELS → join 進 pit_panel。
    # 是「add 欄」非「overwrite」：vintage 本就被 fm_cols 排除,pit_panel 裡沒有這些欄可覆寫,且兩邊零欄名碰撞。
    # join 後 apply_transforms(mode='fm') 會自動吃到 vintage 欄,算其 FM 變化量（CPI/M2/PCE yoy、NFCI level+diff）。
    # 結果：FM 對 revised 序列用 first-release（無 revision look-ahead）、unrevised 序列用 calibrated 固定-lag。
    # ⚠️ 只動 features 路徑;state 路徑（上方 transformed_state）維持 latest revised（dashboard monitoring 用最新值合理）。
    # vintage_dict 空 → 守門跳過 → 退回純 standard features（graceful,純內部 gap,權威 parquet 永不缺）。
    if vintage_dict:
        vintage_levels = build_vintage_features(vintage_dict, query_dates=pit_panel.index)
        pit_panel = pit_panel.join(vintage_levels)
        logger.info(
            f"Vintage first-release levels merged: +{vintage_levels.shape[1]} cols {list(vintage_levels.columns)}"
        )

    # ── build-time coverage guard（壞 panel 不落地）─────────────────────────────────────────────
    # archive-spliced 深史 series 覆蓋低於下限 → 寫 parquet 前就 raise。
    # 封住 12-PASS no-look-ahead gate 的盲區: 該 gate 不檢查 credit 覆蓋, splice 被洗時 pipeline 照樣「成功」。
    # 只擋「欄在、值被洗」(first_valid 變晚 / n 變少 = 沖洗特徵);欄整條不在 panel(config 改 lag_map)則 skip,不誤擋。
    # 之後若新增一條 splice 深史 series, 在此加對應 floor。
    COVERAGE_FLOORS = {"BAMLH0A0HYM2": ("2000-01-01", 5000)}  # series_id: (first_valid 須 ≤, 非NaN 筆數下限)
    for _sid, (_max_fv, _min_n) in COVERAGE_FLOORS.items():
        if _sid not in pit_panel.columns:
            continue
        _fv = pit_panel[_sid].first_valid_index()
        _n = int(pit_panel[_sid].notna().sum())
        if _fv is None or _fv > pd.Timestamp(_max_fv) or _n < _min_n:
            raise RuntimeError(
                f"[coverage guard] {_sid} 深史不足:first_valid={_fv}(須 ≤ {_max_fv}), "
                f"non-NaN={_n}(須 ≥ {_min_n}) → 疑似被洗,中止不寫 parquet。還原: 以 cache 備份回復; credit 深史 series 勿以 --refresh 覆蓋。"
            )
    logger.info(f"[coverage guard] protected 深史 OK: {list(COVERAGE_FLOORS)}")

    pit_panel.to_parquet(out_dir / "macro_panel_pit.parquet")
    logger.info(f"PIT-aligned panel: {pit_panel.shape}")

    # FM = contemporaneous (CRR86):change_t 對 return_t,同期不額外 lag。FM 用 change transforms (diff/yoy/pct)
    #   都不吃 lag → lag=0 對 FM 即顯式 contemporaneous; PIT 由上游 pit_align 處理。
    transformed_features = apply_transforms(pit_panel, ALL_INDICATORS, lag=0, mode="fm")
    transformed_features.to_parquet(out_dir / "macro_panel_features.parquet")
    logger.info(f"Transformed (features, contemporaneous): {transformed_features.shape}")

    # ---- Step 4: Dashboards → 由 rebuild_dashboards.py 產出（in-sync scorecard + 8-panel）----
    # main.py 純 pipeline，不在此 build dashboard（避免與 rebuild 面板數 / 檔名 drift）。
    logger.info("Dashboards: 由 rebuild_dashboards.py 產出（main.py 不 build）")

    # ---- Step 5: Print summary ----
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Output directory: {out_dir.absolute()}")
    print(f"Raw panel:        {fred_panel.shape[0]} days × {fred_panel.shape[1]} indicators")
    print(f"Transformed:      {transformed_state.shape[1]} columns (incl. Z-scores, YoY)")
    print(f"PIT features:     {transformed_features.shape[1]} columns (contemporaneous, ML-ready)")
    print(f"Latest date:      {fred_panel.index[-1]:%Y-%m-%d}")
    print("\nLatest snapshots:")
    latest = fred_panel.iloc[-1].dropna()
    for k, v in latest.head(15).items():
        print(f"  {k:<25} {v:>12.4f}")
    print("\nDashboard files:")
    for f in sorted(out_dir.glob("*.html")):
        print(f"  → {f.name}")
    print("=" * 70)
    return transformed_state


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    run_pipeline(args)
