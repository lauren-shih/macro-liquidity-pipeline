"""
fetch_long_fxvolmacro.py
========================
Dashboard-local 深史 (1996+) for Panel 5「FX & Volatility & CPI/PCE」的 5 條 series：
    DEXTAUS    USD/TWD               (D, FRED ~1983+)
    DTWEXEMEGS Nominal EM USD Index  (D, FRED ~2006+ → 1996 起跑只會回到 2006，正常)
    VIXCLS     VIX                   (D, FRED 1990+)
    CPIAUCSL   CPI                   (M, FRED 1947+)
    PCEPILFE   Core PCE              (M, FRED 1959+)

跟 fetch_long_m2.py / fetch_long_curve.py 同一招：寫到獨立的 ./cache_dashboard/，
main.py 不會 glob 這個 dir → 絕不污染 pipeline、絕不回流 FM / features / canonical。
dashboard.py 的 load_display_*（Panel 5 build 時會加）會讀 cache_dashboard/<SID>.parquet
combine_first 回 panel，補出 1996+ 深史（顯示 1997 亞洲金融 / 2008 / 2020 等段）。

⚠️ VINTAGE 注意：CPIAUCSL / PCEPILFE 是 VINTAGE_SERIES（config line 139），
   但這裡抓的是 **latest-revised**（給 dashboard 顯示用，跟 long-M2 完全一樣的處理）。
   FM 的 CPI / PCE 走 macro_panel_pit 裡的 vintage first-release，兩者物理分離、絕不混。
   護欄（鎖死）：cache_dashboard 的 latest-revised 深史 = dashboard-local，不回流 FM / config。

跑法（於 src/ 目錄）：
    python fetch_long_fxvolmacro.py
"""
from dotenv import load_dotenv
from fred_loader import FredLoader

load_dotenv()  # 灌 FRED key（.env 的 FRED_API_KEY）

# 獨立 dir：main.py 不 glob → 不污染 pipeline（與 fetch_long_m2 / fetch_long_curve 同）
loader = FredLoader(cache_dir="./cache_dashboard")

SERIES = ["DEXTAUS", "DTWEXEMEGS", "VIXCLS", "CPIAUCSL", "PCEPILFE"]
panel = loader.fetch_many(SERIES, start="1996-01-01", force_refresh=True)  # latest-revised

print(f"Panel 5 long: {panel.shape}  {panel.index.min().date()} → {panel.index.max().date()}")
print("--- 各 series 實際覆蓋（DTWEXEMEGS 從 ~2006 起 = FRED 該 index 起點，正常） ---")
for c in SERIES:
    if c in panel.columns and panel[c].notna().any():
        s = panel[c].dropna()
        print(f"  {c:11s} {s.index.min().date()} → {s.index.max().date()}  (n={len(s)})")
    else:
        print(f"  {c:11s} ⚠ 無資料 / 全 NaN（檢查 series_id 或 FRED 授權）")

print("\n存到 cache_dashboard/<SID>.parquet 各一檔（dashboard-local，latest-revised，絕不回流 FM）")
