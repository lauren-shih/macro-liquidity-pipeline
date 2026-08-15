"""
fetch_long_liquidity.py
=======================
Dashboard-local 深史 for Panel 2「Liquidity Monitor」的 Fed 資產負債表 / 便利機制 series。
跟 fetch_long_fxvolmacro.py / fetch_long_curve.py 同一招：寫到獨立的 ./cache_dashboard/，
main.py 不會 glob 這個 dir → 絕不污染 pipeline、絕不回流 FM / features / canonical。
dashboard.py 的 load_display_deep() 會讀 cache_dashboard/<SID>.parquet combine_first 回 panel，
補出深史（涵蓋 2008 GFC Fed 大放水：WALCL/Reserves 從 2002 起就有，QE1 整段在內）。

要抓的 7 條（FRED 最早覆蓋，括號內為實際起點，跑完看下方印出的真值為準）：
    WALCL            Fed Total Assets        (W, Wed level, ~2002-12)
    WDTGAL           TGA (Treasury G.A.)     (W, Wed level, ~2002-12)
    RRPONTSYD        ON RRP                  (D, ~2003-02；2013 前多為 0)
    WRBWFRBL         Reserve Balances        (W, Wed level, ~2002-12)
    RPONTSYD         SRF / Repo Ops          (D, ~2003；2019 前多為 0)
    WLCFLPCL         Discount Window 借款金額  (W, ~2002；2008 GFC 借款飆升)
    H41RESPPALDKNWW  BTFP                    (W, 2023-03 才有 → 之前一定空白, 正常)

→ Fed 資產負債表四條（WALCL/WDTGAL/RRPONTSYD/WRBWFRBL）≈ 2002 起 = Panel 2 的 x 軸錨點(A1)。
  BTFP(H41RESPPALDKNWW) 2023 前空白無解(它就那時才有)、SRF/RRP 早年多為 0 = 正常。

⚠️ 全部抓 latest-revised（給 dashboard 顯示用）。這些序列非 VINTAGE_SERIES、不進 FM，
   護欄（鎖死）：cache_dashboard 的深史 = dashboard-local，不回流 FM / config / canonical。

跑法（於 src/ 目錄）：
    python fetch_long_liquidity.py
"""
from dotenv import load_dotenv
from fred_loader import FredLoader

load_dotenv()  # 灌 FRED key（.env 的 FRED_API_KEY）

# 獨立 dir：main.py 不 glob → 不污染 pipeline（與其他 fetch_long_* 同）
loader = FredLoader(cache_dir="./cache_dashboard")

SERIES = [
    "WALCL",            # Fed Total Assets
    "WDTGAL",           # TGA
    "RRPONTSYD",        # ON RRP
    "WRBWFRBL",         # Reserve Balances (Wed level)
    "RPONTSYD",         # SRF / Repo Ops
    "WLCFLPCL",         # Discount Window primary credit 借款金額 (was DPCREDIT=利率%, 抓錯)
    "H41RESPPALDKNWW",  # BTFP (2023+)
]
panel = loader.fetch_many(SERIES, start="1990-01-01", force_refresh=True)  # latest-revised

print(f"Panel 2 liquidity long: {panel.shape}  "
      f"{panel.index.min().date()} → {panel.index.max().date()}")
print("--- 各 series 實際覆蓋（BTFP 2023 起、RRP/SRF 早年多 0 = 正常） ---")
for c in SERIES:
    if c in panel.columns and panel[c].notna().any():
        s = panel[c].dropna()
        print(f"  {c:16s} {s.index.min().date()} → {s.index.max().date()}  (n={len(s)})")
    else:
        print(f"  {c:16s} ⚠ 無資料 / 全 NaN（檢查 series_id 或 FRED 授權）")

print("\n存到 cache_dashboard/<SID>.parquet 各一檔（dashboard-local，latest-revised，絕不回流 FM）")
print("→ 跑完回 dashboard.py 看 Panel 2 (rebuild_dashboards.py)，x 軸應從 ~2002 起、含 2008 GFC")
