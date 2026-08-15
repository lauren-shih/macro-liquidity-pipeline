"""
fetch_long_curve.py
===================
Dashboard-local 深史 yield curve 序列，給 Panel 4 (Yield Curve & Real Rates) 當長史用。
跟 fetch_long_m2.py 同一招：用 FredLoader 寫到獨立的 ./cache_dashboard/，main.py 不會 glob
→ 絕不污染 pipeline、絕不回流 FM / features / canonical。純 dashboard 顯示。

為什麼能用 FRED（不像 SP500 要 Yahoo）：
  Treasury / TIPS 利率序列（DGS2 / DGS10 / T10Y2Y / T10Y3M / DFII10 / T10YIE）是 FRED 自有、
  無第三方授權限制，長史可直接抓：
    DGS2 / DGS10 / T10Y2Y / T10Y3M  → 1976+（這裡從 1996 起，跟 long-SP500 / long-M2 對齊）
    DFII10 / T10YIE                  → 只有 2003+（TIPS 2003 才發行；FRED 自然從 2003 回，正常）

跑法（於 src/ 目錄）：
    python fetch_long_curve.py
  跑完會在 ./cache_dashboard/ 產生 6 支 parquet（DGS2.parquet 等），dashboard 的
  load_display_curve() 會自動讀它們補長史；不跑 → Panel 4 graceful 退回 panel (2018+)。
"""
from dotenv import load_dotenv
from fred_loader import FredLoader

load_dotenv()  # 灌 FRED key（.env 的 FRED_API_KEY）

SERIES = ["DGS2", "DGS10", "DFII10", "T10YIE", "T10Y2Y", "T10Y3M"]
START = "1996-01-01"  # 跟 long-SP500 / long-M2 對齊（DFII10 / T10YIE 會自然從 2003 起）

loader = FredLoader(cache_dir="./cache_dashboard")  # 獨立 dir，不被 main.py glob → 不污染 pipeline
panel = loader.fetch_many(SERIES, start=START, force_refresh=True)  # latest-revised

print(f"long-curve: {panel.shape}  {panel.index.min().date()} → {panel.index.max().date()}")
print("\n各序列最早有效日 (DFII10 / T10YIE 預期 ~2003, 其餘 ~1996):")
for c in SERIES:
    fv = panel[c].first_valid_index() if c in panel.columns else None
    print(f"  {c:9s} {fv.date() if fv is not None else '(無)'}")
print("\n存到 cache_dashboard/<SID>.parquet（dashboard-local，latest-revised，絕不回流 FM）")
