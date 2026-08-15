r"""
fetch_long_cftc.py
==================
Dashboard-local 深史 CFTC TFF Leveraged Funds 淨部位 (2Y / 10Y UST 期貨)，給 Panel 7
(Leveraged Funds Basis Trade Monitor) 當長史用。
跟 fetch_long_curve.py / fetch_long_m2.py 同一招：寫到獨立 ./cache_dashboard/，main.py 不會 glob
→ 絕不污染 pipeline、絕不回流 FM / features / canonical。純 dashboard 顯示 (Architecture B)。

為什麼能拉到 2006：
  CFTC Disaggregated / TFF 報告歷史資料可回溯到 2006-06-13 (CFTC 官方)。
  CFTCLoader.fetch_contract(code, start_date) 直接帶 start_date='2006-01-01' 即可 (Socrata API)。
  contract code: 042601 = UST 2Y Note → TFF_2Y_LEVERAGED;043602 = UST 10Y Note → TFF_10Y_LEVERAGED
    (與 cftc_loader.TFF_CONTRACTS / config series_id 一致)。
  欄位 lev_net = Leveraged Funds 淨部位 (long − short)，與 build_summary_panel 的 canonical 同。

顯示窗 vs baseline 分離 (見 dashboard plot_basis_trade)：
  這支抓滿 2006 = 餵 R2/R3 expanding z 的 baseline (有 ~9 年暖機 → 顯示窗 2015 一開窗即穩)；
  dashboard anchor 下限 = 2015 (實際顯示從 2015；R4 repo 因 SOFR 2018-04 才有 → 2015–2018 repo area 空白可接受)。
  fetch 拉滿 2006 = 一次抓好，未來若要做更早的部位分析直接重用 (不用再 fetch)。

跑法（於 src/ 目錄）：
    python fetch_long_cftc.py
  跑完會在 ./cache_dashboard/ 產生 TFF_2Y_LEVERAGED.parquet + TFF_10Y_LEVERAGED.parquet，
  dashboard 的 load_display_deep() 會自動讀它們補長史；不跑 → Panel 7 graceful 退回 panel (2018+)。
"""
from pathlib import Path

from cftc_loader import CFTCLoader

START = "2006-01-01"                                      # CFTC TFF 歷史回溯起點 (官方 2006-06-13)
CACHE = Path("./cache_dashboard")                         # 獨立 dir，不被 main.py glob → 不污染 pipeline
CACHE.mkdir(exist_ok=True)

# contract code → canonical sid (與 cftc_loader.TFF_CONTRACTS / config series_id 一致)
CONTRACTS = {"042601": "TFF_2Y_LEVERAGED", "043602": "TFF_10Y_LEVERAGED"}

loader = CFTCLoader()
print(f"fetch-long CFTC TFF (start={START}) → cache_dashboard/  (純顯示，絕不回流 FM)\n")
for code, sid in CONTRACTS.items():
    df = loader.fetch_contract(code, start_date=START)   # date-indexed weekly; 含 lev_net
    s = df["lev_net"].rename(sid)                        # Leveraged Funds 淨部位 (= canonical TFF_*_LEVERAGED)
    s.to_frame().to_parquet(CACHE / f"{sid}.parquet")    # 單欄 parquet → load_display_deep 讀 deep[sid]
    fv = s.first_valid_index()
    print(f"  {sid:20s} {fv.date() if fv is not None else '(無)'} → {s.index.max().date()}  "
          f"n={int(s.notna().sum())} (weekly)")

print("\n存到 cache_dashboard/TFF_*_LEVERAGED.parquet (dashboard-local，latest，絕不回流 FM)")
print("→ 重跑 rebuild_dashboards.py 即可看到 Panel 7 顯示從 2015、R2/R3 z 從 2015 開窗即穩。")
