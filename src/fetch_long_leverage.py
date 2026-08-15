"""
fetch_long_leverage.py
======================
Dashboard-local 深史 for Panel 6「Leverage Monitor」的 NFCI 家族 5 條 series。
跟 fetch_long_liquidity.py / fetch_long_fxvolmacro.py 同一招：寫到獨立的 ./cache_dashboard/，
main.py 不會 glob 這個 dir → 絕不污染 pipeline、絕不回流 FM / features / canonical。
dashboard.py 的 load_display_deep() 會讀 cache_dashboard/<SID>.parquet combine_first 回 panel，
補出深史（NFCI 家族 FRED 可達 1971；Panel 6 錨 1997 → cover dotcom / GFC / COVID / Fed 暴力升息）。

要抓的 5 條（FRED 週頻 W, 1971+；跑完看下方印出真值為準）：
    NFCILEVERAGE   NFCI Leverage Subindex   （Row 1 主角：系統性金融槓桿 debt & equity，thr 2）
    NFCI           NFCI Headline            （Row 2：105-indicator 綜合條件指標，Fed 自己也看）
    NFCIRISK       NFCI Risk Subindex       （Row 2：波動度 & 資金風險溢酬）
    NFCICREDIT     NFCI Credit Subindex     （Row 2：信貸標準 & 利差）
    ANFCI          Adjusted NFCI            （Row 2：剔除 GDP/inflation 商業循環後的純金融條件）

關係（順帶）：NFCI(headline) ≈ NFCILEVERAGE + NFCIRISK + NFCICREDIT（三子指數約略加總成 headline）→
  Panel 6 Row 1(Leverage) + Row 2(Risk/Credit) 跨兩列即可重建 headline 分解；ANFCI = cycle-adjusted 對照版。

⚠️ 全部抓 latest-revised（給 dashboard 顯示用）。NFCI 家族在 config 為 VINTAGE? 否 → 這些深史是
   dashboard-local，不進 FM。護欄（鎖死）：cache_dashboard 的深史不回流 FM / config / canonical。
   （pipeline 的 canonical NFCI 仍由 main.py 走正常 vintage/PIT 流程產到 transformed parquet；
     load_display_deep 只是把「canonical 近端 + 此處深史前段」combine 顯示。）

跑法（於 src/ 目錄）：
    python fetch_long_leverage.py
"""
from dotenv import load_dotenv
from fred_loader import FredLoader

load_dotenv()  # 灌 FRED key（.env 的 FRED_API_KEY）

# 獨立 dir：main.py 不 glob → 不污染 pipeline（與其他 fetch_long_* 同）
loader = FredLoader(cache_dir="./cache_dashboard")

SERIES = [
    "NFCILEVERAGE",   # NFCI Leverage Subindex (Row 1)
    "NFCI",           # NFCI Headline
    "NFCIRISK",       # NFCI Risk Subindex
    "NFCICREDIT",     # NFCI Credit Subindex
    "ANFCI",          # Adjusted NFCI
]
panel = loader.fetch_many(SERIES, start="1970-01-01", force_refresh=True)  # latest-revised, 全史

print(f"Panel 6 NFCI long: {panel.shape}  "
      f"{panel.index.min().date()} → {panel.index.max().date()}")
print("--- 各 series 實際覆蓋（NFCI 家族應全部 ~1971 起、週頻） ---")
for c in SERIES:
    if c in panel.columns and panel[c].notna().any():
        s = panel[c].dropna()
        print(f"  {c:14s} {s.index.min().date()} → {s.index.max().date()}  "
              f"(n={len(s)}, latest={s.iloc[-1]:+.2f})")
    else:
        print(f"  {c:14s} ⚠ 無資料 / 全 NaN（檢查 series_id 或 FRED 授權）")

print("\n存到 cache_dashboard/<SID>.parquet 各一檔（dashboard-local，latest-revised，絕不回流 FM）")
print("→ 跑完回 dashboard.py 看 Panel 6 (rebuild_dashboards.py)，NFCI 兩列應從 ~1997 起、含 dotcom/GFC/COVID")
print("  （margin 列 R6~R9 仍從 2010 起 = fc_margin 資料起點，左側空白為資料本身限制，非 bug）")
