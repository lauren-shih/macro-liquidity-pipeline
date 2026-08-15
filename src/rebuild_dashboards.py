"""
rebuild_dashboards.py — 只重畫 dashboard,絕不碰資料（dashboard 樣式迭代專用）。

做什麼：
    只「讀」 output/macro_panel_transformed.parquet（apply_transforms 一律輸出 raw 欄,
    故這份已含 SOFR/EFFR/WALCL/M2SL... 所有 dashboard 需要的原始欄）→ 只「寫」 *.html。

不做什麼（= 零風險）：
    ✗ 不連 FRED（不 fetch、無 revision drift）
    ✗ 不覆寫任何 parquet（transformed / pit / features 全不動 → factor-model canonical 不受影響）
    ✗ 不碰 cache（OAS 深史 cache 安全）

什麼時候用哪個：
    • 改 dashboard.py 樣式 / hover / 版面後想看效果 → 跑這支（快、安全、無網路）
    • 真的要更新「資料」到最新 → 才跑 main.py（會 fetch + 覆寫 parquet）
    • 永遠不要對 credit 跑 --refresh（雖已雙層保護,沒理由冒險）

用法：
    cd 4.Macro_Pipeline/src
    python rebuild_dashboards.py
"""
from pathlib import Path
import pandas as pd

# Panel plotting imports (all panels active).
from dashboard import plot_repo_plumbing, plot_net_liquidity, plot_credit_stress_hy, plot_credit_stress_other
from dashboard import plot_macro_scorecard, trading_days_only
from dashboard import plot_yield_curve, plot_fx_vol_inflation, plot_leverage_monitor, plot_basis_trade

SRC_DIR = Path(__file__).resolve().parent      # = .../4.Macro_Pipeline/src
OUT_DIR = SRC_DIR / "output"
SRC_PARQUET = OUT_DIR / "macro_panel_transformed.parquet"

if not SRC_PARQUET.exists():
    raise FileNotFoundError(
        f"找不到 {SRC_PARQUET}。這支只重畫、不產資料 → 需要先有 transformed parquet。\n"
        f"（若從沒跑過 pipeline,先跑一次 main.py 產出 parquet,之後樣式迭代都用這支。）"
    )

panel = pd.read_parquet(SRC_PARQUET)           # ← 唯一的讀；之後完全不寫 parquet
print(f"[read-only] {SRC_PARQUET.name}: "
      f"{panel.shape[0]} days × {panel.shape[1]} cols  "
      f"({panel.index.min():%Y-%m-%d} → {panel.index.max():%Y-%m-%d})")

# 只留真實交易日:丟掉週末 / 假日 / 今日尚無市場資料的 phantom 列
# (DFEDTARU/L 在 FRED 是 7 天序列,把週末/假日空列帶進 parquet → 市場序列在那些列 NaN)。
# 修掉 ① x-unified hover 在週末抓最近真實點的 look-ahead 假象;② scorecard card 8 as-of 抓到 06-29。
# 純顯示層過濾,parquet 上面已讀完、之後完全不寫 → 不碰任何資料、不影響 factor-model canonical。
_before = len(panel)
panel = trading_days_only(panel)
print(f"[trading-days-only] 丟 {_before - len(panel)} 個非交易日列 → "
      f"{len(panel)} days  (尾 {panel.index.max():%Y-%m-%d})")

# ── 只寫 HTML ─────────────────────────────────────────────────────────────
plot_macro_scorecard(panel,  OUT_DIR / "00_macro_scorecard.html")   # 主頁 scorecard
plot_repo_plumbing(panel, OUT_DIR / "01_repo_plumbing.html")
plot_net_liquidity(panel, OUT_DIR / "02_net_liquidity.html")
plot_credit_stress_hy(panel,    OUT_DIR / "03-1_credit_stress_hy_oas.html")
plot_credit_stress_other(panel, OUT_DIR / "03-2_credit_stress_other_oas.html")
plot_yield_curve(panel,         OUT_DIR / "04_yield_curve.html")    # Panel 4 (深史需先跑 fetch_long_curve.py)
plot_fx_vol_inflation(panel,   OUT_DIR / "05_fx_vol_inflation.html")  # Panel 5 (深史需先跑 fetch_long_fxvolmacro.py)
plot_leverage_monitor(panel,   OUT_DIR / "06_leverage_monitor.html")  # Panel 6 (NFCI 深史需先跑 fetch_long_leverage.py; margin 讀 repo root finra xlsx)
plot_basis_trade(panel,        OUT_DIR / "07_leveraged_funds_basis_trade.html")   # Panel 7 (TFF 週頻; DGS/SOFR/EFFR/M2 皆 transformed parquet 內 → 無額外深史前置)

print("[done] dashboards rebuilt — HTML only, 沒有 fetch、沒有覆寫任何 parquet、沒動 cache。")
print(f"       → {OUT_DIR / '00_macro_scorecard.html'}")
print(f"       → {OUT_DIR / '01_repo_plumbing.html'}")
print(f"       → {OUT_DIR / '02_net_liquidity.html'}")
print(f"       → {OUT_DIR / '03-1_credit_stress_hy_oas.html'}")
print(f"       → {OUT_DIR / '03-2_credit_stress_other_oas.html'}")
print(f"       → {OUT_DIR / '04_yield_curve.html'}")
print(f"       → {OUT_DIR / '05_fx_vol_inflation.html'}")
print(f"       → {OUT_DIR / '06_leverage_monitor.html'}")
print(f"       → {OUT_DIR / '07_leveraged_funds_basis_trade.html'}")
