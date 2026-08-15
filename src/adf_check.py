#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
adf_check.py — ADF 定態性檢定(z-score 窗口判定 + FM Δ-form 統計後盾)
================================================================================
A 段:z 窗口判定(定版 8 個 z 目標)
    z on level(6): HY OAS / IG OAS / EM HY OAS / credit_quality_spread / VIX / repo_spread
    z on Δ(2):    UST_10Y / UST_2Y leveraged-funds net positioning(native weekly Δ)
    判讀:ADF p<0.05 → stationary → expanding;p>=0.05 → non-stationary → rolling(252d)

B 段:FM active 因子的「變化量應為 stationary」統計後盾(月頻,對齊 FM panel)
    ΔHY OAS / Δrepo_spread / ΔVIX / ΔT10Y2Y / ΔNet Liquidity / DEXTAUS pct1m

跑法（repo root）：python src/adf_check.py
輸出:terminal SUMMARY + adf_summary.csv(repo root)
紀律:只讀 cache、不打 API、不改任何檔
"""
from pathlib import Path
import sys
import pandas as pd

try:
    import pyarrow  # noqa: F401  # 讀 cache parquet 需要；缺了會在 read_parquet 才炸、訊息不直觀
except ImportError:
    sys.exit("此直譯器沒有 pyarrow — 請改用 repo-local 環境（conda env / venv）內的 python 執行")

try:
    from statsmodels.tsa.stattools import adfuller
except ImportError:
    sys.exit("statsmodels 不在此環境 — pip install statsmodels 後重試")

ROOT = Path(__file__).resolve().parents[1]
_SRC = ROOT / "src"

# ── cache 路徑 ────────────────────────────────────────────────────────────────
# canonical cache = src/cache，與 main.py 的 CACHE_DIR 同一個（含 OAS 深史 splice）。
# ⚠️ 讀取路徑以 src/cache 為準；若磁碟上另有同名 cache 目錄，本 script 可能靜默讀到
#    過期快照（缺 rename 後的新 series、信用族群只剩淺歷史），而且不會報錯 —— 因此下方一律印出實際使用的路徑與新鮮度，讓基底問題一眼可見。
CACHE = _SRC / "cache"
if not CACHE.exists():
    sys.exit("canonical cache 不存在: %s — 請確認 layout（main.py 的 CACHE_DIR = SRC_DIR/'cache'）" % CACHE)

_files = sorted(CACHE.glob("*.parquet"))
if not _files:
    sys.exit("cache 目錄沒有任何 parquet: %s" % CACHE)
_newest = max(f.stat().st_mtime for f in _files)
print("cache dir  :", CACHE)
print("cache files:", len(_files), "| 最新檔案時間:",
      pd.Timestamp.fromtimestamp(_newest).strftime("%Y-%m-%d %H:%M"))

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from transformations import (
    compute_net_liquidity,
    compute_repo_spread,
    compute_credit_quality_spread,
)

# ---------------------------------------------------------------- loaders
def load_fred(series_id):
    p = CACHE / (series_id + ".parquet")
    if not p.exists():
        return None
    s = pd.read_parquet(p)["value"]
    s = pd.to_numeric(s, errors="coerce").dropna()
    s.index = pd.to_datetime(s.index)
    return s.sort_index()

def load_cftc(key):
    """key: 'UST_10Y' / 'UST_2Y' → 讀 output/cftc_tff_panel.parquet 的 TFF_<tenor>_LEVERAGED（native weekly）。
    回傳 (Series|None, err_msg|None)。

    讀 main.py Step 2 產的 wide panel：欄名 = canonical TFF_<tenor>_LEVERAGED
    （build_summary_panel 命名），名實相符、單一來源；只「讀」、不打 API、不改檔。
    """
    tenor = key.replace("UST_", "")                       # 'UST_10Y' → '10Y' / 'UST_2Y' → '2Y'
    col = "TFF_%s_LEVERAGED" % tenor
    fp = _SRC / "output" / "cftc_tff_panel.parquet"
    if not fp.exists():
        return None, "output/cftc_tff_panel.parquet 不存在（main.py 未跑過 CFTC）— SKIP"
    df = pd.read_parquet(fp)
    if col not in df.columns:
        return None, "%s 不在 cftc_tff_panel（cols=%s）" % (col, list(df.columns)[:8])
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    s.index = pd.to_datetime(s.index)
    return s.sort_index(), None

# ---------------------------------------------------------------- ADF core
def adf_row(s, label, section, transform, note="", min_n=60):
    out = dict(section=section, series=label, transform=transform, n=0,
               start="", end="", adf_stat=None, p_value=None,
               verdict="", recommendation="", note=note)
    if s is None:
        out["verdict"] = "SKIP: cache missing"
        return out
    s = s.dropna()
    out["n"] = len(s)
    if len(s) < min_n:
        out["verdict"] = "SKIP: n<%d" % min_n
        return out
    out["start"], out["end"] = str(s.index.min().date()), str(s.index.max().date())
    stat, p = adfuller(s, autolag="AIC", regression="c")[:2]
    out["adf_stat"], out["p_value"] = round(stat, 3), round(p, 4)
    stationary = p < 0.05
    out["verdict"] = "stationary" if stationary else "non-stationary"
    if section == "A":
        out["recommendation"] = "expanding" if stationary else "rolling(252d)"
    else:
        out["recommendation"] = "OK as FM factor" if stationary else "!! re-examine form"
    return out

rows, warnings = [], []

# ---------------------------------------------------------------- load all
ids = ["BAMLH0A0HYM2", "BAMLC0A0CM", "BAMLEMHYHYLCRPIUSOAS", "BAMLH0A1HYBB",
       "BAMLH0A3HYC", "VIXCLS", "SOFR", "EFFR", "T10Y2Y",
       "WALCL", "WDTGAL", "RRPONTSYD", "DEXTAUS"]
data = {i: load_fred(i) for i in ids}
miss = [i for i, v in data.items() if v is None]
if miss:
    warnings.append("cache 缺 %s — 對應列 SKIP" % miss)

# deep-history gate(OAS 歷史太淺 → ADF 樣本不足以代表長期性質)
for sid in ["BAMLH0A0HYM2", "BAMLC0A0CM", "BAMLH0A1HYBB", "BAMLH0A3HYC", "BAMLEMHYHYLCRPIUSOAS"]:
    s = data.get(sid)
    if s is not None and s.index.min().year >= 2019:
        warnings.append("%s cache 起點 %s 偏淺 —— 該 series 的深歷史在資料源政策變更後已無法自 API 重新取得；"
                        "下方 ADF 判定僅反映此區間，請據此解讀" % (sid, s.index.min().date()))

# ---------------------------------------------------------------- Section A
rows.append(adf_row(data["BAMLH0A0HYM2"], "HY OAS", "A", "level"))
rows.append(adf_row(data["BAMLC0A0CM"], "IG OAS", "A", "level"))
rows.append(adf_row(data["BAMLEMHYHYLCRPIUSOAS"], "EM HY OAS", "A", "level"))

if data["BAMLH0A3HYC"] is not None and data["BAMLH0A1HYBB"] is not None:
    qual = compute_credit_quality_spread(data["BAMLH0A3HYC"], data["BAMLH0A1HYBB"]).dropna()
    rows.append(adf_row(qual, "credit_quality_spread (CCC-BB)", "A", "level"))
else:
    rows.append(adf_row(None, "credit_quality_spread (CCC-BB)", "A", "level"))

rows.append(adf_row(data["VIXCLS"], "VIX", "A", "level"))

if data["SOFR"] is not None and data["EFFR"] is not None:
    repo = compute_repo_spread(data["SOFR"], data["EFFR"])
    rows.append(adf_row(repo, "repo_spread (SOFR-EFFR)", "A", "level(bps)"))
else:
    repo = None
    rows.append(adf_row(None, "repo_spread (SOFR-EFFR)", "A", "level(bps)"))

for key, label in [("UST_10Y", "TFF_10Y lev net"), ("UST_2Y", "TFF_2Y lev net")]:
    s, err = load_cftc(key)
    if err:
        warnings.append(err)
        rows.append(adf_row(None, label, "A", "weekly Δ"))
    else:
        rows.append(adf_row(s.diff(), label, "A", "weekly Δ",
                            note="變異數 regime caveat:若 expanding 稀釋近期波動 → rolling 覆寫 + documented reason"))

# ---------------------------------------------------------------- Section B(月頻,對齊 FM panel)
def month_end(s):
    return None if s is None else s.resample("ME").last()

b_items = []
b_items.append(("dHY OAS",  month_end(data["BAMLH0A0HYM2"]), "monthly Δ", ""))
b_items.append(("drepo_spread", month_end(repo) if repo is not None else None, "monthly Δ(bps)", ""))
b_items.append(("dVIX", month_end(data["VIXCLS"]), "monthly Δ", ""))
b_items.append(("dT10Y2Y", month_end(data["T10Y2Y"]), "monthly Δ", ""))
if all(data[k] is not None for k in ["WALCL", "WDTGAL", "RRPONTSYD"]):
    netliq = compute_net_liquidity(data["WALCL"], data["WDTGAL"], data["RRPONTSYD"]).dropna()
    b_items.append(("dNet Liquidity", month_end(netliq), "monthly Δ($bn)", "WDTGAL = H.4.1 週三水位"))
else:
    b_items.append(("dNet Liquidity", None, "monthly Δ($bn)", ""))
b_items.append(("DEXTAUS pct1m", month_end(data["DEXTAUS"]), "monthly pct1m", ""))

for label, s_me, tf, note in b_items:
    if s_me is None:
        rows.append(adf_row(None, label, "B", tf, note))
    elif "pct1m" in tf:
        rows.append(adf_row(s_me.pct_change().dropna(), label, "B", tf, note, min_n=48))
    else:
        rows.append(adf_row(s_me.diff().dropna(), label, "B", tf, note, min_n=48))

# ---------------------------------------------------------------- output
res = pd.DataFrame(rows)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 60)
print("\n" + "=" * 96)
print("SECTION A — z-score 窗口判定(stationary→expanding / non-stationary→rolling 252d)")
print("=" * 96)
print(res[res["section"] == "A"].drop(columns=["section"]).to_string(index=False))
print("\n" + "=" * 96)
print("SECTION B — FM active 因子變化量定態性(月頻;應全 stationary)")
print("=" * 96)
print(res[res["section"] == "B"].drop(columns=["section"]).to_string(index=False))
if warnings:
    print("\n--- WARNINGS ---")
    for w in warnings:
        print(" *", w)
out_dir = ROOT / "results"
out_dir.mkdir(exist_ok=True)
out_csv = out_dir / "adf_summary.csv"
res.to_csv(out_csv, index=False, encoding="utf-8-sig")
print("\nsaved:", out_csv)
