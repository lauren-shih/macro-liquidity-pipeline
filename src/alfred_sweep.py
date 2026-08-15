"""
alfred_sweep.py — ALFRED publication-lag sweep（config 內 FRED series 全量掃描）
==================================================================================
兩個常見量測瑕疵已內建修正：
  (1) ancient-period 污染：若用「realtime_start > bound」排除邊界夾擠的 heuristic，
      對「有深歷史 + 被修訂」的 series 不可靠 → H41/DTWEXEMEGS/RRPONTSYAWARD
      跑出上千天垃圾 lag。改成「只量近 N 年的 period」(DATE_FLOOR)，不管 ALFRED 怎麼
      夾擠都 bulletproof —— 近期 period 的 first-release 必然落在 realtime 視窗內、非夾擠。
  (2) ValueError：對某些 series 抓全 vintage matrix 會炸（FRED API 100k row 上限／
      或該 series 在 ALFRED 根本沒 archival vintage）→ 縮短 realtime 視窗減少 response
      size + 把【真正的 error message】抓出來（str(e)），分辨「太大」vs「沒 vintage」。

  ⚠️ 三個 monthly vintage 目標（CPI/M2/PCE）不受此瑕疵影響【結果穩健】
     （CPI=41 / M2=52 / PCE=58）；本修正只讓其餘 34 個也乾淨 + 顯示 FAIL 真正原因，
     不改已確認的 R15-PCE 結論。重跑後 monthly 應重現 ~41/52/58。

  ⚠️ 量到的 lag 是「FRED period date（月頻=月初 1 號）→ first-release 公布日」的天數。
     已驗證 fred_loader / main 皆【不】resample 到月底 → panel 是 month-START index，
     故 gap_vs_config 是真實 leak（非慣例假象）。

目的 / 判讀 narrative
------------------------------------------------------------------
對 config 的 37 個 FRED series 實測真實 publication lag，與 config.Indicator.lag
並排。最關鍵 = PCEPILFE（R15-PCE，config 無此 lag → fallback 1）。
  - daily/weekly 預期 gap≈0、revised≈0%      → 固定 lag 已校準正確，不需 vintage
  - monthly（CPI/M2/PCE）gap 大 + revised 高  → 要 vintage 才 PIT-correct
  - 計算型/index 型（DGS*/T10Y*/VIXCLS/SP500…）多半 ALFRED 無 vintage → FAIL/empty（符合預期）

執行（從 repo root，.env 要有 FRED_API_KEY）：
  python src/alfred_sweep.py
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

import pandas as pd

# ── 定位 src/（含 config.py），不管 script 放哪都能 import ──
_here = Path(__file__).resolve()
for _cand in (_here.parent / "src", _here.parent.parent / "src", _here.parent):
    if (_cand / "config.py").exists():
        sys.path.insert(0, str(_cand))
        break

from config import by_source, ALL_INDICATORS       # SSoT：FRED series 清單 ＋ 每支的 publication lag
from fred_loader_vintage import FredVintageLoader  # 現在哪些走 vintage(config SSoT,現為 8 支)

# pipeline 現在假設的 lag —— SSoT = config.Indicator.lag。
# （早期版本從 pit_safe import DEFAULT_LAG_MAP；該常數已於重構移除：pit_safe 現在不再持有
#   預設 lag，改由呼叫端組好 lag_map 傳入、缺欄即 raise，lag 的唯一來源統一收斂到 config。
#   此處與 main.py 用同一份來源組表，故兩者永遠一致。）
DEFAULT_LAG_MAP = {ind.series_id: ind.lag for ind in ALL_INDICATORS if ind.lag is not None}

from dotenv import load_dotenv, find_dotenv
from fredapi import Fred

# realtime 視窗縮到 4 年（而非 8 年）→ response 小很多，降低 100k-row ValueError。
REALTIME_START = "2022-01-01"
# 只量「date >= DATE_FLOOR」的近期 period → bulletproof 排掉 ancient-period 污染
# （近期 period 的 first-release 必落在 realtime 視窗內，非邊界夾擠，lag 一定乾淨）。
DATE_FLOOR = "2022-06-01"
VINTAGE_NOW = FredVintageLoader.VINTAGE_SERIES

load_dotenv(find_dotenv())
api_key = os.getenv("FRED_API_KEY")
if not api_key:
    sys.exit("FRED_API_KEY 找不到 — 確認 repo 內 .env 有設定 FRED_API_KEY=...")
fred = Fred(api_key=api_key)

fred_inds = by_source("FRED")
print(f"Sweeping {len(fred_inds)} FRED series "
      f"(realtime_start={REALTIME_START}, 只量 date >= {DATE_FLOOR})\n")

rows = []
for ind in fred_inds:
    sid = ind.series_id
    try:
        df = fred.get_series_all_releases(sid, realtime_start=REALTIME_START)

        if df is None or len(df) == 0:
            rows.append({"series_id": sid, "freq": ind.frequency, "status": "NO-VINTAGE(empty)", "n": 0})
            print(f"  {sid:<22} NO-VINTAGE（ALFRED 回空）")
            continue

        df = df.copy()
        df["realtime_start"] = pd.to_datetime(df["realtime_start"])
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

        # first release per period = stable sort + drop_duplicates（Croushore-safe，
        # 對齊 pit_safe_vintage 的 first-release 取法；刻意不用 groupby.agg('first')）
        df = df.sort_values(["date", "realtime_start"], kind="stable")
        first = df.drop_duplicates("date", keep="first").copy()
        last = (df.drop_duplicates("date", keep="last")[["date", "value"]]
                  .rename(columns={"value": "latest_value"}))

        # ★ 只留近期 period（date floor）→ 取代不可靠的「realtime_start > bound」夾擠排除
        clean = first[first["date"] >= pd.Timestamp(DATE_FLOOR)].copy()
        clean["lag_days"] = (clean["realtime_start"] - clean["date"]).dt.days

        n = len(clean)
        if n == 0:
            rows.append({"series_id": sid, "freq": ind.frequency, "status": "NO-recent-period", "n": 0})
            print(f"  {sid:<22} 近期視窗無 period")
            continue

        median_lag = int(clean["lag_days"].median())
        p90_lag = int(clean["lag_days"].quantile(0.90))

        # revision magnitude：多少比例 period 的 first-release != latest（只比兩邊都非 NaN）
        merged = clean[["date", "value", "lag_days"]].merge(last, on="date", how="left")
        m = merged["value"].notna() & merged["latest_value"].notna()
        revised = ((merged.loc[m, "value"].round(4) != merged.loc[m, "latest_value"].round(4)).mean()
                   if m.any() else float("nan"))

        cfg_lag = DEFAULT_LAG_MAP.get(sid)
        cfg_disp = str(cfg_lag) if cfg_lag is not None else "miss->1"
        cfg_eff = cfg_lag if cfg_lag is not None else 1
        gap = median_lag - cfg_eff

        rows.append({
            "series_id": sid, "freq": ind.frequency, "status": "OK", "n": n,
            "median_lag": median_lag, "p90_lag": p90_lag,
            "config_lag": cfg_disp, "gap_vs_config": gap,
            "revised_pct": round(revised * 100, 1) if revised == revised else None,
            "vintage_now": "Y" if sid in VINTAGE_NOW else "",
        })
        star = "  *" if abs(gap) >= 20 else ""
        rev_s = f"{revised*100:>5.1f}%" if revised == revised else "  n/a"
        print(f"  {sid:<22} median_lag={median_lag:>4}d  p90={p90_lag:>4}d  "
              f"config={cfg_disp:<8}  gap={gap:>+5}d  revised={rev_s}{star}")

    except Exception as e:
        # ★ 把真正的 error message 抓出來（只記 type 會看不出是「太大」還是「沒 vintage」）
        msg = str(e).replace("\n", " ").strip()[:60]
        rows.append({"series_id": sid, "freq": ind.frequency, "status": f"FAIL: {msg}", "n": 0})
        print(f"  {sid:<22} FAIL: {msg}")

out = pd.DataFrame(rows)
if "gap_vs_config" in out.columns:
    out["_g"] = out["gap_vs_config"].abs()
    out = out.sort_values("_g", ascending=False, na_position="last").drop(columns="_g")

_out_dir = _here.parent.parent / "results"          # repo root/results（與出貨位置一致）
_out_dir.mkdir(exist_ok=True)
csv_path = (_out_dir / "sweep_alfred.csv").resolve()  # 覆蓋舊的
out.to_csv(csv_path, index=False)

print("\n" + "=" * 92)
print("SUMMARY（依 |gap vs config| 由大到小）")
print("=" * 92)
with pd.option_context("display.max_rows", None, "display.width", 240):
    print(out.to_string(index=False))
print(f"\nsaved -> {csv_path}")
print("\n* monthly（PCEPILFE/CPIAUCSL/M2SL）應重現大 gap + 高 revised%（R15-PCE）")
print("  daily/weekly 應 gap≈0；FAIL 看 status 訊息分辨「太大」vs「沒 vintage」")
print("  污染列（H41/DTWEXEMEGS/RRPONTSYAWARD）這次應變乾淨")
