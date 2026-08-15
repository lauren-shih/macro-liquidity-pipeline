"""
cftc_loader.py
==============
CFTC Traders in Financial Futures (TFF) Report loader.

What this captures
------------------
Hedge funds 在美債期貨的 long / short 部位。當前(2024-2025)的核心議題:
  Leveraged Funds 在 10Y UST futures 的 net short 創歷史新高,
  反映 cash-futures basis trade 規模 ~$1T。
  這是 Yellen / FSOC / Fed 都在警告的 systemic risk.

Data source
-----------
CFTC public Socrata API (no key required):
  https://publicreporting.cftc.gov/resource/gpe5-46if.json
  (Financial Futures Combined - Long Format)

Why this matters for liquidity monitoring
-----------------------------------------
這份 report 是 NY Fed Liberty Street Economics 多次引用的核心資料,
能跑這份報告自動化 = 你已經在做 institutional-grade macro 監控。

Contract market codes
---------------------
 042601  UST 10Y Note (CBT)
 042602  UST 2Y Note (CBT)
 020601  UST 5Y Note (CBT)
 020604  UST Bond (CBT, ~30Y)
 134741  Ultra UST 10Y (CBT)
"""

from __future__ import annotations
import logging
from pathlib import Path
import pandas as pd
import requests

logger = logging.getLogger(__name__)


# Map of futures we care about.
# ⚠️ Code→合約對應已用 Socrata `contract_market_name` 逐一驗證（不可憑文件推斷 —
#    例如 134741 實為 SOFR-3M 利率期貨、042602 為無效空 code）。
# FM registered（config）: TFF_10Y_LEVERAGED ← 043602、TFF_2Y_LEVERAGED ← 042601。
# 其餘三支（5Y / T-Bond / Ultra-Bond）= monitoring-only, 顯示於 leverage monitor 面板。
# 註：「Ultra 10Y Note」另有獨立合約代碼，刻意未納入；不影響兩支註冊指標。
#     若未來要納入 monitoring，先從 Socrata contract_market_name 做 ground-truth 驗證（勿憑記憶 / 文件補）。
TFF_CONTRACTS: dict[str, str] = {
    "042601": "UST_2Y",          # Socrata: UST 2Y NOTE      → FM: TFF_2Y_LEVERAGED
    "044601": "UST_5Y",          # Socrata: UST 5Y NOTE      → monitoring
    "043602": "UST_10Y",         # Socrata: UST 10Y NOTE     → FM: TFF_10Y_LEVERAGED
    "020601": "UST_Bond",        # Socrata: UST BOND         → monitoring (經典 T-Bond)
    "020604": "UST_Ultra_Bond",  # Socrata: ULTRA UST BOND   → monitoring
}


class CFTCLoader:
    """
    CFTC Traders in Financial Futures (TFF) report loader.
    No API key needed - CFTC Socrata is public.
    """

    BASE_URL = "https://publicreporting.cftc.gov/resource/gpe5-46if.json"

    def __init__(
        self,
        cache_dir: str | Path = "./cache",
        timeout: int = 30,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout

    def fetch_contract(
        self,
        contract_code: str,  # 傳入哪個合約代碼(例如："042601")
        start_date: str = "2006-01-01",  # 抓哪天之後的資料
    ) -> pd.DataFrame:
        """
        Fetch TFF data for a single contract market code.

        Returns columns:
            date, lev_long, lev_short, lev_net, dealer_long, dealer_short,
            asset_mgr_long, asset_mgr_short, other_long, other_short
        """
        cache_path = self.cache_dir / f"cftc_tff_{contract_code}.parquet"

        # ===========================================================================================================================
        # Socrata SoQL 查詢（SoQL = Socrata 版 SQL, 參數以 $ 開頭）: 按合約 code + 起日過濾、按日期升冪、上限 5 萬筆。
        # CFTC 無官方 Python SDK（對比 fredapi）→ 本 loader 自建 REST client。
        # ===========================================================================================================================
        params = {
            "$where": (
                f"cftc_contract_market_code='{contract_code}' " f"AND report_date_as_yyyy_mm_dd >= '{start_date}T00:00:00'"
            ),
            "$order": "report_date_as_yyyy_mm_dd ASC",
            "$limit": 50000,
        }

        try:
            logger.info(f'  [CFTC] Fetching contract {contract_code} ({TFF_CONTRACTS.get(contract_code, "unknown")})')
            resp = requests.get(self.BASE_URL, params=params, timeout=self.timeout)
            resp.raise_for_status()
            raw = resp.json()
        except Exception as e:  # except區塊 → 失敗就fallback用cache(production容錯思維，骨架跟fred同套路)
            logger.error(f"  [CFTC FAIL] {contract_code}: {e}")
            if cache_path.exists():
                logger.warning(f"  [fallback] Using cached: {contract_code}")
                return pd.read_parquet(cache_path)
            raise

        if not raw:
            logger.warning(f"  [CFTC EMPTY] {contract_code}")
            return pd.DataFrame()

        df = pd.DataFrame(raw)

        # Standardise columns
        df["date"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"])

        # ===========================================================================================================================
        # Numeric conversion — cols_map 已對照 Socrata 實際 schema 逐欄驗證（ground truth, 讀文件猜不到）:
        #   CFTC 欄位命名不統一 — lev_money / asset_mgr / other_rept 三組「無」_all 後綴, dealer 組「有」_all 後綴。
        #   ⚠️ 勿「好心」把 dealer 的 _all 拿掉 — 那是對照 API 實際 schema 後的正確命名。
        # ===========================================================================================================================
        cols_map = {
            "lev_money_positions_long": "lev_long",  # 無 _all
            "lev_money_positions_short": "lev_short",  # 無 _all
            "dealer_positions_long_all": "dealer_long",  # 有 _all (dealer組特例)
            "dealer_positions_short_all": "dealer_short",  # 有 _all (dealer組特例)
            "asset_mgr_positions_long": "asset_mgr_long",  # 無 _all
            "asset_mgr_positions_short": "asset_mgr_short",  # 無 _all
            "other_rept_positions_long": "other_long",  # 無 _all
            "other_rept_positions_short": "other_short",  # 無 _all
        }
        for src, dst in cols_map.items():
            if src in df.columns:
                df[dst] = pd.to_numeric(df[src], errors="coerce")  # 強制轉換成數字，轉不動的「降級」成"NaN"，絕不報錯中斷crash

        # Compute net positions
        df["lev_net"] = df["lev_long"] - df["lev_short"]
        df["dealer_net"] = df["dealer_long"] - df["dealer_short"]
        df["asset_mgr_net"] = df["asset_mgr_long"] - df["asset_mgr_short"]

        keep = [
            "date",
            "lev_long",
            "lev_short",
            "lev_net",
            "dealer_long",
            "dealer_short",
            "dealer_net",
            "asset_mgr_long",
            "asset_mgr_short",
            "asset_mgr_net",
        ]
        df = df[[c for c in keep if c in df.columns]].sort_values("date").reset_index(drop=True)
        df.set_index("date", inplace=True)

        # Persist
        df.to_parquet(cache_path)
        logger.info(f"  [CFTC OK ] {contract_code}: {len(df)} weekly observations")
        return df

    def fetch_all(self, start_date: str = "2006-01-01") -> dict[str, pd.DataFrame]:
        """Fetch all configured contracts. Returns dict[label, df]."""
        out = {}
        for code, label in TFF_CONTRACTS.items():
            try:
                out[label] = self.fetch_contract(code, start_date)
            except Exception as e:
                logger.error(f"Failed to fetch {label}: {e}")
        return out

    @staticmethod
    def build_summary_panel(contracts_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Create a wide panel of net positions only (date-indexed, weekly).

        Leveraged-funds net → canonical ``TFF_<tenor>_LEVERAGED``，對齊 config series_id /
        pit_safe lag_map / DB ingest ``startswith('TFF_')``（R18 wire；命名一致化 C6）。
        registered 2（10Y/2Y）剛好對上 config；其餘 3 contract（5Y/Bond_30Y/Ultra_10Y）同規則命名
        但未註冊 → 不在 lag_map → 不進 fm_cols → 無害未用（留供未來註冊）。
        Asset-manager net 維持 ``UST_<tenor>_asset_mgr_net``（monitoring-only，留 Cluster ⑥ dashboard）。

        Columns: TFF_10Y_LEVERAGED, TFF_2Y_LEVERAGED, ..., UST_10Y_asset_mgr_net, ...
        """
        out = pd.DataFrame()
        for label, df in contracts_data.items():  # label = "UST_10Y" / "UST_2Y" / "UST_5Y" / "UST_Bond_30Y" / "UST_Ultra_10Y"
            if df.empty:
                continue
            tenor = label.removeprefix("UST_")  # "10Y" / "2Y" / "5Y" / "Bond_30Y" / "Ultra_10Y"
            out[f"TFF_{tenor}_LEVERAGED"] = df["lev_net"]       # registered 2 對上 config series_id
            out[f"{label}_asset_mgr_net"] = df.get("asset_mgr_net")  # monitoring-only,維持 UST_ 命名
        return out.sort_index()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    loader = CFTCLoader()
    df_10y = loader.fetch_contract("042601", start_date="2024-01-01")
    print(f"\n10Y UST Futures TFF data: {len(df_10y)} weeks")
    if not df_10y.empty:
        print(df_10y[["lev_net", "asset_mgr_net"]].tail())
