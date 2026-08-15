"""
FINRA Margin Statistics loader (R13 / Cluster ⑤).

讀 FINRA 官方月度 Excel「Customer Margin Balances」→ margin debt + free credit 月頻 series。
FINRA 無公開即時 API → 手動定期下載 Excel 覆蓋(月頻、不爬蟲);loader 只負責「讀 + parse + 對齊」,
不負責「去哪抓它」。取代已退役的 BOGZ1FL663067003Q(Z.1 broker receivables,季頻、落後 ~5 月、
定義≠margin debt) → FINRA Debit Balances 才是 canonical margin debt。

來源(sheet "Customer Margin Balances",欄序 by position):
    A Year-Month  /  B Debit Balances(=margin debt)  /  C Free Credit Cash  /  D Free Credit Margin
單位:$ millions(FINRA 頁面明示 "shown in $ millions";最新 ~1.3T = 1,304,281 mn)。
reference = 每月最後營業日(Rule 4521(d) settlement-date basis)→ index 用 business month-end
            (= 真實 reference date,且保證是 business day → 對齊 pipeline 的 business-day 網格)。
無回溯修訂 → fixed lag(config lag=25,M+第三週 ~15-21d 保守 round-up;月頻下對 merge 無差;
            實測 pin 待後續),不需 vintage(對比 CPI/M2/PCE 走 ALFRED)。

產出三欄供下游:
    FINRA_MARGIN_DEBT  → config Indicator(mon=['yoy'] dashboard + FM 原料餵 Margin/M2,lag=25)
    FINRA_FC_CASH      → compute_margin_net_credit 原料(非 config Indicator,MON-only composite)
    FINRA_FC_MARGIN    → 同上
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# 產出欄名:margin debt = config series_id(下游 apply_transforms 靠它對欄);FC 兩欄非 Indicator,給 composite。
MARGIN_DEBT_COL = "FINRA_MARGIN_DEBT"
FC_CASH_COL = "FINRA_FC_CASH"
FC_MARGIN_COL = "FINRA_FC_MARGIN"

_SHEET = "Customer Margin Balances"


class FinraLoader:
    """讀 FINRA margin statistics Excel → 月頻 wide DataFrame(business month-end index)。"""

    def __init__(self, data_path: str = "../finra_margin_statistics.xlsx"):
        # main.py 在 src/ 跑 → Excel 在 repo root → 預設 '../';路徑慣例同其他相對路徑,不另開 data/。
        self.data_path = Path(data_path)

    def load(self) -> pd.DataFrame:
        """
        回月頻 wide DataFrame,index = business month-end DatetimeIndex(ascending),欄 = 三個 FINRA 欄(全 $mn)。

        ⚠️ Excel 是降序排列(最新月在最上)→ 必 sort_index() 轉 ascending,
            否則下游 pct_change / yoy 會對顛倒的時序算 → 符號全反的 silent bug。
        """
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"FINRA Excel 不存在: {self.data_path.resolve()}。"
                f" 請從 finra.org 下載 margin statistics Excel 放此路徑(月頻手動覆蓋)。"
            )

        raw = pd.read_excel(self.data_path, sheet_name=_SHEET, header=0)
        # 用「欄位位置」取前 4 欄(header 文字長,by-position 比 by-name 穩):
        #   0=Year-Month / 1=Debit Balances / 2=Free Credit Cash / 3=Free Credit Margin
        raw = raw.iloc[:, :4].copy()
        raw.columns = ["year_month", MARGIN_DEBT_COL, FC_CASH_COL, FC_MARGIN_COL]

        # Year-Month("2026-04" 字串 或 datetime) → business month-end(每月最後營業日 = FINRA reference)
        ym = pd.to_datetime(raw["year_month"], errors="coerce")
        idx = ym + pd.offsets.BMonthEnd(0)  # BMonthEnd(0):非月末營業日則前滾到當月最後營業日

        out = raw[[MARGIN_DEBT_COL, FC_CASH_COL, FC_MARGIN_COL]].apply(
            pd.to_numeric, errors="coerce"
        )
        out.index = idx
        out.index.name = "date"

        out = out[out.index.notna()]
        out = out.sort_index()                          # ★ 降序 → 升序(命門)
        out = out[~out.index.duplicated(keep="last")]   # 防重複月(以最後一筆為準)

        logger.info(
            f"FINRA loaded: {out.shape[0]} months "
            f"[{out.index.min():%Y-%m} → {out.index.max():%Y-%m}], "
            f"latest margin debt = {out[MARGIN_DEBT_COL].iloc[-1]:,.0f} $mn"
        )
        return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    df = FinraLoader().load()
    print(df.tail())
    print(f"\nshape: {df.shape}, asc check: {df.index.is_monotonic_increasing}")
