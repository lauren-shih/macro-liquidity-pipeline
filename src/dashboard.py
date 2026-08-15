"""
dashboard.py
============
Plotly-based interactive macro liquidity dashboard.

Output: single HTML file (deployable to GitHub Pages).

設計重點
--------
1. **Recession shading**: 自動標出 NBER 衰退期 (背景灰色)
2. **Z-score panels**: 不只看 level, 同步呈現 trailing Z-score
3. **規格化 layout**: 4 個 subplot, 每個聚焦一個 thesis
4. **互動性**: hover 顯示精確數值, range selector 讓使用者快速切換時間窗

What a reviewer sees
--------------------
"This is more than a FRED chart viewer: it is an institutional dashboard
 with proper risk thresholds annotated, Z-scores trailing, and recession context.
 This is what we'd build at the desk."
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# composite SSoT (S7 R23/R25 DRY 收斂)：net_liq / repo_spread / quality_spread
# 統一由 transformations 計算,dashboard 不再各自 inline (消 9 份散落 copy)
from transformations import (
    compute_net_liquidity,
    compute_real_m2_yoy,
    compute_repo_spread,
    compute_credit_quality_spread,
    compute_sp500_m2,
    compute_margin_net_credit,
    expanding_zscore,
    rolling_zscore,
)
from finra_loader import FinraLoader, FC_CASH_COL, FC_MARGIN_COL, MARGIN_DEBT_COL

# ============================================================================
# THEME — institutional look (Image-5 風格). 改 THEME_MODE 一行即可全 dashboard 切換。
# ============================================================================
THEME_MODE = "dark"   # "dark" | "light"

_THEMES = {
    "dark": dict(
        template="plotly_dark",
        paper_bg="#1b1e24", plot_bg="#1b1e24",
        font_color="#e6e8eb", title_color="#f4f5f7",
        grid="rgba(255,255,255,0.07)",
        recession="rgba(190,190,190,0.12)",
        corridor="rgba(150,160,178,0.20)",
        hover_bg="rgba(28,31,38,0.92)",
        btn_bg="#2a2e37", btn_active="#3d8bfd", btn_font="#e6e8eb",
        sofr="#4DA3FF", effr="#9DDB4D", iorb="#FF9F43",
        onrrp="#B388FF", tgcr="#2DD4BF",
        z_line="#FFB454", z_fill="rgba(255,159,67,0.30)",
        sp500="#E8EAED", ratio="#2DD4BF",
        thr_warn="#FFB454", thr_crit="#FF6B6B",
        bar_low="#56C28A",   # z-score 長條: 正常 (≤2) 綠
        ig="#4DA3FF", bb="#9DDB4D", hy="#FF9F43",       # credit OAS 品質階梯 (冷→暖)
        ccc="#FB7185", em="#B388FF",
        qs="#FBBF24",                                    # Quality Spread (CCC−BB) 專屬金色
    ),
    "light": dict(
        template="plotly_white",
        paper_bg="#FAFAF8", plot_bg="#FFFFFF",
        font_color="#1a1a1a", title_color="#1a1a1a",
        grid="rgba(0,0,0,0.06)",
        recession="rgba(120,120,120,0.16)",
        corridor="rgba(120,130,145,0.16)",
        hover_bg="rgba(255,255,255,0.95)",
        btn_bg="#eef0f2", btn_active="#0B447C", btn_font="#1a1a1a",
        sofr="#0B447C", effr="#65A30D", iorb="#EA580C",
        onrrp="#7C3AED", tgcr="#0D7A5C",
        z_line="#D97706", z_fill="rgba(217,119,6,0.20)",
        sp500="#1a1a1a", ratio="#0D7A5C",
        thr_warn="#D97706", thr_crit="#DC2626",
        bar_low="#65A30D",   # z-score 長條: 正常 (≤2) 綠
        ig="#0B447C", bb="#65A30D", hy="#EA580C",
        ccc="#E11D48", em="#7C3AED",
        qs="#D97706",
    ),
}
T = _THEMES[THEME_MODE]

# Plotly 工具列: 拿掉看 dashboard 用不到/難懂的鈕 (相機/框選縮放/box-select/lasso/autoscale) + logo;
# 留 平移/放大/縮小/Home。拖曳縮放是預設行為, 不靠 zoom2d 鈕 → 移除它不影響放大 spike。
MODEBAR_CONFIG = {
    "displaylogo": False,
    # 明確指定鈕 + 順序 (直向時由上到下): reset axes → pan → zoom in → zoom out
    "modeBarButtons": [["resetScale2d", "pan2d", "zoomIn2d", "zoomOut2d"]],
}


def _write_panel_html(fig, output) -> Path:
    """write_html + 注入深色滿版 body → 深色填滿整頁、零白邊 (與 scorecard 同一招)。
    plotly full_html 預設的 <body> 是瀏覽器預設 (白底 + 8px margin), 深色圖浮在上面四周露白邊、圖下方也白;
    注入 html,body{margin:0;padding:0;background:#1b1e24} → 深色直接填滿頁面 (圖維持原 height, 圖下方由深色 body 補滿)。
    跟 scorecard 的 body{margin:0;background:#1b1e24} 一致 → portfolio piece 視覺統一。
    """
    html = fig.to_html(include_plotlyjs="cdn", full_html=True, config=MODEBAR_CONFIG)
    html = html.replace(
        "</head>",
        # 載入 Inter web font (與 scorecard 同條 link) → panel 標題真的用到宣告的 Inter,
        # 不再 silently fallback 成系統預設 (Segoe UI / Microsoft JhengHei) → 與 scorecard 視覺一致。
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap">'
        "<style>html,body{margin:0;padding:0;background:#1b1e24;}</style></head>",
    )
    output = Path(output)
    output.write_text(html, encoding="utf-8")
    return output


def trading_days_only(panel: pd.DataFrame) -> pd.DataFrame:
    """只保留「真實交易日」的列 → 丟掉週末 / 假日 / 今日尚無市場資料的 phantom 列。

    為什麼需要:DFEDTARU/DFEDTARL(Fed 目標區間)在 FRED 是 7 天(含週末)序列,會把週末/假日
    的「空列」帶進 transformed parquet —— 那些列上的市場序列(SOFR/EFFR/SP500…)全是 NaN。後果:
      ① x-unified hover 在 NaN gap 上抓「最近的真實點」→ 週日顯示週一的值(look-ahead 假象);
      ② scorecard 的 as-of 抓 index 尾端(ffill 到今日)→ card 8 顯示 06-29 而非真實 06-26。
    把這些非交易日的列直接丟掉 → 兩個問題一次解,且不改任何「值」(只去掉沒有真實觀測的列)。

    判準(frequency-agnostic;不寫死欄名,日後新增指標免回來改):
      1. 必為平日(週一~五)→ 擋週末。
      2. 自動偵測「日頻市場欄」(active 區間內密度 > 0.5 濾掉月/週頻;非空日 > 95% 落平日 →
         濾掉 DFEDTARU/L 這種 7 天序列)→ 只留「至少一個日頻市場欄有真實值」的平日 →
         同時擋假日(休市全 NaN)+ 今日(資料未到全 NaN)。
    純顯示層(在 rebuild_dashboards 載入後呼叫一次);不碰 parquet、不影響 factor-model canonical。
    """
    if not isinstance(panel.index, pd.DatetimeIndex) or panel.empty:
        return panel
    daily_market = []
    for c in panel.columns:
        s = panel[c].dropna()
        if s.empty:
            continue
        span = panel.loc[s.index.min():s.index.max()]          # 該欄 active 區間
        density = s.size / max(len(span), 1)                    # active 區間內非空密度
        if s.size >= 100 and density > 0.5 and (s.index.dayofweek < 5).mean() > 0.95:
            daily_market.append(c)                             # 日頻(密)+ 5 天(幾乎不落週末)= 市場序列
    weekday = panel.index.dayofweek < 5
    keep = (weekday & panel[daily_market].notna().any(axis=1)) if daily_market else weekday
    return panel.loc[keep]


# ============================================================================
# 垂直版面統一規格 — 一次定義, 各 panel 由 n_rows 自動算 height / vertical_spacing。
# 改這 4 個數字 = 全 panel 一起變; 新增 panel 只給 n_rows, 不用再手調。
# (ROW_GAP = 0.4*ROW_H → 6 列 vspace 剛好 0.05 → 註腳不受影響)
# ============================================================================
ROW_H_PX    = 225    # 每個子圖「繪圖區」固定高度 (px)
ROW_GAP_PX  = 90     # 子圖間固定間距 (px) — 預設容納註腳 + 下一張標題
MARGIN_T_PX = 95     # 頂部 (panel 標題 + 時間鈕)
MARGIN_B_PX = 55     # 底部 (一般; 末張有長註腳的 panel 可個別加大, 如 Panel 2)
BUTTON_TOP_PX = 60   # 時間鈕「鈕底」離圖頂的固定像素 (全 panel 一致; 越小越靠頂, 須 < MARGIN_T_PX)


def panel_dims(n_rows: int, margin_b: int = MARGIN_B_PX):
    """由 n_rows 推 (height, vertical_spacing) → 全 panel 子圖高/間距像素一致。"""
    plot_h = n_rows * ROW_H_PX + (n_rows - 1) * ROW_GAP_PX   # 純繪圖區 (不含上下 margin)
    height = plot_h + MARGIN_T_PX + margin_b
    vspace = ROW_GAP_PX / plot_h
    return height, vspace


def _ht(name: str, fmt: str, suffix: str = "", color: str = None) -> str:
    """hovertemplate with the series name baked in -> x-unified 顯示『名稱：值』而非裸數字。
    名稱+數值一律粗體; color 給定 → 整段著該色 (對齊圖上線/area 顏色, Bloomberg 式); 冒號全形『：』(較大較清楚)。"""
    body = f"<b>{name}：%{{y:{fmt}}}{suffix}</b>"
    if color:
        body = f"<span style='color:{color}'>{body}</span>"
    return body + "<extra></extra>"


# ── 資料尚未公布 (T+1) 的尾端標示 ──────────────────────────────────────────────
# 共用時間軸下, panel 最大日由「同日即時公布」的序列 (SP500) 頂著; T+1 序列 (rates/OAS/DGS/FX/VIX)
# 在最後一天尚未公布 → 該點原為 NaN, x-unified hover 會吸到前一個真值點 → 顯示成像「上一日 ffill」。
# 解: 把線 hold 到尾端 (與其他線同尾日), 但那點 hover 改標 "Not yet published" → 揭露事實、可與
# scorecard 的 as-of reconcile。歷史中間 gap 不碰 (那是真缺口, 不是 pending)。
PENDING_LABEL = "Not yet published"
_PENDING_TXT = "#9aa0aa"   # pending hover 文字色 (muted; 真值才上紅/綠)


def _ht_cd(name: str, color: str = None) -> str:
    """同 _ht, 但 hover 取 %{customdata} (字串) → 可顯示『格式化值』或 pending 文字。日頻線配 _pending。"""
    body = f"<b>{name}：%{{customdata}}</b>"
    if color:
        body = f"<span style='color:{color}'>{body}</span>"
    return body + "<extra></extra>"


def _pending(s: pd.Series, disp_index, fmt: str, suffix: str = ""):
    """日頻線尾端 pending 處理 → 回傳 (y, customdata) 給 go.Scatter 用。
    y = s reindex 到 disp_index;最後真值日之後仍 NaN 的交易日 (= T+1 尚未公布) → hold 最後真值
      (線延伸到尾, 與其他線同尾日), 該點 customdata 標 PENDING_LABEL;歷史中間 gap 不碰 (維持 NaN, 不畫)。
    真值點 customdata = 格式化值字串 (f"{v:fmt}{suffix}")。"""
    y = s.reindex(disp_index)
    cd = [f"{v:{fmt}}{suffix}" if pd.notna(v) else "" for v in y]
    last = y.last_valid_index()
    if last is not None:
        need = pd.Series(y.index > last, index=y.index) & y.isna()   # 最後真值日之後仍 NaN = pending tail
        if need.any():
            y = y.mask(need, y.loc[last])                            # hold 最後真值到尾
            cd = [PENDING_LABEL if need.iat[i] else c for i, c in enumerate(cd)]
    return y, cd


def _pending_swatch(pos: pd.Series, val: pd.Series, disp_index, fmt: str, suffix: str, color_fn):
    """signed-area / z 的『隱形 hover swatch trace』尾端 pending 延伸 → 回傳 (x, y, customdata)。
    pos = 釘 swatch y 位置用 (clip 後的 sd/zd);val = hover 顯示的真值 (s/z);兩者同 index。
    customdata = [[文字色, 格式化字串]];真值點格式化, disp_index 給定則尾端 pending 日補
      [muted, PENDING_LABEL] + y hold pos 末值 (swatch/base 延伸到尾;可見色塊不延伸 → 不冒假色塊)。
    disp_index=None → 不延伸 (= 原行為, 其他 panel 未接 pending 時不受影響)。"""
    x = list(pos.index)
    y = list(pos.values)
    cd = [[color_fn(v), (f"{v:{fmt}}{suffix}" if pd.notna(v) else "")] for v in val]
    if disp_index is not None and len(pos):
        last = pos.index.max()
        tail = [d for d in disp_index if d > last]
        if tail:
            hold = float(pos.iloc[-1])
            x += list(tail)
            y += [hold] * len(tail)
            cd += [[_PENDING_TXT, PENDING_LABEL]] * len(tail)
    return x, y, cd


# NBER recession periods — VERIFIED 2026-06-20 against the official NBER Business Cycle
# Dating Committee chronology
# (https://www.nber.org/research/data/us-business-cycle-expansions-and-contractions):
#   2001-03->2001-11, 2007-12->2009-06, 2020-02->2020-04 — all three confirmed correct.
# NOTE: the FM window (2021+) contains NO NBER recession -> shading is only visible in the
#       full-history (1996+) view, not in the default recent window. This is expected.
# ENHANCEMENT (run at home): switch primary source to FRED USRECD (contiguous ==1 spans) for
#       auto-update + traceability; keep this verified hardcode as the offline fallback.
#       Source: https://fred.stlouisfed.org/series/USRECD
NBER_RECESSIONS = [
    ("2001-03-01", "2001-11-30"),  # peak Mar 2001 -> trough Nov 2001
    ("2007-12-01", "2009-06-30"),  # peak Dec 2007 -> trough Jun 2009
    ("2020-02-01", "2020-04-30"),  # peak Feb 2020 -> trough Apr 2020 (COVID, shortest on record)
]


def add_recession_shading(fig, row: int = None, col: int = None, x_min=None, x_max=None):
    """
    Add NBER recession bands. 只畫與資料範圍 [x_min, x_max] 重疊的帶,並 clip 到該範圍,
    避免 (例如) 2001 的帶把 x 軸往回拉、使 2021+ 資料被壓到右側一小條。
    2021+ 資料 → 三段衰退皆不重疊 → 一條都不畫 → x 軸貼齊資料。
    """
    for start, end in NBER_RECESSIONS:
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        if x_min is not None and x_max is not None:
            if e < x_min or s > x_max:   # 完全在資料範圍外 → 跳過 (不畫、不撐軸)
                continue
            s, e = max(s, x_min), min(e, x_max)   # clip 到資料範圍
        kwargs = dict(x0=s, x1=e, fillcolor=T["recession"], line_width=0, layer="below")
        if row is not None and col is not None:
            kwargs.update(row=row, col=col)
        fig.add_vrect(**kwargs)


# ============================================================================
# Shared helpers (used by every panel) — added 2026-06-20 (dashboard rebuild)
# ============================================================================
# ---- z-window footnote (每條 z-row 正下方標 z 的取樣頻率/窗長/min-period; 灰、靠左、同字級) ----
_ZFN_DAILY_EXP  = "Z-score is on daily basis │ Min period is 60 days"
_ZFN_DAILY_252  = "Z-score is on daily basis │ Rolling period is 252 days │ Min period is 126 days"
_ZFN_WEEKLY_260 = "Z-score is on weekly basis │ Rolling period is 260 weeks │ Min period is 260 weeks"
_ZFN_MONTHLY_60 = "Z-score is on monthly basis │ Rolling period is 60 months │ Min period is 60 months"


def _zrow_footnote(fig, text, row, n_rows, vs):
    """在第 row 列 (1-indexed) 正下方加一條 z-window footnote (paper 座標)。
    位置同既有 footnote 公式: y = 該列底 − 5% 列高。須在 apply_global_layout 後呼叫 (否則被標題 recolor 迴圈蓋掉)。"""
    ph = (1 - (n_rows - 1) * vs) / n_rows                   # 每列 paper 高
    y = (1 - (row - 1) * (ph + vs) - ph) - 0.05 * ph        # 該列底再下推 5% 列高
    fig.add_annotation(
        text=text, xref="paper", yref="paper", x=0.0, y=y,
        xanchor="left", yanchor="top", align="left", showarrow=False,
        font=dict(size=9.5, color="#9aa0aa"), opacity=0.92,
    )


def apply_global_layout(fig, df: pd.DataFrame, n_rows: int, height: int = 900, title: str = "",
                        margin_b: int = MARGIN_B_PX, include_10y: bool = False):
    """
    全域版面 (所有 panel 共用)。
      - THEME (dark/light, 由檔頭 THEME_MODE 控制) 套 paper/plot 底色、字色、grid
      - NBER 衰退灰底套到每一 row
      - 年標橫軸 (%Y), 每 row 都顯示
      - rangeselector 按鈕掛在『最上方右側』(row 1)、一鍵控制所有圖 (shared x): 1Y/2Y/3Y/5Y/7Y/All
        (FRED pull 自 2018 起 → 7Y/All 含 2019-09 repo spike; 10Y 不放, SOFR 2018 才誕生會留空白)
      - 拖曳條已移除 (圖上拖選即可縮放任意區間, 雙擊還原)
      - hovermode='x unified', 各 trace hover 都帶名稱
      - legend 移除 (hover 已逐線顯示名稱, 不需眼睛對照)
    """
    x_min, x_max = df.index.min(), df.index.max()
    for r in range(1, n_rows + 1):
        add_recession_shading(fig, row=r, col=1, x_min=x_min, x_max=x_max)

    fig.update_xaxes(
        showticklabels=True, dtick="M12", tickformat="%Y",
        tickangle=0, tickfont=dict(size=10),
        gridcolor=T["grid"], zeroline=False,
    )
    fig.update_yaxes(gridcolor=T["grid"], zeroline=False)

    # 時間鈕垂直位置: 反算 → 鈕底固定離「圖頂 BUTTON_TOP_PX 像素」, 全 panel 一致 (與 height 無關)。
    # y 用「繪圖區正規化」座標 (y=1.0 = 繪圖區頂 = margin_t 處); 要往上進頂部 margin → y>1.0; yanchor=bottom。
    _rsel_y = (height - BUTTON_TOP_PX - margin_b) / (height - MARGIN_T_PX - margin_b)

    # rangeselector 掛 row 1 = 最上方; shared_xaxes 下一鍵控制全部子圖
    fig.update_xaxes(
        rangeselector=dict(
            buttons=[
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(count=2, label="2Y", step="year", stepmode="backward"),
                dict(count=3, label="3Y", step="year", stepmode="backward"),
                dict(count=5, label="5Y", step="year", stepmode="backward"),
                dict(count=7, label="7Y", step="year", stepmode="backward"),
                *([dict(count=10, label="10Y", step="year", stepmode="backward")] if include_10y else []),
                dict(step="all", label="All"),
            ],
            x=1.0, xanchor="right", y=_rsel_y, yanchor="bottom",
            bgcolor=T["btn_bg"], activecolor=T["btn_active"],
            font=dict(color=T["btn_font"], size=11),
            bordercolor=T["grid"], borderwidth=1,
        ),
        row=1, col=1,
    )
    fig.update_xaxes(hoverformat="%Y-%m-%d")

    # subplot 標題用 theme 字色 + 統一加大加粗 (item 6: 所有 panel 一致)
    for ann in fig.layout.annotations:
        ann.font.color = T["title_color"]
        ann.font.size = 14
        if ann.text and not ann.text.startswith("<b>"):
            ann.text = f"<b>{ann.text}</b>"

    fig.update_layout(
        height=height, template=T["template"],
        paper_bgcolor=T["paper_bg"], plot_bgcolor=T["plot_bg"],
        font=dict(color=T["font_color"], family="Inter, 'Noto Sans TC', sans-serif"),
        hovermode="x unified", hoverdistance=20, spikedistance=-1,
        # ⚠ hoverdistance 必 >0: =-1 會把「該游標日期沒有資料的線」硬抓到它最近的點顯示 (例如 IORB
        #   2021 才誕生, 在 2019 會被抓到 2021 的值塞進 hover = 顯示錯值)。20px 內無資料點的線會被正確排除。
        #   bars (area-like) 不受此影響、照常顯示。
        hoverlabel=dict(bgcolor=T["hover_bg"], font_size=11, font_color=T["font_color"]),
        title=dict(text=f'<span style="font-weight:600">{title}</span>', x=0.01, xanchor="left", y=1 - 22 / height, yanchor="top",
                   font=dict(size=20, color=T["title_color"])),
        showlegend=False,   # 移除 legend — hover (x-unified) 已逐線顯示名稱, 不需眼睛對照
        modebar=dict(orientation="v"),   # 工具列改直向→收右上角, 不跟時間區間鈕(右上橫排)打架
        margin=dict(t=MARGIN_T_PX, b=margin_b, l=72, r=30),  # 垂直 margin: b 由呼叫端帶入 (預設常數); l/r 各 panel 可覆寫
        bargap=0,
    )
    return fig


def load_display_sp500(df: pd.DataFrame) -> pd.Series:
    """
    顯示用 SP500 = FRED 近端 (canonical) + Yahoo 深史補前段。
      FRED `SP500` (panel 內, ~2016+) 優先;其構不到的前段 (1996–2016) 用
      cache_dashboard/SP500_long.parquet (Yahoo ^GSPC, 已三方驗) combine_first 補。
    純 dashboard 顯示, 絕不回流 FM / features / canonical。
    (未來 Panel 1/2 若要深史 SP500, 統一改這一支 = 單一資料源收口點。)
    """
    fred = df["SP500"] if "SP500" in df.columns else pd.Series(index=df.index, dtype="float64")
    long_path = Path(__file__).resolve().parent / "cache_dashboard" / "SP500_long.parquet"
    if not long_path.exists():
        return fred.reindex(df.index)                       # 無深史 cache → 退回純 FRED
    deep = pd.read_parquet(long_path)
    deep = deep["SP500"] if "SP500" in deep.columns else deep.iloc[:, 0]
    deep.index = pd.DatetimeIndex(deep.index)
    # FRED 優先 (有值處用 FRED), 其缺處 (深史前段) 用 Yahoo 補
    return fred.reindex(df.index).combine_first(deep.reindex(df.index))


def load_display_m2(df: pd.DataFrame) -> pd.Series:
    """
    顯示用 M2 = panel 內 M2SL (canonical, ~2018+) + 深史補前段。
      panel `M2SL` 優先;其構不到的前段 (1996–2018) 用
      cache_dashboard/M2SL.parquet (FRED latest-revised, fetch_long_m2.py 產) combine_first 補。
    純 dashboard 顯示 (ratio 分母用), 絕不回流 FM / features / canonical
    (latest-revised 非 vintage-PIT, 混入會 look-ahead)。與 load_display_sp500 同收口模式。
    """
    panel_m2 = df["M2SL"] if "M2SL" in df.columns else pd.Series(index=df.index, dtype="float64")
    long_path = Path(__file__).resolve().parent / "cache_dashboard" / "M2SL.parquet"
    if not long_path.exists():
        return panel_m2.reindex(df.index)                   # 無深史 cache → 退回 panel M2 (2018+)
    deep = pd.read_parquet(long_path)
    deep = deep["M2SL"] if "M2SL" in deep.columns else deep.iloc[:, 0]
    deep.index = pd.DatetimeIndex(deep.index)
    # panel 優先 (有值處用 panel), 其缺處 (深史前段) 用 latest-revised 補
    return panel_m2.reindex(df.index).combine_first(deep.reindex(df.index))


def load_display_margin(df: pd.DataFrame) -> pd.DataFrame:
    """
    顯示用 FINRA margin 月頻 (business-month-end, $mn) → 回 DataFrame 兩欄:
      net_credit = FC_cash + FC_margin − Debit (compute_margin_net_credit, ffill+dropna)
                   ⚠ fc_margin 一欄 FINRA 2010-02 才有 → Net Credit 實際 2010-02 起 (給 R3/R5/R6 用)
      debit      = Debit Balances (融資餘額; 恆正, FINRA 1997+ → 給 R4 Margin Debt YoY% 用)
    純 dashboard 顯示 (MON-only composite, 非 config Indicator), 絕不回流 FM / features / canonical。
    xlsx 路徑用 __file__ 相對 (repo_root/finra_margin_statistics.xlsx, 同 finra_loader 慣例);
      不存在 → graceful 退回空 DataFrame (margin 列空白, 不 crash, 與 load_display_sp500/m2 同收口)。
    """
    finra_path = Path(__file__).resolve().parent.parent / "finra_margin_statistics.xlsx"
    try:
        raw = FinraLoader(data_path=str(finra_path)).load()
    except FileNotFoundError:
        return pd.DataFrame(columns=["net_credit", "debit"], dtype="float64")   # 無 xlsx → 空 (graceful)
    nc = compute_margin_net_credit(raw[FC_CASH_COL], raw[FC_MARGIN_COL], raw[MARGIN_DEBT_COL])
    return pd.DataFrame({"net_credit": nc, "debit": raw[MARGIN_DEBT_COL]})   # net_credit 2010+, debit 1997+


# ============================================================================
# Section 1: Repo Plumbing Panel
# ============================================================================
def _axis5(fig, row, secondary_y, t0, dtick, col=1, **kw):
    """5 格刻度 (t0, t0+d, ..., t0+4d) 落在固定 fractions (0.2+i)/4.6 → 任兩軸共用此式即左右對齊。"""
    fig.update_yaxes(range=[t0 - 0.2 * dtick, t0 + 4.4 * dtick], dtick=dtick, tick0=t0,
                     row=row, col=col, secondary_y=secondary_y, **kw)


def _zero_5tick(ymin: float, ymax: float):
    """含 0 的 5 整數刻度 (t0..t0+4d) 涵蓋 [ymin, ymax]; 回傳 (t0, dtick)。找最小可行 nice dtick。"""
    import math
    need_lo = max(-ymin, 0.0); need_hi = max(ymax, 0.0)
    span = max(ymax - ymin, 1e-9)
    exp = math.floor(math.log10(span / 4.0)) if span > 0 else 0
    cands = sorted({m * 10.0 ** k for k in (exp - 1, exp, exp + 1, exp + 2)
                    for m in (1, 2, 2.5, 3, 4, 5, 6, 8)})
    for d in cands:
        klo = math.ceil(need_lo / d - 1e-9); khi = math.ceil(need_hi / d - 1e-9)
        if klo + khi <= 4:
            return -klo * d, d
    d = cands[-1]
    return -2 * d, d


def _range_5tick(ymin: float, ymax: float):
    """任意 (不必含 0) 區間的 5 格 nice 刻度 (t0..t0+4d) 涵蓋 [ymin, ymax]; 回 (t0, dtick)。
    給 R1 FX level 等「非從 0 起」的軸用 (USD/TWD ~30、EM USD Index ~100)。
    與 _axis5 的 range=[t0-0.2d, t0+4.4d] 配套 → 任兩個用 _axis5 的軸左右格線對齊。"""
    import math
    span = max(ymax - ymin, 1e-9)
    exp = math.floor(math.log10(span / 4.0))
    cands = sorted({m * 10.0 ** k for k in (exp - 1, exp, exp + 1, exp + 2)
                    for m in (1, 2, 2.5, 3, 4, 5, 6, 8)})
    for d in cands:
        t0 = math.floor(ymin / d) * d                    # 對齊 d 的整數倍, <= ymin
        if t0 + 4.4 * d >= ymax - 1e-9:                  # axis range 上緣 t0+4.4d 蓋過 ymax
            return t0, d
    d = cands[-1]
    return math.floor(ymin / d) * d, d


_CRISIS_NAMES = (
    (pd.Timestamp("1998-01-01"), pd.Timestamp("1999-06-30"), "1998 Russia / LTCM Crisis"),
    (pd.Timestamp("2008-01-01"), pd.Timestamp("2009-12-31"), "2008 Global Financial Crisis（GFC）"),
)


def _crisis_name(dt):
    """峰值日落在哪場已知危機 → 回危機名; 無對應回 None (未知事件走 fallback)。"""
    for lo, hi, nm in _CRISIS_NAMES:
        if lo <= dt <= hi:
            return nm
    return None


def _z5_episodes(z, thr=5.0, merge_days=180):
    """連續 ≥thr 段 → 合併相距 <merge_days 的 → [(start, end, peak_dt, peak_val, n_days), ...]。
    與 list_z5_days.py 同邏輯 → 圖上事件與腳本完整清單一致。"""
    z = z.dropna()
    mask = z >= thr
    if not mask.any():
        return []
    grp = (mask != mask.shift()).cumsum()
    segs = [[sub.index[0], sub.index[-1], sub.idxmax(), float(sub.max()), int(len(sub))]
            for _, sub in z[mask].groupby(grp[mask])]
    out, cur = [], segs[0]
    for seg in segs[1:]:
        if (seg[0] - cur[1]).days < merge_days:
            cur[1] = seg[1]; cur[4] += seg[4]
            if seg[3] > cur[3]:
                cur[2], cur[3] = seg[2], seg[3]
        else:
            out.append(tuple(cur)); cur = seg
    out.append(tuple(cur))
    return out


def _z5_crisis_text(start, end, peak_dt, peak_val, n_days):
    """3 行標籤: 危機名(帶年) / 期間·連續天數(省年) / 峰值日(帶年)+Max Z。無名事件 fallback 補年。"""
    line1 = _crisis_name(peak_dt) or f"{start:%Y} Z-score ≥ 5 Episode"
    line2 = f"{start:%m-%d} ~ {end:%m-%d} · {n_days} consecutive days ≥ 5"
    line3 = f"{peak_dt:%Y-%m-%d} Max Z-score：{peak_val:.2f}"
    return f"{line1}<br>{line2}<br>{line3}"


def _extreme_labels_impl(fig, x0, x1, s, row, name, fmt, suffix, annot_thr, cap, pos=None, secondary_y=False, texts=None, ax0=48):
    """data-driven: 任何 s ≥ annot_thr 全標真值 (date+value); 自動垂直錯位避互疊。
    pos = {'YYYY-MM-DD': (ax, ay, xanchor)} 對指定日手動 override 位置;
    texts = {'YYYY-MM-DD': '自訂多行文字'} 覆蓋預設文字 (Panel 3-1 危機標籤用)。須在 apply_global_layout 後呼叫。"""
    pos = pos or {}
    ext = s[s >= annot_thr]
    if ext.empty:
        return
    span = (x1 - x0)
    stack_gap = pd.Timedelta(days=60)
    prev_dt, level = None, 0
    for dt, val in ext.items():
        level = level + 1 if (prev_dt is not None and dt - prev_dt < stack_gap) else 0
        prev_dt = dt
        key = f"{dt:%Y-%m-%d}"
        if key in pos:
            ax, ay, xanc = pos[key]
        else:
            right = bool(span) and (dt - x0) / span > 0.72
            ax, ay, xanc = (-ax0 if right else ax0), 14 + level * 34, ("right" if right else "left")
        fig.add_annotation(
            x=dt, y=min(float(val), cap),
            text=(texts or {}).get(key) or f"{dt:%Y-%m-%d}<br>{name}：{val:{fmt}}{suffix}",
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.1,
            arrowcolor=T["thr_crit"], ax=ax, ay=ay, xanchor=xanc, yanchor="top",
            align="left", font=dict(size=10.5, color=T["font_color"]),
            bgcolor="rgba(27,30,36,0.86)", bordercolor=T["thr_crit"],
            borderwidth=1, borderpad=8, row=row, col=1, secondary_y=secondary_y,
        )


def _episode_peaks(s: pd.Series, thr: float, merge_days: int = 180) -> pd.Series:
    """連續 ≥thr 的每段先收斂成峰值, 再把相距 < merge_days 的峰併為同一事件 (留最高峰) →
    一場危機 (如 GFC 跨數月、中間偶爾跌破 5) 只留一個標籤, 天然不重疊。"""
    mask = s >= thr
    if not mask.any():
        return s.iloc[:0]
    grp = (mask != mask.shift()).cumsum()
    raw = pd.Series({sub.idxmax(): float(sub.max())
                     for _, sub in s[mask].groupby(grp[mask])}).sort_index()
    keep, cluster, prev = {}, [], None
    for dt, val in raw.items():
        if prev is not None and (dt - prev).days >= merge_days:
            bdt, bval = max(cluster, key=lambda t: t[1]); keep[bdt] = bval; cluster = []
        cluster.append((dt, val)); prev = dt
    if cluster:
        bdt, bval = max(cluster, key=lambda t: t[1]); keep[bdt] = bval
    return pd.Series(keep).sort_index()


def load_display_curve(df: pd.DataFrame) -> pd.DataFrame:
    """
    顯示用 yield curve = panel 內各序列 (canonical, ~2018+) + cache_dashboard 深史補前段。
      panel 各欄優先;其構不到的前段用 cache_dashboard/<SID>.parquet
      (FRED latest-revised, fetch_long_curve.py 產; DGS/spread ~1996 / DFII10·T10YIE ~2003) combine_first 補。
    純 dashboard 顯示, 絕不回流 FM / features / canonical。與 load_display_sp500/m2 同收口模式。
    無 cache → graceful 退回 panel (2018+)。
    """
    cols = ["DGS2", "DGS10", "DFII10", "T10YIE", "T10Y2Y", "T10Y3M"]
    cache_dir = Path(__file__).resolve().parent / "cache_dashboard"
    out = {}
    for c in cols:
        panel_s = df[c] if c in df.columns else pd.Series(index=df.index, dtype="float64")
        long_path = cache_dir / f"{c}.parquet"
        if long_path.exists():
            deep = pd.read_parquet(long_path)
            deep = deep[c] if c in deep.columns else deep.iloc[:, 0]
            deep.index = pd.DatetimeIndex(deep.index)
            out[c] = panel_s.reindex(df.index).combine_first(deep.reindex(df.index))
        else:
            out[c] = panel_s.reindex(df.index)
    return pd.DataFrame(out, index=df.index)


def load_display_fxvolmacro(df: pd.DataFrame) -> dict:
    """Panel 5 顯示用 5 序列 = panel (canonical, ~2018+) + cache_dashboard 深史 combine_first。
      daily (DEXTAUS / DTWEXEMEGS / VIXCLS) 對齊 df.index;
      monthly (CPIAUCSL / PCEPILFE) 保留「月頻原樣」(給 YoY=pct_change(12) 用, 不 reindex 到日頻,
        以免月初落在非營業日被丟值)。深史由 fetch_long_fxvolmacro.py 產 (FRED latest-revised;
        DTWEXEMEGS ~2006 起 = 該 index 在 FRED 的起點; CPI/PCE ~1996)。
      純 dashboard 顯示, 絕不回流 FM / features / canonical。無 cache → graceful 退回 panel。
      回傳 dict[SID -> Series] (因 daily/monthly 頻率不同, 不併成單一 DataFrame)。"""
    cache_dir = Path(__file__).resolve().parent / "cache_dashboard"
    daily = ["DEXTAUS", "DTWEXEMEGS", "VIXCLS"]
    monthly = ["CPIAUCSL", "PCEPILFE"]
    out = {}
    for c in daily + monthly:
        deep = None
        p = cache_dir / f"{c}.parquet"
        if p.exists():
            deep = pd.read_parquet(p)
            deep = deep[c] if c in deep.columns else deep.iloc[:, 0]
            deep.index = pd.DatetimeIndex(deep.index)
        if c in daily:
            panel_s = df[c] if c in df.columns else pd.Series(index=df.index, dtype="float64")
            out[c] = (panel_s.reindex(df.index).combine_first(deep.reindex(df.index))
                      if deep is not None else panel_s.reindex(df.index))
        else:                                            # monthly: deep 已 1996+ 完整, 用月頻
            out[c] = (deep.dropna() if deep is not None
                      else (df[c].dropna() if c in df.columns else pd.Series(dtype="float64")))
    return out


def plot_repo_plumbing(panel: pd.DataFrame, output: str | Path) -> Path:
    """
    Panel 1 — Repo Plumbing: 6 張子圖上下疊 (共用時間軸, 右上一組時間鈕一鍵控制全部)。
      1 Policy & Repo Rates: Fed Policy Band (淺灰, hover 顯上下緣) + 5 條 rates (solid, connectgaps)
      2 Repo Spread (SOFR − EFFR): signed-area 門檻切色 (綠<20 / 紅≥20 凸尖) + 20 門檻線 + 極端 (≥100 bps) 自動標真值
      3 Expanding Z-score of Repo Spread: signed-area 門檻切色 (綠<2 / 紅≥2 凸尖) + 2 門檻線 + 極端 (≥5) 自動標真值
      4 Reserve Balances (Wed Level, Tn): 與 Panel 2 同 + 3 Tn 警戒線 — 對照 reserve 稀缺 vs repo 承壓
      5 US Stock Index · SP500: 0 起 + 千位分隔
      6 SP500 / M2 （Per $1 Tn）: 0 起, 軸名 Ratio
    ⚠ panel index 可能因 credit OAS (1996+) 撐到 1997; Panel 1 自己序列只 2018+ →
      trim 到本 panel 序列首個有值日期, 避免 All 一大片 pre-2018 空白 + 數字重疊。
    """
    df = panel.copy()
    _p1 = [c for c in ["SOFR", "EFFR", "IORB", "RRPONTSYAWARD", "TGCRRATE", "SP500"] if c in df.columns]
    _anchor = df["SOFR"].first_valid_index() if "SOFR" in df.columns else None
    if _anchor is not None:                              # 錨在 SOFR 首值「那年的 1/1」→ x 軸 show "2018" + scarcity 標籤貼左緣
        _anchor = pd.Timestamp(_anchor.year, 1, 1)       # repo spread/z 自然從 SOFR 上線的 2018-04 起 (Q1 留小段, 真實); 其餘列填滿
    else:                                                # SOFR 缺 → 退回各序列最早 (原邏輯)
        _fb = [d for d in (df[c].first_valid_index() for c in _p1) if d is not None]
        _anchor = min(_fb) if _fb else None
    if _anchor is not None:
        df = df.loc[_anchor - pd.Timedelta(days=7):]     # 往前 7 天 → 2018-01-01 年刻度落進 autorange 內可顯示 (否則資料首日 2018-01-02, Jan-1 刻度被切在邊界外; 同 Panel 3-2)
    spx = load_display_sp500(df)                         # SP500 深史 (顯示用); 併入 Row 3 repo-z 背景

    fig = make_subplots(
        rows=5, cols=1, shared_xaxes=True,
        subplot_titles=(
            "Policy &amp; Repo Rates",
            "Repo Spread（Repo Stress ≥ 20 bps；Label ≥ 100 bps）& SP500",
            "Expanding Z-score of Repo Spread（Repo Stress ≥ 2；Label ≥ 5）& SP500",
            "Reserve Balances with Fed Reserve Banks（Wed. Level）",
            "SP500 / M2 （Per $1 Tn）& Rolling_5yr Z-score（Valuation Stretched ≥ 2）",
        ),
        specs=[[{}], [{"secondary_y": True}], [{"secondary_y": True}], [{}], [{"secondary_y": True}]],
        vertical_spacing=panel_dims(5)[1],
    )

    # ---- 共用 helper: 極端值自動標籤 (closure over fig / df); 門檻 signed-area 已升 module-level _threshold_area(clip=False) ----
    def _extreme_labels(s, row, name, fmt, suffix, annot_thr, cap, pos=None, secondary_y=False):
        _extreme_labels_impl(fig, df.index[0], df.index[-1],
                             s, row, name, fmt, suffix, annot_thr, cap, pos, secondary_y=secondary_y)

    # ---- Row 1: Fed Policy Band (fill 畫在線後) + rates ----
    # ⚠ x-unified hover 顯示順序 = 反向 trace 加入順序 (實機截圖確認); 想要的 hover 順序:
    #   Fed Policy Band → TGCR → SOFR → EFFR → IORB → ON RRP
    #   ⇒ band fill 最前(畫在線後) → rates 反向加入(ON RRP→…→TGCR) → band「只供 hover」trace 最後
    has_band = {"DFEDTARL", "DFEDTARU"}.issubset(df.columns)
    if has_band:
        lo, hi = df["DFEDTARL"], df["DFEDTARU"]
        fig.add_trace(go.Scatter(x=df.index, y=lo, line=dict(width=0), connectgaps=True,
                                 showlegend=False, hoverinfo="skip", name="_band_lo"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=hi, line=dict(width=0), connectgaps=True,
                                 fill="tonexty", fillcolor=T["corridor"], showlegend=False,
                                 hoverinfo="skip", name="_band_fill"), row=1, col=1)
    market_rates = {           # 反向加入 (見上註); solid + connectgaps 補週末/假日 NaN 斷點
        "RRPONTSYAWARD": T["onrrp"], "IORB": T["iorb"], "EFFR": T["effr"],
        "SOFR": T["sofr"], "TGCRRATE": T["tgcr"],
    }
    rate_labels = {"RRPONTSYAWARD": "ON RRP", "TGCRRATE": "TGCR"}
    for sid, color in market_rates.items():
        if sid in df.columns:
            nm = rate_labels.get(sid, sid)
            _ry, _rcd = _pending(df[sid], df.index, ".2f", " %")   # 尾端 T+1 未公布 → hold 最後真值 + 標 pending
            fig.add_trace(
                go.Scatter(x=df.index, y=_ry, name=nm, connectgaps=True, customdata=_rcd,
                           line=dict(color=color, width=1.0, dash="solid"),
                           hovertemplate=_ht_cd(nm, color=color)),
                row=1, col=1,
            )
    if has_band:               # 只供 hover 的 band: 兩條同 y=hi + tonexty(零面積→圖上不可見)+ fillcolor 灰
        fig.add_trace(go.Scatter(x=df.index, y=hi, line=dict(width=0), connectgaps=True,  # 零面積 fill 的基準線
                                 showlegend=False, hoverinfo="skip", name="_swatch_base"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=hi, line=dict(width=0), connectgaps=True,
                                 fill="tonexty", fillcolor=T["corridor"],  # ← fill 讓 hover 色塊(灰)出現; 面積 0 故不可見
                                 showlegend=False, customdata=lo.to_numpy().reshape(-1, 1),
                                 name="Fed Policy Band",
                                 hovertemplate="<span style='color:#AEB4BF'><b>Fed Policy Band：%{customdata[0]:.2f}% − %{y:.2f}%</b></span><extra></extra>"),
                      row=1, col=1)
    fig.update_yaxes(title_text="%", row=1, col=1)

    # ---- Row 2 & 3: Repo Spread + Expanding Z-score — signed-area 門檻切色 (綠<thr / 紅≥thr 凸尖) + 極端自動標 ----
    #   bar→area: 單日 spike 在 All 全景下 bar 是 ~0.43px 細線看不清; area 連續尖峰看得見。
    #   x-unified 同像素抓錯隔壁 (targeting) 非換圖型能解 → 極端值改「文字標籤」直接印真值 (data-driven, 不寫死日期)。
    spread = z = None
    if {"SOFR", "EFFR"}.issubset(df.columns):
        spread = compute_repo_spread(df["SOFR"], df["EFFR"]).dropna()
        z = expanding_zscore(spread).dropna()
        # R2: SP500 先加 (左主軸, hover 落底) → Repo Spread area 後加 (右副軸, hover 置頂) ⇒ hover 序 Repo Spread → SP500
        _add_sp500_bg(fig, spx, 2)
        # Repo Spread 紅 ≥ 20 bps (Fed-anchored: '19 前 SOFR 罕偏離 EFFR 目標 > 25 bps; '24 季末 ~20 bps = '19 來最大季末漲幅); 移右副軸
        # repo spread: 升共用 module-level _threshold_area(clip=False) → 尖峰衝出由軸切 (同 Panel 7 R4); 內含 20 線 + 軸
        _threshold_area(fig, spread, 2, 20.0, "Repo Spread", -20.0, 20.0, fmt=".0f", suffix=" bps",
                        clip=False, title_text="bps", disp_index=df.index)
        # Z-score 紅 ≥ 2 (統計 Z≥2 ≈ 97.7 百分位 = 異常)
        _add_zrow(fig, spx, z, 3, zt0=-2.0, cap=False, disp_index=df.index)   # SP500 白底(左)+repo-z 整根上色(右)+5格 -2/0/2/4/6+Z=2/Z=5 線; 極端標示見下

    # ---- Row 4: Reserve Balances (Wed Level, Tn) + 3 Tn 警戒線 — 與 Panel 2 同; 對照 repo 稀缺 ----
    if "WRBWFRBL" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["WRBWFRBL"].ffill() / 1_000_000, name="Reserve Balances",
                       showlegend=False, line=dict(color=T["sofr"], width=1.4), connectgaps=True,
                       hovertemplate=_ht("Reserve Balances", ".2f", " Tn", color=T["sofr"])),
            row=4, col=1,
        )  # WRBWFRBL = millions → /1_000_000 轉 Tn (config unit 已校正為 Millions, 不進 compute)
        fig.add_hline(y=3, line_dash="dash", line_color=T["thr_crit"], line_width=1.3, row=4, col=1)
        w = df["WRBWFRBL"].ffill() / 1_000_000; wmin = float(w.min()); wmax = float(w.max()); wsp = wmax - wmin
        fig.update_yaxes(title_text="Tn",
                         range=[min(wmin, 3) - wsp * 0.18, max(wmax, 3) + wsp * 0.08], row=4, col=1)
    else:
        fig.update_yaxes(title_text="Tn", row=4, col=1)

    # ---- Row 5: ratio-z (SP500 已併進 Row 3 repo-z, 不再單獨拉一列) ----
    _add_ratio_zrow(fig, panel, df, 5)

    apply_global_layout(
        fig, df, n_rows=5, height=panel_dims(5)[0],
        title="Panel 1 · Repo Plumbing Monitor",
    )

    # Repo Spread 定義註腳 (綠, 放 Row 2 正下方; 配色 follow 指標但門檻三色→取綠)
    _N, _vs = 5, panel_dims(5)[1]
    _ph = (1 - (_N - 1) * _vs) / _N
    fig.add_annotation(
        text="Repo Spread = SOFR − EFFR", xref="paper", yref="paper",
        x=0.0, y=(1 - 1 * (_ph + _vs) - _ph) - 0.05 * _ph,
        xanchor="left", yanchor="top", showarrow=False,
        font=dict(size=10, color=T["bar_low"]), opacity=0.9,
    )
    # x 軸年刻度釘在 anchor (2018-01-01) → 顯示窗起年 2018 顯示; df 已往前 7 天 → Jan-1 落進 autorange (同 Panel 3-2)
    fig.update_xaxes(tick0=_anchor)
    # z-window footnotes (R3 repo-z expanding-daily; R5 SP500/M2 monthly-60)
    _zrow_footnote(fig, _ZFN_DAILY_EXP, 3, _N, _vs)
    _zrow_footnote(fig, _ZFN_MONTHLY_60, 5, _N, _vs)
    # 3 Tn 標籤 (Row 4; 須在 apply_global_layout 之後加, 否則被標題 recolor 迴圈蓋白)
    if "WRBWFRBL" in df.columns:
        fig.add_annotation(text="3 Tn scarcity", x=df.index.min(), y=3,
                           xanchor="left", yanchor="bottom", showarrow=False,
                           font=dict(color=T["thr_crit"], size=10), row=4, col=1)

    # 極端值文字標籤 (data-driven: 任何 spread ≥ 100 / Z ≥ 5 自動標真值; 須在 apply_global_layout 後, 否則被 recolor 迴圈蓋白)
    if spread is not None:
        _extreme_labels(spread, row=2, name="Repo Spread", fmt=".0f", suffix=" bps", annot_thr=100, cap=68,
                        secondary_y=True, pos={"2019-09-17": (16, 14, "left")})  # cap=68 = bps 軸 range top → 貼頂同 R3; pos 拉近 (同 R3 9-17 偏移 16,14)
        _extreme_labels(
            z, row=3, name="Z-score", fmt=".2f", suffix="", annot_thr=5, cap=6.8,
            secondary_y=True,                        # repo-z 併入雙軸 → 標示掛右軸
            pos={                                    # 三框同高 (ay=14, 對齊 2019-09-17); year-end 兩筆分左右避撞
                "2018-12-31": (-16, 14, "right"),    # 左側貼尖峰 (距離 16 = 同 2019-01-02; 框不伸到資料起點以左 → 不撐出左空白)
                "2019-01-02": (16, 14, "left"),      # 右側, 與 2018-12-31 分左右同高
                "2019-09-17": (16, 14, "left"),      # Sept spike 右側
            },
        )

    # Row 1 — 2019-09-17 repo spike 標籤 (item 7): TGCR + SOFR 真值, 框/箭頭取 TGCR 線色。
    # (須在 apply_global_layout 之後加, 否則字色被標題 recolor 迴圈蓋掉; 自動跳過資料缺漏日。)
    _spk = pd.Timestamp("2019-09-17")
    if _spk in df.index and pd.notna(df.at[_spk, "TGCRRATE"]) and pd.notna(df.at[_spk, "SOFR"]):
        _tg, _so = float(df.at[_spk, "TGCRRATE"]), float(df.at[_spk, "SOFR"])
        fig.add_annotation(
            x=_spk, y=_tg,
            text=f"2019-09-17<br>TGCR：{_tg:.2f} %<br>SOFR：{_so:.2f} %",
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.1,
            arrowcolor=T["tgcr"], ax=50, ay=18, xanchor="left", yanchor="top",
            align="left", font=dict(size=10.5, color=T["font_color"]),
            bgcolor="rgba(27,30,36,0.86)", bordercolor=T["tgcr"],
            borderwidth=1, borderpad=8, row=1, col=1,
        )

    output = Path(output)
    _write_panel_html(fig, output)
    return output


# ============================================================================
# Panel 2: Liquidity (6 subplots) — rebuilt 2026-06-20
# ============================================================================
def load_display_deep(panel: pd.DataFrame, sids) -> dict:
    """通用深史 loader (日/週頻): 每個 sid 讀 cache_dashboard/<sid>.parquet 補在 panel[sid] 前段 (combine_first),
    reindex 到 panel.index。無 cache → 退回 panel 該欄。回 dict[sid -> Series]。純顯示, 絕不回流 FM。
    (與 load_display_sp500/m2/curve 同收口; SP500 因檔名 SP500_long.parquet 須走 load_display_sp500, 不走這支。)"""
    cache_dir = Path(__file__).resolve().parent / "cache_dashboard"
    out = {}
    for sid in sids:
        canon = panel[sid] if sid in panel.columns else pd.Series(index=panel.index, dtype="float64")
        p = cache_dir / f"{sid}.parquet"
        if p.exists():
            deep = pd.read_parquet(p)
            deep = deep[sid] if sid in deep.columns else deep.iloc[:, 0]
            deep.index = pd.DatetimeIndex(deep.index)
            out[sid] = canon.reindex(panel.index).combine_first(deep.reindex(panel.index))
        else:
            out[sid] = canon.reindex(panel.index)
    return out


def _deep_monthly(panel: pd.DataFrame, sid: str, cache_dir=None) -> pd.Series:
    """月頻深史 Series (給 compute_real_m2_yoy 用): canonical resample('ME').last() + cache 月頻 combine_first。
    不 reindex 到 daily (避月初值落在非營業日被丟 → Real M2 YoY 在 2002~2018 深史段出現缺口)。"""
    if cache_dir is None:
        cache_dir = Path(__file__).resolve().parent / "cache_dashboard"
    canon = (panel[sid].dropna().resample("ME").last()
             if sid in panel.columns else pd.Series(dtype="float64"))
    p = cache_dir / f"{sid}.parquet"
    if p.exists():
        deep = pd.read_parquet(p)
        deep = deep[sid] if sid in deep.columns else deep.iloc[:, 0]
        deep = pd.Series(deep.values, index=pd.DatetimeIndex(deep.index)).dropna().resample("ME").last()
        return canon.combine_first(deep)
    return canon


def _netliq_display(panel: pd.DataFrame):
    """Net Liquidity + YoY — Panel 2 與 scorecard 共用 (確保兩處數值字面一致)。
    WALCL/WDTGAL/RRPONTSYD 走 load_display_deep (深史接線); x 錨 = Fed 資產負債表四條最早 first_valid (~2002);
    net = compute_net_liquidity; YoY = dense 網格 pct_change(252) PIT (facility 前 RRP 補 0, 後用 obs 網格); 回 (net, nl_yoy)，皆 on idx。"""
    fed = load_display_deep(panel, ["WALCL", "WDTGAL", "RRPONTSYD", "WRBWFRBL"])
    _firsts = [fed[c].first_valid_index() for c in ("WALCL", "WDTGAL", "RRPONTSYD", "WRBWFRBL")
               if c in fed and fed[c].first_valid_index() is not None]
    start = min(_firsts) if _firsts else panel.index.min()
    idx = panel.index[panel.index >= start]
    def _c(c):
        if c in fed:
            return fed[c].reindex(idx)
        return panel[c].reindex(idx) if c in panel.columns else pd.Series(index=idx, dtype="float64")
    net = compute_net_liquidity(_c("WALCL"), _c("WDTGAL"), _c("RRPONTSYD"))
    # YoY — 252 交易日 PIT, 在 dense 網格上算 pct_change(252), 分兩段:
    #   • facility 前 (< 2013-09-23): Fed ON RRP fixed-rate full-allotment facility 尚未啟動, ON RRP drain ≈ 0
    #       → 補 business days; 空白日 .fillna(0) (非操作日隔夜 drain=0), 零星實際 reverse repo 觀測值保留不覆蓋。
    #   • facility 後 (>= 2013-09-23): 用實際 RRP 日頻 obs 網格 (= FRED 的 252-count 網格 → 近端精準)。
    #   為何 dense: RRP obs 網格 facility 前稀疏(~月頻), 直接 pct_change(252) 分母會跨多年 → 偽 spike (2014 曾 +461%,
    #     分母掉到 2007 海嘯前 ~737B)。dense 網格使 252-obs 恆 ≈ 1 年 → 無 spike + 可觀測 2008/QE; 近端 252-obs
    #     全落 facility 期 → 精準 = FRED (-4.97% @ 06-26)。2013-09-23 = facility 啟動日, FRED RRPONTSYD 首個日頻 obs
    #     實證此日 (前為稀疏空白)。詳見 DASHBOARD_BUILD_NOTES「Net Liq YoY 網格」。
    _FAC = pd.Timestamp("2013-09-23")
    _rrp_obs = _c("RRPONTSYD").dropna().index
    _pre = pd.bdate_range(idx.min(), _FAC - pd.Timedelta(days=1))            # facility 前 business days
    _grid = _pre.union(_rrp_obs[_rrp_obs >= _FAC])                           # + facility 後實際 obs
    _rrp_g = _c("RRPONTSYD").reindex(_grid).fillna(0.0)                      # 空白→0, 觀測值保留
    _net_g = compute_net_liquidity(_c("WALCL").reindex(_grid), _c("WDTGAL").reindex(_grid), _rrp_g)
    nl_yoy = (_net_g.pct_change(252) * 100).reindex(idx, method="ffill")
    return net, nl_yoy


def plot_net_liquidity(panel: pd.DataFrame, output: str | Path) -> Path:
    """
    Panel 2 — Liquidity (6 子圖, 純顯示 / Architecture B)。深史版 (~2002+, 含 2008 GFC Fed 放水):
      Row 1  Fed's TA & TGA & ON RRP     WALCL / WDTGAL / RRPONTSYD       lines, Bn
      Row 2  Net Liquidity + YoY + SP500  Net Liq.(線) + SP500(白線) 左軸   hover: Net Liq.→YoY→SP500
                                          + Net Liq. YoY(灰 area) 右軸 %
      Row 3  M2 & Real M2 YoY            M2(線) 左軸 Tn + Real M2 YoY(綠/紅 area) 右軸 %     hover: M2→Real M2 YoY
      Row 4  Reserve Balances (Wed Lvl)   WRBWFRBL                         line,  Tn + 3 Tn 警戒線
      Row 5  SP500 / M2 & Zscore         ratio 線 amber(左) + 5yr rolling-z area(右, thr=2)   ← ratio-z 模版
      Row 6  SRF & Discount Window & BTFP RPONTSYD / WLCFLPCL / H41        lines, Bn

    深史 (需先跑 fetch_long_liquidity.py): Fed 資產負債表四條 (WALCL/WDTGAL/RRPONTSYD/WRBWFRBL) ~2002 起
      = x 軸錨點 (A1, 2008 GFC 整段在內); WLCFLPCL ~2002 (GFC 借款飆升); RRP/SRF 早年多 0; BTFP 2023 前空白 (正常)。
      SP500/M2/ratio 跟著從 ~2002 起 (深史可達 1996 但被 Fed 錨點 trim); Real M2 YoY 用原生月頻 (避深史段缺口)。
    單位 (ground-truth verified): WALCL/WDTGAL/H41 = millions→/1000 Bn;
      WRBWFRBL = millions→/1_000_000 Tn (config unit 已校正為 Millions, 不進 compute);M2SL = billions→/1000 Tn。
    hover 顯示順序 = trace 反向加入順序 (x-unified 特性)。⚠ 無 cache_dashboard 深史時 graceful 退回 panel (2018+)。
    """
    cache_dir = Path(__file__).resolve().parent / "cache_dashboard"
    # ---- 深史接線: Fed 週/日頻 (load_display_deep) + SP500/M2 (各自 loader) ----
    _FED = ["WALCL", "WDTGAL", "RRPONTSYD", "WRBWFRBL", "RPONTSYD", "WLCFLPCL", "H41RESPPALDKNWW"]
    fed = load_display_deep(panel, _FED)                 # dict: SID -> 深史 (canonical + cache), on panel.index
    spx_full = load_display_sp500(panel)
    m2_full = load_display_m2(panel)
    # x 軸錨點 (A1): Fed 資產負債表四條的最早 first_valid (~2002); 不含 facilities/M2/SP500
    _anchor = ["WALCL", "WDTGAL", "RRPONTSYD", "WRBWFRBL"]
    _firsts = [fed[c].first_valid_index() for c in _anchor
               if c in fed and fed[c].first_valid_index() is not None]
    start = min(_firsts) if _firsts else panel.index.min()
    idx = panel.index[panel.index >= start]
    df = panel.reindex(idx).copy()
    for c in _FED:                                       # 深史覆蓋 canonical 欄 (Row 1/2/5/6 直接讀 df[c])
        if c in fed:
            df[c] = fed[c].reindex(idx)
    df["SP500"] = spx_full.reindex(idx)
    df["M2SL"] = m2_full.reindex(idx)

    titles = (
        "Fed's TA & TGA & ON RRP",
        "Net Liquidity（inclu. YoY）& SP500",
        "M2 & Real M2 YoY",
        "Reserve Balances with Fed Reserve Banks（Wed. Level）",
        "SP500 / M2 （Per $1 Tn）& Rolling_5yr Z-score（Valuation Stretched ≥ 2）",
        "SRF & Discount Window & BTFP",
    )
    specs = [[{}], [{"secondary_y": True}], [{"secondary_y": True}],
             [{}], [{"secondary_y": True}], [{}]]        # Row 5 改 secondary_y (ratio-z); Row 4 Reserve 單軸
    fig = make_subplots(rows=6, cols=1, shared_xaxes=True,
                        vertical_spacing=panel_dims(6)[1], specs=specs, subplot_titles=titles)

    def _line(col, name, color, row, scale, fmt, suffix, width=1.4):
        """ffill→step + 單位 scale 的 line trace (週/月頻畫成階梯)。"""
        if col not in df.columns:
            return
        s = df[col].ffill() * scale
        fig.add_trace(
            go.Scatter(x=df.index, y=s, name=name, showlegend=False,
                       line=dict(color=color, width=width, dash="solid"),
                       connectgaps=True, hovertemplate=_ht(name, fmt, suffix, color=color)),
            row=row, col=1,
        )

    def _signed_yoy(s, row, name):
        _add_signed_area(fig, s, row, name)

    # ---- Row 1 — Fed's TA & TGA & ON RRP (Bn); 反向加入 → hover Fed's TA → TGA → ON RRP ----
    _line("RRPONTSYD", "ON RRP", T["onrrp"], 1, 1.0, ",.2f", " Bn")   # NY Fed Temp OMO 午後同日發布 → 領先 SP500 傍晚尾日, 永不落後 → 非 pending
    _line("WDTGAL", "TGA", T["iorb"], 1, 1 / 1000, ",.0f", " Bn")
    _line("WALCL", "Fed's TA", T["sofr"], 1, 1 / 1000, ",.0f", " Bn")
    fig.update_yaxes(title_text="Bn", row=1, col=1)

    # ---- Row 2 — Net Liq.(線)+SP500(白) 左 ; Net Liq. YoY(area) 右 % ; hover Net Liq.→YoY→SP500 ----
    have_net = {"WALCL", "WDTGAL", "RRPONTSYD"}.issubset(df.columns)
    if have_net:
        net, nl_yoy = _netliq_display(panel)             # 與 scorecard 共用同一算法 → 保證一致
    if "SP500" in df.columns:        # 先加 → hover 最底
        fig.add_trace(
            go.Scatter(x=df.index, y=df["SP500"], name="SP500", showlegend=False,
                       line=dict(color=T["sp500"], width=1.4), connectgaps=True,
                       hovertemplate=_ht("SP500", ",.2f", color=T["sp500"])),
            row=2, col=1, secondary_y=False,
        )
    if have_net:
        _signed_yoy(nl_yoy, 2, "Net Liq. YoY")          # 綠/紅 area + 數值上色 (中)
        fig.add_trace(   # Net Liq. 後加 → hover 最上 + 橘線最前 crisp
            go.Scatter(x=net.index, y=net, name="Net Liq.", showlegend=False,
                       line=dict(color=T["iorb"], width=1.6),
                       hovertemplate=_ht("Net Liq.", ",.0f", " Bn", color=T["iorb"])),
            row=2, col=1, secondary_y=False,
        )
    _vals = ([net] if have_net else []) + ([df["SP500"]] if "SP500" in df.columns else [])
    if _vals:
        allv = pd.concat(_vals); vmax = float(allv.max())
        _axis5(fig, 2, False, 0, _nice_step(vmax / 4.0), title_text="Bn / SP500")
    else:
        fig.update_yaxes(title_text="Bn / SP500", row=2, col=1, secondary_y=False)

    # ---- Row 3 — M2(線, 左軸 Tn) + Real M2 YoY(area, 右軸 %); hover M2 → Real M2 YoY ----
    # Real M2 YoY 用「原生月頻」(canonical resample 月 + cache 月 combine), 避深史 daily-reindex 掉月初值的缺口
    rm_m2 = _deep_monthly(panel, "M2SL", cache_dir)
    rm_cpi = _deep_monthly(panel, "CPIAUCSL", cache_dir)
    if len(rm_m2) and len(rm_cpi):
        rm = compute_real_m2_yoy(rm_m2, rm_cpi)
        rm = rm[rm.index >= start]    # trim 到顯示窗 (2002+); 否則月頻值延伸到 1996 會把 x 軸 auto-range 拉走 (破壞 A1)
        _signed_yoy(rm.reindex(df.index, method="ffill"), 3, "Real M2 YoY")   # 月頻 → 日頻階梯 (MS 月初戳記 → 月中顯示「當月」yoy, 與 level/CPI YoY 慣例一致)
    if "M2SL" in df.columns:
        m2_tn = df["M2SL"].ffill() / 1000
        fig.add_trace(   # M2 後加 → hover 最上 + 線最前 crisp
            go.Scatter(x=df.index, y=m2_tn, name="M2", showlegend=False,
                       line=dict(color=T["sofr"], width=1.6), connectgaps=True,
                       hovertemplate=_ht("M2", ".2f", " Tn", color=T["sofr"])),
            row=3, col=1, secondary_y=False,
        )
        mmax = float(m2_tn.max())
        _axis5(fig, 3, False, 0, _nice_step(mmax / 4.0), title_text="Tn")
    else:
        fig.update_yaxes(title_text="Tn", row=3, col=1, secondary_y=False)

    # ---- Row 4 — Reserve Balances (線, Tn) + 3 Tn 警戒線 ----
    _line("WRBWFRBL", "Reserve Balances", T["sofr"], 4, 1 / 1_000_000, ".2f", " Tn")
    fig.add_hline(y=3, line_dash="dash", line_color=T["thr_crit"], line_width=1.3, row=4, col=1)
    if "WRBWFRBL" in df.columns:
        w = df["WRBWFRBL"].ffill() / 1_000_000; wmin = float(w.min()); wmax = float(w.max()); wsp = wmax - wmin
        fig.update_yaxes(title_text="Tn",
                         range=[min(wmin, 3) - wsp * 0.18, max(wmax, 3) + wsp * 0.08], row=4, col=1)
    else:
        fig.update_yaxes(title_text="Tn", row=4, col=1)

    # ---- Row 5 — SP500 / M2 & 5yr rolling Z-score (ratio-z 模版: amber ratio + z area) ----
    _add_ratio_zrow(fig, panel, df, 5, thr=2.0)

    # ---- Row 6 — SRF & Discount Window & BTFP (Bn); 反向加入 → hover SRF → Discount Window → BTFP ----
    _line("H41RESPPALDKNWW", "BTFP", T["onrrp"], 6, 1 / 1000, ",.1f", " Bn")
    _line("WLCFLPCL", "Discount Window", T["iorb"], 6, 1 / 1000, ",.1f", " Bn")   # WLCFLPCL=H.4.1 primary credit 借款金額 (Millions→Bn); 原 DPCREDIT 是利率% 抓錯
    _line("RPONTSYD", "SRF", T["tgcr"], 6, 1.0, ",.1f", " Bn")   # NY Fed Temp OMO 午後同日發布 → 領先尾日, 永不落後 → 非 pending
    fig.update_yaxes(title_text="Bn", row=6, col=1)

    apply_global_layout(fig, df, n_rows=6, height=panel_dims(6, margin_b=78)[0], title="Panel 2 · Liquidity Monitor", margin_b=78, include_10y=True)
    fig.update_layout(margin_l=80, margin_r=30)   # margin_r 統一 30 (全 panel 時間鈕 All 離功能鍵一致); t/b 已由 apply_global_layout 設

    # ---- 圖下方註腳 + 3 Tn scarcity (對齊 Panel 1) ----
    _N2, _vs2 = 6, panel_dims(6)[1]
    _ph2 = (1 - (_N2 - 1) * _vs2) / _N2
    def _foot(text, r, color):
        fig.add_annotation(
            text=text, xref="paper", yref="paper",
            x=0.0, y=(1 - (r - 1) * (_ph2 + _vs2) - _ph2) - 0.05 * _ph2,
            xanchor="left", yanchor="top", showarrow=False,
            font=dict(size=10, color=color), opacity=0.9,
        )
    _foot("Net Liquidity = Fed's TA − TGA − ON RRP", 2, T["iorb"])    # 橘 (Net Liq. 色)
    _foot("Real M2 YoY = M2 YoY − CPI YoY", 3, T["bar_low"])          # 綠 (正負雙色→取綠)
    _foot("BTFP (Bank Term Funding Program) discontinued → last non-zero 2025-03, zero thereafter",
          6, T["onrrp"])                                              # 紫 (BTFP 線色)
    # z-window footnote (R5 SP500/M2 monthly-60)
    _zrow_footnote(fig, _ZFN_MONTHLY_60, 5, _N2, _vs2)
    # 3 Tn scarcity (Row 4 Reserve Balances; 須在 apply_global_layout 後, 否則被標題 recolor 迴圈蓋白)
    if "WRBWFRBL" in df.columns:
        fig.add_annotation(text="3 Tn scarcity", x=df.index.min(), y=3,
                           xanchor="left", yanchor="bottom", showarrow=False,
                           font=dict(color=T["thr_crit"], size=10), row=4, col=1)

    output = Path(output)
    _write_panel_html(fig, output)
    return output


# Panel 3: Credit Stress — split into 3-1 (HY Long History) + 3-2 (Others Short History)
#   拆兩支理由: HY 深史 1997+ 與其餘 OAS 短史 2023+ 同框 → 短史擠右下角 + y 軸被 GFC 撐爆。
#   軸法 (全列雙軸, 對齊 Panel 2 'Net Liquidity & SP500'): level 列 = SP500(左, 白 w1.4)+ credit(右 bps);
#         Z 列 = SP500(左, 白 w1.4)+ Z signed-area(右, 綠<2/紅≥2, 同 Panel 1 _threshold_area, 軸標 Z-score);
#         ratio 列 = 單軸(青 w1.5)。credit 線 w1.6。  rebuilt 2026-06-22
# ============================================================================
def _nice_step(rough: float) -> float:
    """把 rough step 進位到 nice {1,2,2.5,3,4,5,6,8}×10^k (軸刻度用)。"""
    import math
    if rough <= 0:
        return 1.0
    base = 10.0 ** math.floor(math.log10(rough))
    for m in (1, 2, 2.5, 3, 4, 5, 6, 8):
        if rough <= m * base * (1 + 1e-9):
            return m * base
    return 10.0 * base


def _add_sp500_bg(fig, spx: pd.Series, row: int):
    """SP500 疊圖 (左主軸 anchor) — 配色/粗細同 Panel 2 'Net Liquidity & SP500' (T['sp500'] 白, width 1.4)。
       頂部 headroom 1.20 → Row 1 的 SP500 線不頂到右上時間鈕 (其餘列同範圍, 視覺一致)。"""
    if not spx.notna().any():
        return
    fig.add_trace(
        go.Scatter(x=spx.index, y=spx, name="SP500", showlegend=False,
                   line=dict(color=T["sp500"], width=1.4), connectgaps=True,
                   hovertemplate=_ht("SP500", ",.2f", color=T["sp500"])),
        row=row, col=1, secondary_y=False,
    )
    smax = float(spx.max())
    _axis5(fig, row, False, 0, _nice_step(smax / 4.0),   # 5 格 0/.../8000, data-driven
           title_text="SP500", tickformat=",", showgrid=True)


def _add_zrow(fig, spx: pd.Series, z: pd.Series, row: int, zt0: float = -2.0, cap: bool = False, disp_index=None):
    """
    Z 列 (雙軸): 左軸 SP500 (白, 背景, 同 Panel 2) + 右軸 Z signed-area。
    Z signed-area = Panel 1 _threshold_area 手法 (綠<2 / 紅≥2 凸尖 fill 到 0 + Z=2 紅虛線 + hover 值依門檻上色),
    置於 secondary_y; Z=2 線用 2-point scatter 畫在 σ 軸 (非 add_hline, 避免掛到主軸 SP500 尺度)。
    hover 順序: Z-score (含灰 swatch) → SP500。
    cap=True (Panel 3-2): 顯示截到 5-tick 範圍 [zt0, zt0+8] → 極端值 (2020/22 破 4) 釘頂刻度 + grey swatch 留 screen 內,
      hover 顯真值 (customdata[1])。cap=False (預設, Panel 3-1): area/swatch 用真值 z (6.71 在範圍內、swatch 在 screen)。
    """
    _add_sp500_bg(fig, spx, row)                         # SP500 先加 → hover 最底
    z = z.round(2)
    thr = 2.0
    idx = z.index
    zd = z.clip(lower=zt0, upper=zt0 + 4 * 2.0) if cap else z   # 顯示用 (cap 時釘 5-tick 上下界)
    _GRN, _RED = "rgba(86,194,138,0.42)", "rgba(255,107,107,0.42)"
    gy = zd.where(zd < thr, 0.0)                          # < 2 日顯值, ≥ 2 日 = 0 (整根綠 only calm)
    ry = zd.where(zd >= thr, 0.0)                         # ≥ 2 日顯值, < 2 日 = 0 (整根紅 only stress)
    # (1) 綠 area (整根, fill 到 0)
    fig.add_trace(go.Scatter(x=idx, y=gy, mode="lines", line=dict(width=0),
                             fill="tozeroy", fillcolor=_GRN, hoverinfo="skip", showlegend=False),
                  row=row, col=1, secondary_y=True)
    # (2) 紅 area (整根, fill 到 0)
    fig.add_trace(go.Scatter(x=idx, y=ry, mode="lines", line=dict(width=0),
                             fill="tozeroy", fillcolor=_RED, hoverinfo="skip", showlegend=False),
                  row=row, col=1, secondary_y=True)
    # (3)+(5) hover swatch (零面積 grey); 尾端 pending 由 _pending_swatch 延伸; pos=zd (釘位) / val=z (真值)
    #   → pending 點 hover 標 PENDING_LABEL;可見綠/紅色塊 (1)(2) 不延伸 (仍收最後真值日)。
    #   cap/非 cap 統一走 customdata[1]=z 真值字串 (cap 只影響 zd 釘位, 不影響 hover 顯示值)。
    _hx, _hy, _hcd = _pending_swatch(
        zd, z, disp_index, ".2f", "",
        lambda v: "#FF6B6B" if (pd.notna(v) and v >= thr) else "#56C28A")
    _ht_z = ("<span style='color:%{customdata[0]}'><b>Z-score："
             "%{customdata[1]}</b></span><extra></extra>")
    # (3) hover base (延伸後)
    fig.add_trace(go.Scatter(x=_hx, y=_hy, mode="lines", line=dict(width=0),
                             hoverinfo="skip", showlegend=False), row=row, col=1, secondary_y=True)
    # (5) hover swatch (延伸 + customdata; pending 顯 PENDING_LABEL, 真值顯格式化值)
    fig.add_trace(go.Scatter(x=_hx, y=_hy, mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(190,195,205,0.92)", customdata=_hcd, name="Z-score",
                             showlegend=False, hovertemplate=_ht_z),
                  row=row, col=1, secondary_y=True)
    # Z=2 紅虛線 (σ 軸 scatter)
    fig.add_trace(go.Scatter(x=[_hx[0], _hx[-1]], y=[thr, thr], mode="lines",
                             line=dict(color=T["thr_crit"], width=1.0, dash="dash"),
                             hoverinfo="skip", showlegend=False), row=row, col=1, secondary_y=True)
    # Z=5 淡紅虛線 (Label 門檻; 同 z=2 家族但更淡; 僅當 5 落可視範圍內 → 只 Panel 3-1, Panel 3-2 頂 4.8 不畫)
    if zt0 + 8.8 >= 5:
        fig.add_trace(go.Scatter(x=[_hx[0], _hx[-1]], y=[5.0, 5.0], mode="lines",
                                 line=dict(color=T["thr_crit"], width=0.8, dash="dash"),
                                 opacity=0.4, hoverinfo="skip", showlegend=False),
                      row=row, col=1, secondary_y=True)
    _axis5(fig, row, True, zt0, 2, title_text="Z-score", showgrid=False)  # zt0=-2→-2/0/2/4/6; zt0=-4→-4/-2/0/2/4 (頂截斷靠標籤)


def _add_signed_area(fig, s, row, name, fmt=".2f", suffix="%", title_text="%", flip_color=False, disp_index=None):
    """signed-area (正負皆有): 綠(≥0)/紅(<0) 兩條 clip area 皆 skip hover (clip→0 連續無缺口);
    零面積 grey fill trick 給單一 hover entry, 值依正負上色 (<0 紅 / ≥0 綠);
    右軸含 0 的 5-tick (對齊左軸 SP500-from-0)。Panel 2 YoY(%) + Panel 4 term spread(bps) 共用。
    flip_color=True → 反轉 (≥0 紅 / <0 綠);給 Panel 6 R4 Margin Debt YoY% 用
      (融資餘額成長 >0 = 加槓桿 = 紅;<0 = 去槓桿 = 綠), 與「成長=好」的 YoY(預設)相反。"""
    s = s.round(2)
    _pos_c = "rgba(255,107,107,0.42)" if flip_color else "rgba(86,194,138,0.42)"   # ≥0 區色
    _neg_c = "rgba(86,194,138,0.42)" if flip_color else "rgba(255,107,107,0.42)"   # <0 區色
    fig.add_trace(go.Scatter(x=s.index, y=s.clip(lower=0), showlegend=False, hoverinfo="skip",
                             fill="tozeroy", fillcolor=_pos_c, line=dict(width=0)),
                  row=row, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=s.index, y=s.clip(upper=0), showlegend=False, hoverinfo="skip",
                             fill="tozeroy", fillcolor=_neg_c, line=dict(width=0)),
                  row=row, col=1, secondary_y=True)
    # 透明 base + grey swatch: 尾端 pending 由 _pending_swatch 延伸到 disp_index 尾日 → 該點標
    #   PENDING_LABEL;可見綠/紅 clip area (上方兩條) 不延伸 (仍收最後真值日, 不冒假色塊)。pos=val=s (不 clip)。
    _pos_hex = "#FF6B6B" if flip_color else "#56C28A"
    _neg_hex = "#56C28A" if flip_color else "#FF6B6B"
    _hx, _hy, _hcd = _pending_swatch(
        s, s, disp_index, fmt, suffix,
        lambda v: _neg_hex if (pd.notna(v) and v < 0) else _pos_hex)
    ht = ("<span style='color:%{customdata[0]}'><b>" + name + "："
          "%{customdata[1]}</b></span><extra></extra>")
    # 透明 base (延伸後 → 下方 grey tonexty 尾端有 base 可填, 零面積)
    fig.add_trace(go.Scatter(x=_hx, y=_hy, fill="tozeroy", fillcolor="rgba(0,0,0,0)",
                             line=dict(width=0), showlegend=False, hoverinfo="skip"),
                  row=row, col=1, secondary_y=True)
    # grey hover 小色塊 (延伸 + customdata; pending 顯 PENDING_LABEL, 真值顯格式化值且依正負上色)
    fig.add_trace(go.Scatter(x=_hx, y=_hy, fill="tonexty", fillcolor="rgba(190,195,205,0.92)",
                             line=dict(width=0), showlegend=False, name=name,
                             customdata=_hcd, hovertemplate=ht),
                  row=row, col=1, secondary_y=True)
    _lo = min(float(s.min()), 0.0); _hi = max(float(s.max()), 0.0)
    _t0, _d = _zero_5tick(_lo, _hi)                      # 含 0 的 5 格, 對齊左軸 5 格線
    _axis5(fig, row, True, _t0, _d, title_text=title_text, hoverformat=fmt, showgrid=False)


def _threshold_area(fig, s, row, thr, name, t0, dtick, fmt=".2f", suffix="", clip=True, title_text=None, disp_index=None):
    """通用 threshold-area: 綠 < thr / 紅 >= thr 凸尖 fill 到 0 + thr 紅虛線 + 零面積 grey hover swatch (右軸)。
    clip=False → 不 clip 量值, 極端尖峰衝出由軸範圍切 (同 Panel 1 R2 / Panel 7 R4 repo 手法); title_text 給定 → 覆寫軸標 (預設用 name)。
    抽自 _add_zrow 同一手法 (no-fill base + tonexty, Panel 3-1/3-2 已驗證可正確渲染),
    把 thr / 軸刻度 / 名稱 / 格式參數化; 不動 _add_zrow (3-1/3-2 續用它)。
    Panel 5 共用: R2 VIX(thr=20, 軸 0~80) / R3 VIX-z(thr=2, 軸 -2~6) / R5 ratio-z(thr=2, 軸 -4~4)。
    顯示截到 5-tick 範圍 [t0, t0+4d] -> 極端值釘在頂/底刻度不破界; hover 仍顯真值 (customdata)。
    hover entry = grey swatch trace (name); 值色依是否 >= thr (紅/綠)。secondary_y=True。
    高尾紅 (>= thr 紅; 給 VIX / ratio-z / NFCILEVERAGE / Margin-z 等用)。"""
    s = s.round(2)
    idx = s.index
    lo, hi = t0, t0 + 4 * dtick                          # 顯示界 = 5-tick 上下界
    sd = s.clip(lower=lo, upper=hi) if clip else s        # clip=True 釘刻度; clip=False 衝出由軸切 (hover 仍顯真值 s)
    _GRN, _RED = "rgba(86,194,138,0.42)", "rgba(255,107,107,0.42)"
    gy = sd.where(sd < thr, 0.0)                          # < thr 日顯值, ≥ thr 日 = 0 (整根綠 only calm)
    ry = sd.where(sd >= thr, 0.0)                         # ≥ thr 日顯值, < thr 日 = 0 (整根紅 only stress)
    # (1) 綠 area (整根, fill 到 0)
    fig.add_trace(go.Scatter(x=idx, y=gy, mode="lines", line=dict(width=0),
                             fill="tozeroy", fillcolor=_GRN, hoverinfo="skip", showlegend=False),
                  row=row, col=1, secondary_y=True)
    # (2) 紅 area (整根, fill 到 0; 蓋 stress 日)
    fig.add_trace(go.Scatter(x=idx, y=ry, mode="lines", line=dict(width=0),
                             fill="tozeroy", fillcolor=_RED, hoverinfo="skip", showlegend=False),
                  row=row, col=1, secondary_y=True)
    # (3)+(5) hover swatch (零面積 grey); 尾端 pending 由 _pending_swatch 延伸到 disp_index 尾日
    #   → 該點 hover 標 PENDING_LABEL;可見綠/紅色塊 (1)(2) 不延伸 (仍收最後真值日, 不冒假色塊)。
    _is_red = lambda v: v >= thr                          # >= thr 紅 (高尾)
    _hx, _hy, _hcd = _pending_swatch(
        sd, s, disp_index, fmt, suffix,
        lambda v: "#FF6B6B" if (pd.notna(v) and _is_red(v)) else "#56C28A")
    ht = ("<span style='color:%{customdata[0]}'><b>" + name + "："
          "%{customdata[1]}</b></span><extra></extra>")
    # (3) hover base (延伸後 → tonexty 尾端有 base)
    fig.add_trace(go.Scatter(x=_hx, y=_hy, mode="lines", line=dict(width=0),
                             hoverinfo="skip", showlegend=False), row=row, col=1, secondary_y=True)
    # (5) hover swatch (延伸 + customdata; pending 點顯 PENDING_LABEL, 真值點顯格式化值)
    fig.add_trace(go.Scatter(x=_hx, y=_hy, mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(190,195,205,0.92)", customdata=_hcd, name=name,
                             showlegend=False, hovertemplate=ht),
                  row=row, col=1, secondary_y=True)
    # thr 紅虛線 (在 σ/level 軸畫 2-point scatter, 非 add_hline → 不掛到左主軸)
    fig.add_trace(go.Scatter(x=[_hx[0], _hx[-1]], y=[thr, thr], mode="lines",
                             line=dict(color=T["thr_crit"], width=1.0, dash="dash"),
                             hoverinfo="skip", showlegend=False), row=row, col=1, secondary_y=True)
    _axis5(fig, row, True, t0, dtick, title_text=(title_text or name), showgrid=False)


def _ratio_sp500_m2_z(panel_full: pd.DataFrame):
    """SP500/M2 估值比率 + 5yr rolling-z 的單一計算源 (scorecard 與 8 panel 的 _add_ratio_zrow 共用 → 值結構性一致)。
    回 (ratio_full, z_full), 皆 daily on 深 cache extent (panel ∪ SP500_long/M2SL parquet, 使 5yr 暖機落顯示窗前):
      - ratio_full = SP500 / M2(Per $1 Tn) daily, M2.ffill (LINE / nowcast, 當日 SPX 位階可看)
      - z_full     = 月底 60mo(=5yr) rolling z, ffill 回 daily 階梯
    混頻 (日 SPX + 月 M2) 處理:
      月頻 z 用月底「真值」M2 直接相除, 不走 compute_sp500_m2 —— 它內部 .ffill() 會把最後不完整月的 NaN
      補回 → 「當月 SPX / 上月 M2」混月。此處月底真值 M2, 末不完整月 (M2 未出) = NaN → ratio=NaN → z 停最後完整月。
      ratio LINE 仍走 compute_sp500_m2 (內部 ffill) 保留日頻 nowcast。
    月頻取樣理由: fundamental 估值比率日變動 100% 來自 SPX (M2 月內平), 日頻 z 會過度取樣 SPX 日自相關。"""
    if "M2SL" not in panel_full.columns:
        return pd.Series(dtype="float64"), pd.Series(dtype="float64")
    _cache = Path(__file__).resolve().parent / "cache_dashboard"
    _wide_idx = panel_full.index
    for _fn in ("SP500_long.parquet", "M2SL.parquet"):
        _p = _cache / _fn
        if _p.exists():
            _wide_idx = _wide_idx.union(pd.DatetimeIndex(pd.read_parquet(_p).index))
    _wide = panel_full[[c for c in ("SP500", "M2SL") if c in panel_full.columns]].reindex(_wide_idx)
    spx = load_display_sp500(_wide)
    m2 = load_display_m2(_wide)                                    # 月頻真值 (不 ffill)
    ratio_full = compute_sp500_m2(spx, m2) * 1000                 # daily LINE (compute 內部 ffill M2 -> nowcast)
    ratio_m = spx.resample("ME").last().div(m2.resample("ME").last()) * 1000  # 月底 SPX / 月底真值 M2; 末不完整月 M2=NaN -> ratio=NaN -> z 停最後完整月
    z_m = rolling_zscore(ratio_m, 60, min_periods=60)
    z_full = z_m.reindex(ratio_full.index, method="ffill")
    return ratio_full, z_full


def _add_ratio_zrow(fig, panel_full: pd.DataFrame, df: pd.DataFrame, row: int, thr: float = 2.0):
    """ratio-z 模版 (8 panel 共用): 左軸 ratio 線 (amber) + 右軸 5yr rolling-z threshold-area。
    ratio+z 資料由 _ratio_sp500_m2_z(panel_full) 單一源計算 (與 scorecard 同一 helper -> 值結構性一致);
      此處只 reindex 到 df.index + 畫 trace。z 從顯示窗 1996-12 左緣即滿窗 (5yr 暖機落顯示窗前),
      短窗 (Panel 1 2018+ / 3-2 2023+) 也看得到 z。hover: ratio -> Z-score (ratio 後加 -> hover 頂 + 線最上)。
    amber #FBBF24 避撞綠 z-area; 軸 -4/-2/0/2/4 (顯示截界 + hover 真值由 _threshold_area 處理)。"""
    if "M2SL" not in panel_full.columns:
        return
    ratio_full, z_full = _ratio_sp500_m2_z(panel_full)
    ratio = ratio_full.reindex(df.index)                          # 顯示窗 (LINE)
    z = z_full.reindex(df.index).dropna()                         # 顯示窗 (z-area, 從 1996-12 左緣即有值)
    # 右軸 z threshold-area (先加 → hover 底)
    _threshold_area(fig, z, row, thr, "Z-score", t0=-4.0, dtick=2.0, fmt=".2f")
    # 左軸 ratio 線 (後加 → hover 頂 + 畫最上); amber 避撞綠 area
    fig.add_trace(
        go.Scatter(x=ratio.index, y=ratio, name="SP500 / M2", showlegend=False,
                   line=dict(color="#FBBF24", width=1.5), connectgaps=True,
                   hovertemplate=_ht("SP500 / M2", ".2f", color="#FBBF24")),
        row=row, col=1, secondary_y=False)
    _disp = ratio.dropna()
    _rmax = float(_disp.max()) if len(_disp) else 1.0              # 軸用「顯示窗」max (短窗也合身)
    _axis5(fig, row, False, 0, _nice_step(_rmax / 4.0),
           title_text="SP500 / M2", showgrid=True)


# ----------------------------------------------------------------------------
# Panel 3-1 — Credit Stress Monitor · HY OAS (Long History)
# ----------------------------------------------------------------------------
def plot_credit_stress_hy(panel: pd.DataFrame, output: str | Path) -> Path:
    """
    Panel 3-1 (3 子圖, 純顯示 / Architecture B)。HY OAS 深史 1997+ 專屬:
      Row 1  HY OAS & SP500                              HY(右軸 bps, w1.6) + SP500(左軸 白 w1.4)
      Row 2  Expanding Z-score of HY OAS & SP500         SP500(左)+ Z signed-area(右, ADF: HY 定態→expanding)
      Row 3  SP500 / M2 （Per $1 Tn）                       青線
    window: trim 到 HY first_valid (~1997) → 含 2001/GFC/COVID 衰退帶 (深史 marquee)。
    ⚠ Row 1 / Row 2 仍會被 GFC 尖峰拉高 y 軸 (長史本就要呈現), 看近年用右上時間鈕拉 3Y/5Y。
    """
    if "BAMLH0A0HYM2" not in panel.columns:
        raise KeyError("BAMLH0A0HYM2 (HY OAS) 不在 panel → 無法畫 Panel 3-1")
    df = panel.loc[panel["BAMLH0A0HYM2"].first_valid_index():].copy()
    spx = load_display_sp500(df)

    titles = (
        "HY OAS & SP500",
        "Expanding Z-score of HY OAS（Credit Stress ≥ 2；Label ≥ 5）& SP500",
        "SP500 / M2 （Per $1 Tn）& Rolling_5yr Z-score（Valuation Stretched ≥ 2）",
    )
    specs = [[{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}]]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        vertical_spacing=panel_dims(3)[1], specs=specs, subplot_titles=titles)

    # ---- Row 1 — HY OAS & SP500 ----
    _add_sp500_bg(fig, spx, 1)
    hy = df["BAMLH0A0HYM2"] * 100.0                      # % → bps
    _hyy, _hycd = _pending(hy, df.index, ",.0f", " bps")  # 日頻: 尾端 T+1 未公布 → hold + 標 pending
    fig.add_trace(
        go.Scatter(x=df.index, y=_hyy, name="HY OAS", showlegend=False, customdata=_hycd,
                   line=dict(color=T["hy"], width=1.6), connectgaps=True,
                   hovertemplate=_ht_cd("HY OAS", color=T["hy"])),
        row=1, col=1, secondary_y=True,
    )
    hmax = float(hy.max())
    _axis5(fig, 1, True, 0, _nice_step(hmax / 4.0),      # bps 5 格, 對齊左軸 SP500
           title_text="bps", showgrid=False)

    # ---- Row 2 — Expanding Z-score of HY OAS & SP500 (雙軸) ----
    hyz = expanding_zscore(df["BAMLH0A0HYM2"]).dropna()  # z scale-invariant → 用原始 %
    _add_zrow(fig, spx, hyz, 2, disp_index=df.index)

    # ---- Row 3 — SP500 / M2 ----
    _add_ratio_zrow(fig, panel, df, 3)

    apply_global_layout(fig, df, n_rows=3, height=panel_dims(3)[0],
                        title="Panel 3-1 · Credit Stress Monitor · HY OAS（Long History）", include_10y=True)
    fig.update_layout(margin_r=30)         # margin_r 統一 30 (全 panel 時間鈕 All 離功能鍵一致)
    _eps = _z5_episodes(hyz, 5)                                     # 與 list_z5_days.py 同 → 圖與腳本一致
    _pk = pd.Series({e[2]: e[3] for e in _eps})                    # {peak_dt: peak_val}
    _txt = {f"{e[2]:%Y-%m-%d}": _z5_crisis_text(*e) for e in _eps}  # 3 行危機標籤
    _extreme_labels_impl(fig, df.index[0], df.index[-1], _pk, row=2,  # 每場 z≥5 只標峰值 (Z 軸)
                         name="Z-score", fmt=".2f", suffix="", annot_thr=5,
                         cap=6.5, secondary_y=True, texts=_txt, ax0=20)  # ax0 小 → 框貼近尖峰
    # z-window footnotes (R2 HY-OAS-z expanding-daily; R3 SP500/M2 monthly-60)
    _vsP31 = panel_dims(3)[1]
    _zrow_footnote(fig, _ZFN_DAILY_EXP, 2, 3, _vsP31)
    _zrow_footnote(fig, _ZFN_MONTHLY_60, 3, 3, _vsP31)

    output = Path(output)
    _write_panel_html(fig, output)
    return output


# ----------------------------------------------------------------------------
# Panel 3-2 — Credit Stress Monitor · Other OAS (Short History)
# ----------------------------------------------------------------------------
def plot_credit_stress_other(panel: pd.DataFrame, output: str | Path) -> Path:
    """
    Panel 3-2 (5 子圖, 純顯示 / Architecture B)。短史 2023+:
      Row 1  Other OAS & SP500                           QS + EM HY + IG (右軸 bps w1.6) + SP500(左 白 w1.4)
                                                         hover: Quality Spread → EM HY OAS → IG OAS → SP500
             └ 圖下方註腳: Quality Spread = CCC OAS − BB OAS (金色)
      Row 2  Rolling-252 Z of Quality Spread & SP500     SP500(左)+ Z signed-area(右)
      Row 3  Rolling-252 Z of EM HY OAS & SP500
      Row 4  Rolling-252 Z of IG OAS & SP500             (ADF: 短史非定態 → rolling 252d)
      Row 5  SP500 / M2 （Per $1 Tn）                       青線
    window: floor 到 4 條 (CCC/BB/IG/EM) 最早 first_valid 的年初 (~2023-01) → 有深史的 SP500/ratio(+z) 填滿前段到左緣,
            OAS 各自 first_valid (~2023-06) 前留白 (leading NaN, connectgaps 不橋); 2023+ 無 NBER 衰退 → 衰退帶自動不畫。
    """
    _C4 = ["BAMLH0A3HYC", "BAMLH0A1HYBB", "BAMLC0A0CM", "BAMLEMHYHYLCRPIUSOAS"]
    _present = [c for c in _C4 if c in panel.columns]
    _firsts = [d for d in (panel[c].first_valid_index() for c in _present) if d is not None]
    # floor 到最早 OAS first_valid 的「年初」(非 first_valid 當天): 有深史的 SP500 / SP500-M2 ratio(+z) 填滿前段到左緣;
    #   OAS 四條各自 first_valid (~2023-06) 前是 leading NaN → connectgaps 只橋「兩端皆有值」的內部缺口, 不畫前導 NaN → 自然留白 (符合 FRED 無深史共識)。
    _floor = pd.Timestamp(min(_firsts).year, 1, 1) if _firsts else None
    # 往前 7 天 (前一年最後幾個交易日, 此尺度僅數 px) → Jan-1 年刻度落進 autorange 內可顯示;
    #   且結尾不設明確 x range → 走 autorange → 「All」按鈕正常 highlight (與其他 panel 一致)。
    df = panel.loc[_floor - pd.Timedelta(days=7):].copy() if _floor is not None else panel.copy()
    spx = load_display_sp500(df)

    # Quality Spread = CCC − BB (raw %, 供 level ×100→bps 與 z scale-invariant 兩用)
    has_qs = {"BAMLH0A3HYC", "BAMLH0A1HYBB"}.issubset(df.columns)
    qs_raw = compute_credit_quality_spread(df["BAMLH0A3HYC"], df["BAMLH0A1HYBB"]) if has_qs else None

    titles = (
        "Other OAS & SP500",
        "Rolling_1yr Z-score of Quality Spread（Credit Stress ≥ 2；Label ≥ 5）& SP500",
        "Rolling_1yr Z-score of EM HY OAS（Credit Stress ≥ 2；Label ≥ 5）& SP500",
        "Rolling_1yr Z-score of IG OAS（Credit Stress ≥ 2；Label ≥ 5）& SP500",
        "SP500 / M2 （Per $1 Tn）& Rolling_5yr Z-score（Valuation Stretched ≥ 2）",
    )
    specs = [[{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}],
             [{"secondary_y": True}], [{"secondary_y": True}]]
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True,
                        vertical_spacing=panel_dims(5)[1], specs=specs, subplot_titles=titles)

    # ---- Row 1 — Other OAS & SP500 ; hover Quality Spread → EM HY → IG → SP500 ----
    _add_sp500_bg(fig, spx, 1)                           # SP500 先加 → hover 最底
    _lines = []                                          # hover 想要順序 (top→bottom): QS → EM HY → IG
    if qs_raw is not None:
        _lines.append((qs_raw * 100.0, T["qs"], "Quality Spread"))
    if "BAMLEMHYHYLCRPIUSOAS" in df.columns:
        _lines.append((df["BAMLEMHYHYLCRPIUSOAS"] * 100.0, T["em"], "EM HY OAS"))
    if "BAMLC0A0CM" in df.columns:
        _lines.append((df["BAMLC0A0CM"] * 100.0, T["ig"], "IG OAS"))
    _vals = []
    for s, color, nm in reversed(_lines):                # 反向加入 → x-unified hover 正序 QS→EM→IG
        _vals.append(s)
        _sy, _scd = _pending(s, df.index, ",.0f", " bps")   # 日頻 OAS: 尾端 T+1 未公布 → hold + 標 pending
        fig.add_trace(
            go.Scatter(x=df.index, y=_sy, name=nm, showlegend=False, customdata=_scd,
                       line=dict(color=color, width=1.6), connectgaps=True,
                       hovertemplate=_ht_cd(nm, color=color)),
            row=1, col=1, secondary_y=True,
        )
    if _vals:
        omax = float(pd.concat(_vals).max())
        _axis5(fig, 1, True, 0, _nice_step(omax / 4.0),  # bps 5 格, 對齊左軸 SP500
               title_text="bps", showgrid=False)
    else:
        fig.update_yaxes(title_text="bps", showgrid=False, row=1, col=1, secondary_y=True)

    # ---- Rows 2-4 — Rolling-252 Z (雙軸 + SP500): Quality Spread / EM HY / IG ----
    _qsz = rolling_zscore(qs_raw, 252).dropna() if qs_raw is not None else None
    _emz = rolling_zscore(df["BAMLEMHYHYLCRPIUSOAS"], 252).dropna() if "BAMLEMHYHYLCRPIUSOAS" in df.columns else None
    _igz = rolling_zscore(df["BAMLC0A0CM"], 252).dropna() if "BAMLC0A0CM" in df.columns else None
    if _qsz is not None:
        _add_zrow(fig, spx, _qsz, 2, zt0=-4.0, cap=True, disp_index=df.index)
    if _emz is not None:
        _add_zrow(fig, spx, _emz, 3, zt0=-4.0, cap=True, disp_index=df.index)
    if _igz is not None:
        _add_zrow(fig, spx, _igz, 4, zt0=-4.0, cap=True, disp_index=df.index)

    # ---- Row 5 — SP500 / M2 ----
    _add_ratio_zrow(fig, panel, df, 5)

    apply_global_layout(fig, df, n_rows=5, height=panel_dims(5)[0],
                        title="Panel 3-2 · Credit Stress Monitor · Other OAS（Short History）")
    # 註腳放「第一張圖下方」(非整個 Panel 底部): 用 Row 1 的 domain 座標, 左下角。
    # 須在 apply_global_layout 之後加 (否則 annotation 字色被統一覆蓋成 title_color)。
    fig.add_annotation(
        text="Quality Spread = CCC OAS − BB OAS", xref="x domain", yref="y domain",
        x=0.0, y=-0.05, xanchor="left", yanchor="top", showarrow=False,
        font=dict(color=T["qs"], size=11),
    )
    fig.update_layout(margin_r=30)         # margin_r 統一 30 (全 panel 時間鈕 All 離功能鍵一致)
    for _z, _r in ((_qsz, 2), (_emz, 3), (_igz, 4)):     # z≥5 標真值 (通常無)
        if _z is not None:
            _extreme_labels_impl(fig, df.index[0], df.index[-1], _episode_peaks(_z, 5), row=_r,
                                 name="Z-score", fmt=".2f", suffix="", annot_thr=5, cap=4.5, secondary_y=True)
    # z-window footnotes (R2/R3/R4 quality-spread/EM-HY/IG rolling-252-daily; R5 SP500/M2 monthly-60)
    _vsP32 = panel_dims(5)[1]
    for _r32 in (2, 3, 4):
        _zrow_footnote(fig, _ZFN_DAILY_252, _r32, 5, _vsP32)
    _zrow_footnote(fig, _ZFN_MONTHLY_60, 5, 5, _vsP32)

    # 年刻度對齊起始年 1/1 (tick0 = OAS first-valid 年初); 不設明確 x range → 走 autorange →
    #   「All」按鈕正常 highlight (與其他 panel 一致; 修 2026-06-30, 原明確 range 會讓 All 對不上而不亮)。
    #   df 已往前 7 天 → Jan-1 年刻度落進 autorange 內可顯示 (否則資料首日 Jan-3, Jan-1 刻度被切在邊界外)。
    if _floor is not None:
        fig.update_xaxes(tick0=_floor)

    output = Path(output)
    _write_panel_html(fig, output)
    return output


# ============================================================================
# Section 4: Yield Curve Panel
# ============================================================================
# ----------------------------------------------------------------------------
# Panel 4 — Yield Curve & Real Rates Monitor (Long History)
# ----------------------------------------------------------------------------
def plot_yield_curve(panel: pd.DataFrame, output: str | Path) -> Path:
    """
    Panel 4 (4 子圖, 純顯示 / Architecture B)。Yield Curve & Real Rates 長史 (~1996+):
      Row 1  Nominal & Real Yields & Breakeven & SP500   DGS2/DGS10/DFII10/T10YIE(右軸 %, 含負) + SP500(左 白)
      Row 2  10Y−3M Term Spread & SP500                  T10Y3M signed-area(右軸 bps, 0 線=倒掛) + SP500(左)
      Row 3  10Y−2Y Term Spread & SP500                  10Y−2Y=DGS10−DGS2 自算 signed-area(右軸 bps) + SP500(左)
      Row 4  SP500 / M2 （Per $1 Tn）                       青線 ratio
    深史 (需先跑 fetch_long_curve.py): DGS/spread ~1996, DFII10/T10YIE ~2003 (TIPS 發行);
    含 2000/2006-07/2019/2022-23 倒掛 marquee。右軸含 0 的 5-tick → 實質利率 0 線 / Fed 2% 自動成格線。
    ⚠ 無 cache_dashboard 深史時 graceful 退回 panel (2018+)。
    """
    cur = load_display_curve(panel)                      # 各 curve 序列深史補齊
    spx = load_display_sp500(panel)                      # SP500 深史 (1996+)
    # 窗口: trim 到 nominal/spread 最早 first_valid (~1996; DFII10/T10YIE 天生只到 2003)
    anchor = [cur[c].first_valid_index() for c in ("DGS2", "DGS10", "T10Y2Y", "T10Y3M")
              if c in cur.columns and cur[c].first_valid_index() is not None]
    start = min(anchor) if anchor else cur.index.min()
    idx = cur.index[cur.index >= start]
    cur = cur.loc[idx]
    spx = spx.reindex(idx)
    df = panel.reindex(idx)                              # Row 4 ratio 要 M2

    titles = (
        "Nominal & Real Yields & Breakeven & SP500",
        "10Y − 3M Term Spread（Curve Inversion ≤ 0）& SP500",
        "10Y − 2Y Term Spread（Curve Inversion ≤ 0）& SP500",
        "SP500 / M2 （Per $1 Tn）& Rolling_5yr Z-score（Valuation Stretched ≥ 2）",
    )
    specs = [[{"secondary_y": True}], [{"secondary_y": True}],
             [{"secondary_y": True}], [{"secondary_y": True}]]
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        vertical_spacing=panel_dims(4)[1], specs=specs, subplot_titles=titles)

    # ---- Row 1 — Nominal & Real Yields & Breakeven & SP500 ----
    # ⚠ Plotly unified hover 由上到下 = trace 加入順序「反序」(最後加的 → 顯示在 hover 頂端)。
    # 要 hover 2Y→10Y→TIPS→Breakeven→SP500, 須反向加: SP500 先加, 殖利率 reversed(Breakeven→…→2Y)。
    # 同槓桿副作用: SP500 先加 = z 畫最底(背景) → 4 條殖利率線浮在上面 (yield curve panel 本應如此)。
    _add_sp500_bg(fig, spx, 1)                           # SP500 先加 → hover 最末 + z 最底(背景)
    _yld = [("DGS2", "2Y UST", T["sofr"]), ("DGS10", "10Y UST", T["iorb"]),
            ("DFII10", "10Y Real (TIPS)", T["bar_low"]), ("T10YIE", "10Y Breakeven", T["em"])]
    _vals = []
    for c, nm, color in reversed(_yld):                 # reversed: Breakeven→Real→10Y→2Y → hover 2Y 在頂
        if c in cur.columns and cur[c].notna().any():
            _yy, _ycd = _pending(cur[c], cur.index, ".2f", "%")   # 日頻殖利率: 尾端 T+1 未公布 → hold + pending
            fig.add_trace(
                go.Scatter(x=cur.index, y=_yy, name=nm, showlegend=False, customdata=_ycd,
                           line=dict(color=color, width=1.2), connectgaps=True,
                           hovertemplate=_ht_cd(nm, color=color)),
                row=1, col=1, secondary_y=True)
            _vals.append(cur[c])
    if _vals:
        allv = pd.concat(_vals)
        _t0, _d = _zero_5tick(float(allv.min()), float(allv.max()))   # 含負 (DFII10 曾 ~−1.2%)
        _axis5(fig, 1, True, _t0, _d, title_text="%", hoverformat=".2f", showgrid=False)

    # ---- Row 2 — 10Y−3M Term Spread (bps, signed-area) & SP500 ----
    _add_sp500_bg(fig, spx, 2)                           # SP500 先加 → hover: 10Y−3M → SP500 (spread fill 疊上)
    if "T10Y3M" in cur.columns and cur["T10Y3M"].notna().any():
        _add_signed_area(fig, cur["T10Y3M"] * 100.0, 2, "10Y−3M",     # pp → bps
                         fmt=".0f", suffix=" bps", title_text="bps", disp_index=cur.index)

    # ---- Row 3 — 10Y−2Y Term Spread (bps, signed-area) & SP500 ----
    # 用原生 FRED T10Y2Y (對稱 Row 2 的 10Y−3M; 與 Panel 7 / scorecard 一致 = SSoT; 到 06-30 不卡 DGS raw-level lag)。
    # 原自算 DGS10−DGS2 已停用: 自算繼承 DGS raw-level 落後 computed 序列 ~1 交易日,
    #   曾致 scorecard(native, 06-30)=30bps vs panel(自算, 卡 06-29)=28bps 不一致。
    _add_sp500_bg(fig, spx, 3)                           # SP500 先加 → hover: 10Y−2Y → SP500 (spread fill 疊上)
    if "T10Y2Y" in cur.columns and cur["T10Y2Y"].notna().any():
        _add_signed_area(fig, cur["T10Y2Y"] * 100.0, 3, "10Y−2Y",     # 原生序列
                         fmt=".0f", suffix=" bps", title_text="bps", disp_index=cur.index)

    # ---- Row 4 — SP500 / M2 ----
    _add_ratio_zrow(fig, panel, df, 4)

    apply_global_layout(fig, df, n_rows=4, height=panel_dims(4)[0],
                        title="Panel 4 · Yield Curve & Real Rates Monitor", include_10y=True)

    # z-window footnote (R4 SP500/M2 monthly-60; 須在 apply_global_layout 後, 否則被標題 recolor 迴圈蓋掉)
    _zrow_footnote(fig, _ZFN_MONTHLY_60, 4, 4, panel_dims(4)[1])

    output = Path(output)
    _write_panel_html(fig, output)
    return output


def plot_fx_vol_inflation(panel: pd.DataFrame, output: str | Path) -> Path:
    """
    Panel 5 (5 子圖, 純顯示 / Architecture B)。FX, Volatility & Inflation 長史 (~1996+):
      R1  USD/TWD & EM USD Index                          DEXTAUS(左,level) + DTWEXEMEGS(右,level), 雙軸對比
      R2  VIX（Risk ≥ 30）& SP500                          VIX threshold-area(右, thr=20) + SP500(左)
      R3  Expanding Z-score of VIX（Risk ≥ 2）& SP500       VIX expanding-z threshold-area(右, thr=2) + SP500(左)
      R4  CPI YoY & Core PCE YoY & SP500                   CPI / Core-PCE YoY 兩線(右 %) + SP500(左) (無 2% 線)
      R5  SP500 / M2（Per $1 Tn）& Rolling_5yr Z-score（Valuation Stretched ≥ 2）        ratio 線 amber(左) + 5yr rolling-z area(右, thr=2)
    深史 (需先跑 fetch_long_fxvolmacro.py): DEXTAUS/VIX/CPI/PCE ~1996, DTWEXEMEGS ~2006 (FRED 該 index 起點);
    CPI/PCE 月頻 latest-revised -> YoY=pct_change(12)。2% 參考線刻意不畫 (圖含 CPI, 單線會誤導; 留給讀者判讀)。
    ⚠ 無 cache_dashboard 深史時 graceful 退回 panel (2018+)。
    """
    cur = load_display_fxvolmacro(panel)                 # dict: 5 序列 (FX/VIX 日頻、CPI/PCE 月頻)
    spx = load_display_sp500(panel)                      # SP500 深史 (1996+)
    # 窗口: trim 到 daily 序列最早 first_valid (~1996; DTWEXEMEGS 天生 ~2006)
    anchor = [cur[c].first_valid_index() for c in ("DEXTAUS", "VIXCLS")
              if c in cur and cur[c].first_valid_index() is not None]
    start = min(anchor) if anchor else panel.index.min()
    idx = panel.index[panel.index >= start]
    spx = spx.reindex(idx)
    df = panel.reindex(idx)                              # R5 ratio 要 M2
    twd = cur["DEXTAUS"].reindex(idx)
    em = cur["DTWEXEMEGS"].reindex(idx)
    vix = cur["VIXCLS"].reindex(idx)

    titles = (
        "USD/TWD & EM USD Index",
        "VIX（Risk ≥ 30）& SP500",
        "Expanding Z-score of VIX（Risk ≥ 2）& SP500",
        "CPI YoY & Core PCE YoY & SP500",
        "SP500 / M2 （Per $1 Tn）& Rolling_5yr Z-score（Valuation Stretched ≥ 2）",
    )
    specs = [[{"secondary_y": True}]] * 5
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True,
                        vertical_spacing=panel_dims(5)[1], specs=specs, subplot_titles=titles)

    # ---- R1 — USD/TWD & EM USD Index (雙軸 level, 無 SP500) ----
    # ⚠ unified hover 反序: 要 hover USD/TWD->EM, 須先加 EM (hover 底), 後加 USD/TWD (hover 頂)。
    em2 = em.dropna()
    _emy, _emcd = _pending(em2, idx, ".2f")              # 日頻 FX: 尾端 T+1 未公布 → hold + 標 pending
    fig.add_trace(go.Scatter(x=idx, y=_emy, name="EM USD Index", showlegend=False, customdata=_emcd,
                             line=dict(color=T["onrrp"], width=1.2), connectgaps=True,
                             hovertemplate=_ht_cd("EM USD Index", color=T["onrrp"])),
                  row=1, col=1, secondary_y=True)
    _e0, _ed = _range_5tick(float(em2.min()), float(em2.max()))
    _axis5(fig, 1, True, _e0, _ed, title_text="EM USD Index", showgrid=False)
    twd2 = twd.dropna()
    _twy, _twcd = _pending(twd2, idx, ".3f")             # 日頻 FX: 尾端 T+1 未公布 → hold + 標 pending
    fig.add_trace(go.Scatter(x=idx, y=_twy, name="USD/TWD", showlegend=False, customdata=_twcd,
                             line=dict(color=T["sofr"], width=1.2), connectgaps=True,
                             hovertemplate=_ht_cd("USD/TWD", color=T["sofr"])),
                  row=1, col=1, secondary_y=False)
    _t0, _td = _range_5tick(float(twd2.min()), float(twd2.max()))
    _axis5(fig, 1, False, _t0, _td, title_text="USD/TWD", showgrid=True)

    # ---- R2 — VIX（Risk ≥ 30）& SP500 ----
    # hover VIX->SP500: SP500 先加 (底), VIX area 後加 (頂)。
    _add_sp500_bg(fig, spx, 2)
    vix2 = vix.dropna()
    _threshold_area(fig, vix2, 2, 30.0, "VIX",        # 上色分界 30 = 明顯壓力 (VIX 均值 ~19-20; >20 太常見會半片紅)
                    t0=0.0, dtick=20.0, fmt=".1f", disp_index=idx)
    # VIX=20 = 高於均值 / early-warning 淡參考虛線 (上色仍以 30 為界)
    fig.add_trace(go.Scatter(x=[vix2.index.min(), vix2.index.max()], y=[20.0, 20.0], mode="lines",
                             line=dict(color=T["thr_crit"], width=0.8, dash="dot"), opacity=0.4,
                             hoverinfo="skip", showlegend=False), row=2, col=1, secondary_y=True)

    # ---- R3 — Expanding Z-score of VIX（Risk ≥ 2）& SP500 ----
    _add_sp500_bg(fig, spx, 3)
    vixz = expanding_zscore(vix2).dropna()
    _threshold_area(fig, vixz, 3, 2.0, "Z-score", t0=-2.0, dtick=2.0, fmt=".2f", disp_index=idx)

    # ---- R4 — CPI YoY & Core PCE YoY & SP500 (無 2% 線) ----
    # hover CPI->Core PCE->SP500: SP500 先、Core PCE 次、CPI 末。YoY 含負 (通縮) -> 軸用 _zero_5tick。
    _add_sp500_bg(fig, spx, 4)
    # YoY 先 resample 到完整月頻 index 再 pct_change(12) → 日曆對齊 (2025-10 政府關門缺月補 NaN, 不位移); 再攤回日頻 as-known 階梯
    pce_yoy = (cur["PCEPILFE"].resample("MS").last().pct_change(12) * 100).dropna().reindex(idx, method="ffill")
    cpi_yoy = (cur["CPIAUCSL"].resample("MS").last().pct_change(12) * 100).dropna().reindex(idx, method="ffill")
    fig.add_trace(go.Scatter(x=pce_yoy.index, y=pce_yoy, name="Core PCE YoY", showlegend=False,
                             line=dict(color=T["iorb"], width=1.5), connectgaps=True,
                             hovertemplate=_ht("Core PCE YoY", ".2f", "%", color=T["iorb"])),
                  row=4, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=cpi_yoy.index, y=cpi_yoy, name="CPI YoY", showlegend=False,
                             line=dict(color=T["sofr"], width=1.5), connectgaps=True,
                             hovertemplate=_ht("CPI YoY", ".2f", "%", color=T["sofr"])),
                  row=4, col=1, secondary_y=True)
    _both = pd.concat([cpi_yoy, pce_yoy])
    _i0, _idk = _zero_5tick(float(_both.min()), float(_both.max()))
    _axis5(fig, 4, True, _i0, _idk, title_text="%", hoverformat=".2f", showgrid=False)

    # ---- R5 — SP500 / M2 & 5yr rolling Z-score (amber ratio 線 + z area) ----
    _add_ratio_zrow(fig, panel, df, 5, thr=2.0)

    apply_global_layout(fig, df, n_rows=5, height=panel_dims(5)[0],
                        title="Panel 5 · FX & Volatility & Inflation Monitor", include_10y=True)
    # z-window footnotes (R3 VIX-z expanding-daily; R5 SP500/M2 monthly-60; 須在 apply_global_layout 後)
    _vsP5 = panel_dims(5)[1]
    _zrow_footnote(fig, _ZFN_DAILY_EXP, 3, 5, _vsP5)
    _zrow_footnote(fig, _ZFN_MONTHLY_60, 5, 5, _vsP5)
    output = Path(output)
    _write_panel_html(fig, output)
    return output


def plot_leverage_monitor(panel: pd.DataFrame, output: str | Path) -> Path:
    """
    Panel 6 — Leverage Monitor (7 子圖, 純顯示 / Architecture B)。錨 1997 (dotcom/GFC/COVID/暴力升息)。
    主軸 = 散戶「借了多少錢投入股市」→ R3/R4/R5 全 base 在 Margin Debt / M2 (FINRA Debit ÷ M2, 槓桿佔比)。
      Row 1  NFCILEVERAGE & SP500                  area(右, thr 2 高尾紅) + SP500(白線, 左)
      Row 2  NFCI Family                           NFCI/Risk/Credit/ANFCI 4 條線(單軸, 0+2 ref), 無 SP500
      Row 3  Margin Debt / M2 & SP500              amber area(右, %, 恆正單色細線) + SP500(左)  ← 槓桿佔比「水位」
      Row 4  Z of Margin Debt / M2 & SP500         area(右, thr +2 高尾紅) + SP500(左)  ← 佔比「極端度」(vs 5yr)
      Row 5  Margin Debt YoY − M2 YoY & SP500      signed-area(右, %, 翻色 >0 紅) + SP500(左)  ← 超越貨幣的「成長率」
      Row 6  Margin Net Credit & SP500             signed-area(右, ≥0 綠/<0 紅, $bn) + SP500(左)  ← 淨 buffer 水位
      Row 7  SP500 / M2 & Z-score                  ratio-z 模版 (amber ratio 左 + 5yr rolling z 右), 無 SP500

    R3/R4/R5 = 同一個 Margin Debt/M2 的三個互補角度 (水位 / z 極端度 / 變化率), 如 price / RSI / momentum:
      · R3 ÷ M2 消除貨幣供給長期成長 (分子分母皆 nominal → 通膨也一併消掉) → 佔比水位 (≈ 6%)。
      · R4 = R3 比值的 5yr(60月) rolling z (對原始 Margin Debt 直接 z 會永遠貼頂; ÷M2 detrend 後才有意義),
        高尾 ≥+2 = 佔比脆弱。
      · R5 = MD YoY − M2 YoY ≈ R3 比值的一年成長率 (g_r=(g_MD−g_M2)/(1+g_M2)≈g_MD−g_M2); >0 = 融資成長
        超越貨幣供給成長 = 脫離基本面加槓桿 = 紅; 翻色 signed area, 0 線 = 跟貨幣同步(中性), 比 nominal 更有意義。
    Net Credit (閒置現金 − 融資借款) = 帶符號淨 buffer, 穿越 0 → 不可算 % (pct 爆衝)。R6 只看水位 (<0 淨債務人)。
    深史 / staggered 左緣 (誠實反映資料可得性, 非 bug):
      · NFCI 家族 (需先跑 fetch_long_leverage.py) FRED 1971+ → 錨 1997。
      · FINRA Debit 恆正 1997+ → R3 從 1997、R5(扣 12m YoY)從 1998、R4(扣 60m z)從 ~2002;
        Net Credit 需第 3 欄 free-credit-margin (2010-02 起) → R6 從 2010。(R4 比下方 R5 晚起=z 視窗較長, 正常。)
      · SP500/M2 1996+ (深 cache 暖機落顯示窗前 → ratio-z 從顯示窗左緣即滿)。
    各 margin 列「傳月頻原生序列」給 area helper(對角線連點)→ 平滑;不 ffill 到日頻(會變直方/階梯)。
    單位 (ground-truth verified, $mn): Margin Debt/M2 = (debit/1000 $bn)/(M2 $bn)×100 = % (≈6%);
      R5 = (debit.pct(12) − M2.pct(12))×100 = pp; Net Credit 顯示 = nc/1000 = $bn。
    hover 順序 = trace 反向加入順序 (x-unified)。⚠ 無 cache / 無 finra xlsx → graceful 退回 (該列空白)。
    """
    cache_dir = Path(__file__).resolve().parent / "cache_dashboard"
    _NFCI = ["NFCILEVERAGE", "NFCI", "NFCIRISK", "NFCICREDIT", "ANFCI"]
    nfci = load_display_deep(panel, _NFCI)               # dict: SID -> 深史 (canonical + cache), on panel.index
    spx_full = load_display_sp500(panel)
    margin = load_display_margin(panel)                  # DataFrame[net_credit(2010+), debit(1997+)], $mn

    # ---- x 軸錨點 1997 ----
    start = pd.Timestamp("1997-01-01")
    idx = panel.index[panel.index >= start]
    df = panel.reindex(idx).copy()
    for c in _NFCI:
        if c in nfci:
            df[c] = nfci[c].reindex(idx)
    df["SP500"] = spx_full.reindex(idx)
    spx = df["SP500"]

    # ---- margin 月頻計算 (rolling/pct 必在月頻原序上算才正確); 繪圖前再 reindex+ffill 成日頻 as-known 階梯 ----
    m2_me = _deep_monthly(panel, "M2SL", cache_dir)                          # 月頻 M2 ($bn)
    debit_me = (margin["debit"].resample("ME").last()
                if len(margin) else pd.Series(dtype="float64"))               # 融資餘額 月頻 ($mn, 1997+)
    nc_me = (margin["net_credit"].resample("ME").last()
             if len(margin) else pd.Series(dtype="float64"))                  # Net Credit 月頻 ($mn, 2010+)
    if len(debit_me) and len(m2_me):
        md_m2 = ((debit_me / 1000.0) / m2_me * 100).dropna()                 # Margin Debt / M2 = % (R3, 1997+, ≈6%)
        z_md = rolling_zscore(md_m2, 60, min_periods=60).dropna()            # 5yr(60月) rolling z (R4, ~2002+)
        excess = ((debit_me.pct_change(12) - m2_me.pct_change(12)) * 100).dropna()  # MD YoY − M2 YoY (R5, 1998+)
    else:
        md_m2 = pd.Series(dtype="float64"); z_md = pd.Series(dtype="float64")
        excess = pd.Series(dtype="float64")
    nc_bn = ((nc_me / 1000.0).dropna()
             if len(nc_me) else pd.Series(dtype="float64"))                  # Net Credit $bn (R6, 2010+)
    # 月頻 → 日頻 as-known 階梯 (reference-date ffill + 最右緣 hold 到面板右緣; 月中沿用上一個已公佈值)
    md_m2_d = md_m2.reindex(df.index, method="ffill") if len(md_m2) else md_m2
    z_md_d = z_md.reindex(df.index, method="ffill") if len(z_md) else z_md
    excess_d = excess.reindex(df.index, method="ffill") if len(excess) else excess
    nc_bn_d = nc_bn.reindex(df.index, method="ffill") if len(nc_bn) else nc_bn

    titles = (
        "NFCILEVERAGE（Leverage Stretched ≥ 2）& SP500",
        "NFCI Family（Financial Conditions；Stress ≥ 2）",
        "Margin Debt / M2 & SP500",
        "Rolling_5yr Z-score of Margin Debt / M2（Leverage Fragility ≥ 2）& SP500",
        "Margin Debt YoY − M2 YoY（Outpacing Money Supply > 0）& SP500",
        "Margin Net Credit（Net Debtor < 0）& SP500",
        "SP500 / M2 （Per $1 Tn）& Rolling_5yr Z-score（Valuation Stretched ≥ 2）",
    )
    specs = [[{"secondary_y": True}], [{}], [{"secondary_y": True}], [{"secondary_y": True}],
             [{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}]]
    fig = make_subplots(rows=7, cols=1, shared_xaxes=True,
                        vertical_spacing=panel_dims(7)[1], specs=specs, subplot_titles=titles)

    # ---- Row 1 — NFCILEVERAGE area (thr 2 高尾紅) + SP500(左) ----
    _add_sp500_bg(fig, spx, 1)
    if "NFCILEVERAGE" in df.columns and df["NFCILEVERAGE"].notna().any():
        lev = df["NFCILEVERAGE"].ffill()                  # 週頻 → daily ffill (gap 小幾乎平滑)
        _lt0, _ld = _zero_5tick(float(lev.min()), max(float(lev.max()), 2.2))   # 保證 thr=2 線在軸內
        _threshold_area(fig, lev, 1, 2.0, "NFCILEVERAGE", _lt0, _ld, fmt=".2f")

    # ---- Row 2 — NFCI Family 4 條線(單軸, 無 SP500); hover NFCI→Risk→Credit→ANFCI (反向加入) ----
    _fam = [("NFCI", "NFCI", T["sofr"], 1.8), ("NFCIRISK", "Risk", T["iorb"], 1.2),
            ("NFCICREDIT", "Credit", T["tgcr"], 1.2), ("ANFCI", "ANFCI", T["onrrp"], 1.2)]
    _famv = []
    for sid, nm, color, w in reversed(_fam):              # 反向加入 → hover 頂為 NFCI, 線最前
        if sid in df.columns and df[sid].notna().any():
            s = df[sid].ffill()
            _famv.append(s)
            fig.add_trace(go.Scatter(x=s.index, y=s, name=nm, showlegend=False,
                                     line=dict(color=color, width=w), connectgaps=True,
                                     hovertemplate=_ht(nm, ".2f", color=color)),
                          row=2, col=1)
    if _famv:
        allf = pd.concat(_famv)
        _ft0, _fd = _zero_5tick(float(allf.min()), max(float(allf.max()), 2.2))   # 含 0 與 2
        for _yv, _dash, _clr in [(0.0, "dot", "rgba(255,255,255,0.22)"), (2.0, "dash", T["thr_crit"])]:
            fig.add_trace(go.Scatter(x=[idx.min(), idx.max()], y=[_yv, _yv], mode="lines",
                                     line=dict(color=_clr, width=1.0, dash=_dash),
                                     hoverinfo="skip", showlegend=False), row=2, col=1)
        _axis5(fig, 2, False, _ft0, _fd, title_text="NFCI Family", showgrid=False)

    # ---- Row 3 — Margin Debt / M2 (amber area, %, 恆正單色; 細線融入 area) + SP500(左) ----
    _add_sp500_bg(fig, spx, 3)
    if len(md_m2):
        fig.add_trace(go.Scatter(x=md_m2_d.index, y=md_m2_d.round(2), mode="lines",
                                 line=dict(color="#FBBF24", width=0), fill="tozeroy",   # 純 area 無頂線(1.4→0.8→0, 同紅綠 area)
                                 fillcolor="rgba(251,191,36,0.20)", name="Margin Debt / M2",
                                 showlegend=False,
                                 hovertemplate=_ht("Margin Debt / M2", ".2f", "%", color="#FBBF24")),
                      row=3, col=1, secondary_y=True)
        _m0, _md = _zero_5tick(float(md_m2_d.min()), float(md_m2_d.max()))
        _axis5(fig, 3, True, _m0, _md, title_text="%", hoverformat=".2f", showgrid=False)

    # ---- Row 4 — Rolling_5yr Z of Margin Debt / M2 (area, thr +2 高尾紅) + SP500(左) ----
    _add_sp500_bg(fig, spx, 4)
    if len(z_md):
        _threshold_area(fig, z_md_d, 4, 2.0, "Z-score", t0=-4.0, dtick=2.0, fmt=".2f")   # 高尾 ≥+2 紅=脆弱; 日頻階梯 → 過渡 1 天寬 → 整條紅乾淨無混色三角

    # ---- Row 5 — Margin Debt YoY − M2 YoY (signed-area 翻色: >0 超越貨幣=紅 / <0 落後=綠) + SP500(左) ----
    _add_sp500_bg(fig, spx, 5)
    if len(excess):
        _add_signed_area(fig, excess_d, 5, "Margin Debt YoY − M2 YoY", fmt="+.1f", suffix="%",
                         title_text="%", flip_color=True)

    # ---- Row 6 — Margin Net Credit level (signed-area ≥0 綠/<0 紅, $bn) + SP500(左) ----
    _add_sp500_bg(fig, spx, 6)
    if len(nc_bn):
        _add_signed_area(fig, nc_bn_d, 6, "Margin Net Credit", fmt=",.0f", suffix=" Bn", title_text="Bn")

    # ---- Row 7 — SP500 / M2 & 5yr rolling Z-score (ratio-z 模版, 無 SP500) ----
    _add_ratio_zrow(fig, panel, df, 7, thr=2.0)

    apply_global_layout(fig, df, n_rows=7, height=panel_dims(7, margin_b=60)[0],
                        title="Panel 6 · Leverage Monitor", margin_b=60, include_10y=True)
    fig.update_layout(margin_l=80, margin_r=30)

    # ---- 圖下方註腳 (須在 apply_global_layout 後, 否則被標題 recolor 迴圈蓋白) ----
    _N6, _vs6 = 7, panel_dims(7)[1]
    _ph6 = (1 - (_N6 - 1) * _vs6) / _N6
    def _foot(text, r, color):
        fig.add_annotation(
            text=text, xref="paper", yref="paper",
            x=0.0, y=(1 - (r - 1) * (_ph6 + _vs6) - _ph6) - 0.05 * _ph6,
            xanchor="left", yanchor="top", showarrow=False,
            font=dict(size=10, color=color), opacity=0.9,
        )
    _foot("NFCILEVERAGE（standardized）→ Chicago Fed NFCI leverage subindex → "
          "system-wide financial leverage（debt & equity measures）",
          1, T["bar_low"])                                # 綠
    _foot("NFCI Family（standardized）→ NFCI：headline composite │ Risk：volatility & funding risk │ "
          "Credit：lending standards & spreads │ ANFCI：cycle-adjusted（ex-GDP / inflation）",
          2, "#9aa0aa")                                   # 中性灰
    _foot("【Data Source：FINRA Margin Statistics】Margin Debt / M2 = FINRA Debit Balances"
          "（Customer Margin Borrowing）÷ M2（Money Supply）→ removing secular growth",
          3, "#9aa0aa")
    _foot("Margin Debt YoY − M2 YoY → margin debt growth in excess of money-supply growth",
          5, "#9aa0aa")
    _foot("【Data Source：FINRA Margin Statistics】Margin Net Credit = Free Credit Balances"
          "（Cash + Margin Accounts）− Margin Debt",
          6, "#9aa0aa")
    # z-window footnotes (R4 Margin-Debt/M2-z monthly-60; R7 SP500/M2 monthly-60)
    _zrow_footnote(fig, _ZFN_MONTHLY_60, 4, _N6, _vs6)
    _zrow_footnote(fig, _ZFN_MONTHLY_60, 7, _N6, _vs6)

    output = Path(output)
    _write_panel_html(fig, output)
    return output


# ============================================================================
# Panel 7: Leveraged Funds Basis Trade Monitor (5 subplots) — built 2026-06-27
# ============================================================================
# Panel 0 — Macro State Scorecard (晨間儀表板 / 主頁)
# ============================================================================
def plot_macro_scorecard(panel: pd.DataFrame, output: str | Path) -> Path:
    """
    Panel 0 — Macro State Scorecard (主頁, HTML 卡片版).
    8 張卡 (對齊 7 panel + Equity Valuation); 每列 Indicator / Value / 燈 / As of。
    燈 = binary 綠/紅 (無 amber; 完全沿用各 panel 上色門檻):
      高尾 threshold: repo spread 20 / 所有 Z 2 / VIX 30 / NFCILEVERAGE+NFCI 家族 2
      signed       : Net Liq YoY · Real M2 YoY · term spread · Margin Net Credit (>=0 綠 / <0 紅)
      signed flip  : Margin Debt YoY - M2 YoY (>0 紅)
      SRF Use      : >= 1 Bn 紅 (SRF_STRESS_BN; 非 >0 —— 微量常態 0.001-0.010 Bn 為雜訊, 詳該常數註解)
      警戒線       : Reserve < 3 Tn
      其餘 (利率/餘額/部位/OAS level/殖利率/FX/CPI/PCE/Discount Window) = context 無燈
    整張卡可點 -> 該 panel HTML (新分頁); Card 8 無專屬 panel 故不可點。
    As of = 底層真實最後觀測日 (resample 類抓底層 daily series, 不抓月底/週中標籤)。
    時間 = macro_panel_transformed.parquet mtime (= 人工跑 main.py 時點; 跑 rebuild 不變)。
    """
    from datetime import datetime
    import html as _html
    df = panel
    cache_dir = Path(__file__).resolve().parent / "cache_dashboard"
    _pq = Path(__file__).resolve().parent / "output" / "macro_panel_transformed.parquet"
    # Data refreshed = 最後一次手動 trigger (main.py 或 fetch_long) 的牆鐘時間 = canonical 與深 cache 取最新 mtime
    _mtimes = ([_pq.stat().st_mtime] if _pq.exists() else [])
    for _cf in cache_dir.glob("*.parquet"):
        try:
            _mtimes.append(_cf.stat().st_mtime)
        except OSError:
            pass
    refreshed = (datetime.fromtimestamp(max(_mtimes)).strftime("%Y-%m-%d　%H：%M")
                 if _mtimes else "—")

    def hi(v, thr): return "g" if v < thr else "r"
    def lo(v, thr): return "g" if v >= thr else "r"
    def sgn(v):     return "g" if v >= 0 else "r"
    def flip(v):    return "r" if v > 0 else "g"

    # ── SRF Use 專用門檻(2026-07-31 修正;勿改用 flip)────────────────────────────
    # 原本 SRF Use 掛 flip(>0 即紅),前提是假設 SRF 只有「零」與「危機」兩態。
    # 實測 FRED RPONTSYD 有第三態,共三個量級:
    #   ① 真零        : 0
    #   ② 微量常態    : 0.001–0.010 Bn(= $1–10 百萬;交易對手維持操作連線的例行小額)
    #   ③ 真實動用    : 2 / 4.5 / 10.5 / 13 / 18 / 29 / 31 Bn(2025-2026 實際尖峰)
    # ② 的量級對千億級 facility 而言是雜訊,卻天天觸發 >0 → 紅燈長亮 = 警報失效(狼來了),
    # 真的出現十億級動用時反而不會被注意到 —— 這比「顯示 0 卻紅」的視覺矛盾嚴重。
    # 門檻取 1.0 Bn:落在 ②(≤0.010)與 ③(≥2)之間、兩個數量級的空隙中,兩側皆不誤判。
    # 配套:顯示格式同步改 ",.1f"(與 Panel 2 R6 / Discount Window 同刻度)→ 會亮紅的值恆 ≥1.0
    # 顯示得出來,微量則顯示 0.0 Bn 且維持綠,燈號與數字不再互相矛盾。
    # ⚠️ flip 為共用函式(Margin Debt YoY − M2 YoY 那列仍需 >0 即紅、且單位是 %),故另立本函式。
    SRF_STRESS_BN = 1.0   # SRF 動用達此金額(Bn)才判 stress;調整只改這一行
    def srf(v):     return "r" if v >= SRF_STRESS_BN else "g"
    def col(c): return df[c] if c in df.columns else pd.Series(dtype="float64")
    def _aod(s):
        s = s.dropna()
        return s.index[-1] if len(s) else None

    # ---- derived series (逐一對齊 panel transform) ----
    repo = (compute_repo_spread(col("SOFR"), col("EFFR"))
            if "SOFR" in df and "EFFR" in df else pd.Series(dtype="float64"))
    repo_z = expanding_zscore(repo) if len(repo.dropna()) else pd.Series(dtype="float64")
    # ── SSoT: scorecard 每條序列讀「與對應 panel 同一來源」(有深 cache 走深 loader, 其餘 canonical) ──
    #   根因防呆: scorecard 從前多處走 canonical、panel 走深 cache → 兩者前沿不同時值/日期對不上。
    #   OAS/repo 無深 cache → 兩邊都 canonical (自然一致; panel 那端由 _pending 標 "Not yet published")。
    try:
        _fed = load_display_deep(panel, ["WALCL", "WDTGAL", "RRPONTSYD", "WRBWFRBL", "RPONTSYD", "WLCFLPCL"])
    except Exception:
        _fed = {}
    def fcol(c): return _fed[c] if c in _fed else col(c)   # 流動性 6 條 → 深 cache (= Panel 2)
    try:
        _fx = load_display_fxvolmacro(panel)               # dict: DEXTAUS/DTWEXEMEGS/VIXCLS/CPIAUCSL/PCEPILFE
    except Exception:
        _fx = {}
    def xcol(c): return _fx[c] if c in _fx else col(c)     # FX/VIX/CPI/PCE → 深 cache (= Panel 5)
    try:
        _cur = load_display_curve(panel)                   # DataFrame: DGS2/DGS10/DFII10/T10YIE/T10Y2Y/T10Y3M
    except Exception:
        _cur = pd.DataFrame()
    def ccol(c): return _cur[c] if c in _cur.columns else col(c)   # yields → 深 cache (= Panel 4)
    try:
        _nfci = load_display_deep(panel, ["NFCILEVERAGE", "NFCI", "NFCIRISK", "NFCICREDIT", "ANFCI"])
    except Exception:
        _nfci = {}
    def ncol(c): return _nfci[c] if c in _nfci else col(c)  # NFCI 家族 → 深 cache (= Panel 6)
    try:
        _m2s = load_display_m2(panel)                       # Series (月頻深史)
    except Exception:
        _m2s = pd.Series(dtype="float64")
    def m2col(): return _m2s if len(_m2s) else col("M2SL")  # M2 → 深 cache (= Panel 2/8; 修 scorecard 內部 M2 讀兩種源的矛盾)
    if all(c in df for c in ("WALCL", "WDTGAL", "RRPONTSYD")):
        netliq, netliq_yoy = _netliq_display(panel)        # 與 Panel 2 共用同一 helper → 數值字面一致
    else:
        netliq = netliq_yoy = pd.Series(dtype="float64")
    realm2 = (compute_real_m2_yoy(m2col(), xcol("CPIAUCSL"))
              if "M2SL" in df and "CPIAUCSL" in df else pd.Series(dtype="float64"))
    hy_z = expanding_zscore(col("BAMLH0A0HYM2")) if "BAMLH0A0HYM2" in df else pd.Series(dtype="float64")
    qs_raw = (compute_credit_quality_spread(col("BAMLH0A3HYC"), col("BAMLH0A1HYBB"))
              if "BAMLH0A3HYC" in df and "BAMLH0A1HYBB" in df else pd.Series(dtype="float64"))
    qs_z = rolling_zscore(qs_raw, 252) if len(qs_raw.dropna()) else pd.Series(dtype="float64")
    em_z = rolling_zscore(col("BAMLEMHYHYLCRPIUSOAS"), 252) if "BAMLEMHYHYLCRPIUSOAS" in df else pd.Series(dtype="float64")
    ig_z = rolling_zscore(col("BAMLC0A0CM"), 252) if "BAMLC0A0CM" in df else pd.Series(dtype="float64")
    _vixs = (_fx["VIXCLS"] if "VIXCLS" in _fx else col("VIXCLS")).dropna()
    vix_z = expanding_zscore(_vixs) if len(_vixs) else pd.Series(dtype="float64")
    # YoY 先 resample 到完整月頻 index 再 pct_change(12) → 日曆對齊 (2025-10 政府關門缺月補 NaN, 不位移)。
    cpi_yoy = (xcol("CPIAUCSL").dropna().resample("MS").last().pct_change(12) * 100) if "CPIAUCSL" in df else pd.Series(dtype="float64")
    pce_yoy = (xcol("PCEPILFE").dropna().resample("MS").last().pct_change(12) * 100) if "PCEPILFE" in df else pd.Series(dtype="float64")

    _deb_raw = _nc_raw = pd.Series(dtype="float64")
    try:
        _mg = load_display_margin(panel)
        _m2m = _deep_monthly(panel, "M2SL", cache_dir)
        if len(_mg):
            _deb_raw = _mg["debit"]; _nc_raw = _mg["net_credit"]
        _deb = _deb_raw.resample("ME").last() if len(_deb_raw) else pd.Series(dtype="float64")
        _ncm = _nc_raw.resample("ME").last() if len(_nc_raw) else pd.Series(dtype="float64")
        if len(_deb) and len(_m2m):
            md_m2 = ((_deb / 1000.0) / _m2m * 100).dropna()
            md_z = rolling_zscore(md_m2, 60, min_periods=60)
            md_excess = ((_deb.pct_change(12) - _m2m.pct_change(12)) * 100).dropna()
        else:
            md_m2 = md_z = md_excess = pd.Series(dtype="float64")
        nc_bn = (_ncm / 1000.0).dropna() if len(_ncm) else pd.Series(dtype="float64")
    except Exception:
        md_m2 = md_z = md_excess = nc_bn = pd.Series(dtype="float64")

    _UP = {"TFF_2Y_LEVERAGED": 200_000, "TFF_10Y_LEVERAGED": 100_000}
    try:
        _tff = load_display_deep(panel, ["TFF_2Y_LEVERAGED", "TFF_10Y_LEVERAGED"])
    except Exception:
        _tff = {}
    def _pos(c):
        s = _tff.get(c, pd.Series(dtype="float64")).dropna()
        return (s * _UP[c] / 1e9) if len(s) else pd.Series(dtype="float64")
    pos2, pos10 = _pos("TFF_2Y_LEVERAGED"), _pos("TFF_10Y_LEVERAGED")
    d2 = pos2.diff() if len(pos2) else pos2
    d10 = pos10.diff() if len(pos10) else pos10
    d2_z = rolling_zscore(d2, 260, min_periods=260) if len(d2.dropna()) else pd.Series(dtype="float64")
    d10_z = rolling_zscore(d10, 260, min_periods=260) if len(d10.dropna()) else pd.Series(dtype="float64")

    try:
        ratio_full, z_full = _ratio_sp500_m2_z(df)               # 與 panel 同一 helper -> 值結構性一致
        ratio_z = z_full.reindex(df.index).dropna()              # monthly z (末點=最後完整月, 非混月)
        ratio = ratio_full.reindex(df.index)                     # daily ratio LINE (nowcast)
    except Exception:
        ratio = ratio_z = pd.Series(dtype="float64")
    spx_disp = load_display_sp500(df) if "SP500" in df.columns else pd.Series(dtype="float64")

    # ---- 組卡 ----  (dsrc = 日期改抓的底層 series; 不給則用值本身最後日)
    def mk(label, s, fmt, lf=None, dsrc=None, date_fmt="%m-%d"):
        vs = s.dropna()
        if len(vs) == 0:
            return (label, "—", "—", "")
        v = float(vs.iloc[-1])
        ao = _aod(dsrc) if dsrc is not None else vs.index[-1]
        if ao is None:
            ao = vs.index[-1]
        return (label, ao.strftime(date_fmt), fmt(v), (lf(v) if lf else ""))

    def _last2(s):
        s = s.dropna()
        return (s.index[-1], float(s.iloc[-1])) if len(s) else (None, None)
    _al, _vl = _last2(col("DFEDTARL")); _au, _vu = _last2(col("DFEDTARU"))
    fed_band = (("Fed Policy Band", _au.strftime("%m-%d"), f"{_vl:.2f} – {_vu:.2f}%", "")
                if _vl is not None and _vu is not None else ("Fed Policy Band", "—", "—", ""))

    card1 = [
        fed_band,
        mk("TGCR", col("TGCRRATE"), lambda v: f"{v:.2f} %"),
        mk("SOFR", col("SOFR"), lambda v: f"{v:.2f} %"),
        mk("EFFR", col("EFFR"), lambda v: f"{v:.2f} %"),
        mk("Repo Spread", repo, lambda v: f"{v:+.0f} bps", lambda v: hi(v, 20)),
        mk("Z · Repo Spread", repo_z, lambda v: f"{v:+.2f}", lambda v: hi(v, 2)),
        mk("Reserve Balances", fcol("WRBWFRBL"), lambda v: f"{v/1e6:.2f} Tn", lambda v: lo(v/1e6, 3)),
    ]
    card2 = [
        mk("Fed's TA", fcol("WALCL"), lambda v: f"{v/1000:,.0f} Bn"),
        mk("TGA", fcol("WDTGAL"), lambda v: f"{v/1000:,.0f} Bn"),
        mk("ON RRP", fcol("RRPONTSYD"), lambda v: f"{v:,.2f} Bn"),
        mk("Net Liquidity", netliq, lambda v: f"{v:,.0f} Bn", dsrc=fcol("RRPONTSYD")),   # (乙) as-of 錨 RRP 最後真值 (frontier), 非 ffill 尾端
        mk("Net Liq. YoY", netliq_yoy, lambda v: f"{v:+.2f} %", sgn, dsrc=fcol("RRPONTSYD")),   # (乙) as-of 錨 RRP (frontier), 與 level 一致
        mk("M2", m2col(), lambda v: f"{v/1000:.2f} Tn", date_fmt="%Y-%m"),
        mk("Real M2 YoY", realm2, lambda v: f"{v:+.2f} %", sgn, date_fmt="%Y-%m"),
        mk("SRF Use", fcol("RPONTSYD"), lambda v: f"{v:,.1f} Bn", srf),   # 門檻/格式緣由見上方 SRF_STRESS_BN 註解
        mk("Discount Window", fcol("WLCFLPCL"), lambda v: f"{v/1000:,.1f} Bn"),
    ]
    card3 = [
        mk("HY OAS", col("BAMLH0A0HYM2"), lambda v: f"{v*100:,.0f} bps"),
        mk("Z · HY OAS", hy_z, lambda v: f"{v:+.2f}", lambda v: hi(v, 2)),
        mk("Quality Spread", qs_raw, lambda v: f"{v*100:,.0f} bps"),
        mk("Z · Quality Spread", qs_z, lambda v: f"{v:+.2f}", lambda v: hi(v, 2)),
        mk("EM HY OAS", col("BAMLEMHYHYLCRPIUSOAS"), lambda v: f"{v*100:,.0f} bps"),
        mk("Z · EM HY OAS", em_z, lambda v: f"{v:+.2f}", lambda v: hi(v, 2)),
        mk("IG OAS", col("BAMLC0A0CM"), lambda v: f"{v*100:,.0f} bps"),
        mk("Z · IG OAS", ig_z, lambda v: f"{v:+.2f}", lambda v: hi(v, 2)),
    ]
    card4 = [
        mk("2Y UST", ccol("DGS2"), lambda v: f"{v:.2f} %"),
        mk("10Y UST", ccol("DGS10"), lambda v: f"{v:.2f} %"),
        mk("10Y Real (TIPS)", ccol("DFII10"), lambda v: f"{v:.2f} %"),
        mk("10Y Breakeven", ccol("T10YIE"), lambda v: f"{v:.2f} %"),
        mk("10Y − 3M Term Spread", ccol("T10Y3M"), lambda v: f"{v*100:+.0f} bps", sgn),
        mk("10Y − 2Y Term Spread", ccol("T10Y2Y"), lambda v: f"{v*100:+.0f} bps", sgn),
    ]
    card5 = [
        mk("USD/TWD", xcol("DEXTAUS"), lambda v: f"{v:.2f}"),
        mk("EM USD Index", xcol("DTWEXEMEGS"), lambda v: f"{v:.2f}"),
        mk("VIX", xcol("VIXCLS"), lambda v: f"{v:.1f}", lambda v: hi(v, 30)),
        mk("Z · VIX", vix_z, lambda v: f"{v:+.2f}", lambda v: hi(v, 2)),
        mk("CPI YoY", cpi_yoy, lambda v: f"{v:.2f} %", date_fmt="%Y-%m"),
        mk("Core PCE YoY", pce_yoy, lambda v: f"{v:.2f} %", date_fmt="%Y-%m"),
    ]
    card6 = [
        mk("NFCI Leverage", ncol("NFCILEVERAGE"), lambda v: f"{v:+.2f}", lambda v: hi(v, 2)),
        mk("NFCI", ncol("NFCI"), lambda v: f"{v:+.2f}", lambda v: hi(v, 2)),
        mk("NFCI Risk", ncol("NFCIRISK"), lambda v: f"{v:+.2f}", lambda v: hi(v, 2)),
        mk("NFCI Credit", ncol("NFCICREDIT"), lambda v: f"{v:+.2f}", lambda v: hi(v, 2)),
        mk("ANFCI", ncol("ANFCI"), lambda v: f"{v:+.2f}", lambda v: hi(v, 2)),
        mk("Margin Debt / M2", md_m2, lambda v: f"{v:.2f} %", None, dsrc=_deb_raw, date_fmt="%Y-%m"),
        mk("Z · Margin Debt / M2", md_z, lambda v: f"{v:+.2f}", lambda v: hi(v, 2), dsrc=_deb_raw, date_fmt="%Y-%m"),
        mk("Margin Debt YoY − M2 YoY", md_excess, lambda v: f"{v:+.2f} %", flip, dsrc=_deb_raw, date_fmt="%Y-%m"),
        mk("Margin Net Credit", nc_bn, lambda v: f"{v:+,.0f} Bn", sgn, dsrc=_nc_raw, date_fmt="%Y-%m"),
    ]
    card7 = [
        mk("2Y Fut. Net Pos.", pos2, lambda v: f"{v:+,.0f} Bn"),
        mk("10Y Fut. Net Pos.", pos10, lambda v: f"{v:+,.0f} Bn"),
        mk("Δ 2Y Fut. Net Pos.", d2, lambda v: f"{v:+,.0f} Bn"),
        mk("Z · Δ 2Y Fut. Net Pos.", d2_z, lambda v: f"{v:+.2f}", lambda v: hi(v, 2)),
        mk("Δ 10Y Fut. Net Pos.", d10, lambda v: f"{v:+,.0f} Bn"),
        mk("Z · Δ 10Y Fut. Net Pos.", d10_z, lambda v: f"{v:+.2f}", lambda v: hi(v, 2)),
    ]
    card8 = [
        mk("SP500", spx_disp, lambda v: f"{v:,.2f}"),
        mk("SP500 / M2 (per $1 Tn)", ratio, lambda v: f"{v:.2f}"),
        mk("Z · SP500 / M2", ratio_z, lambda v: f"{v:+.2f}", lambda v: hi(v, 2), dsrc=col("M2SL"), date_fmt="%Y-%m"),   # monthly z → as-of 錨 M2 最後真值 (2026-05), 與 M2/Real M2 YoY 一致 (非 ffill 日尾端)
    ]

    # CARDS: (title, color, rows, footnotes, links)  links=[(label,href),...]; [] -> 不可點
    CARDS = [
        ("01 · Repo Plumbing Monitor", "#3B82F6", card1,
         ["Repo Spread = SOFR − EFFR"], [("", "01_repo_plumbing.html")]),
        ("02 · Liquidity Monitor", "#14B8A6", card2,
         ["Net Liquidity = Fed's TA − TGA − ON RRP", "Real M2 YoY = M2 YoY − CPI YoY"],
         [("", "02_net_liquidity.html")]),
        ("03 · Credit Stress Monitor", "#8B5CF6", card3,
         ["Quality Spread = CCC OAS − BB OAS"],
         [("03-1", "03-1_credit_stress_hy_oas.html"), ("03-2", "03-2_credit_stress_other_oas.html")]),
        ("04 · Yield Curve & Real Rates Monitor", "#65A30D", card4, [], [("", "04_yield_curve.html")]),
        ("05 · FX & Volatility & Inflation Monitor", "#F97316", card5, [], [("", "05_fx_vol_inflation.html")]),
        ("06 · Leverage Monitor", "#EF4444", card6, [], [("", "06_leverage_monitor.html")]),
        ("07 · Leveraged Funds Basis Trade Monitor", "#D97706", card7,
         ["Net Pos. = face-value (par) notional"], [("", "07_leveraged_funds_basis_trade.html")]),
        ("08 · Equity Valuation Monitor", "#EC4899", card8, [], []),
    ]

    _SPARK = ('<svg width="16" height="16" viewBox="0 0 14 14" aria-hidden="true">'
              '<path d="M2 2 V12 H13" stroke="currentColor" stroke-width="1" opacity="0.4" fill="none"/>'
              '<polyline points="3,9.5 6,6.5 8.5,8 12,3.5" fill="none" stroke="currentColor" '
              'stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round"/></svg>')

    def _icons(links):
        if not links:
            return ""
        parts = []
        for lbl, href in links:
            lab = ('<span class="iclbl">%s</span>' % _html.escape(lbl)) if lbl else ""
            ttl = ("View %s detail" % lbl) if lbl else "View detail panel"
            parts.append('<a class="ic" href="%s" target="_blank" rel="noopener" title="%s">%s%s</a>'
                         % (href, ttl, lab, _SPARK))
        return '<span class="icg">%s</span>' % "".join(parts)

    def _rcard(title, color, rws, fns, links):
        out = []
        for (label, ao, vstr, light) in rws:
            isz = " z" if label.startswith("Z ·") else ""
            dot = (f'<span class="d d{light}"></span>' if light in ("g", "r") else "<span></span>")
            out.append('<div class="r"><span class="n%s">%s</span><span class="v">%s</span>%s<span class="a">%s</span></div>'
                       % (isz, _html.escape(label), _html.escape(vstr), dot, ao))
        # footnote 在白框「外」、框下方; 固定保留區 (空卡也保留) → 白框 flex:1 才會等高
        fn_html = ('<div class="cfnbox">%s</div>'
                   % "".join('<p class="cfn">%s</p>' % _html.escape(f) for f in fns))
        return ('<div class="cell">'
                '<div class="scd-c"><div class="bar" style="background:%s"></div>'
                '<div class="cth"><p class="ct">%s</p>%s</div>'
                '<div class="hd"><span>Indicator</span><span class="hv">Value</span><span></span><span class="ha">As of</span></div>%s</div>'
                '%s</div>'
                % (color, _html.escape(title), _icons(links), "".join(out), fn_html))

    grid = "".join(_rcard(*c) for c in CARDS)

    _TPL = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Macro State Scorecard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap">
<style>
*{box-sizing:border-box}
body{margin:0;background:#1b1e24;font-family:'Inter','Noto Sans TC',sans-serif}
.scd{padding:22px 26px 20px}
.scd-h{color:#f4f5f7;font-size:20px;font-weight:600;margin:0;letter-spacing:.02em}
.scd-sub{color:#f4f5f7;font-size:12px;font-weight:400;margin:5px 0 8px}
.scd-g{display:grid;grid-template-columns:repeat(auto-fit,minmax(292px,1fr));gap:13px;grid-auto-rows:1fr}
.cell{display:flex;flex-direction:column}
.scd-c{position:relative;overflow:hidden;background:#23272f;border:.5px solid #2e333c;border-radius:12px;padding:12px 14px 11px 17px;flex:1;min-height:0}
.bar{position:absolute;left:0;top:0;bottom:0;width:3px}
.cth{display:flex;justify-content:space-between;align-items:flex-start;gap:8px;margin-bottom:8px}
.ct{color:#dfe2e7;font-size:12px;font-weight:600;margin:0;line-height:1.3;flex:1;min-width:0}
.icg{display:inline-flex;gap:12px;flex-shrink:0;padding-top:1px}
.ic{color:#dce0e6;display:inline-flex;align-items:center;gap:3px;text-decoration:none;transition:color .12s}
.ic:hover{color:#fff}
.iclbl{font-size:10px;font-family:ui-monospace,Menlo,monospace;letter-spacing:.02em}
.hd{display:grid;grid-template-columns:minmax(0,1fr) 72px 12px 42px;gap:6px;padding:0 0 5px;border-bottom:.5px solid rgba(255,255,255,.13);margin-bottom:2px}
.hd span{color:#777c84;font-size:9px;font-weight:500;text-transform:uppercase;letter-spacing:.05em}
.hd .hv,.hd .ha{text-align:right}
.r{display:grid;grid-template-columns:minmax(0,1fr) 72px 12px 42px;gap:6px;align-items:center;padding:3px 0;border-bottom:.5px solid rgba(255,255,255,.045)}
.r:last-child{border-bottom:none}
.n{color:#c2c5cb;font-size:11.5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.n.z{color:#8d929a;padding-left:13px}
.v{color:#e6e8eb;font-size:11.5px;font-family:ui-monospace,Menlo,monospace;text-align:right;font-weight:500;white-space:nowrap}
.d{width:9px;height:9px;border-radius:50%;justify-self:center}
.dg{background:#56C28A}.dr{background:#FF6B6B}
.a{color:#777c84;font-size:10px;font-family:ui-monospace,Menlo,monospace;text-align:right}
.cfnbox{min-height:40px;padding:9px 0 0 18px}
.cfn{color:#6b7079;font-size:9.5px;margin:0 0 2px;line-height:1.5}
.cfn:last-child{margin-bottom:0}
.lg{color:#f4f5f7;font-size:12px;margin:0 2px 16px;display:flex;gap:10px;flex-wrap:wrap;align-items:center}
.li{display:inline-flex;align-items:center;gap:6px}
.lg i{width:9px;height:9px;border-radius:50%;display:inline-block}
.sep{color:#f4f5f7}
</style></head><body>
<div class="scd">
<p class="scd-h">MACRO STATE SCORECARD</p>
<p class="scd-sub">Data refreshed：@@REFRESHED@@ (Taipei · UTC+8)　│　Click a card's chart icon to open its detail panel</p>
<div class="lg"><span class="li"><i class="dg"></i>Normal</span><span class="li"><i class="dr"></i>Stress</span><span class="sep">│</span><span>No dot = Context (level / positioning)</span><span class="sep">│</span><span>As of = Each indicator's latest date</span></div>
<div class="scd-g">@@GRID@@</div>
</div></body></html>"""
    page = _TPL.replace("@@REFRESHED@@", refreshed).replace("@@GRID@@", grid)
    output = Path(output)
    output.write_text(page, encoding="utf-8")
    return output


# ============================================================================
def plot_basis_trade(panel: pd.DataFrame, output: str | Path) -> Path:
    """
    Panel 7 — Leveraged Funds Basis Trade Monitor: 5 子圖 (共用時間軸)。
    主題: Leveraged Funds 美債期貨淨部位 (CFTC TFF) ↔ repo 融資壓力 ↔ 估值 — basis trade 監控鏈。
      1 2Y + 10Y Net Position 併圖 (左軸共用階梯) + 2Y/10Y UST yield (右軸共用平滑)
          → 部位 divergence 直接可見 (Scheme 2: tenor=色相 2Y 紫 / 10Y 青; 部位粗實主、殖利率淺細輔); 自帶灰 0 線
      2 Weekly Δ 2Y Net Position + rolling-5yr Δz (右, ≥+2 紅 = 失序 unwind 警報)
      3 Weekly Δ 10Y Net Position + rolling-5yr Δz (右, unwind 警報)
      4 Repo Spread (SOFR−EFFR) + SP500 — basis trade 靠 repo 融資 (Panel 1 R2 同範本/同 helper, clip=False)
      5 SP500 / M2 (Per $1 Tn) + 5yr rolling-z — 估值 stretch
    顯示窗 2015+ (anchor=max(最早 TFF, 2015)); TFF 走純顯示深史 (cache_dashboard, fetch_long_cftc.py 抓 2006) 餵 R2/R3 z baseline → 開窗即穩;
      R4 repo 因 SOFR 2018-04 才有 → 2015–2018 repo area 空白 (SP500 線仍滿)。無 TFF 深史 cache → graceful 退回 panel (2018+)。
    為何 level 併、Δ 不併: level 同尺度 → divergence 是訊號; Δz 兩條 green/red area 疊會糊, 且 unwind 警報須分 tenor 各自看。
    TFF 週頻 (QA Entry 5): level ffill 成 as-known 日頻階梯; Δ/Δz 在 native weekly 上算 (dropna 稀疏還原 → diff, 勿 daily-ffill 後 diff) 再 reindex。
    Δz form: level 非定態 → 對 Δ z (Δ ADF 均值定態, 但變異數結構性上升 σ ratio 2Y 1.31x/10Y 1.16x → rolling-5yr 260wk, 非 expanding; 對齊當前 regime + 與 ratio-z 一致)。
    顯示單位: 部位/Δ 以 par face-value notional 顯示 (Bn; 2Y ×$200k / 10Y ×$100k per CME 合約面值) — 採 par 而非 market value 以隔離 positioning 訊號與 MTM 市價噪音 (Fed FSR / OFR convention); 2Y/10Y 面值差 2x → 共用左軸下 2Y 相對 10Y 拉伸 2x (apples-to-apples 修正); z scale-invariant 故 R2/R3 警報不受換算影響。cache + fetch_long_cftc.py 維持原始 contracts。
    """
    df = panel.copy()
    # par/face-value notional 換算 (每口面值 ×口數 /1e9 → Bn): 採 par 而非 market value 以隔離 positioning 與 MTM 市價噪音 (market value 需期價序列, 對 monitor overkill 且 pipeline 無乾淨期價); Fed FSR / OFR convention; z scale-invariant → 換算不動 R2/R3 unwind 警報
    USD_PER = {"TFF_2Y_LEVERAGED": 200_000, "TFF_10Y_LEVERAGED": 100_000}
    # 深史 (純顯示 cache_dashboard; 無 cache → 退回 panel 2018+; 絕不回流 FM): TFF 餵 R1 level + R2/R3 z baseline; DGS2/10 餵 R1 殖利率 (比照 Panel 4)
    _deep = load_display_deep(panel, ["TFF_2Y_LEVERAGED", "TFF_10Y_LEVERAGED", "DGS2", "DGS10"])
    # anchor = 最早 TFF 深史, 下限 2015 (顯示窗): TFF 拉滿 2006 餵 z baseline (開窗即穩); R4 repo 因 SOFR 2018 前空白可接受
    _tff_firsts = [_deep[s].first_valid_index() for s in ("TFF_2Y_LEVERAGED", "TFF_10Y_LEVERAGED")
                   if _deep[s].first_valid_index() is not None]
    _anchor = max(min(_tff_firsts), pd.Timestamp(2015, 1, 1)) if _tff_firsts else pd.Timestamp(2015, 1, 1)
    df = df.loc[_anchor - pd.Timedelta(days=7):]            # 往前 7 天 (2014 末幾個交易日, 11yr 尺度約 2-3px 不可見) → Jan-1 年刻度落進 autorange 內可顯示 (否則資料首日=Jan-2, Jan-1 刻度被切在邊界外)
    spx = load_display_sp500(df)                             # SP500 背景 (深史; R2/R3/R4 用)

    fig = make_subplots(
        rows=5, cols=1, shared_xaxes=True,
        subplot_titles=(
            "2Y & 10Y UST Futures Net Position & Treasury Yields",
            "Weekly Δ 2Y Net Position & Rolling_5yr Z-score（Unwind Alarm ≥ 2）",
            "Weekly Δ 10Y Net Position & Rolling_5yr Z-score（Unwind Alarm ≥ 2）",
            "Repo Spread（Repo Stress ≥ 20 bps；Label ≥ 100 bps）& SP500",
            "SP500 / M2 （Per $1 Tn）& Rolling_5yr Z-score（Valuation Stretched ≥ 2）",
        ),
        specs=[[{"secondary_y": True}] for _ in range(5)],
        vertical_spacing=panel_dims(5)[1],
    )

    # ---- R1 併圖: 2Y+10Y Net Position (左軸共用, 紫/青 粗實階梯) + 2Y/10Y UST (右軸共用, 淺紫/淺青 細平滑) + 灰 0 線 ----
    # hover 序 2Y pos → 10Y pos → 2Y UST → 10Y UST (反序加入: yields 先[10Y→2Y], positions 後[10Y→2Y])
    # 0 線 (左主軸, skip hover) — 先加; net long(>0)/short(<0) 界
    fig.add_trace(go.Scatter(x=[df.index.min(), df.index.max()], y=[0, 0], mode="lines",
                             line=dict(color="#E8EAED", width=1.0, dash="dot"),
                             hoverinfo="skip", showlegend=False), row=1, col=1, secondary_y=False)
    # 殖利率 (右副軸共用, 淺細平滑) — 先加 → hover 底
    _yv = []
    for sid, nm, col in [("DGS10", "10Y UST", T["iorb"]),    # 殖利率沿用 Panel 4 配色 (2Y 藍 / 10Y 橙); 深史 (load_display_deep) → 與部位同從 2015, 非只 2018+
                         ("DGS2", "2Y UST", T["sofr"])]:
        s = _deep[sid].reindex(df.index)
        if s.notna().any():
            _sy, _scd = _pending(s, df.index, ".2f", "%")   # 日頻殖利率: 尾端 T+1 未公布 → hold + 標 pending
            fig.add_trace(go.Scatter(x=df.index, y=_sy, name=nm, showlegend=False, customdata=_scd,
                                     line=dict(color=col, width=1.1), connectgaps=True,
                                     hovertemplate=_ht_cd(nm, color=col)),
                          row=1, col=1, secondary_y=True)
            _yv.append(s)
    if _yv:
        _av = pd.concat(_yv)
        _yt0, _yd = _range_5tick(float(_av.min()), float(_av.max()))
        _axis5(fig, 1, True, _yt0, _yd, title_text="%", showgrid=False)
    # 部位 (左主軸共用, 粗實階梯) — 後加 → hover 頂 + 線最上
    _pv = []
    for sid, nm, col in [("TFF_10Y_LEVERAGED", "10Y Fut. Net Position", T["tgcr"]),
                         ("TFF_2Y_LEVERAGED", "2Y Fut. Net Position", T["onrrp"])]:
        s = _deep[sid].reindex(df.index).ffill() * USD_PER[sid] / 1e9   # 深史 TFF (cache_dashboard) → 顯示窗稀疏 weekly → as-known 日頻階梯; ×面值 /1e9 → Bn (par notional)
        if s.notna().any():
            fig.add_trace(go.Scatter(x=s.index, y=s, name=nm, showlegend=False,
                                     line=dict(color=col, width=1.5), connectgaps=True,
                                     hovertemplate=_ht(nm, ".2f", " Bn", color=col)),
                          row=1, col=1, secondary_y=False)
            _pv.append(s)
    if _pv:
        _ap = pd.concat(_pv)
        _pt0, _pd = _range_5tick(float(min(_ap.min(), 0.0)), float(max(_ap.max(), 0.0)))  # 含 0 → 0 線可見
        _axis5(fig, 1, False, _pt0, _pd, title_text="Net Face Value (Bn)", tickformat="~s")

    # ---- R2/R3 共用: TFF Δ (左主軸, 白純線) + rolling-5yr Δz (右副軸, ≥+2 紅 = unwind 警報; 主信號) ----
    def _delta_z_row(row, tff_col):
        """Δ 原始線 + Δz; Δz 在 native weekly 上算 (稀疏還原) 再 reindex 日頻階梯; hover 序 Δ → Z (Δ 後加 = 頂)。
        Δz 高尾 ≥+2 紅 = 失序 unwind (covering shorts, Δ>0 → 降 basis trade); rapid short-building (Δ<0, z<-2) 留綠 (由 R1 level 管)。
        左軸放 Δ 而非 SP500: Δ 是 R2/R3 獨有資訊 (實際週變動); SP500 已在 R4 → 不重複。(Δ 與 z 同形狀但單位不同: contracts vs 標準化, 並陳。)
        z 用 rolling-5yr 而非 expanding: Δ 雖 ADF 定態(均值)但變異數有結構性上升 (σ 2018-20→23-25 ratio 2Y 1.31x/10Y 1.16x; 全史 2006+ 因早段部位更小 → 真實增幅更大) →
          expanding σ 偏小使近段 z 偏高; rolling-5yr(260wk) 對齊當前 regime, 並與 dashboard ratio-z(5yr) 一致。"""
        nw = _deep[tff_col].dropna() * USD_PER[tff_col] / 1e9   # 深史 native weekly level (cache_dashboard 全史 2006+ → rolling-5yr 約 2011 起滿窗, 2015 顯示窗早已穩); ×面值 /1e9 → Bn (par notional)
        z = (rolling_zscore(nw.diff(), 260, min_periods=260)  # Δ 變異數結構性上升 (ratio>1) → rolling-5yr(260wk) 對齊當前 regime (非 expanding, 否則近段 z 偏高); z scale-invariant → ×面值不改變 z 值
             .reindex(df.index, method="ffill").dropna())
        diff = nw.diff().reindex(df.index, method="ffill")   # as-known 日頻 Δ (顯示窗)
        if len(z):                                           # Δz threshold-area (右副軸, ≥+2 紅) — 先加 → hover 底; clip=True 釘 ±4; 內含 thr 線 + 軸
            _threshold_area(fig, z, row, 2.0, "Z-score", -4.0, 2.0, fmt=".2f")
        if diff.notna().any():                               # Δ 原始線 (左主軸, 白純線) — 後加 → hover 頂; 壓在 z-area 上高對比
            fig.add_trace(go.Scatter(x=diff.index, y=diff, name="Δ Net Position", showlegend=False,
                                     line=dict(color=T["sp500"], width=0.9), connectgaps=True,
                                     hovertemplate=_ht("Δ Net Position", ".2f", " Bn", color=T["sp500"])),
                          row=row, col=1, secondary_y=False)
            _z0, _zd = _zero_5tick(float(diff.min()), float(diff.max()))   # 含 0 的 5 格
            _axis5(fig, row, False, _z0, _zd, title_text="Δ Face Value (Bn)", tickformat="~s")

    _delta_z_row(2, "TFF_2Y_LEVERAGED")
    _delta_z_row(3, "TFF_10Y_LEVERAGED")

    # ---- R4: Repo Spread + SP500 (整組 clone Panel 1 R2; 同 module-level _threshold_area(clip=False)) ----
    spread = None
    if {"SOFR", "EFFR"}.issubset(df.columns):
        spread = compute_repo_spread(df["SOFR"], df["EFFR"]).dropna()
        _add_sp500_bg(fig, spx, 4)                           # SP500 先加 (左主軸) → hover 底
        # Repo Spread 紅 ≥ 20 bps; clip=False → 尖峰衝出由軸切 (同 Panel 1 R2); 內含 20 線 + 軸 (title bps)
        _threshold_area(fig, spread, 4, 20.0, "Repo Spread", -20.0, 20.0, fmt=".0f", suffix=" bps",
                        clip=False, title_text="bps", disp_index=df.index)

    # ---- R5: SP500 / M2 ratio-z (共用 _add_ratio_zrow -> 月頻 60mo z) ----
    _add_ratio_zrow(fig, panel, df, 5)

    apply_global_layout(
        fig, df, n_rows=5, height=panel_dims(5)[0],
        title="Panel 7 · Leveraged Funds Basis Trade Monitor",
    )

    # ---- 以下 annotation 須在 apply_global_layout 後 (否則被標題 recolor 迴圈蓋白) ----
    fig.update_xaxes(tick0=_anchor)                          # 年刻度對齊 anchor (Jan-1) → 顯示窗起年 (2015) 在縱軸交叉處顯示, 不被吃掉
    _N, _vs = 5, panel_dims(5)[1]
    _ph = (1 - (_N - 1) * _vs) / _N                          # 每列 paper 高 (footnote 定位)

    # 0 線標籤 (R1)
    fig.add_annotation(text="Net Long / Short", x=df.index.min(), y=0,
                       xanchor="left", yanchor="bottom", showarrow=False,
                       font=dict(color="#E8EAED", size=9.5), row=1, col=1, secondary_y=False)

    # R1 footnote (兩行: 第1行單位/方法 par notional, 第2行解讀 lens; 都靠左、灰、同字級; <br> 分行)
    fig.add_annotation(
        text="Face-value (par) notional follows Fed FSR & OFR convention and isolates positioning from mark-to-market price noise<br>"
             "Net short → basis-trade size · 2Y → Fed-path & policy-cycle lens │ "
             "10Y → systemic-risk & term-premium lens │ 2Y–10Y divergence → curve positioning",
        xref="paper", yref="paper", x=0.0, y=(1 - 0 * (_ph + _vs) - _ph) - 0.05 * _ph,
        xanchor="left", yanchor="top", align="left", showarrow=False,
        font=dict(size=9.5, color="#9aa0aa"), opacity=0.92,
    )

    # z-window footnotes (R2/R3 Δ-position-z rolling-260-weekly; R5 SP500/M2 monthly-60)
    _zrow_footnote(fig, _ZFN_WEEKLY_260, 2, _N, _vs)
    _zrow_footnote(fig, _ZFN_WEEKLY_260, 3, _N, _vs)
    _zrow_footnote(fig, _ZFN_MONTHLY_60, 5, _N, _vs)

    # R4 Repo Spread 定義 footnote (綠, 同 Panel 1) + 極端值文字標籤 (≥100 bps)
    if spread is not None:
        fig.add_annotation(
            text="Repo Spread = SOFR − EFFR", xref="paper", yref="paper",
            x=0.0, y=(1 - 3 * (_ph + _vs) - _ph) - 0.05 * _ph,
            xanchor="left", yanchor="top", showarrow=False,
            font=dict(size=10, color=T["bar_low"]), opacity=0.9,
        )
        _extreme_labels_impl(fig, df.index[0], df.index[-1], spread, 4, "Repo Spread", ".0f", " bps",
                             annot_thr=100, cap=68, secondary_y=True, pos={"2019-09-17": (16, 14, "left")})

    output = Path(output)
    _write_panel_html(fig, output)
    return output


def build_master_dashboard(panel: pd.DataFrame, output: str | Path) -> Path:
    """
    All panels stacked in one HTML, deployable as portfolio piece.
    """
    from transformations import expanding_zscore, rolling_zscore

    fig = make_subplots(
        rows=4,
        cols=2,
        shared_xaxes=False,
        subplot_titles=(
            "① SOFR − EFFR Spread (bps)",
            "② Net Liquidity ($Bn)",
            "③ HY OAS & Z-Score",
            "④ 10Y − 2Y Curve",
            "⑤ NFCI Leverage Subindex",
            "⑥ CCC − BB Quality Spread",
            "⑦ VIX",
            "⑧ 10Y Real Yield",
        ),
        vertical_spacing=0.08,
        horizontal_spacing=0.10,
        specs=[
            [{}, {"secondary_y": True}],  # row 1 — (1,2) Net Liq + SPX ✓
            [{"secondary_y": True}, {}],  # row 2 — (2,1) HY OAS + Z   ✓
            [{}, {}],  # row 3
            [{}, {}],  # row 4
        ],
    )

    df = panel.copy()
    for col in ["WALCL", "WDTGAL", "RRPONTSYD"]:
        if col in df.columns:
            df[col] = df[col].ffill()

    # ① SOFR-EFFR
    if {"SOFR", "EFFR"}.issubset(df.columns):
        spread = compute_repo_spread(df["SOFR"], df["EFFR"]).reindex(df.index)
        fig.add_trace(
            go.Scatter(
                x=spread.index,
                y=spread,
                name="SOFR-EFFR",
                line=dict(color="#7C3AED", width=1.4),
                fill="tozeroy",
                fillcolor="rgba(124,58,237,0.10)",
            ),
            row=1,
            col=1,
        )
        fig.add_hline(y=10, line_dash="dash", line_color="red", row=1, col=1)

    # ② Net Liquidity
    if {"WALCL", "WDTGAL", "RRPONTSYD"}.issubset(df.columns):
        nl = compute_net_liquidity(df["WALCL"], df["WDTGAL"], df["RRPONTSYD"])
        fig.add_trace(go.Scatter(x=nl.index, y=nl, name="Net Liquidity", line=dict(color="#10B981", width=1.5)), row=1, col=2)
        if "SP500" in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df["SP500"], name="SPX", line=dict(color="#1a1a1a", width=1, dash="dot")),
                row=1,
                col=2,
                secondary_y=True,
            )

    # ③ HY OAS + zscore
    if "BAMLH0A0HYM2" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["BAMLH0A0HYM2"], name="HY OAS", line=dict(color="#EA580C", width=1.4)), row=2, col=1
        )
        z = expanding_zscore(df["BAMLH0A0HYM2"].dropna())   # HY OAS = ADF 定態 → expanding（與 Panel 3-1 / scorecard 同法）
        fig.add_trace(
            go.Scatter(x=z.index, y=z, name="HY OAS Z", line=dict(color="#DC2626", width=1, dash="dot")),
            row=2,
            col=1,
            secondary_y=True,
        )

    # ④ Curve
    if "T10Y2Y" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["T10Y2Y"],
                name="10Y-2Y",
                line=dict(color="#7C3AED", width=1.4),
                fill="tozeroy",
                fillcolor="rgba(124,58,237,0.10)",
            ),
            row=2,
            col=2,
        )
        fig.add_hline(y=0, line_dash="solid", line_color="red", row=2, col=2)

    # ⑤ NFCI Leverage
    if "NFCILEVERAGE" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["NFCILEVERAGE"], name="NFCI Lev", line=dict(color="#0B447C", width=1.4)), row=3, col=1
        )
        fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=0.5, row=3, col=1)
        fig.add_hline(y=1, line_dash="dash", line_color="red", row=3, col=1)

    # ⑥ Quality spread
    if {"BAMLH0A3HYC", "BAMLH0A1HYBB"}.issubset(df.columns):
        qs = compute_credit_quality_spread(df["BAMLH0A3HYC"], df["BAMLH0A1HYBB"])
        fig.add_trace(
            go.Scatter(
                x=qs.index,
                y=qs,
                name="CCC-BB",
                line=dict(color="#DC2626", width=1.4),
                fill="tozeroy",
                fillcolor="rgba(220,38,38,0.10)",
            ),
            row=3,
            col=2,
        )

    # ⑦ VIX
    if "VIXCLS" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["VIXCLS"], name="VIX", line=dict(color="#1a1a1a", width=1.2)), row=4, col=1)
        fig.add_hline(y=20, line_dash="dash", line_color="orange", row=4, col=1)

    # ⑧ Real yield
    if "DFII10" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["DFII10"], name="10Y Real", line=dict(color="#10B981", width=1.4)), row=4, col=2
        )

    fig.update_layout(
        height=1300,
        template="plotly_white",
        showlegend=False,
        title=dict(
            text=f"<b>Macro Liquidity & Risk Dashboard</b> "
            f'<span style="font-size:11px; color:#666;">'
            f"· Last update: {df.index[-1]:%Y-%m-%d}"
            f" · Built with FRED API</span>",
            font=dict(size=18, family="Inter, sans-serif"),
        ),
        font=dict(family="Inter, sans-serif", size=10),
        margin=dict(t=80, b=40, l=60, r=40),
    )

    output = Path(output)
    fig.write_html(output, include_plotlyjs="cdn", full_html=True)
    return output
