from dotenv import load_dotenv
from fred_loader import FredLoader

load_dotenv()  # 灌 FRED key（.env 的 FRED_API_KEY）
loader = FredLoader(cache_dir="./cache_dashboard")  # 獨立 dir，不被 main.py glob → 不污染 pipeline
panel = loader.fetch_many(["M2SL"], start="1990-01-01", force_refresh=True)  # latest-revised
print(f"long-M2: {panel.shape}  {panel.index.min().date()} → {panel.index.max().date()}")
print(panel.tail())
print("\n存到 cache_dashboard/M2SL.parquet（dashboard-local，latest-revised，絕不回流 FM）")