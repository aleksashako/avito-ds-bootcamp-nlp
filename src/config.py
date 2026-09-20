from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset"   # исходные parquet-файлы
ART_DIR = ROOT / "artifacts"  # промежуточные результаты: сплиты, индексы, эмбеддинги
ART_DIR.mkdir(exist_ok=True)

TRAIN_PATH = DATA_DIR / "train.parquet"
BENCH_QUERIES_PATH = DATA_DIR / "benchmark_queries.parquet"
BENCH_ITEMS_PATH = DATA_DIR / "benchmark_items.parquet"

SEED = 42
TOP_K = 50  # по условию задачи

SEARCH_KEY = [
    "search_query",
    "search_location_id",
    "search_is_delivery_search",
    "search_infm_params_text",
    "search_category",
]

ITEM_COLS = [
    "item_id", "item_title_raw", "item_description_raw", "item_infm_params_text",
    "item_category_id", "item_microcat_id", "item_price", "item_rating",
    "item_rating_reviews_count", "item_location_id", "item_latitude", "item_longitude",
    "item_is_phone_hidden", "item_is_message_forbidden",
]
