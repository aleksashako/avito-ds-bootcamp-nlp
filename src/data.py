import pandas as pd

from .config import BENCH_ITEMS_PATH, BENCH_QUERIES_PATH, TRAIN_PATH

NUMERIC_AS_TEXT = ["item_price", "item_latitude", "item_longitude"]


def _fix_types(df: pd.DataFrame) -> pd.DataFrame:
    """ поправляем типы данных для числовых значений, как в колабе """
    for c in NUMERIC_AS_TEXT:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    for c in ["item_title_raw", "item_description_raw", "item_infm_params_text",
              "search_query", "search_infm_params_text"]:
        if c in df.columns:
            df[c] = df[c].fillna("")
    return df


def load_train() -> pd.DataFrame:
    return _fix_types(pd.read_parquet(TRAIN_PATH))


def load_bench_queries() -> pd.DataFrame:
    return _fix_types(pd.read_parquet(BENCH_QUERIES_PATH))


def load_bench_items() -> pd.DataFrame:
    return _fix_types(pd.read_parquet(BENCH_ITEMS_PATH))

