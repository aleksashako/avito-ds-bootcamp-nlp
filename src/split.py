"""
Офлайн-разбиения, которые повторяют устройство benchmark

Что известно о benchmark и как строить описала в readme
"""
import numpy as np
import pandas as pd

from .config import ART_DIR, ITEM_COLS, SEARCH_KEY, SEED
from .data import load_bench_items, load_train

SHARE_SEEN = 0.37 # доля знакомых текстов, как в benchmark
EMPTY_SHARE = {"seen": 0.44, "unseen": 0.74} # доля запросов без фильтров в каждой группе
SPLITS = {"val": dict(n=3000, seed=SEED), "rank": dict(n=6000, seed=SEED + 1)}


def _pick(pool: pd.DataFrame, n: int, empty_share: float, used: set, rng) -> pd.DataFrame:
    """Выбрать n поисков, по одному на текст, с нужной долей пустых фильтров."""
    pool = pool[~pool.search_query.isin(used)].sample(frac=1, random_state=rng)
    n_empty = int(round(n * empty_share))
    empty = pool[pool.is_empty].drop_duplicates("search_query").head(n_empty)
    rest = pool[~pool.is_empty & ~pool.search_query.isin(set(empty.search_query))]
    rest = rest.drop_duplicates("search_query").head(n - len(empty))
    return pd.concat([empty, rest])


def make_split(train: pd.DataFrame, base_idx: np.ndarray, name: str, n: int, seed: int):
    """base_idx — строки train, из которых делаем разбиение (позиции в полном train)."""
    rng = np.random.RandomState(seed)
    base = train.iloc[base_idx]

    # --- поиски и их релевантные объявления
    searches = (base.groupby(SEARCH_KEY, sort=False).item_id.agg(lambda s: sorted(set(s)))
                .rename("items").reset_index())
    searches["is_empty"] = searches.search_infm_params_text.eq("")
    searches["text_n_searches"] = searches.search_query.map(searches.search_query.value_counts())

    n_seen = int(round(n * SHARE_SEEN))
    used: set = set()
    # знакомые тексты: у текста должен остаться хотя бы один поиск в fit
    seen = _pick(searches[searches.text_n_searches >= 2], n_seen, EMPTY_SHARE["seen"], used, rng)
    used |= set(seen.search_query)
    # новые тексты: выбираем равномерно по уникальным текстам
    unseen = _pick(searches, n - n_seen, EMPTY_SHARE["unseen"], used, rng)

    val = pd.concat([seen.assign(seen_text=1), unseen.assign(seen_text=0)])
    val = val.sample(frac=1, random_state=rng).reset_index(drop=True)
    val["query_id"] = [f"{name}{i:0{16 - len(name)}d}" for i in range(len(val))]  # 16 символов

    # --- fit: убираем все строки новых текстов и отложенные поиски знакомых
    bkey = pd.MultiIndex.from_frame(base[SEARCH_KEY])
    drop = base.search_query.isin(set(unseen.search_query)) | bkey.isin(pd.MultiIndex.from_frame(seen[SEARCH_KEY]))
    fit_idx = base_idx[~drop.values]

    # --- релевантные и корпус
    truth = val[["query_id", "items"]].explode("items").rename(columns={"items": "item_id"})
    bench_items = load_bench_items()
    held = base[ITEM_COLS].drop_duplicates("item_id")
    held = held[held.item_id.isin(set(truth.item_id))]
    corpus = pd.concat([bench_items[ITEM_COLS], held]).drop_duplicates("item_id").reset_index(drop=True)

    q = val[["query_id"] + SEARCH_KEY + ["seen_text"]].copy()
    q["has_filters"] = (~val.is_empty).astype(int)
    q.to_parquet(ART_DIR / f"{name}_queries.parquet", index=False)
    truth.to_parquet(ART_DIR / f"{name}_truth.parquet", index=False)
    corpus.to_parquet(ART_DIR / f"{name}_corpus.parquet", index=False)
    np.save(ART_DIR / f"{name}_fit_idx.npy", fit_idx)

    print(f"[{name}] запросов: {len(q)} (знакомых {q.seen_text.mean():.2f}, с фильтрами "
          f"{q.has_filters.mean():.2f}); релевантных на запрос {len(truth) / len(q):.2f}")
    print(f"[{name}] fit: {len(fit_idx)} строк; корпус: {len(corpus)} "
          f"(benchmark + {len(corpus) - len(bench_items)} отложенных)")
    in_fit = q.search_query.isin(set(train.search_query.iloc[fit_idx]))
    print(f"[{name}] проверка: знакомые тексты в fit {in_fit[q.seen_text == 1].mean():.2f}, "
          f"новые тексты в fit {in_fit[q.seen_text == 0].mean():.2f} (должно быть 0)")
    return fit_idx


def main():
    train = load_train()
    val_fit = make_split(train, np.arange(len(train)), "val", **SPLITS["val"])
    make_split(train, val_fit, "rank", **SPLITS["rank"])


if __name__ == "__main__":
    main()
