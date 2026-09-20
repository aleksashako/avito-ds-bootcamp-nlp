"""baseline: BM25 по полям объявления + гео-фильтр

Логика отбора top-K для запроса:
1. считаем BM25 запроса по всему корпусу
2. сначала берём объявления из «разрешённых» локаций с ненулевым BM25 (по убыванию скора)
3. если их меньше K, добираем совпадения из остальных локаций

Запуск:
  python -m src.baseline  # сравнение вариантов на валидации
  python -m src.baseline --submit   # answer.csv для benchmark лучшим вариантом
"""
import argparse

import numpy as np
import pandas as pd

from .bm25 import BM25MultiField
from .config import ART_DIR, ROOT, TOP_K
from .data import load_bench_items, load_bench_queries, load_train
from .geo import LocationModel
from .metrics import recall_by_slice, recall_curve, recall_per_query
from .submission import validate_answer, write_answer

FIELDS = {"title": "item_title_raw", "params": "item_infm_params_text", "desc": "item_description_raw"}

# варианты весов полей (title, inf_params, desc)
FIELD_VARIANTS = {
    "title": (1.0, 0.0, 0.0),
    "title+params": (1.0, 0.3, 0.0),
    "title+params+desc": (1.0, 0.3, 0.3),
}
GEO_VARIANTS = ["none", "exact", "model"]
BEST = ("title+params+desc", "model")  

def top_k(score: np.ndarray, allowed: np.ndarray, k: int) -> np.ndarray:
    """ сначала совпадения в разрешённых локациях, потом остальные совпадения"""
    key = score + 1e4 * (allowed & (score > 0))
    k = min(k, len(key))
    idx = np.argpartition(-key, k - 1)[:k]
    idx = idx[np.argsort(-key[idx])]
    return idx[score[idx] > 0]

def build_index(items: pd.DataFrame) -> BM25MultiField:
    df = pd.DataFrame({f: items[c] for f, c in FIELDS.items()})
    index = BM25MultiField({f: 1.0 for f in FIELDS}).fit(df)
    return index

def retrieve(queries, items, index, geo, variants, depth):
    """ для каждого варианта возвращает {query_id: [item_id…]} глубины depth """
    item_ids = items.item_id.values
    item_loc = items.item_location_id.values
    preds = {v: {} for v in variants}
    for q in queries.itertuples():
        per_field = {f: index.fields[f].score(q.search_query) for f in FIELDS}
        masks = {
            "none": np.zeros(len(items), bool),
            "exact": item_loc == q.search_location_id,
            "model": np.isin(item_loc, list(geo.allowed(q.search_location_id))),
        }
        for fv, gv in variants:
            w = dict(zip(FIELDS, FIELD_VARIANTS[fv]))
            score = sum(w[f] * per_field[f] for f in FIELDS if w[f])
            preds[(fv, gv)][q.query_id] = item_ids[top_k(score, masks[gv], depth)].tolist()
    return preds

def run_validation():
    train = load_train()
    fit = train.iloc[np.load(ART_DIR / "val_fit_idx.npy")]
    vq = pd.read_parquet(ART_DIR / "val_queries.parquet")
    truth = pd.read_parquet(ART_DIR / "val_truth.parquet").groupby("query_id").item_id.agg(set).to_dict()
    corpus = pd.read_parquet(ART_DIR / "val_corpus.parquet")

    index = build_index(corpus)
    geo = LocationModel().fit(fit)
    variants = [(f, g) for f in FIELD_VARIANTS for g in GEO_VARIANTS]
    preds = retrieve(vq, corpus, index, geo, variants, depth=1000)

    rows = {f"{f} | geo={g}": recall_curve(preds[(f, g)], truth, [50, 1000]) for f, g in variants}
    print("\nRecall@K на валидации:")
    print(pd.DataFrame(rows).T.round(4).to_string())

    # срезы для лучшего варианта
    locsize = corpus.item_location_id.value_counts()
    vq["loc_items"] = pd.cut(vq.search_location_id.map(locsize).fillna(0),
                             [-1, 0, 500, 3000, 1e9], labels=["0", "1-500", "500-3k", ">3k"])
    per_q = recall_per_query(preds[BEST], truth, TOP_K)
    print(f"\nСрезы для {BEST}:")
    print(recall_by_slice(per_q, vq, ["seen_text", "has_filters", "loc_items"]).to_string())

def make_submission():
    train = load_train()
    bq, items = load_bench_queries(), load_bench_items()
    index = build_index(items)
    geo = LocationModel().fit(train) 
    preds = retrieve(bq, items, index, geo, [BEST], depth=TOP_K)[BEST]
    # запросы, где BM25 не нашёл ничего: такие строки остаются пустыми или короткими
    path = ROOT / "answer.csv"
    write_answer(preds, bq.query_id.tolist(), path)
    validate_answer(path, bq, items)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", action="store_true")
    args = ap.parse_args()
    make_submission() if args.submit else run_validation()
