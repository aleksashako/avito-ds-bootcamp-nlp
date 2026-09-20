"""Оценка источников кандидатов на валидации

Для каждого источника s и силы гео-приора бета ранжируем корпус по
    key = s / max(s) + β · log P(item_loc | search_loc)
(нормировка на максимум делает β сравнимым между источниками) и считаем:
  * recall@50  — насколько хорош источник сам по себе;
  * recall@300 — сколько нужных объявлений он приносит в пул для ранкера.
Потом объединяем top-N лучших конфигураций всех источников и смотрим полноту
объединения: это потолок, которого может достичь ранкер, выбирая 50 из пула.

Запуск: python -m src.eval_sources [dense-ключи через запятую]
"""
import sys
import time

import numpy as np
import pandas as pd

from .config import ART_DIR
from .data import load_train
from .sources import Retriever

BETAS = [0.0, 0.03, 0.1, 0.3, 1.0]
DEPTHS = [50, 100, 300]


def main(dense_keys):
    train = load_train()
    fit = train.iloc[np.load(ART_DIR / "val_fit_idx.npy")]
    vq = pd.read_parquet(ART_DIR / "val_queries.parquet")
    truth = pd.read_parquet(ART_DIR / "val_truth.parquet")
    corpus = pd.read_parquet(ART_DIR / "val_corpus.parquet")
    pos = pd.Series(np.arange(len(corpus)), index=corpus.item_id.values)
    rel = truth.assign(p=pos.loc[truth.item_id].values).groupby("query_id").p.agg(list).to_dict()

    print("строим источники…")
    R = Retriever(dense_keys=dense_keys).fit(fit, corpus, cache="val")
    qemb = R.encode_queries(vq)

    ranks = {} # (source, β) -> список рангов релевантных объявлений (по всем запросам)
    tops = {} # (source, β) -> {query_id: top-300 индексов}
    t = time.time()
    for i, q in enumerate(vq.itertuples()):
        sc = R.score_all(q, {k: v[i] for k, v in qemb.items()})
        geo = sc.pop("geo")
        r = np.array(rel[q.query_id])
        for name, s in sc.items():
            if name.startswith("_") or name.startswith("bm25_lemma_"):
                continue  # скаляры запроса и отдельные поля BM25 — не самостоятельные источники
            m = s.max()
            s = s / m if m > 0 else s
            for b in BETAS:
                key = s + b * geo
                # пессимистичный ранг: при равенстве скоров считаем, что объявление ниже всех
                # «соседей по скору» (иначе объявление со скором 0 среди тысяч нулей
                # получило бы ранг 0 и завысило recall)
                ranks.setdefault((name, b), []).extend(((key[None, :] >= key[r, None]).sum(1) - 1).tolist())
                top = np.argpartition(-key, 300)[:300]
                tops.setdefault((name, b), {})[q.query_id] = top[np.argsort(-key[top])].astype(np.int32)
        if i % 500 == 0:
            print(f"  {i}/{len(vq)} запросов, {time.time() - t:.0f} c")

    # recall по отдельным (объявление, запрос) нельзя усреднять напрямую — пересчитываем по запросам
    n_rel = np.array([len(rel[q]) for q in vq.query_id])
    qid_of_pair = np.repeat(np.arange(len(vq)), n_rel)

    def recall(rk, k):
        hit = np.bincount(qid_of_pair, weights=(np.array(rk) < k), minlength=len(vq))
        return float(np.mean(hit / n_rel))

    rows = []
    for (name, b), rk in ranks.items():
        rows.append(dict(source=name, beta=b, **{f"R@{k}": recall(rk, k) for k in DEPTHS}))
    res = pd.DataFrame(rows)
    res.to_csv(ART_DIR / "eval_sources.csv", index=False)
    best = res.loc[res.groupby("source")["R@300"].idxmax()].sort_values("R@50", ascending=False)
    print("\nЛучший β для каждого источника (по R@300):")
    print(best.round(4).to_string(index=False))
    print("\nВлияние β (R@50):")
    print(res.pivot(index="source", columns="beta", values="R@50").round(4).to_string())

    # --- полнота объединения источников
    print("\nПолнота объединения top-N лучших конфигураций:")
    cfg = list(best[["source", "beta"]].itertuples(index=False, name=None))
    for n in [50, 100, 200, 300]:
        hits, sizes = [], []
        for q in vq.query_id:
            u = set()
            for c in cfg:
                u.update(tops[c][q][:n].tolist())
            r = rel[q]
            hits.append(np.mean([x in u for x in r]))
            sizes.append(len(u))
        print(f"  top-{n} от каждого: recall={np.mean(hits):.4f}, средний размер пула={np.mean(sizes):.0f}")

    # вклад каждого источника: насколько падает полнота пула top-100, если его убрать
    print("\nУникальный вклад источника (падение recall пула top-100 без него):")
    base = None
    for drop in [None] + cfg:
        hits = []
        for q in vq.query_id:
            u = set()
            for c in cfg:
                if c != drop:
                    u.update(tops[c][q][:100].tolist())
            hits.append(np.mean([x in u for x in rel[q]]))
        v = np.mean(hits)
        if drop is None:
            base = v
        else:
            print(f"  без {drop[0]:12s}: {v:.4f} (−{base - v:.4f})")


if __name__ == "__main__":
    keys = tuple(sys.argv[1].split(",")) if len(sys.argv) > 1 and sys.argv[1] else ()
    main(keys)
