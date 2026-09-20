"""
Пулы кандидатов для разбиений val и rank

  python -m src.pools

Пулы нужны на шаге слияния (fusion.py):
  * rank: на нём подбираются веса взвешенной суммы и обучается логистическая регрессия.
    Логи для rank-запросов посчитаны по rank-fit, из которого убраны сами эти запросы,
    поэтому условия такие же, как на benchmark;
  * val: на нём сравниваются способы слияния (он не участвует ни в каком подборе)
"""
import time

import numpy as np
import pandas as pd

from .config import ART_DIR
from .data import load_train
from .features import ItemStats, build_pool
from .sources import Retriever


def load_split(name: str, train: pd.DataFrame):
    """fit-часть train, запросы, корпус и {query_id: множество индексов нужных объявлений}."""
    fit = train.iloc[np.load(ART_DIR / f"{name}_fit_idx.npy")]
    q = pd.read_parquet(ART_DIR / f"{name}_queries.parquet")
    corpus = pd.read_parquet(ART_DIR / f"{name}_corpus.parquet")
    truth = pd.read_parquet(ART_DIR / f"{name}_truth.parquet")
    pos = pd.Series(np.arange(len(corpus)), index=corpus.item_id.values)
    rel = truth.assign(p=pos.loc[truth.item_id].values).groupby("query_id").p.agg(set).to_dict()
    return fit, q, corpus, rel


def build_pools():
    train = load_train()
    for name in ["val", "rank"]:
        t = time.time()
        fit, q, corpus, rel = load_split(name, train)
        print(f"[{name}] строим источники…")
        R = Retriever().fit(fit, corpus, cache=name)
        stats = ItemStats().fit(fit, corpus)
        pool = build_pool(R, stats, q, rel)
        pool.to_parquet(ART_DIR / f"pool_{name}.parquet", index=False)
        hit = pool.groupby("query_id").label.sum()
        n_rel = pd.Series({k: len(v) for k, v in rel.items()})
        print(f"[{name}] пул: {len(pool)} строк, {len(pool) / len(q):.0f} на запрос, "
              f"полнота пула {(hit.reindex(n_rel.index).fillna(0) / n_rel).mean():.4f}, "
              f"{time.time() - t:.0f} c")

if __name__ == "__main__":
    build_pools()