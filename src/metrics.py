"""
ВЫчисление метрики Recall@K 
Для каждого запроса: |top-K ∩ релевантные| / |релевантные|, затем среднее по запросам

порядок внутри не важен
"""

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from .config import TOP_K


def recall_per_query(preds: Dict[str, Sequence[str]], truth: Dict[str, set], k: int = 50) -> pd.Series:
    """ Recall@k для каждого запроса """
    out = {}
    for qid, rel in truth.items():
        top = set(list(preds.get(qid, []))[:k])
        out[qid] = len(top & rel) / len(rel)
    return pd.Series(out, name=f"recall@{k}")


def recall_at_k(preds, truth, k: int = 50) -> float:
    return float(recall_per_query(preds, truth, k).mean())


def recall_curve(preds, truth, ks: List[int] = (10, 50, 100, 200, 500, 1000)) -> pd.Series:
    """Recall на разных глубинах: показывает потолок источника кандидатов

    Если на глубине 1000 recall высокий, а на 50 низкий, значит нужное объявление
    находится, но плохо ранжируется, и это лечится переранжированием.
    Если низкий и на 1000, источник его просто не видит.
    """
    return pd.Series({k: recall_at_k(preds, truth, k) for k in ks}, name="recall")


def recall_by_slice(per_query: pd.Series, queries: pd.DataFrame, slice_cols: List[str]) -> pd.DataFrame:
    """ Средний recall по срезам запросов """
    df = queries.set_index("query_id")[slice_cols].join(per_query)
    res = []
    for c in slice_cols:
        g = df.groupby(c, observed=True)[per_query.name].agg(["mean", "size"])
        g.index = [f"{c}={v}" for v in g.index]
        res.append(g)
    return pd.concat(res).round(4)


# --- те же вычисления, но по «пулу кандидатов» (таблица строк "запрос - объявление"),
# как на шаге слияния: там метрика пересчитывается сотни раз, поэтому считается на numpy.
def positions_in_query(query_ids: np.ndarray, score: np.ndarray, tie: np.ndarray = None) -> np.ndarray:
    """Место каждой строки пула внутри своего запроса при сортировке по убыванию score (0 = лучшее).

    query_ids — id запросов (строки) или уже готовые целочисленные коды запросов.
    tie — дополнительный ключ для одинаковых скоров (по умолчанию порядок строк), чтобы
    результат был детерминированным.
    """
    codes = query_ids if np.issubdtype(query_ids.dtype, np.integer) else pd.factorize(query_ids)[0]
    tie = np.arange(len(score)) if tie is None else tie
    order = np.lexsort((tie, -score, codes))
    c = codes[order]
    starts = np.r_[0, np.flatnonzero(np.diff(c)) + 1]
    pos_sorted = np.arange(len(c)) - np.repeat(starts, np.diff(np.r_[starts, len(c)]))
    pos = np.empty(len(c), np.int64)
    pos[order] = pos_sorted
    return pos


class Recall50:
    """Recall@50 по пулу; знаменатель - все нужные объявления запроса, включая не попавшие в пул

    Коды запросов считаются один раз при создании
    """

    def __init__(self, pool: pd.DataFrame, n_rel: pd.Series):
        self.codes, uniq = pd.factorize(pool.query_id.values)
        self.label = pool.label.values
        self.denom = n_rel.reindex(uniq).values          # нужных объявлений у запроса
        self.n_missing = len(n_rel) - len(uniq)           # запросы без строк в пуле: recall 0

    def __call__(self, score: np.ndarray) -> float:
        top = positions_in_query(self.codes, score) < TOP_K
        hit = np.bincount(self.codes, weights=self.label * top, minlength=len(self.denom))
        return float((hit / self.denom).sum() / (len(self.denom) + self.n_missing))


def recall50(pool: pd.DataFrame, score: np.ndarray, n_rel: pd.Series) -> float:
    return Recall50(pool, n_rel)(score)
