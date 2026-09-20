"""
Слияние кандидатов, выбираем из пула только 50 объявлений

каждый источник (BM25, логи, e5, …) находит свою часть нужных объявлений: вместе они
находят 95,8% (полнота пула на val)
Ответ ограничен 50 объявлениями, поэтому нужно правило, по которому из объединения
выбираются 50. Это не ранжирование для пользователя (порядок внутри 50 не важен и на метрику
не влияет)

  python -m src.fusion   # подобрать/обучить на rank, сравнить на val, сохранить в artifacts/
"""
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import ART_DIR, SEED
from .data import load_train
from .features import DENSE_SOURCES, POOL_SOURCES
from .metrics import Recall50, positions_in_query, recall50
from .pools import load_split

SOURCES = list(POOL_SOURCES)


def feature_cols(pool: pd.DataFrame, classic: bool = False):
    cols = [c for c in pool.columns if c not in {"query_id", "item_idx", "label"}]
    if classic:
        cols = [c for c in cols if not any(d in c for d in DENSE_SOURCES)]
    return cols


def restrict_classic(pool: pd.DataFrame) -> pd.DataFrame:
    """Пул без dense-источника: кандидаты, найденные хотя бы одним не-dense источником."""
    other = [f"in_{s}" for s in SOURCES if s not in DENSE_SOURCES]
    return pool[pool[other].sum(axis=1) > 0]


def source_key(pool: pd.DataFrame, src: str) -> np.ndarray:
    """Ключ источника — тот же, по которому он набирал кандидатов в пул (скор/макс + β·гео)."""
    return (pool[f"n_{src}"] + POOL_SOURCES[src] * pool.geo_logp).values


# RRF
class RRF:
    name = "rrf"

    def __init__(self, k: int = 60):
        self.k = k  # default k = 60

    def fit(self, pool, n_rel):
        return self  # обучения нет

    def score(self, pool: pd.DataFrame) -> np.ndarray:
        q = pool.query_id.values
        return sum(1.0 / (self.k + 1 + positions_in_query(q, source_key(pool, s))) for s in SOURCES)


# взвешенная сумма
class WeightedSum:
    name = "weighted"
    GRID = {**{s: [0, 0.25, 0.5, 1, 2] for s in SOURCES}, "geo": [0.03, 0.05, 0.1, 0.2, 0.3]}

    def __init__(self, weights: dict = None):
        self.w = weights

    def score(self, pool: pd.DataFrame) -> np.ndarray:
        s = sum(self.w[src] * pool[f"n_{src}"].values for src in SOURCES)
        return s + self.w["geo"] * pool.geo_logp.values

    def fit(self, pool: pd.DataFrame, n_rel: pd.Series):
        """Покоординатный перебор: по очереди меняем один вес, оставляем лучшее значение."""
        metric = Recall50(pool, n_rel)
        self.w = {**{s: 1.0 for s in SOURCES}, "geo": 0.1}  # старт: все источники равны
        best = metric(self.score(pool))
        for _ in range(3):  # три прохода по всем весам
            for p, values in self.GRID.items():
                for v in values:
                    old = self.w[p]
                    self.w[p] = v
                    r = metric(self.score(pool))
                    if r > best + 1e-5:
                        best = r
                    else:
                        self.w[p] = old
        return self

    def save(self):
        (ART_DIR / "fusion_weighted.json").write_text(json.dumps(self.w, indent=2))

    @classmethod
    def load(cls):
        return cls(json.loads((ART_DIR / "fusion_weighted.json").read_text()))


#  логистическая регрессия
class LogRegFusion:
    name = "logreg"

    def __init__(self, classic: bool = False):
        self.classic = classic

    def fit(self, pool: pd.DataFrame, n_rel=None):
        self.cols = feature_cols(pool, self.classic)
        # стандартизация: признаки в разных шкалах (км, BM25, доли), а после неё веса сравнимы
        self.model = make_pipeline(StandardScaler(),
                                   LogisticRegression(C=1.0, max_iter=1000, random_state=SEED))
        self.model.fit(pool[self.cols].values, pool.label.values)
        return self

    def score(self, pool: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(pool[self.cols].values)[:, 1]

    def coefficients(self) -> pd.Series:
        return pd.Series(self.model[-1].coef_[0], index=self.cols).sort_values(key=abs, ascending=False)

    def save(self):
        joblib.dump((self.cols, self.model), ART_DIR / "fusion_logreg.joblib")

    @classmethod
    def load(cls):
        obj = cls()
        obj.cols, obj.model = joblib.load(ART_DIR / "fusion_logreg.joblib")
        return obj


def load_fuser(name: str):
    return {"rrf": RRF, "weighted": WeightedSum.load, "logreg": LogRegFusion.load}[name]()


# сравнение
def main():
    train = load_train()
    n_rel = {}
    for name in ["rank", "val"]:
        _, _, _, rel = load_split(name, train)
        n_rel[name] = pd.Series({k: len(v) for k, v in rel.items()})
    p_rank = pd.read_parquet(ART_DIR / "pool_rank.parquet")
    p_val = pd.read_parquet(ART_DIR / "pool_val.parquet")
    res = {}

    res["0. лучший одиночный источник (BM25 + гео)"] = recall50(p_val, source_key(p_val, "bm25_lemma"), n_rel["val"])
    res["1. RRF"] = recall50(p_val, RRF().score(p_val), n_rel["val"])

    t = time.time()
    ws = WeightedSum().fit(p_rank, n_rel["rank"])
    ws.save()
    res["2. взвешенная сумма"] = recall50(p_val, ws.score(p_val), n_rel["val"])
    print(f"взвешенная сумма: веса подобраны за {time.time() - t:.0f} c: {ws.w}")

    t = time.time()
    lr = LogRegFusion().fit(p_rank)
    lr.save()
    res["3. логистическая регрессия"] = recall50(p_val, lr.score(p_val), n_rel["val"])
    print(f"логистическая регрессия: обучена за {time.time() - t:.0f} c; крупнейшие веса:")
    print("  " + ", ".join(f"{c}={v:+.2f}" for c, v in lr.coefficients().head(12).items()))

    # тот же лучший способ без dense-источника: сколько даёт нейросеть e5
    lr_c = LogRegFusion(classic=True).fit(restrict_classic(p_rank))
    pv_c = restrict_classic(p_val)
    res["3. логистическая регрессия без e5 (только классика)"] = recall50(pv_c, lr_c.score(pv_c), n_rel["val"])

    hit = p_val.groupby("query_id").label.sum()
    res["потолок: полнота пула"] = float((hit.reindex(n_rel["val"].index).fillna(0) / n_rel["val"]).mean())
    print("\nRecall@50 на val:")
    print(pd.Series(res).round(4).to_string())


if __name__ == "__main__":
    main()
