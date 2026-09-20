"""Источники кандидатов (retrievers) с единым интерфейсом

Каждый источник для запроса возвращает плотный вектор скоров по всему корпусу (np.ndarray
длины n_items). Так любой кандидат, откуда бы он ни пришёл, получает скоры ВСЕХ источников,
и на этапе ранжирования у каждого кандидата будет полный набор признаков

Источники:
  * bm25_stem / bm25_lemma - лексический поиск по полям (из bm25.py);
  * char - TF-IDF по символьным n-граммам заголовка: устойчив к опечаткам;
  * dense_* - косинус эмбеддингов (dense.py);
  * log_* - источники по логам train через похожие запросы (kNN по текстам запросов):
      log_items - объявления, которые выбирали по похожим запросам (если они есть в корпусе);
      log_mc - P(microcat | запрос): какие подкатегории выбирали по похожим запросам;

Гео - априорная вероятность GeoPrior, которая добавляется к скору
"""
from typing import Dict

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .bm25 import BM25Field, BM25MultiField

FIELDS = {"title": "item_title_raw", "params": "item_infm_params_text", "desc": "item_description_raw"}
BM25_WEIGHTS = {"title": 1.0, "params": 0.3, "desc": 0.3}


# гео
class GeoPrior:
    """ log P(item_loc | search_loc) по логам train, со сглаживанием """

    def __init__(self, floor: float = 1e-4):
        self.floor = floor

    def fit(self, train: pd.DataFrame, corpus: pd.DataFrame) -> "GeoPrior":
        self.loc_codes, self.item_code = np.unique(corpus.item_location_id.values, return_inverse=True)
        code_of = pd.Series(np.arange(len(self.loc_codes)), index=self.loc_codes)
        cnt = train.groupby(["search_location_id", "item_location_id"]).size()
        prob = cnt / cnt.groupby(level=0).transform("sum")
        self.table = {}
        for s, grp in prob.groupby(level=0):
            v = np.full(len(self.loc_codes), np.log(self.floor), np.float32)
            il = grp.index.get_level_values(1)
            known = il.isin(code_of.index)
            v[code_of[il[known]].values] = np.log(np.maximum(grp.values[known], self.floor))
            self.table[s] = v
        return self

    def logp(self, search_loc: int) -> np.ndarray:
        v = self.table.get(search_loc)
        if v is None:  # локации нет в train: разрешаем только её саму
            v = np.full(len(self.loc_codes), np.log(self.floor), np.float32)
        v = v.copy()
        own = np.searchsorted(self.loc_codes, search_loc)
        if own < len(self.loc_codes) and self.loc_codes[own] == search_loc:
            v[own] = max(v[own], np.log(0.5))  # своя локация — всегда высокий приоритет
        return v[self.item_code]


# лексика
class CharScorer:
    def fit(self, corpus: pd.DataFrame) -> "CharScorer":
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                                   sublinear_tf=True, dtype=np.float32, lowercase=True)
        self.X = self.vec.fit_transform(corpus.item_title_raw.str.replace("ё", "е")).T.tocsr()
        return self

    def score(self, query: str) -> np.ndarray:
        q = self.vec.transform([query.replace("ё", "е")])
        return np.asarray((q @ self.X).todense()).ravel()


# логи
class LogScorer:
    """kNN по текстам запросов train: «что выбирали люди, искавшие похожее».

    1. Тексты запросов fit-train индексируются char-TF-IDF; для запроса берём top-M соседей
       с весами w_j = sim_j^power (точное совпадение текста = sim 1).
    2. По соседям агрегируем три матрицы «текст запроса × …»:
       выбранные объявления, их microcat, слова их заголовков.
    """

    def __init__(self, n_neighbors: int = 30, power: float = 3.0, n_expand_terms: int = 30):
        self.m, self.power, self.n_terms = n_neighbors, power, n_expand_terms

    def fit(self, train: pd.DataFrame, corpus: pd.DataFrame, title_bm25: BM25Field) -> "LogScorer":
        texts, t_idx = np.unique(train.search_query.values, return_inverse=True)
        self.texts = texts
        self.qvec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True,
                                    dtype=np.float32)
        self.Q = self.qvec.fit_transform(texts).T.tocsr()
        n_t = len(texts)
        ones = np.ones(len(train), np.float32)

        # тексты × объявления корпуса (только те объявления train, что есть в корпусе)
        pos = pd.Series(np.arange(len(corpus)), index=corpus.item_id.values)
        ip = pos.reindex(train.item_id.values).values
        ok = ~np.isnan(ip)
        self.T_item = sp.csr_matrix((ones[ok], (t_idx[ok], ip[ok].astype(int))), shape=(n_t, len(corpus)))

        # тексты × microcat
        self.mc_codes, mc_idx = np.unique(
            np.r_[train.item_microcat_id.values, corpus.item_microcat_id.values], return_inverse=True)
        self.item_mc = mc_idx[len(train):]
        self.T_mc = sp.csr_matrix((ones, (t_idx, mc_idx[:len(train)])), shape=(n_t, len(self.mc_codes)))

        # тексты × слова заголовков выбранных объявлений (в словаре BM25-индекса заголовков)
        self.title_bm25 = title_bm25
        tv = title_bm25.vec.transform(train.item_title_raw.values)  # строки train × слова
        tv.data[:] = 1.0
        agg = sp.csr_matrix((ones, (t_idx, np.arange(len(train)))), shape=(n_t, len(train)))
        T_term = (agg @ tv).tocsr()
        # TF-IDF-подобное взвешивание: общие слова («услуги», «ремонт») важны меньше
        T_term = T_term.multiply(title_bm25.idf[None, :]).tocsr()
        self.T_term = normalize(T_term, norm="l1")
        return self

    def neighbors(self, query: str):
        sims = np.asarray((self.qvec.transform([query]) @ self.Q).todense()).ravel()
        top = np.argpartition(-sims, self.m)[: self.m]
        top = top[sims[top] > 0.3]  # отсекаем совсем непохожие тексты
        w = sims[top] ** self.power
        return top, w

    def scores(self, query: str) -> Dict[str, np.ndarray]:
        top, w = self.neighbors(query)
        n_items = self.T_item.shape[1]
        if len(top) == 0:
            z = np.zeros(n_items, np.float32)
            return {"log_items": z, "log_mc": z, "log_expand": z, "_log_maxsim": 0.0, "_log_n": 0}
        # взвешенная сумма строк соседей: (1 × M) @ (M × …)
        W = sp.csr_matrix(w[None, :].astype(np.float32))
        items = (W @ self.T_item[top]).toarray().ravel()
        mc = (W @ self.T_mc[top]).toarray().ravel()
        mc = mc / mc.sum()
        terms = (W @ self.T_term[top]).toarray().ravel()
        # оставляем самые весомые слова, включая слова самого запроса
        keep = np.argsort(-terms)[: self.n_terms]
        keep = keep[terms[keep] > 0]
        expand = self.title_bm25.W[:, keep] @ (terms[keep] / terms[keep].max())
        return {"log_items": items.astype(np.float32),
                "log_mc": mc[self.item_mc].astype(np.float32),
                "log_expand": np.asarray(expand).ravel().astype(np.float32),
                # скаляры запроса (ключи с «_» — не источники): насколько запрос «знаком» логам
                "_log_maxsim": float(w.max() ** (1 / self.power)), "_log_n": len(top)}


# dense
class DenseScorer:
    def __init__(self, key: str):
        self.key = key

    def fit(self, corpus: pd.DataFrame) -> "DenseScorer":
        from .dense import load_corpus_emb, load_model
        self.E = load_corpus_emb(self.key, corpus.item_id.values)
        self.model = load_model(self.key)
        return self

    def encode(self, queries) -> np.ndarray:
        from .dense import encode_queries
        return encode_queries(self.key, list(queries), self.model)

    def score_vec(self, qv: np.ndarray) -> np.ndarray:
        return self.E @ qv


# всё вместе
class Retriever:
    """Собирает все источники; для запроса возвращает {имя_источника: вектор скоров}."""

    def __init__(self, dense_keys=("e5",), use_lemma=True):
        self.dense_keys, self.use_lemma = dense_keys, use_lemma

    def fit(self, train: pd.DataFrame, corpus: pd.DataFrame, cache: str = None,
            verbose=True) -> "Retriever":
        """cache — имя файла в artifacts/ для лексических индексов корпуса (BM25, char)

        Индексы зависят только от корпуса, а их построение (лемматизация описаний) занимает
        минуты, поэтому кешируем. Логи и гео зависят от train и строятся заново
        """
        import time
        import joblib
        from .config import ART_DIR
        t = time.time()
        path = ART_DIR / f"lex_{cache}.joblib" if cache else None
        if path is not None and path.exists():
            self.bm25, self.char = joblib.load(path)
        else:
            df = pd.DataFrame({f: corpus[c] for f, c in FIELDS.items()})
            self.bm25 = {"stem": BM25MultiField(BM25_WEIGHTS, tokenizer="stem").fit(df)}
            if self.use_lemma:
                self.bm25["lemma"] = BM25MultiField(BM25_WEIGHTS, tokenizer="lemma").fit(df)
            self.char = CharScorer().fit(corpus)
            if path is not None:
                joblib.dump((self.bm25, self.char), path)
        verbose and print(f"  BM25 + char: {time.time() - t:.0f} c")
        self.log = LogScorer().fit(train, corpus, self.bm25["stem"].fields["title"])
        verbose and print(f"  logs: {time.time() - t:.0f} c")
        self.geo = GeoPrior().fit(train, corpus)
        self.dense = {k: DenseScorer(k).fit(corpus) for k in self.dense_keys}
        verbose and print(f"  dense: {time.time() - t:.0f} c")
        return self

    def encode_queries(self, queries: pd.DataFrame) -> Dict[str, np.ndarray]:
        return {k: d.encode(queries.search_query) for k, d in self.dense.items()}

    def score_all(self, q, qemb: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        text = q.search_query
        out = {"bm25_stem": self.bm25["stem"].score(text)}
        if "lemma" in self.bm25:
            # поля по отдельности — полезные признаки для ранкера, их взвешенная сумма — источник
            per_field = {f: m.score(text) for f, m in self.bm25["lemma"].fields.items()}
            out["bm25_lemma"] = sum(BM25_WEIGHTS[f] * v for f, v in per_field.items())
            out.update({f"bm25_lemma_{f}": v for f, v in per_field.items()})
        out["bm25_title"] = self.bm25["stem"].fields["title"].score(text)
        out["char"] = self.char.score(text)
        out.update(self.log.scores(text))
        for k, d in self.dense.items():
            out[f"dense_{k}"] = d.score_vec(qemb[k])
        out["geo"] = self.geo.logp(q.search_location_id)
        return out
