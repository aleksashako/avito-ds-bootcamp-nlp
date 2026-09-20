"""
Реализация Okapi BM25 (Best Match 25)
"""
from typing import Dict, List

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

from .text import TOKENIZERS


class BM25Field:
    def __init__(self, k1: float = 1.2, b: float = 0.75, tokenizer: str = "stem"):
        self.k1, self.b = k1, b
        self.tokenize = TOKENIZERS[tokenizer]

    def fit(self, texts: List[str]) -> "BM25Field":
        # analyzer=tokenize: нормализация вместо встроенной в sklearn
        self.vec = CountVectorizer(analyzer=self.tokenize, dtype=np.float32)
        tf = self.vec.fit_transform(texts).tocsr()
        n_docs = tf.shape[0]
        dl = np.asarray(tf.sum(axis=1)).ravel()
        avgdl = dl.mean()
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        self.idf = np.log1p((n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)

        # BM25-насыщение tf с нормировкой на длину документа, построчно
        norm = self.k1 * (1 - self.b + self.b * dl / avgdl)
        rows = np.repeat(np.arange(n_docs), np.diff(tf.indptr))
        data = tf.data * (self.k1 + 1) / (tf.data + norm[rows]) * self.idf[tf.indices]

        self.W = sp.csr_matrix((data.astype(np.float32), tf.indices, tf.indptr), shape=tf.shape).tocsc()
        self.vocab = self.vec.vocabulary_
        return self

    def term_ids(self, query: str) -> List[int]:
        return sorted({self.vocab[t] for t in self.tokenize(query) if t in self.vocab})

    def score(self, query: str) -> np.ndarray:
        ids = self.term_ids(query)
        if not ids:
            return np.zeros(self.W.shape[0], dtype=np.float32)
        return np.asarray(self.W[:, ids].sum(axis=1)).ravel()


class BM25MultiField:
    def __init__(self, weights: Dict[str, float], k1: float = 1.2, b: float = 0.75,
                 tokenizer: str = "stem"):
        self.weights = weights
        self.fields = {f: BM25Field(k1, b, tokenizer) for f in weights}

    def fit(self, items) -> "BM25MultiField":
        for f, m in self.fields.items():
            m.fit(items[f].tolist())
        return self

    def score(self, query: str, weights: Dict[str, float] = None) -> np.ndarray:
        weights = weights or self.weights
        return sum(w * self.fields[f].score(query) for f, w in weights.items() if w)
