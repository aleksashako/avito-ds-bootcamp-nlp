"""Пул кандидатов и признаки для ранкера.

Пул запроса = объединение top-N каждого источника (ключ источника: s/max(s) + β·logP(гео),
β подобран на валидации в eval_sources). Для каждого кандидата собираем признаки:

  источники (для всех, а не только для «своего» источника): сырой скор, скор/максимум по
      запросу, флаг «попал в top-N источника», число источников, в top-N которых попал;
  BM25 по полям (title / params / description) и доля слов запроса, найденных в заголовке
      и в описании;
  фильтры поиска: BM25 текста фильтров (search_infm_params_text) по параметрам объявления —
      насколько объявление подходит под «Вид услуги …», «Тип услуги …»;
  гео: log P(item_loc | search_loc), та же ли локация, расстояние в км до центра локации
      поиска, «локальность» подкатегории (доля выборов в своей локации для этой microcat:
      сантехника ищут рядом, копирайтера — где угодно);
  объявление: рейтинг, число отзывов, цена, скрыт ли телефон, запрещены ли сообщения,
      категория, длина текстов, популярность объявления и его подкатегории в логах;
  запрос: число слов, есть ли фильтры, категория 0, похожесть ближайшего запроса из логов.
"""
import time
from typing import Dict

import numpy as np
import pandas as pd

from .sources import Retriever

# источник -> β (сила гео-приора), подобрано в eval_sources
POOL_SOURCES = {
    "bm25_lemma": 0.1, "log_expand": 0.1, "dense_e5": 0.03, "char": 0.1,
    "bm25_title": 0.1, "log_mc": 0.3, "log_items": 0.03,
}
N_PER_SOURCE = 100
DENSE_SOURCES = {"dense_e5"}


class ItemStats:
    """Статичные признаки объявлений корпуса и статистики по логам train."""

    def fit(self, train: pd.DataFrame, corpus: pd.DataFrame) -> "ItemStats":
        c = corpus
        f = pd.DataFrame(index=c.index)
        f["it_rating"] = c.item_rating.fillna(-1)
        f["it_reviews_log"] = np.log1p(c.item_rating_reviews_count.fillna(0))
        f["it_price_log"] = np.log1p(c.item_price.fillna(-1).clip(lower=0))
        f["it_phone_hidden"] = c.item_is_phone_hidden.astype(np.float32)
        f["it_msg_forbidden"] = c.item_is_message_forbidden.astype(np.float32)
        f["it_cat114"] = (c.item_category_id == 114).astype(np.float32)
        f["it_title_len"] = c.item_title_raw.str.len()
        f["it_desc_len_log"] = np.log1p(c.item_description_raw.str.len())
        # популярность объявления в логах (для большинства объявлений корпуса = 0)
        f["it_clicks_log"] = np.log1p(c.item_id.map(train.item_id.value_counts()).fillna(0))
        # популярность подкатегории и её «локальность»
        mc_share = train.item_microcat_id.value_counts(normalize=True)
        f["mc_share"] = c.item_microcat_id.map(mc_share).fillna(0)
        same = (train.search_location_id == train.item_location_id).groupby(train.item_microcat_id).mean()
        f["mc_local"] = c.item_microcat_id.map(same).fillna(same.mean())
        self.item_feats = f.astype(np.float32)

        # координаты: центр каждой локации = медиана координат её объявлений (train + корпус)
        coords = pd.concat([
            train[["item_location_id", "item_latitude", "item_longitude"]],
            c[["item_location_id", "item_latitude", "item_longitude"]],
        ]).dropna().groupby("item_location_id").median()
        self.loc_center = coords
        # для «региональных» локаций поиска (без своих объявлений) — центр самой частой
        # локации объявлений, которые из неё выбирали
        top_item_loc = (train.groupby("search_location_id").item_location_id
                        .agg(lambda s: s.value_counts().index[0]))
        self.search_center = {}
        for s, il in top_item_loc.items():
            src = s if s in coords.index else il
            if src in coords.index:
                self.search_center[s] = coords.loc[src].values
        self.lat = c.item_latitude.values.astype(np.float64)
        self.lon = c.item_longitude.values.astype(np.float64)
        self.item_loc = c.item_location_id.values
        return self

    def dist_km(self, search_loc: int, idx: np.ndarray) -> np.ndarray:
        cen = self.search_center.get(search_loc)
        if cen is None and search_loc in self.loc_center.index:
            cen = self.loc_center.loc[search_loc].values
        if cen is None:
            return np.full(len(idx), -1, np.float32)
        lat, lon = self.lat[idx], self.lon[idx]
        d = np.hypot(lat - cen[0], (lon - cen[1]) * np.cos(np.radians(cen[0]))) * 111.0
        return np.where(np.isnan(d), -1, d).astype(np.float32)


def _top(key: np.ndarray, n: int) -> np.ndarray:
    top = np.argpartition(-key, n)[:n]
    return top[np.argsort(-key[top])]


def build_pool(R: Retriever, stats: ItemStats, queries: pd.DataFrame,
               truth: Dict[str, set] = None, verbose=True) -> pd.DataFrame:
    """Пул кандидатов с признаками для всех запросов. truth: query_id -> {индексы объявлений}."""
    qemb = R.encode_queries(queries)
    lem_fields = R.bm25["lemma"].fields
    frames = []
    t = time.time()
    for i, q in enumerate(queries.itertuples()):
        sc = R.score_all(q, {k: v[i] for k, v in qemb.items()})
        geo = sc.pop("geo")
        scal = {k: sc.pop(k) for k in list(sc) if k.startswith("_")}

        # --- пул: объединение top-N источников
        in_top = {}
        for name, beta in POOL_SOURCES.items():
            s = sc[name]
            m = s.max()
            key = (s / m if m > 0 else s) + beta * geo
            in_top[name] = _top(key, N_PER_SOURCE)
        idx = np.unique(np.concatenate(list(in_top.values())))

        f = {"item_idx": idx}
        for name, s in sc.items():
            v = s[idx]
            m = s.max()
            f[f"s_{name}"] = v
            f[f"n_{name}"] = v / m if m > 0 else v
        for name, top in in_top.items():
            f[f"in_{name}"] = np.isin(idx, top).astype(np.float32)
        f["n_sources"] = sum(f[f"in_{n}"] for n in POOL_SOURCES)

        # --- доля слов запроса в заголовке и в описании
        for fld in ("title", "desc"):
            ids = lem_fields[fld].term_ids(q.search_query)
            if ids:
                hit = (lem_fields[fld].W[:, ids][idx] > 0).sum(axis=1)
                f[f"cover_{fld}"] = np.asarray(hit).ravel() / len(ids)
            else:
                f[f"cover_{fld}"] = np.zeros(len(idx))
        # --- фильтры поиска против параметров объявления
        f["filter_bm25"] = (lem_fields["params"].score(q.search_infm_params_text)[idx]
                            if q.search_infm_params_text else np.zeros(len(idx)))
        # --- гео
        f["geo_logp"] = geo[idx]
        f["same_loc"] = (stats.item_loc[idx] == q.search_location_id).astype(np.float32)
        f["dist_km"] = stats.dist_km(q.search_location_id, idx)
        # --- запрос
        f["q_words"] = len(q.search_query.split())
        f["q_has_filters"] = float(bool(q.search_infm_params_text))
        f["q_cat0"] = float(q.search_category == 0)
        f["q_log_maxsim"] = scal["_log_maxsim"]
        f["q_log_n"] = scal["_log_n"]
        f["pool_size"] = len(idx)

        df = pd.DataFrame(f)
        df.insert(0, "query_id", q.query_id)
        if truth is not None:
            df["label"] = np.isin(idx, list(truth.get(q.query_id, ()))).astype(np.int8)
        frames.append(df)
        if verbose and i % 1000 == 0:
            print(f"  пул: {i}/{len(queries)} запросов, {time.time() - t:.0f} c")

    pool = pd.concat(frames, ignore_index=True)
    item = stats.item_feats.iloc[pool.item_idx.values].reset_index(drop=True)
    pool = pd.concat([pool, item], axis=1)
    num = pool.columns.difference(["query_id", "item_idx", "label"])
    pool[num] = pool[num].astype(np.float32)
    return pool
