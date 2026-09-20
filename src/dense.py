"""Dense-источник: эмбеддинги готовыми open-source энкодерами (без дообучения)

Модели (обе обучены под поиск «короткий запрос - документ» и требуют префиксов):
  * intfloat/multilingual-e5-small - многоязычная, 118 млн параметров, префиксы
    "query: " / "passage: "
  * deepvk/USER2-small - русскоязычная (на основе RuModernBERT-small), 34 млн параметров,
    префиксы "search_query: " / "search_document: "
Веса скачиваются с HuggingFace один раз и дальше используются локально

Эмбеддинги кешируются в artifacts/ (см. ensure_corpus_emb): каждое объявление кодируется
один раз и используется во всех разбиениях и в сабмите.
"""
import sys
import time

import numpy as np
import pandas as pd

from .config import ART_DIR

MODELS = {
    "e5": dict(name="intfloat/multilingual-e5-small", q_prefix="query: ", d_prefix="passage: "),
    "user2": dict(name="deepvk/USER2-small", q_prefix="search_query: ", d_prefix="search_document: "),
}
MAX_LEN = 128


def item_text(items: pd.DataFrame) -> list:
    return (items.item_title_raw + ". " + items.item_infm_params_text.str.slice(0, 150) + ". "
            + items.item_description_raw.str.slice(0, 300)).tolist()


def load_model(key: str):
    import torch
    from sentence_transformers import SentenceTransformer
    torch.manual_seed(0)
    try:
        # сначала строго из локального кеша: при повторных запусках никаких сетевых обращений
        m = SentenceTransformer(MODELS[key]["name"], device="cpu", local_files_only=True)
    except Exception:
        # первый запуск: однократно скачиваем открытые веса с HuggingFace
        m = SentenceTransformer(MODELS[key]["name"], device="cpu")
    m.max_seq_length = MAX_LEN
    return m


def encode_queries(key: str, texts: list, model=None) -> np.ndarray:
    model = model or load_model(key)
    texts = [MODELS[key]["q_prefix"] + t for t in texts]
    return model.encode(texts, batch_size=256, normalize_embeddings=True,
                        convert_to_numpy=True).astype(np.float32)


def ensure_corpus_emb(key: str, corpus: pd.DataFrame):
    """ 
    Досчитывает эмбеддинги объявлений корпуса, которых ещё нет в кеше artifacts/emb_{key}.npy.
    Кеш общий для всех корпусов (val, rank, benchmark): эмбеддинг объявления не зависит
    от остального корпуса, поэтому каждое объявление кодируется ровно один раз
    """
    emb_path, ids_path = ART_DIR / f"emb_{key}.npy", ART_DIR / f"emb_{key}_ids.parquet"
    if emb_path.exists():
        ids, emb = pd.read_parquet(ids_path).item_id, np.load(emb_path)
    else:
        ids, emb = pd.Series([], dtype=str), np.zeros((0, 0), np.float16)
    new = corpus[~corpus.item_id.isin(set(ids))]
    if len(new) == 0:
        return
    model = load_model(key)
    texts = [MODELS[key]["d_prefix"] + t for t in item_text(new)]
    # sentence-transformers сам сортирует тексты по длине внутри encode (меньше паддинга)
    t = time.time()
    e = model.encode(texts, batch_size=128, normalize_embeddings=True,
                     convert_to_numpy=True, show_progress_bar=False).astype(np.float16)
    emb = e if len(ids) == 0 else np.vstack([emb, e])
    np.save(emb_path, emb)
    pd.DataFrame({"item_id": np.r_[ids.values, new.item_id.values]}).to_parquet(ids_path, index=False)
    print(f"{key}: закодировано {len(new)} новых объявлений за {time.time() - t:.0f} c")


def load_corpus_emb(key: str, item_ids) -> np.ndarray:
    """Эмбеддинги в порядке item_ids (корпус валидации или benchmark)."""
    ids = pd.read_parquet(ART_DIR / f"emb_{key}_ids.parquet").item_id
    pos = pd.Series(np.arange(len(ids)), index=ids.values)
    emb = np.load(ART_DIR / f"emb_{key}.npy")
    return emb[pos.loc[list(item_ids)].values].astype(np.float32)


if __name__ == "__main__":
    # python -m src.dense e5 val rank bench - закодировать корпуса указанных разбиений
    if len(sys.argv) < 3:
        print("Использование: python -m src.dense <модель> <разбиения…>")
        print("Например:      python -m src.dense e5 val rank bench")
        sys.exit(1)
    key, names = sys.argv[1], sys.argv[2:]
    for name in names:
        if name == "bench":
            from .data import load_bench_items
            c = load_bench_items()
        else:
            c = pd.read_parquet(ART_DIR / f"{name}_corpus.parquet")
        ensure_corpus_emb(key, c)
