"""Финальный пайплайн: собирает answer.csv для benchmark

  python -m src.predict [logreg|weighted|rrf]     # по умолчанию logreg

Всё детерминировано (фиксированные seed, CPU-инференс, стабильная сортировка), поэтому
повторный запуск даёт тот же answer.csv; в конце печатается его md5 для сверки
"""
import hashlib
import sys
import time

from .config import ROOT, TOP_K
from .data import load_bench_items, load_bench_queries, load_train
from .dense import ensure_corpus_emb
from .features import ItemStats, build_pool
from .fusion import load_fuser
from .metrics import positions_in_query
from .sources import Retriever
from .submission import validate_answer, write_answer


def main(fuser_name: str = "logreg"):
    t = time.time()
    train, bq, items = load_train(), load_bench_queries(), load_bench_items()
    ensure_corpus_emb("e5", items)  # ничего не делает, если эмбеддинги уже в кеше

    print("строим источники на всём train…")
    R = Retriever().fit(train, items, cache="bench")
    stats = ItemStats().fit(train, items)
    pool = build_pool(R, stats, bq)
    print(f"пул: {len(pool)} строк, {len(pool) / len(bq):.0f} на запрос")

    score = load_fuser(fuser_name).score(pool)
    # при равных скорах порядок задаёт item_idx — результат детерминирован
    top = pool[positions_in_query(pool.query_id.values, score, tie=pool.item_idx.values) < TOP_K]
    item_ids = items.item_id.values
    preds = {q: item_ids[g.values].tolist() for q, g in top.groupby("query_id").item_idx}

    path = ROOT / "answer.csv"
    write_answer(preds, bq.query_id.tolist(), path)
    validate_answer(path, bq, items)
    print(f"слияние: {fuser_name}; md5(answer.csv) = {hashlib.md5(path.read_bytes()).hexdigest()}, "
          f"{time.time() - t:.0f} c")


if __name__ == "__main__":
    main(*sys.argv[1:])
