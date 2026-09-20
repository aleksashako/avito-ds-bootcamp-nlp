import re
from typing import Dict, Sequence

import pandas as pd

from .config import TOP_K

ITEM_RE = re.compile(r"^[0-9a-f]{16}$")


def write_answer(preds: Dict[str, Sequence[str]], query_ids: Sequence[str], path) -> pd.DataFrame:
    ans = pd.DataFrame({
        "query_id": list(query_ids),
        "answer": [" ".join(list(preds[q])[:TOP_K]) for q in query_ids],
    })
    ans.to_csv(path, index=False, encoding="utf-8")
    return ans


def validate_answer(path, bench_queries: pd.DataFrame, bench_items: pd.DataFrame) -> None:

    ans = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    assert list(ans.columns) == ["query_id", "answer"], f"колонки: {list(ans.columns)}"
    assert ans.query_id.is_unique, "повторы query_id"
    expected = set(bench_queries.query_id)
    got = set(ans.query_id)
    assert got == expected, f"нет {len(expected - got)} query_id, лишних {len(got - expected)}"
    assert ans.query_id.str.len().eq(16).all(), "query_id не из 16 символов"

    corpus = set(bench_items.item_id)
    n_items = []
    for qid, a in zip(ans.query_id, ans.answer):
        ids = a.split(" ") if a else []
        assert len(ids) <= TOP_K, f"{qid}: больше {TOP_K} item_id"
        assert len(ids) == len(set(ids)), f"{qid}: повторы item_id"
        bad = [i for i in ids if not ITEM_RE.match(i) or i not in corpus]
        assert not bad, f"{qid}: item_id не из корпуса или в неверном формате: {bad[:3]}"
        n_items.append(len(ids))
    print(f"answer.csv OK: {len(ans)} запросов, item_id в строке: "
          f"min={min(n_items)}, mean={sum(n_items) / len(n_items):.1f}")
