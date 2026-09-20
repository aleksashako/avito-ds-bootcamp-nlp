"""
Нормализация текста запросов и объявлений
"""
import re
from functools import lru_cache

from nltk.stem.snowball import SnowballStemmer

TOKEN_RE = re.compile(r"[a-zа-я0-9]+")
_ru = SnowballStemmer("russian")
_en = SnowballStemmer("english")


@lru_cache(maxsize=2_000_000)
def stem(word: str) -> str:
    if word.isdigit():
        return word
    if word.isascii():
        return _en.stem(word)
    return _ru.stem(word)


def tokenize(text: str) -> list:
    text = text.lower().replace("ё", "е")
    return [stem(w) for w in TOKEN_RE.findall(text)]


# --- Лемматизация (pymorphy3): приводит слово к словарной форме ("телевизоров" -> "телевизор").
# Точнее стемминга на русском (стеммер режет окончания по правилам и иногда склеивает
# разные слова или разделяет формы одного), но медленнее — тоже кешируем по словам.
_morph = None


@lru_cache(maxsize=2_000_000)
def lemma(word: str) -> str:
    global _morph
    if word.isdigit() or word.isascii():
        return stem(word)
    if _morph is None:
        import pymorphy3
        _morph = pymorphy3.MorphAnalyzer()
    return _morph.parse(word)[0].normal_form.replace("ё", "е")


def tokenize_lemma(text: str) -> list:
    text = text.lower().replace("ё", "е")
    return [lemma(w) for w in TOKEN_RE.findall(text)]


TOKENIZERS = {"stem": tokenize, "lemma": tokenize_lemma}
