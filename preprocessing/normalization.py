import re
from typing import Any, Iterable

import pandas as pd


ANALOG_SUFFIXES: tuple[str, ...] = (
    "A", "А",            # латиница + кириллица
    "B", "C", "F", "K",
    "GT", "IN",
    " Б/У",
    "Б/У",
    "-L1",
    " JR", " NSK",
)

ANALOG_PREFIXES: tuple[str, ...] = (
    "W",     # WG0300002  → G0300002
    "AVX",   # AVX10X1125 → 10X1125
    "4Т-",   # 4Т-33006   → 33006
)


def normalize(val: Any) -> str | None:
    """
    Приводит значение к стандартному виду: strip + upper.
    Возвращает None для пустых / NaN-значений.
    """
    if pd.isna(val) or str(val).strip() in ("", "None", "nan"):
        return None
    return str(val).strip().upper()


def article_forms(val: Any) -> list[str]:
    """
    Возвращает все «формы» артикула: оригинал + варианты без суффиксов/префиксов
    + варианты без ведущих нулей.
    """
    base = normalize(val)
    if base is None:
        return []

    forms: list[str] = []
    seen: set[str] = set()

    def add_form(candidate: str | None) -> None:
        if not candidate:
            return

        candidate = candidate.strip()
        if not candidate or candidate in seen:
            return

        seen.add(candidate)
        forms.append(candidate)

        stripped = candidate.lstrip("0")
        if stripped and stripped != candidate and stripped[0] not in "-/. _" and stripped not in seen:
            seen.add(stripped)
            forms.append(stripped)

    add_form(base)

    for suf in ANALOG_SUFFIXES:
        if base.endswith(suf) and len(base) > len(suf):
            root = base[: -len(suf)].rstrip()
            add_form(root)

    for pre in ANALOG_PREFIXES:
        if base.startswith(pre) and len(base) > len(pre):
            root = base[len(pre):].lstrip()
            add_form(root)

    return forms


def extract_articles(text: str, match_keys: Iterable[str]) -> list[str]:
    """
    Извлекает список артикулов из текстовой строки.
    Использует словарь matches как множество паттернов.
    """
    # LettersDigits/Digits → два отдельных артикула
    text = re.sub(r"([A-Za-z]+)(\d+)/(\d+)", r"\1\2 \1\3", text)

    # Специальные кейсы
    if "kw1634" in text.lower() and "сдвоенный" in text.lower():
        return ["kw1634in", "kw1634"]
    if "ST40722" in text:
        return ["ST40722(1)", "ST40722(2)"]

    found = []
    for key in match_keys:
        if re.search(re.escape(key), text):
            found.append(key)
    return found


def safe_to_int(x: Any) -> Any:
    """
    Конвертирует x в int если x — целое число без дробной части.
    Иначе возвращает x без изменений.
    """
    if pd.isna(x):
        return x
    try:
        ix = int(x)
        return ix if ix == x else x
    except (TypeError, ValueError):
        return x
