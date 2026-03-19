import re
from typing import Any, Iterable


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
    import pandas as pd
    if pd.isna(val) or str(val).strip() in ("", "None", "nan"):
        return None
    return str(val).strip().upper()


def article_forms(val: Any) -> list[str]:
    """
    Возвращает все «формы» артикула: оригинал + варианты без суффиксов/префиксов
    + вариант без ведущих нулей.

    Используется при построении графа аналогов — артикулы, которые
    отличаются только суффиксом/префиксом, считаются связанными.

    Example:
        >>> article_forms("WG0300002")
        ["WG0300002", "G0300002"]

        >>> article_forms("00773607A")
        ["00773607A", "773607A", "773607"]
    """
    base = normalize(val)
    if base is None:
        return []

    forms: list[str] = [base]

    # ведущие нули: "00773607100401010" → "773607100401010"
    stripped = base.lstrip("0")
    if stripped and stripped != base and stripped[0] not in "-/. _":
        forms.append(stripped)

    for suf in ANALOG_SUFFIXES:
        s = suf.upper()
        if base.endswith(s) and len(base) > len(s):
            root = base[: -len(s)].rstrip()
            if root and root not in forms:
                forms.append(root)

    for pre in ANALOG_PREFIXES:
        p = pre.upper()
        if base.startswith(p) and len(base) > len(p):
            root = base[len(p):].lstrip()
            if root and root not in forms:
                forms.append(root)

    return forms


def extract_articles(text: str, match_keys: Iterable[str]) -> list[str]:
    """
    Извлекает список артикулов из текстовой строки.
    Использует словарь matches как множество паттернов.
    """
    # LettersDigits/Digits → два отдельных артикула
    text = re.sub(r"([A-Za-z]+)(\d+)/(\d+)", r"\1\2 \1\3", text)

    # Специальные кейсы (домен-специфичная логика)
    if "kw1634" in text.lower() and "сдвоенный" in text.lower():
        return ["kw1634in", "kw1634"]
    if "ST40722" in text:
        return ["ST40722(1)", "ST40722(2)"]

    found: list[str] = []
    for key in match_keys:
        if re.search(re.escape(key), text):
            found.append(key)
    return found


def extract_model_case_insensitive(machine: str, brand: str) -> str | None:
    """
    Извлекает модель из строки названия машины, отсекая название бренда.
    Возвращает None если brand не найден в machine.

    Example:
        >>> extract_model_case_insensitive("Genie GS-1932", "Genie")
        "GS-1932"
    """
    import pandas as pd
    if pd.isna(machine) or pd.isna(brand):
        return None

    machine_lower = machine.lower()
    brand_lower   = brand.lower()
    idx = machine_lower.find(brand_lower)
    if idx == -1:
        return None

    return machine[idx + len(brand):].strip()


def safe_to_int(x: Any) -> Any:
    """
    Конвертирует x в int если x — целое число без дробной части.
    Иначе возвращает x без изменений.

    Безопасная альтернатива int(x) без риска ValueError.
    """
    import pandas as pd
    if pd.isna(x):
        return x
    try:
        ix = int(x)
        return ix if ix == x else x
    except (TypeError, ValueError):
        return x
