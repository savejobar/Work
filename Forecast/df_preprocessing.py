import re
import os
import ast
from collections import defaultdict

from typing import Optional, List, Iterable, Any, Dict, Set, Tuple
import numpy as np
import pandas as pd


_ANALOG_SUFFIXES = (
    "A", "А",           # латиница + кириллица
    "B", "C", "F", "K",
    "GT", "IN",
    " Б/У",
    "Б/У",
    "-L1",
    " JR", " NSK",
)
_ANALOG_PREFIXES = (
    "W",      # WG0300002  → G0300002
    "AVX",    # AVX10X1125 → 10X1125
    "4Т-",    # 4Т-33006   → 33006
)

_EXCLUDED_SUFFIXES = ("-01", "-02", "-03")


def load_repair_parts(file_csv: str, file_excel: str, skiprows: int = 9) -> pd.DataFrame | None:

    try:
        data = pd.read_csv(file_csv, skiprows=range(skiprows), dtype=str)
        print(f"Файл успешно загружен из CSV: {file_csv}")
        return data
    except FileNotFoundError:
        try:
            data = pd.read_excel(file_excel, skiprows=range(skiprows), dtype=str)
            print(f"Файл успешно загружен из Excel: {file_excel}")
            return data
        except FileNotFoundError:
            print("Файлы не найдены. Проверьте путь к CSV и Excel.")
            return None
    except Exception as e:
        print(f"Ошибка при чтении CSV: {e}")
        return None
    

def extract_model_case_insensitive(machine: Optional[str], brand: Optional[str]) -> Optional[str]:

    if pd.isna(machine) or pd.isna(brand):
        return None

    machine_lower = machine.lower()
    brand_lower = brand.lower()
    idx = machine_lower.find(brand_lower)
    if idx == -1:
        return None

    return machine[idx + len(brand):].strip()


def extract_articles(text: str, match_keys: Iterable[str]) -> List[str]:

    # Преобразование паттерна LettersDigits/Digits в отдельные артикула
    text = re.sub(r'([A-Za-z]+)(\d+)/(\d+)', r'\1\2 \1\3', text)

    # Специальные кейсы
    if ('kw1634' in text.lower()) and 'сдвоенный' in text.lower():
        return ['kw1634in', 'kw1634']
    if 'ST40722' in text:
        return ['ST40722(1)', 'ST40722(2)']

    found: List[str] = []
    for key in match_keys:
        pattern = re.escape(key)
        if re.search(pattern, text):
            found.append(key)

    return found


def add_original_to_extended(row: pd.Series) -> str | None:

    main_art = str(row["Номенклатура.Артикул"]).strip()
    extended = row["Номенклатура.Оригинальный номер расширенный"]
    original = row["Номенклатура.Оригинальный номер"]

    extended_parts = []
    if pd.notna(extended) and str(extended).strip() != "":
        extended_parts = re.split(r'\s+', str(extended))
        extended_parts = [part.strip() for part in extended_parts if part.strip()]

    original_parts = []
    if pd.notna(original) and str(original).strip() != "":
        original_parts = [str(original).strip()]

    extended_upper = [p.upper() for p in extended_parts]

    for part in original_parts:
        if part.upper() not in extended_upper and part.upper() != main_art.upper():
            extended_parts.append(part)

    extended_parts = sorted(list(dict.fromkeys(extended_parts)))

    return " ".join(extended_parts) if extended_parts else np.nan


def find_all_analogs(start: Any, graph: Dict[Any, Set[Any]]) -> Tuple[Any, ...]:

    visited = set()
    stack = [start]

    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            stack.extend(graph[node] - visited)

    return tuple(sorted(visited))


def normalize(val) -> str | None:
    if pd.isna(val) or str(val).strip() in ("", "None", "nan"):
        return None
    return str(val).strip().upper()  


def join_ordered(series):

    vals = series.dropna().astype(str).str.strip()
    vals = [v for v in vals if v]
    seen = set()
    result = []
    for v in vals:
        if v not in seen:
            seen.add(v)
            result.append(v)
    return "; ".join(result) if result else None


def shortest_value(series):

    vals = series.dropna().astype(str).str.strip()
    if len(vals) == 0:
        return None
    return min(vals, key=len)


def count_unique(series):

    vals = series.dropna().astype(str).str.strip()
    vals = [v for v in vals if v]
    return len(set(vals))


def merge_parts_df1(df):

    df = df.copy()
    df['Количество'] = df['Количество'].astype(float).astype(int)
    df['Лот.CRM наработка'] = df['Лот.CRM наработка'].apply(lambda x: str(x).replace(',',''))
    df['Лот.CRM наработка'] = df['Лот.CRM наработка'].astype(float).astype('Int64')
    df['Лот.CRM год выпуска'] = df['Лот.CRM год выпуска'].astype(float).astype('Int64')

    meta = (
        df.groupby("Номер группы", as_index=False)
        .agg({
            "Номенклатура": shortest_value,
            "Машина.Бренд": join_ordered,
            "Машина.Серия техники": join_ordered,
            "Тип подъемника": join_ordered,
            "Тип двигателя": join_ordered,
            'Номенклатура.Тип техники': join_ordered,
            'Документ.Склад': join_ordered,
            'Номера машин': join_ordered,
            "all_analogs": "first"
        })
        .rename(columns={
        'Машина.Бренд': 'Бренды',
        'Машина.Серия техники': 'Серии техники',
        'Тип подъемника': 'Типы подъемников',
        'Тип двигателя': 'Типы двигателя',
        'Документ.Склад': 'Склады',
        'all_analogs': 'Список аналогов'
    })
    )

    qty = (
        df.groupby(
            ["Год", "Месяц", "Номер группы"],
            as_index=False
        )
        .agg({
            "Количество": "sum",
            'Лот.CRM год выпуска': 'mean',
            'Средняя наработка (лет)': 'mean',
            'Серийный номер': count_unique,
            'Лот.CRM наработка': 'mean'
        })
        .rename(columns={
        'Серийный номер':  "Количество уникальных номеров машин, где запчасть применялась",
        "Количество": 'Ремонт',
        'Лот.CRM год выпуска' : 'Средний год выпуска',
        'Лот.CRM наработка': 'Средняя наработка техники, где запчасть применялась (ч)'
        })
    )

    qty['Средний год выпуска'] = qty['Средний год выпуска'].round(1)
    qty['Средняя наработка (лет)'] = qty['Средняя наработка (лет)'].round(1)
    qty['Средняя наработка техники, где запчасть применялась (ч)'] = qty['Средняя наработка техники, где запчасть применялась (ч)'].round(1) 
    result = qty.merge(meta, on="Номер группы", how="left")

    return result


def merge_parts_df2(df):
    df = df.copy()

    meta = (
        df.groupby("Номер группы", as_index=False)
        .agg({
            "Номенклатура": shortest_value,
            'Склад': join_ordered,
            "Список аналогов": lambda x: max(x.dropna(), key=len, default=None),
            "Тип техники": join_ordered,
            'Артикул': 'first',
        })
    )

    qty = (
        df.groupby(["Год", "Месяц", "Номер группы"], as_index=False)
        .agg({
            "Расход": "sum",
            "Приход": "sum",
            "Конечный остаток": 'last'
        })
    )

    for col in ["Расход", "Приход", "Конечный остаток", 'Номер группы']:
        qty[col] = qty[col].apply(lambda x: int(x) if x.is_integer() else x)

    result = qty.merge(meta, on="Номер группы", how="left")
    return result


def prepare_sales(agg: pd.DataFrame, df_quarter: pd.DataFrame) -> pd.DataFrame:
    agg['Год'] = agg['Год'].astype(int)
    agg['Месяц'] = agg['Месяц'].astype(int)
    df_quarter['Год'] = df_quarter['Год'].astype(int)
    df_quarter['Месяц'] = df_quarter['Месяц'].astype(int)

    merged = agg.merge(
        df_quarter[['Год','Месяц','Номер группы','Ремонт']],
        on=['Год','Месяц','Номер группы'],
        how='left'
    )

    merged['Продажа'] = merged['Расход'] - merged['Ремонт'].fillna(0)

    merged = merged.drop(columns=['Расход', 'Ремонт'])

    return merged


def normalize_analog_lists(df, col_group="Номер группы", col_analogs="Список аналогов"):

    df = df.copy()

    group_max_analogs = (
        df.groupby(col_group)[col_analogs]
        .apply(lambda s: max(
            (v for v in s if isinstance(v, tuple)),
            key=len,
            default=None
        ))
        .to_dict()
    )

    df[col_analogs] = df.apply(
        lambda row: group_max_analogs.get(row[col_group], row[col_analogs]),
        axis=1
    )

    return df


def article_forms(val) -> list[str]:
    base = normalize(val)
    if base is None:
        return []

    forms: list[str] = [base]

    # ведущие нули: "00773607100401010" → "773607100401010"
    stripped = base.lstrip("0")
    if stripped and stripped != base and stripped[0] not in "-/. _":
        forms.append(stripped)

    for suf in _ANALOG_SUFFIXES:
        s = suf.upper()
        if base.endswith(s) and len(base) > len(s):
            root = base[: -len(s)].rstrip()
            if root and root not in forms:
                forms.append(root)

    for pre in _ANALOG_PREFIXES:
        p = pre.upper()
        if base.startswith(p) and len(base) > len(p):
            root = base[len(p):].lstrip()
            if root and root not in forms:
                forms.append(root)

    return forms

# внизу хуета

def find_mixed_groups(df_raw: pd.DataFrame,
                      col_group: str = 'Номер группы',
                      col_name:  str = 'Номенклатура',
                      min_unique: int = 2) -> pd.DataFrame:
    """
    Найти группы аналогов, где одна группа содержит детали
    с принципиально разными названиями — признак ложного аналога.
    
    Применять на сырых данных ДО вызова merge_parts_df1/df2.
    """
    # Все уникальные названия по каждой группе
    group_names = (
        df_raw.groupby(col_group)[col_name]
        .apply(lambda s: sorted(s.dropna().astype(str).str.strip().unique()))
        .reset_index()
        .rename(columns={col_name: 'все_названия'})
    )
    group_names['кол_во_названий'] = group_names['все_названия'].apply(len)
    
    # Оставляем только группы с расхождением
    mixed = group_names[group_names['кол_во_названий'] >= min_unique].copy()
    
    # Считаем схожесть: если все названия похожи — это просто опечатки,
    # если нет — скорее всего разные детали
    import re
    def avg_similarity(names: list[str]) -> float:
        if len(names) < 2:
            return 1.0
        scores = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                w1 = set(re.split(r'\W+', names[i].lower())) - {''}
                w2 = set(re.split(r'\W+', names[j].lower())) - {''}
                scores.append(len(w1 & w2) / len(w1 | w2) if w1 | w2 else 1.0)
        return round(sum(scores) / len(scores), 2)
    
    mixed['схожесть_названий'] = mixed['все_названия'].apply(avg_similarity)
    
    return (
        mixed
        .sort_values('схожесть_названий')   # самые подозрительные первые
        [['Номер группы', 'кол_во_названий', 'схожесть_названий', 'все_названия']]
        .reset_index(drop=True)
    )


def _parse_analogs(val) -> list[str]:
    """Распарсить поле «Список аналогов» в список строк."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return []
    s = str(val).strip()
    if s in ("", "nan", "None"):
        return []
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, (tuple, list)):
            return [str(x).strip() for x in parsed if str(x).strip()]
        return [str(parsed).strip()]
    except (ValueError, SyntaxError):
        return [s]


def _norm(art: str) -> str:
    """Нормализация артикула для сравнений: верхний регистр, без пробелов."""
    return str(art).strip().upper()


def _name_similarity(n1: str, n2: str) -> float:
    """Доля общих слов относительно объединения множеств слов двух строк."""
    if not n1 or not n2:
        return 0.0
    w1 = set(re.split(r'\W+', n1.lower())) - {""}
    w2 = set(re.split(r'\W+', n2.lower())) - {""}
    if not w1 or not w2:
        return 0.0
    return len(w1 & w2) / len(w1 | w2)


def _confidence(sim: float) -> str:
    if sim >= 0.5:
        return "HIGH"
    if sim >= 0.15:
        return "MEDIUM"
    return "LOW"


def audit_analog_groups(
    df: pd.DataFrame,
    col_group: str = "Номер группы",
    col_article: str = "Артикул",
    col_name: str = "Номенклатура",
    col_analogs: str = "Список аналогов",
    # Суффиксы для LETTER_SUFFIX_SPLIT — проверяются как суффикс И как префикс
    letter_suffixes: tuple[str, ...] = (
        "A", "А",             # латиница + кириллица
        "B", "C", "F", "K",   # другие буквенные ревизии
        "GT", "IN",           # составные буквенные суффиксы
        " Б/У",               # бывшие в употреблении
        "-01", "-02", "-03",
        "-R", "-L", "-L1",
        " JR", " NSK",
    ),
    # Строки, которые проверяются ТОЛЬКО как префикс (не как суффикс)
    prefix_only: tuple[str, ...] = (
        "W",                  # WG0300002 vs G0300002
        "AVX",                # AVX10X1125 vs 10X1125
        "4Т-",                # 4Т-33006 vs 33006
    ),
    # Числовые суффиксы для NUMERIC_SUFFIX_SPLIT
    numeric_suffix_max_len: int = 3,  # 001, 01, 1 — но не 0001
    numeric_suffix_min_len: int = 1,
    # Максимальное кол-во ведущих нулей для LEADING_ZEROS_SPLIT
    leading_zeros_max: int = 4,
) -> pd.DataFrame:

    unique = (
        df[[col_group, col_article, col_name, col_analogs]]
        .drop_duplicates(subset=[col_group])
        .copy()
        .reset_index(drop=True)
    )

    unique["_parsed"] = unique[col_analogs].apply(_parse_analogs)
    unique["_name_lc"] = unique[col_name].fillna("").astype(str).str.strip().str.lower()

    # Вспомогательные индексы
    # article (norm) -> group
    art_to_group: dict[str, str] = {}
    # group -> set of norm articles in its analog list
    group_to_arts: dict[str, set[str]] = {}
    # group -> row (для быстрого доступа к имени)
    group_to_row: dict[str, pd.Series] = {}

    for _, row in unique.iterrows():
        grp = str(row[col_group])
        group_to_row[grp] = row
        arts = {_norm(a) for a in row["_parsed"]}
        group_to_arts[grp] = arts
        for a in arts:
            # Если коллизия — запишем специальный маркер
            if a in art_to_group and art_to_group[a] != grp:
                art_to_group[a] = "__DUPLICATE__"
            else:
                art_to_group[a] = grp

    issues: list[dict] = []

    def _row_info(grp: str) -> tuple[str, str, str]:
        """(article, name, analogs_str) для группы."""
        r = group_to_row[grp]
        return (
            str(r[col_article]),
            str(r[col_name]),
            str(r[col_analogs]),
        )

    def _add(
        issue_type: str,
        grp_a: str,
        grp_b: str | None,
        art_a: str,
        art_b: str | None,
        sim: float,
        comment: str,
        suffix_or_prefix: str = "",
    ):
        art_a_main, name_a, analogs_a = _row_info(grp_a)
        art_b_main, name_b, analogs_b = _row_info(grp_b) if grp_b else ("", "", "")
        issues.append(
            {
                "issue_type": issue_type,
                "confidence": _confidence(sim),
                "name_similarity": round(sim, 2),
                "group_a": grp_a,
                "group_b": grp_b or "",
                "article_a": art_a,
                "article_b": art_b or "",
                "suffix_or_prefix": suffix_or_prefix,
                "name_a": name_a,
                "name_b": name_b,
                "analogs_a": analogs_a,
                "analogs_b": analogs_b,
                "comment": comment,
            }
        )

    dup_arts = {a for a, g in art_to_group.items() if g == "__DUPLICATE__"}
    if dup_arts:
        # Нужно восстановить, в каких именно группах артикул встречается
        art_to_all_groups: dict[str, list[str]] = defaultdict(list)
        for _, row in unique.iterrows():
            grp = str(row[col_group])
            for a in row["_parsed"]:
                an = _norm(a)
                if an in dup_arts:
                    art_to_all_groups[an].append(grp)

        seen_dup_pairs: set[tuple] = set()
        for art, groups in art_to_all_groups.items():
            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    ga, gb = groups[i], groups[j]
                    pair = tuple(sorted([ga, gb]))
                    if pair in seen_dup_pairs:
                        continue
                    seen_dup_pairs.add(pair)
                    n_a = group_to_row[ga][col_name]
                    n_b = group_to_row[gb][col_name]
                    sim = _name_similarity(n_a, n_b)
                    _add(
                        "DUPLICATE_IN_GROUPS",
                        ga, gb, art, art, sim,
                        f"Артикул «{art}» встречается в аналогах обеих групп",
                    )

    for _, row in unique.iterrows():
        grp = str(row[col_group])
        own = _norm(str(row[col_article]))
        if own in ("NAN", "", "NONE"):
            continue
        if own not in group_to_arts[grp]:
            _add(
                "SELF_MISSING",
                grp, None, str(row[col_article]), None, 0.0,
                f"Собственный артикул «{row[col_article]}» отсутствует в списке аналогов группы",
            )


    all_arts_norm = list(art_to_group.keys())
    seen_letter_pairs: set[tuple] = set()

    # Суффиксная часть
    for base_art in all_arts_norm:
        if art_to_group.get(base_art) == "__DUPLICATE__":
            continue
        for suf in letter_suffixes:
            candidate = base_art + suf.upper()
            if candidate not in art_to_group:
                continue
            if art_to_group.get(candidate) == "__DUPLICATE__":
                continue
            ga = art_to_group[base_art]
            gb = art_to_group[candidate]
            if ga == gb:
                continue
            pair = tuple(sorted([ga + "|" + base_art, gb + "|" + candidate]))
            if pair in seen_letter_pairs:
                continue
            seen_letter_pairs.add(pair)
            n_a = group_to_row[ga][col_name]
            n_b = group_to_row[gb][col_name]
            sim = _name_similarity(n_a, n_b)
            _add(
                "LETTER_SUFFIX_SPLIT",
                ga, gb, base_art, candidate, sim,
                f"«{base_art}» и «{candidate}» (суффикс «{suf.strip()}») находятся в разных группах",
                suffix_or_prefix=suf.strip(),
            )

    all_prefixes = [
        s for s in letter_suffixes
        if not s.startswith(" ")
    ] + list(prefix_only)
    seen_prefix_pairs: set[tuple] = set()

    for base_art in all_arts_norm:
        if art_to_group.get(base_art) == "__DUPLICATE__":
            continue
        # base_art должен быть достаточно длинным, чтобы префикс имел смысл
        if len(base_art) < 3:
            continue
        for pre in all_prefixes:
            candidate = pre.upper() + base_art
            if candidate not in art_to_group:
                continue
            if art_to_group.get(candidate) == "__DUPLICATE__":
                continue
            ga = art_to_group[base_art]
            gb = art_to_group[candidate]
            if ga == gb:
                continue
            pair = tuple(sorted([ga + "|" + base_art, gb + "|" + candidate]))
            if pair in seen_prefix_pairs:
                continue
            seen_prefix_pairs.add(pair)
            n_a = group_to_row[ga][col_name]
            n_b = group_to_row[gb][col_name]
            sim = _name_similarity(n_a, n_b)
            _add(
                "LETTER_SUFFIX_SPLIT",
                ga, gb, base_art, candidate, sim,
                f"«{base_art}» и «{candidate}» (префикс «{pre.strip()}») находятся в разных группах",
                suffix_or_prefix=f"prefix:{pre.strip()}",
            )

    seen_num_pairs: set[tuple] = set()

    for base_art in all_arts_norm:
        if art_to_group.get(base_art) == "__DUPLICATE__":
            continue
        ga = art_to_group[base_art]
        for candidate in all_arts_norm:
            if candidate == base_art:
                continue
            if art_to_group.get(candidate) == "__DUPLICATE__":
                continue
            gb = art_to_group[candidate]
            if ga == gb:
                continue
            if not candidate.startswith(base_art):
                continue
            suffix = candidate[len(base_art):]
            if not (
                numeric_suffix_min_len <= len(suffix) <= numeric_suffix_max_len
                and suffix.isdigit()
            ):
                continue
            pair = tuple(sorted([ga + "|" + base_art, gb + "|" + candidate]))
            if pair in seen_num_pairs:
                continue
            seen_num_pairs.add(pair)
            n_a = group_to_row[ga][col_name]
            n_b = group_to_row[gb][col_name]
            sim = _name_similarity(n_a, n_b)
            _add(
                "NUMERIC_SUFFIX_SPLIT",
                ga, gb, base_art, candidate, sim,
                f"«{base_art}» и «{candidate}» (числовой суффикс «{suffix}») находятся в разных группах",
                suffix_or_prefix=suffix,
            )

    seen_zero_pairs: set[tuple] = set()

    for base_art in all_arts_norm:
        if art_to_group.get(base_art) == "__DUPLICATE__":
            continue
        # base_art не должен сам начинаться с нуля (иначе будет дублирование)
        if base_art.startswith("0"):
            continue
        ga = art_to_group[base_art]
        for n_zeros in range(1, leading_zeros_max + 1):
            candidate = "0" * n_zeros + base_art
            if candidate not in art_to_group:
                continue
            if art_to_group.get(candidate) == "__DUPLICATE__":
                continue
            gb = art_to_group[candidate]
            if ga == gb:
                continue
            pair = tuple(sorted([ga + "|" + base_art, gb + "|" + candidate]))
            if pair in seen_zero_pairs:
                continue
            seen_zero_pairs.add(pair)
            n_a = group_to_row[ga][col_name]
            n_b = group_to_row[gb][col_name]
            sim = _name_similarity(n_a, n_b)
            zeros = "0" * n_zeros
            _add(
                "LEADING_ZEROS_SPLIT",
                ga, gb, base_art, candidate, sim,
                f"«{base_art}» и «{candidate}» отличаются ведущими нулями «{zeros}»",
                suffix_or_prefix=f"prefix:{zeros}",
            )

    name_to_groups: dict[str, list[str]] = defaultdict(list)
    for _, row in unique.iterrows():
        name = row["_name_lc"]
        if name and name not in ("nan", "none", ""):
            name_to_groups[name].append(str(row[col_group]))

    seen_name_pairs: set[tuple] = set()
    for name, groups in name_to_groups.items():
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                ga, gb = groups[i], groups[j]
                pair = tuple(sorted([ga, gb]))
                if pair in seen_name_pairs:
                    continue
                seen_name_pairs.add(pair)
                art_a = str(group_to_row[ga][col_article])
                art_b = str(group_to_row[gb][col_article])
                _add(
                    "SAME_NAME_SPLIT",
                    ga, gb, art_a, art_b, 1.0,
                    f"Одинаковое название «{name}» в двух разных группах",
                )

    main_art_to_group: dict[str, str] = {}
    for _, row in unique.iterrows():
        an = _norm(str(row[col_article]))
        if an not in ("NAN", "", "NONE"):
            main_art_to_group[an] = str(row[col_group])

    seen_xref_pairs: set[tuple] = set()
    for _, row in unique.iterrows():
        ga = str(row[col_group])
        arts_a = group_to_arts[ga]
        for art in arts_a:
            if art not in main_art_to_group:
                continue
            gb = main_art_to_group[art]
            if gb == ga:
                continue
            arts_b = group_to_arts[gb]
            # Асимметрия: A ссылается на B через art, но B не ссылается на A
            arts_a_minus_self = arts_a - {_norm(str(row[col_article]))}
            if not arts_a_minus_self.intersection(arts_b - {art}):
                pair = tuple(sorted([ga, gb]))
                if pair in seen_xref_pairs:
                    continue
                seen_xref_pairs.add(pair)
                n_a = str(row[col_name])
                n_b = str(group_to_row[gb][col_name])
                sim = _name_similarity(n_a, n_b)
                art_b_main = str(group_to_row[gb][col_article])
                _add(
                    "CROSS_REF_ASYMMETRY",
                    ga, gb,
                    str(row[col_article]), art_b_main,
                    sim,
                    f"Группа {ga} содержит «{art}» (главный арт. группы {gb}), "
                    f"но группа {gb} не содержит артикулов группы {ga}",
                )

    if not issues:
        return pd.DataFrame(
            columns=[
                "issue_type", "confidence", "name_similarity",
                "group_a", "group_b", "article_a", "article_b",
                "suffix_or_prefix",
                "name_a", "name_b", "analogs_a", "analogs_b", "comment",
            ]
        )

    result = pd.DataFrame(issues)

    # Сортировка: сначала HIGH, потом по типу проблемы
    conf_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    type_order = {
        "DUPLICATE_IN_GROUPS": 0,
        "SELF_MISSING": 1,
        "SAME_NAME_SPLIT": 2,
        "LETTER_SUFFIX_SPLIT": 3,
        "NUMERIC_SUFFIX_SPLIT": 4,
        "LEADING_ZEROS_SPLIT": 5,
        "CROSS_REF_ASYMMETRY": 6,
    }
    result["_c_ord"] = result["confidence"].map(conf_order)
    result["_t_ord"] = result["issue_type"].map(type_order).fillna(99)
    result = (
        result.sort_values(["_c_ord", "_t_ord", "name_similarity"],
                           ascending=[True, True, False])
        .drop(columns=["_c_ord", "_t_ord"])
        .reset_index(drop=True)
    )

    return result


def print_audit_summary(audit_df: pd.DataFrame) -> None:
    """Печать читаемого отчёта по результатам audit_analog_groups()."""
    if audit_df.empty:
        print("✅ Проблем не обнаружено.")
        return

    total = len(audit_df)
    print(f"{'='*70}")
    print(f"  АУДИТ АНАЛОГОВ  —  найдено проблем: {total}")
    print(f"{'='*70}")

    for issue_type, group in audit_df.groupby("issue_type", sort=False):
        counts = group["confidence"].value_counts()
        high   = counts.get("HIGH", 0)
        medium = counts.get("MEDIUM", 0)
        low    = counts.get("LOW", 0)
        print(f"\n▶  {issue_type}  ({len(group)} шт.)  "
              f"HIGH={high}  MEDIUM={medium}  LOW={low}")
        has_suf = "suffix_or_prefix" in group.columns
        print(f"   {'Гр.A':>6}  {'Гр.B':>6}  {'Артикул A':<22}  {'Артикул B':<22}  {'Suf/Pre':<10}  {'Conf':<7}  Sim")
        print(f"   {'-'*6}  {'-'*6}  {'-'*22}  {'-'*22}  {'-'*10}  {'-'*7}  ---")
        for _, r in group.iterrows():
            suf = str(r.get("suffix_or_prefix", ""))[:10] if has_suf else ""
            print(
                f"   {r['group_a']:>6}  {str(r['group_b']):>6}  "
                f"{str(r['article_a'])[:22]:<22}  {str(r['article_b'])[:22]:<22}  "
                f"{suf:<10}  {r['confidence']:<7}  {r['name_similarity']:.2f}"
            )
            print(f"         ↳ {r['name_a'][:60]}")
            if r["group_b"]:
                print(f"           {r['name_b'][:60]}")

    print(f"\n{'='*70}")
    print("Колонки результирующего датафрейма:")
    print("  issue_type, confidence, name_similarity,")
    print("  group_a, group_b, article_a, article_b, suffix_or_prefix,")
    print("  name_a, name_b, analogs_a, analogs_b, comment")
    print(f"{'='*70}")



def export_to_excel(audit_df: pd.DataFrame, path: str) -> None:
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
        from openpyxl.utils import get_column_letter
    except ImportError as e:
        raise ImportError("Установите openpyxl: pip install openpyxl") from e

    # ── Константы ──────────────────────────────────────────────────────────
    CLR = {
        "header_bg":    "1F3864",
        "header_fg":    "FFFFFF",
        "HIGH_bg":      "FFD7D7",
        "HIGH_fg":      "C00000",
        "MEDIUM_bg":    "FFF2CC",
        "MEDIUM_fg":    "7F6000",
        "LOW_bg":       "E2EFDA",
        "LOW_fg":       "375623",
        "alt_row":      "F2F7FC",
        "summary_hdr":  "2E75B6",
        "border":       "BFBFBF",
    }
    TAB_COLORS = {
        "SAME_NAME_SPLIT":      "C00000",
        "LETTER_SUFFIX_SPLIT":  "FF7C00",
        "NUMERIC_SUFFIX_SPLIT": "FFD700",
        "LEADING_ZEROS_SPLIT":  "70AD47",
        "DUPLICATE_IN_GROUPS":  "7030A0",
        "SELF_MISSING":         "00B0F0",
        "CROSS_REF_ASYMMETRY":  "808080",
    }
    ISSUE_NAMES = {
        "SAME_NAME_SPLIT":      "Одинаковое название, разные группы",
        "LETTER_SUFFIX_SPLIT":  "Буквенный суффикс / префикс",
        "NUMERIC_SUFFIX_SPLIT": "Числовой суффикс (ревизия)",
        "LEADING_ZEROS_SPLIT":  "Ведущие нули",
        "DUPLICATE_IN_GROUPS":  "Дубликат артикула в нескольких группах",
        "SELF_MISSING":         "Артикул отсутствует в своих аналогах",
        "CROSS_REF_ASYMMETRY":  "Асимметричная перекрёстная ссылка",
    }
    ISSUE_ORDER = [
        "SAME_NAME_SPLIT", "LETTER_SUFFIX_SPLIT", "NUMERIC_SUFFIX_SPLIT",
        "LEADING_ZEROS_SPLIT", "DUPLICATE_IN_GROUPS", "SELF_MISSING",
        "CROSS_REF_ASYMMETRY",
    ]
    CONF_RU = {"HIGH": "🔴 Высокая", "MEDIUM": "🟡 Средняя", "LOW": "🟢 Низкая"}
    COLS = [
        ("confidence",       "Уровень",       12),
        ("group_a",          "Группа A",       10),
        ("article_a",        "Артикул A",      22),
        ("name_a",           "Название A",     38),
        ("group_b",          "Группа B",       10),
        ("article_b",        "Артикул B",      22),
        ("name_b",           "Название B",     38),
        ("suffix_or_prefix", "Суффикс/Пре.",   13),
        ("name_similarity",  "Схожесть",       10),
        ("comment",          "Комментарий",    45),
    ]

    THIN = Side(style="thin",   color=CLR["border"])
    MED  = Side(style="medium", color="888888")

    def _fill(hex_color: str) -> PatternFill:
        return PatternFill("solid", fgColor=hex_color)

    def _border(*sides: str) -> Border:
        return Border(**{s: THIN for s in sides})

    def _hdr_font(size: int = 11) -> Font:
        return Font(name="Arial", bold=True, color=CLR["header_fg"], size=size)

    def _body_font(bold: bool = False, color: str = "000000", size: int = 10) -> Font:
        return Font(name="Arial", bold=bold, color=color, size=size)

    def _write_header_row(ws, titles: list[str], row: int = 1) -> None:
        for c, title in enumerate(titles, 1):
            cell = ws.cell(row=row, column=c, value=title)
            cell.font      = _hdr_font()
            cell.fill      = _fill(CLR["header_bg"])
            cell.alignment = Alignment(horizontal="center", vertical="center",
                                       wrap_text=True)
            cell.border    = Border(bottom=MED, top=MED, left=THIN, right=THIN)
        ws.row_dimensions[row].height = 30

    # ── Книга ──────────────────────────────────────────────────────────────
    wb = Workbook()
    wb.remove(wb.active)

    # ── Лист «Сводка» ──────────────────────────────────────────────────────
    ws_sum = wb.create_sheet("📊 Сводка")
    ws_sum.sheet_properties.tabColor = CLR["summary_hdr"]
    ws_sum.freeze_panes = "A3"

    ws_sum.merge_cells("A1:F1")
    tc = ws_sum["A1"]
    tc.value     = "АУДИТ АНАЛОГОВ — СВОДНЫЙ ОТЧЁТ"
    tc.font      = Font(name="Arial", bold=True, size=14, color="FFFFFF")
    tc.fill      = _fill(CLR["header_bg"])
    tc.alignment = Alignment(horizontal="center", vertical="center")
    ws_sum.row_dimensions[1].height = 36

    _write_header_row(
        ws_sum,
        ["Тип проблемы", "Описание", "🔴 Высокая", "🟡 Средняя", "🟢 Низкая", "Всего"],
        row=2,
    )

    total_h = total_m = total_l = 0
    summary_rows: list[list] = []
    for issue_type in ISSUE_ORDER:
        grp = audit_df[audit_df["issue_type"] == issue_type]
        if grp.empty:
            continue
        cnts = grp["confidence"].value_counts()
        h = int(cnts.get("HIGH",   0))
        m = int(cnts.get("MEDIUM", 0))
        l = int(cnts.get("LOW",    0))
        total_h += h; total_m += m; total_l += l
        summary_rows.append([issue_type, ISSUE_NAMES.get(issue_type, ""), h, m, l, h+m+l])

    for r_idx, row_data in enumerate(summary_rows, 3):
        alt = (r_idx % 2 == 0)
        for c_idx, val in enumerate(row_data, 1):
            cell = ws_sum.cell(row=r_idx, column=c_idx, value=val)
            cell.border    = _border("left", "right", "bottom")
            cell.alignment = Alignment(vertical="center",
                                       wrap_text=(c_idx <= 2))
            cell.fill = _fill(CLR["alt_row"]) if alt else PatternFill()
            if c_idx == 1:
                cell.font = _body_font(bold=True,
                                       color=TAB_COLORS.get(val, "000000"))
            elif c_idx == 3:
                cell.font = _body_font(bold=bool(val), color=CLR["HIGH_fg"])
            elif c_idx == 4:
                cell.font = _body_font(bold=bool(val), color=CLR["MEDIUM_fg"])
            elif c_idx == 5:
                cell.font = _body_font(bold=bool(val), color=CLR["LOW_fg"])
            elif c_idx == 6:
                cell.font = _body_font(bold=True)
            else:
                cell.font = _body_font()

    # итоговая строка
    tot = len(summary_rows) + 3
    for c_idx, val in enumerate(
        ["ИТОГО", "", total_h, total_m, total_l, total_h + total_m + total_l], 1
    ):
        cell = ws_sum.cell(row=tot, column=c_idx, value=val)
        cell.font   = _body_font(bold=True, size=11)
        cell.fill   = _fill("D9E1F2")
        cell.border = Border(top=MED, bottom=MED, left=THIN, right=THIN)
        cell.alignment = Alignment(horizontal="center", vertical="center")

    ws_sum.column_dimensions["A"].width = 26
    ws_sum.column_dimensions["B"].width = 42
    for col in ["C", "D", "E", "F"]:
        ws_sum.column_dimensions[col].width = 14

    # ── Листы по типам ─────────────────────────────────────────────────────
    for issue_type in ISSUE_ORDER:
        grp = audit_df[audit_df["issue_type"] == issue_type]
        if grp.empty:
            continue

        tab_name = issue_type.replace("_SPLIT", "").replace("_", " ")[:28]
        ws = wb.create_sheet(tab_name)
        ws.sheet_properties.tabColor = TAB_COLORS.get(issue_type, "808080")
        ws.freeze_panes = "A3"

        # строка-заголовок листа
        n_cols = len(COLS)
        ws.merge_cells(start_row=1, start_column=1,
                       end_row=1,   end_column=n_cols)
        tc = ws.cell(row=1, column=1,
                     value=f"{ISSUE_NAMES.get(issue_type, issue_type)}  ({len(grp)} шт.)")
        tc.font      = Font(name="Arial", bold=True, size=12, color="FFFFFF")
        tc.fill      = _fill(TAB_COLORS.get(issue_type, "808080"))
        tc.alignment = Alignment(horizontal="left", vertical="center", indent=1)
        ws.row_dimensions[1].height = 28

        _write_header_row(ws, [c[1] for c in COLS], row=2)

        CENTER_FIELDS = {"confidence", "group_a", "group_b",
                         "suffix_or_prefix", "name_similarity"}
        WRAP_FIELDS   = {"name_a", "name_b", "comment"}

        for r_idx, (_, row_data) in enumerate(grp.iterrows(), 3):
            conf = str(row_data.get("confidence", "LOW"))
            alt  = (r_idx % 2 == 0)

            for c_idx, (field, _, _width) in enumerate(COLS, 1):
                val = row_data.get(field, "")
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    val = ""
                if field == "confidence":
                    val = CONF_RU.get(str(val), val)
                if field == "name_similarity" and val != "":
                    try:
                        val = float(val)
                    except (TypeError, ValueError):
                        pass

                cell = ws.cell(row=r_idx, column=c_idx, value=val)
                cell.border    = _border("left", "right", "bottom")
                cell.alignment = Alignment(
                    vertical="center",
                    wrap_text=(field in WRAP_FIELDS),
                    horizontal="center" if field in CENTER_FIELDS else "left",
                )

                if field == "confidence":
                    cell.fill = _fill(CLR[f"{conf}_bg"])
                    cell.font = _body_font(bold=True, color=CLR[f"{conf}_fg"])
                elif alt:
                    cell.fill = _fill(CLR["alt_row"])
                    cell.font = _body_font()
                else:
                    cell.font = _body_font()

                if field == "name_similarity" and val != "":
                    cell.number_format = "0.00"

            ws.row_dimensions[r_idx].height = 28

        for c_idx, (_, _, w) in enumerate(COLS, 1):
            ws.column_dimensions[get_column_letter(c_idx)].width = w

    wb.save(path)
    print(f"✅ Отчёт сохранён: {path}  ({len(audit_df)} проблем, {len(wb.sheetnames)} листов)")