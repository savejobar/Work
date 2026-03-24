import re
from collections import defaultdict
from typing import Any

import pandas as pd

from preprocessing.normalization import article_forms


def find_all_analogs(
    start: Any,
    graph: dict[Any, set[Any]],
) -> tuple[Any, ...]:
    """
    Обход в глубину по графу аналогов начиная с узла start.
    """
    visited = set()
    stack = [start]

    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            stack.extend(graph[node] - visited)

    return tuple(sorted(visited))


def build_analog_graph(df: pd.DataFrame) -> defaultdict[str, set[str]]:
    """
    Строит граф связей между артикулами-аналогами.

    Рёбра соединяют:
      - разные формы одного артикула (с суффиксом/префиксом и без);
      - артикул и каждый из его аналогов из колонки «Аналоги».
    """
    graph: defaultdict[str, set[str]] = defaultdict(set)

    for _, row in df.iterrows():
        part_forms = article_forms(row["Номенклатура.Артикул"])

        # связываем формы одного артикула между собой
        for i, pf_i in enumerate(part_forms):
            for pf_j in part_forms[i + 1:]:
                graph[pf_i].add(pf_j)
                graph[pf_j].add(pf_i)

        # связываем с аналогами
        for analog_raw in (row["Аналоги"] or []):
            for af in article_forms(analog_raw):
                for pf in part_forms:
                    if pf != af:
                        graph[pf].add(af)
                        graph[af].add(pf)

    return graph


def _merge_analog_tuples(series: pd.Series) -> tuple | None:
    """
    Объединяет все tuple-значения из серии в один отсортированный tuple.
    """
    tuples = [v for v in series if isinstance(v, tuple)]
    if not tuples:
        return None
    merged = sorted({item for tpl in tuples for item in tpl})
    return tuple(merged)


def normalize_analog_lists(
    df: pd.DataFrame,
    col_group: str = "Номер группы",
    col_analogs: str = "Список аналогов",
) -> pd.DataFrame:
    """
    Для каждой группы выбирает самый длинный кортеж аналогов
    и проставляет его всем строкам группы.
    """
    df = df.copy()

    group_max = (
        df.groupby(col_group)[col_analogs]
        .apply(_merge_analog_tuples)
        .to_dict()
    )

    mapped = df[col_group].map(group_max)
    has_tuple = mapped.apply(lambda x: isinstance(x, tuple))
    df.loc[has_tuple, col_analogs] = mapped[has_tuple]

    return df


def consolidate_extended_article_numbers(row: pd.Series) -> str | None:
    """
    Объединяет основной, оригинальный и расширенный номера запчасти
    в единую строку через пробел. Возвращает None если результат пустой.
    """
    main_art = str(row["Номенклатура.Артикул"]).strip().upper()
    extended = row["Номенклатура.Оригинальный номер расширенный"]
    original = row["Номенклатура.Оригинальный номер"]

    extended_parts: list[str] = []
    if pd.notna(extended) and str(extended).strip():
        extended_parts = [
            p.strip()
            for p in re.split(r"\s+", str(extended))
            if p.strip()
        ]

    original_val = (
        str(original).strip()
        if pd.notna(original) and str(original).strip()
        else None
    )

    extended_upper = {p.upper() for p in extended_parts}

    if original_val:
        up_original = original_val.upper()
        if up_original not in extended_upper and up_original != main_art:
            extended_parts.append(original_val)

    unique_parts = sorted(set(extended_parts))
    return " ".join(unique_parts) if unique_parts else None
