import os
from pathlib import Path

import pandas as pd
import pytest

from features.aggregation import aggregate_stock_groups, fill_missing_months
from readers.loaders import preprocess_stock_report


# ----------------------------
# Helpers
# ----------------------------

def _row(df: pd.DataFrame, group_id: int, year: int, month: int) -> pd.Series:
    hit = df.loc[
        (df["Номер группы"] == group_id)
        & (df["Год"] == year)
        & (df["Месяц"] == month)
    ]
    assert len(hit) == 1, f"Ожидалась ровно 1 строка для group={group_id}, {year}-{month}, получено {len(hit)}"
    return hit.iloc[0]


def assert_stock_aggregation_invariants(raw: pd.DataFrame, agg: pd.DataFrame) -> None:
    """
    Общие инварианты, которые должны держаться всегда.
    Это можно вызывать и в тестах, и прямо после агрегации в debug-режиме.
    """
    assert not agg.empty, "aggregate_stock_groups вернула пустой DataFrame"
    assert {"Год", "Месяц", "Номер группы", "Расход", "Конечный остаток"}.issubset(agg.columns)

    dupes = agg.duplicated(["Год", "Месяц", "Номер группы"]).sum()
    assert dupes == 0, f"После агрегации остались дубли group-month: {dupes}"

    # Поток за месяц должен сохраняться строго как сумма raw Расход по group-month.
    expected_flow = (
        raw.groupby(["Год", "Месяц", "Номер группы"], as_index=False)["Расход"]
        .sum()
        .rename(columns={"Расход": "Расход_expected"})
    )

    chk = agg.merge(expected_flow, on=["Год", "Месяц", "Номер группы"], how="left")
    bad = chk[chk["Расход"] != chk["Расход_expected"]]
    assert bad.empty, (
        "Нарушено сохранение потока Расход.\n"
        f"Проблемные строки:\n{bad[['Год', 'Месяц', 'Номер группы', 'Расход', 'Расход_expected']].to_string(index=False)}"
    )


# ----------------------------
# Unit tests for aggregate_stock_groups
# ----------------------------

def test_aggregate_stock_groups_takes_last_stock_within_month_and_ffills_by_part():
    """
    Кейс:
    - в январе по детали A две записи -> должен взяться последний остаток (8), а расход суммироваться (1+2)
    - в феврале появляется деталь B с остатком 7
    - остаток A должен протянуться в февраль
    Итог по группе в феврале = A(8) + B(7) = 15
    """
    df = pd.DataFrame([
        {
            "Год": 2024, "Месяц": 1, "Номер группы": 1,
            "Номенклатура": "Фильтр A", "Артикул": "A",
            "Список аналогов": "('A', 'B')",
            "Расход": 1, "Конечный остаток": 10,
        },
        {
            "Год": 2024, "Месяц": 1, "Номер группы": 1,
            "Номенклатура": "Фильтр A", "Артикул": "A",
            "Список аналогов": "('A', 'B')",
            "Расход": 2, "Конечный остаток": 8,   # последняя запись января по A
        },
        {
            "Год": 2024, "Месяц": 2, "Номер группы": 1,
            "Номенклатура": "Фильтр B", "Артикул": "B",
            "Список аналогов": "('A', 'B')",
            "Расход": 1, "Конечный остаток": 7,
        },
    ])

    res = (
        aggregate_stock_groups(df)
        .sort_values(["Номер группы", "Год", "Месяц"])
        .reset_index(drop=True)
    )

    assert_stock_aggregation_invariants(df, res)

    jan = _row(res, 1, 2024, 1)
    feb = _row(res, 1, 2024, 2)

    assert jan["Расход"] == 3
    assert jan["Конечный остаток"] == 8

    assert feb["Расход"] == 1
    assert feb["Конечный остаток"] == 15


def test_aggregate_stock_groups_does_not_backfill_part_before_first_appearance():
    """
    Кейс:
    - деталь B существует с января (остаток 4)
    - деталь A появляется только в марте (остаток 5)
    - в феврале A не должна внезапно участвовать в сумме
    Ожидание:
      янв = 4
      фев = 4
      мар = 9
    """
    df = pd.DataFrame([
        {
            "Год": 2024, "Месяц": 1, "Номер группы": 7,
            "Номенклатура": "Фильтр B", "Артикул": "B",
            "Список аналогов": "('A', 'B')",
            "Расход": 1, "Конечный остаток": 4,
        },
        {
            "Год": 2024, "Месяц": 2, "Номер группы": 7,
            "Номенклатура": "Фильтр B", "Артикул": "B",
            "Список аналогов": "('A', 'B')",
            "Расход": 1, "Конечный остаток": pd.NA,
        },
        {
            "Год": 2024, "Месяц": 3, "Номер группы": 7,
            "Номенклатура": "Фильтр A", "Артикул": "A",
            "Список аналогов": "('A', 'B')",
            "Расход": 0, "Конечный остаток": 5,
        },
    ])

    res = (
        aggregate_stock_groups(df)
        .sort_values(["Номер группы", "Год", "Месяц"])
        .reset_index(drop=True)
    )

    assert_stock_aggregation_invariants(df, res)

    jan = _row(res, 7, 2024, 1)
    feb = _row(res, 7, 2024, 2)
    mar = _row(res, 7, 2024, 3)

    assert jan["Конечный остаток"] == 4
    assert feb["Конечный остаток"] == 4
    assert mar["Конечный остаток"] == 9


def test_aggregate_stock_groups_keeps_nan_before_any_stock_fact_in_group_month():
    """
    Кейс:
    - в январе по группе есть только расход, но нет ни одного факта остатка
    - первый остаток появляется только в феврале
    Январский остаток не должен подменяться нулем.
    """
    df = pd.DataFrame([
        {
            "Год": 2024, "Месяц": 1, "Номер группы": 10,
            "Номенклатура": "Фильтр X", "Артикул": "X",
            "Список аналогов": "('X',)",
            "Расход": 3, "Конечный остаток": pd.NA,
        },
        {
            "Год": 2024, "Месяц": 2, "Номер группы": 10,
            "Номенклатура": "Фильтр X", "Артикул": "X",
            "Список аналогов": "('X',)",
            "Расход": 1, "Конечный остаток": 6,
        },
    ])

    res = (
        aggregate_stock_groups(df)
        .sort_values(["Номер группы", "Год", "Месяц"])
        .reset_index(drop=True)
    )

    assert_stock_aggregation_invariants(df, res)

    jan = _row(res, 10, 2024, 1)
    feb = _row(res, 10, 2024, 2)

    assert jan["Расход"] == 3
    assert pd.isna(jan["Конечный остаток"]), "До первого факта остатка должен оставаться NaN, а не 0"
    assert feb["Конечный остаток"] == 6


# ----------------------------
# Unit tests for fill_missing_months
# ----------------------------

def test_fill_missing_months_zeroes_sale_and_repair_only_from_first_nonzero():
    """
    Кейс:
    - остаток известен уже в январе
    - первая ненулевая Продажа только в апреле
    - первый ненулевой Ремонт только в мае
    Ожидание:
    - фев/мар синтетические
    - stock тянется с января
    - Продажа в фев/мар остается NaN
    - Ремонт в фев/мар/апр остается NaN
    - в мае Продажа = 0, потому что после старта продаж
    """
    df = pd.DataFrame([
        # Группа 1
        {
            "Год": 2024, "Месяц": 1, "Номер группы": 1,
            "Номенклатура": "Фильтр 1", "Артикул": "A1", "Список аналогов": "('A1',)",
            "Конечный остаток": 10, "Продажа": pd.NA, "Ремонт": pd.NA,
        },
        {
            "Год": 2024, "Месяц": 4, "Номер группы": 1,
            "Номенклатура": "Фильтр 1", "Артикул": "A1", "Список аналогов": "('A1',)",
            "Конечный остаток": 7, "Продажа": 3, "Ремонт": pd.NA,
        },
        {
            "Год": 2024, "Месяц": 5, "Номер группы": 1,
            "Номенклатура": "Фильтр 1", "Артикул": "A1", "Список аналогов": "('A1',)",
            "Конечный остаток": pd.NA, "Продажа": pd.NA, "Ремонт": 2,
        },

        # Техническая группа, чтобы в all_months были февраль и март
        {
            "Год": 2024, "Месяц": 2, "Номер группы": 99,
            "Номенклатура": "stub", "Артикул": "stub", "Список аналогов": "('stub',)",
            "Конечный остаток": 1, "Продажа": pd.NA, "Ремонт": pd.NA,
        },
        {
            "Год": 2024, "Месяц": 3, "Номер группы": 99,
            "Номенклатура": "stub", "Артикул": "stub", "Список аналогов": "('stub',)",
            "Конечный остаток": 1, "Продажа": pd.NA, "Ремонт": pd.NA,
        },
    ])

    filled = (
        fill_missing_months(df)
        .sort_values(["Номер группы", "Год", "Месяц"])
        .reset_index(drop=True)
    )

    g1 = filled[filled["Номер группы"] == 1].reset_index(drop=True)
    assert list(zip(g1["Год"], g1["Месяц"])) == [
        (2024, 1), (2024, 2), (2024, 3), (2024, 4), (2024, 5)
    ]

    jan = _row(g1, 1, 2024, 1)
    feb = _row(g1, 1, 2024, 2)
    mar = _row(g1, 1, 2024, 3)
    apr = _row(g1, 1, 2024, 4)
    may = _row(g1, 1, 2024, 5)

    assert jan["Конечный остаток"] == 10
    assert feb["Конечный остаток"] == 10
    assert mar["Конечный остаток"] == 10
    assert apr["Конечный остаток"] == 7
    assert may["Конечный остаток"] == 7

    assert pd.isna(feb["Продажа"])
    assert pd.isna(mar["Продажа"])
    assert apr["Продажа"] == 3
    assert may["Продажа"] == 0

    assert pd.isna(feb["Ремонт"])
    assert pd.isna(mar["Ремонт"])
    assert pd.isna(apr["Ремонт"])
    assert may["Ремонт"] == 2

    assert feb["is_synthetic"] == 1
    assert mar["is_synthetic"] == 1
    assert jan["is_synthetic"] == 0
    assert apr["is_synthetic"] == 0
    assert may["is_synthetic"] == 0


def test_fill_missing_months_keeps_stock_nan_before_first_stock_even_if_sales_started_earlier():
    """
    Кейс:
    - старт группы в январе идет из Продажи
    - первый факт остатка только в марте
    Ожидание:
    - январь и февраль по stock остаются NaN
    - Продажа после первого ненулевого месяца добивается нулем
    """
    df = pd.DataFrame([
        # Группа 5
        {
            "Год": 2024, "Месяц": 1, "Номер группы": 5,
            "Номенклатура": "Фильтр 5", "Артикул": "A5", "Список аналогов": "('A5',)",
            "Конечный остаток": pd.NA, "Продажа": 2, "Ремонт": pd.NA,
        },
        {
            "Год": 2024, "Месяц": 3, "Номер группы": 5,
            "Номенклатура": "Фильтр 5", "Артикул": "A5", "Список аналогов": "('A5',)",
            "Конечный остаток": 5, "Продажа": pd.NA, "Ремонт": pd.NA,
        },

        # Техническая группа для февраля
        {
            "Год": 2024, "Месяц": 2, "Номер группы": 99,
            "Номенклатура": "stub", "Артикул": "stub", "Список аналогов": "('stub',)",
            "Конечный остаток": 1, "Продажа": pd.NA, "Ремонт": pd.NA,
        },
    ])

    filled = (
        fill_missing_months(df)
        .sort_values(["Номер группы", "Год", "Месяц"])
        .reset_index(drop=True)
    )

    g5 = filled[filled["Номер группы"] == 5].reset_index(drop=True)
    assert list(zip(g5["Год"], g5["Месяц"])) == [
        (2024, 1), (2024, 2), (2024, 3)
    ]

    jan = _row(g5, 5, 2024, 1)
    feb = _row(g5, 5, 2024, 2)
    mar = _row(g5, 5, 2024, 3)

    assert pd.isna(jan["Конечный остаток"])
    assert pd.isna(feb["Конечный остаток"])
    assert mar["Конечный остаток"] == 5

    assert jan["Продажа"] == 2
    assert feb["Продажа"] == 0
    assert mar["Продажа"] == 0


# ----------------------------
# Smoke test on real source xlsx
# ----------------------------

@pytest.mark.smoke
def test_aggregate_stock_groups_smoke_on_real_source_xlsx():
    """
    Smoke на реальном источнике.
    Здесь НЕ тестируется grouping аналогов.
    Мы просто подаем в aggregate_stock_groups реальный формат отчета из 1С
    и создаем тех. Номер группы из article-like ключа, чтобы проверить инварианты агрегации.

    Запуск:
        STOCK_SOURCE_XLSX=/abs/path/to/Остатки и обороты.xlsx pytest -m smoke -q
    """
    src = os.getenv("STOCK_SOURCE_XLSX")
    if not src:
        pytest.skip("Не задан STOCK_SOURCE_XLSX")
    src_path = Path(src)
    if not src_path.exists():
        pytest.skip(f"Файл не найден: {src_path}")

    raw = preprocess_stock_report(str(src_path)).copy()

    # Это не тест на группировку аналогов.
    # Нам нужен валидный group id для smoke проверки самой агрегации.
    group_key = (
        raw["Артикул"]
        .astype("string")
        .fillna(raw["Оригинальный номер"].astype("string"))
        .fillna(raw["Номенклатура"].astype("string"))
        .str.strip()
    )
    raw["Номер группы"] = pd.factorize(group_key)[0] + 1
    raw["Список аналогов"] = pd.NA

    res = aggregate_stock_groups(raw)

    assert_stock_aggregation_invariants(raw, res)
    assert res["Конечный остаток"].notna().any(), "На реальном источнике не найдено ни одного остатка"
    assert res["Расход"].notna().any(), "На реальном источнике не найдено ни одного расхода"