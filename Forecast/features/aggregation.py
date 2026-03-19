import pandas as pd

from preprocessing.normalization import safe_to_int


def join_ordered(series: pd.Series) -> str | None:
    """
    Объединяет уникальные непустые элементы Series через '; ',
    сохраняя порядок первого появления.
    """
    vals   = series.dropna().astype(str).str.strip()
    seen:   set[str]  = set()
    result: list[str] = []
    for v in vals:
        if v and v not in seen:
            seen.add(v)
            result.append(v)
    return "; ".join(result) if result else None


def shortest_value(series: pd.Series) -> str | None:
    """Выбирает самое короткое строковое значение из Series."""
    vals = series.dropna().astype(str).str.strip()
    return min(vals, key=len) if len(vals) > 0 else None


def count_unique(series: pd.Series) -> int:
    """Подсчитывает количество уникальных непустых строковых значений."""
    vals = series.dropna().astype(str).str.strip()
    return len({v for v in vals if v})


def aggregate_repair_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Агрегирует данные по группам аналогов (ремонт).

    Рассчитывает суммарное количество деталей в ремонт,
    средние показатели по технике и объединяет метаданные.

    Returns:
        DataFrame с одной строкой на (Год, Месяц, Номер группы).
    """
    df = df.copy()

    meta = (
        df.groupby("Номер группы", as_index=False)
        .agg(
            Номенклатура=("Номенклатура",             shortest_value),
            Бренды=("Машина.Бренд",                   join_ordered),
            Серии_техники=("Машина.Серия техники",    join_ordered),
            Типы_подъемников=("Тип подъемника",       join_ordered),
            Типы_двигателя=("Тип двигателя",          join_ordered),
            Тип_техники=("Номенклатура.Тип техники",  join_ordered),
            Склады_ремонт=("Документ.Склад",          join_ordered),
            Номера_машин=("Номера машин",              join_ordered),
            Список_аналогов=("all_analogs",            "first"),
        )
        .rename(columns={
            "Серии_техники":    "Серии техники",
            "Типы_подъемников": "Типы подъемников",
            "Типы_двигателя":   "Типы двигателя",
            "Тип_техники":      "Номенклатура.Тип техники",
            "Номера_машин":     "Номера машин",
            "Список_аналогов":  "Список аналогов",
        })
    )

    qty = (
        df.groupby(["Год", "Месяц", "Номер группы"], as_index=False)
        .agg(
            Ремонт=("Количество",                  "sum"),
            Средний_год_выпуска=("Лот.CRM год выпуска",    "mean"),
            Средняя_наработка_лет=("Средняя наработка (лет)", "mean"),
            Уникальных_машин=("Серийный номер",    count_unique),
            Средняя_наработка_ч=("Лот.CRM наработка", "mean"),
        )
        .rename(columns={
            "Средний_год_выпуска":   "Средний год выпуска",
            "Средняя_наработка_лет": "Средняя наработка (лет)",
            "Уникальных_машин": (
                "Количество уникальных номеров машин, где запчасть применялась"
            ),
            "Средняя_наработка_ч": (
                "Средняя наработка техники, где запчасть применялась (ч)"
            ),
        })
    )

    round_cols = [
        "Средний год выпуска",
        "Средняя наработка (лет)",
        "Средняя наработка техники, где запчасть применялась (ч)",
    ]
    qty[round_cols] = qty[round_cols].round(1)

    return qty.merge(meta, on="Номер группы", how="left")


def aggregate_stock_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Агрегирует складские показатели (остатки и обороты) по группам аналогов.

    Безопасно конвертирует целые числа без потери данных через safe_to_int().

    Returns:
        DataFrame с одной строкой на (Год, Месяц, Номер группы).
    """
    df = df.copy()

    meta = (
        df.groupby("Номер группы", as_index=False)
        .agg(
            Номенклатура=("Номенклатура",  shortest_value),
            Склады_продажа=("Склады_продажа", join_ordered),
            Список_аналогов=("Список аналогов", lambda x: max(
                x.dropna(), key=len, default=None
            )),
            Тип_техники=("Тип техники", join_ordered),
            Артикул=("Артикул", "first"),
        )
        .rename(columns={
            "Список_аналогов": "Список аналогов",
            "Тип_техники":     "Тип техники",
        })
    )

    qty = (
        df.groupby(["Год", "Месяц", "Номер группы"], as_index=False)
        .agg(
            Расход=("Расход",                "sum"),
            Приход=("Приход",                "sum"),
            Конечный_остаток=("Конечный остаток", "last"),
        )
        .rename(columns={"Конечный_остаток": "Конечный остаток"})
    )

    for col in ["Расход", "Приход", "Конечный остаток", "Номер группы"]:
        if col in qty.columns:
            qty[col] = qty[col].apply(safe_to_int)

    return qty.merge(meta, on="Номер группы", how="left")


def calculate_external_sales(
    stock_agg: pd.DataFrame,
    repair_agg: pd.DataFrame,
) -> pd.DataFrame:
    """
    Рассчитывает чистые внешние продажи:
        Продажа = Расход (склад) − Ремонт (внутреннее потребление)

    Args:
        stock_agg:  Агрегированный склад (из aggregate_stock_groups).
        repair_agg: Агрегированные ремонты (из aggregate_repair_groups).

    Returns:
        DataFrame с колонкой «Продажа» и без «Расход» / «Ремонт».
    """
    merged = stock_agg.merge(
        repair_agg[["Год", "Месяц", "Номер группы", "Ремонт"]],
        on=["Год", "Месяц", "Номер группы"],
        how="left",
    )
    merged["Продажа"] = merged["Расход"] - merged["Ремонт"].fillna(0)
    return merged.drop(columns=["Расход", "Ремонт"])


def fill_missing_months(df: pd.DataFrame) -> pd.DataFrame:
    """
    Заполняет пропущенные месяцы в временном ряду каждой группы.

    Правила заполнения:
      - Метаданные (Бренды, Склады и т.д.) — берём из первой известной строки группы.
      - Количественные (Продажа, Ремонт, Приход) — 0.
      - Конечный остаток — ffill внутри группы.
      - Строки до первой активности группы (Продажа > 0 или Ремонт > 0)
        не добавляются.

    Добавляет колонку is_synthetic:
        1    = синтетическая строка (добавлена этой функцией)
        <NA> = реальная строка из исходных данных

    Returns:
        DataFrame с заполненными месяцами, отсортированный по Год, Месяц, Номер группы.
    """
    META_COLS = [
        "Бренды", "Серии техники", "Типы подъемников", "Типы двигателя",
        "Номенклатура.Тип техники", "Склады_продажа", "Номера машин",
        "Склады_ремонт", "Тип техники", "Артикул", "Номенклатура",
        "Список аналогов",
    ]
    ZERO_COLS  = ["Продажа", "Ремонт", "Приход"]
    FFILL_COLS = ["Конечный остаток"]

    meta_cols_present  = [c for c in META_COLS  if c in df.columns]
    zero_cols_present  = [c for c in ZERO_COLS  if c in df.columns]
    ffill_cols_present = [c for c in FFILL_COLS if c in df.columns]

    all_months = (
        df[["Год", "Месяц"]]
        .drop_duplicates()
        .sort_values(["Год", "Месяц"])
        .reset_index(drop=True)
    )
    all_months["_ym"] = all_months["Год"] * 100 + all_months["Месяц"]

    active = df[(df["Продажа"] > 0) | (df["Ремонт"] > 0)]
    group_start = (
        active.groupby("Номер группы")
        .apply(lambda g: (g["Год"] * 100 + g["Месяц"]).min())
        .rename("_start_ym")
        .reset_index()
    )

    grid = group_start.merge(all_months, how="cross")
    grid = grid[grid["_ym"] >= grid["_start_ym"]].drop(
        columns=["_start_ym", "_ym"]
    )

    df = df.copy()
    df["_original"] = 1

    merged = grid.merge(df, on=["Год", "Месяц", "Номер группы"], how="left")

    if meta_cols_present:
        meta = df.groupby("Номер группы")[meta_cols_present].first()
        for col in meta_cols_present:
            merged[col] = merged[col].combine_first(
                merged["Номер группы"].map(meta[col])
            )

    if zero_cols_present:
        merged[zero_cols_present] = merged[zero_cols_present].fillna(0)

    merged = merged.sort_values(["Номер группы", "Год", "Месяц"])

    for col in ["Продажа", "Ремонт"]:
        if col not in merged.columns:
            continue
        mask = merged.groupby("Номер группы")[col].transform(
            lambda s: (s > 0).cummax()
        )
        merged.loc[~mask, col] = pd.NA

    if ffill_cols_present:
        merged[ffill_cols_present] = (
            merged.groupby("Номер группы")[ffill_cols_present]
            .transform(lambda s: s.ffill())
        )

    merged = (
        merged
        .sort_values(["Год", "Месяц", "Номер группы"])
        .reset_index(drop=True)
    )
    merged["is_synthetic"] = (
        merged["_original"].isna().replace(False, pd.NA).astype("Int8")
    )
    merged.drop(columns="_original", inplace=True)

    return merged
