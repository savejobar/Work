import pandas as pd

from preprocessing.normalization import safe_to_int


def shortest_value(series: pd.Series) -> str | None:
    """
    Выбирает самое короткое строковое значение из Series.
    """
    vals = series.dropna().astype(str).str.strip()
    return min(vals, key=len) if len(vals) > 0 else None


def aggregate_repair_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Агрегирует данные по группам аналогов (ремонт).

    Ремонт:
    - "Ремонт" — только по подъемникам, это идет в прогноз ремонта
    - "Ремонт не подъемники" — внутреннее потребление по прочей технике
    - "Ремонт всего" — сумма двух колонок, нужна для корректного расчета продаж
    """
    df = df.copy()

    meta = (
        df.groupby("Номер группы", as_index=False)
        .agg(
            Номенклатура=("Номенклатура", shortest_value),
            Список_аналогов=(
                "all_analogs",
                lambda x: max(
                    (v for v in x.dropna() if isinstance(v, tuple)),
                    key=len,
                    default=None,
                ),
            ),
            Артикул=("Номенклатура.Артикул", shortest_value),
        )
        .rename(columns={"Список_аналогов": "Список аналогов"})
    )

    df["Количество_подъемники"] = df["Количество"].where(df["Машина_подъемник"], 0)
    df["Количество_неподъемники"] = df["Количество"].where(~df["Машина_подъемник"], 0)

    qty = (
        df.groupby(["Год", "Месяц", "Номер группы"], as_index=False)
        .agg(
            Ремонт=("Количество_подъемники", "sum"),
            **{
                "Ремонт не подъемники": ("Количество_неподъемники", "sum"),
                "Ремонт всего": ("Количество", "sum"),
            },
        )
    )

    res = qty.merge(meta, on="Номер группы", how="left")

    for col in ["Ремонт", "Ремонт не подъемники", "Ремонт всего", "Номер группы"]:
        if col in res.columns:
            res[col] = res[col].apply(safe_to_int)

    return res


def aggregate_stock_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Агрегирует складские показатели по группам аналогов.

    Логика по остатку:
    1. Внутри месяца берём последнюю запись по каждому Артикул в группе.
    2. Протягиваем остаток по Артикул вперед по месяцам.
    3. Складываем остатки всех Артикул группы.
    """
    df = df.copy()
    df["_src_order"] = range(len(df))
    df["_ym"] = df["Год"] * 100 + df["Месяц"]

    meta = (
        df.groupby("Номер группы", as_index=False)
        .agg(
            Номенклатура=("Номенклатура", shortest_value),
            Список_аналогов=(
                "Список аналогов",
                lambda x: max(
                    (v for v in x.dropna() if isinstance(v, tuple)),
                    key=len,
                    default=None,
                ),
            ),
            Артикул=("Артикул", shortest_value),
        )
        .rename(columns={"Список_аналогов": "Список аналогов"})
    )

    qty = (
        df.groupby(["Год", "Месяц", "Номер группы"], as_index=False)
        .agg(
            Расход=("Расход", "sum"),
            Приход=("Приход", "sum"),
        )
    )

    stock_by_code = (
        df
        .sort_values(
            ["Номер группы", "Код", "Год", "Месяц", "_src_order"],
            kind="mergesort",
        )
        .groupby(["Номер группы", "Код", "Год", "Месяц"], as_index=False)
        .tail(1)
        [["Номер группы", "Код", "Год", "Месяц", "_ym", "Конечный остаток"]]
    )

    all_months = (
        df[["Год", "Месяц", "_ym"]]
        .drop_duplicates()
        .sort_values(["Год", "Месяц"])
        .reset_index(drop=True)
    )

    article_start = (
        stock_by_code
        .groupby(["Номер группы", "Код"], as_index=False)["_ym"]
        .min()
        .rename(columns={"_ym": "_start_ym"})
    )

    stock_grid = article_start.merge(all_months, how="cross")
    stock_grid = stock_grid.loc[stock_grid["_ym"] >= stock_grid["_start_ym"]]

    stock_grid = (
        stock_grid
        .merge(
            stock_by_code[["Номер группы", "Код", "Год", "Месяц", "Конечный остаток"]],
            on=["Номер группы", "Код", "Год", "Месяц"],
            how="left",
        )
        .sort_values(["Номер группы", "Код", "_ym"], kind="mergesort")
        .reset_index(drop=True)
    )

    stock_grid["Конечный остаток"] = (
        stock_grid
        .groupby(["Номер группы", "Код"])["Конечный остаток"]
        .ffill()
    )


    stock = (
        stock_grid
        .groupby(["Год", "Месяц", "Номер группы"], as_index=False)["Конечный остаток"]
        .sum(min_count=1)
    )

    res = qty.merge(
        stock,
        on=["Год", "Месяц", "Номер группы"],
        how="outer",
    )

    for col in ["Расход", "Конечный остаток", "Номер группы", "Приход"]:
        if col in res.columns:
            res[col] = res[col].apply(safe_to_int)

    return res.merge(meta, on="Номер группы", how="left")


def calculate_external_sales(
    stock_agg: pd.DataFrame,
    repair_agg: pd.DataFrame,
) -> pd.DataFrame:
    """
    Рассчитывает чистые внешние продажи:

    Продажа = Расход (склад) − Ремонт всего
    """
    merged = stock_agg.merge(
        repair_agg[["Год", "Месяц", "Номер группы", "Ремонт всего"]],
        on=["Год", "Месяц", "Номер группы"],
        how="left",
    )

    merged["Продажа"] = merged["Расход"] - merged["Ремонт всего"].fillna(0)

    return merged.drop(columns=["Расход", "Ремонт всего"])


def fill_flow_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Заполняет пропуски в потоковых колонках (например, Продажа/Ремонт).

    До первого ненулевого значения по группе оставляет NaN.
    Начиная с первого ненулевого значения, заполняет пропуски нулем.
    """
    df = df.copy()
    df["_ym"] = df["Год"] * 100 + df["Месяц"]

    for col in cols:
        numeric_col = pd.to_numeric(df[col], errors="coerce").fillna(0)

        first_nonzero = (
            df.loc[numeric_col != 0]
            .groupby("Номер группы")["_ym"]
            .min()
        )

        start_map = df["Номер группы"].map(first_nonzero)

        df.loc[
            start_map.notna() & (df["_ym"] < start_map),
            col,
        ] = pd.NA

        df.loc[
            start_map.notna() & (df["_ym"] >= start_map) & df[col].isna(),
            col,
        ] = 0

    return df.drop(columns="_ym")
