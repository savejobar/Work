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
    """
    df = df.copy()

    meta = (
        df.groupby("Номер группы", as_index=False)
        .agg(Номенклатура=("Номенклатура", shortest_value))
    )

    qty = (
        df.groupby(["Год", "Месяц", "Номер группы"], as_index=False)
        .agg(Ремонт=("Количество", "sum"))
    )

    return qty.merge(meta, on="Номер группы", how="left")


def aggregate_stock_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Агрегирует складские показатели по группам аналогов.

    Логика:
    1. Внутри месяца берём последнее состояние остатка по каждой позиции группы.
    2. Затем протягиваем остаток по каждой позиции вперёд по месяцам (ffill).
    3. После этого суммируем остатки по группе.
    """
    df = df.copy()

    df["_src_order"] = range(len(df))

    df["_part_key"] = df["Артикул"].where(df["Артикул"].notna() & df["Артикул"].ne(""), df["Номенклатура"])

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



    # 1. Последний остаток по позиции внутри месяца.
    stock_by_part = (
        df.loc[df["Конечный остаток"].notna()]
        .sort_values(
            ["Номер группы", "_part_key", "Год", "Месяц", "_src_order"],
            kind="mergesort",
        )
        .groupby(["Номер группы", "_part_key", "Год", "Месяц"], as_index=False)
        .tail(1)
        [["Номер группы", "_part_key", "Год", "Месяц", "Конечный остаток"]]
    )

    # 2. Протягиваем остаток по каждой позиции только начиная с первого месяца её появления.
    all_months = (
        df[["Год", "Месяц"]]
        .drop_duplicates()
        .sort_values(["Год", "Месяц"])
        .reset_index(drop=True)
    )
    all_months["_ym"] = all_months["Год"] * 100 + all_months["Месяц"]

    part_start = (
        stock_by_part.assign(_ym=stock_by_part["Год"] * 100 + stock_by_part["Месяц"])
        .groupby(["Номер группы", "_part_key"], as_index=False)["_ym"]
        .min()
        .rename(columns={"_ym": "_start_ym"})
    )

    stock_grid = part_start.merge(all_months, how="cross")
    stock_grid = stock_grid[stock_grid["_ym"] >= stock_grid["_start_ym"]]

    stock_grid = (
        stock_grid.merge(
            stock_by_part,
            on=["Номер группы", "_part_key", "Год", "Месяц"],
            how="left",
        )
        .sort_values(["Номер группы", "_part_key", "_ym"], kind="mergesort")
        .reset_index(drop=True)
    )

    stock_grid["Конечный остаток"] = (
        stock_grid.groupby(["Номер группы", "_part_key"])["Конечный остаток"]
        .ffill()
    )

    # 3. Сумма остатков по группе. До первого факта оставляем NaN, а не подменяем его нулём.
    stock_summed = (
        stock_grid.groupby(["Год", "Месяц", "Номер группы"], as_index=False)["Конечный остаток"]
        .sum(min_count=1)
    )

    # Поток за месяц суммируем отдельно.
    qty = (
        df.groupby(["Год", "Месяц", "Номер группы"], as_index=False)
        .agg(Расход=("Расход", "sum"))
    )

    res = qty.merge(stock_summed, on=["Год", "Месяц", "Номер группы"], how="outer")

    for col in ["Расход", "Конечный остаток", "Номер группы"]:
        if col in res.columns:
            res[col] = res[col].apply(safe_to_int)

    return res.merge(meta, on="Номер группы", how="left")


def calculate_external_sales(
    stock_agg: pd.DataFrame,
    repair_agg: pd.DataFrame,
) -> pd.DataFrame:
    """
    Рассчитывает чистые внешние продажи:
    Продажа = Расход (склад) − Ремонт (внутреннее потребление)
    """
    merged = stock_agg.merge(
        repair_agg[["Год", "Месяц", "Номер группы", "Ремонт"]],
        on=["Год", "Месяц", "Номер группы"],
        how="left",
    )
    merged["Продажа"] = merged["Расход"] - merged["Ремонт"].fillna(0)

    return merged.drop(columns=["Расход", "Ремонт"])


def fill_missing_months(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if df.empty:
        return df.copy()

    df["_ym"] = df["Год"] * 100 + df["Месяц"]

    all_months = (
        df[["Год", "Месяц"]]
        .drop_duplicates()
        .sort_values(["Год", "Месяц"])
        .reset_index(drop=True)
    )
    all_months["_ym"] = all_months["Год"] * 100 + all_months["Месяц"]

    start_candidates = []

    # Остаток: старт с первого ИЗВЕСТНОГО значения, не с первого ненулевого
    if "Конечный остаток" in df.columns:
        first_stock = (
            df.loc[df["Конечный остаток"].notna()]
            .groupby("Номер группы")["_ym"]
            .min()
            .rename("stock_start")
        )
        start_candidates.append(first_stock)
    else:
        first_stock = pd.Series(dtype="Int64")

    # Продажа/Ремонт: старт с первого НЕНУЛЕВОГО значения независимо по каждой колонке
    first_nonzero_by_col: dict[str, pd.Series] = {}
    for col in ["Продажа", "Ремонт"]:
        if col in df.columns:
            numeric_col = pd.to_numeric(df[col], errors="coerce").fillna(0)
            first_nonzero = (
                df.loc[numeric_col != 0]
                .groupby("Номер группы")["_ym"]
                .min()
                .rename(f"{col}_start")
            )
            first_nonzero_by_col[col] = first_nonzero
            start_candidates.append(first_nonzero)
        else:
            first_nonzero_by_col[col] = pd.Series(dtype="Int64")

    if start_candidates:
        start_df = pd.concat(start_candidates, axis=1)
        group_start = start_df.min(axis=1, skipna=True)
    else:
        group_start = pd.Series(dtype="Int64")

    fallback_start = df.groupby("Номер группы")["_ym"].min()
    group_start = group_start.reindex(fallback_start.index).fillna(fallback_start)
    group_start = group_start.rename("_start_ym").reset_index()

    grid = group_start.merge(all_months, how="cross")
    grid = grid.loc[grid["_ym"] >= grid["_start_ym"]].drop(columns="_start_ym")

    merged = (
        grid.merge(df.drop(columns="_ym"), on=["Год", "Месяц", "Номер группы"], how="left")
        .sort_values(["Номер группы", "Год", "Месяц"], kind="mergesort")
        .reset_index(drop=True)
    )

    merged["_ym"] = merged["Год"] * 100 + merged["Месяц"]

    meta_cols = ["Номенклатура", "Артикул", "Список аналогов"]
    for col in [c for c in meta_cols if c in merged.columns]:
        merged[col] = merged.groupby("Номер группы")[col].ffill().bfill()

    # Остаток: тянем как состояние
    if "Конечный остаток" in merged.columns:
        merged["Конечный остаток"] = (
            merged.groupby("Номер группы")["Конечный остаток"].ffill()
        )

        if not first_stock.empty:
            stock_start_map = merged["Номер группы"].map(first_stock)
            merged.loc[
                stock_start_map.notna() & (merged["_ym"] < stock_start_map),
                "Конечный остаток",
            ] = pd.NA

    # Продажа/Ремонт: нули только с первого ненулевого значения КАЖДОЙ колонки
    for col in ["Продажа", "Ремонт"]:
        if col not in merged.columns:
            continue

        start_map = merged["Номер группы"].map(first_nonzero_by_col[col])

        # До старта колонки оставляем NaN
        merged.loc[
            start_map.notna() & (merged["_ym"] < start_map),
            col,
        ] = pd.NA

        # После старта колонки пропуски заполняем нулем
        merged.loc[
            start_map.notna() & (merged["_ym"] >= start_map) & merged[col].isna(),
            col,
        ] = 0

    value_cols = [c for c in ["Конечный остаток", "Продажа", "Ремонт"] if c in merged.columns]
    merged["is_synthetic"] = merged[value_cols].isna().all(axis=1).astype("Int8")

    return merged.drop(columns="_ym").reset_index(drop=True)
