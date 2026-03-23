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

    # Метаданные группы делаем детерминированными, без зависимости от случайного порядка строк.
    meta = (
        df.groupby("Номер группы", as_index=False)
        .agg(
            Номенклатура=("Номенклатура", shortest_value),
            Список_аналогов=(
                "Список аналогов",
                lambda x: max(x.dropna().astype(str), key=len, default=None),
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
    """
    Заполняет пропущенные месяцы по группам.

    Правила:
    - Конечный остаток тянем вперёд как состояние (ffill), но не рисуем искусственный 0 до первого факта.
    - Продажа и Ремонт заполняются нулём только начиная с первого НЕНУЛЕВОГО значения
    в соответствующей колонке внутри группы.
    """
    df = df.copy()

    if df.empty:
        df["is_synthetic"] = pd.Series(dtype="Int8")
        return df

    df["_original"] = 1
    df["_ym"] = df["Год"] * 100 + df["Месяц"]

    all_months = (
        df[["Год", "Месяц", "_ym"]]
        .drop_duplicates()
        .sort_values(["Год", "Месяц"])
        .reset_index(drop=True)
    )


    start_candidates = []

    if "Конечный остаток" in df.columns:
        first_stock = (
            df.loc[df["Конечный остаток"].notna()]
            .groupby("Номер группы")["_ym"]
            .min()
            .rename("stock_start")
        )
        start_candidates.append(first_stock)
    else:
        first_stock = pd.Series(dtype="float64")

    first_nonzero_by_col: dict[str, pd.Series] = {}
    for col in ["Продажа", "Ремонт"]:
        if col in df.columns:
            first_nonzero = (
                df.loc[df[col].fillna(0) != 0]
                .groupby("Номер группы")["_ym"]
                .min()
                .rename(f"{col}_start")
            )
            first_nonzero_by_col[col] = first_nonzero
            start_candidates.append(first_nonzero)
        else:
            first_nonzero_by_col[col] = pd.Series(dtype="float64")

    if start_candidates:
        start_df = pd.concat(start_candidates, axis=1)
        group_start = start_df.min(axis=1, skipna=True)
    else:
        group_start = pd.Series(dtype="float64")

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

    if "Конечный остаток" in merged.columns:
        merged["Конечный остаток"] = (
            merged.groupby("Номер группы")["Конечный остаток"]
            .ffill()
        )

        if not first_stock.empty:
            stock_start_map = merged["Номер группы"].map(first_stock)
            merged.loc[
                stock_start_map.notna() & (merged["_ym"] < stock_start_map),
                "Конечный остаток",
            ] = pd.NA

    for col in ["Продажа", "Ремонт"]:
        if col not in merged.columns:
            continue

        start_map = merged["Номер группы"].map(first_nonzero_by_col[col])
        fill_mask = start_map.notna() & (merged["_ym"] >= start_map) & merged[col].isna()
        merged.loc[fill_mask, col] = 0

    merged["is_synthetic"] = merged["_original"].isna().astype("Int8")
    merged = merged.drop(columns=[c for c in ["_original", "_ym"] if c in merged.columns])

    for col in ["Продажа", "Ремонт", "Конечный остаток", "Номер группы"]:
        if col in merged.columns:
            merged[col] = merged[col].apply(safe_to_int)

    return merged.reset_index(drop=True)
