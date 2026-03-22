import pandas as pd

from preprocessing.normalization import safe_to_int


def shortest_value(series: pd.Series) -> str | None:
    """Выбирает самое короткое строковое значение из Series."""
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
    Агрегирует складские показатели (остатки и обороты) по группам аналогов.
    """
    df = df.copy()

    meta = (
        df.groupby("Номер группы", as_index=False)
        .agg(
            Номенклатура=("Номенклатура", shortest_value),
            Список_аналогов=("Список аналогов", lambda x: max(
                x.dropna(), key=len, default=None
            )),
            Артикул=("Артикул", "first"),
        )
        .rename(columns={"Список_аналогов": "Список аналогов"})
    )

    qty = (
        df.groupby(["Год", "Месяц", "Номер группы"], as_index=False)
        .agg(
            Расход=("Расход", "sum"),
            Конечный_остаток=("Конечный остаток", "last"),
        )
        .rename(columns={"Конечный_остаток": "Конечный остаток"})
    )

    for col in ["Расход", "Конечный остаток", "Номер группы"]:
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
    """
    META_COLS = ["Номенклатура", "Артикул", "Список аналогов"]
    ZERO_COLS = ["Продажа", "Ремонт"]
    FFILL_COLS = ["Конечный остаток"]

    meta_cols_present = [c for c in META_COLS  if c in df.columns]
    zero_cols_present = [c for c in ZERO_COLS  if c in df.columns]
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
        active.assign(_start_ym=active["Год"] * 100 + active["Месяц"])
        .groupby("Номер группы", as_index=False)["_start_ym"]
        .min()
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
