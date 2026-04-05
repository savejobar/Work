from collections import defaultdict

import pandas as pd

from features.aggregation import (
    aggregate_repair_groups,
    aggregate_stock_groups,
    fill_flow_columns,
)
from preprocessing.cleaning import (
    normalize_nomenclatures_repair_parts,
    normalize_nomenclatures_stock_report,
    fill_missing_article,
)
from preprocessing.grouping import (
    build_analog_graph,
    consolidate_extended_article_numbers,
    find_all_analogs,
    normalize_analog_lists,
)
from preprocessing.normalization import article_forms, normalize
from readers.loaders import preprocess_repair_parts, preprocess_stock_report


def _build_article_lookup(
    df1: pd.DataFrame,
) -> tuple[dict[str, int], dict[str, tuple[str, ...]]]:
    """
    Строит два словаря для быстрого поиска группы и списка аналогов
    по нормализованному артикулу.
    """
    article_to_group = {}
    article_to_analogs = {}

    for _, row in df1.drop_duplicates("Номенклатура.Артикул").iterrows():
        group = row["Номер группы"]
        analogs = row["all_analogs"]
        if not isinstance(analogs, tuple):
            raise TypeError(f"all_analogs must be tuple, got {type(analogs).__name__}")

        for art in analogs:
            article_to_group[art] = group
            article_to_analogs[art] = analogs

        main = row["Номенклатура.Артикул"]
        if pd.notna(main) and main:
            article_to_group[main] = group
            article_to_analogs[main] = analogs

    return article_to_group, article_to_analogs


def _lookup_group(
    art_norm: str | None,
    orig_norm: str | None,
    article_to_group: dict[str, int],
    article_to_analogs: dict[str, tuple[str, ...]],
) -> tuple[int | None, tuple[str, ...] | None]:
    """
    Ищет группу и аналоги по нормализованному артикулу или ориг.номеру.
    Проверяет все формы артикула (с суффиксом, без и т.д.).
    """
    for val in [art_norm, orig_norm]:
        if not val:
            continue
        for form in article_forms(val):
            if form in article_to_group:
                return article_to_group[form], article_to_analogs[form]
    return None, None


def _enrich_analogs(
    row: pd.Series,
    article_to_group: dict[str, int],
) -> tuple[str, ...] | None:
    """
    Добавляет артикул из df2 в список аналогов группы,
    если его ещё нет ни в аналогах, ни в article_to_group.
    """
    art = row["Артикул"]
    analogs = row["Список аналогов"]
    if pd.isnull(art) or not isinstance(analogs, tuple):
        return analogs
    if art not in analogs and art not in article_to_group:
        return tuple(sorted(analogs + (art,)))
    return analogs


def _drop_rows_without_identifiers(
    df: pd.DataFrame,
    article_col: str,
    original_col: str,
) -> pd.DataFrame:
    """
    Удаляет строки, где после нормализации не осталось ни одного идентификатора.
    """
    return df.loc[
        df[article_col].notna() | df[original_col].notna()
    ].copy()


def _sync_group_membership(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Синхронизирует состав аналогов между repair- и stock-данными
    для уже найденных групп.
    """
    art_by_group = (
        df2[df2["Номер группы"].notna()]
        .assign(Артикул_norm=lambda d: d["Артикул"].apply(normalize))
        .dropna(subset=["Артикул_norm"])
        .groupby("Номер группы")["Артикул_norm"]
        .apply(set)
        .to_dict()
    )

    df1_group_idx: defaultdict[int, list[int]] = defaultdict(list)
    for i, g in df1["Номер группы"].dropna().items():
        df1_group_idx[int(g)].append(i)

    df2_group_idx: defaultdict[int, list[int]] = defaultdict(list)
    for i, g in df2["Номер группы"].dropna().items():
        df2_group_idx[int(g)].append(i)

    for grp, arts in art_by_group.items():
        grp = int(grp)
        idxs_df1 = df1_group_idx.get(grp, [])
        if not idxs_df1:
            continue

        existing = df1.at[idxs_df1[0], "all_analogs"]

        new_arts = arts - set(existing)
        if not new_arts:
            continue

        new_analogs = tuple(sorted(set(existing) | new_arts))
        for i in idxs_df1:
            df1.at[i, "all_analogs"] = new_analogs
        for i in df2_group_idx.get(grp, []):
            df2.at[i, "Список аналогов"] = new_analogs

    return df1, df2


def run_full_pipeline(
    repair_path: str,
    stock_path: str,
) -> pd.DataFrame:
    """
    Полный ETL-пайплайн: загрузка → нормализация → группировка → агрегация.
    """
    df1 = preprocess_repair_parts(repair_path)
    df1 = normalize_nomenclatures_repair_parts(df1)

    for col in [
        "Номенклатура.Артикул",
        "Номенклатура.Оригинальный номер",
        "Номенклатура.Оригинальный номер расширенный",
    ]:
        df1[col] = df1[col].apply(normalize)
    
    df1 = fill_missing_article(df1, "Номенклатура.Артикул", "Номенклатура.Оригинальный номер")
    df1 = _drop_rows_without_identifiers(
        df1,
        "Номенклатура.Артикул",
        "Номенклатура.Оригинальный номер",
    )

    df1["Номенклатура.Оригинальный номер расширенный"] = df1.apply(
        consolidate_extended_article_numbers, axis=1
    )
    df1["Аналоги"] = (
        df1["Номенклатура.Оригинальный номер расширенный"]
        .fillna("")
        .str.upper()
        .str.split()
    )

    graph = build_analog_graph(df1)
    df1["all_analogs"] = df1["Номенклатура.Артикул"].apply(
        lambda x: find_all_analogs(x, graph)
    )
    df1 = df1.drop(columns="Аналоги")

    group_mapping: dict = {
        analog_tuple: idx
        for idx, analog_tuple in enumerate(df1["all_analogs"].unique(), start=1)
    }
    df1["Номер группы"] = df1["all_analogs"].apply(group_mapping.get).astype("Int64")
    df_quarter = aggregate_repair_groups(df1)

    df2 = preprocess_stock_report(stock_path)
    df2 = normalize_nomenclatures_stock_report(df2)

    df2["Артикул"] = df2["Артикул"].apply(normalize)
    df2["Оригинальный номер"] = df2["Оригинальный номер"].apply(normalize)
    df2 = fill_missing_article(df2, "Артикул", "Оригинальный номер")
    df2 = _drop_rows_without_identifiers(df2, "Артикул", "Оригинальный номер")

    article_to_group, article_to_analogs = _build_article_lookup(df1)

    if df2.empty:
        raise ValueError("Stock report is empty after normalization")
    
    groups, analogs_list = zip(
        *df2.apply(
            lambda r: _lookup_group(
                r["Артикул"], r["Оригинальный номер"],
                article_to_group, article_to_analogs,
            ),
            axis=1,
        )
    )
    
    df2["Номер группы"] = pd.Series(groups, index=df2.index, dtype="Int64")
    df2["Список аналогов"] = analogs_list

    df2["Список аналогов"] = df2.apply(
        lambda r: _enrich_analogs(r, article_to_group), axis=1
    )
    df2 = normalize_analog_lists(df2)

    df1, df2 = _sync_group_membership(df1, df2)
    article_to_group, article_to_analogs = _build_article_lookup(df1)

    unmatched_idx = df2[df2["Номер группы"].isna()].index
    relinked_idx: list[int] = []

    for idx in unmatched_idx:
        grp, analogs = _lookup_group(
            normalize(df2.at[idx, "Артикул"]),
            normalize(df2.at[idx, "Оригинальный номер"]),
            article_to_group,
            article_to_analogs,
        )
        if grp is None:
            continue

        df2.at[idx, "Номер группы"] = grp
        df2.at[idx, "Список аналогов"] = analogs
        relinked_idx.append(idx)

    if relinked_idx:
        df2.loc[relinked_idx, "Список аналогов"] = (
            df2.loc[relinked_idx]
            .apply(lambda r: _enrich_analogs(r, article_to_group), axis=1)
        )
        df2 = normalize_analog_lists(df2)
        df1, df2 = _sync_group_membership(df1, df2)
        article_to_group, article_to_analogs = _build_article_lookup(df1)

    graph_new = defaultdict(set)
    unmatched_idx = df2[df2["Номер группы"].isna()].index

    for idx in unmatched_idx:
        art = normalize(df2.at[idx, "Артикул"])
        orig = normalize(df2.at[idx, "Оригинальный номер"])

        if art is None:
            if orig is not None:
                art = orig
            else:
                continue

        all_forms = article_forms(art) + (article_forms(orig) if orig else [])
        if any(f in article_to_group for f in all_forms):
            continue

        for af in article_forms(art):
            for bf in article_forms(art):
                if af != bf:
                    graph_new[af].add(bf)
                    graph_new[bf].add(af)

        if orig is not None:
            for af in article_forms(art):
                for bf in article_forms(orig):
                    if af != bf:
                        graph_new[af].add(bf)
                        graph_new[bf].add(af)

    for idx in unmatched_idx:
        art = normalize(df2.at[idx, "Артикул"])
        orig = normalize(df2.at[idx, "Оригинальный номер"])
        if art is None:
            art = orig

        all_forms = article_forms(art) if art else []
        if any(f in article_to_group for f in all_forms):
            continue

        df2.at[idx, "Список аналогов"] = (
            find_all_analogs(art, graph_new) if art is not None else tuple()
        )

    unique_new = df2.loc[df2["Номер группы"].isna(), "Список аналогов"].drop_duplicates()
    new_group_start = int(df1["Номер группы"].max()) + 1
    new_group_map = {
        grp: new_group_start + i
        for i, grp in enumerate(unique_new)
    }
    mask_new = df2["Номер группы"].isna()
    df2.loc[mask_new, "Номер группы"] = df2.loc[mask_new, "Список аналогов"].apply(
        lambda x: new_group_map.get(x)
    )
    df2["Номер группы"] = df2["Номер группы"].astype("Int64")
    agg = aggregate_stock_groups(df2)

    agg = agg.rename(columns={"Расход": "Продажа"})

    final = pd.merge(
        df_quarter, agg,
        on=["Год", "Месяц", "Номер группы"],
        how="outer",
        suffixes=("_df", "_agg"),
    )

    cols_to_combine = [
        c for c in df_quarter.columns
        if c in agg.columns and c not in ["Год", "Месяц", "Номер группы"]
    ]
    for col in cols_to_combine:
        final[col] = final[f"{col}_df"].combine_first(final[f"{col}_agg"])

    cols_to_drop = [
        c for c in final.columns
        if c.endswith("_df") or c.endswith("_agg")
    ]
    final.drop(columns=cols_to_drop, inplace=True)

    final["Продажа"] = final["Продажа"].fillna(0)
    final["Ремонт"]  = final["Ремонт"].fillna(0)

    flow_order = [
        "Приход",
        "Продажа",
        "Ремонт",
        "Ремонт не подъемники",
        "Ремонт всего",
        "Конечный остаток",
    ]
    meta_order = [c for c in final.columns if c not in flow_order]
    final = final[meta_order + [c for c in flow_order if c in final.columns]]

    final = normalize_analog_lists(
        final, col_group="Номер группы", col_analogs="Список аналогов"
    )

    name_map = (
        final.groupby("Номер группы")["Номенклатура"]
        .apply(lambda s: min(s.dropna().astype(str), key=len, default=None))
    )
    final["Номенклатура"] = final["Номер группы"].map(name_map)

    article_map = (
    final.groupby("Номер группы")["Артикул"]
    .apply(lambda s: max(s.dropna().astype(str), key=len, default=None))
    )
    final["Артикул"] = final["Номер группы"].map(article_map)

    df = fill_flow_columns(final, ["Продажа", "Ремонт", "Ремонт не подъемники", "Ремонт всего"])

    return df