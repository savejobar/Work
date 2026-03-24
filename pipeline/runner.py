import ast
from collections import defaultdict

import pandas as pd

from features.aggregation import (
    aggregate_repair_groups,
    aggregate_stock_groups,
    calculate_external_sales,
    fill_missing_months,
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
) -> tuple[dict[str, int], dict[str, tuple]]:
    """
    Строит два словаря для быстрого поиска группы и списка аналогов
    по нормализованному артикулу.
    """
    article_to_group = {}
    article_to_analogs = {}

    for _, row in df1.drop_duplicates("Номенклатура.Артикул").iterrows():
        group = row["Номер группы"]
        raw = row["all_analogs"]
        try:
            analogs = raw if isinstance(raw, tuple) else ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            analogs = ()

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
    article_to_analogs: dict[str, tuple],
) -> tuple[int | None, tuple | None]:
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
) -> tuple | None:
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
    
    mask_df1 = (
    df1["Номенклатура.Артикул"].isna()
    & df1["Номенклатура.Оригинальный номер"].notna()
    )
    df1.loc[mask_df1, "Номенклатура.Артикул"] = df1.loc[
        mask_df1, "Номенклатура.Оригинальный номер"
    ]

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
    df1["Номер группы"] = df1["all_analogs"].apply(group_mapping.get)
    df_quarter = aggregate_repair_groups(df1)

    df2 = preprocess_stock_report(stock_path)
    df2 = normalize_nomenclatures_stock_report(df2)

    df2["Артикул"] = df2["Артикул"].apply(normalize)
    df2["Оригинальный номер"] = df2["Оригинальный номер"].apply(normalize)
    df2 = fill_missing_article(df2, "Артикул", "Оригинальный номер")

    mask_df2 = df2["Артикул"].isna() & df2["Оригинальный номер"].notna()
    df2.loc[mask_df2, "Артикул"] = df2.loc[mask_df2, "Оригинальный номер"]


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
    
    df2["Номер группы"] = groups
    df2["Список аналогов"] = analogs_list

    df2["Список аналогов"] = df2.apply(
        lambda r: _enrich_analogs(r, article_to_group), axis=1
    )
    df2 = normalize_analog_lists(df2)

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

    art_by_group = (
        df2[df2["Номер группы"].notna()]
        .assign(Артикул_norm=lambda d: d["Артикул"].apply(normalize))
        .dropna(subset=["Артикул_norm"])
        .groupby(df2["Номер группы"].dropna().astype(str))["Артикул_norm"]
        .apply(set)
        .to_dict()
    )

    df1_group_idx: defaultdict[str, list] = defaultdict(list)
    for i, g in df1["Номер группы"].astype(str).items():
        df1_group_idx[g].append(i)

    df2_group_idx: defaultdict[str, list] = defaultdict(list)
    for i, g in df2["Номер группы"].dropna().astype(str).items():
        df2_group_idx[g].append(i)

    for grp, arts in art_by_group.items():
        idxs_df1 = df1_group_idx.get(grp, [])
        if not idxs_df1:
            continue
        existing = df1.at[idxs_df1[0], "all_analogs"]
        if not isinstance(existing, tuple):
            continue
        new_arts = arts - set(existing)
        if not new_arts:
            continue
        new_analogs = tuple(sorted(existing + tuple(new_arts)))
        for i in idxs_df1:
            df1.at[i, "all_analogs"] = new_analogs
        for i in df2_group_idx.get(grp, []):
            df2.at[i, "Список аналогов"] = new_analogs

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

    agg = aggregate_stock_groups(df2)
    agg = calculate_external_sales(agg, df_quarter)

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

    meta_order = [
        c for c in final.columns
        if c not in ["Приход", "Продажа", "Ремонт", "Конечный остаток"]
    ]
    final = final[meta_order + ["Приход", "Продажа", "Ремонт", "Конечный остаток"]]

    final = normalize_analog_lists(
        final, col_group="Номер группы", col_analogs="Список аналогов"
    )

    name_map = (
        final.groupby("Номер группы")["Номенклатура"]
        .apply(lambda s: min(s.dropna().astype(str), key=len, default=None))
    )
    final["Номенклатура"] = final["Номер группы"].map(name_map)

    df = fill_flow_columns(df, ["Продажа", "Ремонт"])

    return df
