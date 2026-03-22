import numpy as np
import pandas as pd

from readers.loaders import load_config


def apply_corrections(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """
    Применяет ручные правки к DataFrame из configs/corrections_{source}.json.

    Поддерживаемые разделы JSON:
        article_remap    – переименование значений колонки «Артикул».
        by_article       – правки по значению «Артикул».
        by_nomenclature  – правки по значению «Номенклатура».
                           Ключ «Номенклатура» внутри объекта переименовывает
                           саму строку и применяется последним.

    Значение null в JSON → NaN в DataFrame.
    """
    corrections = load_config(f"corrections_{source}.json")
    if not corrections:
        return df.copy()

    df = df.copy()

    nom_col = "Номенклатура"
    art_col = "Артикул" if source == "stock" else "Номенклатура.Артикул"

    # Переименование значений артикула
    article_remap: dict[str, str] = corrections.get("article_remap", {})
    if article_remap and art_col in df.columns:
        df[art_col] = df[art_col].replace(article_remap)

    # Правки по значению Артикула
    for art_val, overrides in corrections.get("by_article", {}).items():
        if art_col not in df.columns:
            break
        mask = df[art_col] == art_val
        if not mask.any():
            continue
        for col, value in overrides.items():
            df.loc[mask, col] = value if value is not None else np.nan

    # Правки по значению Номенклатуры
    for nom_val, overrides in corrections.get("by_nomenclature", {}).items():
        mask = df[nom_col] == nom_val
        if not mask.any():
            continue
        # Сначала все поля, кроме переименования номенклатуры
        for col, value in overrides.items():
            if col == nom_col:
                continue
            df.loc[mask, col] = value if value is not None else np.nan
        # Переименование самой номенклатуры — в последнюю очередь,
        # чтобы маска не потеряла строки до применения остальных правок
        if nom_col in overrides and overrides[nom_col] is not None:
            df.loc[mask, nom_col] = overrides[nom_col]

    return df
