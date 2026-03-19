import pandas as pd

from readers.loaders import get_matches
from preprocessing.corrections import apply_corrections
from preprocessing.normalization import extract_articles


# ── Внутренние функции разбивки комплектов ────────────────────────────────────

def _split_complects_repair(df: pd.DataFrame) -> pd.DataFrame:
    """
    Находит строки-комплекты в df1 (ремонт), взрывает их в отдельные артикулы
    и возвращает объединённый DataFrame.

    Комплект — строка, где в Номенклатуре перечислено несколько артикулов
    (через «+», специальные паттерны и т.д.).

    Raises:
        ValueError: если после explode количество строк не соответствует ожидаемому.
    """
    matches = get_matches()

    mask_plus = (
        df["Номенклатура"].str.contains(r"\+", na=False)
        & ~df["Номенклатура"].str.contains(
            r"^(?:Колесо|РВД|[Оо]богреватель|Контроллер|Deutz"
            r"|Выключатель|Батарея|Рукав"
            r"|Фильтр воздушный к-кт \(внутр\.\+внешн\.\) 1351230502)",
            na=False,
        )
    )
    mask_st_units = df["Номенклатура"].str.contains(
        r"Фильтр воздушный ST40111/40110", case=False, na=False
    )
    mask_brackets = df["Номенклатура"].str.contains(
        r"(95*165*340/70*90*335,", case=False, na=False, regex=False
    )
    mask_hid = df["Номенклатура"].str.contains(
        r"Фильтр воздушный ST40111ST40110", case=False, na=False
    )

    complects = df[mask_plus | mask_st_units | mask_brackets | mask_hid].copy()
    df        = df[~df.index.isin(complects.index)]

    complects["Номенклатура.Артикул"] = complects["Номенклатура"].apply(
        lambda x: extract_articles(x, matches.keys())
    )
    complects["Номенклатура.Артикул"] = complects["Номенклатура"].apply(
        lambda x: extract_articles(x, matches.keys())
    )

    # ── ОТЛАДКА (удалить после исправления) ──────────────────────────────
    import sys
    counts = complects["Номенклатура.Артикул"].apply(len)
    print(f"Всего комплектов: {len(complects)}", file=sys.stderr)
    print(f"Дают 1 артикул: {(counts==1).sum()}", file=sys.stderr)
    print(f"Дают 2 артикула: {(counts==2).sum()}", file=sys.stderr)
    for _, row in complects[counts != 2].head(5).iterrows():
        print(f"ПРОБЛЕМА: '{row['Номенклатура']}' → {row['Номенклатура.Артикул']}", file=sys.stderr)
    print(f"matches keys: {list(matches.keys())}", file=sys.stderr)
    # ─────────────────────────────────────────────────────────────────────


    df_exploded = complects.explode("Номенклатура.Артикул").reset_index(drop=True)
    df_exploded["Номенклатура.Оригинальный номер"] = df_exploded[
        "Номенклатура.Артикул"
    ].map(lambda x: matches[x].split()[0] if x in matches else None)
    df_exploded["Номенклатура.Оригинальный номер расширенный"] = df_exploded[
        "Номенклатура.Артикул"
    ].map(matches)

    expected = complects.shape[0] * 2
    actual   = df_exploded.shape[0]
    if actual != expected:
        raise ValueError(
            f"Ожидалось {expected} строк после explode комплектов (ремонт), "
            f"получено {actual}. "
            "Проверьте configs/matches.json и маски определения комплектов."
        )

    return pd.concat([df, df_exploded], ignore_index=True)


def _split_complects_stock(df: pd.DataFrame) -> pd.DataFrame:
    """
    Находит строки-комплекты в df2 (склад), взрывает их в отдельные артикулы
    и возвращает объединённый DataFrame.

    Raises:
        ValueError: если после explode количество строк не соответствует ожидаемому.
    """
    matches = get_matches()

    mask_plus = (
        df["Номенклатура"].str.contains(r"\+", na=False)
        & ~df["Номенклатура"].str.contains(
            r"^(Колесо|РВД|[Оо]богреватель|Распределитель|Насос|Комплект"
            r"|Кабель|Гидроцилиндр|Датчик|Коллектор"
            r"|Фильтр топливный PERKINS"
            r"|Фильтр воздушный \(внешний\+внутренний\) A5541S)",
            na=False,
        )
    )
    mask_st_units = df["Номенклатура"].str.contains(
        r"Фильтр воздушный ST40111/40110", case=False, na=False
    )
    mask_brackets = df["Номенклатура"].str.contains(
        r"(95*165*340/70*90*335,", case=False, na=False, regex=False
    )
    mask_hid = df["Номенклатура"].str.contains(
        r"Фильтр воздушный ST40111ST40110", case=False, na=False
    )
    mask_filter = df["Номенклатура"].str.contains(
        r"Фильтр воздушный 6666375/6666376", case=False, na=False
    )

    complects = df[
        mask_plus | mask_st_units | mask_brackets | mask_hid | mask_filter
    ].copy()
    df = df[~df.index.isin(complects.index)]

    complects["Артикул"] = complects["Номенклатура"].apply(
        lambda x: extract_articles(x, matches.keys())
    )
    df_exploded = complects.explode("Артикул").reset_index(drop=True)
    df_exploded["Оригинальный номер"] = df_exploded["Артикул"].map(
        lambda x: matches[x].split()[0] if x in matches else None
    )

    expected = complects.shape[0] * 2
    actual   = df_exploded.shape[0]
    if actual != expected:
        raise ValueError(
            f"Ожидалось {expected} строк после explode комплектов (склад), "
            f"получено {actual}. "
            "Проверьте configs/matches.json и маски определения комплектов."
        )

    return pd.concat([df, df_exploded], ignore_index=True)


# ── Публичные функции ─────────────────────────────────────────────────────────

def normalize_nomenclatures_repair_parts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Очищает и нормализует справочник номенклатуры в DataFrame ремонтов.

    Порядок операций:
      1. Применяет правки из configs/corrections_repair.json.
      2. Убирает префикс «DIFA» и лишние пробелы.
      3. Разбивает строки-комплекты на отдельные артикулы.

    Returns:
        Очищенный DataFrame.
    """
    df = apply_corrections(df, source="repair")

    cols = [
        "Номенклатура.Артикул",
        "Номенклатура.Оригинальный номер",
        "Номенклатура.Оригинальный номер расширенный",
    ]
    df[cols] = df[cols].replace(r"DIFA\s*", "", regex=True)
    df[cols] = (
        df[cols]
        .replace(r"\s+", " ", regex=True)
        .apply(lambda x: x.str.strip())
    )

    df = _split_complects_repair(df)
    return df


def normalize_nomenclatures_stock_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Очищает и нормализует справочник номенклатуры в DataFrame склада.

    Порядок операций:
      1. Применяет правки из configs/corrections_stock.json.
      2. Убирает префикс «DIFA» и лишние пробелы.
      3. Разбивает строки-комплекты на отдельные артикулы.

    Важно: corrections применяются ДО вычисления масок комплектов —
    переименованные строки корректно попадают в нужные маски.

    Returns:
        Очищенный DataFrame.
    """
    df = apply_corrections(df, source="stock")

    cols = ["Артикул", "Оригинальный номер"]
    df[cols] = df[cols].replace(r"DIFA\s*", "", regex=True)
    df[cols] = (
        df[cols]
        .replace(r"\s+", " ", regex=True)
        .apply(lambda x: x.str.strip())
    )

    df = _split_complects_stock(df)
    return df
