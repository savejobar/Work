import pandas as pd

from readers.loaders import get_matches
from preprocessing.corrections import apply_corrections
from preprocessing.normalization import extract_articles


def _split_complects(
        df: pd.DataFrame,
        art_col: str,
        orig_col: str,
        ext_col: str | None = None,
) -> pd.DataFrame:
    """
    Находит строки-комплекты, разбивает их на отдельные артикулы
    и возвращает объединённый DataFrame.

    Комплект — строка номенклатуры содержащая '+' или совпадающая
    со специальными масками (ST40111/40110, сдвоенные фильтры и т.д.).
    Каждый комплект взрывается ровно в 2 строки через explode —
    если это не так, выбрасывается ValueError.
    """
    matches = get_matches()
    mask_plus = (
    df["Номенклатура"].str.contains(r"\+", na=False)
    & ~df["Номенклатура"].str.contains(
        r"^(?:Колесо|РВД|Обогреватель|Контроллер|Deutz|Набор зарядное|Набор уплотнений|Амортизатор|Клемма|Ryobi|Кнопка|Элемент"
        r"|Выключатель|Батарея|Рукав|Зажим|Шина|Болт|Кожух|Патрубок|Гидроруль|Крепеж|Тяга|Палец|Кольцо|Фонарь|Подшипник|SATA|Разъ[её]м|Индуктивный|Подстаканник|Секция|Устройство|Предохранитель|Кольца|Вкладыши|Обод|Уплотнение|Ролик|ПВД|Провод|Жгут|Карта|Бесконтактный"
        r"|Распределитель|Насос|Комплект|Кабель|Гидроцилиндр|Датчик|Коллектор|Микроконтроллер|\*\*\*Шина|Розетка|Лента|Колодка|Адаптер|Накопитель|Коронка|Переходник|Прокладка|Реле|БРС|Ароматизатор|Ремкомплект|Шпилька|Диск|Манжета|Аккумулятор|Поршень|Крепление"
        r"|Фильтр топливный PERKINS"
        r"|Фильтр воздушный \(внешний\+внутренний\) A5541S"
        r"|Фильтр воздушный к-кт \(внутр\.\+внешн\.\) 1351230502"
        r"|Фильтр воздушный \(к-т внешний\+внутренний\) A5613S"
        r")",
        na=False,
    )
)
    mask_st_units = df["Номенклатура"].str.contains(r"Фильтр воздушный ST40111/40110", case=False, na=False)
    mask_brackets = df["Номенклатура"].str.contains(r"(95*165*340/70*90*335,", case=False, na=False, regex=False)
    mask_hid = df["Номенклатура"].str.contains(r"Фильтр воздушный ST40111ST40110", case=False, na=False)
    mask_filter = (
    df["Номенклатура"].str.contains(r"Фильтр воздушный 6666375/6666376", case=False, na=False)
    | df["Номенклатура"].str.contains(r"Фильтр воздушный комплект DIFA4337DIFA433701", case=False, na=False)
    ) 

    complects = df[mask_plus | mask_st_units | mask_brackets | mask_hid | mask_filter].copy()
    df = df[~df.index.isin(complects.index)]

    complects[art_col] = complects["Номенклатура"].apply(
        lambda x: extract_articles(x, matches.keys())
    )
    df_exploded = complects.explode(art_col).reset_index(drop=True)
    df_exploded[orig_col] = df_exploded[art_col].map(
        lambda x: matches[x].split()[0] if x in matches else None
    )
    if ext_col:
        df_exploded[ext_col] = df_exploded[art_col].map(matches)

    expected = complects.shape[0] * 2
    actual = df_exploded.shape[0]
    
    if actual != expected:
        problem_info = "\n".join(complects["Номенклатура"].unique().tolist())
        raise ValueError(
            f"Ожидалось {expected} строк после explode комплектов, получено {actual}.\n"
            f"Проблемные номенклатуры:\n{problem_info}"
        )


    return pd.concat([df, df_exploded], ignore_index=True)


def normalize_nomenclatures_repair_parts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Нормализует номенклатуру в DataFrame ремонтов.
    """
    df = apply_corrections(df, source="repair")

    cols = [
        "Номенклатура.Артикул",
        "Номенклатура.Оригинальный номер",
        "Номенклатура.Оригинальный номер расширенный",
    ]
    df[cols] = (
        df[cols]
        .replace(r"DIFA\s*", "", regex=True)
        .replace(r"\s+", " ", regex=True)
        .apply(lambda x: x.str.strip())
    )

    df = _split_complects(
        df,
        art_col="Номенклатура.Артикул",
        orig_col="Номенклатура.Оригинальный номер",
        ext_col="Номенклатура.Оригинальный номер расширенный",
    )

    return df


def normalize_nomenclatures_stock_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Нормализует номенклатуру в DataFrame склада.
    """
    df = apply_corrections(df, source="stock")

    cols = ["Артикул", "Оригинальный номер"]
    df[cols] = (
        df[cols]
        .replace(r"DIFA\s*", "", regex=True)
        .replace(r"\s+", " ", regex=True)
        .apply(lambda x: x.str.strip())
    )

    df = _split_complects(
        df,
        art_col="Артикул",
        orig_col="Оригинальный номер",
    )

    return df