import json
import os

import numpy as np
import pandas as pd


_CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",      
    "configs"
)


def load_config(file_name: str) -> dict:
    """Загружает JSON из папки configs. Возвращает {} при отсутствии файла."""
    config_path = os.path.join(_CONFIG_DIR, file_name)
    if os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


_matches_cache: dict | None = None


def get_matches() -> dict:
    """
    Ленивая загрузка configs/matches.json.

    Raises:
        FileNotFoundError: если файл не найден или пуст.
    """
    global _matches_cache
    if _matches_cache is None:
        _matches_cache = load_config("matches.json")
        if not _matches_cache:
            raise FileNotFoundError(
                f"configs/matches.json не найден или пуст. "
                f"Ожидаемый путь: "
                f"{os.path.abspath(os.path.join(_CONFIG_DIR, 'matches.json'))}"
            )
    return _matches_cache


def load_dataset(file_excel: str, skiprows: int = 8) -> pd.DataFrame:
    """
    Загружает DataFrame из Excel-файла.

    Raises:
        FileNotFoundError: если файл не найден.
    """
    try:
        data = pd.read_excel(file_excel, skiprows=range(skiprows), dtype=str)
        print(f"Файл успешно загружен из Excel: {file_excel}")
        return data
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Файл не найден: {file_excel!r}. Проверьте путь."
        )


def preprocess_repair_parts(file_path: str, skiprows: int = 8) -> pd.DataFrame:
    """
    Загружает и предобрабатывает отчёт «Запчасти списанные в ремонт».

    Что делает:
      - Загружает Excel, убирает строки без артикула/ориг.номера.
      - Фильтрует только записи по подъёмникам.
      - Добавляет колонки: Тип подъемника, Тип двигателя, Год, Месяц, Квартал.
      - Парсит числовые поля (год выпуска, наработка, количество).

    Returns:
        Предобработанный DataFrame.
    """
    from preprocessing.normalization import extract_model_case_insensitive

    data = load_dataset(file_path, skiprows)
    df = data.copy()
    df.columns = df.columns.str.strip()

    # Оставляем только строки с хотя бы одним идентификатором
    df = df[
        ~(df["Номенклатура.Оригинальный номер"].isna()
          & df["Номенклатура.Артикул"].isna())
    ]
    # Только подъёмники
    df = df[df["Машина"].str.contains(
        r"подъ[её]мник", case=False, na=False, regex=True
    )]

    df["Номенклатура"] = df["Номенклатура"].str.strip()

    # ── Тип подъемника ────────────────────────────────────────────────────
    import numpy as np
    type_conditions = [
        df["Машина"].str.contains("ножничный",       case=False, na=False),
        df["Машина"].str.contains("коленчатый",      case=False, na=False),
        df["Машина"].str.contains("телескопический", case=False, na=False),
        df["Машина"].str.contains("мачтовый",        case=False, na=False),
    ]
    lift_type_labels = ["ножничный", "коленчатый", "телескопический", "мачтовый"]
    df["Тип подъемника"] = np.select(
        type_conditions, lift_type_labels, default="другое"
    )

    # ── Тип двигателя ─────────────────────────────────────────────────────
    engine_conditions = [
        df["Машина"].str.contains("электрический", case=False, na=False),
        df["Машина"].str.contains("дизельный",     case=False, na=False),
    ]
    engine_labels = ["Электрический", "Дизельный"]
    df["Тип двигателя"] = np.select(
        engine_conditions, engine_labels, default="Дизельный"
    )

    df = df.drop(columns=[c for c in df.columns if "Unnamed" in c])

    # ── Временные поля ────────────────────────────────────────────────────
    df["Дата"]    = pd.to_datetime(df["Дата"], format="%d.%m.%Y %H:%M:%S")
    df["Год"]     = df["Дата"].dt.year.astype("Int64")
    df["Квартал"] = df["Дата"].dt.quarter.astype("Int64")
    df["Месяц"]   = df["Дата"].dt.month.astype("Int64")

    # ── Числовые поля ─────────────────────────────────────────────────────
    df["Лот.CRM год выпуска"] = (
        df["Лот.CRM год выпуска"]
        .apply(lambda x: int(str(x).replace(",", "")) if isinstance(x, str) else x)
        .astype(float)
        .astype("Int64")
    )
    df["Средняя наработка (лет)"] = df["Год"] - df["Лот.CRM год выпуска"]
    df["Лот.CRM наработка"] = (
        df["Лот.CRM наработка"]
        .apply(lambda x: str(x).replace(",", "") if isinstance(x, str) else x)
        .astype(float)
        .astype("Int64")
    )
    df["Количество"] = df["Количество"].astype(float).astype(int)

    df["Номера машин"] = df.apply(
        lambda row: extract_model_case_insensitive(
            row["Машина"], row["Машина.Бренд"]
        ),
        axis=1,
    )

    return df


def preprocess_stock_report(file_path: str, skiprows: int = 8) -> pd.DataFrame:
    """
    Загружает и предобрабатывает отчёт «Остатки и обороты».

    Что делает:
      - Разбирает структуру Excel с вложенными заголовками периодов.
      - Добавляет колонки Год, Месяц, Квартал через ffill по заголовкам.
      - Конвертирует числовые колонки (Расход, Приход, Конечный остаток).

    Returns:
        Предобработанный DataFrame.
    """
    data = load_dataset(file_path, skiprows)
    df = data.copy()
    df.columns = df.columns.str.strip()

    df["Склады_продажа"] = df["Склад"]
    df["Номенклатура"]   = df["Номенклатура"].str.strip()

    # ── Парсинг периодов из заголовочных строк ────────────────────────────
    period_mask = df["Номенклатура"].str.contains(
        r"\d{4} г\.|\d квартал \d{4} г\."
        r"|Январь|Февраль|Март|Апрель|Май|Июнь"
        r"|Июль|Август|Сентябрь|Октябрь|Ноябрь|Декабрь",
        na=False,
    )

    df["year"]    = df["Номенклатура"].str.extract(r"\b(20[2-9]\d)\b")
    df["quarter"] = df["Номенклатура"].str.extract(r"(\d квартал)")
    df["month"]   = df["Номенклатура"].str.extract(
        r"(Январь|Февраль|Март|Апрель|Май|Июнь"
        r"|Июль|Август|Сентябрь|Октябрь|Ноябрь|Декабрь)"
    )
    df[["year", "quarter", "month"]] = df[["year", "quarter", "month"]].ffill()
    df = df[~period_mask]

    df["Квартал"] = df["quarter"].str.extract(r"(\d)").astype(int)

    month_map = {
        "Январь": 1,  "Февраль": 2,  "Март": 3,    "Апрель": 4,
        "Май":    5,  "Июнь":    6,  "Июль": 7,    "Август": 8,
        "Сентябрь": 9, "Октябрь": 10, "Ноябрь": 11, "Декабрь": 12,
    }
    df["Месяц"] = df["month"].map(month_map).astype("Int64")
    df["Год"]   = df["year"].astype("Int64")

    df.drop(
        columns=["Unnamed: 0", "month", "quarter", "year", "Склад"],
        inplace=True,
        errors="ignore",
    )
    df = df.loc[df["Артикул"].notna() | df["Оригинальный номер"].notna()]

    for col in ["Расход", "Приход", "Конечный остаток"]:
        df[col] = df[col].str.replace(",", "", regex=False).astype(float)

    return df
