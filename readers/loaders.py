import json
import os

import pandas as pd
from typing import Any


_CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "configs"
)

REPAIR_COLS = [
    "Дата", 
    "Год", 
    "Месяц", 
    "Номенклатура", 
    "Номенклатура.Артикул", 
    "Номенклатура.Оригинальный номер", 
    "Номенклатура.Оригинальный номер расширенный",
    "Машина",
    "Количество"
    ]

STOCK_COLS  = [
    "Год",
    "Месяц",
    "Номенклатура", 
    "Артикул", 
    "Оригинальный номер",
    "Приход",
    "Расход",
    "Конечный остаток"
    ]

DANGEROUS_EXCEL_PREFIXES: tuple[str, ...] = ("=", "+", "-", "@")

MAX_UPLOAD_BYTES: int = 50 * 1024 * 1024 


def _uploaded_size(uploaded_file: Any) -> int:
    """
    Возвращает размер загруженного файла в байтах.
    """
    size = getattr(uploaded_file, "size", None)
    if size is not None:
        return int(size)
    return len(uploaded_file.getbuffer())


def validate_upload_size(uploaded_file: Any, label: str) -> None:
    """
    Проверяет, что размер загруженного файла не превышает допустимый лимит.
    """
    size = _uploaded_size(uploaded_file)
    if size > MAX_UPLOAD_BYTES:
        raise ValueError(
            f"{label}: файл слишком большой ({size / 1024 / 1024:.1f} MB). "
            f"Лимит: {MAX_UPLOAD_BYTES / 1024 / 1024:.0f} MB."
        )
    

def load_config(file_name: str) -> dict:
    """
    Загружает JSON из папки configs. Возвращает {} при отсутствии файла.
    """
    config_path = os.path.join(_CONFIG_DIR, file_name)
    if os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


_matches_cache: dict[str, str] | None = None


def sanitize_excel_value(value: Any) -> Any:
    """
    Экранирует строковые значения, которые Excel может интерпретировать как формулу.
    """
    if not isinstance(value, str):
        return value

    if value.lstrip().startswith(DANGEROUS_EXCEL_PREFIXES):
        return f"'{value}"

    return value


def sanitize_excel_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Возвращает копию DataFrame, безопасную для экспорта в Excel..
    """
    safe_df: pd.DataFrame = df.copy()

    for col in safe_df.columns:
        if pd.api.types.is_object_dtype(safe_df[col]) or pd.api.types.is_string_dtype(safe_df[col]):
            safe_df[col] = safe_df[col].map(sanitize_excel_value)

    return safe_df


def get_matches() -> dict:
    """
    Ленивая загрузка configs/matches.json.
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


def load_dataset(
    file_excel: str,
    target_col: str = "Номенклатура",
    max_skip: int = 20,
) -> pd.DataFrame:
    """
    Загружает DataFrame из Excel-файла.
    Поддерживает одно- и двухстрочные заголовки.
    """
    preview = pd.read_excel(
        file_excel, header=None, nrows=max_skip, dtype=str, engine="calamine"
    )

    for n, row in preview.iterrows():
        if target_col in row.values:
            if n > 0:
                prev_row = preview.iloc[n - 1]
                if any(v in str(prev_row.values) for v in ["Начальный остаток", "Приход", "Расход", "Конечный остаток"]):
                    df = pd.read_excel(
                        file_excel,
                        skiprows=range(n - 1),
                        header=[0, 1],
                        dtype=str,
                        engine="calamine",
                    )
                    df.columns = [
                        " ".join(
                            str(c) for c in col
                            if "Unnamed" not in str(c) and "nan" not in str(c)
                        ).strip()
                        for col in df.columns
                    ]
                    return df

            return pd.read_excel(
                file_excel, skiprows=range(n), dtype=str, engine="calamine"
            )

    raise ValueError(f"Колонка '{target_col}' не найдена в первых {max_skip} строках")


def preprocess_repair_parts(file_path: str) -> pd.DataFrame:
    """
    Загружает и предобрабатывает отчёт «Запчасти списанные в ремонт».
    """
    data = load_dataset(file_path)
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

    if df.empty:
        raise ValueError(
            "В отчете 'Запчасти списанные в ремонт' не найдено строк по подъёмной технике."
        )

    df["Номенклатура"] = df["Номенклатура"].str.strip()

    # Временные поля
    df["Дата"] = pd.to_datetime(df["Дата"], format="%d.%m.%Y %H:%M:%S")
    df["Год"] = df["Дата"].dt.year.astype("Int64")
    df["Месяц"] = df["Дата"].dt.month.astype("Int64")

    df["Количество"] = df["Количество"].astype(float).astype(int)

    return df[REPAIR_COLS]


def preprocess_stock_report(file_path: str) -> pd.DataFrame:
    """
    Загружает и предобрабатывает отчёт «Остатки и обороты».
    """
    data = load_dataset(file_path)
    df = data.copy()
    df.columns = df.columns.str.strip()
    
    df.columns = [
    col.replace("По месяцам ", "").strip() if "Номенклатура" in col else col
    for col in df.columns
    ]
    
    df["Номенклатура"] = df["Номенклатура"].str.strip()

    # Парсинг периодов из заголовочных строк
    period_mask = df["Номенклатура"].str.contains(
        r"\d{4} г\.|\d квартал \d{4} г\."
        r"|Январь|Февраль|Март|Апрель|Май|Июнь"
        r"|Июль|Август|Сентябрь|Октябрь|Ноябрь|Декабрь",
        na=False,
    )

    period_labels = df["Номенклатура"].where(period_mask)
    df["year"] = period_labels.str.extract(r"\b(20[2-9]\d)\b")
    df["month"] = period_labels.str.extract(
        r"(Январь|Февраль|Март|Апрель|Май|Июнь"
        r"|Июль|Август|Сентябрь|Октябрь|Ноябрь|Декабрь)"
    )
    df[["year", "month"]] = df[["year", "month"]].ffill()
    df = df[~period_mask]

    month_map = {
        "Январь": 1, "Февраль": 2, "Март": 3, "Апрель": 4,
        "Май": 5, "Июнь": 6, "Июль": 7, "Август": 8,
        "Сентябрь": 9, "Октябрь": 10, "Ноябрь": 11, "Декабрь": 12,
    }
    df["Месяц"] = df["month"].map(month_map).astype("Int64")
    df["Год"] = df["year"].astype("Int64")

    df = df.loc[df["Артикул"].notna() | df["Оригинальный номер"].notna()]

    for col in ["Расход", "Приход", "Конечный остаток", "Начальный остаток"]:
        df[col] = df[col].str.replace(",", "", regex=False).astype(float)
        df.loc[df[col] < 0, col] = 0
        
        if  col != 'Конечный остаток':
            df[col] = df[col].fillna(0)
    
    mask = df["Конечный остаток"].isna()
    df.loc[mask, "Конечный остаток"] = (
        df.loc[mask, "Начальный остаток"]
        + df.loc[mask, "Приход"]
        - df.loc[mask, "Расход"]
    )

    return df[STOCK_COLS]
