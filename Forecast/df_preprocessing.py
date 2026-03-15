import re
import os

from typing import Optional, List, Iterable, Any, Dict, Set, Tuple
import numpy as np
import pandas as pd


def load_repair_parts(file_csv: str, file_excel: str, skiprows: int = 9) -> pd.DataFrame | None:

    try:
        data = pd.read_csv(file_csv, skiprows=range(skiprows), dtype=str)
        print(f"Файл успешно загружен из CSV: {file_csv}")
        return data
    except FileNotFoundError:
        try:
            data = pd.read_excel(file_excel, skiprows=range(skiprows), dtype=str)
            print(f"Файл успешно загружен из Excel: {file_excel}")
            return data
        except FileNotFoundError:
            print("Файлы не найдены. Проверьте путь к CSV и Excel.")
            return None
    except Exception as e:
        print(f"Ошибка при чтении CSV: {e}")
        return None
    

def extract_model_case_insensitive(machine: Optional[str], brand: Optional[str]) -> Optional[str]:

    if pd.isna(machine) or pd.isna(brand):
        return None

    machine_lower = machine.lower()
    brand_lower = brand.lower()
    idx = machine_lower.find(brand_lower)
    if idx == -1:
        return None

    return machine[idx + len(brand):].strip()


def extract_articles(text: str, match_keys: Iterable[str]) -> List[str]:

    # Преобразование паттерна LettersDigits/Digits в отдельные артикула
    text = re.sub(r'([A-Za-z]+)(\d+)/(\d+)', r'\1\2 \1\3', text)

    # Специальные кейсы
    if ('kw1634' in text.lower()) and 'сдвоенный' in text.lower():
        return ['kw1634in', 'kw1634']
    if 'ST40722' in text:
        return ['ST40722(1)', 'ST40722(2)']

    found: List[str] = []
    for key in match_keys:
        pattern = re.escape(key)
        if re.search(pattern, text):
            found.append(key)

    return found


def add_original_to_extended(row: pd.Series) -> str | None:

    main_art = str(row["Номенклатура.Артикул"]).strip()
    extended = row["Номенклатура.Оригинальный номер расширенный"]
    original = row["Номенклатура.Оригинальный номер"]

    extended_parts = []
    if pd.notna(extended) and str(extended).strip() != "":
        extended_parts = re.split(r'\s+', str(extended))
        extended_parts = [part.strip() for part in extended_parts if part.strip()]

    original_parts = []
    if pd.notna(original) and str(original).strip() != "":
        original_parts = [str(original).strip()]

    extended_upper = [p.upper() for p in extended_parts]

    for part in original_parts:
        if part.upper() not in extended_upper and part.upper() != main_art.upper():
            extended_parts.append(part)

    extended_parts = sorted(list(dict.fromkeys(extended_parts)))

    return " ".join(extended_parts) if extended_parts else np.nan


def find_all_analogs(start: Any, graph: Dict[Any, Set[Any]]) -> Tuple[Any, ...]:

    visited = set()
    stack = [start]

    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            stack.extend(graph[node] - visited)

    return tuple(sorted(visited))


def normalize(val):

    if pd.isna(val) or str(val).strip() in ("", "None", "nan"):
        return None
    return str(val).strip().upper()


def join_ordered(series):

    vals = series.dropna().astype(str).str.strip()
    vals = [v for v in vals if v]
    seen = set()
    result = []
    for v in vals:
        if v not in seen:
            seen.add(v)
            result.append(v)
    return "; ".join(result) if result else None


def shortest_value(series):

    vals = series.dropna().astype(str).str.strip()
    if len(vals) == 0:
        return None
    return min(vals, key=len)


def count_unique(series):

    vals = series.dropna().astype(str).str.strip()
    vals = [v for v in vals if v]
    return len(set(vals))


def merge_parts_df1(df):

    df = df.copy()
    df['Количество'] = df['Количество'].astype(float).astype(int)
    df['Лот.CRM наработка'] = df['Лот.CRM наработка'].apply(lambda x: str(x).replace(',',''))
    df['Лот.CRM наработка'] = df['Лот.CRM наработка'].astype(float).astype('Int64')
    df['Лот.CRM год выпуска'] = df['Лот.CRM год выпуска'].astype(float).astype('Int64')

    meta = (
        df.groupby("Номер группы", as_index=False)
        .agg({
            "Номенклатура": shortest_value,
            "Машина.Бренд": join_ordered,
            "Машина.Серия техники": join_ordered,
            "Тип подъемника": join_ordered,
            "Тип двигателя": join_ordered,
            'Номенклатура.Тип техники': join_ordered,
            'Документ.Склад': join_ordered,
            'Номера машин': join_ordered,
            "all_analogs": "first"
        })
        .rename(columns={
        'Машина.Бренд': 'Бренды',
        'Машина.Серия техники': 'Серии техники',
        'Тип подъемника': 'Типы подъемников',
        'Тип двигателя': 'Типы двигателя',
        'Документ.Склад': 'Склады',
        'all_analogs': 'Список аналогов'
    })
    )

    qty = (
        df.groupby(
            ["Год", "Месяц", "Номер группы"],
            as_index=False
        )
        .agg({
            "Количество": "sum",
            'Лот.CRM год выпуска': 'mean',
            'Средняя наработка (лет)': 'mean',
            'Серийный номер': count_unique,
            'Лот.CRM наработка': 'mean'
        })
        .rename(columns={
        'Серийный номер':  "Количество уникальных номеров машин, где запчасть применялась",
        "Количество": 'Ремонт',
        'Лот.CRM год выпуска' : 'Средний год выпуска',
        'Лот.CRM наработка': 'Средняя наработка техники, где запчасть применялась (ч)'
        })
    )

    qty['Средний год выпуска'] = qty['Средний год выпуска'].round(1)
    qty['Средняя наработка (лет)'] = qty['Средняя наработка (лет)'].round(1)
    qty['Средняя наработка техники, где запчасть применялась (ч)'] = qty['Средняя наработка техники, где запчасть применялась (ч)'].round(1) 
    result = qty.merge(meta, on="Номер группы", how="left")

    return result


def merge_parts_df2(df):
    df = df.copy()

    meta = (
        df.groupby("Номер группы", as_index=False)
        .agg({
            "Номенклатура": shortest_value,
            'Склад': join_ordered,
            "Список аналогов": lambda x: max(x.dropna(), key=len, default=None),
            "Тип техники": join_ordered,
            'Артикул': 'first',
        })
    )

    qty = (
        df.groupby(["Год", "Месяц", "Номер группы"], as_index=False)
        .agg({
            "Расход": "sum",
            "Приход": "sum",
            "Конечный остаток": 'last'
        })
    )

    for col in ["Расход", "Приход", "Конечный остаток", 'Номер группы']:
        qty[col] = qty[col].apply(lambda x: int(x) if x.is_integer() else x)

    result = qty.merge(meta, on="Номер группы", how="left")
    return result


def prepare_sales(agg: pd.DataFrame, df_quarter: pd.DataFrame) -> pd.DataFrame:
    agg['Год'] = agg['Год'].astype(int)
    agg['Месяц'] = agg['Месяц'].astype(int)
    df_quarter['Год'] = df_quarter['Год'].astype(int)
    df_quarter['Месяц'] = df_quarter['Месяц'].astype(int)

    merged = agg.merge(
        df_quarter[['Год','Месяц','Номер группы','Ремонт']],
        on=['Год','Месяц','Номер группы'],
        how='left'
    )

    merged['Продажа'] = merged['Расход'] - merged['Ремонт'].fillna(0)

    merged = merged.drop(columns=['Расход', 'Ремонт'])

    return merged


def normalize_analog_lists(df, col_group="Номер группы", col_analogs="Список аналогов"):

    df = df.copy()

    group_max_analogs = (
        df.groupby(col_group)[col_analogs]
        .apply(lambda s: max(
            (v for v in s if isinstance(v, tuple)),
            key=len,
            default=None
        ))
        .to_dict()
    )

    df[col_analogs] = df.apply(
        lambda row: group_max_analogs.get(row[col_group], row[col_analogs]),
        axis=1
    )

    return df