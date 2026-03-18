import re
import os
import ast
from collections import defaultdict

from typing import Optional, List, Iterable, Any, Dict, Set, Tuple
import numpy as np
import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings

warnings.filterwarnings('ignore')


_ANALOG_SUFFIXES = (
    "A", "А",           # латиница + кириллица
    "B", "C", "F", "K",
    "GT", "IN",
    " Б/У",
    "Б/У",
    "-L1",
    " JR", " NSK",
)
_ANALOG_PREFIXES = (
    "W",      # WG0300002  → G0300002
    "AVX",    # AVX10X1125 → 10X1125
    "4Т-",    # 4Т-33006   → 33006
)


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


def normalize(val) -> str | None:
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
        'Документ.Склад': 'Склады_ремонт',
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
            'Склады_продажа': join_ordered,
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
            "Конечный остаток": 'sum'
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


def article_forms(val) -> list[str]:
    base = normalize(val)
    if base is None:
        return []

    forms: list[str] = [base]

    # ведущие нули: "00773607100401010" → "773607100401010"
    stripped = base.lstrip("0")
    if stripped and stripped != base and stripped[0] not in "-/. _":
        forms.append(stripped)

    for suf in _ANALOG_SUFFIXES:
        s = suf.upper()
        if base.endswith(s) and len(base) > len(s):
            root = base[: -len(s)].rstrip()
            if root and root not in forms:
                forms.append(root)

    for pre in _ANALOG_PREFIXES:
        p = pre.upper()
        if base.startswith(p) and len(base) > len(p):
            root = base[len(p):].lstrip()
            if root and root not in forms:
                forms.append(root)

    return forms


def fill_missing_months(df: pd.DataFrame) -> pd.DataFrame:

    META_COLS = [
        'Бренды', 'Серии техники', 'Типы подъемников', 'Типы двигателя',
        'Номенклатура.Тип техники', 'Склады_продажа', 'Номера машин', 'Склады_ремонт',
        'Тип техники', 'Артикул', 'Номенклатура', 'Список аналогов',
    ]
    ZERO_COLS  = ['Продажа', 'Ремонт', 'Приход']
    FFILL_COLS = ['Конечный остаток']

    meta_cols_present  = [c for c in META_COLS  if c in df.columns]
    zero_cols_present  = [c for c in ZERO_COLS  if c in df.columns]
    ffill_cols_present = [c for c in FFILL_COLS if c in df.columns]

    all_months = (
        df[['Год', 'Месяц']]
        .drop_duplicates()
        .sort_values(['Год', 'Месяц'])
        .reset_index(drop=True)
    )
    all_months['_ym'] = all_months['Год'] * 100 + all_months['Месяц']

    active = df[(df['Продажа'] > 0) | (df['Ремонт'] > 0)]
    group_start = (
        active.groupby('Номер группы')
        .apply(lambda g: (g['Год'] * 100 + g['Месяц']).min())
        .rename('_start_ym')
        .reset_index()
    )

    grid = group_start.merge(all_months, how='cross')
    grid = grid[grid['_ym'] >= grid['_start_ym']].drop(columns=['_start_ym', '_ym'])

    df = df.copy()
    df['_original'] = 1

    merged = grid.merge(df, on=['Год', 'Месяц', 'Номер группы'], how='left')

    if meta_cols_present:
        meta = df.groupby('Номер группы')[meta_cols_present].first()
        for col in meta_cols_present:
            merged[col] = merged[col].combine_first(
                merged['Номер группы'].map(meta[col])
            )

    if zero_cols_present:
        merged[zero_cols_present] = merged[zero_cols_present].fillna(0)

    merged = merged.sort_values(['Номер группы', 'Год', 'Месяц'])
    for col in ['Продажа', 'Ремонт']:
        if col not in merged.columns:
            continue
        mask = merged.groupby('Номер группы')[col].transform(lambda s: (s > 0).cummax())
        merged.loc[~mask, col] = pd.NA

    if ffill_cols_present:
        merged[ffill_cols_present] = (
            merged.groupby('Номер группы')[ffill_cols_present]
            .transform(lambda s: s.ffill())
        )

    merged = merged.sort_values(['Год', 'Месяц', 'Номер группы']).reset_index(drop=True)

    merged['is_synthetic'] = merged['_original'].isna().replace(False, pd.NA).astype('Int8')
    merged.drop(columns='_original', inplace=True)

    return merged