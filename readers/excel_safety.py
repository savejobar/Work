import pandas as pd
from typing import Any

DANGEROUS_EXCEL_PREFIXES: tuple[str, ...] = ("=", "+", "-", "@")


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