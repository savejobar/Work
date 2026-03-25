from __future__ import annotations

import io
import tempfile
import hashlib
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from forecasting.runner import MONTH_RU
from readers.loaders import sanitize_excel_dataframe, validate_upload_size
    

def _clear_search_state() -> None:
    """
    Сбрасывает состояние поиска при загрузке нового датасета.
    """
    keys_to_drop = [
        "search_mode",
        "submitted_mode",
        "submitted_articles",
        "article_input",
        "articles_file",
    ]
    for key in keys_to_drop:
        st.session_state.pop(key, None)

    search_select_keys = [
        key for key in list(st.session_state.keys())
        if key.startswith("search_select_")
    ]
    for key in search_select_keys:
        st.session_state.pop(key, None)


def render_sidebar() -> pd.DataFrame | None:
    """
    Отрисовывает сайдбар с загрузкой файлов и кнопкой запуска пайплайна.
    """
    with st.sidebar:
        st.markdown("## Источники данных")

        f1 = st.file_uploader(
            "Запчасти списанные в ремонт",
            type=["xlsx", "xls"],
            key="repair_file",
        )
        f2 = st.file_uploader(
            "Остатки и обороты",
            type=["xlsx", "xls"],
            key="stock_file",
        )

        st.divider()

        if f1 and f2:
            if st.button("Обработать данные", type="primary", width="stretch"):
                _run_pipeline(f1, f2)
        else:
            st.info("Загрузите оба файла для начала работы")

        pipeline_notice = st.session_state.get("pipeline_notice")
        if pipeline_notice:
            st.warning(pipeline_notice)

        if "df_main" in st.session_state:
            df = st.session_state["df_main"]
            ym = df["Год"] * 100 + df["Месяц"]
            min_row = df.loc[ym.idxmin()]
            max_row = df.loc[ym.idxmax()]
            start = f"{MONTH_RU[int(min_row['Месяц'])]} {int(min_row['Год'])}"
            end = f"{MONTH_RU[int(max_row['Месяц'])]} {int(max_row['Год'])}"
            st.success(f"Данные загружены: {len(df):,} строк")
            st.caption(f"Групп: {df['Номер группы'].nunique():,}")
            st.caption(f"Период: {start} — {end}")

            st.divider()

            if "processed_excel" not in st.session_state:
                safe_df = sanitize_excel_dataframe(df)
                buf = io.BytesIO()
                safe_df.to_excel(buf, index=False, engine="openpyxl")
                buf.seek(0)
                st.session_state["processed_excel"] = buf.getvalue()

            st.download_button(
                label="Скачать обработанные данные",
                data=st.session_state["processed_excel"],
                file_name="processed_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                width="stretch",
            )

    return st.session_state.get("df_main")


def _run_pipeline(f1, f2) -> None:
    from pipeline.runner import run_full_pipeline
    from app.logger import SessionLogger

    log = SessionLogger()

    try:
        validate_upload_size(f1, "Запчасти списанные в ремонт")
        validate_upload_size(f2, "Остатки и обороты")
    except ValueError as e:
        st.error(str(e))
        return

    log.info(f"Загрузка файлов: {f1.name}, {f2.name}")

    with st.spinner("Обрабатываем данные..."):
        try:
            b1 = f1.getvalue()
            b2 = f2.getvalue()

            dataset_version = hashlib.sha256(b1 + b"||" + b2).hexdigest()

            with tempfile.TemporaryDirectory() as tmp:
                suffix1 = Path(f1.name).suffix.lower()
                suffix2 = Path(f2.name).suffix.lower()

                with tempfile.NamedTemporaryFile(dir=tmp, suffix=suffix1, delete=False) as fh1:
                    fh1.write(b1)
                    p1 = fh1.name

                with tempfile.NamedTemporaryFile(dir=tmp, suffix=suffix2, delete=False) as fh2:
                    fh2.write(b2)
                    p2 = fh2.name

                df = run_full_pipeline(repair_path=p1, stock_path=p2)

            log.info(f"Пайплайн завершён: {len(df)} строк, {df['Номер группы'].nunique()} групп")
            st.session_state["df_main"] = df
            st.session_state["dataset_version"] = dataset_version

            st.session_state.pop("processed_excel", None)
            st.session_state.pop("forecast_results", None)
            st.session_state.pop("forecast_key", None)
            st.session_state.pop("batch_excel", None)
            st.session_state.pop("batch_key", None)
            st.session_state.pop("pipeline_notice", None)
            _clear_search_state()

            st.rerun()

        except Exception as e:
            error_code = type(e).__name__
            if "df_main" in st.session_state:
                st.session_state["pipeline_notice"] = (
                    "Новая загрузка не удалась, показан предыдущий набор данных."
                )
            else:
                st.session_state.pop("pipeline_notice", None)
            log.error(f"PIPELINE_ERROR code={error_code}")
            st.error(f"Ошибка при обработке: {e}")

