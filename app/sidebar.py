from __future__ import annotations

import io
import tempfile
import hashlib
from pathlib import Path

import pandas as pd
import streamlit as st

from forecasting.runner import MONTH_RU
from readers.loaders import sanitize_excel_dataframe, validate_upload_size
    

REPAIR_REQUIRED_RAW_COLS = [
    "Дата",
    "Номенклатура",
    "Номенклатура.Артикул",
    "Номенклатура.Оригинальный номер",
    "Номенклатура.Оригинальный номер расширенный",
    "Машина",
]

STOCK_REQUIRED_RAW_COLS = [
    "Номенклатура",
    "Артикул",
    "Оригинальный номер",
    "Код",
    "Документ движения (Регистратор)",
    "Контрагент",
]


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
        "search_lookup_signature",
        "search_lookup_results",
        "search_not_found",
    ]
    for key in keys_to_drop:
        st.session_state.pop(key, None)

    search_select_keys = [
        key for key in list(st.session_state.keys())
        if key.startswith("search_select_")
    ]
    for key in search_select_keys:
        st.session_state.pop(key, None)


@st.cache_data(show_spinner=False)
def build_processed_excel(df: pd.DataFrame) -> bytes:
    safe_df = sanitize_excel_dataframe(df)
    export_df = safe_df.rename(columns={"Ремонт": "Ремонт подъемников"})

    buf = io.BytesIO()
    export_df.to_excel(buf, index=False, engine="openpyxl")
    buf.seek(0)
    return buf.getvalue()


def render_sidebar() -> pd.DataFrame | None:
    """
    Отрисовывает сайдбар с загрузкой файлов и кнопкой запуска пайплайна.
    """
    with st.sidebar:
        st.markdown("## Источники данных")

        with st.expander("Требования к файлам", expanded=False):
            st.markdown("**Запчасти списанные в ремонт**")
            st.caption("Обязательные поля в исходном файле:")
            st.code("\n".join(REPAIR_REQUIRED_RAW_COLS), language="text")

            st.markdown("**Остатки и обороты**")
            st.caption("Обязательные поля в исходном файле:")
            st.code("\n".join(STOCK_REQUIRED_RAW_COLS), language="text")

            st.warning(
                "Не забудьте предобработать оба файла, приведя их к текстовому формату в Excel. "
                "С другими колонками или другой структурой будет ошибка."
            )

        st.markdown("**Отчет: Запчасти списанные в ремонт**")
        f1 = st.file_uploader(
            "Запчасти списанные в ремонт",
            type=["xlsx", "xls"],
            key="repair_file",
            label_visibility="collapsed",
        )
        st.markdown("**Отчет: Остатки и обороты**")
        f2 = st.file_uploader(
            "Остатки и обороты",
            type=["xlsx", "xls"],
            key="stock_file",
            label_visibility="collapsed",
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
                if st.button("Подготовить файл Excel", width="stretch", key="prepare_processed_excel"):
                    with st.spinner("Готовим файл для скачивания..."):
                        st.session_state["processed_excel"] = build_processed_excel(df)
                    st.rerun()
            else:
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

