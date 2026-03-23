from __future__ import annotations

import os
import io
import tempfile

import pandas as pd
import streamlit as st

from forecasting.runner import MONTH_RU


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
                buf = io.BytesIO()
                df.to_excel(buf, index=False, engine="openpyxl")
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
    """
    Запускает полный ETL-пайплайн из загруженных файлов.
    """
    from pipeline.runner import run_full_pipeline
    from app.logger import SessionLogger

    log = SessionLogger()
    log.info(f"Загрузка файлов: {f1.name}, {f2.name}")

    with st.spinner("Обрабатываем данные..."):
        try:
            b1 = f1.getvalue()
            b2 = f2.getvalue()

            with tempfile.TemporaryDirectory() as tmp:
                p1 = os.path.join(tmp, f1.name)
                p2 = os.path.join(tmp, f2.name)

                with open(p1, "wb") as fh:
                    fh.write(b1)
                with open(p2, "wb") as fh:
                    fh.write(b2)

                df = run_full_pipeline(repair_path=p1, stock_path=p2)

            log.info(f"Пайплайн завершён: {len(df)} строк, {df['Номер группы'].nunique()} групп")
            st.session_state["df_main"] = df
            st.session_state.pop("processed_excel", None)
            st.rerun()

        except Exception as e:
            log.error(f"Ошибка пайплайна: {e}")
            st.error(f"Ошибка при обработке: {e}")
