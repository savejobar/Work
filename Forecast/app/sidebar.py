from __future__ import annotations

import os
import tempfile

import pandas as pd
import streamlit as st


def render_sidebar() -> pd.DataFrame | None:
    """
    Отрисовывает сайдбар с загрузкой файлов и кнопкой запуска пайплайна.

    Returns:
        DataFrame после run_full_pipeline() или None если данные ещё не загружены.
    """
    with st.sidebar:
        st.markdown("## Источник данных")

        f1 = st.file_uploader(
            "Запчасти списанные в ремонт (.xlsx)",
            type=["xlsx"],
            key="repair_file",
        )
        f2 = st.file_uploader(
            "Остатки и обороты (.xlsx)",
            type=["xlsx"],
            key="stock_file",
        )

        st.divider()

        if f1 and f2:
            if st.button("Обработать данные", type="primary", use_container_width=True):
                _run_pipeline(f1, f2)
        else:
            st.info("Загрузите оба файла для начала работы")

        # Показываем статус загрузки
        if "df_main" in st.session_state:
            df = st.session_state["df_main"]
            st.success(f"Данные загружены: {len(df):,} строк")
            st.caption(
                f"Групп: {df['Номер группы'].nunique():,}  |  "
                f"Период: {int(df['Год'].min())}-{int(df['Год'].max())}"
            )

    return st.session_state.get("df_main")


def _run_pipeline(f1, f2) -> None:
    """Записывает файлы во временную папку и запускает run_full_pipeline."""
    # Импорт здесь чтобы не замедлять старт приложения
    from pipeline.runner import run_full_pipeline

    with st.spinner("Обрабатываем данные..."):
        try:
            with tempfile.TemporaryDirectory() as tmp:
                p1 = os.path.join(tmp, "repair.xlsx")
                p2 = os.path.join(tmp, "stock.xlsx")

            with open(p1, "wb") as fh:
                fh.write(f1.getvalue())  # getvalue() не зависит от позиции курсора
            with open(p2, "wb") as fh:
                fh.write(f2.getvalue())

                df = run_full_pipeline(repair_path=p1, stock_path=p2)

            st.session_state["df_main"] = df
            st.rerun()

        except Exception as e:
            st.error(f"Ошибка при обработке: {e}")
