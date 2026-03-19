"""
app/main.py
============
Точка входа Streamlit-приложения.

Запуск:
    streamlit run app/main.py
"""

import streamlit as st

st.set_page_config(
    page_title="Прогноз запчастей",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Импорты после set_page_config
from app.sidebar     import render_sidebar
from app.components  import (
    render_search,
    render_params,
    render_metrics,
    render_table,
    render_outliers,
)
from app.charts      import render_chart
from forecasting.runner import run_group_forecast


# ── Заголовок ─────────────────────────────────────────────────────────────────

st.title(" Прогноз спроса на запасные части")
st.caption(
    "Загрузите два исходных отчёта → найдите запчасть по артикулу → получите прогноз"
)

# ── Сайдбар: загрузка и пайплайн ─────────────────────────────────────────────

df = render_sidebar()

if df is None:
    st.info("Загрузите данные в боковой панели, чтобы начать")
    st.stop()

# ── Параметры прогноза ────────────────────────────────────────────────────────

steps, iqr_factor, croston_threshold, show_clean = render_params()

st.divider()

# ── Поиск запчасти ────────────────────────────────────────────────────────────

group_id = render_search(df)  # теперь возвращает int | None

if not group_id:
    st.info("Введите артикул для построения прогноза")
    st.stop()

# ── Прогноз ───────────────────────────────────────────────────────────────────

with st.spinner("Строим прогноз..."):
    try:
        result = run_group_forecast(
            df,
            group_id=group_id,   # ← было article=article
            steps=steps,
            iqr_factor=iqr_factor,
            croston_threshold=croston_threshold,
        )
    except Exception as e:
        st.error(str(e))
        st.stop()

# ── Вывод результатов ─────────────────────────────────────────────────────────

st.markdown(f"### {result.nomenclature}")

render_metrics(result)

st.divider()

tab_chart, tab_table = st.tabs(["График", "Таблица"])

with tab_chart:
    render_chart(result, show_clean=show_clean)

with tab_table:
    render_table(result)

render_outliers(result)
