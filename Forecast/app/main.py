import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import streamlit as st

st.set_page_config(
    page_title="Прогноз запчастей",
    layout="wide",
    initial_sidebar_state="expanded",
)

from app.sidebar import render_sidebar
from app.components import (
    render_search,
    render_params,
    render_metrics,
    render_table,
    render_outliers,
    render_summary_table,
)
from app.charts import render_chart
from forecasting.runner import run_group_forecast
from readers.exporters import build_batch_excel
from app.logger import SessionLogger
log = SessionLogger()

@st.fragment
def download_section(results, show_clean: bool, steps: int, iqr_factor: float, croston_threshold: float):
    results_key = str([r.group_id for r in results]) + str(show_clean) + str(steps) + str(iqr_factor) + str(croston_threshold)
    if "batch_excel" not in st.session_state or st.session_state.get("batch_key") != results_key:
        st.session_state["batch_excel"] = build_batch_excel(results, show_clean=show_clean)
        st.session_state["batch_key"] = results_key

    st.download_button(
        label="Скачать Excel с прогнозами",
        data=st.session_state["batch_excel"],
        file_name="forecast_batch.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
    )


st.title("Прогноз спроса на запасные части")
st.caption(
    "Загрузите два исходных отчёта → найдите запчасть по артикулу или загрузите список запчастей → получите прогноз"
)

# Сайдбар: загрузка и пайплайн

df = render_sidebar()

if df is None:
    st.info("Загрузите данные в боковой панели, чтобы начать")
    st.stop()

# Параметры прогноза

steps, iqr_factor, croston_threshold, show_clean = render_params()

st.divider()

# Поиск запчасти

group_ids = render_search(df)

if not group_ids:
    st.info("Введите артикул для построения прогноза")
    st.stop()

is_batch = len(group_ids) > 1

# Кэш ключ — все параметры которые влияют на прогноз
forecast_key = str(group_ids) + str(steps) + str(iqr_factor) + str(croston_threshold)

if "forecast_results" not in st.session_state or st.session_state.get("forecast_key") != forecast_key:
    results = []

    if is_batch:
        progress = st.progress(0, text="Начинаем обработку...")

    for i, group_id in enumerate(group_ids):
        try:
            if is_batch:
                meta = df[df["Номер группы"] == group_id].iloc[0]
                progress.progress(
                    int((i / len(group_ids)) * 100),
                    text=f"Обрабатываем: {str(meta['Номенклатура'])[:50]}...",
                )
            result = run_group_forecast(
                df,
                group_id=group_id,
                steps=steps,
                iqr_factor=iqr_factor,
                croston_threshold=croston_threshold,
            )
            results.append(result)
            log.info(f"Прогноз группы {group_id} | {result.nomenclature[:40]} | продажи={result.sale.method} | ремонт={result.repair.method}")  # ← добавь сюда
        except Exception as e:
            log.error(f"Ошибка прогноза группы {group_id}: {e}")  # ← добавь
            st.error(f"Ошибка для группы {group_id}: {e}")
            continue

    if is_batch:
        progress.progress(100, text="Готово!")
        progress.empty()

    st.session_state["forecast_results"] = results
    st.session_state["forecast_key"] = forecast_key

results = st.session_state["forecast_results"]

if not results:
    st.error("Не удалось построить ни одного прогноза")
    st.stop()

# Вывод результатов
if is_batch:
    download_section(results, show_clean=show_clean, steps=steps, iqr_factor=iqr_factor, croston_threshold=croston_threshold)
    st.markdown("### Сводная таблица")
    render_summary_table(results)
else:
    result = results[0]
    st.markdown(f"### {result.nomenclature}")
    render_metrics(result)
    st.divider()
    tab_chart, tab_table = st.tabs(["График", "Таблица"])
    with tab_chart:
        render_chart(result, show_clean=show_clean)
    with tab_table:
        render_table(result)
    render_outliers(result)