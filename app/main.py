import pandas as pd
from pathlib import Path
import streamlit as st

_FAQ_PATH = Path(__file__).with_name("faq.md")

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
from forecasting.runner import run_group_forecast, _get_train_end
from readers.exporters import build_batch_excel
from app.logger import SessionLogger


def apply_current_month_policy(df: pd.DataFrame, include_current_month: bool) -> pd.DataFrame:
    """
    Возвращает рабочий DataFrame для прогноза.

    Если include_current_month=False, исключает из обучения текущий календарный
    месяц относительно системной даты (`pd.Timestamp.today()`). Поведение не
    зависит от того, какой месяц последний в данных.
    """
    if include_current_month or df.empty:
        return df

    today = pd.Timestamp.today()
    current_year = int(today.year)
    current_month = int(today.month)

    return df.loc[
        ~((df["Год"] == current_year) & (df["Месяц"] == current_month))
    ].copy()

log = SessionLogger()

@st.fragment
def download_section(
    results,
    show_clean: bool,
    steps: int,
    iqr_factor: float | None,
    croston_threshold: float,
    include_current_month: bool,
    forecast_anchor: str,
):
    dataset_version = st.session_state.get("dataset_version", "no_dataset")
    results_key = (
        str(dataset_version)
        + str([r.group_id for r in results])
        + str(show_clean)
        + str(steps)
        + str(iqr_factor)
        + str(croston_threshold)
        + str(include_current_month)
        + str(forecast_anchor)
    )

    if "batch_excel" not in st.session_state or st.session_state.get("batch_key") != results_key:
        st.session_state["batch_excel"] = build_batch_excel(results, show_clean=show_clean)
        st.session_state["batch_key"] = results_key

    st.download_button(
        label="Скачать Excel с прогнозами",
        data=st.session_state["batch_excel"],
        file_name="forecast.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
    )

st.title("Прогноз спроса на запасные части")

st.caption(
    "Загрузите два исходных отчёта → найдите запчасть по артикулу или загрузите список запчастей → получите прогноз"
)

with st.expander("FAQ - как работает прогноз?"):
    st.markdown(_FAQ_PATH.read_text(encoding="utf-8"))
    
# Сайдбар: загрузка и пайплайн

df = render_sidebar()

if df is None:
    st.info("Загрузите данные в боковой панели, чтобы начать")
    st.stop()

# Параметры прогноза
steps, iqr_factor, croston_threshold, show_clean, include_current_month = render_params()

# Рабочий DataFrame для прогноза
df_work = apply_current_month_policy(df, include_current_month)

if df_work.empty:
    st.error("После исключения текущего месяца данных для прогноза не осталось")
    st.stop()

st.divider()

# Поиск запчасти
group_ids = render_search(df_work)

if not group_ids:
    st.info("Введите артикул для построения прогноза")
    st.stop()

is_batch = len(group_ids) > 1

dataset_version = st.session_state.get("dataset_version", "no_dataset")
forecast_start = pd.Timestamp.today().normalize().replace(day=1)
forecast_anchor = forecast_start.strftime("%Y-%m")
forecast_key = (
    str(dataset_version)
    + str(group_ids)
    + str(steps)
    + str(iqr_factor)
    + str(croston_threshold)
    + str(include_current_month)
    + str(forecast_anchor)
)

if "forecast_results" in st.session_state and st.session_state.get("forecast_key") == forecast_key:
    results = st.session_state["forecast_results"]
else:
    results = []
    had_errors = False
    train_end_year, train_end_month = _get_train_end(df_work)
    group_frames = {
        int(group_id): grp.sort_values(["Год", "Месяц"], kind="mergesort").copy()
        for group_id, grp in (
            df_work[df_work["Номер группы"].isin(group_ids)]
            .groupby("Номер группы", sort=False)
        )
    }

    if is_batch:
        progress = st.progress(0, text="Начинаем обработку...")

    for i, group_id in enumerate(group_ids):
        try:
            group_key = int(group_id)
            grp = group_frames.get(group_key)
            if grp is None or grp.empty:
                raise ValueError(f"Группа {group_id} не найдена")

            if is_batch:
                meta = grp.iloc[0]
                progress.progress(
                    int((i / len(group_ids)) * 100),
                    text=f"Обрабатываем: {str(meta['Номенклатура'])[:50]}...",
                )

            result = run_group_forecast(
                grp=grp,
                group_id=group_key,
                train_end_year=train_end_year,
                train_end_month=train_end_month,
                forecast_start=forecast_start,
                steps=steps,
                iqr_factor=iqr_factor,
                croston_threshold=croston_threshold,
            )

            results.append(result)
            log.info(
                f"Прогноз группы {group_id} | {result.nomenclature[:40]} | "
                f"продажи={result.sale.method} | ремонт={result.repair.method}"
            )
        except Exception as e:
            had_errors = True
            log.error(f"Ошибка прогноза группы {group_id}: {e}")
            st.error(f"Ошибка для группы {group_id}: {e}")
            continue

    if is_batch:
        progress.progress(100, text="Готово!")
        progress.empty()

    if not had_errors:
        st.session_state["forecast_results"] = results
        st.session_state["forecast_key"] = forecast_key

if not results:
    st.error("Не удалось построить ни одного прогноза")
    st.stop()

# Вывод результатов
download_section(
    results,
    show_clean=show_clean,
    steps=steps,
    iqr_factor=iqr_factor,
    croston_threshold=croston_threshold,
    include_current_month=include_current_month,
    forecast_anchor=forecast_anchor,
)

if is_batch:
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
