import pandas as pd
from pathlib import Path
import sys
import streamlit as st

_FAQ_PATH = Path(__file__).with_name("faq.md")
_REPO_ROOT = Path(__file__).resolve().parents[1]

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

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
def download_section(results_key: str):
    if "batch_excel" not in st.session_state:
        return
    if st.session_state.get("batch_key") != results_key:
        return

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

ending_stock_map = (
    df.dropna(subset=["Конечный остаток"])
    .sort_values(["Номер группы", "Год", "Месяц"], kind="mergesort")
    .groupby("Номер группы")["Конечный остаток"]
    .last()
    .to_dict()
)

# Рабочий DataFrame для прогноза
df_work = apply_current_month_policy(df, include_current_month)

if df_work.empty:
    st.error("После исключения текущего месяца данных для прогноза не осталось")
    st.stop()

st.divider()

# Поиск запчасти
group_ids, progress_slot = render_search(df_work, include_current_month=include_current_month)

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
batch_results_key = forecast_key + str(show_clean)

if "forecast_results" in st.session_state and st.session_state.get("forecast_key") == forecast_key:
    results = st.session_state["forecast_results"]
    if progress_slot is not None:
        progress_slot.empty()
else:
    results = []
    had_errors = False
    st.session_state.pop("batch_excel", None)
    st.session_state.pop("batch_key", None)

    progress = None
    if is_batch and progress_slot is not None:
        progress = progress_slot.progress(
            34,
            text="Этап 2/3: Подготавливаем данные для прогноза...",
        )

    group_frames = {}
    train_end_year, train_end_month = _get_train_end(df_work)

    for i, group_id in enumerate(group_ids):
        group_key = int(group_id)
        grp = (
            df_work[df_work["Номер группы"] == group_key]
            .sort_values(["Год", "Месяц"], kind="mergesort")
            .copy()
        )
        group_frames[group_key] = grp

        if is_batch and progress is not None:
            name = f"группа {group_id}"
            if grp is not None and not grp.empty:
                name = str(grp.iloc[0]["Номенклатура"])[:50]
            pct = 34 + int(((i + 1) / max(len(group_ids), 1)) * 10)
            progress.progress(
                pct,
                text=(
                    f"Этап 2/3: Подготавливаем данные ({i + 1}/{len(group_ids)}) "
                    f"— {name}..."
                ),
            )

    if is_batch and progress is not None:
        progress.progress(
            45,
            text=f"Этап 2/3: Прогнозируем группы (0/{len(group_ids)})...",
        )

    for i, group_id in enumerate(group_ids):
        try:
            group_key = int(group_id)
            grp = group_frames.get(group_key)
            if grp is None or grp.empty:
                raise ValueError(f"Группа {group_id} не найдена")

            if is_batch and progress is not None:
                meta = grp.iloc[0]
                pct = 45 + int(((i + 1) / max(len(group_ids), 1)) * 35)
                progress.progress(
                    pct,
                    text=(
                        f"Этап 2/3: Прогнозируем ({i + 1}/{len(group_ids)}) "
                        f"— {str(meta['Номенклатура'])[:50]}..."
                    ),
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

            stock_override = ending_stock_map.get(group_key)
            if pd.notna(stock_override):
                result.ending_stock = float(stock_override)

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

    if results:
        if is_batch and progress is not None:
            progress.progress(81, text=f"Этап 3/3: Подготавливаем Excel-отчет (0/{len(results)})...")

            def update_excel_progress(done: int, total: int, result) -> None:
                label = str(result.nomenclature)[:50] if getattr(result, "nomenclature", None) else str(result.article)[:50]
                pct = 81 + int((done / max(total, 1)) * 18)
                progress.progress(
                    pct,
                    text=(
                        f"Этап 3/3: Подготавливаем Excel-отчет ({done}/{total}) "
                        f"— {label}..."
                    ),
                )
        else:
            update_excel_progress = None

        st.session_state["batch_excel"] = build_batch_excel(
            results,
            show_clean=show_clean,
            progress_callback=update_excel_progress,
        )
        st.session_state["batch_key"] = batch_results_key

    if is_batch and progress is not None:
        progress.progress(100, text="Готово!")
        progress.empty()
    elif progress_slot is not None:
        progress_slot.empty()

    if not had_errors:
        st.session_state["forecast_results"] = results
        st.session_state["forecast_key"] = forecast_key

if not results:
    st.error("Не удалось построить ни одного прогноза")
    st.stop()

# Вывод результатов
download_section(batch_results_key)

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
