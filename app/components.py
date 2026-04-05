from __future__ import annotations
import re 
from typing import Any

import pandas as pd
import streamlit as st

from forecasting.runner import GroupForecastResult, build_result_summary, forecast_table
from readers.loaders import validate_upload_size

 
def render_search(
    df: pd.DataFrame,
    include_current_month: bool,
) -> tuple[list[int] | None, Any]:
    """
    Возвращает список номеров групп или None.
    """
    from forecasting.runner import find_groups_by_article

    st.markdown("### Поиск запчасти")

    mode = st.radio(
        "Режим",
        ["Одиночный", "Списком"],
        horizontal=True,
        label_visibility="collapsed",
        key="search_mode",
    )

    if "submitted_articles" not in st.session_state:
        st.session_state["submitted_articles"] = None
    if "submitted_mode" not in st.session_state:
        st.session_state["submitted_mode"] = mode

    if st.session_state["submitted_mode"] != mode:
        st.session_state["submitted_mode"] = mode
        st.session_state["submitted_articles"] = None

    uploaded = None

    if mode == "Одиночный":
        with st.form("article_search_form", border=False):
            col_inp, col_btn = st.columns([3, 1], vertical_alignment="bottom")

            with col_inp:
                st.text_input(
                    "Артикул",
                    placeholder="Введите артикул или его часть...",
                    label_visibility="collapsed",
                    key="article_input",
                )

            with col_btn:
                submitted = st.form_submit_button("Прогноз", use_container_width=True)

    else:
        with st.form("article_search_form_file", border=False):
            uploaded = st.file_uploader(
                "Загрузите Excel со списком артикулов",
                type=["xlsx", "xls"],
                key="articles_file",
            )
            submitted = st.form_submit_button("Прогноз", use_container_width=True)

    progress_slot = st.empty() if mode == "Списком" else None

    if submitted:
        if mode == "Одиночный":
            raw = st.session_state.get("article_input", "").strip()
            if not raw:
                st.session_state["submitted_articles"] = None
                if progress_slot is not None:
                    progress_slot.empty()
                return None, progress_slot

            st.session_state["submitted_articles"] = [
                a for a in re.split(r"[,\s]+", raw) if a
            ]

        else:
            if uploaded is None:
                st.session_state["submitted_articles"] = None
                st.caption("Файл должен содержать колонку 'Артикул'")
                if progress_slot is not None:
                    progress_slot.empty()
                return None, progress_slot

            try:
                validate_upload_size(uploaded, "Список артикулов", max_bytes=5 * 1024 * 1024)
                arts_df = pd.read_excel(uploaded, dtype=str, engine="calamine")
            except Exception as e:
                st.session_state["submitted_articles"] = None
                st.error(f"Ошибка чтения файла: {e}")
                if progress_slot is not None:
                    progress_slot.empty()
                return None, progress_slot

            normalized_cols = {
                str(col).strip().casefold(): col
                for col in arts_df.columns
            }

            article_col = (
            normalized_cols.get("артикул")
            or normalized_cols.get("номенклатура.артикул")
            )
            if article_col is None:
                st.session_state["submitted_articles"] = None
                st.error("В файле нет колонки 'Артикул'")
                if progress_slot is not None:
                    progress_slot.empty()
                return None, progress_slot

            articles = [
                a
                for a in arts_df[article_col].dropna().str.strip().tolist()
                if a
            ]
            articles = list(dict.fromkeys(articles))

            st.session_state["submitted_articles"] = articles
            st.caption(f"Загружено артикулов: {len(articles)}")

    articles = st.session_state.get("submitted_articles")
    if not articles:
        if progress_slot is not None:
            progress_slot.empty()
        return None, progress_slot

    dataset_version = st.session_state.get("dataset_version", "no_dataset")
    search_signature = (
        str(dataset_version)
        + str(mode)
        + str(articles)
        + str(include_current_month)
    )

    if st.session_state.get("search_lookup_signature") != search_signature:
        progress = None
        if progress_slot is not None:
            progress = progress_slot.progress(
                1,
                text=f"Этап 1/3: Ищем группы по артикулам (0/{len(articles)})...",
            )

        search_results = []
        not_found = []
        for i, art in enumerate(articles, start=1):
            hits = find_groups_by_article(df, art)
            search_results.append({"article": art, "hits": hits})
            if not hits:
                not_found.append(art)

            if progress is not None:
                pct = max(1, int((i / max(len(articles), 1)) * 33))
                progress.progress(
                    pct,
                    text=f"Этап 1/3: Ищем группы по артикулам ({i}/{len(articles)}) — {art}...",
                )

        st.session_state["search_lookup_signature"] = search_signature
        st.session_state["search_lookup_results"] = search_results
        st.session_state["search_not_found"] = not_found
    else:
        search_results = st.session_state.get("search_lookup_results", [])
        not_found = st.session_state.get("search_not_found", [])

    group_ids = []

    for i, item in enumerate(search_results):
        art = item["article"]
        hits = item["hits"]

        if not hits:
            continue

        if len(hits) == 1:
            group_ids.append(hits[0]["Номер группы"])
            continue

        option_map = {
            str(h["Номер группы"]): h
            for h in hits
        }

        selected_key = st.selectbox(
            f"Найдено несколько для '{art}':",
            options=list(option_map.keys()),
            format_func=lambda key: (
                f"{option_map[key]['Совпадение']} — "
                f"{str(option_map[key]['Номенклатура'])[:50]}"
            ),
            key=f"search_select_{i}_{art}",
        )

        group_ids.append(option_map[selected_key]["Номер группы"])

    if not_found:
        st.warning(f"Не найдено: {', '.join(not_found)}")

    group_ids = list(dict.fromkeys(group_ids))

    return (group_ids if group_ids else None), progress_slot


def render_params() -> tuple[int, float | None, float, bool, bool]:
    """
    Блок параметров прогноза.
    """
    st.markdown("### Параметры прогноза")

    no_outliers = st.session_state.get("no_outliers", True)

    col1, col2, col3, gap1, col4, gap2, col5 = st.columns(
        [1, 1, 1, 0.3, 0.9, 0.1, 1.2],
        vertical_alignment="bottom",
    )

    with col1:
        steps = st.slider(
            "Горизонт (мес)", min_value=1, max_value=12, value=3,
            help="Количество месяцев вперёд для прогноза",
        )
    with col2:
        iqr_slider_value = st.slider(
            "IQR ×", min_value=1.0, max_value=5.0, value=1.5, step=0.1,
            help="Определяет чувствительность к выбросам. Малое значение — больше точек удаляется. Влияет на стабильность прогноза.",
            disabled=no_outliers,
            key="iqr_factor_slider",
        )
    with col3:
        croston_threshold = st.slider(
            "Порог нулей для TSB", min_value=0.1, max_value=0.9,
            value=0.50, step=0.05,
            help="Если доля нулей в серии выше порога — применяется TSB",
        )
    with col4:
        show_clean = st.toggle(
            "Очищенный ряд",
            value=True,
            help="Отображать серию после замены выбросов",
        )
    with col5:
        include_current_month = st.toggle(
            "Текущий месяц",
            value=True,
            help="Если выключено, из расчёта ремонтов и продаж исключается текущий календарный месяц по системной дате, но берется последний конечный остаток по каждому коду в пределах группы. Это полезно, если месяц ещё не завершён и данные неполные.",
        )

    sub_cols = st.columns([1, 1, 1, 0.2, 0.9, 0.2, 1.2])
    with sub_cols[1]:
        no_outliers = st.toggle(
            "Не удалять выбросы",
            value=True,
            key="no_outliers",
        )

    iqr_factor = None if no_outliers else iqr_slider_value

    return steps, iqr_factor, croston_threshold, show_clean, include_current_month


def render_metrics(result: GroupForecastResult) -> None:
    """
    Карточки с суммарными прогнозными значениями.
    """
    summary = build_result_summary(result)
    n_months = len(result.fc_months)

    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("Номер группы", result.group_id)
    c2.metric("Артикул", result.article[:20])
    c3.metric("Конечный остаток", f"{result.ending_stock:,.1f}")
    c4.metric(f"Продажи ({n_months} мес.)", f"{summary['sale_total']:,.1f}")
    c5.metric(f"Ремонт ({n_months} мес.)", f"{summary['repair_total']:,.1f}")
    c6.metric("Итого спрос", f"{summary['total_demand']:,.1f}")
    c7.metric("Нужно заказать", f"{summary['need_to_order']:,.1f}")


def render_table(result: GroupForecastResult) -> None:
    """
    Таблица прогноза по месяцам с итоговой строкой.
    """
    df_table = forecast_table(result)

    num_cols = ["Прогноз продаж", "Прогноз ремонта", "Итого спрос"]
    
    # Выделяем итоговую строку жирным через стилизацию
    def highlight_total(row):
        if row["Период"] == "ИТОГО":
            return ["font-weight: bold; background-color: #cbd5e1; color: #0f172a"] * len(row)
        return [""] * len(row)

    st.dataframe(
        df_table.style
        .apply(highlight_total, axis=1)
        .format({col: "{:.1f}" for col in num_cols}),
        width="stretch",
        hide_index=True,
    )


def render_outliers(result: GroupForecastResult) -> None:
    """
    Раскрывающийся блок с деталями выбросов.
    """
    sale_out = result.sale.outliers
    repair_out = result.repair.outliers

    if not sale_out and not repair_out:
        return

    with st.expander(
        f"Выбросы: продажи ({len(sale_out)}) | ремонт ({len(repair_out)})"
    ):
        col_s, col_r = st.columns(2)

        with col_s:
            st.markdown("**Продажи**")
            if sale_out:
                st.dataframe(pd.DataFrame(sale_out), hide_index=True, width="stretch")
            else:
                st.caption("Выбросов не обнаружено")

        with col_r:
            st.markdown("**Ремонт**")
            if repair_out:
                st.dataframe(pd.DataFrame(repair_out), hide_index=True, width="stretch")
            else:
                st.caption("Выбросов не обнаружено")


def render_summary_table(results: list) -> None:
    """
    Сводная таблица прогнозов по всем запчастям.
    """
    from forecasting.runner import MONTH_RU

    fc_months = results[0].fc_months
    rows = []
    for result in results:
        summary = build_result_summary(result)
        row = {
            "Артикул": result.article,
            "Номенклатура": result.nomenclature[:50],
            "Конечный остаток": round(float(result.ending_stock), 1),
            "Метод продаж": result.sale.method,
            "Метод ремонта": result.repair.method,
        }
        for i, (y, m) in enumerate(fc_months):
            lbl = f"{MONTH_RU[m]} {y}"
            row[f"{lbl} Продажи"] = round(float(result.sale.forecast.iloc[i]), 1)
            row[f"{lbl} Ремонт"] = round(float(result.repair.forecast.iloc[i]), 1)
        row["Итого продажи"] = summary["sale_total"]
        row["Итого ремонт"] = summary["repair_total"]
        row["Итого спрос"] = summary["total_demand"]
        row["Нужно заказать"] = summary["need_to_order"]
        rows.append(row)

    df_out = pd.DataFrame(rows)

    num_cols = [c for c in df_out.columns if df_out[c].dtype in ["float64", "float32"]]

    st.dataframe(
        df_out.style
        .set_properties(
            subset=["Нужно заказать"],
            **{
                "background-color": "#dcfce7",
                "color": "#166534",
                "font-weight": "bold",
            },
        )
        .format({col: "{:.1f}" for col in num_cols}),
        width="stretch",
        hide_index=True,
    )
