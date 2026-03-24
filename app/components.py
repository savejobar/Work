from __future__ import annotations
import re 

import pandas as pd
import streamlit as st

from forecasting.runner import GroupForecastResult, build_result_summary, forecast_table

NO_OUTLIER_REMOVAL = float('inf')


def render_search(df: pd.DataFrame) -> list[int] | None:
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
                submitted = st.form_submit_button("Найти", use_container_width=True)

    else:
        with st.form("article_search_form_file", border=False):
            uploaded = st.file_uploader(
                "Загрузите Excel со списком артикулов",
                type=["xlsx", "xls"],
                key="articles_file",
            )
            submitted = st.form_submit_button("Найти", use_container_width=True)

    if submitted:
        if mode == "Одиночный":
            raw = st.session_state.get("article_input", "").strip()
            if not raw:
                st.session_state["submitted_articles"] = None
                return None

            st.session_state["submitted_articles"] = [
                a for a in re.split(r"[,\s]+", raw) if a
            ]

        else:
            if uploaded is None:
                st.session_state["submitted_articles"] = None
                st.caption("Файл должен содержать колонку 'Артикул'")
                return None

            try:
                arts_df = pd.read_excel(uploaded, dtype=str, engine="calamine")
            except Exception as e:
                st.session_state["submitted_articles"] = None
                st.error(f"Ошибка чтения файла: {e}")
                return None

            if "Артикул" not in arts_df.columns:
                st.session_state["submitted_articles"] = None
                st.error("В файле нет колонки 'Артикул'")
                return None

            st.session_state["submitted_articles"] = [
                a for a in arts_df["Артикул"].dropna().str.strip().tolist() if a
            ]
            st.caption(f"Загружено артикулов: {len(st.session_state['submitted_articles'])}")

    articles = st.session_state.get("submitted_articles")
    if not articles:
        return None

    group_ids = []
    not_found = []

    for i, art in enumerate(articles):
        hits = find_groups_by_article(df, art)

        if not hits:
            not_found.append(art)
            continue

        if len(hits) == 1:
            group_ids.append(hits[0]["Номер группы"])
            continue

        selected = st.selectbox(
            f"Найдено несколько для '{art}':",
            options=hits,
            format_func=lambda h: f"[{h['Номер группы']}] {h['Артикул']} — {str(h['Номенклатура'])[:50]}",
            key=f"search_select_{i}_{art}",
        )
        group_ids.append(selected["Номер группы"])


    if not_found:
        st.warning(f"Не найдено: {', '.join(not_found)}")

    group_ids = list(dict.fromkeys(group_ids))

    return group_ids if group_ids else None


def render_params() -> tuple[int, float, float, bool, bool]:
    """
    Блок параметров прогноза.
    """
    st.markdown("### Параметры прогноза")

    col1, col2, col3, gap1, col4, gap2, col5 = st.columns([1, 1, 1, 0.2, 0.8, 0.2, 1])

    with col1:
        steps = st.slider(
            "Горизонт (мес)", min_value=1, max_value=12, value=3,
            help="Количество месяцев вперёд для прогноза",
        )

    with col2:
        iqr_factor = st.slider(
            "IQR ×", min_value=1.0, max_value=5.0, value=1.5, step=0.1,
            help="Определяет чувствительность к выбросам. Малое значение — больше точек удаляется. Влияет на стабильность прогноза."
        )
        no_outliers = st.toggle("Не удалять выбросы", value=True)
        if no_outliers:
            iqr_factor = NO_OUTLIER_REMOVAL

    with col3:
        croston_threshold = st.slider(
            "Порог нулей для TSB", min_value=0.1, max_value=0.8,
            value=0.40, step=0.05,
            help="Если доля нулей в серии выше порога — применяется TSB/Croston",
        )

    with col4:
        show_clean = st.toggle(
            "Показать очищенный ряд",
            value=True,
            help="Отображать серию после замены выбросов",
        )

    with col5:
        include_last_month = st.toggle(
            "Учитывать последний месяц",
            value=True,
            help="Если выключено, текущий календарный месяц полностью исключается из обучения. Тогда прогноз начинается с текущего месяца.",
        )


    return steps, iqr_factor, croston_threshold, show_clean, include_last_month


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
