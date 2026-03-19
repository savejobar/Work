from __future__ import annotations

import pandas as pd
import streamlit as st

from forecasting.runner import GroupForecastResult, forecast_table


def render_search(df: pd.DataFrame) -> int | None:
    """
    Возвращает выбранный Номер группы или None.
    """
    from forecasting.runner import find_groups_by_article

    st.markdown("### Поиск запчасти")

    col_inp, col_hint = st.columns([3, 1])
    with col_hint:
        if st.button("Пример", use_container_width=True):
            sample = df["Артикул"].dropna().iloc[0] if not df.empty else ""
            st.session_state["article_input"] = str(sample)

    with col_inp:
        article = st.text_input(
            "Артикул",
            placeholder="Введите артикул или его часть...",
            label_visibility="collapsed",
            key="article_input",
        )

    if not article.strip():
        return None

    hits = find_groups_by_article(df, article.strip())

    if not hits:
        st.error(f"Артикул '{article}' не найден")
        return None

    if len(hits) == 1:
        st.caption(f"Найдено: {hits[0]['Артикул']} — {hits[0]['Номенклатура'][:60]}")
        return hits[0]["Номер группы"]

    # Несколько совпадений — показываем выбор
    st.info(f"Найдено {len(hits)} совпадений. Выберите нужное:")
    options = {
        f"{h['Артикул']} — {str(h['Номенклатура'])[:50]}": h["Номер группы"]
        for h in hits
    }
    selected = st.selectbox(
        "Выберите запчасть",
        options=list(options.keys()),
        label_visibility="collapsed",
    )
    return options[selected]


def render_params() -> tuple[int, float, float, bool]:
    """
    Блок параметров прогноза.

    Returns:
        (steps, iqr_factor, croston_threshold, show_clean)
    """
    st.markdown("### Параметры прогноза")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        steps = st.slider(
            "Горизонт (мес.)", min_value=1, max_value=12, value=3,
            help="Количество месяцев вперёд для прогноза",
        )
    with col2:
        iqr_factor = st.slider(
            "IQR ×", min_value=1.0, max_value=5.0, value=1.5, step=0.1,
        )
        no_outliers = st.toggle("Не удалять выбросы", value=False)
        if no_outliers:
            iqr_factor = float("inf")  # порог бесконечный → ничего не срабатывает
    with col3:
        croston_threshold = st.slider(
            "Порог нулей для TSB", min_value=0.1, max_value=0.8,
            value=0.40, step=0.05,
            help="Если доля нулей в серии выше порога — применяется TSB/Croston",
        )
    with col4:
        show_clean = st.toggle(
            "Показать очищенный ряд", value=True,
            help="Отображать серию после замены выбросов",
        )

    return steps, iqr_factor, croston_threshold, show_clean


def render_metrics(result: GroupForecastResult) -> None:
    """Карточки с суммарными прогнозными значениями."""
    sale_total   = round(float(result.sale.forecast.sum()),   1)
    repair_total = round(float(result.repair.forecast.sum()), 1)
    total        = round(sale_total + repair_total, 1)
    n_months     = len(result.fc_months)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Номер группы", result.group_id)
    c2.metric("Артикул",      result.article[:20])
    c3.metric(f"Продажи ({n_months} мес.)", f"{sale_total:,.1f}")
    c4.metric(f"Ремонт ({n_months} мес.)",  f"{repair_total:,.1f}")
    c5.metric("Итого спрос",               f"{total:,.1f}")


def render_table(result: GroupForecastResult) -> None:
    """Таблица прогноза по месяцам с итоговой строкой."""
    df_table = forecast_table(result)

    # Выделяем итоговую строку жирным через стилизацию
    def highlight_total(row):
        if row["Период"] == "ИТОГО":
            return ["font-weight: bold; background-color: #cbd5e1; color: #0f172a"] * len(row)
        return [""] * len(row)

    st.dataframe(
        df_table.style.apply(highlight_total, axis=1),
        use_container_width=True,
        hide_index=True,
    )


def render_outliers(result: GroupForecastResult) -> None:
    """Раскрывающийся блок с деталями выбросов."""
    sale_out   = result.sale.outliers
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
                st.dataframe(pd.DataFrame(sale_out), hide_index=True, use_container_width=True)
            else:
                st.caption("Выбросов не обнаружено")

        with col_r:
            st.markdown("**Ремонт**")
            if repair_out:
                st.dataframe(pd.DataFrame(repair_out), hide_index=True, use_container_width=True)
            else:
                st.caption("Выбросов не обнаружено")
