from __future__ import annotations
import re 

import pandas as pd
import streamlit as st

from forecasting.runner import GroupForecastResult, forecast_table


def render_search(df: pd.DataFrame) -> list[int] | None:
    """
    Возвращает список номеров групп или None.
    Поддерживает одиночный поиск и пакетный режим через Excel.
    """
    from forecasting.runner import find_groups_by_article

    st.markdown("### Поиск запчасти")

    mode = st.radio(
        "Режим",
        ["Одиночный", "Списком"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if mode == "Одиночный":
        col_inp, col_btn = st.columns([3, 1])
        with col_inp:
            article = st.text_input(
                "Артикул",
                placeholder="Введите артикул или его часть...",
                label_visibility="collapsed",
                key="article_input",
            )
        with col_btn:
            st.button("Найти", width="stretch")

        if not article.strip():
            return None

        articles = [a for a in re.split(r"[,\s]+", article.strip()) if a]

    else:
        uploaded = st.file_uploader(
            "Загрузите Excel со списком артикулов",
            type=["xlsx", "xls"],
            key="articles_file",
        )
        if uploaded is None:
            st.caption("Файл должен содержать колонку 'Артикул'")
            return None

        try:
            arts_df = pd.read_excel(uploaded, dtype=str, engine="calamine")
            if "Артикул" not in arts_df.columns:
                st.error("В файле нет колонки 'Артикул'")
                return None
            articles = arts_df["Артикул"].dropna().str.strip().tolist()
            st.caption(f"Загружено артикулов: {len(articles)}")
        except Exception as e:
            st.error(f"Ошибка чтения файла: {e}")
            return None

    # Поиск групп
    group_ids = []
    not_found = []

    for i, art in enumerate(articles):
        hits = find_groups_by_article(df, art)
        if not hits:
            not_found.append(art)
        elif len(hits) == 1:
            group_ids.append(hits[0]["Номер группы"])
        else:
            options = {
                f"{h['Артикул']} — {str(h['Номенклатура'])[:50]}": h["Номер группы"]
                for h in hits
            }
            selected = st.selectbox(
                f"Найдено несколько для '{art}':",
                options=list(options.keys()),
                key=f"select_{i}_{art}",
            )
            group_ids.append(options[selected])

    if not_found:
        st.warning(f"Не найдено: {', '.join(not_found)}")

    return group_ids if group_ids else None


def render_params() -> tuple[int, float, float, bool]:
    """
    Блок параметров прогноза.
    """
    st.markdown("### Параметры прогноза")

    col1, col2, col3, col4 = st.columns(4)

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
    """
    Карточки с суммарными прогнозными значениями.
    """
    sale_total = round(float(result.sale.forecast.sum()), 1)
    repair_total = round(float(result.repair.forecast.sum()), 1)
    total = round(sale_total + repair_total, 1)
    n_months = len(result.fc_months)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Номер группы", result.group_id)
    c2.metric("Артикул", result.article[:20])
    c3.metric(f"Продажи ({n_months} мес.)", f"{sale_total:,.1f}")
    c4.metric(f"Ремонт ({n_months} мес.)", f"{repair_total:,.1f}")
    c5.metric("Итого спрос", f"{total:,.1f}")


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
    """Сводная таблица прогнозов по всем запчастям."""
    from forecasting.runner import MONTH_RU

    fc_months = results[0].fc_months
    rows = []
    for result in results:
        row = {
            "Артикул": result.article,
            "Номенклатура": result.nomenclature[:50],
            "Метод продаж": result.sale.method,
            "Метод ремонта": result.repair.method,
        }
        for y, m in fc_months:
            i = fc_months.index((y, m))
            lbl = f"{MONTH_RU[m]} {y}"
            row[f"{lbl} Продажи"] = round(float(result.sale.forecast.iloc[i]), 1)
            row[f"{lbl} Ремонт"] = round(float(result.repair.forecast.iloc[i]), 1)
        row["Итого продажи"] = round(float(result.sale.forecast.sum()), 1)
        row["Итого ремонт"] = round(float(result.repair.forecast.sum()), 1)
        row["Итого спрос"] = round(float(result.sale.forecast.sum() + result.repair.forecast.sum()), 1)
        rows.append(row)

    st.dataframe(
        pd.DataFrame(rows),
        width="stretch",
        hide_index=True,
    )