from __future__ import annotations
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

from forecasting.models import ForecastResult, forecast_series

MONTH_RU = {
    1: "Янв", 2: "Фев", 3: "Мар", 4: "Апр",
    5: "Май", 6: "Июн", 7: "Июл", 8: "Авг",
    9: "Сен", 10: "Окт", 11: "Ноя", 12: "Дек",
}


def _month_label(year: int, month: int) -> str:
    """
    Возвращает название месяца на русском с годом, например 'Янв 2024'.
    """
    return f"{MONTH_RU[month]} {year}"


def _next_months(year: int, month: int, n: int) -> list[tuple[int, int]]:
    """
    Возвращает список из n следующих месяцев после (year, month).
    """
    result, y, m = [], year, month
    for _ in range(n):
        m += 1
        if m > 12:
            m, y = 1, y + 1
        result.append((y, m))
    return result


def _month_start(year: int, month: int) -> pd.Timestamp:
    """
    Возвращает первый день месяца.
    """
    return pd.Timestamp(year=year, month=month, day=1)


def _months_between(start: pd.Timestamp, end: pd.Timestamp) -> int:
    """
    Возвращает число полных календарных месяцев между началом месяцев.
    """
    return (end.year - start.year) * 12 + (end.month - start.month)


def _get_train_end(df: pd.DataFrame) -> tuple[int, int]:
    """
    Возвращает год и месяц последней строки датафрейма.

    Граница обучения определяется уже подготовленным df_work:
    если тумблер исключения последнего месяца выключен, текущий месяц
    отфильтрован заранее, и прогноз начнётся с него; если тумблер включён,
    последняя строка участвует в обучении.
    """
    last = df.sort_values(["Год", "Месяц"]).iloc[-1]
    return int(last["Год"]), int(last["Месяц"])


def _build_monthly_group_frame(
    grp: pd.DataFrame,
    group_id: int,
    train_end_year: int,
    train_end_month: int,
) -> pd.DataFrame:
    """
    Достраивает для группы непрерывную месячную сетку до train_end.

    Потоковые колонки остаются пустыми в отсутствующих месяцах и
    интерпретируются позже в _get_series(). Конечный остаток протягивается
    вперёд, чтобы расчёт need_to_order брал последний известный остаток на
    общей календарной оси.
    """
    if grp.empty:
        return grp.copy()

    train_end = pd.Timestamp(year=train_end_year, month=train_end_month, day=1)

    monthly = grp.copy()
    monthly["_date"] = pd.to_datetime(
        monthly["Год"].astype(str)
        + "-"
        + monthly["Месяц"].astype(str).str.zfill(2)
        + "-01"
    )
    monthly = monthly[monthly["_date"] <= train_end].sort_values("_date")

    if monthly.empty:
        return monthly

    full_index = pd.date_range(start=monthly["_date"].min(), end=train_end, freq="MS")
    monthly = monthly.set_index("_date").reindex(full_index)

    monthly["Год"] = monthly.index.year
    monthly["Месяц"] = monthly.index.month
    monthly["Номер группы"] = group_id

    if "Конечный остаток" in monthly.columns:
        monthly["Конечный остаток"] = (
            pd.to_numeric(monthly["Конечный остаток"], errors="coerce").ffill()
        )

    for col in ["Артикул", "Номенклатура"]:
        if col in monthly.columns:
            monthly[col] = monthly[col].ffill().bfill()

    return monthly


def _get_series(
    grp: pd.DataFrame,
    col: str,
) -> pd.Series:
    """
    Извлекает непрерывный месячный ряд для одной колонки группы.

    После первого ненулевого значения пропущенные месяцы трактуются как нули.
    """
    if grp.empty or col not in grp.columns:
        return pd.Series(dtype=float)

    values = pd.to_numeric(grp[col], errors="coerce")

    nonzero = values[values.fillna(0) > 0]
    if nonzero.empty:
        return pd.Series(dtype=float)

    start = nonzero.index[0]
    values = values.loc[start:].fillna(0).astype(float)
    values.index = pd.DatetimeIndex(values.index, freq="MS")
    return values


def _draw_panel(
    ax: plt.Axes,
    fc_result: ForecastResult,
    title: str,
    show_clean: bool,
) -> None:
    """
    Рисует одну панель графика: исторический ряд, выбросы и прогноз.
    """
    raw = fc_result.series_raw
    clean = fc_result.series_clean
    fc = fc_result.forecast

    if raw is None or raw.empty:
        ax.set_title(f"{title} — нет данных")
        return

    ax.plot(raw.index, raw.values, color="#3b82f6", lw=2, marker="o",
            markersize=3, label="История", zorder=3)

    if show_clean and fc_result.outliers:
        ax.plot(clean.index, clean.values, color="#8b5cf6", lw=1.5,
                linestyle="--", label="После очистки", zorder=3)

    for i, o in enumerate(fc_result.outliers):
        ts = pd.Timestamp(o["date"] + "-01")
        ax.scatter([ts], [o["original"]], color="#ef4444", s=80,
                   zorder=5, marker="x",
                   label="Выброс" if i == 0 else None)

    last_date = raw.index[-1]
    fc_x = [last_date] + list(fc.index)
    last_val = (
        float(fc_result.series_clean.values[-1])
        if fc_result.series_clean is not None
        else float(raw.values[-1])
    )
    fc_y = [last_val] + list(fc.values)
    ax.plot(fc_x, fc_y, color="#f97316", lw=2.5, linestyle="--",
            marker="D", markersize=6, zorder=4,
            label=f"Прогноз ({fc_result.method})")

    for x, y in zip(fc.index, fc.values):
        ax.annotate(f"{y:,.1f}", xy=(x, y), xytext=(0, 8),
                    textcoords="offset points", ha="center",
                    fontsize=8, color="#c2410c")

    ax.axvline(last_date, color="#94a3b8", lw=1, linestyle=":")
    zero_pct = int(round((raw == 0).sum() / max(len(raw), 1) * 100))
    ax.set_title(
        f"{title}  |  {fc_result.method}\n"
        f"Итого: {fc.sum():,.1f}  |  Нулей: {zero_pct}%",
        fontsize=10,
    )
    ax.margins(y=0.15)
    ax.set_xlabel("Период")
    ax.set_ylabel("Количество")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.1f}"))
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(45)
        lbl.set_ha("right")


@dataclass
class GroupForecastResult:
    """
    Результат прогноза по группе аналогов (продажи + ремонт).
    """
    group_id: int
    article: str
    nomenclature: str
    analogs: str
    sale: ForecastResult
    repair: ForecastResult
    fc_months: list[tuple[int, int]]
    ending_stock: float = 0.0


def build_result_summary(result: GroupForecastResult) -> dict[str, float]:
    """
    Возвращает единые итоговые метрики для UI и экспорта.
    """
    sale_total = round(float(result.sale.forecast.sum()), 1)
    repair_total = round(float(result.repair.forecast.sum()), 1)
    total_demand = round(sale_total + repair_total, 1)
    need_to_order = max(0.0, round(total_demand - result.ending_stock, 1))

    return {
        "sale_total": sale_total,
        "repair_total": repair_total,
        "total_demand": total_demand,
        "need_to_order": need_to_order,
    }


def find_groups_by_article(df: pd.DataFrame, article: str) -> list[dict]:
    article_up = str(article).strip().upper()
    if not article_up:
        return []

    hits = []

    group_rows = df.drop_duplicates("Номер группы")[
        ["Номер группы", "Артикул", "Номенклатура", "Список аналогов"]
    ]

    for _, row in group_rows.iterrows():
        group_id = int(row["Номер группы"])
        base_article = str(row["Артикул"]).strip()
        nomenclature = str(row["Номенклатура"])

        matched = None

        if article_up in base_article.upper():
            matched = base_article
        else:
            analogs = row["Список аналогов"]
            if isinstance(analogs, tuple):
                for analog in analogs:
                    analog_str = str(analog).strip()
                    if article_up in analog_str.upper():
                        matched = analog_str
                        break

        if matched is not None:
            hits.append({
                "Номер группы": group_id,
                "Артикул": base_article,
                "Номенклатура": nomenclature,
                "Совпадение": matched,
            })

    hits.sort(
        key=lambda h: (
            0 if str(h["Совпадение"]).upper() == article_up else 1,
            0 if str(h["Совпадение"]).upper().startswith(article_up) else 1,
            str(h["Совпадение"]).upper(),
            h["Номер группы"],
        )
    )
    return hits


def run_group_forecast(
    grp: pd.DataFrame,
    group_id: int,
    train_end_year: int,
    train_end_month: int,
    forecast_start: pd.Timestamp,
    steps: int = 3,
    iqr_factor: float | None = 1.5,
    croston_threshold: float = 0.40,
) -> GroupForecastResult:
    """
    Строит прогноз продаж и ремонта для группы аналогов.
    """
    steps = max(1, min(12, steps))

    if grp.empty:
        raise ValueError(f"Группа {group_id} не найдена")

    meta_row = grp.iloc[0]
    nomenclature = str(meta_row["Номенклатура"])[:80]
    article_out = str(meta_row["Артикул"])
    raw_analogs = meta_row.get("Список аналогов")

    if isinstance(raw_analogs, tuple):
        analogs_out = ", ".join(str(x) for x in raw_analogs if pd.notna(x))
    elif pd.notna(raw_analogs):
        analogs_out = str(raw_analogs)
    else:
        analogs_out = ""

    grp_monthly = _build_monthly_group_frame(grp, group_id, train_end_year, train_end_month)

    natural_start = _month_start(*_next_months(train_end_year, train_end_month, 1)[0])
    forecast_start = max(forecast_start.normalize(), natural_start)
    lead_months = _months_between(natural_start, forecast_start)
    total_steps = steps + lead_months

    fc_months_full = _next_months(train_end_year, train_end_month, total_steps)
    fc_start_full = _month_start(*fc_months_full[0])

    sale_series = _get_series(grp_monthly, "Продажа")
    repair_series = _get_series(grp_monthly, "Ремонт")

    sale_result = forecast_series(
        sale_series,
        total_steps,
        fc_start_full,
        iqr_factor,
        croston_threshold,
    )
    repair_result = forecast_series(
        repair_series,
        total_steps,
        fc_start_full,
        iqr_factor,
        croston_threshold,
    )

    sale_result.forecast = sale_result.forecast.iloc[lead_months:lead_months + steps]
    repair_result.forecast = repair_result.forecast.iloc[lead_months:lead_months + steps]
    fc_months = fc_months_full[lead_months:lead_months + steps]

    ending_stock = 0.0
    if "Конечный остаток" in grp_monthly.columns:
        vals = grp_monthly["Конечный остаток"].dropna()
        if not vals.empty:
            ending_stock = float(vals.iloc[-1])

    return GroupForecastResult(
        group_id=group_id,
        article=article_out,
        nomenclature=nomenclature,
        analogs=analogs_out,
        sale=sale_result,
        repair=repair_result,
        fc_months=fc_months,
        ending_stock=ending_stock,
    )


def forecast_table(result: GroupForecastResult) -> pd.DataFrame:
    """
    Строит таблицу прогноза по месяцам с итоговой строкой.
    """
    rows = []
    for i, (y, m) in enumerate(result.fc_months):
        sv = round(float(result.sale.forecast.iloc[i]), 1)
        rv = round(float(result.repair.forecast.iloc[i]), 1)
        rows.append({
            "Период": _month_label(y, m),
            "Прогноз продаж": sv,
            "Прогноз ремонта": rv,
            "Итого спрос": round(sv + rv, 1),
        })
    rows.append({
        "Период": "ИТОГО",
        "Прогноз продаж": round(float(result.sale.forecast.sum()), 1),
        "Прогноз ремонта": round(float(result.repair.forecast.sum()), 1),
        "Итого спрос": round(
            float(result.sale.forecast.sum() + result.repair.forecast.sum()), 1
        ),
    })
    return pd.DataFrame(rows)


def plot_forecast(
    result: GroupForecastResult,
    show_clean: bool = True,
    figsize: tuple[int, int] = (14, 8),
) -> plt.Figure:
    """
    Строит двухпанельный график прогноза: продажи и ремонт.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(
        f"{result.nomenclature}\nАртикул: {result.article}  |  Группа: {result.group_id}",
        fontsize=12, fontweight="bold",
    )
    _draw_panel(axes[0], result.sale, "Продажи", show_clean)
    _draw_panel(axes[1], result.repair, "Ремонт", show_clean)
    fig.tight_layout()
    return fig
