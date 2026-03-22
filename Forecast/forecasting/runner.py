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


def _get_train_end(df: pd.DataFrame) -> tuple[int, int]:
    """
    Возвращает год и месяц последней строки датафрейма.
    Используется как граница обучающей выборки для модели прогноза.
    """
    last = df.sort_values(["Год", "Месяц"]).iloc[-1]
    return int(last["Год"]), int(last["Месяц"])


def _get_series(
    df: pd.DataFrame,
    group_id: int,
    col: str,
    train_end_year: int,
    train_end_month: int,
) -> pd.Series:
    """
    Извлекает временной ряд для группы и колонки.
    Начало ряда — первый месяц с col > 0 для данной группы.
    """
    train_end_ym = train_end_year * 100 + train_end_month

    grp = df[df["Номер группы"] == group_id].copy()
    grp["_ym"] = grp["Год"] * 100 + grp["Месяц"]
    grp = grp[grp["_ym"] <= train_end_ym].sort_values("_ym")

    if grp.empty:
        return pd.Series(dtype=float)

    values = grp[col].fillna(0).astype(float)

    # Обрезаем до первого ненулевого значения этой конкретной колонки
    first_nonzero = (values > 0).idxmax()
    if values[first_nonzero] == 0:
        # Колонка полностью нулевая — возвращаем пустую серию
        return pd.Series(dtype=float)

    grp = grp.loc[first_nonzero:]
    values = values.loc[first_nonzero:]

    dates = pd.to_datetime(
        grp["Год"].astype(str) + "-" + grp["Месяц"].astype(str).str.zfill(2) + "-01"
    )
    return pd.Series(values.values, index=pd.DatetimeIndex(dates, freq="MS"))


def _draw_panel(
    ax: plt.Axes,
    fc_result: ForecastResult,
    fc_months: list[tuple[int, int]],
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
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(45)
        lbl.set_ha("right")


@dataclass
class GroupForecastResult:
    """Результат прогноза по группе аналогов (продажи + ремонт)."""
    group_id: int
    article: str
    nomenclature: str
    sale: ForecastResult
    repair: ForecastResult
    fc_months: list[tuple[int, int]]


def find_groups_by_article(df: pd.DataFrame, article: str) -> list[dict]:
    """Возвращает все группы где артикул содержит введённую строку."""
    article_up = article.strip().upper()
    mask = df["Артикул"].astype(str).str.upper().str.contains(article_up, na=False)
    hits = (
        df[mask]
        .drop_duplicates("Номер группы")[["Номер группы", "Артикул", "Номенклатура"]]
        .to_dict("records")
    )
    # Также ищем в аналогах если по артикулу ничего нет
    if not hits and "Список аналогов" in df.columns:
        for _, row in df.drop_duplicates("Номер группы").iterrows():
            analogs = row["Список аналогов"]
            if isinstance(analogs, tuple):
                if any(article_up in str(a).upper() for a in analogs):
                    hits.append({
                        "Номер группы": int(row["Номер группы"]),
                        "Артикул": str(row["Артикул"]),
                        "Номенклатура": str(row["Номенклатура"]),
                    })
    return hits


def run_group_forecast(
    df: pd.DataFrame,
    group_id: int,
    steps: int = 3,
    iqr_factor: float = 1.5,
    croston_threshold: float = 0.40,
) -> GroupForecastResult:
    """
    Строит прогноз продаж и ремонта для группы аналогов.
    """
    steps = max(1, min(12, steps))

    meta_row = df[df["Номер группы"] == group_id].iloc[0]
    nomenclature = str(meta_row["Номенклатура"])[:80]
    article_out = str(meta_row["Артикул"])

    train_end_year, train_end_month = _get_train_end(df)
    fc_months = _next_months(train_end_year, train_end_month, steps)
    fc_start = pd.Timestamp(year=fc_months[0][0], month=fc_months[0][1], day=1)

    sale_series = _get_series(df, group_id, "Продажа", train_end_year, train_end_month)
    repair_series = _get_series(df, group_id, "Ремонт", train_end_year, train_end_month)

    sale_result = forecast_series(sale_series, steps, fc_start, iqr_factor, croston_threshold)
    repair_result = forecast_series(repair_series, steps, fc_start, iqr_factor, croston_threshold)

    return GroupForecastResult(
        group_id=group_id,
        article=article_out,
        nomenclature=nomenclature,
        sale=sale_result,
        repair=repair_result,
        fc_months=fc_months,
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
    _draw_panel(axes[0], result.sale, result.fc_months, "Продажи", show_clean)
    _draw_panel(axes[1], result.repair, result.fc_months, "Ремонт", show_clean)
    fig.tight_layout()
    return fig