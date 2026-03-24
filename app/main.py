import pandas as pd
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
from forecasting.runner import run_group_forecast, _get_train_end
from readers.exporters import build_batch_excel
from app.logger import SessionLogger


def apply_last_month_policy(df: pd.DataFrame, include_last_month: bool) -> pd.DataFrame:
    """
    Возвращает рабочий DataFrame для прогноза.

    Если include_last_month=False, исключает из обучения текущий календарный
    месяц целиком. Это позволяет использовать полный прошлый месяц как конец
    истории и строить прогноз, начиная с текущего месяца.
    """
    if include_last_month or df.empty:
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
    iqr_factor: float,
    croston_threshold: float,
    include_last_month: bool,
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
        + str(include_last_month)
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

with st.expander("FAQ — как работает прогноз?"):
    st.markdown("""
**Какие методы прогнозирования используются?**

Для каждой группы запчастей система строит **два отдельных прогноза**:
- по **продажам**
- по **ремонту**

Модель для каждого ряда выбирается **автоматически**, исходя из структуры истории: длины ряда, доли нулей, наличия тренда и сезонности.

---

**Как система выбирает модель?**

Выбор идёт по принципу “от более специализированной модели к более простой”:

1. Если ряд **сильно прерывистый** и в нём высокая доля нулевых месяцев, используется **TSB**.
2. Если данных достаточно много и видна возможная сезонность, проверяется **Holt-Winters**.
3. Если сезонность не нужна, но есть тренд, используется **Holt**.
4. Если ряд короткий и без выраженной структуры, используется **ETS**.
5. Если статистические модели не дают устойчивого результата, берётся резервный вариант — **Среднее за 6 последних месяцев**.

Итоговая модель выбирается **отдельно для продаж и ремонта**, потому что их поведение часто различается: например, продажи могут быть редкими и “рваными”, а ремонт — более стабильным.

---

**Когда выбирается TSB?**

**TSB (Teunter-Syntetos-Babai)** применяется для **нерегулярного спроса**, когда в ряде много месяцев без движения.

Если доля нулевых месяцев превышает заданный порог, ряд считается прерывистым, и система использует именно TSB.

Формально проверяется:

`zero_share = число месяцев, где спрос = 0 / общее число месяцев`

Если:

`zero_share >= порог для TSB`

то выбирается **TSB**.

Например:
- 12 месяцев истории
- продажи были только в 5 месяцах
- значит 7 месяцев из 12 — нулевые
- `7 / 12 = 58%`

Если порог TSB равен `40%`, такой ряд пойдёт в TSB.

TSB моделирует спрос как произведение двух компонент:
- вероятности того, что спрос вообще возникнет
- среднего размера спроса, если он возник

Упрощённо:

`Прогноз = p_t * z_t`

где:
- `p_t` — сглаженная вероятность ненулевого спроса
- `z_t` — сглаженный размер спроса

Обе величины обновляются экспоненциально через параметры сглаживания `α` и `β`.

---

**Когда выбирается Holt-Winters?**

**Holt-Winters** используется, если:
- ряд достаточно длинный
- есть смысл искать **сезонность**
- в данных не слишком много нулей

Обычно он рассматривается при истории **от 24 месяцев**, потому что сезонную модель трудно оценивать на коротком ряде.

Модель раскладывает ряд на компоненты:
- уровень
- тренд
- сезонность

В зависимости от структуры ряда могут использоваться:
- **аддитивная** сезонность: `y_t = level + trend + season + error`
- **мультипликативная** сезонность: `y_t = level * trend * season * error`

Если проверяются несколько вариантов Holt-Winters, лучший выбирается по **AIC**:
- чем ниже AIC (Информационный критерий Акаике), тем лучше баланс между качеством подгонки и сложностью модели

Упрощённо:

`AIC = 2k - 2ln(L)`

где:
- `k` — число параметров модели
- `L` — значение функции правдоподобия

То есть система не просто берёт “самую сложную” модель, а ищет наиболее оправданную статистически.

---

**Когда выбирается Holt?**

**Holt** применяется, если:
- данных уже достаточно для оценки тренда
- сезонность либо не просматривается, либо не нужна
- ряд не настолько прерывистый, чтобы использовать TSB

Обычно эта модель уместна, когда история имеет хотя бы несколько наблюдений подряд и в ней есть плавное изменение уровня вверх или вниз.

Holt использует две сглаживаемые компоненты:
- уровень `l_t`
- тренд `b_t`

Прогноз на `h` шагов вперёд:

`ŷ_(t+h) = l_t + h * b_t`

Это хороший компромисс для рядов без ярко выраженной сезонности, но с направленным движением.

---

**Когда выбирается ETS?**

**ETS** — это более простой вариант экспоненциального сглаживания.

Он используется, если:
- ряд короткий
- сезонности нет
- тренд выражен слабо
- более сложные модели не дали устойчивого результата

По сути ETS подходит для рядов, где нужно аккуратно сгладить шум и получить базовый краткосрочный прогноз без переусложнения.

В простейшем случае прогноз строится через сглаженный уровень:

`l_t = α y_t + (1 - α) l_(t-1)`

---

**Когда используется Mean-6m?**

**Mean-6m** — это резервный, максимально устойчивый вариант.

Если статистические модели:
- не смогли обучиться
- дали нестабильный результат
- или история слишком слабая для надёжной оценки

система берёт среднее за последние 6 месяцев:

`Прогноз = (y_(t-5) + y_(t-4) + ... + y_t) / 6`

Этот метод не пытается “угадать структуру”, а даёт понятную базовую оценку на основе последней истории.

---

**Как подбираются параметры моделей?**

Параметры подбираются **автоматически** внутри используемой модели.

Это означает:
- для TSB подбираются коэффициенты сглаживания вероятности и размера спроса
- для Holt / ETS подбираются коэффициенты сглаживания уровня и, при необходимости, тренда
- для Holt-Winters дополнительно подбираются параметры сезонности и тип сезонной модели

Пользователь вручную не задаёт эти коэффициенты: система выбирает их так, чтобы модель лучше описывала исторический ряд.

---

**Как обрабатываются выбросы?**

Перед построением прогноза ряд можно очистить от аномально больших значений.

Выброс определяется по правилу межквартильного размаха:

`Верхняя граница = Q3 + k * IQR`

где:
- `IQR = Q3 - Q1`
- `Q1` — первый квартиль
- `Q3` — третий квартиль
- `k` — выбранный пользователем IQR-фактор

Если значение выше этой границы, оно считается выбросом и заменяется более устойчивой оценкой — медианой соседних наблюдений.

Чем меньше `IQR`, тем чувствительнее система к всплескам.
Если включён режим **«Не удалять выбросы»**, очистка не применяется.

---

Формула итогового спроса:

`Total demand = Sales forecast + Repair forecast`

Такой подход точнее, чем прогнозировать один общий поток, потому что:
- продажи и ремонт часто имеют разную частоту
- могут подчиняться разным паттернам
- и могут требовать разных моделей

---

**Почему это важно для запчастей?**

Спрос на запчасти редко бывает “гладким”:
- одни позиции продаются регулярно
- другие нужны только под ремонт
- третьи лежат без движения месяцами, а потом резко всплескивают

Поэтому система не использует одну универсальную формулу для всех, а подбирает модель под характер конкретного ряда. Это делает прогноз устойчивее и ближе к реальной логике складского спроса.
""")
    
# Сайдбар: загрузка и пайплайн

df = render_sidebar()

if df is None:
    st.info("Загрузите данные в боковой панели, чтобы начать")
    st.stop()

# Параметры прогноза
steps, iqr_factor, croston_threshold, show_clean, include_last_month = render_params()

# Рабочий DataFrame для прогноза
df_work = apply_last_month_policy(df, include_last_month)

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
    + str(include_last_month)
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
    include_last_month=include_last_month,
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
