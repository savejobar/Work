import warnings
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


CROSTON_THRESHOLD = 0.40
TSB_DEFAULT_ALPHA = 0.1
TSB_DEFAULT_BETA = 0.1
MIN_SEASONAL_OBS = 24
MIN_HOLT_OBS = 4


@dataclass
class ForecastResult:
    """
    Результат прогноза одной серии.
    """
    forecast: pd.Series # прогнозные значения с DatetimeIndex
    method: str # название метода
    aic: float | None = None # AIC лучшей модели (если применимо)
    outliers: list = field(default_factory=list)
    series_raw: pd.Series | None = None
    series_clean: pd.Series | None = None


def remove_outliers_local(
    series: pd.Series,
    iqr_factor: float = 1.5,
) -> tuple[pd.Series, list[dict]]:
    """
    Заменяет выбросы (значения выше Q3 + iqr_factor*IQR) медианой соседей.
    """
    s = series.copy()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    upper = q3 + iqr_factor * iqr
    mask = s > upper
    log: list[dict] = []

    for idx in s[mask].index:
        orig = s[idx]
        pos = s.index.get_loc(idx)
        nbrs = [
            s.iloc[pos + off]
            for off in [-3, -2, -1, 1, 2, 3]
            if 0 <= pos + off < len(s) and not mask.iloc[pos + off]
        ]
        repl = float(np.median(nbrs)) if nbrs else float(s.median())
        s.iloc[pos] = repl
        log.append({
            "date": idx.strftime("%Y-%m"),
            "original": round(float(orig), 1),
            "replacement": round(repl, 1),
            "threshold": round(float(upper), 1),
        })

    return s, log


def _tsb_forecast(
    series: pd.Series,
    steps: int,
    fc_index: pd.DatetimeIndex,
) -> tuple[pd.Series, str]:
    """
    TSB (Teunter-Syntetos-Babai) для прерывистого спроса.

    В отличие от Croston, TSB обновляет вероятность спроса на каждом периоде,
    включая нулевые месяцы. Это позволяет прогнозу снижаться для "затухающих"
    позиций с длинным хвостом нулей.

    Параметры alpha и beta подбираются по сетке, минимизируя in-sample MSE
    одношаговых прогнозов.
    """
    vals = np.asarray(series, dtype=float)
    nz = np.flatnonzero(vals > 0)

    if len(nz) == 0:
        return pd.Series([0.0] * steps, index=fc_index), "TSB"

    start = int(nz[0])

    # После первого ненулевого значения _get_series обычно уже обрезает ряд,
    # но здесь дополнительно защищаемся от внешних вызовов с ведущими нулями.
    vals = vals[start:]

    if len(vals) == 1:
        fc_val = max(0.0, round(float(vals[0]), 2))
        return (
            pd.Series([fc_val] * steps, index=fc_index),
            f"TSB(α={TSB_DEFAULT_ALPHA:.2f}, β={TSB_DEFAULT_BETA:.2f})",
        )

    def fit_tsb(alpha: float, beta: float) -> tuple[float, float, float]:
        """
        Возвращает сглаженный размер спроса z, вероятность спроса p и MSE.
        """
        z = float(vals[0])
        p = 1.0
        errors: list[float] = []

        for y in vals[1:]:
            forecast = p * z
            errors.append((float(y) - forecast) ** 2)

            has_demand = 1.0 if y > 0 else 0.0
            if has_demand:
                z = alpha * float(y) + (1 - alpha) * z
            p = beta * has_demand + (1 - beta) * p

        mse = float(np.mean(errors)) if errors else 0.0
        return z, p, mse

    best_mse = np.inf
    best_alpha = TSB_DEFAULT_ALPHA
    best_beta = TSB_DEFAULT_BETA
    best_z = float(vals[0])
    best_p = 1.0

    for alpha in np.arange(0.05, 0.55, 0.05):
        for beta in np.arange(0.05, 0.55, 0.05):
            z, p, mse = fit_tsb(float(alpha), float(beta))
            if mse < best_mse:
                best_mse = mse
                best_alpha = float(alpha)
                best_beta = float(beta)
                best_z = z
                best_p = p

    fc_val = max(0.0, round(best_p * best_z, 2))
    method = f"TSB(α={best_alpha:.2f}, β={best_beta:.2f})"
    return pd.Series([fc_val] * steps, index=fc_index), method


def _ets_forecast(
    series: pd.Series,
    steps: int,
    fc_index: pd.DatetimeIndex,
    n_obs: int,
) -> tuple[pd.Series, str, float | None]:
    """
    Подбирает лучшую ETS/Holt-Winters модель по AIC.

    Кандидаты зависят от длины серии:
        >= 24 мес → включает сезонные модели (Holt-Winters add/mul)
        >= 4 мес  → Holt и простой ETS
        < 4 мес   → Mean-6m
    """
    from statsmodels.tsa.holtwinters import ExponentialSmoothing as ETS

    has_zeros = (series == 0).any()

    # Формируем список кандидатов в зависимости от длины серии
    candidates: list[tuple] = []

    if n_obs >= MIN_SEASONAL_OBS:
        # Holt-Winters: аддитивный тренд + аддитивная сезонность
        candidates.append(("add", "add", 12, "HW-add"))
        # Holt-Winters: аддитивный тренд + мультипликативная сезонность
        if not has_zeros:
            candidates.append(("add", "mul", 12, "HW-mul"))

    if n_obs >= MIN_HOLT_OBS:
        candidates.append(("add", None, None, "Holt"))
        candidates.append((None, None, None, "ETS"))

    best_fc: pd.Series | None = None
    best_aic: float = np.inf
    best_name: str = "Mean-6m"

    for trend, seasonal, sp, name in candidates:
        if sp and n_obs < 2 * sp:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ETS(series, trend=trend, seasonal=seasonal, seasonal_periods=sp)
                fitted = model.fit(optimized=True)
                if fitted.aic < best_aic:
                    best_aic = fitted.aic
                    best_fc = fitted.forecast(steps).clip(lower=0).round(2)
                    best_name = name
        except Exception as e:
            logging.warning(f"{name} failed: {e}")

    if best_fc is None:
        # Запасной вариант — среднее последних 6 месяцев
        val = max(0.0, round(float(series.tail(6).mean()), 2))
        best_fc = pd.Series([val] * steps, index=fc_index)
        best_aic = None
        best_name = "Mean-6m"
    else:
        best_fc.index = fc_index

    return best_fc, best_name, best_aic if best_aic != np.inf else None


def forecast_series(
    series: pd.Series,
    steps: int,
    fc_start_date: pd.Timestamp,
    iqr_factor: float | None = 1.5,
    croston_threshold: float = CROSTON_THRESHOLD,
) -> ForecastResult:
    """
    Прогнозирует временной ряд, автоматически выбирая лучший метод.

    Алгоритм выбора:
        1. Очищает выбросы через IQR (iqr_factor регулирует чувствительность).
           Если iqr_factor=None, очистка выбросов отключается.
        2. Если доля нулей > croston_threshold → TSB для прерывистого спроса.
        3. Иначе → лучшая ETS/Holt-Winters по AIC.
    """
    s_raw = series.clip(lower=0)
    n_obs = len(s_raw)

    # Очистка выбросов
    if iqr_factor is None:
        s_clean = s_raw.copy()
        outliers = []
    else:
        s_clean, outliers = remove_outliers_local(s_raw, iqr_factor)

    fc_index = pd.date_range(fc_start_date, periods=steps, freq="MS")
    zero_ratio = (s_clean == 0).sum() / max(n_obs, 1)

    if zero_ratio > croston_threshold:
        forecast, method = _tsb_forecast(s_clean, steps, fc_index)
        aic = None
    else:
        forecast, method, aic = _ets_forecast(s_clean, steps, fc_index, n_obs)

    return ForecastResult(
        forecast=forecast,
        method=method,
        aic=aic,
        outliers=outliers,
        series_raw=s_raw,
        series_clean=s_clean,
    )
