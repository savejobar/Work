"""
app/charts.py
==============
Отрисовка графиков в Streamlit.
Тонкая обёртка над forecasting.runner.plot_forecast.
"""

from __future__ import annotations

import streamlit as st

from forecasting.runner import GroupForecastResult, plot_forecast


def render_chart(result: GroupForecastResult, show_clean: bool = True) -> None:
    """Отрисовывает двухпанельный график прогноза в Streamlit."""
    fig = plot_forecast(result, show_clean=show_clean, figsize=(14, 5))
    st.pyplot(fig, use_container_width=True)
    # Освобождаем память
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.close(fig)
