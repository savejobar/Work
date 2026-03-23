from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st

from forecasting.runner import GroupForecastResult, plot_forecast


def render_chart(result: GroupForecastResult, show_clean: bool = True) -> None:
    """
    Отрисовывает двухпанельный график прогноза в Streamlit.
    """
    fig = plot_forecast(result, show_clean=show_clean, figsize=(14, 5))
    st.pyplot(fig, width="stretch")
    plt.close(fig)