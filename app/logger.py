import logging
import uuid
from datetime import datetime, timezone

import streamlit as st


def _get_user() -> str:
    """
    Возвращает email пользователя или уникальный ID сессии.
    """
    try:
        if hasattr(st, "user") and st.user.is_logged_in:
            email = st.user.email
            if email:
                return email
    except AttributeError:
        pass
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())
    return st.session_state["session_id"]


@st.cache_resource(ttl=3600, show_spinner=False)
def _get_sheet():
    """
    Возвращает первый лист Google таблицы.
    """
    import gspread
    from google.oauth2.service_account import Credentials

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=scopes,
    )
    client = gspread.authorize(creds)
    return client.open_by_key(st.secrets["google_sheet_id"]).sheet1


def _get_console_logger() -> logging.Logger:
    """
    Возвращает логгер для вывода в консоль.
    """
    logger = logging.getLogger("forecast_app")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        logger.propagate = False
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(handler)
    return logger


def _write_log(level: str, message: str, user: str) -> None:
    """
    Записывает строку лога в Google Sheets.
    При временной ошибке один раз сбрасывает кэш sheet и повторяет запись.
    """
    row = [
        datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        user,
        level,
        message,
    ]

    try:
        sheet = _get_sheet()
        sheet.append_row(row)
    except Exception:
        try:
            _get_sheet.clear()
            sheet = _get_sheet()
            sheet.append_row(row)
        except Exception as second_error:
            _get_console_logger().error(
                f"Logging failed after retry: {second_error}"
            )
            

class SessionLogger:
    """
    Логгер который пишет в консоль и в Google Sheets.
    """

    def info(self, msg: str) -> None:
        user = _get_user()
        _get_console_logger().info(f"{user} | {msg}")
        _write_log("INFO", msg, user)

    def warning(self, msg: str) -> None:
        user = _get_user()
        _get_console_logger().warning(f"{user} | {msg}")
        _write_log("WARNING", msg, user)

    def error(self, msg: str) -> None:
        user = _get_user()
        _get_console_logger().error(f"{user} | {msg}")
        _write_log("ERROR", msg, user)