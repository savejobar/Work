import logging
import uuid
from datetime import datetime, timezone

import streamlit as st


def _get_user() -> str:
    """
    Возвращает email пользователя или уникальный ID сессии.
    """
    try:
        if hasattr(st, "user") and st.user and st.user.is_logged_in:
            user_id = (
                getattr(st.user, "email", None)
                or getattr(st.user, "id", None)
                or getattr(st.user, "username", None)
            )
            if user_id:
                return user_id
    except AttributeError:
        pass
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())
    return st.session_state["session_id"]


@st.cache_resource(ttl=3600)
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
    print(f"DEBUG _write_log called: {level} | {user} | {message}")  # ← добавь
    try:
        sheet = _get_sheet()
        print(f"DEBUG sheet obtained: {sheet}")  # ← добавь
        sheet.append_row([...])
        print("DEBUG append_row success")  # ← добавь
    except Exception as e:
        print(f"DEBUG exception: {e}")  # ← добавь
        _get_console_logger().error(f"Logging failed: {e}")


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