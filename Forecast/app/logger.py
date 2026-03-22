import logging
import uuid
from datetime import datetime, timezone
import threading

import streamlit as st


def _get_user() -> str:
    """
    Возвращает email пользователя или уникальный ID сессии.
    """
    try:
        if hasattr(st, "user") and st.user:
            if st.user.is_logged_in:
                email = getattr(st.user, "email", None)
                if email:
                    return email
    except AttributeError:
        pass
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())
    return st.session_state["session_id"]


@st.cache_resource
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


def _write_log_async(level: str, message: str, user: str) -> None:
    """
    Асинхронная запись лога — не блокирует UI.
    """
    threading.Thread(
        target=_write_log,
        args=(level, message, user),
        daemon=True,
    ).start()


def _write_log(level: str, message: str, user: str) -> None:
    """
    Записывает строку в Google Sheets.
    """
    try:
        sheet = _get_sheet()
        sheet.append_row([
            datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            user,
            level,
            message,
        ])
    except Exception as e:
        _get_console_logger().error(f"Logging failed: {e}")  # ← не теряем ошибки


class SessionLogger:
    """
    Логгер который пишет в консоль синхронно и в Google Sheets асинхронно.
    """

    def info(self, msg: str) -> None:
        user = _get_user()
        _get_console_logger().info(f"{user} | {msg}")
        _write_log_async("INFO", msg, user)

    def warning(self, msg: str) -> None:
        user = _get_user()
        _get_console_logger().warning(f"{user} | {msg}")
        _write_log_async("WARNING", msg, user)

    def error(self, msg: str) -> None:
        user = _get_user()
        _get_console_logger().error(f"{user} | {msg}")
        _write_log_async("ERROR", msg, user)