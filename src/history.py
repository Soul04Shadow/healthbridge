"""Utilities for managing persistent chat history."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from langchain_community.chat_message_histories import SQLChatMessageHistory

_DB_PATH = Path(os.environ.get("CHAT_HISTORY_DB_PATH", "data/chat_history.sqlite")).expanduser().resolve()
_CONNECTION_STRING = f"sqlite:///{_DB_PATH}"


def _ensure_storage() -> None:
    """Ensure that the directory for the history database exists."""
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_message_history(session_id: str) -> SQLChatMessageHistory:
    """Return the message history for a specific session."""
    _ensure_storage()
    return SQLChatMessageHistory(session_id=session_id, connection=_CONNECTION_STRING)


def message_history_factory(config: Mapping[str, Any] | None) -> SQLChatMessageHistory:
    """Factory compatible with RunnableWithMessageHistory.

    Args:
        config: Runnable configuration containing a ``session_id`` in the
            ``configurable`` namespace.

    Returns:
        An instance of :class:`SQLChatMessageHistory` bound to the provided session.
    """
    session_id = None
    if config is not None:
        if isinstance(config, Mapping):
            configurable = config.get("configurable")
        else:
            configurable = getattr(config, "configurable", None)
        if isinstance(configurable, Mapping):
            session_id = configurable.get("session_id")
        else:
            session_id = getattr(configurable, "session_id", None)
    if not session_id:
        raise ValueError("A session_id must be provided in the runnable configuration.")
    return load_message_history(str(session_id))
