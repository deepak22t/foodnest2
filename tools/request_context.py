"""Per-request context for the API (e.g. current user) so tool calls see the same user as the HTTP handler."""

from contextvars import ContextVar

current_user_id: ContextVar[str] = ContextVar("current_user_id", default="default")


def get_current_user_id() -> str:
    return current_user_id.get()
