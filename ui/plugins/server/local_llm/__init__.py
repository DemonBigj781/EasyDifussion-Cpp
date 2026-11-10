"""Local instruction-driven prompt processor backed by llama.cpp."""

from .service import (
    ProcessRequest,
    get_settings,
    health,
    process_prompt,
    save_settings,
    shutdown,
)

__all__ = [
    "ProcessRequest",
    "get_settings",
    "health",
    "process_prompt",
    "save_settings",
    "shutdown",
]
