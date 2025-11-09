"""
Serialization helpers for the compose pipeline.

These utilities provide stable JSON fallbacks so refactored modules can safely
emit diagnostics without depending on the legacy monolith.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _json_default(obj: Any) -> Any:
    """
    JSON serializer fallback that mirrors the behaviour of the legacy script.

    * objects exposing ``to_dict`` are converted via that method
    * sets are converted to sorted lists for determinism
    * ``pathlib.Path`` instances are stringified
    """
    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        try:
            return to_dict()
        except Exception:
            pass
    if isinstance(obj, set):
        return sorted(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


__all__ = ["_json_default"]
