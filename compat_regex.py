"""Compatibility shim for the optional third-party :mod:`regex` package.

The composer prefers the enhanced ``regex`` module when it is installed so
callers can rely on features like full Unicode properties and fuzzy matching.
When the dependency is unavailable we gracefully fall back to the standard
library :mod:`re` module which provides the subset of functionality used by the
pipeline.  Downstream code imports ``compat_regex.regex`` and therefore does
not have to repeat the try/except logic.
"""
from __future__ import annotations

import importlib
import types

__all__ = ["regex"]

try:  # pragma: no cover - trivial import wrapper
    regex = importlib.import_module("regex")
except ModuleNotFoundError:  # pragma: no cover - fallback path
    regex = importlib.import_module("re")

# Expose commonly expected attributes even when falling back to :mod:`re`.
for attr in ("Regex", "Pattern", "Match"):
    if hasattr(regex, attr):
        globals()[attr] = getattr(regex, attr)

# Ensure ``from compat_regex import regex as _re`` style imports behave like the
# underlying module.
if isinstance(regex, types.ModuleType):
    __spec__ = regex.__spec__  # type: ignore[attr-defined]
