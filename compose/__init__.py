"""
Core package for the CN compose pipeline refactor.

Submodules are gradually extracted from the legacy monolithic script.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

__all__ = [
    "attacks",
    "workers",
    "leakage",
    "serialize",
    "utils",
    "state",
    "quota",
    "rng",
    "mismatch",
    "constants",
    "audit",
    "carriers",
    "effects",
    "capabilities",
    "cli",
    "dsl_runtime",
    "payload",
    "sources",
    "knobs",
    "dedupe_helpers",
    "surface_noise",
    "symmetry",
    "conversation",
    "effects_eval",
    "adv_mutate",
    "balance",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)


if TYPE_CHECKING:  # pragma: no cover - for static analyzers/mypy
    from . import (
        attacks,
        workers,
        leakage,
        serialize,
        utils,
        audit,
        carriers,
        capabilities,
        cli,
        constants,
        dsl_runtime,
        effects,
        knobs,
        mismatch,
        payload,
        quota,
        rng,
        sources,
        state,
        symmetry,
    )
