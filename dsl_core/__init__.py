"""Facade module for the refactored DSL core package.

This keeps the legacy surface area intact while functionality is migrated
into dedicated submodules that expose the previous dsl_core.py entry points.
"""
from __future__ import annotations

import base64
import codecs
import json
import logging
import math
import os
import random
import re
import string
import urllib  # Legacy surface expects urllib namespace exposed
import urllib.parse  # noqa: F401 -- ensure urllib.parse registered alongside base package
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Iterable, Optional, Protocol, Tuple
from types import ModuleType, SimpleNamespace

import numpy as np

from stable_random import stable_rng

from . import spec as _spec
from . import utils as _utils
from . import anchors as _anchors
from . import textops as _textops
from . import sandbox as _sandbox
from . import invariants as _invariants
from . import soft as _soft
from . import renderers as _renderers
from . import generator as _generator


_logger = logging.getLogger(__name__)


def _public_names(mod) -> Iterable[str]:
    names = getattr(mod, "__all__", None)
    if names is None:
        filtered: list[str] = []
        for name in dir(mod):
            if name.startswith("_"):
                continue
            value = getattr(mod, name)
            if isinstance(value, ModuleType):
                continue
            filtered.append(name)
        names = filtered
    return names


def _merge_exports(sources: Iterable[Tuple[str, object]]):
    namespace: dict[str, Tuple[object, str, int]] = {}
    conflicts: dict[str, dict[str, object]] = {}
    order_map = {label: idx for idx, (label, _) in enumerate(sources)}
    priority_chain = " -> ".join(label for label, _ in sources)
    for label, module in sources:
        order = order_map[label]
        for name in _public_names(module):
            value = getattr(module, name)
            if name in namespace:
                kept_label = namespace[name][1]
                entry = conflicts.setdefault(
                    name,
                    {
                        "kept": kept_label,
                        "kept_order": order_map[kept_label],
                        "skipped": [],
                    },
                )
                entry["skipped"].append({"label": label, "order": order})
                continue
            namespace[name] = (value, label, order)
    export_conflicts = {
        symbol: {
            "kept": meta["kept"],
            "kept_order": meta["kept_order"],
            "skipped": sorted(meta["skipped"], key=lambda x: x["order"]),
        }
        for symbol, meta in conflicts.items()
    }
    if export_conflicts:
        warn_env = bool(os.getenv("DSL_CORE_WARN_CONFLICTS"))
        log_level = logging.WARNING if warn_env else logging.INFO
        _logger.log(
            log_level,
            "dsl_core export conflicts detected (priority %s; kept earlier modules)",
            priority_chain,
            extra={
                "export_conflicts": export_conflicts,
                "export_priority": [label for label, _ in sources],
            },
        )
        if warn_env:
            msgs = [
                (
                    f"{symbol}: kept {meta['kept']!r} (order {meta['kept_order']}) "
                    f"over {[item['label'] for item in meta['skipped']]}"
                )
                for symbol, meta in export_conflicts.items()
            ]
            warnings.warn(
                "dsl_core duplicate exports detected:\n  " + "\n  ".join(msgs),
                RuntimeWarning,
                stacklevel=2,
            )
    # strip provenance info
    return {name: value for name, (value, _, _) in namespace.items()}, export_conflicts


_legacy_shim_exports = {
    "Any": Any,
    "Callable": Callable,
    "Iterable": Iterable,
    "Optional": Optional,
    "Protocol": Protocol,
    "Tuple": Tuple,
    "Counter": Counter,
    "defaultdict": defaultdict,
    "dataclass": dataclass,
    "field": field,
    "lru_cache": lru_cache,
    "stable_rng": stable_rng,
    "random": random,
    "json": json,
    "re": re,
    "math": math,
    "base64": base64,
    "codecs": codecs,
    "urllib": urllib,
    "string": string,
    "np": np,
}
_legacy_shims = SimpleNamespace(**_legacy_shim_exports)
_legacy_shims.__all__ = tuple(_legacy_shim_exports.keys())

_EXPORT_SOURCES = [
    ("spec", _spec),
    ("utils", _utils),
    ("anchors", _anchors),
    ("textops", _textops),
    ("sandbox", _sandbox),
    ("invariants", _invariants),
    ("soft", _soft),
    ("renderers", _renderers),
    ("generator", _generator),
    ("legacy_shims", _legacy_shims),
]

_namespace, EXPORT_CONFLICTS = _merge_exports(_EXPORT_SOURCES)
globals().update(_namespace)
__all__ = sorted(_namespace.keys())
