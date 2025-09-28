# -*- coding: utf-8 -*-
"""Stable RNG helpers shared across data generation modules."""
from __future__ import annotations

import hashlib
import random
from contextlib import contextmanager
from typing import Iterable, Optional

__all__ = [
    "stable_seed_hex",
    "stable_seed_int",
    "stable_rng",
    "derive_rng",
    "RandomBinder",
    "bind_random_module",
    "random_module_binding",
]

_RANDOM_BIND_FUNCS = (
    "random",
    "randrange",
    "randint",
    "choice",
    "choices",
    "sample",
    "shuffle",
    "uniform",
    "betavariate",
    "expovariate",
    "gammavariate",
    "gauss",
    "lognormvariate",
    "normalvariate",
    "paretovariate",
    "triangular",
    "weibullvariate",
    "getrandbits",
    "randbytes",
)

_MISSING = object()


def stable_seed_hex(*parts: object) -> str:
    """Return a deterministic hex seed by hashing the provided parts."""
    joined = "||".join("" if p is None else str(p) for p in parts)
    return hashlib.blake2b(joined.encode("utf-8"), digest_size=16).hexdigest()

def stable_seed_int(*parts: object) -> int:
    """Integer variant of :func:stable_seed_hex."""
    return int(stable_seed_hex(*parts), 16)

def stable_rng(*parts: object) -> random.Random:
    """Build a `random.Random` seeded from the hashed *parts*."""
    return random.Random(stable_seed_hex(*parts))

def derive_rng(rng: Optional[random.Random], *parts: object) -> random.Random:
    """Return *rng* if provided, otherwise derive one via :func:stable_rng."""
    if rng is not None:
        return rng
    return stable_rng(*parts)


class RandomBinder:
    """Bind module-level `random` helpers to a stable RNG instance."""

    def __init__(self, namespace: str = "compose_attacks") -> None:
        self.namespace = namespace
        self.seed_value: Optional[int] = None
        self.rng: Optional[random.Random] = None
        self._orig_attrs: dict[str, object] = {}
        self._bound = False

    def configure(self, seed: int) -> random.Random:
        self.seed_value = int(seed)
        self.rng = stable_rng(self.namespace, self.seed_value)
        self._bind()
        return self.rng

    def reseed(self, seed: Optional[int] = None) -> random.Random:
        if seed is None:
            if self.seed_value is None:
                raise RuntimeError("RandomBinder not configured")
            seed = self.seed_value
        return self.configure(int(seed))

    def seed(self, seed: Optional[int] = None) -> None:
        """Compatibility hook for module-level `random.seed`."""
        self.reseed(seed)
        return None

    def unbind(self) -> None:
        """Restore the original module-level `random` helpers."""
        if not self._bound:
            return
        for name, original in self._orig_attrs.items():
            if original is _MISSING:
                try:
                    delattr(random, name)
                except AttributeError:
                    pass
            else:
                setattr(random, name, original)
        self._orig_attrs.clear()
        self._bound = False

    def _bind(self) -> None:
        if self.rng is None:
            raise RuntimeError("RandomBinder requires configure() first")
        for name in _RANDOM_BIND_FUNCS:
            fn = getattr(self.rng, name, None)
            if fn is not None:
                self._set_random_attr(name, fn)
        if hasattr(self.rng, "_randbelow"):
            self._set_random_attr("_randbelow", getattr(self.rng, "_randbelow"))
        self._set_random_attr("seed", self.seed)
        self._bound = True

    def _set_random_attr(self, name: str, value: object) -> None:
        if name not in self._orig_attrs:
            self._orig_attrs[name] = getattr(random, name, _MISSING)
        setattr(random, name, value)


def bind_random_module(namespace: str, seed: int) -> RandomBinder:
    """Return a `RandomBinder` and bind `random` helpers to a stable RNG."""
    binder = RandomBinder(namespace=namespace)
    binder.configure(seed)
    return binder


@contextmanager
def random_module_binding(namespace: str, seed: int):
    """Context manager that binds module-level RNG helpers for the duration of the context."""
    binder = RandomBinder(namespace=namespace)
    try:
        binder.configure(seed)
        yield binder
    finally:
        binder.unbind()