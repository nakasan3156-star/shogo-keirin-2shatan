"""Runtime memory/performance guards without changing PR31 prediction logic."""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable

from . import pr31_runtime
from . import production_runtime_fix

_BASE_MODEL_LOADER: Callable[[], dict[str, Any]] = pr31_runtime._load_bundle
_INSTALLED = False


@lru_cache(maxsize=1)
def _cached_model_bundle() -> dict[str, Any]:
    """Load the frozen PR31 bundle once per process and reuse the same object."""
    return _BASE_MODEL_LOADER()


def clear_pdf_text_cache() -> None:
    """Drop request-scoped temp-PDF text after each API calculation."""
    production_runtime_fix._cached_extract_text.cache_clear()


def install_runtime_memory_guard() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    pr31_runtime._load_bundle = _cached_model_bundle
    _INSTALLED = True
