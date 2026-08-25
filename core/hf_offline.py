"""Carrega modelos da Hugging Face pelo cache local; só usa a rede se faltar arquivo.

O huggingface_hub 1.x lê HF_HUB_OFFLINE uma vez na importação. Mudar só o
os.environ depois disso não bloqueia HTTP — é preciso alterar também a constante.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, TypeVar

T = TypeVar("T")

_OFFLINE_ENV = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")


def _hf_offline_modules() -> list[Any]:
    modules: list[Any] = []
    try:
        import huggingface_hub.constants as hf_const

        modules.append(hf_const)
    except Exception:
        pass
    return modules


@contextmanager
def huggingface_offline(enabled: bool) -> Iterator[None]:
    saved_env = {key: os.environ.get(key) for key in _OFFLINE_ENV}
    modules = _hf_offline_modules()
    saved_const = [(mod, getattr(mod, "HF_HUB_OFFLINE", None)) for mod in modules]
    try:
        if enabled:
            for key in _OFFLINE_ENV:
                os.environ[key] = "1"
        else:
            for key in _OFFLINE_ENV:
                os.environ.pop(key, None)
        for mod in modules:
            if hasattr(mod, "HF_HUB_OFFLINE"):
                mod.HF_HUB_OFFLINE = bool(enabled)
        yield
    finally:
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        for mod, value in saved_const:
            if value is None:
                continue
            try:
                mod.HF_HUB_OFFLINE = value
            except Exception:
                pass


def is_cache_miss(exc: BaseException) -> bool:
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    if "outofmemory" in name or "out of memory" in text:
        return False
    if "gatedrepo" in name or "gated repo" in text:
        return False
    markers = (
        "offline",
        "local_files_only",
        "localentrynotfound",
        "cannot find an appropriate cached snapshot",
        "not found in the local cache",
        "cannot find the requested files",
        "is not in the cache",
        "huggingface hub is currently offline",
        "failed to locate the",
        "couldn't find a valid snapshot",
        "no cached snapshot",
        "could not be found in huggingface",
    )
    if any(m in name for m in ("offline", "localentrynotfound")):
        return True
    return any(m in text for m in markers)


def load_with_cache_first(
    *,
    offline_call: Callable[[], T],
    online_call: Callable[[], T],
    log: Callable[[str], None],
    label: str,
) -> T:
    """Tenta o cache sem HTTP; se o modelo não existir localmente, baixa da Hub."""
    try:
        with huggingface_offline(True):
            result = offline_call()
        log(f"{label}: cache local (sem Hugging Face).")
        return result
    except Exception as exc:
        if not is_cache_miss(exc):
            if "outofmemory" in type(exc).__name__.lower() or "out of memory" in str(exc).lower():
                raise
            log(f"{label}: cache local falhou ({exc}). Tentando Hugging Face…")
        else:
            log(f"{label}: não está no cache. Baixando da Hugging Face…")
        with huggingface_offline(False):
            return online_call()
