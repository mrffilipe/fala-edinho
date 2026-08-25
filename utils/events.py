"""Eventos enviados da thread de trabalho para a GUI via queue.Queue."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Event:
    """Envelope único para logs, progresso, estágios e erros."""

    kind: str
    message: str = ""
    progress: float = 0.0
    stage: str = ""
    payload: Any = None


# kind: log | stage | progress | warning | error | done | file_done | oom_retry | setup_done
