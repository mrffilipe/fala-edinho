"""Persistência local de configurações (token HuggingFace, device, FFmpeg)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from utils.paths import config_path

# Valores padrão — o token HF nunca é commitado; fica só no disco do usuário.
DEFAULTS: dict[str, Any] = {
    "hf_token": "",
    "setup_complete": False,
    "device": "cpu",
    "cuda_tag": "cpu",
    "ffmpeg_bin_dir": "",
    "last_model": "small",
    "last_language": "auto",
    "last_compute_type": "",
    "last_batch_size": 0,
    "diarize": False,
    "mix_audio": "first_two",
    "last_job_seconds": 0,
}


def load_config() -> dict[str, Any]:
    path = config_path()
    data = dict(DEFAULTS)
    if path.is_file():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                data.update(raw)
        except (OSError, json.JSONDecodeError):
            pass
    return data


def save_config(updates: dict[str, Any] | None = None, **kwargs: Any) -> dict[str, Any]:
    data = load_config()
    if updates:
        data.update(updates)
    data.update(kwargs)
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return data


def hf_token() -> str:
    return str(load_config().get("hf_token") or "").strip()
