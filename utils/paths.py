"""Caminhos de dados da aplicação (config, FFmpeg local, pasta de saída)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

APP_NAME = "FalaEdinho"
_LEGACY_APP_NAME = "PyWhisperX-GUI"

# Flag do CreateProcess no Windows para não abrir janela de console em subprocessos.
CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


def _user_data_root() -> Path:
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = str(Path.home() / "Library" / "Application Support")
    else:
        base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base)


def app_data_dir() -> Path:
    """Pasta persistente por usuário: config, FFmpeg extraído, etc."""
    path = _user_data_root() / APP_NAME
    path.mkdir(parents=True, exist_ok=True)
    _migrate_legacy_app_data(path)
    return path


def _migrate_legacy_app_data(new_dir: Path) -> None:
    """Copia config.json da pasta antiga do PyWhisperX-GUI, se ainda não existir."""
    new_cfg = new_dir / "config.json"
    if new_cfg.is_file():
        return
    old_cfg = _user_data_root() / _LEGACY_APP_NAME / "config.json"
    if not old_cfg.is_file():
        return
    try:
        shutil.copy2(old_cfg, new_cfg)
    except OSError:
        pass


def config_path() -> Path:
    return app_data_dir() / "config.json"


def ffmpeg_dir() -> Path:
    path = app_data_dir() / "ffmpeg"
    path.mkdir(parents=True, exist_ok=True)
    return path


def prepend_to_path(directory: str | Path) -> None:
    """Coloca um diretório no início do PATH deste processo (e de subprocessos)."""
    directory = str(Path(directory).resolve())
    current = os.environ.get("PATH", "")
    parts = current.split(os.pathsep)
    if directory not in parts:
        os.environ["PATH"] = directory + os.pathsep + current


def open_folder(path: str | Path) -> None:
    """Abre a pasta no explorador de arquivos do sistema."""
    folder = Path(path)
    if folder.is_file():
        folder = folder.parent
    folder.mkdir(parents=True, exist_ok=True)
    target = str(folder)

    if sys.platform == "win32":
        os.startfile(target)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", target])
    else:
        subprocess.Popen(["xdg-open", target])


def run_hidden(
    args: list[str],
    *,
    timeout: float | None = 20,
) -> subprocess.CompletedProcess[str]:
    """Executa um comando capturando stdout/stderr, sem flash de console no Windows."""
    kwargs: dict = {
        "args": args,
        "capture_output": True,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "timeout": timeout,
    }
    if sys.platform == "win32":
        kwargs["creationflags"] = CREATE_NO_WINDOW
    return subprocess.run(**kwargs)
