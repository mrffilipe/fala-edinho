"""Verificação de FFmpeg no PATH e download do binário quando estiver ausente."""

from __future__ import annotations

import os
import shutil
import sys
import tarfile
import zipfile
from collections.abc import Callable
from pathlib import Path

import requests

from utils.config import load_config, save_config
from utils.paths import ffmpeg_dir, prepend_to_path, run_hidden

LogFn = Callable[[str], None]
ProgressFn = Callable[[float, str], None]

# Builds oficiais estáticas — Windows é o alvo principal do FalaEdinho.
_FFMPEG_URLS_WINDOWS = [
    "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip",
    "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip",
]
_FFMPEG_URLS_LINUX = [
    "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz",
]

_BINARY_NAMES = {"ffmpeg", "ffmpeg.exe"}


def which_ffmpeg() -> str | None:
    return shutil.which("ffmpeg")


def ffmpeg_works(binary: str | None = None) -> bool:
    cmd = binary or which_ffmpeg()
    if not cmd:
        return False
    try:
        proc = run_hidden([cmd, "-version"], timeout=15)
        return proc.returncode == 0 and "ffmpeg" in (proc.stdout or "").lower()
    except Exception:
        return False


def apply_local_ffmpeg_to_path() -> str | None:
    """Garante que um FFmpeg local já extraído entre no PATH deste processo."""
    cfg = load_config()
    saved = str(cfg.get("ffmpeg_bin_dir") or "").strip()
    candidates: list[Path] = []
    if saved:
        candidates.append(Path(saved))
    candidates.append(ffmpeg_dir())

    for root in candidates:
        if not root.exists():
            continue
        found = _find_ffmpeg_binary(root)
        if found and ffmpeg_works(str(found)):
            prepend_to_path(found.parent)
            save_config(ffmpeg_bin_dir=str(found.parent))
            return str(found)
    return None


def ensure_ffmpeg(log: LogFn | None = None, progress: ProgressFn | None = None) -> str:
    """
    Retorna o caminho do executável FFmpeg.
    Se não estiver no PATH, baixa um build estático (Windows/Linux) para a pasta da app.
    """
    log = log or (lambda _m: None)
    progress = progress or (lambda _p, _s: None)

    existing = which_ffmpeg()
    if existing and ffmpeg_works(existing):
        log(f"FFmpeg encontrado no PATH: {existing}")
        return existing

    local = apply_local_ffmpeg_to_path()
    if local:
        log(f"FFmpeg local reutilizado: {local}")
        return local

    if sys.platform == "darwin":
        raise RuntimeError(
            "FFmpeg não está no PATH. No macOS instale com: brew install ffmpeg"
        )

    urls = _FFMPEG_URLS_WINDOWS if sys.platform == "win32" else _FFMPEG_URLS_LINUX
    last_error: Exception | None = None
    for url in urls:
        try:
            log(f"Baixando FFmpeg de {url}")
            archive = _download(url, log, progress)
            binary = _extract_ffmpeg(archive, log, progress)
            prepend_to_path(binary.parent)
            save_config(ffmpeg_bin_dir=str(binary.parent))
            if not ffmpeg_works(str(binary)):
                raise RuntimeError("FFmpeg extraído não executou corretamente.")
            log(f"FFmpeg instalado em {binary}")
            return str(binary)
        except Exception as exc:
            last_error = exc
            log(f"Falha no download/extração ({url}): {exc}")

    raise RuntimeError(
        "Não foi possível obter o FFmpeg automaticamente. "
        "Instale-o e garanta que 'ffmpeg' esteja no PATH."
        + (f" Último erro: {last_error}" if last_error else "")
    )


def _download(url: str, log: LogFn, progress: ProgressFn) -> Path:
    dest_dir = ffmpeg_dir()
    suffix = ".tar.xz" if url.endswith(".tar.xz") else ".zip"
    dest = dest_dir / f"ffmpeg_download{suffix}"
    progress(0.02, "Baixando FFmpeg")

    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length") or 0)
        downloaded = 0
        with dest.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=256 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                downloaded += len(chunk)
                if total:
                    # Reserva 0.02–0.75 da barra para o download.
                    progress(0.02 + 0.73 * (downloaded / total), "Baixando FFmpeg")
    log(f"Arquivo salvo: {dest} ({dest.stat().st_size} bytes)")
    return dest


def _extract_ffmpeg(archive: Path, log: LogFn, progress: ProgressFn) -> Path:
    target = ffmpeg_dir() / "dist"
    target.mkdir(parents=True, exist_ok=True)
    progress(0.78, "Extraindo FFmpeg")

    if archive.suffixes[-2:] == [".tar", ".xz"] or str(archive).endswith(".tar.xz"):
        with tarfile.open(archive, "r:xz") as tar:
            _safe_extract_tar(tar, target)
    else:
        with zipfile.ZipFile(archive) as zf:
            _safe_extract_zip(zf, target)

    binary = _find_ffmpeg_binary(target)
    if binary is None:
        raise RuntimeError("ffmpeg não foi encontrado dentro do arquivo baixado.")
    progress(0.95, "FFmpeg extraído")
    log(f"Binário: {binary}")
    return binary


def _safe_extract_zip(zf: zipfile.ZipFile, dest: Path) -> None:
    dest = dest.resolve()
    for info in zf.infolist():
        member_path = (dest / info.filename).resolve()
        if not str(member_path).startswith(str(dest)):
            continue
        zf.extract(info, dest)


def _safe_extract_tar(tar: tarfile.TarFile, dest: Path) -> None:
    dest = dest.resolve()
    for member in tar.getmembers():
        member_path = (dest / member.name).resolve()
        if not str(member_path).startswith(str(dest)):
            continue
        try:
            tar.extract(member, dest, set_attrs=False)
        except TypeError:
            tar.extract(member, dest)


def _find_ffmpeg_binary(root: Path) -> Path | None:
    if root.is_file() and root.name.lower() in _BINARY_NAMES:
        return root
    if not root.exists():
        return None
    for path in root.rglob("*"):
        if path.is_file() and path.name.lower() in _BINARY_NAMES:
            if os.access(path, os.X_OK) or path.suffix.lower() == ".exe":
                return path
            return path
    return None
