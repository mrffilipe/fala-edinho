"""Carrega áudio para o WhisperX: uma faixa ou mistura (ShadowPlay / multi-stream)."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from utils.paths import CREATE_NO_WINDOW, run_hidden

SAMPLE_RATE = 16000
MIX_OFF = "off"
MIX_FIRST_TWO = "first_two"
MIX_ALL = "all"


def count_audio_streams(path: str) -> int:
    """Conta streams de áudio com ffprobe. Fallback 1 se a sonda falhar."""
    probe = _ffprobe_bin()
    try:
        proc = run_hidden(
            [
                probe,
                "-v",
                "error",
                "-select_streams",
                "a",
                "-show_entries",
                "stream=index",
                "-of",
                "json",
                path,
            ],
            timeout=60,
        )
        if proc.returncode != 0:
            return 1
        data = json.loads(proc.stdout or "{}")
        streams = data.get("streams") or []
        return max(len(streams), 1) if streams else 1
    except Exception:
        return 1


def load_audio_for_whisper(path: str, mix_mode: str = MIX_FIRST_TWO):
    """
    Decodifica para mono float32 16 kHz (mesmo contrato de whisperx.load_audio).
    mix_mode: off | first_two | all
    """
    n_streams = count_audio_streams(path)
    mode = (mix_mode or MIX_OFF).strip().lower()
    if mode not in {MIX_OFF, MIX_FIRST_TWO, MIX_ALL}:
        mode = MIX_FIRST_TWO

    should_mix = mode != MIX_OFF and n_streams >= 2
    mix_count = 0
    if should_mix:
        mix_count = 2 if mode == MIX_FIRST_TWO else n_streams
        mix_count = min(mix_count, n_streams)

    if should_mix and mix_count >= 2:
        pcm = _decode_mixed(path, mix_count)
        if mode == MIX_FIRST_TWO:
            message = f"{n_streams} faixas detectadas; misturando as duas primeiras (0+1) em mono 16 kHz."
        else:
            message = f"{n_streams} faixas detectadas; misturando todas ({mix_count}) em mono 16 kHz."
        return _pcm_to_float32(pcm), message

    pcm = _decode_default(path)
    if n_streams <= 1:
        message = "1 faixa de áudio; carregando em mono 16 kHz."
    elif mode == MIX_OFF:
        message = f"{n_streams} faixas detectadas; mistura desligada — usando só a primeira."
    else:
        message = f"{n_streams} faixas detectadas; mistura não aplicada — usando a primeira."
    return _pcm_to_float32(pcm), message


def _decode_default(path: str) -> bytes:
    return _run_ffmpeg_pcm(
        [
            "ffmpeg",
            "-nostdin",
            "-threads",
            "0",
            "-i",
            path,
            "-f",
            "s16le",
            "-ac",
            "1",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(SAMPLE_RATE),
            "-",
        ]
    )


def _decode_mixed(path: str, n_inputs: int) -> bytes:
    labels = "".join(f"[0:a:{i}]" for i in range(n_inputs))
    filt = (
        f"{labels}amix=inputs={n_inputs}:duration=longest:dropout_transition=0:normalize=0"
    )
    return _run_ffmpeg_pcm(
        [
            "ffmpeg",
            "-nostdin",
            "-threads",
            "0",
            "-i",
            path,
            "-filter_complex",
            filt,
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "-",
        ]
    )


def _pcm_to_float32(pcm: bytes):
    import numpy as np

    if not pcm:
        raise RuntimeError("FFmpeg devolveu áudio vazio.")
    return np.frombuffer(pcm, np.int16).flatten().astype(np.float32) / 32768.0


def _run_ffmpeg_pcm(args: list[str]) -> bytes:
    kwargs: dict = {
        "args": args,
        "capture_output": True,
        "timeout": None,
    }
    if sys.platform == "win32":
        kwargs["creationflags"] = CREATE_NO_WINDOW
    proc = subprocess.run(**kwargs)
    if proc.returncode != 0:
        err = ""
        if proc.stderr:
            err = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(err or f"ffmpeg saiu com código {proc.returncode}")
    return proc.stdout or b""


def _ffprobe_bin() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        name = "ffprobe.exe" if sys.platform == "win32" else "ffprobe"
        sibling = Path(ffmpeg).with_name(name)
        if sibling.is_file():
            return str(sibling)
    return shutil.which("ffprobe") or "ffprobe"
