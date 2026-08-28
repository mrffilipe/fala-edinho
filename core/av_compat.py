"""PyAV e outras DLLs nativas podem ser bloqueadas pelo Smart App Control no Windows."""

from __future__ import annotations

import importlib.machinery
import sys
import types


class AppControlBlockedError(RuntimeError):
    """Smart App Control / WDAC impediu carregar uma extensão nativa."""


_HELP = (
    "O Windows Smart App Control está bloqueando DLLs do Python usadas pelo WhisperX "
    "(PyAV, SciPy, etc.). Isso não é falha do modelo nem da Hugging Face.\n\n"
    "Como resolver (uma vez):\n"
    "1. Configurações → Privacidade e segurança → Segurança do Windows\n"
    "2. Controle de aplicativos e do navegador → Smart App Control\n"
    "3. Desativado\n"
    "4. Reinicie o Windows e abra o FalaEdinho de novo."
)


def is_app_control_block(exc: BaseException) -> bool:
    text = str(exc).lower()
    needles = (
        "política de controle de aplicativo",
        "politica de controle de aplicativo",
        "application control policy",
        "dll load failed while importing",
        "wdac",
        "smart app control",
    )
    return any(n in text for n in needles)


def ensure_av_importable() -> str:
    """
    Garante que `import av` não derrube o faster-whisper.

    O FalaEdinho decodifica áudio com FFmpeg; o PyAV só é exigido no import.
    Retorna 'native' ou 'ffmpeg' (stub).
    """
    try:
        import av  # type: ignore

        if not getattr(av, "_falaedinho_stub", False):
            return "native"
    except Exception as exc:
        if not is_app_control_block(exc):
            raise
        _purge_prefix("av")

    _install_av_stub()
    return "ffmpeg"


def warmup_whisperx_imports() -> str:
    """Stub do PyAV + importa whisperx.asr. Levanta AppControlBlockedError se o Windows bloquear."""
    mode = ensure_av_importable()
    try:
        import whisperx.asr  # noqa: F401
    except Exception as exc:
        if is_app_control_block(exc):
            raise AppControlBlockedError(_HELP) from exc
        raise
    return mode


def _purge_prefix(prefix: str) -> None:
    for name in list(sys.modules):
        if name == prefix or name.startswith(prefix + "."):
            sys.modules.pop(name, None)


def _install_av_stub() -> None:
    av = types.ModuleType("av")
    av.__spec__ = importlib.machinery.ModuleSpec("av", loader=None)
    av.__file__ = "<falaedinho-av-stub>"
    av._falaedinho_stub = True  # type: ignore[attr-defined]
    av.__version__ = "0+falaedinho-ffmpeg"
    sys.modules["av"] = av
