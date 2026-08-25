"""Pipeline WhisperX em thread: transcrição, alinhamento, diarização e exportação."""

from __future__ import annotations

import gc
import logging
import os
import queue
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from utils.events import Event
from core.hf_offline import load_with_cache_first

# Estágios exatamente como na especificação (exibidos na barra de progresso).
STAGE_LOAD = "Carregando modelo"
STAGE_TRANSCRIBE = "Transcrevendo"
STAGE_ALIGN = "Alinhando timestamps"
STAGE_DIARIZE = "Diarizando"
STAGE_EXPORT = "Exportando"

DIARIZE_MODEL_URL = "https://huggingface.co/pyannote/speaker-diarization-community-1"
DIARIZE_GATED_HELP = (
    "Diarização indisponível: o modelo Pyannote é gated (403). "
    "Entre na mesma conta do token, abra "
    f"{DIARIZE_MODEL_URL} e clique em Agree and access repository. "
    "O token precisa ter permissão Read. "
    "A transcrição será exportada sem identificação de locutores."
)

WRITER_OPTIONS = {
    "max_line_width": None,
    "max_line_count": None,
    "highlight_words": False,
}


@dataclass
class JobOptions:
    model: str = "small"
    language: str | None = None  # None = auto-detect
    batch_size: int = 16
    compute_type: str = "float16"
    device: str = "cpu"
    diarize: bool = False
    min_speakers: int | None = None
    max_speakers: int | None = None
    hf_token: str = ""
    output_formats: list[str] = field(default_factory=lambda: ["srt", "txt"])
    mix_audio: str = "first_two"  # off | first_two | all


def is_oom_error(exc: BaseException) -> bool:
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    needles = (
        "outofmemory",
        "out of memory",
        "cuda out of memory",
        "cudnn_status_alloc_failed",
        "cuda error: out of memory",
    )
    if "outofmemory" in name:
        return True
    return any(n in text for n in needles)


def _clear_cuda() -> None:
    gc.collect()
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _emit(q: queue.Queue[Event], **kwargs: Any) -> None:
    q.put(Event(**kwargs))


class _QueueLogHandler(logging.Handler):
    def __init__(self, event_queue: queue.Queue[Event]) -> None:
        super().__init__()
        self.event_queue = event_queue
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if record.name.startswith(("httpx", "httpcore", "huggingface_hub", "urllib3")):
                if record.levelno < logging.WARNING:
                    return
            msg = self.format(record)
            if _is_noisy_console(msg):
                return
            if msg.strip():
                _emit(self.event_queue, kind="log", message=msg)
        except Exception:
            pass


class _QueueStdWriter:
    """Redireciona print() da thread de trabalho para a janela de log."""

    def __init__(self, event_queue: queue.Queue[Event], original: Any) -> None:
        self.event_queue = event_queue
        self.original = original

    def write(self, data: str) -> int:
        text = data.replace("\r\n", "\n").replace("\r", "\n")
        for line in text.split("\n"):
            if _is_noisy_console(line):
                continue
            if line.strip():
                _emit(self.event_queue, kind="log", message=line.rstrip())
        if self.original is not None:
            try:
                self.original.write(data)
            except Exception:
                pass
        return len(data)

    def flush(self) -> None:
        if self.original is not None:
            try:
                self.original.flush()
            except Exception:
                pass


def _is_noisy_console(text: str) -> bool:
    """HTTP HEAD/404, tqdm e avisos recorrentes do HF/Lightning não ajudam no log da GUI."""
    stripped = text.strip()
    if not stripped:
        return True
    if stripped.startswith("HTTP Request:"):
        return True
    lowered = stripped.lower()
    if "huggingface_hub cache-system uses symlinks" in lowered:
        return True
    if "lightning automatically upgraded" in lowered:
        return True
    if "reproducibilitywarning" in lowered or "tensorfloat-32" in lowered:
        return True
    if "std(): degrees of freedom" in lowered:
        return True
    if "warnings.warn(" in stripped:
        return True
    if stripped.startswith("UserWarning:") or stripped.startswith("  warnings.warn"):
        return True
    # Barras tqdm: "file:  45%|####    | 12.3MB/s"
    if "%|" in stripped and ("it/s" in stripped or "/s" in stripped or "B/s" in stripped):
        return True
    if stripped.startswith("Loading weights:") and "%|" in stripped:
        return True
    return False


def _quiet_third_party_loggers() -> None:
    for name in ("httpx", "httpcore", "huggingface_hub", "urllib3", "filelock"):
        logging.getLogger(name).setLevel(logging.WARNING)


def run_jobs(files: list[str], options: JobOptions, event_queue: queue.Queue[Event]) -> None:
    """Ponto de entrada da worker thread. Sempre envia um evento `done` no finally."""
    _apply_hf_token_env(options.hf_token)
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    _quiet_third_party_loggers()
    handler = _QueueLogHandler(event_queue)
    handler.setLevel(logging.INFO)
    root_logger = logging.getLogger()
    previous_level = root_logger.level
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)

    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = _QueueStdWriter(event_queue, old_stdout)  # type: ignore[assignment]
    sys.stderr = _QueueStdWriter(event_queue, old_stderr)  # type: ignore[assignment]

    opened_dirs: list[str] = []
    try:
        opened_dirs = _run_jobs_inner(files, options, event_queue)
        _emit(
            event_queue,
            kind="done",
            message="Processamento concluído.",
            progress=1.0,
            payload={"output_dirs": opened_dirs},
        )
    except Exception as exc:
        _emit(
            event_queue,
            kind="error",
            message=_friendly_error(exc),
            payload={"traceback": traceback.format_exc()},
        )
        _emit(event_queue, kind="done", message="Encerrado com erro.", payload={"output_dirs": opened_dirs})
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        root_logger.removeHandler(handler)
        root_logger.setLevel(previous_level)


def _run_jobs_inner(
    files: list[str],
    options: JobOptions,
    q: queue.Queue[Event],
) -> list[str]:
    import whisperx  # type: ignore

    from core.audio import load_audio_for_whisper
    from utils.paths import open_folder

    device = options.device
    compute_type = options.compute_type
    language = options.language or None
    batch_size = max(1, int(options.batch_size))

    _emit(q, kind="stage", stage=STAGE_LOAD, progress=0.02, message=STAGE_LOAD)
    _emit(q, kind="log", message=f"Carregando modelo {options.model} em {device} ({compute_type})…")
    _emit(
        q,
        kind="log",
        message="Cache local primeiro: Hugging Face só entra se o modelo ainda não estiver no disco.",
    )

    def _load_asr(local: bool):
        kwargs: dict[str, Any] = {
            "whisper_arch": options.model,
            "device": device,
            "compute_type": compute_type,
            "local_files_only": local,
        }
        if options.hf_token:
            kwargs["use_auth_token"] = options.hf_token
        try:
            return whisperx.load_model(**kwargs)
        except TypeError:
            kwargs.pop("use_auth_token", None)
            return whisperx.load_model(options.model, device, compute_type=compute_type, local_files_only=local)

    try:
        model = load_with_cache_first(
            offline_call=lambda: _load_asr(True),
            online_call=lambda: _load_asr(False),
            log=lambda m: _emit(q, kind="log", message=m),
            label=f"Whisper {options.model}",
        )
    except Exception as exc:
        if is_oom_error(exc):
            _clear_cuda()
            raise RuntimeError(
                "Falta de VRAM ao carregar o modelo. Escolha um modelo menor "
                "(tiny/base/small) ou compute type int8."
            ) from exc
        raise

    align_model = None
    align_meta = None
    align_lang: str | None = None
    diarize_model = None
    output_dirs: list[str] = []
    n_files = max(len(files), 1)

    try:
        for index, raw_path in enumerate(files):
            path = str(Path(raw_path))
            file_base = index / n_files
            file_span = 1.0 / n_files
            _emit(q, kind="log", message=f"[{index + 1}/{len(files)}] {path}")

            try:
                audio, mix_msg = load_audio_for_whisper(path, options.mix_audio)
                _emit(q, kind="log", message=mix_msg)
            except Exception as exc:
                _emit(
                    q,
                    kind="warning",
                    message=(
                        f"Falha ao misturar/carregar áudio ({exc}). "
                        "Usando whisperx.load_audio (somente a primeira faixa)."
                    ),
                )
                audio = whisperx.load_audio(path)

            def file_progress(local: float, stage: str, message: str = "") -> None:
                _emit(
                    q,
                    kind="stage",
                    stage=stage,
                    progress=min(0.99, file_base + file_span * local),
                    message=message or stage,
                )

            result, batch_size = _transcribe_with_oom_retry(
                model, audio, batch_size, language, q, file_progress
            )
            detected = result.get("language") or language or "en"
            result["language"] = detected
            _emit(q, kind="log", message=f"Idioma: {detected}")

            file_progress(0.55, STAGE_ALIGN)
            try:
                if align_model is None or align_lang != detected:
                    def _load_align(local: bool):
                        return whisperx.load_align_model(
                            language_code=detected,
                            device=device,
                            model_cache_only=local,
                        )

                    align_model, align_meta = load_with_cache_first(
                        offline_call=lambda: _load_align(True),
                        online_call=lambda: _load_align(False),
                        log=lambda m: _emit(q, kind="log", message=m),
                        label=f"Alinhamento ({detected})",
                    )
                    align_lang = detected
                result = whisperx.align(
                    result["segments"],
                    align_model,
                    align_meta,
                    audio,
                    device,
                    return_char_alignments=False,
                )
                result["language"] = detected
            except Exception as exc:
                _emit(
                    q,
                    kind="warning",
                    message=(
                        f"Alinhamento indisponível para '{detected}' ({exc}). "
                        "Exportando timestamps de segmento do Whisper."
                    ),
                )

            if options.diarize:
                file_progress(0.72, STAGE_DIARIZE)
                if not options.hf_token:
                    _emit(
                        q,
                        kind="warning",
                        message=(
                            "Diarização ativada sem token HuggingFace. "
                            f"Gere um token em https://huggingface.co/settings/tokens e aceite {DIARIZE_MODEL_URL}. "
                            "Exportando sem locutores."
                        ),
                    )
                else:
                    try:
                        if diarize_model is None:
                            diarize_model = load_with_cache_first(
                                offline_call=lambda: _load_diarization(
                                    options.hf_token, device
                                ),
                                online_call=lambda: _load_diarization(
                                    options.hf_token, device
                                ),
                                log=lambda m: _emit(q, kind="log", message=m),
                                label="Diarização Pyannote",
                            )
                        diarize_kwargs: dict[str, Any] = {}
                        if options.min_speakers is not None:
                            diarize_kwargs["min_speakers"] = options.min_speakers
                        if options.max_speakers is not None:
                            diarize_kwargs["max_speakers"] = options.max_speakers
                        diarize_segments = diarize_model(audio, **diarize_kwargs)
                        result = whisperx.assign_word_speakers(diarize_segments, result)
                        result["language"] = detected
                    except Exception as exc:
                        if is_oom_error(exc):
                            _clear_cuda()
                            _emit(
                                q,
                                kind="warning",
                                message=(
                                    "VRAM insuficiente na diarização. "
                                    "Reduza o modelo ASR ou desative a diarização. "
                                    "Exportando sem locutores."
                                ),
                            )
                        elif _is_gated_hf_error(exc):
                            _emit(q, kind="warning", message=DIARIZE_GATED_HELP)
                        else:
                            _emit(
                                q,
                                kind="warning",
                                message=(
                                    f"Diarização falhou ({exc}). "
                                    "Exportando a transcrição sem rótulos de locutor."
                                ),
                            )

            file_progress(0.88, STAGE_EXPORT)
            out_dir = str(Path(path).resolve().parent)
            _export(result, path, out_dir, options.output_formats, q)
            output_dirs.append(out_dir)
            _emit(q, kind="file_done", message=f"Concluído: {Path(path).name}", payload={"dir": out_dir})
            file_progress(1.0, STAGE_EXPORT, f"Exportado em {out_dir}")
    finally:
        del model
        if align_model is not None:
            del align_model
        if diarize_model is not None:
            del diarize_model
        _clear_cuda()

    unique_dirs = list(dict.fromkeys(output_dirs))
    for folder in unique_dirs:
        try:
            open_folder(folder)
        except Exception as exc:
            _emit(q, kind="warning", message=f"Não foi possível abrir a pasta {folder}: {exc}")
    return unique_dirs


def _transcribe_with_oom_retry(
    model: Any,
    audio: Any,
    batch_size: int,
    language: str | None,
    q: queue.Queue[Event],
    file_progress,
) -> tuple[dict[str, Any], int]:
    current = max(1, batch_size)
    last_exc: BaseException | None = None
    while current >= 1:
        try:
            file_progress(0.20, STAGE_TRANSCRIBE, f"{STAGE_TRANSCRIBE} (batch_size={current})")
            kwargs: dict[str, Any] = {"batch_size": current}
            if language:
                kwargs["language"] = language
            result = model.transcribe(audio, **kwargs)
            return result, current
        except Exception as exc:
            last_exc = exc
            if not is_oom_error(exc) or current <= 1:
                if is_oom_error(exc):
                    _clear_cuda()
                    raise RuntimeError(
                        "Estouro de VRAM mesmo com batch_size=1. "
                        "Reduza o modelo (medium → small → base) ou use compute type int8."
                    ) from exc
                raise
            _clear_cuda()
            nxt = max(1, current // 2)
            _emit(
                q,
                kind="oom_retry",
                message=(
                    f"VRAM insuficiente (Out of Memory). "
                    f"Reduzindo batch_size de {current} para {nxt} e tentando novamente."
                ),
                payload={"from": current, "to": nxt},
            )
            current = nxt
    raise RuntimeError("Falha na transcrição.") from last_exc


def _apply_hf_token_env(token: str) -> None:
    """Deixa o token visível para huggingface_hub (downloads e modelos gated)."""
    token = (token or "").strip()
    if not token:
        return
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token


def _is_gated_hf_error(exc: BaseException) -> bool:
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    if "gatedrepo" in name:
        return True
    needles = (
        "gated repo",
        "not in the authorized list",
        "restricted and you are not",
        "403 forbidden",
        "cannot access gated",
    )
    return any(n in text for n in needles)


def _load_diarization(token: str, device: str) -> Any:
    """Compatível com APIs token= (README atual) e use_auth_token= (versões antigas)."""
    try:
        from whisperx.diarize import DiarizationPipeline  # type: ignore
    except Exception:
        import whisperx  # type: ignore

        DiarizationPipeline = whisperx.DiarizationPipeline

    try:
        return DiarizationPipeline(token=token, device=device)
    except TypeError:
        return DiarizationPipeline(use_auth_token=token, device=device)


def _export(
    result: dict[str, Any],
    audio_path: str,
    output_dir: str,
    formats: list[str],
    q: queue.Queue[Event],
) -> None:
    from whisperx.utils import get_writer  # type: ignore

    os.makedirs(output_dir, exist_ok=True)
    selected = [fmt.lstrip(".").lower() for fmt in formats if fmt]
    if not selected:
        selected = ["srt"]

    for fmt in selected:
        writer = get_writer(fmt, output_dir)
        writer(result, audio_path, WRITER_OPTIONS)
        _emit(q, kind="log", message=f"Exportado .{fmt} → {output_dir}")


def _friendly_error(exc: BaseException) -> str:
    if is_oom_error(exc):
        return (
            "Erro de VRAM (CUDA Out of Memory). "
            "Reduza o batch_size, troque o compute type para int8 ou use um modelo menor."
        )
    if _is_gated_hf_error(exc):
        return DIARIZE_GATED_HELP
    return str(exc)
