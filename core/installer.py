"""First-run engine: checagem de ambiente e bootstrap de PyTorch + WhisperX + FFmpeg."""

from __future__ import annotations

import importlib
import importlib.util
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field

from core.ffmpeg import apply_local_ffmpeg_to_path, ensure_ffmpeg, ffmpeg_works, which_ffmpeg
from core.hardware import HardwareInfo, detect_hardware, torch_tag_fallbacks
from utils.config import save_config
from utils.paths import CREATE_NO_WINDOW

LogFn = Callable[[str], None]
ProgressFn = Callable[[float, str], None]

WHISPERX_GIT = "git+https://github.com/m-bain/whisperx.git"
WHISPERX_PYPI = "whisperx"
MIN_PYTHON = (3, 10)

# Dependências do WhisperX 3.8+ sem torch/torchaudio: o bootstrap já instala o wheel CUDA.
# ctranslate2>=4.6.1 é o primeiro com wheels cp314; a 3.2.0 do PyPI pedia 4.4.0 (inexistente aqui).
WHISPERX_RUNTIME_DEPS = [
    "ctranslate2>=4.6.1",
    "faster-whisper>=1.2.0",
    "nltk>=3.9.1",
    "numpy>=2.1.0",
    "omegaconf>=2.3.0",
    "pandas>=2.2.3",
    "pyannote-audio>=4.0.0",
    "huggingface-hub>=0.28.1",
    "transformers>=4.48.0",
]

# Índice oficial de wheels CUDA/CPU do PyTorch.
TORCH_INDEX = "https://download.pytorch.org/whl/{tag}"


@dataclass
class EnvironmentStatus:
    python_ok: bool = False
    git_ok: bool = False
    ffmpeg_ok: bool = False
    torch_ok: bool = False
    torch_cuda: bool = False
    whisperx_ok: bool = False
    device: str = "cpu"
    cuda_tag: str = "cpu"
    python_version: str = ""
    hardware: HardwareInfo | None = None
    messages: list[str] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        """Git só é obrigatório na instalação, não para executar depois do setup."""
        return self.python_ok and self.ffmpeg_ok and self.torch_ok and self.whisperx_ok


class BootstrapError(RuntimeError):
    """Falha recuperável do setup (Git ausente, permissão, pip, etc.)."""


def check_python() -> tuple[bool, str]:
    version = sys.version.split()[0]
    ok = sys.version_info >= MIN_PYTHON
    return ok, version


def check_git() -> tuple[bool, str]:
    path = shutil.which("git")
    if not path:
        return False, ""
    try:
        proc = subprocess.run(
            [path, "--version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            creationflags=CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        if proc.returncode == 0:
            return True, (proc.stdout or proc.stderr).strip()
    except Exception:
        return False, path
    return False, path


def _module_present(name: str) -> bool:
    """True se o pacote está no site-packages (não importa — evita cache de ImportError)."""
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def _can_import_subprocess(name: str, timeout: int = 180, log: LogFn | None = None) -> bool:
    """Valida o import em um processo novo (enxerga pacotes recém-instalados pelo pip)."""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", f"import {name}"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            creationflags=CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
    except Exception as exc:
        if log:
            log(f"Falha ao importar {name}: {exc}")
        return False
    if proc.returncode != 0:
        if log:
            log((proc.stderr or proc.stdout or f"import {name} falhou").strip()[-2000:])
        return False
    return True


def _purge_ml_imports() -> None:
    importlib.invalidate_caches()
    for name in list(sys.modules):
        if (
            name in {"torch", "torchaudio", "whisperx"}
            or name.startswith("torch.")
            or name.startswith("torchaudio.")
            or name.startswith("whisperx.")
        ):
            sys.modules.pop(name, None)


def check_environment() -> EnvironmentStatus:
    """Avalia se o WhisperX já pode rodar ou se o setup automático é necessário."""
    status = EnvironmentStatus()
    status.python_ok, status.python_version = check_python()
    if not status.python_ok:
        status.messages.append(
            f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ é obrigatório (encontrado {status.python_version})."
        )
    else:
        status.messages.append(f"Python {status.python_version} OK")
        if sys.version_info >= (3, 14):
            status.messages.append(
                "Aviso: o WhisperX oficial declara Python >=3.10,<3.14. "
                "O setup tentará um modo de compatibilidade no 3.14; "
                "se falhar, use Python 3.12."
            )
        elif sys.version_info >= (3, 13):
            status.messages.append(
                "Aviso: Python 3.13+ pode não ter wheels oficiais do WhisperX. "
                "Se o setup falhar, use Python 3.10, 3.11 ou 3.12."
            )

    status.git_ok, git_info = check_git()
    status.messages.append(f"Git: {git_info or 'não encontrado'}")

    apply_local_ffmpeg_to_path()
    ffmpeg = which_ffmpeg()
    status.ffmpeg_ok = bool(ffmpeg and ffmpeg_works(ffmpeg))
    status.messages.append(f"FFmpeg: {ffmpeg if status.ffmpeg_ok else 'ausente'}")

    status.hardware = detect_hardware()
    status.cuda_tag = status.hardware.torch_index_tag
    status.device = status.hardware.recommended_device
    status.messages.extend(status.hardware.notes)

    status.torch_ok = _module_present("torch")
    if status.torch_ok:
        try:
            import torch  # type: ignore

            status.torch_cuda = bool(torch.cuda.is_available())
            status.messages.append(
                f"PyTorch {torch.__version__} | CUDA disponível: {status.torch_cuda}"
            )
            if status.torch_cuda:
                status.device = "cuda"
            else:
                status.device = "cpu"
        except Exception as exc:
            status.torch_ok = False
            status.messages.append(f"PyTorch instalado, mas falhou ao importar: {exc}")
    else:
        status.messages.append("PyTorch não instalado")

    status.whisperx_ok = _module_present("whisperx")
    status.messages.append("WhisperX OK" if status.whisperx_ok else "WhisperX não instalado")
    return status


def bootstrap(log: LogFn, progress: ProgressFn, hf_token: str | None = None) -> EnvironmentStatus:
    """
    Instala FFmpeg, PyTorch (CUDA ou CPU) e WhisperX.
    Deve ser chamado em uma thread, nunca no main thread da GUI.
    """
    token = (hf_token or "").strip()
    if token:
        save_config(hf_token=token)

    progress(0.02, "Verificando Python")
    py_ok, py_ver = check_python()
    if not py_ok:
        raise BootstrapError(
            f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ é necessário. Versão atual: {py_ver}"
        )
    log(f"Python {py_ver}")

    progress(0.06, "Verificando Git")
    git_ok, git_info = check_git()
    if git_ok:
        log(git_info or "Git OK")
    else:
        log(
            "Git não está no PATH. A instalação via repositório git+ falhará; "
            "será tentado o fallback PyPI (whisperx)."
        )

    progress(0.10, "Verificando FFmpeg")
    try:
        ffmpeg_path = ensure_ffmpeg(log=log, progress=lambda p, s: progress(0.10 + 0.15 * p, s))
        log(f"FFmpeg pronto: {ffmpeg_path}")
    except PermissionError as exc:
        raise BootstrapError(
            f"Sem permissão para instalar o FFmpeg. Execute como administrador ou instale manualmente. ({exc})"
        ) from exc
    except Exception as exc:
        raise BootstrapError(str(exc)) from exc

    progress(0.28, "Detectando hardware")
    hw = detect_hardware()
    for note in hw.notes:
        log(note)

    preferred = hw.torch_index_tag
    installed_tag = _install_torch(preferred, log, progress)

    progress(0.72, "Instalando WhisperX")
    _install_whisperx(git_ok, log)

    if installed_tag != "cpu":
        if _torch_import_ok(expect_cuda=True, log=log):
            log("PyTorch CUDA permanece disponível após o WhisperX; sem reinstalação forçada.")
        else:
            progress(0.88, "Reinstalando PyTorch CUDA")
            log("O WhisperX substituiu o PyTorch CUDA. Reinstalando o wheel com GPU.")
            installed_tag = _install_torch(
                installed_tag, log, progress, reinstall=True, start=0.88, end=0.96
            )

    progress(0.97, "Validando ambiente")
    _purge_ml_imports()
    status = check_environment()
    if not status.ready:
        details = "; ".join(status.messages)
        raise BootstrapError(f"Setup concluído, mas o ambiente ainda não está pronto: {details}")

    # Se pedimos CUDA e o torch não enxerga a GPU, ainda assim marcamos o setup como
    # completo em CPU — o usuário consegue transcrever; o log já avisou.
    device = "cuda" if status.torch_cuda else "cpu"
    if preferred != "cpu" and device == "cpu":
        log(
            "Aviso: GPU NVIDIA foi detectada, mas o PyTorch não está usando CUDA. "
            "O FalaEdinho seguirá em CPU. Verifique o driver NVIDIA."
        )

    extras = {
        "setup_complete": True,
        "device": device,
        "cuda_tag": installed_tag if device == "cuda" else "cpu",
    }
    if token:
        extras["hf_token"] = token
    save_config(extras)

    progress(1.0, "Setup concluído")
    log("Ambiente WhisperX pronto.")
    return check_environment()


def _install_torch(
    preferred_tag: str,
    log: LogFn,
    progress: ProgressFn,
    *,
    reinstall: bool = False,
    start: float = 0.30,
    end: float = 0.70,
) -> str:
    tags = torch_tag_fallbacks(preferred_tag)
    last_error = ""
    n = len(tags)
    for i, tag in enumerate(tags):
        frac = start + (end - start) * (i / max(n, 1))
        progress(frac, f"Instalando PyTorch ({tag})")
        index = TORCH_INDEX.format(tag=tag)
        args = ["install", "torch", "torchaudio", "--index-url", index]
        if reinstall:
            args.extend(["--force-reinstall", "--no-cache-dir"])
        log(f"pip {' '.join(args)}")
        rc, output = _run_pip(args, log)
        _purge_ml_imports()
        if rc == 0 and _torch_import_ok(expect_cuda=(tag != "cpu"), log=log):
            log(f"PyTorch ({tag}) instalado com sucesso.")
            return tag
        last_error = output[-4000:]
        if tag != "cpu":
            log(f"Falhou PyTorch {tag}. Tentando próximo candidato…")

    # Último recurso: PyTorch padrão do PyPI (geralmente CPU).
    progress(end, "Instalando PyTorch (PyPI/CPU)")
    rc, output = _run_pip(["install", "torch", "torchaudio"], log)
    _purge_ml_imports()
    if rc == 0 and _torch_import_ok(expect_cuda=False, log=log):
        log("PyTorch CPU instalado via PyPI (fallback).")
        return "cpu"
    raise BootstrapError(
        "Não foi possível instalar o PyTorch. "
        f"Última saída do pip:\n{last_error or output[-4000:]}"
    )


def _torch_import_ok(*, expect_cuda: bool, log: LogFn) -> bool:
    script = (
        "import torch;"
        "print(torch.__version__);"
        "print('1' if torch.cuda.is_available() else '0')"
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            creationflags=CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
    except Exception as exc:
        log(f"Falha ao validar torch: {exc}")
        return False
    if proc.returncode != 0:
        log((proc.stderr or proc.stdout or "import torch falhou").strip())
        return False
    lines = [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]
    version = lines[0] if lines else "?"
    cuda_flag = bool(lines) and lines[-1] == "1"
    log(f"import torch OK ({version}) | CUDA={cuda_flag}")
    if expect_cuda and not cuda_flag:
        log("torch.cuda.is_available() = False para este wheel.")
        return False
    return True


def _install_whisperx(git_ok: bool, log: LogFn) -> None:
    """
    Python 3.10–3.13: git+ oficial, depois PyPI.
    Python 3.14+: o metadata exige <3.14. Sem --ignore-requires-python o pip
    desce para whisperx 3.2.0 (ctranslate2==4.4.0), que não tem wheel cp314.
    """
    py314_plus = sys.version_info >= (3, 14)
    attempts: list[list[str]] = []

    if git_ok:
        if py314_plus:
            log(
                "Python 3.14+ detectado. WhisperX declara requires-python <3.14. "
                "Instalando do GitHub com --ignore-requires-python, sem rebaixar o PyTorch CUDA."
            )
            attempts.append(["install", "--ignore-requires-python", "--no-deps", WHISPERX_GIT])
        else:
            attempts.append(["install", WHISPERX_GIT])
            attempts.append(["install", "--ignore-requires-python", WHISPERX_GIT])
            attempts.append(["install", "--ignore-requires-python", "--no-deps", WHISPERX_GIT])

    if py314_plus:
        attempts.append(["install", "--ignore-requires-python", "--no-deps", "whisperx>=3.8.0"])
    else:
        attempts.append(["install", WHISPERX_PYPI])

    last_output = ""
    for args in attempts:
        log(f"pip {' '.join(args)}")
        rc, output = _run_pip(args, log)
        last_output = output
        if rc != 0:
            log("Tentativa falhou.")
            continue
        if "--no-deps" in args:
            _install_whisperx_runtime_deps(log)
        if _can_import_subprocess("whisperx", log=log):
            log("WhisperX instalado.")
            return
        log("Pacote WhisperX presente, mas o import falhou. Tentando próxima estratégia…")

    hint = ""
    if py314_plus:
        hint = (
            "\nO WhisperX oficial ainda não declara suporte a Python 3.14 "
            f"({sys.version.split()[0]}). A solução mais estável é Python 3.12:\n"
            "  py -3.12 -m venv .venv\n"
            "  .venv\\Scripts\\activate\n"
            "  pip install -r requirements.txt\n"
            "  python main.py\n"
        )
    raise BootstrapError(
        "Falha ao instalar o WhisperX." + hint + f"\nÚltima saída do pip:\n{last_output[-2500:]}"
    )


def _install_whisperx_runtime_deps(log: LogFn) -> None:
    """Instala deps do pipeline sem mexer no torch/torchaudio CUDA já escolhidos."""
    log("Instalando dependências do WhisperX (ctranslate2, faster-whisper, pyannote, …)")
    rc, output = _run_pip(["install", *WHISPERX_RUNTIME_DEPS], log)
    if rc == 0:
        return
    log("Instalação em lote falhou; tentando pacote a pacote.")
    for dep in WHISPERX_RUNTIME_DEPS:
        d_rc, d_out = _run_pip(["install", dep], log)
        if d_rc != 0:
            log(f"Aviso: não foi possível instalar {dep}.")
            log(d_out[-800:])


def _run_pip(args: list[str], log: LogFn) -> tuple[int, str]:
    """Executa pip em subprocesso, faz streaming do log e tenta --user em erro de permissão."""
    rc, output = _stream_pip(args, log)
    combined = output.lower()
    if rc != 0 and any(token in combined for token in ("permission", "access is denied", "errno 13")):
        log("Erro de permissão no pip — tentando novamente com --user.")
        user_args = _inject_user_flag(args)
        rc, output = _stream_pip(user_args, log)
    return rc, output


def _inject_user_flag(args: list[str]) -> list[str]:
    if "--user" in args:
        return args
    # ["install", ...] → ["install", "--user", ...]
    if args and args[0] == "install":
        return [args[0], "--user", *args[1:]]
    return ["--user", *args]


def _stream_pip(args: list[str], log: LogFn) -> tuple[int, str]:
    cmd = [sys.executable, "-m", "pip", *args]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"

    collected: list[str] = []
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
            creationflags=CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
    except PermissionError as exc:
        raise BootstrapError(
            f"Sem permissão para executar pip ({exc}). Tente um ambiente virtual ou terminal elevado."
        ) from exc

    assert proc.stdout is not None
    for line in proc.stdout:
        text = line.rstrip()
        if text:
            log(text)
            collected.append(text)
    rc = proc.wait()
    return rc, "\n".join(collected)
