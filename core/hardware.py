"""Detecção de GPU NVIDIA, versão CUDA do driver e mapeamento para wheels PyTorch."""

from __future__ import annotations

import csv
import io
import os
import re
import shutil
import sys
import threading
from dataclasses import dataclass, field

from utils.paths import run_hidden

# nvidia-smi clássico: "CUDA Version: 12.6"
# nvidia-smi recente (R610+): "CUDA UMD Version: 13.3"
_CUDA_VERSION_RE = re.compile(r"CUDA (?:UMD )?Version:\s*(\d+)\.(\d+)", re.IGNORECASE)
_DRIVER_RE = re.compile(r"(?:Driver Version|KMD Version):\s*([\d.]+)", re.IGNORECASE)


@dataclass
class HardwareInfo:
    has_nvidia: bool = False
    gpu_name: str = ""
    driver_version: str = ""
    cuda_major: int | None = None
    cuda_minor: int | None = None
    torch_index_tag: str = "cpu"
    recommended_device: str = "cpu"
    notes: list[str] = field(default_factory=list)

    @property
    def cuda_label(self) -> str:
        if self.cuda_major is None:
            return "n/d"
        minor = self.cuda_minor if self.cuda_minor is not None else 0
        return f"{self.cuda_major}.{minor}"


def map_cuda_to_torch_tag(major: int, minor: int) -> str:
    """Mapeia a CUDA do driver para o índice de wheels do PyTorch (spec: cu118/cu121/cu124)."""
    version = (major, minor)
    # Drivers novos (12.8+/13.x) tentam wheels recentes primeiro; o pip faz fallback.
    if version >= (12, 8) or major >= 13:
        return "cu128"
    if version >= (12, 6):
        return "cu126"
    if version >= (12, 4):
        return "cu124"
    if version >= (12, 1):
        return "cu121"
    if version >= (11, 8):
        return "cu118"
    return "cpu"


def torch_tag_fallbacks(preferred: str) -> list[str]:
    """Ordem de tentativa se o wheel preferido falhar no pip."""
    chain = ["cu128", "cu126", "cu124", "cu121", "cu118", "cpu"]
    if preferred == "cpu":
        return ["cpu"]
    if preferred in chain:
        return chain[chain.index(preferred) :]
    return chain


def detect_hardware() -> HardwareInfo:
    """Identifica NVIDIA + CUDA via nvidia-smi, com fallback wmic/CIM no Windows."""
    info = HardwareInfo()

    smi = shutil.which("nvidia-smi")
    if smi:
        _fill_from_nvidia_smi(info, smi)
    elif sys.platform == "win32":
        _fill_from_windows_wmi(info)
        if info.has_nvidia:
            info.notes.append(
                "GPU NVIDIA detectada via WMI, mas nvidia-smi não está no PATH. "
                "Instale os drivers NVIDIA para habilitar CUDA."
            )
    else:
        info.notes.append("nvidia-smi nao encontrado - assumindo CPU.")

    if info.has_nvidia and info.cuda_major is not None:
        tag = map_cuda_to_torch_tag(info.cuda_major, info.cuda_minor or 0)
        if tag == "cpu":
            info.notes.append(
                f"Driver CUDA {info.cuda_label} é antigo demais para os wheels atuais. "
                "Usando PyTorch CPU. Atualize o driver NVIDIA para usar GPU."
            )
            info.torch_index_tag = "cpu"
            info.recommended_device = "cpu"
        else:
            info.torch_index_tag = tag
            info.recommended_device = "cuda"
            info.notes.append(
                f"GPU {info.gpu_name or 'NVIDIA'} | driver {info.driver_version or 'n/d'} | "
                f"CUDA {info.cuda_label} -> PyTorch {tag}"
            )
    elif info.has_nvidia:
        # NVIDIA sem versão CUDA parseada: tenta cu121 (exemplo da spec) e cai no pip fallback.
        info.torch_index_tag = "cu121"
        info.recommended_device = "cuda"
        info.notes.append(
            "GPU NVIDIA detectada, mas a versão CUDA do driver não foi lida. "
            "Tentando wheel cu121."
        )
    else:
        info.torch_index_tag = "cpu"
        info.recommended_device = "cpu"
        if not info.notes:
            info.notes.append("Nenhuma GPU NVIDIA compatível encontrada. PyTorch será instalado em modo CPU.")

    _enrich_with_torch(info)
    return info


def _fill_from_nvidia_smi(info: HardwareInfo, smi: str) -> None:
    try:
        query = run_hidden(
            [smi, "--query-gpu=name,driver_version", "--format=csv,noheader"],
            timeout=15,
        )
        if query.returncode == 0 and query.stdout.strip():
            first = query.stdout.strip().splitlines()[0]
            parts = [p.strip() for p in first.split(",")]
            if parts:
                info.gpu_name = parts[0]
                info.has_nvidia = True
            if len(parts) > 1:
                info.driver_version = parts[1]
    except Exception as exc:
        info.notes.append(f"Falha ao consultar nvidia-smi (query): {exc}")

    try:
        full = run_hidden([smi], timeout=15)
        text = (full.stdout or "") + "\n" + (full.stderr or "")
        match = _CUDA_VERSION_RE.search(text)
        if match:
            info.cuda_major = int(match.group(1))
            info.cuda_minor = int(match.group(2))
            info.has_nvidia = True
        driver = _DRIVER_RE.search(text)
        if driver and not info.driver_version:
            info.driver_version = driver.group(1)
        if not info.gpu_name:
            # Linha da tabela costuma conter o nome após o ID do GPU.
            for line in text.splitlines():
                if "NVIDIA" in line and any(tag in line for tag in ("GeForce", "RTX", "Quadro", "Tesla", "A100", "A6000")):
                    info.gpu_name = line.strip()
                    info.has_nvidia = True
                    break
    except Exception as exc:
        info.notes.append(f"Falha ao executar nvidia-smi: {exc}")


def _fill_from_windows_wmi(info: HardwareInfo) -> None:
    """Fallback: lista controladores de vídeo via wmic ou PowerShell CIM."""
    names: list[str] = []
    try:
        proc = run_hidden(
            ["wmic", "path", "win32_VideoController", "get", "Name"],
            timeout=20,
        )
        if proc.returncode == 0:
            for line in proc.stdout.splitlines():
                line = line.strip()
                if line and line.lower() != "name":
                    names.append(line)
    except Exception:
        names = []

    if not names:
        try:
            proc = run_hidden(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name",
                ],
                timeout=20,
            )
            if proc.returncode == 0:
                names = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
        except Exception:
            names = []

    nvidia = [n for n in names if "nvidia" in n.lower()]
    if nvidia:
        info.has_nvidia = True
        info.gpu_name = nvidia[0]


def _enrich_with_torch(info: HardwareInfo) -> None:
    """Se o torch já estiver instalado, confirma se CUDA realmente funciona."""
    try:
        import torch  # type: ignore
    except Exception:
        return

    cuda_ok = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
    if cuda_ok:
        info.recommended_device = "cuda"
        try:
            info.gpu_name = info.gpu_name or torch.cuda.get_device_name(0)
        except Exception:
            pass
        info.has_nvidia = True
        info.notes.append("torch.cuda.is_available() = True")
    else:
        if info.recommended_device == "cuda":
            info.notes.append(
                "Wheel CUDA pode estar instalado, mas torch.cuda.is_available() = False. "
                "O pipeline usará CPU até o setup reinstalar o PyTorch correto."
            )
        info.notes.append("torch.cuda.is_available() = False")


@dataclass
class RuntimeSnapshot:
    """Métricas ao vivo de CPU, RAM e GPU para o cabeçalho da GUI."""

    cpu_name: str = ""
    cpu_percent: float | None = None
    cpu_count: int | None = None
    ram_used_gb: float | None = None
    ram_total_gb: float | None = None
    ram_percent: float | None = None
    gpu_name: str = ""
    gpu_util: float | None = None
    vram_used_gb: float | None = None
    vram_total_gb: float | None = None
    gpu_temp_c: float | None = None
    has_nvidia: bool = False


class RuntimeSampler:
    """Amostra CPU/RAM/GPU sem bloquear demais a GUI (nvidia-smi no máximo a cada ciclo)."""

    def __init__(self) -> None:
        self._cpu_name = ""
        self._cpu_times = _cpu_idle_total()
        threading.Thread(target=self._load_cpu_name, daemon=True).start()

    def _load_cpu_name(self) -> None:
        try:
            self._cpu_name = detect_cpu_name()
        except Exception:
            self._cpu_name = "CPU"

    def sample(self) -> RuntimeSnapshot:
        snap = RuntimeSnapshot(cpu_name=self._cpu_name or "CPU", cpu_count=os.cpu_count())
        self._fill_cpu(snap)
        self._fill_ram(snap)
        self._fill_gpu(snap)
        return snap

    def _fill_cpu(self, snap: RuntimeSnapshot) -> None:
        times = _cpu_idle_total()
        if times is None:
            return
        idle, total = times
        prev = self._cpu_times
        self._cpu_times = times
        if prev is None:
            return
        d_idle = idle - prev[0]
        d_total = total - prev[1]
        if d_total <= 0:
            return
        busy = max(0.0, 1.0 - (d_idle / d_total))
        snap.cpu_percent = busy * 100.0

    def _fill_ram(self, snap: RuntimeSnapshot) -> None:
        ram = _ram_bytes()
        if ram is None:
            return
        used, total = ram
        if total <= 0:
            return
        snap.ram_used_gb = used / (1024**3)
        snap.ram_total_gb = total / (1024**3)
        snap.ram_percent = used / total * 100.0

    def _fill_gpu(self, snap: RuntimeSnapshot) -> None:
        if not _fill_gpu_nvidia_smi(snap):
            _fill_gpu_torch(snap)


def detect_cpu_name() -> str:
    """Nome do processador (cacheável; WMI/CIM no Windows)."""
    if sys.platform == "win32":
        name = _windows_cpu_name()
        if name:
            return name
    elif sys.platform == "darwin":
        try:
            proc = run_hidden(["sysctl", "-n", "machdep.cpu.brand_string"], timeout=5)
            if proc.returncode == 0 and proc.stdout.strip():
                return proc.stdout.strip()
        except Exception:
            pass
    else:
        try:
            with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except Exception:
            pass
    try:
        import platform

        brand = (platform.processor() or "").strip()
    except Exception:
        brand = ""
    return brand or "CPU"


def _windows_cpu_name() -> str:
    try:
        proc = run_hidden(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_Processor | Select-Object -ExpandProperty Name",
            ],
            timeout=5,
        )
        if proc.returncode == 0:
            for line in proc.stdout.splitlines():
                line = line.strip()
                if line:
                    return line
    except Exception:
        pass
    try:
        proc = run_hidden(["wmic", "cpu", "get", "Name"], timeout=5)
        if proc.returncode == 0:
            for line in proc.stdout.splitlines():
                line = line.strip()
                if line and line.lower() != "name":
                    return line
    except Exception:
        pass
    return ""


def _cpu_idle_total() -> tuple[int, int] | None:
    if sys.platform == "win32":
        return _windows_cpu_idle_total()
    try:
        with open("/proc/stat", encoding="utf-8") as fh:
            parts = fh.readline().split()
        if not parts or parts[0] != "cpu":
            return None
        nums = [int(x) for x in parts[1:11]]
        idle = nums[3] + (nums[4] if len(nums) > 4 else 0)
        total = sum(nums)
        return idle, total
    except Exception:
        return None


def _windows_cpu_idle_total() -> tuple[int, int] | None:
    import ctypes
    from ctypes import wintypes

    class FILETIME(ctypes.Structure):
        _fields_ = [("dwLowDateTime", wintypes.DWORD), ("dwHighDateTime", wintypes.DWORD)]

    idle = FILETIME()
    kernel = FILETIME()
    user = FILETIME()
    if not ctypes.windll.kernel32.GetSystemTimes(
        ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)
    ):
        return None

    def _to_int(ft: FILETIME) -> int:
        return (ft.dwHighDateTime << 32) + ft.dwLowDateTime

    idle_i = _to_int(idle)
    # Kernel time on Windows includes idle.
    total = _to_int(kernel) + _to_int(user)
    return idle_i, total


def _ram_bytes() -> tuple[int, int] | None:
    if sys.platform == "win32":
        return _windows_ram_bytes()
    try:
        info: dict[str, int] = {}
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                key, raw, *_rest = line.replace(":", " ").split()
                if key in {"MemTotal", "MemAvailable"}:
                    info[key] = int(raw) * 1024
        total = info.get("MemTotal")
        avail = info.get("MemAvailable")
        if total and avail is not None:
            return total - avail, total
    except Exception:
        pass
    return None


def _windows_ram_bytes() -> tuple[int, int] | None:
    import ctypes

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
        return None
    total = int(stat.ullTotalPhys)
    used = total - int(stat.ullAvailPhys)
    return used, total


def _parse_smi_number(raw: str) -> float | None:
    text = (raw or "").strip()
    if not text or text.upper() in {"N/A", "[N/A]", "NA"}:
        return None
    try:
        return float(text.replace(",", "."))
    except ValueError:
        return None


def _fill_gpu_nvidia_smi(snap: RuntimeSnapshot) -> bool:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return False
    try:
        proc = run_hidden(
            [
                smi,
                "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            timeout=8,
        )
    except Exception:
        return False
    if proc.returncode != 0 or not proc.stdout.strip():
        return False
    first = proc.stdout.strip().splitlines()[0]
    try:
        row = next(csv.reader(io.StringIO(first)))
    except Exception:
        return False
    if len(row) < 3:
        return False
    snap.has_nvidia = True
    snap.gpu_name = row[0].strip()
    used_mib = _parse_smi_number(row[1])
    total_mib = _parse_smi_number(row[2])
    if used_mib is not None:
        snap.vram_used_gb = used_mib / 1024.0
    if total_mib is not None:
        snap.vram_total_gb = total_mib / 1024.0
    if len(row) > 3:
        snap.gpu_util = _parse_smi_number(row[3])
    if len(row) > 4:
        snap.gpu_temp_c = _parse_smi_number(row[4])
    return True


def _fill_gpu_torch(snap: RuntimeSnapshot) -> None:
    try:
        import torch  # type: ignore

        if not torch.cuda.is_available():
            return
        snap.has_nvidia = True
        snap.gpu_name = snap.gpu_name or torch.cuda.get_device_name(0)
        free, total = torch.cuda.mem_get_info(0)
        snap.vram_total_gb = total / (1024**3)
        snap.vram_used_gb = (total - free) / (1024**3)
    except Exception:
        return


def shorten_hw_name(name: str, limit: int = 40) -> str:
    text = " ".join((name or "").replace("NVIDIA ", "").split())
    for token in ("(R)", "(TM)", "(C)"):
        text = text.replace(token, "")
    text = " ".join(text.split())
    text = re.sub(r"^\d+(st|nd|rd|th)\s+Gen\s+", "", text, flags=re.I)
    if len(text) <= limit:
        return text
    return text[: max(1, limit - 1)] + "…"


def format_gb(value: float | None) -> str:
    if value is None:
        return "n/d"
    return f"{value:.1f}"


@dataclass
class RuntimeView:
    """Textos prontos para o selo do canto e os três cartões GPU/CPU/RAM."""

    badge: str
    gpu_main: str
    gpu_sub: str
    cpu_main: str
    cpu_sub: str
    ram_main: str
    ram_sub: str


def format_runtime_view(snap: RuntimeSnapshot, device: str) -> RuntimeView:
    gpu = shorten_hw_name(snap.gpu_name) if snap.gpu_name else ""
    if device == "cuda" and (gpu or snap.has_nvidia):
        badge = f"{gpu or 'GPU NVIDIA'}  ·  CUDA"
    elif snap.has_nvidia and gpu:
        badge = f"{gpu}  ·  CPU"
    else:
        badge = "CPU  ·  sem CUDA"

    if snap.vram_used_gb is not None and snap.vram_total_gb is not None:
        pct = ""
        if snap.vram_total_gb > 0:
            pct = f" ({snap.vram_used_gb / snap.vram_total_gb * 100:.0f}%)"
        gpu_main = f"VRAM  {format_gb(snap.vram_used_gb)} / {format_gb(snap.vram_total_gb)} GB{pct}"
        extras = []
        if snap.gpu_util is not None:
            extras.append(f"uso {snap.gpu_util:.0f}%")
        if snap.gpu_temp_c is not None:
            extras.append(f"{snap.gpu_temp_c:.0f} °C")
        gpu_sub = "  ·  ".join(extras) if extras else "VRAM"
    elif snap.has_nvidia or device == "cuda":
        gpu_main = "VRAM n/d"
        gpu_sub = gpu or "NVIDIA"
    else:
        gpu_main = "Indisponível"
        gpu_sub = "Pipeline em CPU"

    cpu_main = shorten_hw_name(snap.cpu_name or "CPU", 36)
    cpu_bits = []
    if snap.cpu_percent is not None:
        cpu_bits.append(f"{snap.cpu_percent:.0f}%")
    if snap.cpu_count:
        cpu_bits.append(f"{snap.cpu_count} núcleos")
    cpu_sub = "  ·  ".join(cpu_bits) if cpu_bits else "uso n/d"

    if snap.ram_used_gb is not None and snap.ram_total_gb is not None:
        ram_main = f"{format_gb(snap.ram_used_gb)} / {format_gb(snap.ram_total_gb)} GB"
        ram_sub = f"{snap.ram_percent:.0f}% em uso" if snap.ram_percent is not None else ""
    else:
        ram_main = "n/d"
        ram_sub = ""

    return RuntimeView(
        badge=badge,
        gpu_main=gpu_main,
        gpu_sub=gpu_sub,
        cpu_main=cpu_main,
        cpu_sub=cpu_sub,
        ram_main=ram_main,
        ram_sub=ram_sub,
    )
