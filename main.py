"""Ponto de entrada do FalaEdinho."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Garante imports `core`, `gui` e `utils` ao executar de qualquer cwd.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    os.chdir(ROOT)
except OSError:
    pass


def _print_startup_instructions() -> None:
    print("=" * 64, flush=True)
    print("FalaEdinho", flush=True)
    print("Como inicializar:", flush=True)
    print("  1. Use Python 3.10 ou superior", flush=True)
    print("  2. pip install -r requirements.txt", flush=True)
    print("  3. python main.py", flush=True)
    print("=" * 64, flush=True)


def main() -> int:
    _print_startup_instructions()

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    if sys.version_info >= (3, 13):
        print(
            "Aviso: Python "
            + sys.version.split()[0]
            + " é recente. WhisperX/PyTorch costumam ter melhor suporte em 3.10-3.12.",
            flush=True,
        )

    if sys.version_info < (3, 10):
        print(
            f"Erro: Python 3.10+ é obrigatório. Versão atual: {sys.version.split()[0]}",
            file=sys.stderr,
        )
        return 1

    try:
        import customtkinter as ctk

        from core.ffmpeg import apply_local_ffmpeg_to_path
        from core.installer import check_environment
        from gui.setup_window import SetupWindow
        from gui.theme import create_root
    except ImportError as exc:
        print("Instale as dependências base com:", file=sys.stderr)
        print("  pip install -r requirements.txt", file=sys.stderr)
        print(f"Detalhe: {exc}", file=sys.stderr)
        return 1

    apply_local_ffmpeg_to_path()
    status = check_environment()

    root = create_root()
    root.title("FalaEdinho")
    root.geometry("1180x800")
    root.minsize(980, 660)

    container = ctk.CTkFrame(root, fg_color="transparent", corner_radius=0)
    container.pack(fill="both", expand=True)

    def show_main() -> None:
        from gui.app import MainWindow

        for child in container.winfo_children():
            child.destroy()
        MainWindow(container).pack(fill="both", expand=True)

    if status.ready:
        print("Ambiente WhisperX detectado. Abrindo a interface principal.", flush=True)
        show_main()
    else:
        print("Ambiente incompleto. Abrindo o Setup Automático.", flush=True)
        SetupWindow(container, on_finished=show_main).pack(fill="both", expand=True)

    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
