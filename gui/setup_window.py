"""Tela de Setup Automático (first-run): FFmpeg, PyTorch e WhisperX."""

from __future__ import annotations

import os
import queue
import sys
import threading
from collections.abc import Callable
from tkinter import messagebox

import customtkinter as ctk

from core.installer import BootstrapError, bootstrap, check_environment
from gui.theme import ACCENT, OK, PANEL, SURFACE
from utils.config import hf_token, save_config
from utils.events import Event

LOG_FONT = ("Consolas", 12) if sys.platform == "win32" else ("Menlo", 12)


class SetupWindow(ctk.CTkFrame):
    """Exibida quando o ambiente WhisperX ainda não está pronto."""

    def __init__(self, master, on_finished: Callable[[], None]) -> None:
        super().__init__(master, fg_color=SURFACE, corner_radius=0)
        self.on_finished = on_finished
        self.event_queue: queue.Queue[Event] = queue.Queue()
        self._busy = False
        self._build()
        if os.environ.get("FALAEDINHO_NO_AUTOSETUP") != "1":
            self.after(400, self._start_bootstrap)
        self.after(80, self._poll)

    def _build(self) -> None:
        status = check_environment()

        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=28, pady=(24, 8))
        ctk.CTkLabel(
            header,
            text="Setup Automático",
            font=ctk.CTkFont(size=26, weight="bold"),
        ).pack(anchor="w")
        ctk.CTkLabel(
            header,
            text="O FalaEdinho vai detectar o hardware, instalar FFmpeg, PyTorch (CUDA ou CPU) e o WhisperX.",
            text_color="#A1A1AA",
            wraplength=900,
            justify="left",
        ).pack(anchor="w", pady=(6, 0))

        checklist = ctk.CTkFrame(self, fg_color=PANEL, corner_radius=12)
        checklist.pack(fill="x", padx=28, pady=12)
        items = [
            ("Python 3.10+", status.python_ok, status.python_version),
            ("Git", status.git_ok, "necessário para pip install git+whisperx"),
            ("FFmpeg", status.ffmpeg_ok, "será baixado se estiver ausente"),
            ("PyTorch", status.torch_ok, "CUDA" if status.torch_cuda else "CPU / não instalado"),
            ("WhisperX", status.whisperx_ok, "pacote oficial"),
        ]
        for label, ok, detail in items:
            row = ctk.CTkFrame(checklist, fg_color="transparent")
            row.pack(fill="x", padx=16, pady=4)
            mark = "OK" if ok else "—"
            color = OK if ok else "#A1A1AA"
            ctk.CTkLabel(row, text=mark, text_color=color, width=36).pack(side="left")
            ctk.CTkLabel(row, text=f"{label}  ·  {detail}").pack(side="left")

        token_frame = ctk.CTkFrame(self, fg_color=PANEL, corner_radius=12)
        token_frame.pack(fill="x", padx=28, pady=(0, 12))
        ctk.CTkLabel(
            token_frame,
            text="HuggingFace Access Token (opcional agora — obrigatório para diarização Pyannote)",
        ).pack(anchor="w", padx=16, pady=(12, 4))
        self.token_entry = ctk.CTkEntry(token_frame, placeholder_text="hf_...", show="*", height=36)
        self.token_entry.pack(fill="x", padx=16, pady=(0, 14))
        existing = hf_token()
        if existing:
            self.token_entry.insert(0, existing)

        self.stage_label = ctk.CTkLabel(self, text="Aguardando início…", text_color=ACCENT)
        self.stage_label.pack(anchor="w", padx=28)
        self.progress = ctk.CTkProgressBar(self, height=14)
        self.progress.pack(fill="x", padx=28, pady=(4, 10))
        self.progress.set(0)

        self.log_box = ctk.CTkTextbox(self, font=LOG_FONT, wrap="word")
        self.log_box.pack(fill="both", expand=True, padx=28, pady=(0, 12))
        for line in status.messages:
            self._append_log(line)

        self.retry_btn = ctk.CTkButton(
            self,
            text="Iniciar Setup Automático",
            command=self._start_bootstrap,
            height=40,
            fg_color=ACCENT,
        )
        self.retry_btn.pack(pady=(0, 20))

    def _append_log(self, message: str) -> None:
        self.log_box.insert("end", message.rstrip() + "\n")
        self.log_box.see("end")

    def _start_bootstrap(self) -> None:
        if self._busy:
            return
        self._busy = True
        self.retry_btn.configure(state="disabled")
        token = self.token_entry.get().strip()
        if token:
            save_config(hf_token=token)

        def worker() -> None:
            def log(msg: str) -> None:
                self.event_queue.put(Event(kind="log", message=msg))

            def progress(value: float, stage: str) -> None:
                self.event_queue.put(Event(kind="stage", stage=stage, progress=value, message=stage))

            try:
                bootstrap(log=log, progress=progress, hf_token=token)
                self.event_queue.put(Event(kind="setup_done", message="Setup concluído."))
            except BootstrapError as exc:
                self.event_queue.put(Event(kind="error", message=str(exc)))
            except Exception as exc:
                self.event_queue.put(Event(kind="error", message=f"Erro inesperado no setup: {exc}"))

        threading.Thread(target=worker, daemon=True).start()

    def _poll(self) -> None:
        if not self.winfo_exists():
            return
        try:
            while True:
                event = self.event_queue.get_nowait()
                self._handle(event)
        except queue.Empty:
            pass
        if self.winfo_exists():
            self.after(80, self._poll)

    def _handle(self, event: Event) -> None:
        if event.kind == "log" and event.message:
            self._append_log(event.message)
        elif event.kind == "stage":
            self.stage_label.configure(text=event.stage or event.message)
            self.progress.set(max(0.0, min(1.0, event.progress)))
        elif event.kind == "error":
            self._busy = False
            self._append_log("ERRO: " + event.message)
            self.stage_label.configure(text="Falha no setup")
            self.retry_btn.configure(state="normal", text="Tentar novamente")
            messagebox.showerror("Setup Automático", event.message)
        elif event.kind == "setup_done":
            self._busy = False
            self.progress.set(1.0)
            self.stage_label.configure(text="Ambiente pronto")
            self._append_log("Setup concluído. Abrindo a interface principal…")
            self.after(400, self.on_finished)
