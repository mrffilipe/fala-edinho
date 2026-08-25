"""Janela principal: fila de arquivos, parâmetros WhisperX, progresso e logs."""

from __future__ import annotations

import queue
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox
from urllib.parse import unquote, urlparse

import customtkinter as ctk

from core.hardware import RuntimeSampler, format_runtime_view
from core.installer import check_environment
from core.whisper_runner import JobOptions, run_jobs
from gui.theme import ACCENT, OK, PANEL, SURFACE
from gui.widgets import add_param_label, bind_positive_int, make_readonly_combo
from utils.config import hf_token, load_config, save_config
from utils.events import Event

ALLOWED_EXT = {".mp4", ".mkv", ".mov", ".mp3", ".wav", ".m4a"}
MODELS = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]
OUTPUT_FORMATS = ["srt", "vtt", "txt", "json", "tsv"]
MIX_SCOPE_LABELS = [
    ("first_two", "Duas primeiras"),
    ("all", "Todas as faixas"),
]
LOG_FONT = ("Consolas", 12) if sys.platform == "win32" else ("Menlo", 12)
HW_FONT = ("Segoe UI", 11) if sys.platform == "win32" else ("Helvetica", 11)
START_LABEL = "Iniciar Transcrição"
START_BUSY_LABEL = "Transcrevendo…"

LANGUAGES: list[tuple[str, str]] = [
    ("auto", "Detectar automaticamente"),
    ("pt", "Português"),
    ("en", "Inglês"),
    ("es", "Espanhol"),
    ("fr", "Francês"),
    ("de", "Alemão"),
    ("it", "Italiano"),
    ("ja", "Japonês"),
    ("zh", "Chinês"),
    ("ko", "Coreano"),
    ("ru", "Russo"),
    ("ar", "Árabe"),
    ("nl", "Holandês"),
    ("pl", "Polonês"),
    ("tr", "Turco"),
    ("uk", "Ucraniano"),
    ("hi", "Hindi"),
    ("sv", "Sueco"),
    ("cs", "Tcheco"),
    ("el", "Grego"),
]

PARAM_HELP = {
    "model": (
        "Tamanho do modelo Whisper. Quanto maior, melhor a qualidade e mais VRAM/tempo.\n"
        "• tiny / base: rápido, menos preciso\n"
        "• small / medium: equilíbrio\n"
        "• large-v2 / large-v3: melhor qualidade\n"
        "Ex.: large-v2 numa RTX 5070."
    ),
    "language": (
        "Idioma falado no áudio. “Detectar automaticamente” analisa os primeiros segundos.\n"
        "Se o material é sempre em português, escolha Português — fica mais estável e um pouco mais rápido."
    ),
    "batch": (
        "Quantos trechos o modelo processa de uma vez. Valores altos aceleram, mas gastam mais VRAM.\n"
        "Padrão: 16 na GPU, 4 na CPU. Se faltar memória, o app reduz o lote sozinho."
    ),
    "compute": (
        "Como os números são calculados na GPU/CPU.\n"
        "• float16: padrão em GPUs modernas (recomendado)\n"
        "• int8: usa menos memória (CPU ou GPU apertada)\n"
        "• float32: máxima precisão, mais lento e pesado"
    ),
    "mix": (
        "Junta várias faixas de áudio numa só antes de transcrever.\n"
        "Útil em gravações com sistema + microfone (ShadowPlay, OBS, etc.). "
        "Desligue para usar só a primeira faixa."
    ),
    "mix_scope": (
        "Quais faixas entram na mistura.\n"
        "• Duas primeiras: típico de jogo + microfone\n"
        "• Todas as faixas: mistura qualquer quantidade de streams"
    ),
    "diarize": (
        "Tenta marcar quem falou em cada trecho (SPEAKER_00, SPEAKER_01…).\n"
        "Precisa de um token Hugging Face e de aceitar o modelo Pyannote na mesma conta. "
        "Sem isso, a transcrição sai sem identificação de locutores."
    ),
    "speakers": (
        "Dica opcional para o Pyannote. Deixe vazio para detectar sozinho.\n"
        "Só números inteiros a partir de 1.\n"
        "Ex.: entrevista a dois → mínimo 2 e máximo 2."
    ),
    "token": (
        "Token com permissão Read: https://huggingface.co/settings/tokens\n"
        "Na mesma conta, aceite o modelo:\n"
        "https://huggingface.co/pyannote/speaker-diarization-community-1"
    ),
    "formats": (
        "Pode marcar vários.\n"
        "• .srt / .vtt: legendas\n"
        "• .txt: texto corrido\n"
        "• .json / .tsv: dados para outros programas"
    ),
}


def _language_label(code: str, name: str) -> str:
    return name if code == "auto" else f"{name} ({code})"


def _format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "—"
    total = int(round(seconds))
    if total < 1:
        return "< 1 s"
    if total < 60:
        return f"{total} s"
    minutes, secs = divmod(total, 60)
    if minutes < 60:
        return f"{minutes} min {secs:02d} s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes:02d} min"


class MainWindow(ctk.CTkFrame):
    def __init__(self, master) -> None:
        super().__init__(master, fg_color=SURFACE, corner_radius=0)
        self.event_queue: queue.Queue[Event] = queue.Queue()
        self.files: list[str] = []
        self._busy = False
        self._oom_alerted = False
        self._file_rows: dict[str, ctk.CTkFrame] = {}
        self._hw_sampler = RuntimeSampler()
        self._hw_job: str | None = None
        self._job_started_at: float | None = None
        self._job_files_total = 0
        self._job_files_done = 0
        self._job_progress = 0.0
        self._eta_smooth: float | None = None
        self._stats_job: str | None = None
        self._last_duration_sec: float | None = None

        env = check_environment()
        cfg = load_config()
        try:
            self._last_duration_sec = float(cfg.get("last_job_seconds") or 0) or None
        except (TypeError, ValueError):
            self._last_duration_sec = None
        self.device = env.device if env.ready else (cfg.get("device") or "cpu")
        default_batch = 16 if self.device == "cuda" else 4
        default_compute = "float16" if self.device == "cuda" else "int8"
        if cfg.get("last_batch_size"):
            default_batch = int(cfg["last_batch_size"])
        if cfg.get("last_compute_type"):
            default_compute = str(cfg["last_compute_type"])

        self.batch_var = tk.IntVar(value=default_batch)
        self.compute_var = tk.StringVar(value=default_compute)
        self.diarize_var = tk.BooleanVar(value=bool(cfg.get("diarize")))
        self._want_diarize = bool(self.diarize_var.get())
        saved_mix = str(cfg.get("mix_audio") or "first_two")
        self.mix_enabled = tk.BooleanVar(value=(saved_mix != "off"))
        self._saved_mix_scope = saved_mix if saved_mix in {"first_two", "all"} else "first_two"
        self.format_vars = {
            fmt: tk.BooleanVar(value=(fmt in ("srt", "txt"))) for fmt in OUTPUT_FORMATS
        }

        self._build(env)
        self._bind_dnd()
        self.after(80, self._poll)
        self._refresh_hardware()
        self._refresh_job_stats()

    def _build(self, env) -> None:
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=20, pady=(12, 6))
        ctk.CTkLabel(
            header,
            text="FalaEdinho",
            font=ctk.CTkFont(size=24, weight="bold"),
        ).pack(side="left", anchor="n", pady=(2, 0))

        badge = ctk.CTkFrame(header, fg_color=PANEL, corner_radius=10)
        badge.pack(side="right")
        self._hw_badge = ctk.CTkLabel(
            badge,
            text="",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=OK if self.device == "cuda" else "#A1A1AA",
            anchor="e",
        )
        self._hw_badge.pack(padx=12, pady=8)

        stats = ctk.CTkFrame(self, fg_color="transparent")
        stats.pack(fill="x", padx=20, pady=(0, 8))
        stats.grid_columnconfigure((0, 1, 2), weight=1, uniform="hw")
        self._hw_gpu_main, self._hw_gpu_sub = self._metric_card(stats, "GPU", 0)
        self._hw_cpu_main, self._hw_cpu_sub = self._metric_card(stats, "CPU", 1)
        self._hw_ram_main, self._hw_ram_sub = self._metric_card(stats, "RAM", 2)

        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=20)
        body.grid_columnconfigure(0, weight=3)
        body.grid_columnconfigure(1, weight=2)
        body.grid_rowconfigure(0, weight=1)

        self._build_files_panel(body)
        self._build_params_panel(body)

        footer = ctk.CTkFrame(self, fg_color="transparent")
        footer.pack(fill="x", padx=20, pady=(6, 8))
        status_row = ctk.CTkFrame(footer, fg_color="transparent")
        status_row.pack(fill="x")
        self.stage_label = ctk.CTkLabel(status_row, text="Pronto", text_color=ACCENT)
        self.stage_label.pack(side="left")
        self.stats_label = ctk.CTkLabel(
            status_row,
            text="",
            text_color="#A1A1AA",
            font=ctk.CTkFont(size=12),
            anchor="e",
        )
        self.stats_label.pack(side="right")
        self.progress = ctk.CTkProgressBar(footer, height=12)
        self.progress.pack(fill="x", pady=(4, 6))
        self.progress.set(0)

        self.log_box = ctk.CTkTextbox(footer, height=118, font=LOG_FONT, wrap="word")
        self.log_box.pack(fill="x")
        self._log("Ambiente pronto. Arraste arquivos ou use o botão para selecionar.")
        for msg in env.messages:
            self._log(msg)

    def _metric_card(self, parent, title: str, column: int) -> tuple[ctk.CTkLabel, ctk.CTkLabel]:
        card = ctk.CTkFrame(parent, fg_color=PANEL, corner_radius=10)
        card.grid(row=0, column=column, sticky="nsew", padx=(0 if column == 0 else 6, 0 if column == 2 else 6))
        ctk.CTkLabel(
            card,
            text=title,
            text_color="#A1A1AA",
            font=ctk.CTkFont(size=11, weight="bold"),
            anchor="w",
        ).pack(anchor="w", padx=12, pady=(8, 0))
        main = ctk.CTkLabel(
            card,
            text="",
            font=ctk.CTkFont(size=14, weight="bold"),
            anchor="w",
            justify="left",
        )
        main.pack(anchor="w", padx=12, pady=(2, 0))
        sub = ctk.CTkLabel(
            card,
            text="",
            font=HW_FONT,
            text_color="#D4D4D8",
            anchor="w",
            justify="left",
        )
        sub.pack(anchor="w", padx=12, pady=(0, 10))
        return main, sub

    def _refresh_hardware(self) -> None:
        if not self.winfo_exists():
            return
        try:
            view = format_runtime_view(self._hw_sampler.sample(), self.device)
            self._hw_badge.configure(text=view.badge)
            self._hw_gpu_main.configure(text=view.gpu_main)
            self._hw_gpu_sub.configure(text=view.gpu_sub)
            self._hw_cpu_main.configure(text=view.cpu_main)
            self._hw_cpu_sub.configure(text=view.cpu_sub)
            self._hw_ram_main.configure(text=view.ram_main)
            self._hw_ram_sub.configure(text=view.ram_sub)
        except Exception:
            pass
        if self.winfo_exists():
            self._hw_job = self.after(1500, self._refresh_hardware)

    def _build_files_panel(self, parent) -> None:
        left = ctk.CTkFrame(parent, fg_color=PANEL, corner_radius=12)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=4)

        ctk.CTkLabel(
            left,
            text="Arquivos",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).pack(anchor="w", padx=14, pady=(12, 6))

        self.drop_zone = ctk.CTkFrame(
            left,
            height=130,
            corner_radius=10,
            border_width=2,
            border_color=ACCENT,
            fg_color="#16181E",
        )
        self.drop_zone.pack(fill="x", padx=14, pady=(0, 8))
        self.drop_zone.pack_propagate(False)
        self.drop_label = ctk.CTkLabel(
            self.drop_zone,
            text="Arraste e solte áudio/vídeo aqui\n.mp4  .mkv  .mov  .mp3  .wav  .m4a",
            text_color="#D4D4D8",
            justify="center",
        )
        self.drop_label.place(relx=0.5, rely=0.5, anchor="center")

        btns = ctk.CTkFrame(left, fg_color="transparent")
        btns.pack(fill="x", padx=14, pady=(0, 8))
        ctk.CTkButton(btns, text="Selecionar arquivos…", command=self._browse, width=180).pack(
            side="left"
        )
        ctk.CTkButton(
            btns,
            text="Limpar fila",
            command=self._clear_files,
            width=110,
            fg_color="#3F3F46",
            hover_color="#52525B",
        ).pack(side="right")

        list_holder = ctk.CTkFrame(left, fg_color="transparent")
        list_holder.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.file_list = ctk.CTkScrollableFrame(list_holder, fg_color="transparent")
        self.file_list.pack(fill="both", expand=True)
        self._list_tail = ctk.CTkFrame(self.file_list, fg_color="transparent", height=64)
        self._list_tail.pack(fill="x")
        self._list_tail.pack_propagate(False)

        self.start_btn = ctk.CTkButton(
            list_holder,
            text=START_LABEL,
            height=44,
            font=ctk.CTkFont(size=15, weight="bold"),
            fg_color=ACCENT,
            command=self._start,
        )
        self.start_btn.place(relx=0.5, rely=1.0, y=-6, anchor="s", relwidth=0.86)
        self.start_btn.lift()
        self.start_btn.bind("<MouseWheel>", self._on_files_mousewheel)

    def _on_files_mousewheel(self, event) -> None:
        """A roda do mouse no botão continua rolando a fila que passa por baixo."""
        try:
            self.file_list._parent_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")  # type: ignore[attr-defined]
        except Exception:
            pass

    def _build_params_panel(self, parent) -> None:
        right = ctk.CTkScrollableFrame(parent, fg_color=PANEL, corner_radius=12)
        right.grid(row=0, column=1, sticky="nsew", padx=(10, 0), pady=4)
        cfg = load_config()

        ctk.CTkLabel(
            right,
            text="Parâmetros WhisperX",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).pack(anchor="w", padx=12, pady=(12, 8))

        add_param_label(right, "Modelo", PARAM_HELP["model"])
        self.model_combo = make_readonly_combo(
            right, MODELS, cfg.get("last_model") or "small"
        )
        self.model_combo.pack(fill="x", padx=12, pady=(0, 10))

        add_param_label(right, "Idioma", PARAM_HELP["language"])
        lang_values = [_language_label(code, name) for code, name in LANGUAGES]
        last_lang = cfg.get("last_language") or "auto"
        match = next(
            (lbl for lbl, (code, _) in zip(lang_values, LANGUAGES) if code == last_lang),
            lang_values[0],
        )
        self.lang_combo = make_readonly_combo(right, lang_values, match)
        self.lang_combo.pack(fill="x", padx=12, pady=(0, 10))

        add_param_label(right, "Tamanho do lote", PARAM_HELP["batch"])
        batch_row = ctk.CTkFrame(right, fg_color="transparent")
        batch_row.pack(fill="x", padx=12, pady=(0, 10))
        self.batch_label = ctk.CTkLabel(batch_row, text=str(self.batch_var.get()), width=36)
        self.batch_label.pack(side="right")
        self.batch_slider = ctk.CTkSlider(
            batch_row,
            from_=1,
            to=32,
            number_of_steps=31,
            command=self._on_batch_slide,
        )
        self.batch_slider.set(self.batch_var.get())
        self.batch_slider.pack(side="left", fill="x", expand=True, padx=(0, 8))

        add_param_label(right, "Precisão numérica", PARAM_HELP["compute"])
        radios = ctk.CTkFrame(right, fg_color="transparent")
        radios.pack(fill="x", padx=12, pady=(0, 10))
        for value, hint in (
            ("float16", "GPUs modernas"),
            ("int8", "CPU / pouca VRAM"),
            ("float32", "máxima precisão"),
        ):
            ctk.CTkRadioButton(
                radios,
                text=f"{value}  ({hint})",
                variable=self.compute_var,
                value=value,
            ).pack(anchor="w", pady=2)

        add_param_label(right, "Faixas de áudio", PARAM_HELP["mix"])
        self.mix_chk = ctk.CTkCheckBox(
            right,
            text="Misturar faixas de áudio",
            variable=self.mix_enabled,
            command=self._toggle_mix,
        )
        self.mix_chk.pack(anchor="w", padx=12, pady=(0, 6))
        add_param_label(right, "Quais faixas misturar", PARAM_HELP["mix_scope"])
        mix_values = [label for _, label in MIX_SCOPE_LABELS]
        mix_label = next(
            (lbl for code, lbl in MIX_SCOPE_LABELS if code == self._saved_mix_scope),
            mix_values[0],
        )
        self.mix_combo = make_readonly_combo(right, mix_values, mix_label)
        self.mix_combo.pack(fill="x", padx=12, pady=(0, 10))
        self._toggle_mix()

        add_param_label(right, "Token Hugging Face", PARAM_HELP["token"])
        self.token_entry = ctk.CTkEntry(right, placeholder_text="hf_...", show="*", height=34)
        self.token_entry.pack(fill="x", padx=12, pady=(0, 10))
        token = hf_token()
        if token:
            self.token_entry.insert(0, token)
        self.token_entry.bind("<KeyRelease>", lambda _e: self._sync_token_dependent())
        self.token_entry.bind("<<Paste>>", lambda _e: self.after(20, self._sync_token_dependent))
        self.token_entry.bind("<FocusOut>", lambda _e: self._sync_token_dependent())

        add_param_label(right, "Diarização", PARAM_HELP["diarize"])
        self.diarize_chk = ctk.CTkCheckBox(
            right,
            text="Identificar locutores",
            variable=self.diarize_var,
            command=self._toggle_diarize,
        )
        self.diarize_chk.pack(anchor="w", padx=12, pady=(0, 8))

        add_param_label(right, "Número de locutores", PARAM_HELP["speakers"])
        speakers = ctk.CTkFrame(right, fg_color="transparent")
        speakers.pack(fill="x", padx=12, pady=(0, 8))
        min_col = ctk.CTkFrame(speakers, fg_color="transparent")
        min_col.pack(side="left", fill="x", expand=True)
        ctk.CTkLabel(min_col, text="Mínimo", text_color="#A1A1AA").pack(anchor="w")
        self.min_speakers = ctk.CTkEntry(min_col, width=90)
        self.min_speakers.pack(anchor="w", pady=(2, 0))
        max_col = ctk.CTkFrame(speakers, fg_color="transparent")
        max_col.pack(side="left", fill="x", expand=True, padx=(16, 0))
        ctk.CTkLabel(max_col, text="Máximo", text_color="#A1A1AA").pack(anchor="w")
        self.max_speakers = ctk.CTkEntry(max_col, width=90)
        self.max_speakers.pack(anchor="w", pady=(2, 0))
        bind_positive_int(self, self.min_speakers)
        bind_positive_int(self, self.max_speakers)

        add_param_label(right, "Formato de saída", PARAM_HELP["formats"])
        formats = ctk.CTkFrame(right, fg_color="transparent")
        formats.pack(fill="x", padx=12, pady=(0, 16))
        for fmt in OUTPUT_FORMATS:
            ctk.CTkCheckBox(formats, text=f".{fmt}", variable=self.format_vars[fmt]).pack(
                side="left", padx=(0, 8)
            )

        self._sync_token_dependent()

    def _on_batch_slide(self, value: float) -> None:
        ivalue = int(round(value))
        self.batch_var.set(ivalue)
        self.batch_label.configure(text=str(ivalue))

    def _token_present(self) -> bool:
        return bool(self.token_entry.get().strip())

    def _sync_token_dependent(self) -> None:
        has_token = self._token_present()
        try:
            self.diarize_chk.configure(state="normal" if has_token else "disabled")
        except Exception:
            pass
        if not has_token:
            self.diarize_var.set(False)
            self.min_speakers.configure(state="disabled")
            self.max_speakers.configure(state="disabled")
            return
        self.diarize_var.set(self._want_diarize)
        self._apply_speaker_state()

    def _toggle_mix(self) -> None:
        try:
            self.mix_combo.configure(state="readonly" if self.mix_enabled.get() else "disabled")
        except Exception:
            pass

    def _toggle_diarize(self) -> None:
        if self._token_present():
            self._want_diarize = bool(self.diarize_var.get())
        self._apply_speaker_state()

    def _apply_speaker_state(self) -> None:
        enabled = self._token_present() and bool(self.diarize_var.get())
        state = "normal" if enabled else "disabled"
        self.min_speakers.configure(state=state)
        self.max_speakers.configure(state=state)

    def _bind_dnd(self) -> None:
        root = self.winfo_toplevel()
        if not getattr(root, "_dnd_enabled", False):
            self.drop_label.configure(
                text="Drag & Drop indisponível\nUse o botão para selecionar arquivos"
            )
            return
        try:
            from tkinterdnd2 import DND_FILES

            for widget in (self.drop_zone, self.drop_label, self):
                widget.drop_target_register(DND_FILES)
                widget.dnd_bind("<<Drop>>", self._on_drop)
        except Exception:
            self.drop_label.configure(
                text="Drag & Drop indisponível\nUse o botão para selecionar arquivos"
            )

    def _on_drop(self, event) -> None:
        try:
            raw_items = self.tk.splitlist(event.data)
        except Exception:
            raw_items = [event.data]
        paths = [_normalize_dropped_path(item) for item in raw_items]
        self._add_files(paths)

    def _browse(self) -> None:
        selected = filedialog.askopenfilenames(
            title="Selecionar áudio ou vídeo",
            filetypes=[
                ("Mídia", "*.mp4 *.mkv *.mov *.mp3 *.wav *.m4a"),
                ("Todos", "*.*"),
            ],
        )
        if selected:
            self._add_files(list(selected))

    def _add_files(self, paths: list[str]) -> None:
        added = 0
        for raw in paths:
            path = str(Path(raw).expanduser())
            suffix = Path(path).suffix.lower()
            if suffix not in ALLOWED_EXT:
                self._log(f"Ignorado (extensão não suportada): {path}")
                continue
            if not Path(path).is_file():
                self._log(f"Arquivo não encontrado: {path}")
                continue
            if path in self.files:
                continue
            self.files.append(path)
            self._append_file_row(path)
            added += 1
        if added:
            self._log(f"{added} arquivo(s) na fila. Total: {len(self.files)}")
            self._refresh_job_stats()

    def _append_file_row(self, path: str) -> None:
        row = ctk.CTkFrame(self.file_list, fg_color="#16181E", corner_radius=8)
        row.pack(fill="x", pady=3, padx=4, before=self._list_tail)
        ctk.CTkLabel(row, text=Path(path).name, anchor="w").pack(
            side="left", fill="x", expand=True, padx=8, pady=6
        )
        ctk.CTkButton(
            row,
            text="×",
            width=28,
            height=28,
            fg_color="#3F3F46",
            command=lambda p=path: self._remove_file(p),
        ).pack(side="right", padx=6)
        self._file_rows[path] = row

    def _remove_file(self, path: str) -> None:
        if path in self.files:
            self.files.remove(path)
        row = self._file_rows.pop(path, None)
        if row is not None:
            row.destroy()
        self._refresh_job_stats()

    def _clear_files(self) -> None:
        self.files.clear()
        for row in self._file_rows.values():
            row.destroy()
        self._file_rows.clear()
        self._refresh_job_stats()

    def _selected_mix_audio(self) -> str:
        if not self.mix_enabled.get():
            return "off"
        raw = self.mix_combo.get()
        for code, label in MIX_SCOPE_LABELS:
            if raw == label:
                return code
        return "first_two"

    def _selected_language(self) -> str | None:
        raw = self.lang_combo.get()
        for code, name in LANGUAGES:
            if raw == _language_label(code, name):
                return None if code == "auto" else code
        return None

    def _optional_int(self, entry: ctk.CTkEntry) -> int | None:
        text = entry.get().strip().lower()
        if not text or text == "auto":
            return None
        try:
            value = int(text)
        except ValueError:
            raise ValueError(f"Valor numérico inválido: '{text}'") from None
        if value < 1:
            raise ValueError("Mínimo e máximo de locutores devem ser números inteiros a partir de 1.")
        return value

    def _start(self) -> None:
        if self._busy:
            return
        if not self.files:
            messagebox.showwarning("Fila vazia", "Adicione pelo menos um arquivo de áudio ou vídeo.")
            return

        formats = [fmt for fmt, var in self.format_vars.items() if var.get()]
        if not formats:
            messagebox.showwarning("Formato", "Selecione ao menos um formato de saída.")
            return

        token = self.token_entry.get().strip()
        diarize = bool(self.diarize_var.get())
        if diarize and not token:
            messagebox.showerror(
                "Token Hugging Face",
                "A identificação de locutores exige um token Hugging Face.\n"
                "1. Token (Read): https://huggingface.co/settings/tokens\n"
                "2. Aceite o modelo na MESMA conta:\n"
                "   https://huggingface.co/pyannote/speaker-diarization-community-1",
            )
            return

        try:
            min_spk = self._optional_int(self.min_speakers) if diarize else None
            max_spk = self._optional_int(self.max_speakers) if diarize else None
        except ValueError as exc:
            messagebox.showerror("Locutores", str(exc))
            return
        if min_spk is not None and max_spk is not None and min_spk > max_spk:
            messagebox.showerror(
                "Locutores",
                "O mínimo de locutores não pode ser maior que o máximo.",
            )
            return

        options = JobOptions(
            model=self.model_combo.get(),
            language=self._selected_language(),
            batch_size=int(self.batch_var.get()),
            compute_type=self.compute_var.get(),
            device=self.device,
            diarize=diarize,
            min_speakers=min_spk,
            max_speakers=max_spk,
            hf_token=token,
            output_formats=formats,
            mix_audio=self._selected_mix_audio(),
        )
        save_config(
            hf_token=token,
            last_model=options.model,
            last_language=options.language or "auto",
            last_compute_type=options.compute_type,
            last_batch_size=options.batch_size,
            diarize=diarize,
            mix_audio=options.mix_audio,
        )

        files = list(self.files)
        self._busy = True
        self._oom_alerted = False
        self._job_started_at = time.monotonic()
        self._job_files_total = len(files)
        self._job_files_done = 0
        self._job_progress = 0.0
        self._eta_smooth = None
        self.start_btn.configure(state="disabled", text=START_BUSY_LABEL)
        self.progress.set(0)
        self._refresh_job_stats()
        self._schedule_stats_tick()
        self._log("Iniciando pipeline WhisperX em thread separada…")
        threading.Thread(
            target=run_jobs,
            args=(files, options, self.event_queue),
            daemon=True,
        ).start()

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
            self._log(event.message)
        elif event.kind == "stage":
            self.stage_label.configure(text=event.stage or event.message or "…")
            if event.progress:
                self._job_progress = max(0.0, min(1.0, event.progress))
                self.progress.set(self._job_progress)
            self._refresh_job_stats()
        elif event.kind == "warning":
            self._log("AVISO: " + event.message)
        elif event.kind == "oom_retry":
            self._log("VRAM: " + event.message)
            if not self._oom_alerted:
                self._oom_alerted = True
                messagebox.showwarning("VRAM (Out of Memory)", event.message)
        elif event.kind == "file_done":
            self._job_files_done = min(self._job_files_total, self._job_files_done + 1)
            self._log(event.message)
            self._refresh_job_stats()
        elif event.kind == "error":
            self._log("ERRO: " + event.message)
            tb = ""
            if isinstance(event.payload, dict):
                tb = event.payload.get("traceback") or ""
            if tb:
                self._log(tb)
            messagebox.showerror("Processamento", event.message)
        elif event.kind == "done":
            success = "conclu" in (event.message or "").lower()
            if success and self._job_started_at is not None:
                self._last_duration_sec = max(0.0, time.monotonic() - self._job_started_at)
                save_config(
                    last_job_seconds=round(self._last_duration_sec, 1),
                )
            self._busy = False
            self._job_started_at = None
            self._eta_smooth = None
            self.start_btn.configure(state="normal", text=START_LABEL)
            self.progress.set(1.0 if success else self.progress.get())
            self.stage_label.configure(text=event.message or "Concluído")
            self._log(event.message)
            self._refresh_job_stats()

    def _schedule_stats_tick(self) -> None:
        if self._stats_job is not None:
            try:
                self.after_cancel(self._stats_job)
            except Exception:
                pass
        self._stats_job = self.after(400, self._on_stats_tick)

    def _on_stats_tick(self) -> None:
        self._stats_job = None
        if not self.winfo_exists():
            return
        self._refresh_job_stats()
        if self._busy:
            self._schedule_stats_tick()

    def _refresh_job_stats(self) -> None:
        if not hasattr(self, "stats_label"):
            return
        parts: list[str] = []
        if self._busy:
            total = max(self._job_files_total, 1)
            current = min(max(self._job_files_done + 1, 1), total)
            parts.append(f"Arquivo {current}/{total}")
            elapsed = None
            if self._job_started_at is not None:
                elapsed = time.monotonic() - self._job_started_at
                parts.append(f"Decorrido {_format_duration(elapsed)}")
            eta = self._estimate_remaining(elapsed)
            if eta is None:
                parts.append("Restante calculando…")
            else:
                parts.append(f"Restante ~{_format_duration(eta)}")
            if self._last_duration_sec:
                parts.append(f"Última {_format_duration(self._last_duration_sec)}")
        else:
            n = len(self.files)
            if n == 1:
                parts.append("1 na fila")
            else:
                parts.append(f"{n} na fila")
            if self._last_duration_sec:
                parts.append(f"Última {_format_duration(self._last_duration_sec)}")
            else:
                parts.append("Última —")
        self.stats_label.configure(text="    ·    ".join(parts))

    def _estimate_remaining(self, elapsed: float | None) -> float | None:
        if elapsed is None or elapsed < 8:
            return None
        progress = self._job_progress
        if progress < 0.08:
            return None
        raw = elapsed * (1.0 - progress) / max(progress, 1e-6)
        if self._eta_smooth is None:
            self._eta_smooth = raw
        else:
            self._eta_smooth = 0.28 * raw + 0.72 * self._eta_smooth
        return max(0.0, self._eta_smooth)

    def _log(self, message: str) -> None:
        self.log_box.insert("end", message.rstrip() + "\n")
        self.log_box.see("end")


def _normalize_dropped_path(item: str) -> str:
    text = item.strip().strip('"')
    if text.lower().startswith("file:"):
        parsed = urlparse(text)
        path = unquote(parsed.path)
        if sys.platform == "win32" and path.startswith("/") and len(path) > 2 and path[2] == ":":
            path = path[1:]
        return path
    return text
