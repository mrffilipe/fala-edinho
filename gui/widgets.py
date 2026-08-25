"""Controles reutilizáveis da GUI: tooltip, select somente leitura, inteiros positivos."""

from __future__ import annotations

import tkinter as tk
import customtkinter as ctk

from gui.theme import ACCENT


class InfoTip:
    """Ícone “i” com dica ao passar o mouse ou clicar."""

    def __init__(self, parent, text: str) -> None:
        self.text = text
        self._tip: tk.Toplevel | None = None
        self._hide_job: str | None = None
        self.button = ctk.CTkButton(
            parent,
            text="i",
            width=22,
            height=22,
            corner_radius=11,
            fg_color="#3F3F46",
            hover_color=ACCENT,
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self._show,
        )
        self.button.bind("<Enter>", self._on_enter)
        self.button.bind("<Leave>", self._on_leave)

    def pack(self, **kwargs):
        return self.button.pack(**kwargs)

    def _on_enter(self, _event=None) -> None:
        self._cancel_hide()
        self._show()

    def _on_leave(self, _event=None) -> None:
        self._hide_job = self.button.after(280, self._hide)

    def _cancel_hide(self) -> None:
        if self._hide_job is not None:
            try:
                self.button.after_cancel(self._hide_job)
            except Exception:
                pass
            self._hide_job = None

    def _show(self) -> None:
        if self._tip is not None:
            return
        try:
            x = self.button.winfo_rootx() + 8
            y = self.button.winfo_rooty() + self.button.winfo_height() + 6
        except Exception:
            return
        tip = tk.Toplevel(self.button)
        tip.wm_overrideredirect(True)
        tip.wm_geometry(f"+{int(x)}+{int(y)}")
        try:
            tip.attributes("-topmost", True)
        except Exception:
            pass
        frame = tk.Frame(tip, background="#1A1C24", highlightbackground="#3B82F6", highlightthickness=1)
        frame.pack()
        label = tk.Label(
            frame,
            text=self.text,
            justify="left",
            wraplength=360,
            background="#1A1C24",
            foreground="#E4E4E7",
            font=("Segoe UI", 9),
            padx=10,
            pady=8,
        )
        label.pack()
        self._tip = tip
        tip.bind("<Enter>", lambda _e: self._cancel_hide())
        tip.bind("<Leave>", self._on_leave)

    def _hide(self) -> None:
        self._hide_job = None
        if self._tip is not None:
            try:
                self._tip.destroy()
            except Exception:
                pass
            self._tip = None


def add_param_label(parent, text: str, help_text: str) -> None:
    row = ctk.CTkFrame(parent, fg_color="transparent")
    row.pack(anchor="w", fill="x", padx=12, pady=(0, 2))
    ctk.CTkLabel(row, text=text).pack(side="left")
    InfoTip(row, help_text).pack(side="left", padx=(6, 0))


def make_readonly_combo(parent, values: list[str], current: str, **kwargs) -> ctk.CTkComboBox:
    """Select fechado: só escolhe da lista, sem digitar."""
    box = ctk.CTkComboBox(parent, values=values, state="readonly", **kwargs)
    box.set(current)

    def on_click(_event=None):
        if str(box.cget("state")) == "disabled":
            return
        box._clicked()  # type: ignore[attr-defined]

    box.bind("<Button-1>", on_click)
    return box


def bind_positive_int(widget, entry: ctk.CTkEntry) -> None:
    """Aceita vazio (automático) ou inteiros ≥ 1. Bloqueia letras, zero e negativos."""

    def allowed(proposed: str) -> bool:
        if proposed == "":
            return True
        if not proposed.isdigit():
            return False
        return not proposed.startswith("0")

    vcmd = (widget.register(allowed), "%P")
    try:
        entry.configure(validate="key", validatecommand=vcmd)
    except Exception:
        pass
