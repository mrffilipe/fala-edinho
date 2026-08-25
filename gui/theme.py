"""Aparência CustomTkinter e janela raiz com suporte a Drag & Drop."""

from __future__ import annotations

import customtkinter as ctk

ACCENT = "#3B82F6"
SURFACE = "#1B1D24"
PANEL = "#242731"
OK = "#22C55E"


def apply_theme() -> None:
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    try:
        ctk.set_widget_scaling(1.0)
    except Exception:
        pass


def create_root():
    """CTk com mixin TkinterDnD. Sem o pacote, cai para uma janela CTk simples."""
    apply_theme()
    try:
        from tkinterdnd2 import TkinterDnD

        class App(ctk.CTk, TkinterDnD.DnDWrapper):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.TkdndVersion = TkinterDnD._require(self)

        root = App()
        root._dnd_enabled = True  # type: ignore[attr-defined]
        return root
    except Exception:
        root = ctk.CTk()
        root._dnd_enabled = False  # type: ignore[attr-defined]
        return root
