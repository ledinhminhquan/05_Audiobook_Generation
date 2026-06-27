"""Gradio UI for the Audiobook Generation system.

Paste text or upload a document, choose a voice/backend, and get the normalized
script, an audio preview, the agent's decision log, and download links. ``gradio``
is imported lazily so the package stays importable without it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from ..config import AppConfig
from ..logging_utils import get_logger

logger = get_logger(__name__)


def build_ui(cfg: Optional[AppConfig] = None):
    import gradio as gr  # lazy
    from ..agent.narrator_agent import NarratorAgent

    cfg = cfg or AppConfig()
    agent = NarratorAgent(cfg, load_model=True)

    def normalize_only(text: str):
        if not text.strip():
            return "", "—"
        job = agent.normalize_preview(text)
        full = "\n".join(f"[{s.kind}] {s.normalized}" for s in job.segments)
        dec = "\n".join(f"{d.id} {d.name}: {d.branch} ({d.detail})" for d in job.decisions)
        return full, dec

    def make_audiobook(text: str, file, title: str, backend: str):
        path = None
        if file is not None:
            path = file.name if hasattr(file, "name") else file
        if not (text and text.strip()) and not path:
            return None, "Provide text or a file.", "—", "{}"
        if backend and backend != "auto":
            agent._tts_backend = backend
            agent._tts = None
        job = agent.process(path=path, text=(text if not path else None),
                            title=title or "Audiobook", synth=True)
        sd = job.to_dict()
        audio = sd["outputs"].get("wav")
        norm = "\n".join(f"[{s['kind']}] {s['normalized']}" for s in sd["segments"])
        dec = "\n".join(f"{d['id']} {d['name']}: {d['branch']} — {d['detail']}" for d in sd["decisions"])
        meta = json.dumps({"status": sd["status"], "metrics": sd["metrics"],
                           "outputs": sd["outputs"]}, indent=2)
        return audio, norm, dec, meta

    with gr.Blocks(title=cfg.project_title) as demo:
        gr.Markdown(f"# 🎧 {cfg.project_title}\nTurn documents into narrated audiobooks "
                    "(trained text-normalization + neural TTS + an agentic pipeline).")
        with gr.Tab("Normalize (preview)"):
            tn_in = gr.Textbox(label="Written text", lines=4,
                               value="He paid $5.2M in 1984; Dr. Vance lives on Oak Dr.")
            tn_btn = gr.Button("Normalize", variant="primary")
            tn_out = gr.Textbox(label="Spoken form (per segment)", lines=6)
            tn_dec = gr.Textbox(label="Agent decisions", lines=3)
            tn_btn.click(normalize_only, [tn_in], [tn_out, tn_dec])
        with gr.Tab("Make audiobook"):
            with gr.Row():
                txt = gr.Textbox(label="Paste text", lines=6)
                up = gr.File(label="…or upload (epub/pdf/txt/md)",
                             file_types=[".epub", ".pdf", ".txt", ".md"])
            with gr.Row():
                title = gr.Textbox(label="Title", value="My Audiobook")
                backend = gr.Dropdown(["auto", "speecht5", "kokoro", "parler", "pyttsx3"],
                                      value="auto", label="TTS backend")
            go = gr.Button("Generate audiobook", variant="primary")
            audio = gr.Audio(label="Preview", type="filepath")
            norm = gr.Textbox(label="Normalized script", lines=6)
            dec = gr.Textbox(label="Agent decision log", lines=4)
            meta = gr.Code(label="Job manifest", language="json")
            go.click(make_audiobook, [txt, up, title, backend], [audio, norm, dec, meta])
    return demo


def launch(server_name: str = "0.0.0.0", server_port: int = 7860, share: bool = False) -> None:
    build_ui().launch(server_name=server_name, server_port=server_port, share=share)


__all__ = ["build_ui", "launch"]
