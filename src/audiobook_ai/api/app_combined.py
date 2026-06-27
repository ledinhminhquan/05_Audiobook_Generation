"""Combined ASGI app: FastAPI REST + the Gradio UI mounted at ``/ui``.

    uvicorn audiobook_ai.api.app_combined:app --host 0.0.0.0 --port 7860
"""

from __future__ import annotations

from ..logging_utils import get_logger
from .main import app

logger = get_logger(__name__)

try:
    import gradio as gr  # lazy/optional
    from .ui import build_ui
    app = gr.mount_gradio_app(app, build_ui(), path="/ui")
    logger.info("Mounted Gradio UI at /ui")
except Exception as exc:  # pragma: no cover
    logger.info("Gradio UI not mounted (%s); REST API still available", exc)

__all__ = ["app"]
