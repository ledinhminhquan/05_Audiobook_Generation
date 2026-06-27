"""Gradio entrypoint (used by the Hugging Face Space and `make ui`).

    python app/gradio_app.py        # local
    # or on a Space, this file is the app entrypoint.
"""

from __future__ import annotations

import os

from audiobook_ai.api.ui import build_ui

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    build_ui().launch(server_name="0.0.0.0", server_port=port)
