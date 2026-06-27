# Audiobook Generation System — API + UI image.
# CPU image by default (SpeechT5 runs on CPU for the demo). For GPU workers, base
# off an nvidia/cuda runtime and `pip install .[ml]` with a CUDA torch wheel.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/root/.cache/huggingface \
    AUDIOBOOK_AI_ARTIFACTS_DIR=/artifacts

# ffmpeg => mp3/m4b export; espeak-ng => pyttsx3 fallback voice
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg espeak-ng git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml requirements.txt README.md ./
COPY src ./src

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e ".[ml,tts,api,report]"

COPY configs ./configs
COPY docs ./docs

EXPOSE 7860
# Combined REST API + Gradio UI (UI mounted at /ui)
CMD ["uvicorn", "audiobook_ai.api.app_combined:app", "--host", "0.0.0.0", "--port", "7860"]
