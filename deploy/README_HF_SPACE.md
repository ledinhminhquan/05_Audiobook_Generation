# Deploying to a Hugging Face Space (Gradio)

The Gradio UI (`app/gradio_app.py`) runs the full agent: paste text / upload a
document → normalized script + audio + decision log + downloads. Default backend
is **SpeechT5** (MIT, CPU-able), so the demo runs on a free/CPU Space; a GPU Space
makes synthesis real-time.

## Option A — Gradio SDK Space (simplest)

1. Create a new Space → SDK **Gradio**.
2. Add these files at the repo root of the Space:
   - `app.py` →
     ```python
     from audiobook_ai.api.ui import build_ui
     demo = build_ui()
     ```
   - `requirements.txt` (copy `requirements_colab.txt` from this repo **plus** `torch`),
   - the `src/` folder (so `audiobook_ai` is importable), or add
     `pip install git+https://github.com/<you>/05_Audiobook_Generation` to requirements.
3. (Optional) Space **Secrets**: set `AUDIOBOOK_AI_LLM_API_KEY` to enable the LLM brain.
4. (Optional) Hardware: pick a **T4/A10 GPU** for fast synthesis; CPU works for the demo.

## Option B — Docker Space (REST API + UI)

1. Create a Space → SDK **Docker**; push this repo (it has a `Dockerfile`).
2. The image serves `audiobook_ai.api.app_combined:app` on port **7860**
   (REST API + Gradio UI mounted at `/ui`).
3. Set the Space port to 7860.

## Notes
- Pre-download models into the image / Space cache for fast cold starts
  (`microsoft/speecht5_tts`, `microsoft/speecht5_hifigan`, `Matthijs/cmu-arctic-xvectors`,
  and your trained TN model).
- To ship a **trained** normalizer, push it to the HF Hub and set
  `AUDIOBOOK_AI_MODEL_DIR` (or place it under `models/tn_normalizer/latest`).
- Keep only **permissive** TTS backends (SpeechT5 / Kokoro / Parler) in a public Space;
  XTTS / MMS-TTS are non-commercial.
