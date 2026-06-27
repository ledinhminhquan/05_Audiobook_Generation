# 🎧 Audiobook Generation System

> Turn long documents (EPUB / PDF / TXT / Markdown) into mastered, chaptered
> **audiobooks** (`.wav` / `.mp3` / `.m4b` + read-along `.srt`) with a **trainable
> text-normalization model**, a **neural TTS backend**, and an **agentic** pipeline.

**NLP in Industry — Final Assignment.** Author: **Le Dinh Minh Quan** (Student `23127460`).
Reference inspiration: [denizsafak/abogen](https://github.com/denizsafak/abogen).

The hard, NLP-heavy part of audiobook production is **text normalization** — a naive
TTS reads `$5.2M`, `1984`, `Dr.`, `Chapter IV`, `3/4`, `9:45 AM` wrong. So the **trainable
core** is a Text-Normalization (TN) seq2seq model that converts *written → spoken* form;
the audio is produced by a pretrained neural TTS (**SpeechT5**); and a deterministic
**agent** orchestrates the whole `document → audiobook` pipeline.

---

## ✅ How this repo meets every assignment requirement

| Requirement | Where it is delivered |
|---|---|
| **Business problem** | [`docs/problem_definition.md`](docs/problem_definition.md) — cost/accessibility/scale, business + technical metrics |
| **Dev infra & tooling** | `src/` package, `pyproject.toml`, `requirements*.txt`, `Makefile`, Docker, CI, modular structure |
| **Data management** | Synthetic TN corpus generator ([`data/tn_corpus.py`](src/audiobook_ai/data/tn_corpus.py)); leakage-free splits; [`docs/data_description.md`](docs/data_description.md), [`docs/data_card.md`](docs/data_card.md) |
| **Model selection & optimization** | ByT5-small TN model + **rule baseline** to beat; tuning; per-class error analysis; [`docs/model_selection.md`](docs/model_selection.md) |
| **Deployment** | FastAPI REST + Gradio UI + CLI + batch + Docker + HF Space; [`docs/deployment.md`](docs/deployment.md) |
| **Agentic AI** | Deterministic FSM with **4 decision points** + optional LLM brain; [`docs/agent_architecture.md`](docs/agent_architecture.md) |
| **Continual learning & monitoring** | [`docs/continual_learning_monitoring.md`](docs/continual_learning_monitoring.md) + [`monitoring/drift_report.py`](src/audiobook_ai/monitoring/drift_report.py) |
| **Privacy & robustness** | [`docs/privacy_robustness.md`](docs/privacy_robustness.md) |
| **Project management** | [`docs/project_plan.md`](docs/project_plan.md) |
| **Ethics & responsible AI** | [`docs/ethics_statement.md`](docs/ethics_statement.md) |
| **Report + slides** | auto-generated `report.pdf` + `slides.pptx` (`autopilot`) |

---

## 🏗️ Pipeline

```
document (epub/pdf/txt/md)
   │  parse (ebooklib / PyMuPDF / pdfplumber)        ── D1 parse-quality routing
   ▼
chapters → segments (pysbd + TTS chunking; narration|dialogue|heading|skippable)
   │  NORMALIZE  (TRAINED ByT5-small, per segment)   ── D2 normalization-confidence
   ▼
voice routing (narrator/dialogue/heading)            ── D4 voice routing
   │  SYNTHESIZE (SpeechT5 / Kokoro / Parler)
   ▼
audio QA (loudness/peak/duration/silence)            ── D3 re-synthesis gate
   │  STITCH + master to −18 LUFS / −3 dBTP (ACX)
   ▼
export  .wav / .mp3 / .m4b (+chapters) / .srt  +  manifest.json
```

## 📦 Models & data (all ids VERIFIED on the HF Hub)

| Role | Id | License |
|---|---|---|
| **TN model (trained)** | `google/byt5-small` (fallback `google-t5/t5-small`) | Apache-2.0 |
| **TN data** | synthetic generator (primary) + optional `DigitalUmuganda/…Eng_Fra` eval | code MIT |
| TTS (primary) | `microsoft/speecht5_tts` + `microsoft/speecht5_hifigan` | MIT |
| Speaker voices | `Matthijs/cmu-arctic-xvectors` | MIT |
| TTS (optional) | `hexgrad/Kokoro-82M`, `parler-tts/parler-tts-mini-v1` | Apache-2.0 |
| Baseline | regex + `num2words` rule normalizer | — |

> Non-commercial models (`coqui/XTTS-v2`, `facebook/mms-tts-eng`) are **excluded** from the default path.

## 🗂️ Repository layout

```
05_Audiobook_Generation/
├── src/audiobook_ai/
│   ├── config.py  cli.py  logging_utils.py
│   ├── data/         document.py · tn_corpus.py · dataset.py · samples.py · download_dataset.py
│   ├── models/       expanders.py · baseline_rules.py · normalizer.py · model_registry.py
│   ├── synthesis/    tts_backend.py · voices.py · stitch.py · subtitles.py
│   ├── training/     train_normalizer.py · evaluate.py · tune.py
│   ├── agent/        state.py · policy.py · tools.py · llm_orchestrator.py · narrator_agent.py
│   ├── api/          schemas.py · dependencies.py · main.py · ui.py · app_combined.py
│   ├── analysis/ autoreport/ monitoring/ automation/ grading/
├── configs/ · data/ · models/ · tests/ · docs/ · notebooks/ · app/ · deploy/ · sample_data/
├── Dockerfile · docker-compose.yml · Makefile · pyproject.toml · requirements*.txt
```

---

## 🚀 Quickstart

```bash
# 1) install (core is tiny; add extras for ML/TTS/serving/reports)
pip install -e ".[ml,tts,api,report]"

# 2) build the synthetic TN corpus
audiobook-ai data --task corpus

# 3) see the baseline you must beat (overall + the hard ambiguous slice)
audiobook-ai evaluate --which test
audiobook-ai evaluate --which hard

# 4) make an audiobook from the built-in sample book (offline-friendly)
audiobook-ai demo-agent                      # uses SpeechT5 if available
audiobook-ai synthesize --file sample_data/sample_book.txt --title "Demo"

# 5) normalize-only preview (fast, no audio)
audiobook-ai normalize --text "He paid \$5.2M in 1984; Dr. Vance lives on Oak Dr."
```

### Train the Text-Normalization model
```bash
audiobook-ai --config configs/train.yaml train      # ByT5-small (auto-resumes)
audiobook-ai evaluate --which test                  # neural vs baseline, per-class
```
On Colab/GPU use the notebook (below) — it auto-profiles H100/A100/L4/T4.

### Serve
```bash
audiobook-ai serve --ui --port 7860        # FastAPI REST + Gradio UI at /ui
# POST /normalize · POST /synthesize · POST /synthesize/file · GET /artifacts/...
```

### One-button report + slides + self-grade
```bash
audiobook-ai autopilot --no-train          # eval → analysis → report.pdf + slides.pptx + bundle
audiobook-ai grade                          # rubric completeness check
```

---

## 🤖 The agent (mandatory agentic component)

A **deterministic finite-state machine** over a versioned job context, with **four
decision points** that act on intermediate outputs, plus an *optional* LLM brain
(`anthropic`) that validates its output and **falls back to rules** (default = zero paid API):

- **D1** parse-quality routing (structured / assisted / degraded)
- **D2** normalization-confidence escalation (low confidence → flag / LLM disambiguation)
- **D3** audio-QA re-synthesis gate (loudness/peak/duration/silence → bounded re-synth)
- **D4** voice routing (narrator / dialogue / heading)

Every step is timed + traced and a full `manifest.json` is written. See
[`docs/agent_architecture.md`](docs/agent_architecture.md).

## ☁️ Colab / H100 training

Open [`notebooks/Audiobook_AI_Colab_Training_H100_AUTOPILOT.ipynb`](notebooks/Audiobook_AI_Colab_Training_H100_AUTOPILOT.ipynb)
— it mounts Drive, installs Colab-safe deps (never touches torch), **auto-profiles the GPU**
(H100/A100/L4/T4 → batch + precision, effective batch held constant), trains resume-safely,
evaluates vs the baseline, runs the agent, and generates the report/slides. Step-by-step:
[`notebooks/COLAB_GUIDE.md`](notebooks/COLAB_GUIDE.md).

## 🧪 Tests

```bash
pytest -q        # CPU-only, no model/network downloads (rule normalizer + placeholder TTS)
```

## 📚 Docs index

`docs/`: problem_definition · data_description · data_card · model_selection ·
tn_quality_evaluation · agent_architecture · deployment · continual_learning_monitoring ·
privacy_robustness · project_plan · ethics_statement · architecture · model_card ·
slide_deck_outline · DESIGN_BRIEF.

## 📝 License

MIT — see [`LICENSE`](LICENSE). Pretrained models keep their own licenses (table above).
Converting a copyrighted book to audio is a derivative work — verify you hold the rights.
