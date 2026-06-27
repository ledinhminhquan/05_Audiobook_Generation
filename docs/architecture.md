# P05 Audiobook Generation — System Architecture

**Project:** Audiobook Generation System — turn long documents (EPUB/PDF/TXT/MD) into mastered, chaptered audiobooks (`.wav` / `.mp3` / `.m4b` + `.srt` subtitles).
**Author:** Le Dinh Minh Quan (23127460) — *NLP in Industry*, final assignment.
**Reference repo:** `github.com/denizsafak/abogen`.

---

## 1. Design Premise

A naive TTS reads `"$5.2M"`, `"1984"`, `"Dr."`, `"Chapter IV"`, `"3/4"`, and `"9:45 AM"` wrong. The genuinely hard, NLP-heavy part of audiobook production is **text normalization** (written → spoken). Therefore the *trainable* component is a **Text-Normalization (TN) seq2seq model**; audio is produced by a **pretrained neural TTS backend**; and a **deterministic agent (FSM)** orchestrates the whole document → audiobook pipeline.

This split drives the architecture: one ML-heavy module that we train and evaluate against a context-blind baseline, surrounded by robust, swappable, pretrained or rule-based components so the pipeline never hard-fails.

---

## 2. End-to-End Component Diagram

```
                          INPUT DOCUMENT  (EPUB / PDF / TXT / MD)
                                     │
                                     ▼
   ┌──────────────────────────────────────────────────────────────────────────┐
   │                    AGENT  (deterministic FSM + optional LLM brain)         │
   │   state.py · policy.py · tools.py · llm_orchestrator.py · narrator_agent.py │
   │   4 decision points (D1–D4) · ToolTrace timing · Decision audit log        │
   └──────────────────────────────────────────────────────────────────────────┘
        │
        ▼
   [PARSE]  data/document.py
   ebooklib (EPUB) · PyMuPDF+pdfplumber (PDF) · txt/md
   └─► (D1) parse_score ∈[0,1] = alpha-ratio + structure + segment-length sanity
            ├─ ≥0.85  structured
            ├─ 0.5–0.85 assisted
            └─ <0.5   degraded (flat-text path)
        │
        ▼
   [CHAPTER DETECT]  TOC / font-size / "Chapter N" regex
        │
        ▼
   [SEGMENT + CLASSIFY]  pysbd sentences + TTS-chunking
   label each segment: narration | dialogue | heading | skippable
        │
        ▼
   [NORMALIZE]  models/normalizer.py  ◄── TRAINED ByT5 (registry: latest pointer)
   per-segment written→spoken + confidence (length-norm sequence prob)
   └─► (D2) confidence < norm_confidence_min (0.55)
            ├─ flag segment
            ├─ optionally escalate → LLM brain (validates, falls back to rules)
            └─ record neural-vs-baseline disagreement
        │
        ▼
   [VOICE ROUTE]  synthesis/voices.py
   └─► (D4) heading/dialogue/narration → distinct x-vector indices (stable per book)
        │
        ▼
   [SYNTHESIZE]  synthesis/tts_backend.py  ◄── SpeechT5 + HiFi-GAN + x-vectors
        │
        ▼
   [AUDIO QA]  empty/NaN · duration ratio · peak/clipping · silence fraction
   └─► (D3) fail → bounded re-synth (max 2: reseed → split → fallback backend)
            then accept-best + flag
        │
        ▼
   [STITCH]  synthesis/stitch.py
   silence gaps · ACX master: −18 LUFS / −3 dBTP loudness
        │
        ▼
   [EXPORT]  synthesis/subtitles.py + stitch.py
   wav · mp3 · m4b (chapter markers) · srt · manifest.json
        │
        ▼
   ┌────────────────────────── SHARED ENTRY POINTS ──────────────────────────┐
   │  FastAPI (api/main.py) · Gradio UI (api/ui.py) · CLI (cli.py)            │
   │  all construct and drive the SAME agent over the SAME pipeline           │
   └─────────────────────────────────────────────────────────────────────────┘
```

**Data flow summary:** a single immutable artifact (text → segments → normalized segments → audio clips → mastered track) accumulates through the FSM. Every transition is timed (`ToolTrace`), every branch (D1–D4) is recorded (`Decision`), and the full record is serialized to `manifest.json` alongside the audio outputs.

---

## 3. Repository Module Map

The repo mirrors the P02/P03/P04 template. Package root: `src/audiobook_ai/`.

| Module | Responsibility |
|---|---|
| `config.py` | Central config dataclasses; resolves paths, env-vars, GPU profile, thresholds (e.g. `norm_confidence_min=0.55`, parse-score bands). |
| `cli.py` | Console-script `audiobook-ai`; subcommands (see §8). |
| `logging_utils.py` | Structured logging; PII-minimizing log policy. |
| **`data/`** | |
| `data/document.py` | Document parsing (EPUB via ebooklib; PDF via PyMuPDF + pdfplumber; txt/md); chapter detection; segmentation/classification helpers. |
| `data/tn_corpus.py` | **Synthetic TN corpus generator** — context-rich `(written, spoken)` pairs across 16 semiotic classes with injected ambiguity. |
| `data/dataset.py` | HF dataset construction, tokenization, leakage-free splits. |
| `data/samples.py` | Curated example inputs for demos/tests. |
| `data/download_dataset.py` | Optional real-data loader + optional eval sanity set. |
| **`models/`** | |
| `models/expanders.py` | Pure-python / num2words expanders (cardinals, ordinals, dates, money, etc.). |
| `models/baseline_rules.py` | **Context-blind rule normalizer** — ordered regex spans + abbreviation dict + expanders. The baseline to beat. |
| `models/normalizer.py` | Trained ByT5 inference wrapper; produces spoken text + per-segment confidence. |
| `models/model_registry.py` | Model registry: `tn_meta.json` + **latest pointer**, `repo@revision` pins; resolves which checkpoint inference loads. |
| **`synthesis/`** | |
| `synthesis/tts_backend.py` | Pluggable TTS backends (SpeechT5 primary; optional/fallback below); handles weights, sample rate, vocoder. |
| `synthesis/voices.py` | Voice routing → x-vector indices (narrator/dialogue/heading). |
| `synthesis/stitch.py` | Clip stitching, silence gaps, ACX loudness master, container export (wav/mp3/m4b). |
| `synthesis/subtitles.py` | `.srt` generation aligned to spoken segments. |
| **`training/`** | |
| `training/train_normalizer.py` | HF `Seq2SeqTrainer` loop (`predict_with_generate`, bf16/tf32, early stopping, resume). |
| `training/evaluate.py` | Sentence EM, macro/per-class accuracy, unrecoverable-error counting. |
| `training/tune.py` | Hyperparameter / config tuning. |
| **`agent/`** | |
| `agent/state.py` | FSM state, `ToolTrace`, `Decision` record types. |
| `agent/policy.py` | Decision logic for D1–D4 (thresholds, routing, escalation). |
| `agent/tools.py` | Tool wrappers (parse, normalize, synthesize, QA, stitch) the agent invokes. |
| `agent/llm_orchestrator.py` | Optional LLM brain (anthropic); validates and falls back to rules. **Off by default.** |
| `agent/narrator_agent.py` | Top-level agent that runs the document → audiobook FSM. |
| **`api/`** | |
| `api/schemas.py` | Pydantic request/response models. |
| `api/dependencies.py` | Shared dependency wiring (loads agent, registry, backends once). |
| `api/main.py` | FastAPI app: `/healthz`, `/readyz`, `/normalize`, `/synthesize`, `/synthesize/file`, `/artifacts`, `/download`. |
| `api/ui.py` | Gradio UI. |
| `api/app_combined.py` | Combined ASGI app mounting the UI at `/ui`. |
| `analysis/` · `autoreport/` · `monitoring/` · `automation/` · `grading/` | Error analysis, report/slide generation, monitoring, autopilot, grading utilities. |

Top-level: `configs/ data/ models/ tests/ docs/ notebooks/ app/ deploy/ Dockerfile docker-compose.yml Makefile pyproject.toml requirements*.txt README.md`.

---

## 4. Config, Env-Vars, and Artifact Wiring (Drive on Colab)

`config.py` is the single point where runtime context is resolved:

- **Paths.** Data, model, and output roots are resolved from config/env-vars. On Colab, artifacts (synthetic corpus, checkpoints, registry, exported audio) are pointed at a **mounted Google Drive** directory so training survives runtime resets and inference can pick up the latest checkpoint across sessions.
- **GPU profile.** The effective batch size is held at **256**; per-device batch / grad-accum / precision are selected from the detected accelerator (see §6).
- **Thresholds.** Decision-point cutoffs (`norm_confidence_min=0.55`, parse-score bands `0.85` / `0.5`, re-synth `max 2` attempts) are config values, not hard-coded constants.
- **LLM brain.** Enabled only when explicitly configured (anthropic credentials). Default is **OFF** → zero paid API, CPU-only, fully reproducible.

The **model registry** (`tn_meta.json` + latest pointer + `repo@revision` pins) is what couples *training output* to *inference input*: training writes a checkpoint and updates the registry; inference (`normalizer.py` via `model_registry.py`) reads the latest pointer to load the exact pinned revision. This is also the versioning mechanism for deployment.

---

## 5. Lazy-Import & Graceful-Degradation Design

Heavy dependencies (PyMuPDF, pdfplumber, torch/transformers, TTS weights, ffmpeg) are **lazily imported** at the point of use, not at package import. This keeps the CLI/API importable in minimal environments and lets each stage degrade independently rather than crash the whole pipeline.

Degradation ladders:

| Stage | Primary | Fallback ladder |
|---|---|---|
| **Parse** | clean structured extraction | scanned/garbled PDF → low `parse_score` → **degraded flat-text path** (D1) |
| **Normalize** | trained ByT5 | rule normalizer (`baseline_rules.py`) as guardrail / when model unavailable |
| **TTS** | `microsoft/speecht5_tts` | optional quality backends → `pyttsx3` (offline OS TTS, no weights) → **`PlaceholderTTS`** (deterministic low-noise floor) |
| **Export** | ffmpeg containers (mp3/m4b) | skip ffmpeg, emit wav/srt/manifest |

The `PlaceholderTTS` floor guarantees the pipeline is **testable with no model and never hard-fails**. Char-level ByT5 is intrinsically robust to numbers/symbols/OOV and adversarial/noisy tokens, complementing the rule guardrail.

---

## 6. Sample-Rate / Format Handling

- **SpeechT5** (primary) synthesizes at **16 kHz**; optional quality backends run at higher rates (`hexgrad/Kokoro-82M` 24 kHz, `parler-tts/parler-tts-mini-v1`).
- `tts_backend.py` reports each backend's native sample rate; `stitch.py` is responsible for resolving a consistent working rate, inserting silence gaps, and producing the final master.
- **Mastering target:** ACX-conformant **−18 LUFS integrated / −3 dBTP true-peak**, applied at stitch time.
- **Export formats:** `wav`, `mp3`, `m4b` (with chapter markers), plus `srt` subtitles and `manifest.json`. Real-Time Factor (RTF) is recorded in the manifest.

---

## 7. Training Artifacts → Inference Coupling

```
training/train_normalizer.py ──writes──► checkpoint(s) (Drive on Colab)
                                            │
                                            ▼
                       models/model_registry.py
                       tn_meta.json + LATEST POINTER + repo@revision pins
                                            │
                              ┌─────────────┴─────────────┐
                              ▼                            ▼
                   models/normalizer.py            api/dependencies.py
                   (loads pinned latest)           (constructs agent once)
                              │                            │
                              └──────────► agent/narrator_agent.py ◄┘
                                           (D2 normalize step)
```

Training never talks to the agent directly. It deposits a checkpoint and advances the **latest pointer** in the registry. At inference, `dependencies.py` builds the agent once; the agent's normalize step (`normalizer.py`) resolves the latest pinned revision via `model_registry.py`. Pinning by `repo@revision` makes deployments reproducible and rollback-able.

**Validated baseline (context-blind rules)** on the synthetic test distribution — the bar the trained model must clear:

| Slice | Sentence exact-match (EM) |
|---|---|
| easy | 0.945 |
| hard | 0.006 |
| overall | 0.712 |

The context-aware ByT5 model is expected to hold high accuracy on **both** slices — beating the baseline overall and especially on the hard, ambiguity-injected slice.

---

## 8. Shared Agent Across API / UI / CLI

All three entry points construct and drive the **same** `narrator_agent` over the **same** pipeline; only the I/O surface differs.

- **FastAPI** (`api/main.py`): `GET /healthz` `/readyz`; `POST /normalize` (text → spoken preview); `POST /synthesize` (json) and `/synthesize/file` (upload); `/artifacts` static mount; `/download`.
- **Gradio UI** (`api/ui.py`), mounted at `/ui` via the combined ASGI app (`api/app_combined.py`).
- **CLI** (`cli.py`, console-script `audiobook-ai`) subcommands: `data`, `train`, `tune`, `evaluate`, `normalize`, `synthesize`, `demo-agent`, `serve`, `benchmark`, `error-analysis`, `audio-quality`, `monitor`, `generate-report`, `generate-slides`, `autopilot`, `grade`.

Shared dependency wiring in `api/dependencies.py` loads the agent, registry, and backends **once**, so the HTTP server, UI, and in-process CLI calls share construction logic and configuration.

---

## 9. Deployment & Validated Run

- **Containers:** Docker (api CPU + worker GPU), `docker-compose`, and a Gradio **HF Space**.
- **Latency:** RTF reported per run in the manifest.
- **Scalability:** chunk-level parallelism + GPU micro-batching.
- **Versioning:** model registry (`tn_meta.json` + latest pointer, `repo@revision` pins).

**Validated end-to-end run** (real SpeechT5, CPU): 3 chapters, 7 spoken segments, 0 flagged, **all 4 decisions (D1–D4) fired**, produced a 67 s WAV + SRT + manifest, **RTF 2.28 on CPU** (≈ 0.1 expected on GPU).

---

## 10. Verified Backend & Model IDs

| Role | ID | License | Notes |
|---|---|---|---|
| Trainable TN model | `google/byt5-small` | apache-2.0 | ~300M, byte/char-level |
| TN fallback (small GPU) | `google-t5/t5-small` | apache-2.0 | 60.5M |
| Primary TTS | `microsoft/speecht5_tts` | MIT | ~144M, 16 kHz |
| Vocoder | `microsoft/speecht5_hifigan` | MIT | |
| Speaker x-vectors | `Matthijs/cmu-arctic-xvectors` | MIT | 7931 embeddings, 7 speakers |
| Optional quality | `hexgrad/Kokoro-82M` | apache-2.0 | 24 kHz |
| Optional quality | `parler-tts/parler-tts-mini-v1` | apache-2.0 | prompt-styled |
| Eval sanity set | `DigitalUmuganda/Text_Normalization_Challenge_Unittests_Eng_Fra` | — | optional |

**Excluded from any commercial path** (non-commercial licenses): `coqui/XTTS-v2` (CPML), `facebook/mms-tts-eng` (CC-BY-NC), `SWivid/F5-TTS` (CC-BY-NC). The commercial path uses only MIT/Apache IDs (license hygiene).
