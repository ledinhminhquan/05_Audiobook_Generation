# P05 Audiobook Generation — Project Plan & Timeline

**Project:** Audiobook Generation System — turn long documents (EPUB/PDF/TXT/MD) into mastered, chaptered audiobooks (`.wav` / `.mp3` / `.m4b` + `.srt` subtitles).
**Author:** Le Dinh Minh Quan (student 23127460)
**Course:** NLP in Industry — Final Assignment
**Reference repo:** `github.com/denizsafak/abogen`

This document covers **Project Management & Teamwork**: the build plan and timeline, a task breakdown across simulated roles for solo execution, and a reflection on how this system scales to a real team.

---

## 1. Project framing (what is actually being built)

The trainable ML heart of this project is **not** the audio model — it is a **Text-Normalization (TN) seq2seq model** that converts written tokens to their spoken form across **16 semiotic classes** (PLAIN, PUNCT, CARDINAL, ORDINAL, DECIMAL, DIGIT, MONEY, MEASURE, DATE, TIME, FRACTION, TELEPHONE, ELECTRONIC, ABBREV, ROMAN, VERBATIM).

A naive TTS reads `"$5.2M"`, `"1984"`, `"Dr."`, `"Chapter IV"`, `"3/4"`, `"9:45 AM"` wrong. Normalization is the hard, NLP-heavy part of audiobook production. Around the trained model sit three pre-built layers:

1. A **pretrained neural TTS backend** (SpeechT5, not trained by us).
2. A **deterministic FSM agent** with **4 decision points** (D1–D4) that orchestrates the whole document → audiobook pipeline.
3. A **deployment surface**: FastAPI + Gradio UI + CLI + Docker.

Because the heavy artifact (the TN model) is decoupled from the audio backends and the agent, the work naturally factors into modules that **can be owned independently** — the key property that makes the team-scaling story below credible.

---

## 2. Project plan & timeline

The plan is organized into **seven phases** mapped to an **8-week** milestone schedule. Phases overlap deliberately: the synthetic-data generator and the rule baseline are stood up early so they can serve as both training data and an evaluation yardstick throughout.

### 2.1 Phases

| Phase | Name | Goal / exit criterion |
|------|------|------------------------|
| P1 | Research & scoping | Confirm the TN-as-trainable-model thesis; pin VERIFIED model/TTS ids and licenses; confirm there is **no permissive English TN-Challenge mirror** on HF Hub, so synthetic data is primary. |
| P2 | Data & baseline | Synthetic corpus generator (`tn_corpus.py`) with injected ambiguity; leakage-free splits; context-blind rule baseline (`baseline_rules.py`) to beat. |
| P3 | Model + training | Train ByT5 TN model with HF `Seq2SeqTrainer`; anti-overfitting controls; beat baseline overall and especially on the hard slice. |
| P4 | Agent | Deterministic FSM with 4 decision points (D1–D4), tracing, optional LLM brain (off by default). |
| P5 | Deployment | FastAPI endpoints, Gradio UI, combined ASGI app, CLI, Docker/compose, HF Space, model registry. |
| P6 | Evaluation | Sentence EM, macro per-class accuracy, per-class accuracy, unrecoverable-error rate; end-to-end RTF + audio-QA pass rate. |
| P7 | Docs & report | This plan, README, report, slides, error analysis, manifest-driven monitoring. |

### 2.2 Milestone schedule (week-by-week)

| Week | Phase(s) | Milestone / deliverable | Definition of done |
|------|----------|--------------------------|--------------------|
| W1 | P1 | Scope locked; ids + licenses verified | `byt5-small` (apache-2.0) primary, `t5-small` fallback chosen; TTS = `microsoft/speecht5_tts` + `speecht5_hifigan` + `Matthijs/cmu-arctic-xvectors`; commercial path is MIT/Apache only. |
| W2 | P2 | Synthetic corpus + baseline | `tn_corpus.py` emits train 60000 / val 4000 / test 4000 / hard 1500; dedup before split; splits leakage-free. |
| W2–W3 | P2/P6 | Baseline numbers recorded | Baseline sentence-EM **easy 0.945 / hard 0.006 / overall 0.712** — the bar to beat. |
| W3–W5 | P3 | Trained TN model | `Seq2SeqTrainer` with `predict_with_generate`, bf16+tf32, cosine + warmup 0.05, label smoothing 0.1, early stopping (patience 4 on `sentence_accuracy`), `load_best_model_at_end`. ByT5 ~3–6 h on one H100. |
| W4–W5 | P4 | Agent FSM | D1 parse-quality routing, D2 normalization-confidence escalation, D3 audio-QA re-synth gate, D4 voice routing; every step traced (`ToolTrace`), every decision recorded (`Decision`). |
| W5–W6 | P5 | Deployment surface | `api/main.py` (`/healthz`, `/readyz`, `/normalize`, `/synthesize`, `/synthesize/file`, `/download`); Gradio UI mounted at `/ui`; CLI console-script `audiobook-ai`; Docker (api CPU + worker GPU). |
| W6–W7 | P6 | Full evaluation | Headline sentence EM beats baseline on both slices; macro-class + per-class accuracy; unrecoverable-error count → 0; validated end-to-end run (3 chapters, 7 segments, 0 flagged, all 4 decisions fired, 67 s WAV + SRT + manifest, RTF 2.28 CPU). |
| W7–W8 | P7 | Docs, report, slides | README, report, slides via `generate-report` / `generate-slides`; `monitor` over `manifest.json`; model registry (`tn_meta.json` + `latest` pointer, `repo@revision` pins). |

> **Buffer / risk:** small-GPU fallback (`t5-small`, fits T4) and the GPU profile table (H100/A100-40/L4/T4, effective batch held at 256) de-risk W3–W5 if H100 time is unavailable. The `PlaceholderTTS` floor and `pyttsx3` fallback keep the pipeline testable even with no model weights, de-risking W5–W6.

---

## 3. Task breakdown by simulated role (solo execution)

Although this is solo work, every task is tagged with the role it would belong to on a real team. This keeps responsibilities legible and maps cleanly onto the module boundaries in `src/audiobook_ai/`.

### 3.1 Roles & responsibilities

| Role | Owns (modules) | Core responsibilities in this project |
|------|----------------|----------------------------------------|
| **PM** | `docs/`, planning | Scope, milestone schedule, success-metric definitions (business + technical), license-hygiene gate (commercial path = MIT/Apache only), ethics/copyright sign-off. |
| **Data Engineer** | `data/` (`tn_corpus.py`, `dataset.py`, `document.py`, `download_dataset.py`) | Synthetic corpus with injected ambiguity (`St.`→Street vs Saint, `1984 people` count vs `in 1984` year, Roman contexts); dedup + leakage-free splits; PLAIN capped ~30%; optional real-data + sanity set `DigitalUmuganda/Text_Normalization_Challenge_Unittests_Eng_Fra`; document parsing (ebooklib / PyMuPDF+pdfplumber / txt+md). |
| **ML Engineer** | `models/`, `training/` | TN model (`byt5-small`, fallback `t5-small`), task prefix `"normalize: "`; training config + GPU profiles; anti-overfitting (early stop, label smoothing, weight decay, held-out hard slice); rule baseline as the bar to beat. |
| **Backend / Deploy** | `synthesis/`, `agent/`, `api/`, `deploy/` | TTS backends + vocoder + x-vector voices; agent FSM (D1–D4) + tracing; FastAPI + Gradio + combined ASGI; CLI; Docker/compose; HF Space; model registry + `repo@revision` pins; RTF in manifest; chunk-level parallelism + GPU micro-batching. |
| **QA** | `tests/`, `analysis/`, `monitoring/` | Audio-QA gate (loudness/peak/duration/empty/silence) + bounded re-synth; metric harness (sentence EM, macro/per-class, unrecoverable errors); ACX conformance (-18 LUFS / -3 dBTP); graceful-degradation tests (degraded parse path, placeholder TTS, skip ffmpeg); end-to-end smoke run. |

### 3.2 Representative task list (role-tagged)

- **[PM]** Verify VERIFIED ids/licenses; define success metrics; record input `sha256` + rights/public-domain check (copyright = derivative work).
- **[Data]** Build `tn_corpus.py`; inject ambiguity across all 16 classes; enforce dedup + leakage-free splits; wire optional real-data loader and sanity set.
- **[ML]** Implement `baseline_rules.py` (ordered regex spans + abbrev dict + `num2words`/pure-python expanders); record baseline EM (0.945 / 0.006 / 0.712); train ByT5; tune; evaluate; beat baseline.
- **[Backend]** Implement TTS backend (SpeechT5 + HiFi-GAN + x-vectors); agent FSM with D1–D4 + `ToolTrace`/`Decision` audit; FastAPI + UI + CLI; Docker; model registry.
- **[QA]** Implement audio-QA checks + bounded re-synth (max 2 attempts: reseed → split → fallback backend); metric harness incl. unrecoverable-error counting; loudness conformance; degradation tests; validate the 3-chapter end-to-end run.

---

## 4. Reflection: scaling to a real team

The solo build is deliberately structured so the simulated roles in §3 become **real parallel workstreams** with minimal rework.

### 4.1 Parallel workstreams & ownership boundaries
The module split is the contract between teams. Three workstreams can run concurrently after W2 because they communicate through stable artifacts, not shared code:

- **Data + ML** own `data/` → `models/` → `training/`. Their output is a versioned model artifact plus metrics.
- **Backend / Deploy** own `synthesis/`, `agent/`, `api/`, `deploy/`. They consume the model only through the **model registry** (`tn_meta.json` + `latest` pointer, `repo@revision` pins) — so they can build the FSM and API against the rule baseline or `PlaceholderTTS` long before the trained model lands.
- **QA** own `tests/`, `analysis/`, `monitoring/` and define the metric harness up front, so "beats baseline" and "unrecoverable-error rate → 0" are enforced as gates rather than negotiated at the end.

The hard interface between Data/ML and Backend is the **normalize step** (text → spoken with a confidence score); the hard interface between Backend and QA is the **`manifest.json`** (per-clip QA, decisions, RTF). Both are already produced by the solo pipeline, so they need no redesign to become team boundaries.

### 4.2 CI, model registry & versioning
- **CI** runs the QA harness on every change: synthetic-corpus regeneration is deterministic, the baseline number is fixed, and the end-to-end smoke run (3 chapters, all 4 decisions fire) is cheap on CPU thanks to `PlaceholderTTS`/`pyttsx3` floors — so a PR can be blocked on a real functional check, not just unit tests.
- **Model registry** decouples training cadence from deployment cadence. New checkpoints publish a new `tn_meta.json`; deploy pins `repo@revision`, enabling safe rollback and reproducible serving.

### 4.3 On-call & monitoring
- Every pipeline step is **timed and traced** (`ToolTrace`) and every decision is **recorded** (`Decision`) into `manifest.json`. The `monitor` CLI command reads these, so a real on-call rotation watches concrete signals: **RTF**, **audio-QA pass rate**, **flagged-segment / escalation rate** (D2), **re-synth rate** (D3), and **loudness conformance** (-18 LUFS / -3 dBTP ACX).
- Graceful degradation is built in (degraded flat-text parse path, rule-normalizer fallback, placeholder TTS, skip-ffmpeg), so the pipeline never hard-fails — turning many would-be pages into quality alerts instead of outages.
- The optional LLM brain is **off by default** (zero paid API, CPU-only), so cost and latency stay predictable as load scales via **chunk-level parallelism + GPU micro-batching**.

### 4.4 What a team would add that solo work defers
PII handling at scale (NER for attribution only, minimized logging, TTL cleanup), voice-cloning **consent gating** (XTTS/F5 are non-commercial and off by default), and a per-title rights/copyright workflow (record input `sha256`, verify public-domain) become dedicated ownership rather than checklist items — a natural fifth workstream (Trust & Safety / Legal) layered on the same audit trail.

---

## 5. Success metrics (tracked across the plan)

| Type | Metric |
|------|--------|
| Business | Cost per finished audio-hour (≪ human narration); accessibility (sight-impaired/dyslexic/commuters); time-to-first-audio (stream chapter 1); catalog conversion speed. |
| Technical | TN sentence accuracy (beat baseline 0.712 overall / 0.006 hard); macro per-class accuracy; unrecoverable-error rate → 0; RTF; audio-QA pass rate; loudness conformance (-18 LUFS / -3 dBTP ACX). |
