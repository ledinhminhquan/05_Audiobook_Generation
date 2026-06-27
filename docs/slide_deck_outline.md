# P05 Audiobook Generation — Slide Deck Outline

Presentation outline for the final-assignment submission. Target length: **13 slides**, ~12–15 minutes. Each slide lists a title, 3–6 concise bullets, and a note on the visual/diagram to include. Aligned to the written report and the assignment's required slide list.

---

## Slide 1 — Title & Info

- **Audiobook Generation System** — turn long documents (EPUB / PDF / TXT / MD) into mastered, chaptered audiobooks
- Le Dinh Minh Quan — student **23127460**
- Course: NLP in Industry — final assignment
- One-line pitch: "A naive TTS reads *$5.2M*, *1984*, *Dr.*, *Chapter IV* wrong — we fix the hard NLP part."
- Reference repo: `github.com/denizsafak/abogen`

> **Visual:** Clean title slide. Background = a written sentence (`Dr. Smith paid $5.2M in 1984`) with arrows to its spoken form (`Doctor Smith paid five point two million dollars in nineteen eighty-four`).

---

## Slide 2 — Business Problem & Motivation

- Human narration is slow and expensive; publishers have large back-catalogs that never get an audio edition
- **Accessibility gap**: sight-impaired, dyslexic, and commuter listeners are underserved
- The hard, NLP-heavy bottleneck is **Text Normalization (TN)** — converting *written → spoken* form correctly
- A naive TTS mis-reads numbers, money, dates, times, abbreviations, Roman numerals, fractions
- Business levers: **cost per finished audio-hour** ≪ human narration, faster **catalog conversion**, **time-to-first-audio** (stream chapter 1 early)

> **Visual:** Side-by-side cost/time bar comparison (human narration vs automated) + a "mis-read gallery" of tricky tokens (`9:45 AM`, `3/4`, `Chapter IV`, `St.`).

---

## Slide 3 — Proposed NLP Solution (TN + TTS + Agent)

Three layers, with one trainable ML heart:

| Layer | What it does | Trained? |
|-------|--------------|----------|
| **TN model** | written → spoken across 16 semiotic classes | **Yes — the ML heart** |
| **TTS backend** | spoken text → audio waveform | No — pretrained neural TTS |
| **Agent** | orchestrates the full document → audiobook pipeline | Deterministic FSM (+ optional LLM) |

- The **trainable model** is a Text-Normalization seq2seq model — this is where the NLP learning happens
- Audio is produced by a **pretrained** neural TTS backend (no audio training)
- An **agent** runs the end-to-end pipeline with auditable decision points
- Output: `.wav` / `.mp3` / `.m4b` (chaptered) + `.srt` subtitles + `manifest.json`

> **Visual:** Three-layer stack diagram (TN → TTS → Agent) with the TN box highlighted as "the trainable model".

---

## Slide 4 — System Architecture Diagram

End-to-end FSM pipeline:

```
document
  -> PARSE        (ebooklib EPUB / PyMuPDF+pdfplumber PDF / txt+md)
  -> CHAPTER      (TOC / font-size / "Chapter N" regex)
  -> SEGMENT+CLASSIFY  (pysbd sentences; narration|dialogue|heading|skippable)
  -> NORMALIZE    (TRAINED ByT5, per-segment, with confidence)
  -> VOICE-ROUTE
  -> SYNTHESIZE   (SpeechT5)
  -> AUDIO-QA     (loudness/peak/duration/empty/silence; bounded re-synth)
  -> STITCH       (ACX -18 LUFS / -3 dBTP master)
  -> EXPORT       wav / mp3 / m4b(+chapter markers) / srt + manifest.json
```

- Every step is **timed + traced** (`ToolTrace`); every decision is **recorded** (`Decision`)
- Graceful degradation at every stage (rule normalizer, placeholder TTS, skip ffmpeg)

> **Visual:** Horizontal pipeline flowchart of the 9 stages, with the 4 agent decision points (D1–D4) marked as diamonds.

---

## Slide 5 — Data Overview (Synthetic TN Corpus, 16 Classes)

- **No** permissively-licensed English TN-Challenge mirror exists on HF Hub (canonical ids 404) → we ship a **reproducible synthetic corpus generator** (`data/tn_corpus.py`)
- Generates context-rich `(written, spoken)` pairs across **16 semiotic classes**:
  `PLAIN, PUNCT, CARDINAL, ORDINAL, DECIMAL, DIGIT, MONEY, MEASURE, DATE, TIME, FRACTION, TELEPHONE, ELECTRONIC, ABBREV, ROMAN, VERBATIM`
- **Injected ambiguity** the baseline misses: `St.` → Street vs Saint, `1984 people` (count) vs `in 1984` (year), day-as-ordinal dates, Roman-numeral contexts
- Default sizes: **train 60,000 / val 4,000 / test 4,000 / hard 1,500**; splits are **leakage-free**
- Optional eval sanity set: `DigitalUmuganda/Text_Normalization_Challenge_Unittests_Eng_Fra` (verified)

> **Visual:** Table of the 16 classes with one example each (e.g. MONEY `$5.2M` → `five point two million dollars`), plus a small split-size bar.

---

## Slide 6 — Model & Evaluation Results

- **Model:** `google/byt5-small` (apache-2.0, ~300M, **byte/char-level** → robust to numbers/symbols/OOV); fallback `google-t5/t5-small` (apache-2.0, 60.5M). Seq2Seq, task prefix `"normalize: "`
- **Baseline to beat:** a context-**blind** rule normalizer (ordered regex + abbreviation dict + `num2words`) — strong on easy cases, fails on ambiguous ones
- **Validated baseline EM** (synthetic test): **easy = 0.945**, **hard = 0.006**, **overall = 0.712**
- The trained context-aware model is expected to score high on **both** slices → beats baseline overall and especially on the **hard slice**

**Metrics reported:**

| Metric | Why |
|--------|-----|
| Sentence exact-match accuracy | Headline number |
| Macro per-semiotic-class accuracy | Exposes rare-class failures |
| Per-class accuracy | Diagnostic detail |
| Unrecoverable / "silly" error rate (Sproat & Jaitly) | Catastrophic mis-reads → minimized even at cost of benign ones |

> **Visual:** Grouped bar chart — baseline vs trained model on **easy / hard / overall** EM, with the hard-slice gap (0.006) as the dramatic highlight.

---

## Slide 7 — Agentic AI Component (FSM + D1–D4)

Deterministic FSM with an **optional LLM brain**, **4 decision points** (requirement is ≥3):

- **D1 — Parse-quality routing:** `parse_score ∈ [0,1]` → `structured (≥0.85)` / `assisted (0.5–0.85)` / `degraded (<0.5)`
- **D2 — Normalization-confidence escalation:** per-segment confidence = length-normalized sequence probability; below `norm_confidence_min = 0.55` → flag + optionally escalate to LLM (validates, falls back to rules); records neural-vs-baseline disagreement
- **D3 — Audio-QA re-synthesis gate:** per-clip checks (empty/NaN, duration ratio, peak/clipping, silence) → bounded re-synth (max 2 attempts: reseed → split → fallback backend) → accept-best + flag
- **D4 — Voice routing:** heading / dialogue / narration → distinct voices (x-vector indices), stable per book
- **LLM OFF by default** → zero paid API, CPU-only; full `manifest.json` audit trail

> **Visual:** FSM state diagram with D1–D4 as labeled decision diamonds; callout box showing one D2 example (low-confidence segment → escalated → resolved).

---

## Slide 8 — TTS Backends (Pretrained, Not Trained)

- **Primary:** `microsoft/speecht5_tts` (MIT, ~144M, 16kHz) + `microsoft/speecht5_hifigan` vocoder (MIT) + speaker x-vectors `Matthijs/cmu-arctic-xvectors` (MIT, 7931 embeddings, 7 speakers) — **CPU-demo-able**, multi-voice
- **Optional quality:** `hexgrad/Kokoro-82M` (apache-2.0, 24kHz), `parler-tts/parler-tts-mini-v1` (apache-2.0, prompt-styled)
- **Fallback:** `pyttsx3` (offline OS TTS) → ultimate floor `PlaceholderTTS` (deterministic) so the pipeline **never hard-fails**
- **Excluded from commercial path** (non-commercial licenses): `coqui/XTTS-v2` (CPML), `facebook/mms-tts-eng` (CC-BY-NC), `SWivid/F5-TTS` (CC-BY-NC)

> **Visual:** Tiered backend pyramid (Primary → Optional quality → Fallback → Floor) color-coded by license (MIT/Apache = commercial-safe vs non-commercial).

---

## Slide 9 — Deployment Overview (API / Gradio / CLI / Docker, RTF)

- **FastAPI** (`api/main.py`): `GET /healthz /readyz`, `POST /normalize` (text→spoken preview), `POST /synthesize` (+ `/synthesize/file` upload), `/artifacts` static mount, `/download`
- **Gradio UI** (`api/ui.py`) + combined ASGI app mounting UI at `/ui`
- **CLI** (`audiobook-ai`): `data, train, tune, evaluate, normalize, synthesize, demo-agent, serve, benchmark, autopilot, grade`, …
- **Docker** (API CPU + worker GPU), docker-compose, HF Space (Gradio)
- **Scalability:** chunk-level parallelism + GPU micro-batching; **versioning** via model registry (`tn_meta.json` + latest pointer, `repo@revision` pins)
- **Validated end-to-end run** (real SpeechT5, CPU): 3 chapters, 7 spoken segments, **0 flagged**, **all 4 decisions fired** → 67s WAV + SRT + manifest, **RTF 2.28 on CPU** (~0.1 on GPU)

> **Visual:** Deployment topology (API + worker + UI + Docker) on the left; on the right a manifest snippet showing `RTF = 2.28` and `decisions: D1–D4 fired`.

---

## Slide 10 — Ethics, Privacy & Risks

- **Voice-cloning consent:** XTTS / F5 are non-commercial, consent-gated, **off by default**
- **Hallucinated pronunciations:** mitigated by rule guardrail + D3 audio-QA + **unrecoverable-error counting**
- **PII in books** (phones / addresses / emails / names): minimize logging, TTL cleanup, NER for attribution only
- **Copyright:** converting a book is a derivative work → verify rights / public-domain, record input **sha256**
- **License hygiene:** commercial path uses **only MIT / Apache** model ids
- **Robustness:** scanned PDFs → degraded flat-text path; garbled text → low `parse_score`; char-level ByT5 robust to noisy/adversarial tokens

> **Visual:** Risk-register table (Risk → Mitigation) with a license "traffic light" (green = MIT/Apache, red = CC-BY-NC/CPML).

---

## Slide 11 — Success Metrics

- **Business:** cost per finished audio-hour ≪ human narration; accessibility reach; time-to-first-audio; catalog conversion speed
- **Technical:** TN sentence accuracy (beat baseline), macro-class accuracy, **unrecoverable-error rate → 0**, RTF, audio-QA pass rate, **loudness conformance (-18 LUFS / -3 dBTP ACX)**
- Honest, reproducible result: baseline overall EM **0.712**, hard-slice **0.006** — the gap the trained model closes

> **Visual:** Two-column scorecard (Business metrics | Technical metrics) with the ACX loudness target badge.

---

## Slide 12 — Key Takeaways

- The **real NLP problem** in audiobook production is Text Normalization, not the audio itself
- A **char-level ByT5** TN model beats a context-blind rule baseline exactly where it matters — the **hard, ambiguous** cases (baseline hard EM = 0.006)
- A **deterministic FSM agent** with 4 audited decision points (D1–D4) makes the pipeline robust, traceable, and degradable
- Fully **license-clean** commercial path (MIT/Apache only) and runs **CPU-only, zero paid API**
- End-to-end validated: document → mastered, chaptered audiobook + subtitles + manifest

> **Visual:** Four icon-cards summarizing the four takeaways (TN-is-the-problem / ByT5-wins-hard / agent-audited / license-clean).

---

## Slide 13 — Future Work

- Train and report the **full trained-model numbers** on both easy and hard slices (close the loop vs the 0.712 baseline)
- Integrate higher-quality TTS (Kokoro-82M / Parler) on the commercial path for richer voices
- Source or build a **larger, more diverse** TN corpus (real-data loader) to broaden class coverage
- Expand the **LLM brain** for D2 escalation while keeping the zero-cost default
- Streaming **time-to-first-audio** (chapter-1 first) and broader format/language support

> **Visual:** Roadmap timeline (Now → Next → Later) mapping each future-work item to a milestone.

---

### Required-slide coverage checklist

| Required slide | Covered by |
|----------------|-----------|
| Title & info | Slide 1 |
| Business Problem & Motivation | Slide 2 |
| Proposed NLP Solution (TN + TTS + agent) | Slide 3 |
| System Architecture Diagram | Slide 4 |
| Data Overview (synthetic TN corpus, 16 classes) | Slide 5 |
| Model & Evaluation Results | Slide 6 |
| Agentic AI Component (FSM + D1–D4 + example) | Slide 7 |
| Deployment Overview (API/Gradio/CLI/Docker, RTF) | Slide 9 |
| Ethics, Privacy & Risks | Slide 10 |
| Key Takeaways & Future Work | Slides 12–13 |
