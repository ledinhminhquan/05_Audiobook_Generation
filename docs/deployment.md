# Deployment — Audiobook Generation System (P05)

This document describes how the Audiobook Generation System is packaged and served:
the delivered deployment formats, the working inference pipeline, input/output
contracts, measured latency, scalability strategy, model versioning, and the
known challenges and limitations.

The system turns long documents (EPUB / PDF / TXT / MD) into mastered, chaptered
audiobooks (`.wav` / `.mp3` / `.m4b` + `.srt` subtitles + `manifest.json`). The
trainable ML heart is a **Text-Normalization (TN) seq2seq model** (written → spoken);
audio is produced by a pretrained neural TTS backend (`microsoft/speecht5_tts`);
a deterministic agent FSM orchestrates the full document → audiobook flow.

---

## 1. Deployment formats delivered

The system ships four interchangeable surfaces over the same inference core, so the
same pipeline can be driven from a service, a browser, a terminal, or a batch job.

| Format | Entry point | Use case |
| --- | --- | --- |
| **REST API** | `src/audiobook_ai/api/main.py` (FastAPI) | Programmatic / service integration |
| **Gradio web demo** | `src/audiobook_ai/api/ui.py` | Interactive browser demo, HF Space |
| **CLI** | `src/audiobook_ai/cli.py` (console script `audiobook-ai`) | Local ops, automation, CI |
| **Batch** | CLI `synthesize` over a directory / autopilot | Bulk catalog conversion |

A combined ASGI app (`src/audiobook_ai/api/app_combined.py`) mounts the Gradio UI at
`/ui` on top of the FastAPI app, so a single process can serve both the API and the
web demo.

### 1.1 REST API endpoints

FastAPI service in `src/audiobook_ai/api/main.py`:

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/healthz` | Liveness probe (process is up) |
| `GET` | `/readyz` | Readiness probe (models loaded, ready to serve) |
| `POST` | `/normalize` | Text → spoken-form preview (TN model only, no audio) |
| `POST` | `/synthesize` | Full synthesis from a JSON body |
| `POST` | `/synthesize/file` | Full synthesis from an uploaded document |
| `GET` | `/artifacts/...` | Static mount serving generated artifacts |
| `GET` | `/download` | Download a produced audiobook / artifact |

`/normalize` exercises only the trained ByT5 normalizer and is the cheapest way to
sanity-check spoken-form output. `/synthesize` and `/synthesize/file` run the full
parse → normalize → synthesize → master pipeline and emit audio plus a manifest.

### 1.2 Gradio web demo

`src/audiobook_ai/api/ui.py` provides an interactive demo: paste/upload a document,
preview normalization, generate audio, and download artifacts. It is the surface
deployed to the **Hugging Face Space**. Because the primary TTS backend
(`microsoft/speecht5_tts`) is CPU-demo-able, the Space runs without a GPU.

### 1.3 CLI

The `audiobook-ai` console script (`src/audiobook_ai/cli.py`) exposes the full
lifecycle as subcommands:

```
audiobook-ai data            # build/refresh the synthetic TN corpus
audiobook-ai train           # train the TN normalizer
audiobook-ai tune            # hyperparameter tuning
audiobook-ai evaluate        # metrics vs. baseline
audiobook-ai normalize       # written -> spoken preview
audiobook-ai synthesize      # document -> audiobook
audiobook-ai demo-agent      # run the agent FSM with traces
audiobook-ai serve           # launch the FastAPI / combined app
audiobook-ai benchmark       # latency / RTF benchmark
audiobook-ai error-analysis  # per-class error breakdown
audiobook-ai audio-quality   # audio-QA report
audiobook-ai monitor         # runtime monitoring
audiobook-ai generate-report # autoreport
audiobook-ai generate-slides # slides
audiobook-ai autopilot       # end-to-end automation
audiobook-ai grade           # self-grading
```

### 1.4 Batch

Batch conversion runs the same pipeline over many inputs via `synthesize` (per file)
or `autopilot` (end-to-end). Each job is independent and writes its own
`manifest.json`, so a catalog can be processed without shared state.

---

## 2. Working inference pipeline

The agent runs a deterministic finite-state machine over each document. Every stage
is timed and traced (`ToolTrace`) and every routing decision is recorded
(`Decision`), all surfaced in the final `manifest.json`.

```
document
  -> PARSE            (ebooklib EPUB / PyMuPDF + pdfplumber PDF / txt + md)
  -> CHAPTER detect   (TOC / font-size / "Chapter N" regex)
  -> SEGMENT+CLASSIFY (pysbd sentences + TTS-chunking;
                       narration | dialogue | heading | skippable)
  -> NORMALIZE        (trained ByT5, per-segment, with confidence)
  -> VOICE-ROUTE      (x-vector indices per role)
  -> SYNTHESIZE       (SpeechT5)
  -> AUDIO-QA         (loudness / peak / duration / empty / silence; bounded re-synth)
  -> STITCH           (silence gaps, ACX -18 LUFS / -3 dBTP master)
  -> EXPORT           (wav / mp3 / m4b + chapter markers / srt + manifest.json)
```

The pipeline embeds four agent decision points (≥3 required):

- **D1 — parse-quality routing:** `parse_score ∈ [0,1]` from alpha-ratio + structure
  signal + segment-length sanity → `structured` (≥0.85) / `assisted` (0.5–0.85) /
  `degraded` (<0.5).
- **D2 — normalization-confidence escalation:** per-segment length-normalized
  sequence probability; below `norm_confidence_min` (0.55) → flag and optionally
  escalate to the LLM brain (off by default), recording neural-vs-baseline disagreement.
- **D3 — audio-QA re-synthesis gate:** per-clip checks (empty/NaN, duration ratio,
  peak/clipping, silence fraction); on fail, bounded re-synth (max 2 attempts,
  escalating reseed → split → fallback backend), then accept-best + flag.
- **D4 — voice routing:** heading / dialogue / narration → distinct x-vector voices,
  stable per book.

Graceful degradation exists at every stage: a rule normalizer backs the neural model,
and `pyttsx3` / a deterministic `PlaceholderTTS` back the neural TTS so the pipeline
never hard-fails and remains testable with no model weights present.

---

## 3. Input / output formats

### 3.1 Inputs

| Input | Parser |
| --- | --- |
| `.epub` | `ebooklib` |
| `.pdf` | `PyMuPDF` + `pdfplumber` |
| `.txt` | plain text |
| `.md` | markdown |
| raw text | direct (via `/normalize`, `/synthesize`, or CLI) |

### 3.2 Outputs

| Output | Description |
| --- | --- |
| `.wav` | Lossless mastered audio (SpeechT5 is 16 kHz) |
| `.mp3` | Compressed delivery format |
| `.m4b` | Chaptered audiobook with chapter markers (requires `ffmpeg`) |
| `.srt` | Subtitle / caption track aligned to audio |
| `manifest.json` | Run metadata, traces, decisions, RTF, artifact index |

All audio is mastered to the ACX target of **−18 LUFS** integrated loudness and
**−3 dBTP** true peak.

### 3.3 JSON schemas

Request/response models live in `src/audiobook_ai/api/schemas.py`.

**`POST /normalize`** — text → spoken preview:

```json
// request
{ "text": "Dr. Smith paid $5.2M in 1984." }

// response
{
  "input": "Dr. Smith paid $5.2M in 1984.",
  "spoken": "Doctor Smith paid five point two million dollars in nineteen eighty-four.",
  "confidence": 0.91
}
```

**`POST /synthesize`** — JSON body → audiobook artifacts:

```json
// request
{
  "text": "Chapter IV. It was 9:45 AM ...",
  "formats": ["wav", "mp3", "m4b", "srt"],
  "voice_routing": true
}

// response (artifact index; mirrors manifest.json)
{
  "manifest": "/artifacts/<job>/manifest.json",
  "artifacts": {
    "wav": "/artifacts/<job>/book.wav",
    "mp3": "/artifacts/<job>/book.mp3",
    "m4b": "/artifacts/<job>/book.m4b",
    "srt": "/artifacts/<job>/book.srt"
  },
  "rtf": 2.28,
  "chapters": 3,
  "segments": 7,
  "flagged": 0
}
```

`POST /synthesize/file` is identical but takes a multipart document upload instead of
the `text` field. The `manifest.json` is the authoritative record: it carries the
per-stage traces, all agent decisions, the loudness master result, and the measured RTF.

---

## 4. User interaction

- **Browser users** open the Gradio UI (`/ui` or the HF Space), upload/paste a document,
  preview normalization, generate, and download.
- **Developers** call the REST API: `/normalize` for a cheap spoken-form preview,
  `/synthesize` or `/synthesize/file` for full jobs, then `/download` / `/artifacts`
  to retrieve outputs. `/healthz` and `/readyz` integrate with orchestrators.
- **Operators** drive everything from the `audiobook-ai` CLI (incl. `serve`,
  `benchmark`, `autopilot`).
- **Batch consumers** feed many documents through `synthesize` / `autopilot`, each
  producing an independent manifest.

---

## 5. Latency

Latency is reported as **Real-Time Factor (RTF)** — processing time divided by audio
duration — and recorded in `manifest.json` for every run.

| Environment | RTF | Interpretation |
| --- | --- | --- |
| **CPU (validated)** | **2.28** | ~2.28 s of compute per 1 s of audio |
| **GPU (estimated)** | **~0.1** | ~10× faster than real time |

These come from a **validated end-to-end run** with the real SpeechT5 backend on CPU:
3 chapters, 7 spoken segments, 0 flagged, all 4 agent decisions fired, producing a
**67 s WAV + SRT + manifest** at **RTF 2.28**.

**Time-to-first-audio** is a first-class metric: chapters are produced and can be
streamed independently, so a listener gets Chapter 1 audio long before the full book
finishes rendering, rather than waiting on the entire document.

---

## 6. Scalability

The workload is **TTS-bound**, and TTS parallelizes cleanly at the segment level:

- **Chunk-level parallelism:** the document is segmented into independent sentence/
  TTS chunks; these synthesize concurrently and are stitched in order at the end.
- **GPU micro-batching:** segments are micro-batched on the GPU to amortize kernel
  launch and keep the device saturated, which is what drives the ~0.1 GPU RTF.
- **Queue-based work distribution:** the deployment splits an **API tier (CPU)** for
  parsing/normalization/serving from a **worker tier (GPU)** for synthesis, connected
  by a job queue, so synthesis throughput scales by adding GPU workers independently of
  the API frontend.

Because each job is self-contained (own manifest, own input sha256), horizontal scaling
needs no shared mutable state.

---

## 7. Model versioning

- **Model registry:** `src/audiobook_ai/models/model_registry.py` tracks trained TN
  models via a `tn_meta.json` metadata record plus a `latest` pointer, so the serving
  layer resolves a known-good normalizer at startup.
- **Pinned external weights:** all pretrained backends are pinned by **`repo@revision`**
  so deployments are reproducible. The primary serving stack pins:
  - TN model: `google/byt5-small` (apache-2.0; fallback `google-t5/t5-small`)
  - TTS: `microsoft/speecht5_tts` (MIT) + `microsoft/speecht5_hifigan` vocoder (MIT)
  - Speaker x-vectors: `Matthijs/cmu-arctic-xvectors` (MIT)
- **License hygiene:** the commercial serving path uses only MIT / Apache ids.
  Non-commercial backends (`coqui/XTTS-v2`, `facebook/mms-tts-eng`, `SWivid/F5-TTS`)
  are excluded from that path and gated off by default.

---

## 8. Packaging: Docker & Hugging Face Space

- **Docker:** two images — an **API image (CPU)** and a **worker image (GPU)** —
  matching the API/worker split in §6. A `docker-compose.yml` wires them together with
  the job queue for local / single-host deployment. Deploy assets live under `deploy/`.
- **Hugging Face Space:** the Gradio UI is published as an HF Space. It runs CPU-only
  thanks to the CPU-demo-able SpeechT5 backend, with the LLM brain off by default
  (zero paid API).

---

## 9. Deployment challenges & limitations

- **TTS RTF dominates long books.** Synthesis is the bottleneck, and at **CPU RTF 2.28**
  a full-length book takes longer than the audio it produces. This is why GPU workers
  (RTF ~0.1), chunk-level parallelism, and chapter-streaming (time-to-first-audio)
  matter — CPU is fine for previews and the demo Space, but not for fast bulk catalog
  conversion.
- **`ffmpeg` dependency for `.m4b`.** Producing chaptered `.m4b` with chapter markers
  requires `ffmpeg` in the runtime image. Where it is unavailable, the pipeline degrades
  gracefully and still emits `.wav` / `.mp3` / `.srt`.
- **GPU memory.** Micro-batch size on the worker tier is bounded by device memory;
  larger batches improve throughput but must fit the GPU, so batch sizing is a
  per-device tuning knob.
- **Out-of-distribution inputs.** Scanned PDFs route through the `degraded` flat-text
  path (D1), and garbled text yields a low `parse_score`. The char-level ByT5 normalizer
  is robust to noisy/adversarial tokens, but parse quality on poor source documents
  remains a real ceiling on output quality.
- **Copyright / rights.** Converting a book is a derivative work; deployments must
  verify rights or public-domain status. The system records the input `sha256` for
  provenance, but rights verification is an operational responsibility, not an automated
  guarantee.
