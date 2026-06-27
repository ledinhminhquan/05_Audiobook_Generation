# Problem Definition — Audiobook Generation System (P05)

**Author:** Le Dinh Minh Quan (student 23127460)
**Course:** NLP in Industry — Final Assignment
**Reference repository:** github.com/denizsafak/abogen

---

## 1. Business Context & Motivation

The audiobook market is one of the fastest-growing segments of publishing, yet
producing a single title remains slow and expensive. Traditional production
depends on a professional narrator in a studio, plus editing, mastering, and QA
passes. This pushes the cost of a finished audio-hour to hundreds of dollars and
the turnaround to days or weeks, which means the **vast majority of written
content never becomes audio at all** — backlist catalogs, self-published books,
technical manuals, course material, and internal documents.

The **Audiobook Generation System** turns long documents
(EPUB / PDF / TXT / MD) into mastered, chaptered audiobooks (`.wav` / `.mp3` /
`.m4b` + `.srt` subtitles). It collapses the parse → normalize → synthesize →
master pipeline into an automated, auditable process, so that converting a book
becomes a matter of compute minutes rather than studio hours.

The core insight that shapes the entire design: **a naive TTS engine reads
written text wrong.** It mispronounces `"$5.2M"`, `"1984"`, `"Dr."`,
`"Chapter IV"`, `"3/4"`, and `"9:45 AM"`. The genuinely hard, NLP-heavy part of
audiobook production is **Text Normalization (TN)** — converting written form to
spoken form. That is where this project places its trainable model. The audio
itself is produced by a *pretrained* neural TTS backend, and an **agent**
orchestrates the full document-to-audiobook pipeline.

---

## 2. Target Users & Stakeholders

| Stakeholder | Need served |
|---|---|
| **Publishers** | Convert large backlist catalogs to audio quickly and cheaply; faster catalog conversion. |
| **Accessibility organizations** | Make text available to sight-impaired and dyslexic readers in spoken form. |
| **Indie / self-published authors** | Produce an audiobook without the cost of a studio narrator. |
| **E-learning providers** | Turn course material and manuals into listenable lessons. |
| **Individuals / commuters** | Convert personal documents and books for hands-free listening. |

These users share one constraint: they have **far more text than they can afford
to narrate by hand.** The system exists to remove that bottleneck while keeping
broadcast-grade audio quality.

---

## 3. The Problem Solved

Given an arbitrary long document, the system must:

1. **Parse** structure out of heterogeneous formats (EPUB via `ebooklib`,
   PDF via PyMuPDF + pdfplumber, plain `txt`/`md`).
2. **Detect chapters** (TOC, font-size, `"Chapter N"` regex).
3. **Segment and classify** content into sentences and chunks tagged as
   narration / dialogue / heading / skippable.
4. **Normalize** each segment from written to spoken form.
5. **Route voices**, **synthesize** audio, run **audio QA**, **stitch**, master
   to broadcast loudness, and **export** `wav` / `mp3` / `m4b` (with chapter
   markers) / `srt` plus a `manifest.json`.

The headline difficulty is **step 4**: turning written tokens into the words a
human would actually say, correctly and in context, across an entire book.

---

## 4. Why NLP Is Required

This is not a problem that string-substitution or hand-written rules can solve.
Three properties make it irreducibly an NLP problem.

### 4.1 Text Normalization is unsolved-by-rules

Written-to-spoken normalization spans **16 semiotic classes** — PLAIN, PUNCT,
CARDINAL, ORDINAL, DECIMAL, DIGIT, MONEY, MEASURE, DATE, TIME, FRACTION,
TELEPHONE, ELECTRONIC, ABBREV, ROMAN, VERBATIM — and the correct reading is
**context-dependent**:

- `"St."` → *Street* or *Saint* depending on surrounding words.
- `"1984"` → *nineteen eighty-four* (a year) vs. *one thousand nine hundred
  eighty-four* (a count, as in `"1984 people"`).
- A `"3"` in a date is an ordinal (*third*); elsewhere a cardinal (*three*).
- Roman numerals (`"Chapter IV"`) flip between numbers and ordinary letters
  depending on context.

A rule engine can handle the easy cases but cannot disambiguate the hard ones.
We make this concrete with a **context-blind rule baseline**
(`src/audiobook_ai/models/baseline_rules.py`: ordered regex spans + an
abbreviation dictionary + `num2words`/pure-Python expanders). On the synthetic
test distribution it scores:

| Slice | Baseline sentence exact-match (EM) |
|---|---|
| Easy | **0.945** |
| Hard (injected ambiguity) | **0.006** |
| Overall | **0.712** |

The collapse on the hard slice (0.006) is the empirical proof that rules are not
enough. The trained, **context-aware** model is expected to hold high accuracy on
*both* slices, beating the baseline overall and especially on the hard slice.
This is the trainable ML heart of the project.

### 4.2 Segmentation & structure are linguistic, not lexical

Splitting a book into the right spoken units — sentence boundaries (`pysbd`),
TTS-sized chunks, and segment *roles* (narration vs. dialogue vs. heading vs.
skippable) — requires language-aware segmentation, not byte counting. The role a
segment plays then drives voice routing and prosody.

### 4.3 Agentic orchestration of a noisy, real-world pipeline

Real documents are messy (scanned PDFs, garbled text, inconsistent structure),
and every stage can fail. The system therefore runs as a **deterministic finite
state machine with an optional LLM brain**, exposing **four decision points**:

| ID | Decision | Logic |
|---|---|---|
| **D1** | Parse-quality routing | `parse_score ∈ [0,1]` from alpha-ratio + structure signal + segment-length sanity → structured (≥0.85) / assisted (0.5–0.85) / degraded (<0.5). |
| **D2** | Normalization-confidence escalation | Per-segment confidence = length-normalized sequence probability; below `norm_confidence_min` (0.55) → flag and optionally escalate to the LLM brain (validates, falls back to rules); records neural-vs-baseline disagreement. |
| **D3** | Audio-QA re-synthesis gate | Per-clip checks (empty/NaN, duration ratio, peak/clipping, silence fraction); fail → bounded re-synth (max 2 attempts; reseed → split → fallback backend), then accept-best + flag. |
| **D4** | Voice routing | Heading / dialogue / narration → distinct x-vector voices, stable per book. |

Every step is timed and traced (`ToolTrace`); every decision is recorded
(`Decision`); the full run is captured in `manifest.json`. The LLM brain is
**off by default** (zero paid API, CPU-only). This orchestration — routing,
confidence escalation, bounded retries, full audit — is precisely the kind of
multi-step reasoning over noisy language that a fixed script cannot encode.

---

## 5. The Model & Backends (context)

- **Trainable model:** a Text-Normalization seq2seq model, `google/byt5-small`
  (apache-2.0, ~300M, byte/char-level → robust to numbers, symbols, OOV), with
  task prefix `"normalize: "`. Fallback on small GPUs: `google-t5/t5-small`
  (apache-2.0, 60.5M).
- **Training data:** a reproducible **synthetic corpus generator**
  (`src/audiobook_ai/data/tn_corpus.py`) producing context-rich
  (written, spoken) pairs across all 16 classes with **injected ambiguity**,
  because no permissively-licensed English TN-Challenge mirror exists on the HF
  Hub. Leakage-free splits (train 60,000 / val 4,000 / test 4,000 / hard 1,500).
- **TTS backend (pretrained, not trained):** `microsoft/speecht5_tts` (MIT,
  ~144M, 16kHz) + `microsoft/speecht5_hifigan` vocoder + speaker x-vectors
  `Matthijs/cmu-arctic-xvectors` (MIT). Graceful fallbacks down to a
  deterministic placeholder so the pipeline never hard-fails.

---

## 6. Success Metrics

Success is measured on two axes: the **business value** delivered and the
**technical quality** of the output.

### 6.1 Business metrics (cost / time / efficiency / accessibility)

- **Cost per finished audio-hour** — must be *far below* human narration.
- **Accessibility reach** — content made available to sight-impaired, dyslexic,
  and on-the-go listeners.
- **Time-to-first-audio** — stream Chapter 1 while later chapters render, rather
  than waiting for the whole book.
- **Catalog conversion speed** — how fast a publisher can convert a backlist.

### 6.2 Technical metrics

| Metric | Target / definition |
|---|---|
| **TN sentence accuracy** | Sentence-level exact-match; must **beat the rule baseline** (overall 0.712, hard 0.006). |
| **Macro per-class accuracy** | Mean accuracy across the 16 semiotic classes — exposes rare-class failures. |
| **Unrecoverable-error rate** | Sproat & Jaitly "silly"/catastrophic mis-reads driven toward **0**, even at the cost of benign errors. |
| **RTF (real-time factor)** | Synthesis latency reported per run in `manifest.json`. |
| **Audio-QA pass rate** | Fraction of clips passing D3 checks without re-synth. |
| **Loudness conformance** | ACX broadcast master: **−18 LUFS integrated / −3 dBTP peak**. |

### 6.3 Validated end-to-end evidence

A real end-to-end run (real SpeechT5, CPU) processed **3 chapters / 7 spoken
segments**, flagged **0**, fired **all 4 decisions**, and produced a **67 s WAV +
SRT + manifest** at **RTF 2.28 on CPU** (≈0.1 on GPU). This confirms the pipeline,
the agent decisions, the loudness master, and the export formats all work
together as specified.
