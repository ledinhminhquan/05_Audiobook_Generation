# Agent Architecture — Audiobook Generation (P05)

**Project:** Audiobook Generation System — turn long documents (EPUB / PDF / TXT / MD) into mastered, chaptered audiobooks (`.wav` / `.mp3` / `.m4b` + `.srt`).
**Author:** Le Dinh Minh Quan (23127460) — *NLP in Industry*, final assignment.
**Reference repo:** `github.com/denizsafak/abogen`.

This document describes the **mandatory agentic component**: a **deterministic finite-state machine (FSM)** that orchestrates the full `document -> audiobook` pipeline, with an **optional LLM brain** that only ever *validates and falls back*. The agent is graded heavily, so the FSM, the `JobState` context, the four decision points (D1–D4), the tool contracts, and the audit trail are all spelled out below, followed by an ASCII flow diagram and a fully worked example interaction.

---

## 1. Design philosophy

The agent is **deterministic-first**. Audiobook production must be reproducible, auditable, and runnable with **zero paid API and CPU-only** (LLM **OFF by default**). Every transition is a pure function of the current `JobState` plus tool outputs; the same input book yields the same decisions and the same manifest. The optional Anthropic LLM brain is a *bounded escalation path* — it is consulted only at D2, it must return a structurally valid result, and on any failure (no key, timeout, malformed output, low confidence) the agent **falls back to the rule normalizer**. The LLM can never take the pipeline off its rails.

Why an FSM and not a free-form agent loop? The pipeline is a known, ordered sequence of stages (parse -> chapter -> segment/classify -> normalize -> voice-route -> synthesize -> audio-QA -> stitch -> export). The "intelligence" lives in **four well-defined decision points** where the path branches on measured quality signals, plus **bounded re-synthesis** with escalating strategies. This gives agentic behavior (routing, escalation, self-correction, fallback, tool use) while staying testable and graceful under degradation.

---

## 2. The FSM

States correspond to pipeline stages. Each state runs one or more **tools** (Section 4), writes its outputs into `JobState`, may consult a **decision point** (Section 5) to choose the next branch, and appends a `ToolTrace` (timing) and, where applicable, a `Decision` to the audit log.

```
PARSE -> CHAPTER -> SEGMENT/CLASSIFY -> NORMALIZE -> VOICE_ROUTE
      -> SYNTHESIZE -> AUDIO_QA -> STITCH -> EXPORT -> DONE
```

| State | Tool(s) | Decision | Branches / effect |
|-------|---------|----------|-------------------|
| `PARSE` | `tool_parse` | **D1** | structured / assisted / degraded path |
| `CHAPTER` | (in `tool_parse`) | — | TOC / font-size / "Chapter N" regex |
| `SEGMENT/CLASSIFY` | `tool_parse` (pysbd + chunking) | — | label each segment narration\|dialogue\|heading\|skippable |
| `NORMALIZE` | `tool_normalize` | **D2** | accept / flag / escalate-to-LLM / baseline fallback |
| `VOICE_ROUTE` | `tool_route_voices` | **D4** | assign x-vector voice per segment |
| `SYNTHESIZE` | `tool_synthesize` | — | render each segment to audio |
| `AUDIO_QA` | `tool_audio_qa` | **D3** | accept / bounded re-synth / accept-best+flag |
| `STITCH` | `tool_stitch` | — | gaps + ACX loudness master |
| `EXPORT` | `tool_stitch` (+ subtitles) | — | wav/mp3/m4b + srt + manifest.json |

The FSM is implemented in `src/audiobook_ai/agent/policy.py` (transitions + decision logic), `state.py` (the context), `tools.py` (the tool contracts), `llm_orchestrator.py` (the optional brain), and driven by `narrator_agent.py`.

---

## 3. The `JobState` context

`JobState` (in `agent/state.py`) is the single mutable context threaded through every state. It accumulates evidence so each decision is a pure function of what is known so far. Key fields:

| Field | Type | Written by | Used by |
|-------|------|-----------|---------|
| `input_path`, `input_sha256` | str | `PARSE` | manifest, copyright record |
| `parse_score` | float `[0,1]` | `tool_parse` | **D1** |
| `parse_mode` | enum `structured\|assisted\|degraded` | **D1** | downstream chaptering/segmentation |
| `chapters[]` | list | `CHAPTER` | stitch chapter markers |
| `segments[]` | list of `{text, label, chapter}` | `SEGMENT/CLASSIFY` | normalize, route, synth |
| `normalized[]` | list of `{spoken, confidence, source}` | `tool_normalize` | **D2**, synth |
| `flags[]` | list of `{segment_id, reason}` | D2 / D3 | manifest QA report |
| `voice_map` | dict `label -> xvector_idx` | `tool_route_voices` | **D4**, synth |
| `clips[]` | list of `{path, qa, attempts}` | synth / QA | **D3**, stitch |
| `traces[]` | list of `ToolTrace` | every tool | audit, RTF |
| `decisions[]` | list of `Decision` | D1–D4 | audit, manifest |
| `manifest` | dict | `EXPORT` | final `manifest.json` |

`source` on a normalized segment records provenance: `neural` (trained ByT5), `llm` (validated Anthropic output), or `baseline` (rule normalizer fallback). This is what makes the D2 path auditable.

---

## 4. Tool contracts

Tools are deterministic functions with explicit input/output schemas. Each emits a `ToolTrace` (name, start, end, duration, ok/err). They live in `agent/tools.py` and wrap the pipeline modules.

| Tool | Input | Output | Notes |
|------|-------|--------|-------|
| `tool_parse` | `input_path` | `text`, `chapters[]`, `segments[]` (labelled), `parse_score` | ebooklib (EPUB) / PyMuPDF + pdfplumber (PDF) / txt+md; pysbd sentence split + TTS chunking; computes the D1 score |
| `tool_normalize` | `segment.text` | `spoken`, `confidence`, `source` | trained ByT5 (`google/byt5-small`, prefix `"normalize: "`); confidence = length-normalized sequence probability; records neural-vs-baseline disagreement |
| `tool_route_voices` | `segment.label`, `voice_map` | `xvector_idx` | maps narration / dialogue / heading -> distinct x-vector indices; stable per book (D4) |
| `tool_synthesize` | `spoken`, `xvector_idx`, `strategy` | `clip_path`, `sr` | SpeechT5 (`microsoft/speecht5_tts` + `speecht5_hifigan`, 16 kHz, x-vectors from `Matthijs/cmu-arctic-xvectors`); `strategy` lets D3 reseed / split / swap backend |
| `tool_audio_qa` | `clip_path`, `expected_dur` | `qa = {empty, nan, dur_ratio, peak, silence_frac}`, `pass` | per-clip checks; feeds D3 |
| `tool_stitch` | `clips[]`, `chapters[]` | mastered audio + `.srt` + `manifest.json` | inserts silence gaps; ACX master **-18 LUFS / -3 dBTP**; exports wav/mp3/m4b with chapter markers |

**Backend fallback chain** (used by `tool_synthesize`, escalated by D3):
`SpeechT5` (primary, MIT) -> optional quality (`hexgrad/Kokoro-82M`, `parler-tts/parler-tts-mini-v1`) -> `pyttsx3` (offline OS TTS) -> `PlaceholderTTS` (deterministic low-noise floor so the pipeline **never hard-fails** and is testable with no model).

---

## 5. Decision points (D1–D4)

The assignment requires **≥3** decision points; **we have 4**. Each is a pure function of `JobState`, records a `Decision` to the audit log, and has an explicit fallback.

### D1 — Parse-quality routing
- **Where:** end of `PARSE`.
- **Input:** `parse_score ∈ [0,1]`, computed from **alpha-ratio** (fraction of alphabetic characters vs. garbage/OCR noise) + **structure signal** (presence of TOC / chapter headings) + **segment-length sanity** (sentences neither degenerate-short nor runaway-long).
- **Thresholds / branches:**

  | `parse_score` | Branch | Behavior |
  |---------------|--------|----------|
  | `>= 0.85` | `structured` | trust TOC + headings, full chaptering & classification |
  | `0.5 – 0.85` | `assisted` | use regex/font-size heuristics, looser trust in structure |
  | `< 0.5` | `degraded` | flat-text path: treat as one stream, minimal structure (e.g. scanned/garbled PDFs) |

- **Fallback:** the `degraded` branch is itself the graceful floor — the book is still narrated, just without rich chaptering.

### D2 — Normalization-confidence escalation
- **Where:** per segment in `NORMALIZE`.
- **Input:** per-segment **confidence = length-normalized sequence probability** of the ByT5 generation; also the **neural-vs-baseline disagreement** signal.
- **Threshold:** `norm_confidence_min = 0.55`.
- **Branches:**
  - `confidence >= 0.55` -> **accept** neural output (`source = neural`).
  - `confidence < 0.55` -> **flag** the segment; **optionally escalate** to the LLM brain (Anthropic). The brain re-normalizes; its output is **validated** (must be well-formed spoken text consistent with the written tokens). If valid -> use it (`source = llm`). If the LLM is off / unavailable / invalid / still low-confidence -> **fall back to the rule normalizer** (`source = baseline`).
- **Fallback:** rule normalizer (`baseline_rules.py`) — never blocks. LLM is **OFF by default** (zero paid API, CPU-only).

### D3 — Audio-QA re-synthesis gate
- **Where:** per clip in `AUDIO_QA`.
- **Input:** per-clip checks — `empty / NaN`, **duration ratio** vs. expected (from token count / spoken length), **peak / clipping**, **silence fraction**.
- **Branch:** any check fails -> **bounded re-synthesis**, **max 2 attempts**, with an **escalating strategy**:
  1. **reseed** (same backend, new generation seed),
  2. **split** (break the segment into shorter chunks),
  3. **fallback backend** (drop down the backend chain).
- **Terminal:** after attempts are exhausted -> **accept-best** clip (lowest-defect attempt) and **flag** it in the manifest QA report.
- **Fallback:** accept-best + flag guarantees forward progress even if synthesis stays imperfect.

### D4 — Voice routing
- **Where:** `VOICE_ROUTE`.
- **Input:** segment `label` (`heading` / `dialogue` / `narration`).
- **Branch:** map each label to a **distinct x-vector index** so headings, dialogue, and narration get different voices.
- **Stability:** the `voice_map` is fixed **per book** so a character/role keeps one voice throughout.
- **Fallback:** if only one voice is available (e.g. minimal backend), all labels collapse to the narrator voice without failing.

---

## 6. Optional LLM brain

- **Engine:** Anthropic (consulted **only** at D2).
- **Role:** *validate and fall back* — it re-normalizes a low-confidence segment, its output is validated for well-formedness/consistency, and any failure routes to the rule normalizer.
- **Default:** **OFF** — the entire system runs **CPU-only with zero paid API**. The brain is a bounded enhancement, never a dependency. This keeps the agent reproducible and the grading run free.

---

## 7. Audit trace & manifest

Every step is **timed and traced** (`ToolTrace`) and every branch is **recorded** (`Decision`). On `EXPORT`, `tool_stitch` writes a full **`manifest.json`** capturing: `input_sha256` (copyright/provenance), chapters, per-segment normalization `source` + confidence, all D1–D4 decisions, all flags, per-clip QA + re-synth attempts, loudness conformance (`-18 LUFS / -3 dBTP`), and **RTF** (real-time factor) for latency reporting. The manifest is the single audit artifact a reviewer (or the `grade` CLI) inspects to verify what the agent did and why.

**Validated end-to-end run** (real SpeechT5, CPU): 3 chapters, 7 spoken segments, **0 flagged**, **all 4 decisions fired**, produced a 67 s WAV + SRT + manifest, **RTF 2.28 on CPU** (~0.1 expected on GPU).

---

## 8. ASCII flow diagram

```
                 ┌──────────────────────────── INPUT: EPUB / PDF / TXT / MD ─┐
                 │  record input_sha256 (copyright / provenance)             │
                 └───────────────────────────────┬───────────────────────────┘
                                                  ▼
                                          ┌───────────────┐
                                          │   tool_parse  │  ebooklib / PyMuPDF
                                          │  + chaptering │  + pdfplumber / txt+md
                                          └───────┬───────┘
                                                  ▼
                                    ╔═════════ D1  parse_score ═════════╗
                                    ║  >=0.85 structured                ║
                                    ║  0.5–0.85 assisted                ║
                                    ║  <0.5  degraded (flat-text)       ║
                                    ╚═════════════════╤═════════════════╝
                                                      ▼
                                 ┌────────────────────────────────────────┐
                                 │ SEGMENT/CLASSIFY (pysbd + chunking)     │
                                 │ label: narration|dialogue|heading|skip  │
                                 └────────────────────┬───────────────────┘
                                                      ▼
                                          ┌───────────────────┐
                                          │  tool_normalize   │  trained ByT5
                                          │  -> spoken, conf  │  "normalize: "
                                          └─────────┬─────────┘
                                                    ▼
                              ╔═══════════ D2  confidence < 0.55 ? ═══════════╗
                              ║  no  -> accept (source=neural)                ║
                              ║  yes -> flag; LLM brain (validate)            ║
                              ║          valid -> source=llm                  ║
                              ║          else  -> baseline rules (fallback)   ║
                              ╚═══════════════════════╤═══════════════════════╝
                                                      ▼
                                          ┌───────────────────┐
                                          │ tool_route_voices │
                                          └─────────┬─────────┘
                                                    ▼
                              ╔═══════════════ D4  voice routing ═════════════╗
                              ║  heading/dialogue/narration -> x-vector idx   ║
                              ║  stable per book                              ║
                              ╚═══════════════════════╤═══════════════════════╝
                                                      ▼
                                          ┌───────────────────┐
                                          │  tool_synthesize  │  SpeechT5 16kHz
                                          └─────────┬─────────┘
                                                    ▼
                                          ┌───────────────────┐
                                          │   tool_audio_qa   │  empty/NaN, dur,
                                          │                   │  peak, silence
                                          └─────────┬─────────┘
                                                    ▼
                  ╔═════════════════════ D3  QA pass ? ════════════════════════╗
                  ║  pass -> accept clip                                       ║
                  ║  fail -> bounded re-synth (<=2): reseed -> split -> backend║
                  ║          exhausted -> accept-best + flag                   ║
                  ╚════════════════════════════╤═══════════════════════════════╝
                                               ▼
                                     ┌───────────────────┐
                                     │    tool_stitch    │  silence gaps +
                                     │  ACX -18 LUFS/-3dB│  chapter markers
                                     └─────────┬─────────┘
                                               ▼
                    EXPORT: wav / mp3 / m4b(+chapters) / srt + manifest.json
                    (ToolTrace timings + D1–D4 Decisions + flags + RTF)
```

---

## 9. Worked example interaction (the sample book)

Input: a short public-domain book; we follow one heading segment, one narration sentence containing the written date **"March 3, 1921"**, and one line of dialogue.

**1. PARSE + D1.** `tool_parse` reads the EPUB with ebooklib, detects a TOC and "Chapter N" headings, and computes `parse_score = 0.93` (high alpha-ratio, clear structure, sane sentence lengths).
> **D1 decision:** `0.93 >= 0.85` -> `parse_mode = structured`. Full chaptering retained (3 chapters). `Decision(D1, score=0.93, branch=structured)` logged.

**2. SEGMENT/CLASSIFY.** pysbd splits sentences; chunker labels:
- `"Chapter One"` -> `heading`
- `"He returned on March 3, 1921, to an empty house."` -> `narration`
- `"\"You're late,\" she said."` -> `dialogue`

**3. NORMALIZE + D2.** `tool_normalize` runs trained ByT5 on each segment.
- Narration: ByT5 emits `"He returned on March third, nineteen twenty-one, to an empty house."` — note `1921` is read as a **year** ("nineteen twenty-one"), and the day **`3`** is read as the **ordinal** "third", because the model used the date context. Confidence `0.88`.
> **D2 decision:** `0.88 >= 0.55` -> **accept**, `source = neural`. (A context-blind baseline would risk "March three, one thousand nine hundred twenty-one" — exactly the failure the trained model fixes.)
- Dialogue normalizes cleanly, confidence `0.91` -> accept.
- Heading is plain, confidence `0.97` -> accept. `Decision(D2, ...)` logged per segment; **0 flagged**.

**4. VOICE_ROUTE + D4.** `tool_route_voices` builds a stable `voice_map`:
`heading -> xvector[heading_idx]`, `narration -> xvector[narrator_idx]`, `dialogue -> xvector[dialogue_idx]` (distinct CMU-Arctic x-vectors).
> **D4 decision:** three distinct voices assigned; map fixed for the whole book. `Decision(D4, voice_map=...)` logged.

**5. SYNTHESIZE.** `tool_synthesize` renders each spoken string with SpeechT5 + HiFi-GAN at 16 kHz using the routed x-vector. The narration clip says "nineteen twenty-one", confirming the normalization carried through to audio.

**6. AUDIO_QA + D3.** `tool_audio_qa` checks each clip: not empty / no NaN, `dur_ratio` within bounds, peak below clipping, silence fraction normal.
> **D3 decision:** all clips **pass** on the first attempt -> accept, no re-synth. (Had the narration clip been truncated, D3 would have tried: reseed -> split -> fallback backend, up to 2 attempts, then accept-best + flag.) `Decision(D3, pass=true)` logged.

**7. STITCH + EXPORT.** `tool_stitch` inserts silence gaps, masters to **-18 LUFS / -3 dBTP** (ACX), writes chapter markers, and exports `.wav` / `.mp3` / `.m4b` + `.srt`. The manifest records `input_sha256`, the 3 chapters, each segment's `source`/confidence, all four `Decision` records, zero flags, loudness conformance, and **RTF 2.28 (CPU)**.

**Net result:** every decision point fired, the hard NLP case (`March 3, 1921` -> "March third, nineteen twenty-one") was handled by the trained context-aware normalizer, and the agent produced an auditable, mastered audiobook with **no human intervention and no paid API**.
