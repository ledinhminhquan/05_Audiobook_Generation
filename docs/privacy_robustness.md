# Data Privacy & Model Robustness

**Project:** Audiobook Generation System (P05) — turn long documents (EPUB/PDF/TXT/MD) into mastered, chaptered audiobooks (`.wav`/`.mp3`/`.m4b` + `.srt`).
**Author:** Le Dinh Minh Quan (23127460) — *NLP in Industry*, final assignment.

This document covers how the system handles **personally identifiable information (PII)** that naturally appears inside books, and how it stays **robust** to noisy, adversarial, and out-of-distribution (OOD) inputs. The guiding principle is the same one that motivates the whole project: a document-to-audiobook pipeline must **never hard-fail and never leak** — it degrades gracefully, and it minimizes what it stores.

---

## 1. Why PII shows up here at all

The trainable component is a **Text-Normalization (TN)** seq2seq model that converts written tokens to spoken form across 16 semiotic classes. Three of those classes are, by construction, the places where PII surfaces inside ordinary prose:

| Semiotic class | Typical written form | PII risk |
|----------------|----------------------|----------|
| `TELEPHONE`    | `+1 (415) 555-0132`  | Phone numbers |
| `ELECTRONIC`   | `jane.doe@example.com`, URLs, handles | Emails, accounts, addresses-as-URLs |
| `ABBREV`       | `Dr.`, `St.`, titles preceding names | Names, honorifics, street addresses |

In addition, the segment classifier labels spans as `narration | dialogue | heading | skippable`, and the **D4 voice-routing** decision point uses light **named-entity recognition (NER) for attribution only** — to decide *who is speaking* so dialogue gets a distinct voice (x-vector index). NER here is a routing signal, **not** an extraction or profiling step: entities are used to pick a voice and then discarded.

A book is also someone's intellectual property. Beyond PII, the system records the input file's **SHA-256** for provenance/rights tracking (converting a book is a derivative work), but it does **not** retain the raw content beyond what the run needs.

---

## 2. PII handling principles

### 2.1 Data minimization
- **Only what the run needs.** Text is parsed, segmented, normalized, synthesized, stitched, and exported. Once artifacts (`.wav`/`.mp3`/`.m4b`/`.srt` + `manifest.json`) are produced, intermediate text buffers carrying `TELEPHONE`/`ELECTRONIC`/`ABBREV` spans are not needed and are not persisted long-term.
- **NER is attribution-scoped.** Entities feed D4 voice routing and are not written to a separate "people in this book" store.

### 2.2 No raw PII in logs
- The pipeline is heavily traced — **every step is timed (`ToolTrace`) and every decision recorded (`Decision`)** for the audit trail and `manifest.json`. These traces record **metrics and decisions**, not raw text: parse scores, per-segment normalization **confidence**, neural-vs-baseline **disagreement flags**, audio-QA verdicts, RTF — never the literal phone number or email that triggered them.
- A segment that contains a `TELEPHONE`/`ELECTRONIC` span is referenced by index and class, so logs stay useful for debugging without becoming a PII sink.

### 2.3 TTL cleanup
- Intermediate working data carrying PII (parsed text, per-segment buffers) is subject to **time-to-live (TTL) cleanup** so it does not accumulate on disk after the audiobook is delivered.

### 2.4 Optional redaction / skip
- The segment classifier already produces a **`skippable`** label; segments dominated by sensitive `TELEPHONE`/`ELECTRONIC` content can be routed to **skip** or to **redaction** before synthesis, so the spoken output does not read out a phone number or email aloud when that is undesired.
- Because this is optional and per-segment, it composes with the rest of the FSM without special-casing.

---

## 3. Robustness to noisy & adversarial input

### 3.1 Char/byte-level model = robust by design
The TN model is **`google/byt5-small`** (VERIFIED, apache-2.0, ~300M), a **byte/character-level** seq2seq model with the task prefix `"normalize: "`. Operating on bytes rather than a fixed word vocabulary makes it inherently **robust to numbers, symbols, OOV tokens, typos, and odd formats** — exactly the garbage that real documents contain (`$5.2M`, `Chapter IV`, `9:45 AM`, `3/4`, `1984`). On small GPUs the fallback base is **`google-t5/t5-small`** (VERIFIED, apache-2.0, 60.5M).

Robustness is also **trained for**: the synthetic corpus generator (`src/audiobook_ai/data/tn_corpus.py`) deliberately **injects ambiguity** — `St.`→Street vs Saint, `1984 people` (count) vs `in 1984` (year), dates with day-as-ordinal, Roman-numeral contexts the rule baseline misses — and holds out a **hard slice** (default 1,500 examples) plus leakage-free splits. The payoff is visible against the context-blind baseline:

| Slice | Baseline sentence exact-match (EM) |
|-------|-----------------------------------|
| easy    | **0.945** |
| hard    | **0.006** |
| overall | **0.712** |

The rule baseline is strong on easy cases and collapses on ambiguous ones (`hard = 0.006`); the trained context-aware ByT5 is expected to hold up on **both** slices. We also track **macro per-class accuracy** (exposes rare-class failure) and the **Sproat & Jaitly "unrecoverable / silly error"** rate — semantically catastrophic mis-reads minimized even at the cost of benign ones.

### 3.2 Out-of-distribution documents → degraded path
The agent's **D1 parse-quality routing** computes a **`parse_score` in [0,1]** from alpha-ratio + structure signal + segment-length sanity, and routes:

| `parse_score` | Route |
|---------------|-------|
| ≥ 0.85 | **structured** (full TOC/chapter pipeline) |
| 0.5 – 0.85 | **assisted** |
| < 0.5 | **degraded** (flat-text path) |

OOD inputs — **scanned PDFs**, **garbled OCR**, **unsupported languages** — naturally produce a **low `parse_score`** because their alpha-ratio and structure signals are poor. Instead of crashing or emitting nonsense chapters, the system drops into the **degraded flat-text path**: no chapter detection is forced, the document is treated as plain text, and the pipeline continues to produce *some* audio rather than failing.

### 3.3 Known failure cases (documented, not hidden)
- **Scanned PDFs** with no embedded text layer → low `parse_score` → degraded path; quality is bounded by what the OCR upstream provided.
- **Garbled OCR** (broken ligatures, hyphenation noise) → low `parse_score`; ByT5's byte-level robustness recovers many tokens, but heavily corrupted spans may still mis-normalize and are flagged.
- **Unsupported languages** → low alpha/structure signals → degraded path; the English-trained TN model is not expected to normalize them correctly, so they are surfaced as a degraded run rather than silently mangled.

---

## 4. Mitigation strategies (defense in depth)

The pipeline assumes things will go wrong and layers cheap, deterministic fallbacks so it **never hard-fails**:

1. **Rule guardrail.** The context-blind rule normalizer (`src/audiobook_ai/models/baseline_rules.py`: ordered regex spans + abbreviation dict + `num2words`/pure-Python expanders) is always available. When the neural model is low-confidence or unavailable, normalization falls back to rules.

2. **D2 normalization-confidence escalation.** Per-segment confidence = length-normalized sequence probability. Below **`norm_confidence_min` = 0.55**, the segment is **flagged**, neural-vs-baseline **disagreement is recorded**, and it can optionally escalate to the LLM brain (Anthropic), which **validates and itself falls back to rules**. *(LLM is OFF by default — zero paid API, CPU-only.)*

3. **D3 audio-QA re-synthesis gate.** Each clip is checked for empty/NaN, duration ratio vs. expected, peak/clipping, and silence fraction. A failure triggers **bounded re-synthesis (max 2 attempts)** with an escalating strategy `reseed → split → fallback backend`, then **accept-best + flag**. This is the main guard against **hallucinated pronunciations** — caught at the audio level and counted toward the unrecoverable-error metric.

4. **Graceful degradation everywhere.**
   - TTS backend ladder: **`microsoft/speecht5_tts`** (MIT) → optional quality backends → **`pyttsx3`** (offline OS TTS, no weights) → ultimate floor **`PlaceholderTTS`** (deterministic low-noise) so the pipeline is **testable with no model** and never crashes for lack of weights.
   - Skip `ffmpeg` if absent; D1 degraded flat-text path for unparseable documents.

5. **License & consent hygiene (robustness against misuse).** Voice cloning (`coqui/XTTS-v2` CPML, `SWivid/F5-TTS` CC-BY-NC, `facebook/mms-tts-eng` CC-BY-NC) is **consent-gated, non-commercial, and off by default**; the commercial path uses **only MIT/Apache** ids.

---

## 5. Summary

| Concern | Primary mechanism |
|---------|-------------------|
| PII in `TELEPHONE`/`ELECTRONIC`/`ABBREV` spans | Minimization, no raw PII in logs (only metrics/decisions), TTL cleanup, optional redaction/skip |
| Name attribution | NER for voice routing (D4) **only**, then discarded |
| Provenance / rights | Input SHA-256 recorded; derivative-work check |
| Typos / odd formats / OOV | Byte-level **ByT5** + ambiguity-injected training + hard slice |
| OOD docs (scanned PDF, garbled OCR, other languages) | Low `parse_score` → **degraded flat-text path** (D1) |
| Hallucinated pronunciations | Rule guardrail (D2) + **audio-QA re-synth gate (D3)** + unrecoverable-error counting |
| Never hard-fail | Backend ladder down to **PlaceholderTTS**; skip-`ffmpeg`; degraded path |

The system is built so that the **worst case is a flagged, degraded, but complete and private run** — not a leak and not a crash.
