<!-- DESIGN BRIEF — P05 Audiobook Generation. Single source of truth (verified research).
     Note: this project mirrors the P02/P03/P04 repo template (src/audiobook_ai/{data,models,
     synthesis,training,agent,api,analysis,autoreport,monitoring,automation,grading}); the module
     names in section 8 are the conceptual stage map, realised inside that template layout. -->

# P05 — Audiobook Generation — Design Brief

> **Status:** Single source of truth for implementation. Every model/dataset id below is marked **VERIFIED** against the Hugging Face Hub. The trainable core is a **ByT5-small Text-Normalization seq2seq** model with a deterministic rule-based baseline to beat; audio is produced by a pretrained neural TTS backend (**SpeechT5** primary, CPU-demo-able); a deterministic agent FSM orchestrates `document → audiobook`.

---

## 1. Problem & business value

**Problem.** Turning long-form documents (EPUB/PDF/TXT/MD) into listenable audiobooks fails on two fronts: (a) **text normalization** — naive TTS reads `$5.2M`, `1984`, `Dr.`, `Chapter IV`, `3/4`, `555-0142` literally or wrongly, which is jarring and sometimes *unrecoverable* for a listener; and (b) **production orchestration** — chapter detection, sentence-safe chunking, multi-voice routing, loudness mastering, and chapter-marked exports are tedious and error-prone. P05 solves both: a **trainable TN model** fixes the reading, and an **agentic pipeline** turns a raw document into a mastered `.m4b`/`.mp3` with synchronized `.srt`.

**Business value.** Self-serve audiobook production at a fraction of human-narration cost; accessibility (sight-impaired, dyslexic, commuters); rapid catalog conversion for publishers; read-along subtitles for language learners. The permissive default stack (SpeechT5/Kokoro + ByT5, all MIT/Apache-2.0) keeps a **commercial-safe path** open.

### Success metrics

**Business**
| Metric | Target |
|---|---|
| Cost per finished audio-hour | ≪ human narration (compute-only) |
| Time-to-first-audio (10-h book) | < a few seconds (stream chapter 1 first) |
| Wall-clock for a 10-h book | minutes (chunk-parallel) to ~1 h (single stream) |
| Listener-rated naturalness (MOS-style) | acceptable on default voice; high on Kokoro/XTTS |
| Unrecoverable-error rate (listener-fatal mis-reads) | → 0 per chapter |

**Technical**
| Metric | Definition | Target |
|---|---|---|
| **RTF** (Real-Time Factor) | `synth_wall_time / audio_duration` | Kokoro ~0.03–0.10 (GPU); XTTS-v2 ~0.3–0.6; SpeechT5 CPU ~0.8–1.5 — reported per-job in `manifest.json` |
| **TN sentence accuracy** | exact-match of full normalized string vs reference | primary headline number; must **beat the rule baseline** |
| **TN macro-class accuracy** | mean per-semiotic-class exact-match (16 classes) | surfaces rare-class failures hidden by PLAIN-heavy micro acc |
| **TN unrecoverable/silly errors** | manual count of semantically catastrophic mis-reads (Sproat & Jaitly) | minimized even at cost of benign errors |
| **Audio-QA pass rate** | clips passing loudness/peak/duration/empty checks first try | high; bounded re-synth budget `N=3` |
| **Loudness conformance** | integrated LUFS / true-peak vs target | default **−18 LUFS, −3 dBTP** (ACX-safe); podcast preset −16 LUFS |

---

## 2. VERIFIED stack table

> Every id below was confirmed live on the Hub. **Non-permissive ids are flagged and must stay out of any commercial path.**

### Text-Normalization dataset
| Id / source | Role | License | Status |
|---|---|---|---|
| **Synthetic generator** (`num2words` + templates, §4) | **PRIMARY training data** — self-contained, license-clean | code (Apache/MIT-clean) | **VERIFIED path** (no Hub dep) |
| `DigitalUmuganda/Text_Normalization_Challenge_Unittests_Eng_Fra` | English eval/sanity set (<1K rows) | (Hub) | **VERIFIED** — too small to train, good eval |
| `github.com/rwsproat/text-normalization-data` / Kaggle `google-nlu/text-normalization` | Optional scale-up; canonical Sproat & Jaitly, 1.1B words, **16 semiotic classes** | restrictive / off-Hub | **VERIFIED off-Hub** (not Hub-loadable) |
| `shubham-Bgs/Text-Normalization-Hindi` | **Schema reference** — exact `input`/`target` columns | apache-2.0 | **VERIFIED** (Hindi; schema only) |
| `google/text_normalization`, `cestwc/text-normalization`, `moinbach7/asr_en_text_normalization` | — | — | **DO NOT EXIST (404)** — never reference |
| `pavanBuduguppa/asr_inverse_text_normalization` | — | GPL-3.0 | **VERIFIED but REJECT** (ITN, copyleft, broken viewer) |

### TN base model
| Id | Role | License | Params | Status |
|---|---|---|---|---|
| `google/byt5-small` | **PRIMARY TN model** — byte/char-level, robust to numbers/symbols/OOV | apache-2.0 | ~300M (byte vocab≈384) | **VERIFIED** |
| `google-t5/t5-small` | Secondary speed baseline (subword) | apache-2.0 | 60.5M | **VERIFIED** (legacy `t5-small` redirects here) |
| `google/flan-t5-small` | Considered; marginal over t5-small once fine-tuned | apache-2.0 | 77.0M | **VERIFIED** |

### TTS backends
| Id | Role | License | Params | SR | Status |
|---|---|---|---|---|---|
| `microsoft/speecht5_tts` | **PRIMARY TTS** — CPU-demo-able, permissive, multi-speaker via x-vector | **MIT** | ~144M | 16 kHz | **VERIFIED** |
| `microsoft/speecht5_hifigan` | Vocoder for SpeechT5 | **MIT** | ~13M | 16 kHz | **VERIFIED** |
| `hexgrad/Kokoro-82M` | **Optional quality upgrade** — permissive, CPU-friendly, native 24 kHz | apache-2.0 | 82M | 24 kHz | **VERIFIED** |
| `hexgrad/Kokoro-82M-v1.1-zh` | Multilingual (Mandarin) Kokoro variant | apache-2.0 | 82M | 24 kHz | **VERIFIED** |
| `parler-tts/parler-tts-mini-v1` | Optional permissive heavy — prompt-styled voices | apache-2.0 | 877.8M | 44.1 kHz | **VERIFIED** |
| `parler-tts/parler-tts-large-v1` | Larger prompt-controlled character voices | apache-2.0 | 2.33B | — | **VERIFIED** |
| `suno/bark-small` | Optional expressive/preset voices | **MIT** | ~400M | 24 kHz | **VERIFIED** |
| `nari-labs/Dia-1.6B` | Optional two-voice dialogue (`[S1]/[S2]` tags) | apache-2.0 | 1.61B | — | **VERIFIED** |
| `canopylabs/orpheus-3b-0.1-ft` | Optional expressive (gated, GPU-only) | apache-2.0 | 3.78B | — | **VERIFIED** (gated) |
| `coqui/XTTS-v2` | Voice-clone (6 s ref) — **NON-COMMERCIAL** | **other / CPML** | ~460M–1.8B | 24 kHz | **VERIFIED — avoid commercial** |
| `facebook/mms-tts-eng` | Tiny multilingual fallback — **NON-COMMERCIAL** | **CC-BY-NC-4.0** | 36.3M | 16 kHz | **VERIFIED — avoid commercial** |
| `SWivid/F5-TTS` | High-fidelity clone (alt) — **NON-COMMERCIAL** | cc-by-nc-4.0 | — | — | **VERIFIED — non-commercial only** |
| `sesame/csm-1b` | Context-aware prosody (gated, alt) | apache-2.0 | 1.55B | — | **VERIFIED** (gated) |
| `pyttsx3` (PyPI) | **Last-resort offline fallback** (OS SAPI5/espeak) | BSD-style | n/a | OS | **VERIFIED** (no weights, never hard-fails) |

### Speaker embeddings
| Id | Role | License | Status |
|---|---|---|---|
| `Matthijs/cmu-arctic-xvectors` | 7,931 × 512-d x-vectors → SpeechT5 speaker voices (7 speakers) | **MIT** | **VERIFIED** |

### Baseline libraries (the system the model must beat)
| Lib | Role | Note |
|---|---|---|
| `num2words` | cardinal/ordinal/year/currency verbalization | core of rule baseline & synthetic generator |
| `inflect` | pluralization, articles, ordinals | ADDRESS/MEASURE phrasing |
| `nemo_text_processing` | NVIDIA WFST/Pynini deterministic TN | strongest production reference baseline (heavy install; conda py3.10) |

### Parsing libraries
| Lib (pip id) | Role |
|---|---|
| `EbookLib` (`ebooklib`) + `beautifulsoup4` | EPUB spine → reading order, headings/paragraphs |
| `PyMuPDF` (`fitz`) | PDF primary: span `size`/`font`/`bbox`, outline `get_toc()` |
| `pdfplumber` | PDF fallback: word-bbox for columns/tables |
| `Markdown` + `charset_normalizer`/`chardet` | md/txt decode + render |
| `pysbd` | sentence segmentation (97.92% GRS); alts `nltk`, `blingfire` (speed); abogen uses `spacy` |
| `dslim/bert-base-NER` (MIT, **VERIFIED**) | PERSON entities → speaker attribution / normalization disambiguation |

### Audio libraries
| Lib | Role |
|---|---|
| `soundfile` + `numpy` | per-segment WAV I/O (24 kHz Kokoro / 16 kHz SpeechT5) |
| `pydub` (+ `ffmpeg`) | stitch/concat/silence/export |
| `static_ffmpeg` | bundles ffmpeg binary |
| `pyloudnorm` | in-process EBU R128 / BS.1770-4 metering |
| `ffmpeg loudnorm` (two-pass) / `ffmpeg-normalize` | post-processing loudness (better for mastering) |
| `mutagen` | m4b/mp3 tags, chapters (`CHAP`/`CTOC`), cover art |
| `MahmoudAshraf/mms-300m-1130-forced-aligner` (**VERIFIED**, cc-by-nc-4.0) | word-level SRT alignment — **NC**; swap to torchaudio MMS aligner for commercial |
| `pyannote/speaker-diarization-3.1` (MIT, **VERIFIED**, gated) | optional audio QA: verify single-speaker dialogue clips |

---

## 3. System pipeline

```
                          ┌───────────────── manifest.json (reproducible build record) ─────────────────┐
                          │                                                                              │
 INPUT (epub/pdf/txt/md)  │                                                                              │
        │                 ▼                                                                              ▼
        ▼      ┌──────────────────┐   ┌──────────────┐   ┌──────────────┐   ┌────────────────────────────┐
 ┌────────────┐│   PARSE          │   │  CHAPTER     │   │  SEGMENT     │   │  NORMALIZE  (TRAINED ByT5)  │
 │  document  ├▶ ebooklib/PyMuPDF ├──▶│ TOC/font/    ├──▶│ pysbd +      ├──▶│ numbers·dates·money·abbr·   │
 └────────────┘│ pdfplumber/md    │   │ regex cascade│   │ TTS-chunk    │   │ roman·units·URLs → words    │
               └──────────────────┘   └──────────────┘   └──────────────┘   └─────────────┬──────────────┘
                                                                                          ▼
 ┌──────────────┐   ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────────────────────────┐
 │  m4b/mp3/srt │   │   STITCH         │   │   AUDIO QA       │   │  TTS  (SpeechT5 default / Kokoro /    │
 │  (+ chapters)│◀──┤ pydub concat +   │◀──┤ loudness·peak·   │◀──┤  XTTS char-voices) per segment       │◀── VOICE-ROUTE
 │  + manifest  │   │ silence gaps +   │   │ duration·empty·  │   │  soundfile WAV + per-seg duration +  │   narration/heading
 └──────────────┘   │ −18 LUFS master  │   │ trim → re-synth  │   │  word timestamps                     │   /dialogue→voice map
        ▲           └──────────────────┘   └───────┬──────────┘   └──────────────────────────────────────┘
        │                                          │ FAIL & retries<N (jitter seed / split / fallback model)
        └──────────────────────────────────────────┘──────────────────────────────────────────────────────▶ back to TTS
```

**Build invariants:** reading order from EPUB spine / PDF page+outline; **never split mid-sentence**; merge tiny fragments (< 8 chars); inter-sentence ~150 ms, inter-paragraph ~350 ms, inter-chapter ~700 ms silence; master to **−18 LUFS / −3 dBTP** (ACX-safe; −16 LUFS podcast preset); m4b chapter `START/END` (ms) must match concatenated chapter offsets; pin every model by `repo@revision`.

---

## 4. Trainable model plan — ByT5-small Text Normalization

### Dataset (primary = synthetic; no Hub dependency)
There is **no usable, permissively-licensed English TN Challenge mirror on the Hub** (canonical ids 404; the only English-adjacent set is GPL ITN with a broken viewer). Therefore the **synthetic generator is the primary training-data path**.

Generate balanced `(written, spoken)` pairs across all **16 semiotic classes** (`PLAIN, PUNCT, DATE, LETTERS/VERBATIM, CARDINAL, ORDINAL, DECIMAL, MEASURE, MONEY, FRACTION, TIME, ELECTRONIC, DIGIT, TELEPHONE, ADDRESS, TRANS`) using `num2words` + `random` + carrier-sentence templates:

```python
n = random.randint(0, 10**9);       pairs.append((f"{n:,}", num2words(n)))                       # CARDINAL
y = random.randint(1000, 2099);     pairs.append((str(y), num2words(y, to='year')))             # YEAR/DATE
n = random.randint(1, 200);         pairs.append((f"{n}th", num2words(n, to='ordinal')))        # ORDINAL
amt = round(random.uniform(.01,9.99),2); pairs.append((f"${amt}M", f"{num2words(amt)} million dollars"))  # MONEY
# DECIMAL→"point" digits · FRACTION→"a over b" · MEASURE→num+unit-dict · TIME→"h:mm"
# TELEPHONE/DIGIT→per-digit · ROMAN→converter · ABBREV/ELECTRONIC→dicts
```
Embed each token in carrier sentences (`"The total was {X} last year."`) so the model learns context; shuffle; **80/10/10** split. Target **~50k–200k** examples; **over-sample rare classes** (TIME, FRACTION, ADDRESS, TELEPHONE). Auto-label with the §6 rule engine, hand-verify a small eval slice.

- **Optional scale-up:** Google TN Challenge from `rwsproat/text-normalization-data` / Kaggle (off-Hub, large, restrictive) → convert to `input`/`target`/`class` columns. TSV schema: `semiotic_class \t input_token \t output_token`, output `<self>` for PLAIN/PUNCT, blank lines separate sentences.
- **Eval sanity set:** `DigitalUmuganda/Text_Normalization_Challenge_Unittests_Eng_Fra` (VERIFIED English unit tests).

### Input / target format
- **Prefix** every source with `"normalize: "` (T5-style task tag).
- **Unicode NFC** before tokenizing (stable bytes); ByT5 tokenizer is byte-level — **do not** add custom digit tokens.
- **Lengths:** sentence-level ByT5 `max_source=256 / max_target=256`; token-level (single semiotic in→out) `128/128`; `t5-small` fallback `192/192`. `DataCollatorForSeq2Seq` dynamic padding; label pad → `-100`.

### Metrics
- **Sentence-level exact-match accuracy** (headline, micro — dominated by PLAIN/PUNCT).
- **Per-class accuracy + macro-average** (exposes rare-class failures).
- **Unrecoverable/silly-error count** (manual, Sproat & Jaitly arXiv:1611.00068) — the critical TTS metric; complement with WER/edit distance and a rule-vs-neural delta table.

### Baseline to beat
Deterministic **regex + abbreviation-dict + `num2words`/`inflect`** pipeline (§6 rule engine): ordered longest-match spans for MONEY/DATE/CARDINAL/DIGIT/ORDINAL/DECIMAL/FRACTION/MEASURE/ROMAN/ABBREV/ELECTRONIC, PLAIN/PUNCT pass-through. Cite `nemo_text_processing` WFST as the production-grade reference baseline. ByT5 must **beat per-class accuracy** while minimizing unrecoverable errors.

### Anti-overfitting
- Keep T5 `dropout_rate=0.1` (raise to 0.15 only if val/train gap widens); **label smoothing 0.1**, **weight_decay 0.01**, **grad clip 1.0**.
- **Early stopping** patience 4 on `sentence_accuracy`, `load_best_model_at_end=True`.
- **Dedup** `(input, target)` pairs (hash-dedup *before* splitting to prevent leakage); **cap PLAIN/PUNCT ≤ 30%** of train so rare classes are learned.
- **Stratified per-class held-out eval**; optionally hold out *entire rare-class instances* (e.g. a chunk of TELEPHONE) to probe compositional generalization. Never let identical sentences span train/val.

### H100 `Seq2SeqTrainer` config dict
```python
CONFIG = {
    "model_id": "google/byt5-small",          # fallback: "google-t5/t5-small"
    "prefix": "normalize: ",
    "max_source_length": 256, "max_target_length": 256,   # token-level → 128/128 ; t5-small → 192/192

    "effective_batch_size": 256,              # HELD CONSTANT across all GPUs
    "per_device_train_batch_size": 32,        # H100 default
    "gradient_accumulation_steps": 8,         # 256 / (32 * num_gpus=1)
    "per_device_eval_batch_size": 64,

    "bf16": True, "tf32": True,               # bf16+tf32 on H100/A100
    "gradient_checkpointing": True,           # use_reentrant=False; model.config.use_cache=False
    "group_by_length": True,                  # ~1.3–1.6x throughput on char seqs

    "learning_rate": 5e-4,                    # ByT5 likes higher LR (3e-4 for t5-small)
    "optim": "adamw_torch_fused",
    "lr_scheduler_type": "cosine", "warmup_ratio": 0.05,
    "weight_decay": 0.01, "label_smoothing_factor": 0.1, "max_grad_norm": 1.0,
    "num_train_epochs": 3,                    # or cap by max_steps≈150k–200k on the full set

    "predict_with_generate": True,
    "generation_max_length": 256, "generation_num_beams": 1,  # greedy eval; beams=4 final test only

    "eval_strategy": "steps", "save_strategy": "steps",
    "eval_steps": 2000, "save_steps": 2000, "save_total_limit": 3, "logging_steps": 100,
    "load_best_model_at_end": True,
    "metric_for_best_model": "sentence_accuracy", "greater_is_better": True,

    "dropout_rate": 0.1, "early_stopping_patience": 4, "seed": 42,
}
```
Wire with `Seq2SeqTrainingArguments` + `Seq2SeqTrainer`, `DataCollatorForSeq2Seq`, `EarlyStoppingCallback(patience=4)`, and `compute_metrics` returning `sentence_accuracy`, `acc_<CLASS>` (all 16), and `macro_class_accuracy`. Resume via `get_last_checkpoint`. Set `torch.backends.cuda.matmul.allow_tf32=True`. Pass `gradient_checkpointing_kwargs={"use_reentrant": False}`.

### GPU-profile table (effective batch held at 256)
| GPU | Precision | tf32 | per-dev bs | grad-accum | Eff. batch | grad-ckpt | Notes |
|---|---|---|---|---|---|---|---|
| **H100 80GB** | bf16 | ✅ | 32 | 8 | 256 | optional (off→faster) | bs=48/accum=6 if seq≤192; ckpt off ≈ +25% speed; VRAM ~45–60 GB (off) / ~28–35 GB (on) |
| **A100 80GB** | bf16 | ✅ | 32 | 8 | 256 | off | ~1.5–2× slower than H100 |
| **A100 40GB** | bf16 | ✅ | 16 | 16 | 256 | on | ckpt on for 256-byte seqs |
| **L4 24GB** | bf16 | ❌ | 8 | 32 | 256 | on | bf16 (Ada); no tf32 benefit |
| **T4 16GB** | fp16 | ❌ | 4 | 64 | 256 | on | **fp16 not bf16** (Turing); keep label-smoothing + grad clip |

Auto-select via `torch.cuda.get_device_name(0)` substring match; effective batch stays 256. **Wall-clock:** ~2–3M-example subset × 3 epochs ≈ **3–6 h on one H100** (or cap `max_steps≈150k–200k`); t5-small fallback ~3–4× faster, lower numeric/date accuracy, fits T4 (<8 GB).

---

## 5. Agent architecture

The agent is a **deterministic finite-state machine** over a shared, append-only, versioned `JobContext`: `state(ctx) -> (ctx', next_state)`. The **LLM is never in the control path** — it returns *advisory hints* at escalation hooks only; on timeout, low confidence, invalid JSON, missing key, or exception, the FSM falls back to the rule result. This guarantees reproducibility + a full audit trace (`audit.jsonl` + rolled-up `manifest.json`).

### State machine
```
INPUT ─▶ S0 LOAD ─▶ S1 PARSE_STRUCTURE ─▶ S2 DETECT_CHAPTERS ─▶ S3 SEGMENT+CLASSIFY ─▶ S4 NORMALIZE ─▶ S5 VOICE_SELECT
              │            │[D1 parse routing]                       │                  [D2 norm conf]     [D4 voice route]
              │                                                                                                  │
 EXPORT ◀─ S9 EXPORT ◀─ S8 STITCH ◀─ S7 AUDIO_QA ◀─────────────────────────────────────────── S6 SYNTHESIZE ◀──┘
 (audio+srt+manifest)                    │[D3 audio-QA gate] ──(FAIL, retries≤N: jitter seed / split / fallback model)──▶ S6
```
S0 sniffs mime + extracts raw text/metadata; S2 uses TOC→font/size→regex→single-chapter cascade; S3 produces typed segments `{narration | dialogue | heading | skippable}` via rule classifier + quote-balance + NER. On unrecoverable QA failure, S7 **degrades gracefully** (keep best clip / insert silence, flag, continue — never crash the book).

### Decision points (≥3; four specified)
- **D1 — Parse-quality / format routing** (on S1): `parse_score ∈ [0,1]` from dict-hit-rate + structure-signal density + non-garbage ratio. `≥0.85` → structured (trust TOC/headings); `0.5–0.85` → LLM-assisted structure repair (fallback heuristic); `<0.5` → flat-text degraded (`degraded=true`); DRM/empty/scanned → **FAIL_FAST 422** (OCR only if `ocr_enabled`).
- **D2 — Normalization-confidence escalation** (on S4 per-token): `conf = max_candidate_prob`; ambiguous if `conf < τ=0.75` or two candidates within δ (e.g. `St.`→Saint/Street, `read`/`lead` homographs). Confident → emit; ambiguous → **batched LLM disambiguation** with sentence + NER context; LLM down/invalid/`conf<0.5` → rule default + `flag=ambiguous_unresolved`.
- **D3 — Audio-QA re-synthesis gate** (on S6 per clip): checks = empty/NaN/silence (RMS floor), duration sanity (`[0.5×,2.0×]` of `chars×sec_per_char`), clipping/true-peak (`>−1 dBTP`), loudness (LUFS window), edge artifacts. Retry budget `N=3`, escalating: (a) new seed/jitter, (b) split + boundary pad, (c) **fallback model** (Kokoro→SpeechT5); at `N` → keep best, flag `qa_failed`, continue.
- **D4 — Voice routing** (on S5 per-segment): `narration`/`heading` → narrator voice (default SpeechT5/Kokoro); `dialogue`+known speaker → that character's pinned profile; `dialogue`+unknown → "unattributed-dialogue" voice; `skippable` → dropped; `>K` distinct speakers → **cluster** minor characters into a voice pool by inferred gender/age, pinning the `character→voice` map so a voice is stable book-wide (determinism).

The FSM records `{decision, branch, rule_score, llm_used, llm_conf, final_reason, tool_version}` for every firing.

### Tool contracts (pydantic, versioned; `tool_version` → manifest)
```python
parse_document(file: bytes, mime: str|None) -> {text, blocks[{kind,text,level,page,font}], meta{title,author,lang}, parse_score, drm}
segment_and_classify(blocks) -> [{id, chapter_id, order, text, type, speaker, speaker_conf, rule_score}]
normalize_text(segment)      -> {text_norm, tokens[{raw,norm,kind,conf,candidates,source}]}        # source: rule|llm
select_voice(segment, voice_map, pool_budget) -> {voice_profile_id, model_id, ref_audio, style, reason}
synthesize(text_norm, voice_profile)          -> {wav, sr, seed, model_id, model_rev}
audio_qa(clip) -> {pass, checks{empty,duration_ratio,true_peak_dbtp,lufs,clipped_pct,lead_sil_ms,tail_sil_ms}, action, retries_used}
stitch(clips, chapters, silence_cfg) -> {book_wav, chapter_offsets[]}
export(book_wav, alignments, manifest) -> {audio, srt, m4b, manifest}
```

### Optional LLM fallback
```python
llm_advise(decision_id: str, payload: dict) -> {choice, confidence, rationale}
# Wrapped: try LLM(temperature=0, timeout=8s, max_retries=1, strict JSON) -> validate -> else rule_result.
# Cache by hash(payload) so repeated runs are bit-stable. Used only at D1/D2/D4 + D3 text-hook; rules always produce a valid answer.
```
Audit guarantees: every LLM use is flagged with its confidence + what the rule would have done; every QA retry traces to the failing check; re-running with the same `config_hash` + pinned seeds reproduces the audio bit-for-bit.

---

## 6. Deployment

### FastAPI service
```
POST /v1/jobs            (multipart file | json {text, config})        -> 202 {job_id}
GET  /v1/jobs/{id}                                                     -> {status, progress, manifest?}
GET  /v1/jobs/{id}/events   (SSE)   -> state/decision/progress stream
GET  /v1/jobs/{id}/stream   (audio/mpeg, chunked) -> chapter-by-chapter as finished
GET  /v1/jobs/{id}/audio | /srt | /manifest                           -> artifacts
POST /v1/synthesize         (sync, text ≤ ~2k chars)                   -> {audio, srt} inline
GET  /healthz /readyz /metrics(Prometheus)
```
Thin API + **Redis + RQ/Celery/arq** queue; GPU workers pull jobs; `Idempotency-Key` dedups resubmits; artifacts to **S3/MinIO** with presigned URLs + TTL. SSE progress mirrors `audit.jsonl`; `/stream` flushes chapter 1 first so playback starts in seconds.

### Gradio UI
Upload (epub/pdf/txt/md) or paste → config (target LUFS, narrator voice, character-voice toggle, model choice, LLM-brain on/off). Live decision log, `gr.Audio` streaming chapters, download buttons (`m4b`/`mp3`/`srt`/`manifest`). **Voice-cast table** overrides D4's `character→voice` map and re-runs only S6–S9 (S1–S5 outputs cached → cheap re-synth).

### CLI batch
```bash
audiobookgen synth book.epub --out ./out --voice af_heart --char-voices \
  --model speecht5 --target-lufs -18 --llm-brain off --workers 4 --format m4b
audiobookgen batch ./library/*.epub --out ./out --parallel 2 --resume
```
YAML profiles; `--resume` skips chapters whose clip hashes already pass QA; exit codes encode QA-flag severity for CI.

### Docker
Multi-stage `base` (CUDA runtime + ffmpeg + espeak-ng) → `deps` (pinned wheels) → `runtime`. Two images: `api` (CPU) and `worker` (GPU). `HF_HOME` on a persistent volume; models baked or volume-mounted by `repo@revision`. `docker-compose`: `api`, `worker`×N (`--gpus all`), `redis`, `minio`; healthchecks gate `readyz` on model-load completion.

### HF Space
**Gradio Space on GPU (T4/A10)**, `app.py` wrapping the same pipeline, LLM brain off by default (`ANTHROPIC_API_KEY` Space secret enables D1/D2/D4 hooks). Default to permissive CPU-able **SpeechT5** (or Kokoro) so the demo runs cheaply; XTTS/Dia behind a toggle; cap input length + concurrency; pre-bake model cache for fast cold starts. Companion **FastAPI Space** (Docker SDK) exposes the REST API.

### Latency / scalability / versioning
- **Latency/RTF:** report per-job RTF in manifest; TTFB on `/stream` < a few seconds by synthesizing chapter 1 first.
- **Scalability:** segments independent post-S5 → **chunk-level fan-out** across workers, order restored at S8 by `order` index; **GPU micro-batching** (~20–50 ms window) groups same-voice/same-model segments; one model pinned per worker (no reload thrash); autoscale by queue depth; **content-addressed clip cache** `hash(text_norm + voice_profile + model_rev + seed)` reuses unchanged clips on re-runs / voice-cast edits.
- **Versioning:** pin every model by `repo@revision` (commit sha) in `models.lock.json` (role→`id@rev`); voice-profile registry versioned independently (cloned-voice ref audio content-hashed); `git_sha` + `config_hash` + `tool_version` complete provenance; canary new revs behind a config flag, A/B by routing a job fraction.

---

## 7. Risks, limitations, ethics

- **Voice-cloning consent.** `coqui/XTTS-v2` (and F5-TTS) clone a voice from ~6 s of audio. **Require explicit, documented consent** for any cloned voice; never clone real people without authorization. Both are **non-commercial (CPML / CC-BY-NC-4.0)** — keep out of any shipped/commercial path; restrict to research/demo with a consent gate. The default narrator path (SpeechT5/Kokoro) uses synthetic/precomputed voices with no cloning.
- **Hallucinated pronunciations / unrecoverable TN errors.** Neural TN (and TTS) can occasionally produce semantically catastrophic mis-reads (wrong number, flipped unit, `£`→"euros", mangled year). Mitigate with: the **rule baseline as a guardrail**, the **D3 duration/sanity QA gate**, manual **unrecoverable-error counting** in eval, and minimizing such errors even at the cost of benign ones. Surface flagged clips in the manifest for human review.
- **PII in books.** Documents may contain phone numbers, addresses, names, emails (TELEPHONE/ADDRESS/ELECTRONIC classes). Do not log raw PII beyond what's needed; honor TTL cleanup on artifacts; the NER step (`dslim/bert-base-NER`) is for attribution only, not retention. Offer a redaction/skip option.
- **Copyright.** Converting a book to audio is a derivative work. **Verify the user holds rights / the work is public-domain or licensed** before production; record input `sha256` + provenance in the manifest. Do not ship a public service that ingests arbitrary copyrighted books without rights checks.
- **License hygiene (load-bearing).** Commercial path = **permissive only**: SpeechT5/HiFi-GAN (MIT), Kokoro/Parler/Dia/Orpheus/CSM (Apache-2.0), ByT5/T5 (Apache-2.0), x-vectors (MIT), NER/diarization (MIT). **Never** ship `coqui/XTTS-v2` (CPML), `facebook/mms-tts-eng` (CC-BY-NC), `SWivid/F5-TTS` (CC-BY-NC), the MMS forced-aligner (CC-BY-NC → swap to torchaudio MMS aligner), or GPL TN data in a commercial deployment.
- **Limitations.** Single-document focus; long-book wall-clock dominated by TTS RTF; chapter detection degrades on poorly structured/scanned PDFs (degraded flat-text path); TN synthetic corpus may under-cover edge cases the off-Hub Google set would catch (optional scale-up exists).

---

## 8. Repo module map

```
src/audiobook_ai/
├── __init__.py
├── config.py                # CONFIG dicts, GPU profiles, LUFS targets, models.lock.json loader, repo@revision pins
│
├── parsing/
│   ├── loader.py            # S0: mime sniff, raw text/metadata, DRM/empty/scanned detection
│   ├── epub.py              # ebooklib spine → reading order; bs4 headings/paragraphs; NCX/nav TOC
│   ├── pdf.py               # PyMuPDF spans/bbox/outline; header/footer/page-num drop; pdfplumber fallback
│   ├── text_md.py           # charset_normalizer decode; Markdown→bs4 heading/para extractor
│   └── chapters.py          # S2: TOC→font/size→regex→single-chapter cascade
│
├── normalization/
│   ├── rules.py             # S6 BASELINE: regex + abbrev dict + num2words/inflect (MONEY/DATE/CARDINAL/…)
│   ├── synthetic.py         # §4 synthetic (written,spoken) corpus generator across 16 classes
│   ├── byt5_infer.py        # TRAINED ByT5-small inference (prefix, NFC, batched)
│   ├── train.py             # Seq2SeqTrainer recipe, GPU-profile auto-select, resume
│   ├── metrics.py           # sentence_accuracy, acc_<CLASS>, macro_class_accuracy, unrecoverable-error tally
│   └── nemo_ref.py          # optional nemo_text_processing WFST reference baseline
│
├── segmentation/
│   ├── segment.py           # S3: pysbd sentences + TTS-window chunking (never split mid-sentence, merge tiny)
│   └── classify.py          # narration|dialogue|heading|skippable; quote-balance + dslim/bert-base-NER speaker attribution
│
├── tts/
│   ├── base.py              # TTSBackend interface (synthesize → wav, sr, seed, model_id, model_rev)
│   ├── speecht5.py          # PRIMARY: microsoft/speecht5_tts + speecht5_hifigan + cmu-arctic-xvectors
│   ├── kokoro.py            # optional Apache-2.0 quality upgrade (KPipeline, 24 kHz)
│   ├── parler.py            # optional permissive prompt-styled voices
│   ├── xtts.py              # optional NON-COMMERCIAL voice clone (consent-gated)
│   ├── dia.py               # optional two-voice dialogue [S1]/[S2]
│   ├── pyttsx3_fallback.py  # last-resort offline (no weights, never hard-fails)
│   └── voice_router.py      # D4: character→voice map, pinning, pool clustering
│
├── audio/
│   ├── qa.py                # S7/D3: empty/NaN, duration sanity, true-peak, LUFS, edge-trim; re-synth gate
│   ├── stitch.py            # S8: pydub concat + sentence/para/chapter silence + chapter offsets
│   ├── loudness.py          # pyloudnorm + ffmpeg loudnorm two-pass → −18 LUFS / −3 dBTP
│   ├── encode.py            # S9: m4b (FFMETADATA chapters) / mp3 (mutagen CHAP/CTOC) / opus / flac
│   └── subtitles.py         # SRT/VTT from segment durations + (optional) forced-aligner word timestamps
│
├── agent/
│   ├── fsm.py               # S0–S9 state machine over JobContext; transition logging
│   ├── context.py           # append-only versioned JobContext
│   ├── decisions.py         # D1–D4 predicates + branch logic
│   ├── llm_brain.py         # optional advisory llm_advise (temperature=0, timeout, JSON-validate, cache, rule fallback)
│   ├── tools.py             # pydantic tool contracts (parse/segment/normalize/select_voice/synthesize/audio_qa/stitch/export)
│   └── audit.py             # audit.jsonl writer + manifest.json roll-up
│
├── api/
│   ├── app.py               # FastAPI: /v1/jobs, /events(SSE), /stream, /synthesize, /healthz /readyz /metrics
│   ├── queue.py             # Redis + RQ/Celery/arq enqueue; idempotency; worker entrypoint
│   └── storage.py           # S3/MinIO artifacts + presigned URLs + TTL
│
├── ui/
│   └── gradio_app.py        # upload/config, live decision log, streaming player, voice-cast override (re-run S6–S9)
│
└── cli.py                   # `audiobookgen synth|batch` (YAML profiles, --resume, CI exit codes)

deploy/   # Dockerfile (api/worker multi-stage), docker-compose.yml, hf_space/app.py
notebooks/ # ByT5 training (Colab/H100), rule-vs-neural eval, synthetic-corpus build
```

**Module → pipeline-stage map:** `parsing/*` = PARSE+CHAPTER (S0–S2) · `segmentation/*` = SEGMENT (S3) · `normalization/*` = NORMALIZE+training (S4, baseline + TRAINED ByT5) · `tts/*` = VOICE-ROUTE+TTS (S5–S6) · `audio/*` = AUDIO-QA+STITCH+EXPORT (S7–S9) · `agent/*` = FSM/decisions/LLM/audit binding it all · `api/ui/cli/deploy` = serving surfaces.
