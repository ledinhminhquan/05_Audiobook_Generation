# Model Selection & Optimization — P05 Audiobook Generation

**Author:** Le Dinh Minh Quan (23127460) — NLP in Industry, final assignment
**Trainable component:** Text Normalization (TN) seq2seq model for TTS
**Reference repo:** github.com/denizsafak/abogen

---

## 1. What we are actually selecting a model for

An audiobook pipeline has many moving parts, but only one part is genuinely a *learned NLP problem*: **text normalization (TN)** — converting written tokens into their spoken form. A naive TTS reads `$5.2M`, `1984`, `Dr.`, `Chapter IV`, `3/4`, and `9:45 AM` wrong. The audio itself is produced by a **pretrained** neural TTS backend (SpeechT5); the document→audiobook orchestration is a deterministic agent FSM. So the model we *train and select* is the TN model, and the model we *select for it* must be robust to numbers, symbols, dates, currencies, and out-of-vocabulary (OOV) tokens.

The task spans **16 semiotic classes**: `PLAIN, PUNCT, CARDINAL, ORDINAL, DECIMAL, DIGIT, MONEY, MEASURE, DATE, TIME, FRACTION, TELEPHONE, ELECTRONIC, ABBREV, ROMAN, VERBATIM`. It is framed as a sequence-to-sequence problem with the task prefix `"normalize: "`.

---

## 2. Model choice: `google/byt5-small`

### 2.1 Candidates considered

| Option | What it is | Why not selected as primary |
|---|---|---|
| **Rule-only** (`baseline_rules.py`) | Ordered regex spans + abbreviation dict + `num2words` / pure-python expanders | Context-blind. Strong on easy cases, collapses on ambiguity (see §5). Kept as a **baseline and as a runtime guardrail**, not as the model. |
| **`google-t5/t5-small`** (VERIFIED, apache-2.0, 60.5M) | Subword (SentencePiece) T5 encoder-decoder | Subword tokenization fragments numbers/symbols/OOV unpredictably; less robust to adversarial/noisy character-level input. Retained as a **GPU-constrained fallback** (T4). |
| `flan-t5` | Instruction-tuned subword T5 | Same subword-vocabulary limitation as t5-small; instruction tuning buys little for a narrow char-faithful transduction task and adds size. Not selected. |
| **`google/byt5-small`** (VERIFIED, apache-2.0, ~300M) | **Byte/character-level** T5 encoder-decoder | **Selected.** Byte vocabulary => no OOV, robust to digits/symbols/spacing, ideal for character-faithful normalization. |

### 2.2 Why byte-level wins for TN

TN is fundamentally about **characters**: a model must reliably distinguish `1984` (year → "nineteen eighty-four") from `1984 people` (count → "one thousand nine hundred eighty-four"), expand `$5.2M`, and read Roman numerals in `Chapter IV`. Subword tokenizers (t5/flan-t5) split such tokens into brittle, vocabulary-dependent pieces and choke on genuinely novel strings. ByT5 operates over raw **bytes**, so:

- **No OOV** — every input is representable; novel numbers/symbols are never "unknown".
- **Char-level robustness** — resilient to adversarial/noisy tokens, odd spacing, and the garbled output of scanned PDFs (which the pipeline routes to a degraded flat-text path).
- **Faithful transduction** — the model edits at the character granularity the task demands.

This robustness is the explicit reason ByT5 is primary: the licenses are clean (apache-2.0 for both byt5-small and the t5-small fallback), keeping the commercial path on MIT/Apache ids only.

---

## 3. Architecture

ByT5-small is a **T5 encoder-decoder Transformer** with a key modification: it discards the SentencePiece subword vocabulary in favor of a **byte vocabulary** (UTF-8 bytes plus a few special tokens). Mechanically:

- **Encoder–decoder, attention-based** seq2seq. The encoder reads the prefixed input `"normalize: <written text>"`; the decoder autoregressively emits the spoken form.
- **Byte tokenization** — input and output are byte sequences. There is no learned subword merge table, hence no OOV and no tokenizer drift across domains.
- **Parameter budget** — ~300M params for byt5-small (vs 60.5M for the t5-small fallback). ByT5 reallocates capacity from a large embedding table toward deeper Transformer layers, which suits character-level reasoning.
- **Trade-off baked into the architecture** — byte sequences are **longer** than subword sequences for the same text, so each example costs more compute/memory (addressed in §4 and §6).

---

## 4. Training procedure

Training uses the Hugging Face **`Seq2SeqTrainer`** with `predict_with_generate=True`, `load_best_model_at_end`, and resume via `get_last_checkpoint`. Precision is **bf16 + tf32** on H100/A100 class GPUs (the T4 path drops to fp16 — see GPU profile).

### 4.1 Hyperparameter configuration

```python
training_config = {
    # precision
    "bf16": True,                  # bf16 on H100/A100 (fp16 only on T4)
    "tf32": True,                  # H100/A100

    # optimization
    "learning_rate": 5e-4,         # byt5-small  (3e-4 for t5-small)
    "effective_batch_size": 256,   # per-device 32 x grad-accum 8 on H100
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.05,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,

    # regularization
    "label_smoothing_factor": 0.1,
    "dropout": 0.1,

    # generation / eval
    "predict_with_generate": True,
    "metric_for_best_model": "sentence_accuracy",

    # checkpointing / stopping
    "early_stopping_patience": 4,  # on sentence_accuracy
    "load_best_model_at_end": True,
    "resume_from_checkpoint": "get_last_checkpoint",
}
```

Learning rate is the one value that differs by base model: **`5e-4` for byt5-small**, **`3e-4` for the t5-small fallback**. The effective batch size is held at **256** across all hardware by trading per-device batch against gradient accumulation (and gradient checkpointing on smaller GPUs).

### 4.2 GPU profile (effective batch held at 256)

| GPU | Precision | Per-device batch | Grad-accum | Grad-checkpoint | Base model |
|---|---|---:|---:|---|---|
| **H100** | bf16 + tf32 | 32 | 8 | off | byt5-small |
| **A100-40** | bf16 | 16 | 16 | on | byt5-small |
| **L4** | bf16 | 8 | 32 | on | byt5-small |
| **T4** | fp16 (!) | 4 | 64 | on | **switch to t5-small** |

**Rough training time:** byt5-small ≈ **3–6 h on a single H100** for the full corpus (or cap `max_steps`); t5-small is ≈ **3–4× faster** and fits a T4.

### 4.3 Anti-overfitting strategy

Multiple defenses are stacked deliberately:

1. **Deduplicate** (written, spoken) pairs **before** splitting — leakage-free splits, no val/test input appears in train.
2. **Cap `PLAIN` ≤ ~30%** of the corpus so the model is forced to learn the hard semiotic classes rather than copying.
3. **Early stopping** (patience 4) on `sentence_accuracy`.
4. **Label smoothing** (0.1) + **weight decay** (0.01) + **dropout** (0.1).
5. A **held-out hard slice** (1,500 examples) that stresses ambiguity, evaluated separately from the easy distribution.

---

## 5. Baseline comparison

### 5.1 The baseline

The model must beat a **context-blind rule normalizer** (`src/audiobook_ai/models/baseline_rules.py`): ordered regex spans + an abbreviation dictionary + `num2words` / pure-python expanders. It is genuinely strong on unambiguous cases but, by construction, cannot use context.

### 5.2 Validated baseline numbers (synthetic test distribution)

| Slice | Baseline sentence exact-match (EM) |
|---|---:|
| **Easy** | **0.945** |
| **Hard** | **0.006** |
| **Overall** | **0.712** |

The hard slice is where context-blindness is fatal: the synthetic corpus **injects ambiguity** the baseline cannot resolve — `St.` → *Street* vs *Saint*, `"1984 people"` (count) vs `"in 1984"` (year), dates with the day as an ordinal, and Roman-numeral contexts the baseline misses. The **0.006** EM on the hard slice is the honest, reproducible evidence that rules alone are insufficient.

### 5.3 Expected neural result

The trained context-aware ByT5 model is expected to reach **high accuracy on BOTH slices**, beating the baseline overall and **especially on the hard slice** — exactly the cases where a learned model's contextual reasoning pays off. This is the central, reproducible claim of the project: *neural ≫ baseline on the hard slice*.

### 5.4 Data note

There is **no permissively-licensed English TN-Challenge mirror** on the HF Hub (the canonical ids `google/text_normalization` and `cestwc/text-normalization` 404). The primary data is therefore a **reproducible synthetic corpus generator** (`src/audiobook_ai/data/tn_corpus.py`) producing context-rich pairs across all 16 classes — defaults: **train 60,000 / val 4,000 / test 4,000 / hard 1,500**. An optional real-data loader plus the optional eval sanity set `DigitalUmuganda/Text_Normalization_Challenge_Unittests_Eng_Fra` (VERIFIED) are available.

---

## 6. Evaluation & error-analysis approach

### 6.1 Metrics

- **Sentence-level exact-match accuracy** — the headline number.
- **Macro per-semiotic-class accuracy** — averages across the 16 classes to expose **rare-class failures** that headline accuracy would hide.
- **Per-class accuracy** — a class-by-class breakdown.
- **Unrecoverable / "silly" errors** (Sproat & Jaitly notion) — semantically catastrophic mis-reads (e.g. reading a year as a count, or mangling a currency). These are minimized **even at the cost of benign errors**, because a single catastrophic mis-read damages a listener's trust far more than a slightly awkward but correct reading.

### 6.2 How errors are analyzed

The `error-analysis` CLI examines failures **per class** and isolates **unrecoverable errors** specifically, alongside **neural-vs-baseline disagreement** (also recorded at runtime by agent decision point D2). This separates "the model paraphrased acceptably" from "the model said something semantically wrong," and tells us which semiotic classes need more synthetic coverage. The target is an **unrecoverable-error rate → 0**.

---

## 7. Trade-offs

### 7.1 Accuracy vs speed

- **Greedy vs beam decoding.** Greedy decoding is fast and is the default for the latency-sensitive pipeline; beam search can lift accuracy on ambiguous inputs at higher per-segment cost. The TN model runs **per-segment with a confidence score** (length-normalized sequence probability), so the pipeline can afford to spend more decoding budget only where confidence is low.
- **ByT5 char sequences are longer.** Byte-level inputs/outputs are longer than subword ones, so ByT5 is **slower and more memory-hungry** per example than t5-small. This is the price of OOV-free robustness, and it is exactly why the GPU profile (§4.2) trades per-device batch against gradient accumulation and gradient checkpointing, and why the T4 path falls back to t5-small.

### 7.2 Complexity vs maintainability

The system is **neural + a rule guardrail**, not neural-only. The trained ByT5 model handles context; the deterministic rule normalizer (and per-clip audio QA at agent decision point D3, plus unrecoverable-error counting) acts as a **guardrail against hallucinated pronunciations** and as a graceful-degradation floor when the model is unavailable. This adds a second code path to maintain, but it buys:

- **Safety** — a learned model never silently emits a catastrophic reading without a deterministic check behind it.
- **Robustness** — the pipeline never hard-fails: if the neural model or a TTS backend is missing, rules and a placeholder backend keep it running.

The trade-off is deliberate: a little extra maintenance surface in exchange for a system that is **safe, testable with no model present, and degrades gracefully** end-to-end.

---

## 8. Summary

| Decision | Choice | Core reason |
|---|---|---|
| Primary base model | `google/byt5-small` (~300M, apache-2.0) | Byte-level robustness for numbers/symbols/OOV |
| Fallback base model | `google-t5/t5-small` (60.5M, apache-2.0) | Fits T4; ~3–4× faster |
| Rejected | rule-only, t5-small-as-primary, flan-t5 | Context-blind or subword-fragile |
| Trainer | HF `Seq2SeqTrainer`, `predict_with_generate` | Generation-aware eval, resumable |
| Effective batch | 256 (held across all GPUs) | Stable optimization regardless of hardware |
| Baseline to beat | 0.945 easy / 0.006 hard / 0.712 overall | Honest, reproducible target; neural wins the hard slice |
| Headline metric | sentence exact-match + macro-class + unrecoverable-error rate | Catches rare-class and catastrophic failures |
