# Text-Normalization Quality Evaluation

This document is the quality deep-dive for the **P05 Audiobook Generation** system. It specifies exactly how the trainable model — a Text-Normalization (TN) seq2seq model that converts *written* text into *spoken* form — is measured, how it is compared against a context-blind rule baseline, and how those numbers translate into listener experience.

The TN model is the ML heart of the project: a naive TTS reads `$5.2M`, `1984`, `Dr.`, `Chapter IV`, `3/4`, and `9:45 AM` wrong. Getting these *right in context* is the hard, NLP-heavy part of audiobook production, so it is the part we evaluate rigorously.

- Model: `google/byt5-small` (byte/char-level, robust to numbers/symbols/OOV), task prefix `"normalize: "`. Fallback on small GPUs: `google-t5/t5-small`.
- Baseline: `src/audiobook_ai/models/baseline_rules.py` — a context-**blind** ordered-regex + abbreviation-dict + `num2words` expander.
- Evaluation entry point: `src/audiobook_ai/training/evaluate.py`.
- Error breakdown: `analysis/error_analysis.py`.

---

## 1. What we are measuring

The TN task spans **16 semiotic classes**: `PLAIN, PUNCT, CARDINAL, ORDINAL, DECIMAL, DIGIT, MONEY, MEASURE, DATE, TIME, FRACTION, TELEPHONE, ELECTRONIC, ABBREV, ROMAN, VERBATIM`.

A correct system must not only expand tokens, but **disambiguate by context** — the same surface form maps to different spoken forms depending on surrounding words:

| Written | Context A → spoken | Context B → spoken |
|---|---|---|
| `St.` | "Main **St.**" → "Main **Street**" | "**St.** Patrick" → "**Saint** Patrick" |
| `1984` | "**1984** people" → "nineteen eighty four"? **count** "one thousand nine hundred eighty four" | "in **1984**" → "nineteen eighty four" (**year**) |
| `IV` | "Chapter **IV**" → "four" (**Roman**) | "**IV** drip" → "I V" (**verbatim/abbrev**) |

A context-blind ruleset cannot resolve these. That gap is precisely what the evaluation is built to expose and quantify.

---

## 2. Metrics (precise definitions)

### 2.1 Sentence exact-match accuracy (EM) — the headline

For each test example, the model's generated spoken string is compared **verbatim** to the gold spoken string. The example scores `1` only on a full, exact match; `0` otherwise.

```
EM = (# sentences where prediction == gold) / (total sentences)
```

EM is strict by design: in an audiobook, a single wrong token inside an otherwise-correct sentence is still an audible defect, so we hold the model to whole-sentence correctness. EM is reported as the headline number and is the metric used for early stopping (`sentence_accuracy`) and `load_best_model_at_end`.

### 2.2 Macro per-class accuracy

EM alone hides rare-class failures: a model can score well overall while being catastrophic on, say, `TELEPHONE` or `ROMAN`, because those classes are infrequent. To surface this, we compute accuracy **within each semiotic class** and then average the per-class accuracies **unweighted**:

```
macro_acc = mean over the 16 classes of ( per_class_correct / per_class_total )
```

Because every class contributes equally regardless of frequency, macro accuracy is the metric that punishes "good on PLAIN, terrible on the rare hard stuff."

### 2.3 Per-class accuracy

The raw per-class numbers (one accuracy per semiotic class) feed the results table in Section 5 and drive `analysis/error_analysis.py`. They tell us *where* the model wins or loses against the baseline, class by class.

### 2.4 Unrecoverable-error count (Sproat & Jaitly "silly" errors)

Following the TN literature, not all errors are equal. We separately count **unrecoverable / silly errors**: semantically catastrophic mis-reads that a listener cannot recover from (a wrong number, a wrong year, a money figure read as a date). These are tracked as an explicit **count**, not just folded into accuracy, because the optimization target is:

> Minimize unrecoverable errors **even at the cost of more benign ones.**

A benign error (a stylistic variant that is still understandable) costs far less than a silly error (a factually wrong reading). Section 6 defines the taxonomy.

---

## 3. Evaluation protocol

### 3.1 Data splits

Because there is **no permissively-licensed English TN-Challenge mirror on the HF Hub** (the canonical ids `google/text_normalization` and `cestwc/text-normalization` are 404), the primary corpus is a reproducible **synthetic generator**, `src/audiobook_ai/data/tn_corpus.py`, producing context-rich `(written, spoken)` pairs across all 16 classes with **injected ambiguity**. Default sizes:

| Split | Size | Purpose |
|---|---|---|
| `train` | 60,000 | training only |
| `val` | 4,000 | model selection / early stopping |
| `test` | 4,000 | headline EM + macro/per-class accuracy |
| `hard` | 1,500 | ambiguous slice — the disambiguation stress test |

Splits are **leakage-free**: no `val`/`test` input appears in `train`, pairs are deduplicated before splitting, and `PLAIN` is capped at roughly 30% so the headline number is not inflated by trivial pass-through tokens.

### 3.2 The three evaluation surfaces

1. **Test split (4,000):** the main distribution. Reports headline EM, macro per-class accuracy, per-class accuracy, and the unrecoverable-error count.
2. **Ambiguous hard slice (1,500):** examples constructed specifically around the disambiguation traps — `St.`→Street vs Saint, `1984 people` (count) vs `in 1984` (year), dates with day-as-ordinal, Roman-numeral contexts the baseline misses. This slice is reported **separately**; it is the headline discriminator between context-aware and context-blind systems.
3. **Optional real sanity set:** `DigitalUmuganda/Text_Normalization_Challenge_Unittests_Eng_Fra` (verified) — an optional eval-only sanity check that the model generalizes beyond the synthetic generator. It is a smoke test, not the primary metric.

Run via the CLI:

```bash
audiobook-ai evaluate     # test + hard, EM + macro/per-class, unrecoverable count
audiobook-ai error-analysis   # per-class breakdown + silly/benign taxonomy
```

---

## 4. Neural-vs-baseline methodology and why the hard slice matters

### 4.1 Methodology

Both systems are evaluated on the **identical** `test` and `hard` inputs with the **identical** metrics, so any difference is attributable to the model, not the data:

1. Run `baseline_rules.py` over `test` and `hard` → baseline EM, macro, per-class.
2. Run the trained ByT5 model over the same inputs → neural EM, macro, per-class.
3. Compute per-class deltas (Section 5) and the change in unrecoverable-error count.
4. Record **neural-vs-baseline disagreement** per segment (also surfaced live in the agent's decision **D2** as a confidence/disagreement signal).

### 4.2 Why the hard slice is the real test

The baseline is *strong on easy cases* and *fails on ambiguous ones* by construction — it has no context. Overall EM therefore flatters the baseline whenever the test mix is dominated by easy tokens. Splitting **easy vs hard** is what makes the comparison honest:

- On **easy** inputs, both systems should be high — the model must not regress on the cases the baseline already handles.
- On **hard** inputs, the baseline collapses; this is where a context-aware seq2seq model earns its keep.

A single blended number would hide both facts. The hard slice isolates the disambiguation skill that justifies training a model at all.

---

## 5. Per-class results — table template

The per-class table below is the deliverable of `analysis/error_analysis.py`, **filled after training**. Each row is a semiotic class; `delta = neural_acc − baseline_acc`. Positive deltas concentrated in the ambiguous classes (`ABBREV`, `ROMAN`, `DATE`, `CARDINAL`/year-vs-count, `MONEY`, `TIME`) are the expected win pattern.

| Class | Baseline acc | Neural acc | Delta |
|---|---|---|---|
| PLAIN | _TBD_ | _TBD_ | _TBD_ |
| PUNCT | _TBD_ | _TBD_ | _TBD_ |
| CARDINAL | _TBD_ | _TBD_ | _TBD_ |
| ORDINAL | _TBD_ | _TBD_ | _TBD_ |
| DECIMAL | _TBD_ | _TBD_ | _TBD_ |
| DIGIT | _TBD_ | _TBD_ | _TBD_ |
| MONEY | _TBD_ | _TBD_ | _TBD_ |
| MEASURE | _TBD_ | _TBD_ | _TBD_ |
| DATE | _TBD_ | _TBD_ | _TBD_ |
| TIME | _TBD_ | _TBD_ | _TBD_ |
| FRACTION | _TBD_ | _TBD_ | _TBD_ |
| TELEPHONE | _TBD_ | _TBD_ | _TBD_ |
| ELECTRONIC | _TBD_ | _TBD_ | _TBD_ |
| ABBREV | _TBD_ | _TBD_ | _TBD_ |
| ROMAN | _TBD_ | _TBD_ | _TBD_ |
| VERBATIM | _TBD_ | _TBD_ | _TBD_ |
| **Macro avg** | _TBD_ | _TBD_ | _TBD_ |

### Validated baseline floor (the number to beat)

The baseline has already been measured on the synthetic `test` distribution. These are the **honest, reproducible** floor values:

| Slice | Baseline sentence EM |
|---|---|
| Easy | **0.945** |
| Hard | **0.006** |
| Overall | **0.712** |

The pattern is exactly as predicted: the context-blind ruleset is near-perfect on easy inputs (0.945) and **almost completely fails on the ambiguous hard slice (0.006)**. The trained context-aware model is expected to reach high accuracy on **both** slices — matching the baseline on easy and dramatically lifting the hard slice — thereby beating the baseline overall **and especially on hard**. The most important single improvement to demonstrate is hard-slice EM rising far above 0.006.

---

## 6. Error taxonomy: silly vs benign

`analysis/error_analysis.py` classifies every mismatch into one of two buckets. This taxonomy is what makes the unrecoverable-error count actionable.

### Silly / unrecoverable errors (minimize at all costs)
Semantically catastrophic — the listener hears something factually wrong and cannot reconstruct the original:
- `$5.2M` read as a date or a bare number.
- `1984` (a year) read as a count, or vice-versa.
- `Chapter IV` read as "Chapter eye-vee" or "Chapter six".
- `9:45 AM` read as a fraction or a ratio.
- Telephone/electronic strings mangled into nonsense.

These drive the optimization. A model that trades several benign errors to remove one silly error is **preferred**.

### Benign errors (tolerable)
Understandable, stylistically-different readings the listener recovers from instantly:
- "three quarters" vs "three fourths" for `3/4`.
- "oh nine forty-five" vs "nine forty-five A M" for `9:45 AM`.
- Minor article/connector phrasing differences that don't change meaning.

These still cost EM points (EM is exact-match), but they do **not** increment the unrecoverable-error count.

---

## 7. Mapping metrics to listener experience

| Metric | What the listener hears |
|---|---|
| **Sentence EM** | Fraction of sentences rendered exactly as intended — directly proportional to "this audiobook sounds correct." |
| **Hard-slice EM** | How the book sounds at its trickiest moments (years, money, abbreviations, chapter numerals) — where naive TTS embarrasses itself. |
| **Macro per-class acc** | Whether rare-but-jarring constructs (phone numbers, Roman numerals, measures) are handled — a single botched rare class is what listeners remember and complain about. |
| **Unrecoverable-error count → 0** | The number of moments a listener is actively misled (wrong figure, wrong year). This is the metric tied most directly to trust; the project target is **zero**. |

In short: EM and macro accuracy measure *how often it's right*; the silly-error count measures *how badly it's wrong when it is wrong*. An audiobook can survive occasional benign rephrasings but not a wrong dollar figure read aloud with confidence — which is why the evaluation weights unrecoverable errors as a first-class, separately-tracked quantity, and why the hard slice, not the blended overall number, is the headline discriminator between the neural model and the baseline.

---

## 8. Reproducing the evaluation

```bash
# 1. Regenerate the corpus (train/val/test/hard, leakage-free, deduped)
audiobook-ai data

# 2. Evaluate the trained model on test + hard slices
audiobook-ai evaluate          # -> src/audiobook_ai/training/evaluate.py

# 3. Per-class table + silly/benign taxonomy + neural-vs-baseline deltas
audiobook-ai error-analysis    # -> analysis/error_analysis.py
```

All numbers in this document are computed from the same synthetic distribution and the same metric definitions for both systems, so the baseline-vs-neural comparison is apples-to-apples and fully reproducible.
