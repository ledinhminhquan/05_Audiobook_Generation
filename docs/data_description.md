# Data Description — P05 Audiobook Generation (Text Normalization)

**Project:** Audiobook Generation System — turn long documents (EPUB/PDF/TXT/MD) into mastered, chaptered audiobooks.
**Author:** Le Dinh Minh Quan (student 23127460) — *NLP in Industry*, final assignment.
**Scope of this document:** the data behind the project's single **trainable** model — a **Text-Normalization (TN)** seq2seq model that converts *written* text into its *spoken* form. The audio backends (SpeechT5, Kokoro, etc.) are **pretrained** and are not covered here because they require no training data from us.

---

## 1. Why this data exists (the problem the data must teach)

A naive TTS engine mis-reads almost every non-plain token. It will say "dollar five point two M" for `$5.2M`, "one thousand nine hundred eighty four" for the title *1984*, "Dee Arr" for `Dr.`, "Chapter eye-vee" for `Chapter IV`, "three slash four" for `3/4`, and "nine colon forty-five A M" for `9:45 AM`. The hard, NLP-heavy part of audiobook production is therefore **text normalization (written → spoken)**.

The model must learn a mapping over **16 semiotic classes** and — critically — must learn it **in context**, because many tokens are ambiguous and can only be resolved from surrounding words. The dataset is engineered specifically to expose and teach that context-dependence.

### The 16 semiotic classes

| # | Class | Written example | Spoken target (illustrative) |
|---|-------|-----------------|------------------------------|
| 1 | PLAIN | `book` | book |
| 2 | PUNCT | `,` | (silence / prosody) |
| 3 | CARDINAL | `27` | twenty seven |
| 4 | ORDINAL | `3rd` | third |
| 5 | DECIMAL | `3.14` | three point one four |
| 6 | DIGIT | `007` | zero zero seven |
| 7 | MONEY | `$5.2M` | five point two million dollars |
| 8 | MEASURE | `5km` | five kilometers |
| 9 | DATE | `1984` (as year) | nineteen eighty four |
| 10 | TIME | `9:45 AM` | nine forty five A M |
| 11 | FRACTION | `3/4` | three quarters |
| 12 | TELEPHONE | phone number | digit-by-digit reading |
| 13 | ELECTRONIC | URL / email | spoken URL form |
| 14 | ABBREV | `Dr.` | doctor |
| 15 | ROMAN | `Chapter IV` | Chapter four |
| 16 | VERBATIM | letters read as-is | letter-by-letter reading |

These are the standard Sproat & Jaitly text-normalization semiotic classes; the model is trained and evaluated against all 16, and a **macro per-class accuracy** metric ensures rare classes are not drowned out by the dominant PLAIN class.

---

## 2. Data sourcing

### 2.1 Primary source — a reproducible **synthetic** corpus generator

The primary training data is **synthetic**, produced by an in-repo generator:

```
src/audiobook_ai/data/tn_corpus.py
```

It emits context-rich `(written, spoken)` sentence pairs covering all 16 classes.

**Why synthetic is the primary source (not a convenience choice):** there is **no permissively-licensed English TN-Challenge mirror on the Hugging Face Hub**. The canonical dataset ids return **404**:

- `google/text_normalization` → 404 (does not exist on the Hub)
- `cestwc/text-normalization` → 404 (does not exist on the Hub)

Because no redistributable, correctly-licensed English TN dataset is available to depend on, a reproducible generator is the only way to guarantee that any grader can rebuild the exact corpus from source with no missing/license-encumbered download. The generator is deterministic and version-controlled, so the dataset is fully reproducible.

### 2.2 Optional real-data loader

The repo also ships an **optional real-data loader** so the model can be trained or fine-tuned on real TN data if a user supplies it locally. This is opt-in and not required for the headline results.

### 2.3 Optional evaluation sanity set

For an independent sanity check, the pipeline can pull a small **verified** evaluation set from the Hub:

```
DigitalUmuganda/Text_Normalization_Challenge_Unittests_Eng_Fra   (VERIFIED)
```

This is used only as an external eval/sanity probe, not as the primary training corpus.

> **Summary of sourcing strategy:** synthetic generator = PRIMARY (reproducible, license-clean, full class coverage with injected ambiguity); optional real loader = opt-in fine-tune path; DigitalUmuganda unittests = optional external sanity check.

---

## 3. How the synthetic pairs are generated (including injected ambiguity)

The generator does **not** simply pair an isolated token with its reading. It builds **full sentences** that place each token in a context, then records the correct spoken form for that context. The whole point is to force the model to use context to disambiguate.

**Injected-ambiguity patterns** (designed to break a context-blind normalizer):

- **Abbreviation polysemy:** `St.` → **Street** vs **Saint**, decided by surrounding words.
- **Year vs count:** `"1984 people"` → cardinal **count** ("one thousand nine hundred eighty four people") vs `"in 1984"` → **year** ("nineteen eighty four").
- **Dates with day-as-ordinal:** the day component must be read as an ordinal in date contexts.
- **Roman-numeral contexts** that the rule baseline systematically misses (e.g. chapter/section/regnal-name Roman numerals).

These ambiguous constructions are concentrated in a dedicated **hard slice**, so we can measure context understanding directly rather than letting easy cases hide the failures.

### Dataset size & language

- **Language:** **English only.**
- **Default split sizes:**

| Split | Size | Role |
|-------|------|------|
| train | 60,000 | model training |
| val   | 4,000  | early-stopping / model selection |
| test  | 4,000  | held-out general evaluation |
| hard  | 1,500  | held-out **ambiguity-stress** slice |

Total ≈ **69,500** generated pairs at defaults.

---

## 4. Preprocessing

Applied consistently across all splits:

1. **Unicode normalization (NFC).** All text is NFC-normalized so visually-identical strings share one byte representation — important because the model is byte/char-level (ByT5).
2. **Task prefix.** Every input is prefixed with `"normalize: "` (the T5/ByT5 seq2seq task-prefix convention).
3. **Deduplication.** Identical `(written, spoken)` pairs are **deduplicated before splitting**, so the same example cannot appear in two splits.
4. **Leakage-free splits.** Splits are constructed so that **no validation or test input appears in the training set** — there is no train/eval overlap, which keeps the reported accuracy honest.

### Justification for the splits

- **60k train** is large enough to cover all 16 classes with many context variants each, while still training the ByT5-small model in a few hours on a single H100.
- **4k val** gives a stable signal for early stopping and model selection without wasting examples that could be training data.
- **4k test** is a like-distribution held-out set for the headline sentence-accuracy number.
- **1.5k hard** is a *separate* held-out slice of deliberately ambiguous cases. Keeping it apart from `test` lets us report two numbers — "does the model work in general?" (test) and "does the model actually use context?" (hard) — instead of letting easy cases inflate a single average.

---

## 5. Handling noisy / biased / imbalanced data (anti-overfitting)

Text normalization is naturally dominated by PLAIN tokens, and a synthetic generator can over-represent whatever it generates most. Several mechanisms keep the data and the model honest:

- **PLAIN class capped at ≈30%** of the corpus, so the model cannot get a high score by trivially copying plain words; the harder semiotic classes get real representation.
- **Deduplication before split** (Section 4) removes exact-duplicate inflation.
- **Held-out hard slice** (1.5k) measures the ambiguous behavior the baseline fails on.
- **Macro per-class accuracy** is tracked alongside sentence accuracy to expose rare-class failures.
- Training-side regularization that protects against memorizing the synthetic data: **early stopping** (patience 4 on sentence accuracy), **label smoothing (0.1)**, **weight decay (0.01)**, and dropout.

The headline error metric explicitly prioritizes the **Sproat & Jaitly "unrecoverable / silly error"** notion — semantically catastrophic mis-reads — minimizing those even at the cost of some benign mismatches.

---

## 6. Baseline reference (what the data must let us beat)

The baseline is a **context-blind rule normalizer**:

```
src/audiobook_ai/models/baseline_rules.py
```

It is ordered regex spans + an abbreviation dictionary + `num2words`/pure-Python expanders. It is strong on easy, unambiguous cases and structurally unable to resolve the injected ambiguity.

**Validated baseline numbers on the synthetic test distribution** (sentence-level exact-match, EM):

| Slice | Baseline EM |
|-------|-------------|
| easy | **0.945** |
| hard | **0.006** |
| overall | **0.712** |

The near-zero `hard` score is the key result: it confirms that the hard slice genuinely requires context and is not solvable by rules. A trained context-aware model is expected to score high on **both** slices, beating the baseline overall and dramatically on the hard slice — an honest, reproducible improvement.

---

## 7. Known limitations & potential biases

These are intrinsic to the dataset and are stated openly so results are not over-claimed:

- **Synthetic-vs-real gap.** The primary corpus is generated, not harvested from real books. Real-world text has distributional quirks (OCR noise, unusual formatting, domain jargon) that the generator only partially reproduces. The optional real-data loader and the DigitalUmuganda sanity set partly mitigate this, but a synthetic↔real domain gap remains.
- **English-only.** The corpus is English exclusively; the model is not trained to normalize other languages.
- **US-centric formats.** Money (`$`), telephone, and related formats follow US conventions. Non-US currency symbols, date orders, and phone formats are under-represented and may normalize incorrectly.
- **Capped class imbalance is still imbalance.** PLAIN is capped at ≈30% rather than eliminated; the natural rarity of some semiotic classes means per-class sample counts still vary, which is why macro-class accuracy is reported.
- **Generator bias.** Any systematic blind spot in the generator (a context pattern it never produces) becomes a blind spot in the model. The held-out hard slice probes for this but cannot prove its absence.

---

## 8. Reproducibility summary

| Property | Value |
|----------|-------|
| Primary source | Synthetic generator `src/audiobook_ai/data/tn_corpus.py` (deterministic) |
| Why synthetic | No permissive English TN-Challenge mirror on HF Hub (`google/text_normalization`, `cestwc/text-normalization` → 404) |
| Optional real loader | Opt-in local real-data path |
| Optional sanity eval | `DigitalUmuganda/Text_Normalization_Challenge_Unittests_Eng_Fra` (VERIFIED) |
| Language | English |
| Splits (train/val/test/hard) | 60,000 / 4,000 / 4,000 / 1,500 |
| Preprocessing | NFC, `"normalize: "` prefix, dedup, leakage-free splits |
| Classes | 16 semiotic classes (PLAIN … VERBATIM) |
| Baseline EM (easy/hard/overall) | 0.945 / 0.006 / 0.712 |

The entire corpus can be regenerated from source with no external downloads, making every reported number reproducible by a grader.
