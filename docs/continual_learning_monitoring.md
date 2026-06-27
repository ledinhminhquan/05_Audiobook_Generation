# Continual Learning & Monitoring

**Project:** P05 Audiobook Generation System
**Author:** Le Dinh Minh Quan (23127460) — NLP in Industry, final assignment
**Scope of this document:** how the trainable Text-Normalization (TN) model keeps improving after deployment, how we detect when it is silently getting worse, and which signals we watch in production.

> The only *trained* component in P05 is the Text-Normalization seq2seq model (`google/byt5-small`, fallback `google-t5/t5-small`) that converts written tokens to spoken form across the 16 semiotic classes (PLAIN, PUNCT, CARDINAL, ORDINAL, DECIMAL, DIGIT, MONEY, MEASURE, DATE, TIME, FRACTION, TELEPHONE, ELECTRONIC, ABBREV, ROMAN, VERBATIM). The TTS backends (`microsoft/speecht5_tts` + HiFi-GAN vocoder) are pretrained and frozen. Continual learning therefore targets the **TN model** and, indirectly, the agent's confidence thresholds. Audio quality is monitored, not retrained.

---

## 1. Why continual learning matters here

The TN model is trained primarily on a **reproducible synthetic corpus** (`src/audiobook_ai/data/tn_corpus.py`) with injected ambiguity (e.g. `St.`→Street vs Saint, `1984 people`→count vs `in 1984`→year, day-as-ordinal dates, Roman-numeral contexts). Synthetic data, however realistic, cannot anticipate every real-world surface form. Real books continually introduce token shapes the generator never produced: new currency formats, exotic measurement units, domain jargon, and OCR garble from scanned PDFs. Without a feedback loop, the model's accuracy on *the distribution we actually serve* drifts away from the distribution it was trained on while offline metrics stay flat. The mechanisms below close that loop.

---

## 2. How new data is collected

New training signal is harvested **automatically during every production run** and routed into a feedback store. Nothing requires a separate labeling campaign to get started; the pipeline already produces the right artifacts.

### 2.1 Sources of new data

| Source | Where it comes from | Signal it carries |
|---|---|---|
| **Low-confidence segments** | Decision **D2** flags any segment whose length-normalized sequence probability falls below `norm_confidence_min = 0.55` | "The model itself is unsure here" — high-value candidates for review |
| **Neural-vs-baseline disagreements** | D2 also records where the trained ByT5 output differs from the context-blind rule normalizer (`src/audiobook_ai/models/baseline_rules.py`) | Either the model fixed an ambiguity the rules miss, or it regressed — both worth inspecting |
| **Audio-QA failures** | Decision **D3** re-synthesis gate (empty/NaN, bad duration ratio, peak/clipping, excess silence) | A QA failure often traces back to a normalization that produced unspeakable text |
| **User corrections** | Edits applied via the `POST /normalize` preview / UI before the user accepts a book | Gold human labels: written input + corrected spoken form |

### 2.2 The feedback store

Each captured item is stored as a structured record: the **written input**, the **model's spoken output**, the **confidence**, the **baseline output** (for disagreement cases), any **human correction**, and provenance (input `sha256`, model registry revision, semiotic class guess, timestamp). Because P05 already records the input `sha256` and a full `manifest.json` per run, every flagged segment is traceable back to its book and model version. PII hygiene from the project's ethics policy applies to this store too: phones/addresses/emails/names are minimized in logs and subject to TTL cleanup, so the feedback store keeps the *normalization pair* (e.g. `(212) 555-0199` → spoken digits) only as long as needed for label review, not indefinitely.

---

## 3. How retraining / fine-tuning occurs

Retraining is **periodic**, not online — the model is never updated mid-book. A scheduled job assembles a fresh corpus, fine-tunes, validates, and promotes a new version through the registry with a canary/AB rollout.

### 3.1 The augmented corpus

The next training run mixes three streams:

1. **Regenerated synthetic corpus** from `tn_corpus.py` (defaults: train 60000, val 4000, test 4000, hard 1500), with the generator extended to cover any newly observed token shapes (see drift mitigations in §6).
2. **Real corrections** harvested from the feedback store (§2) — human-verified `(written, spoken)` pairs, which carry the most weight because they reflect true production distribution.
3. **Optional real-data eval sanity set** `DigitalUmuganda/Text_Normalization_Challenge_Unittests_Eng_Fra` (verified) for an external cross-check.

The same anti-overfitting and anti-leakage discipline used for the first model is preserved: dedup pairs before split, cap `PLAIN ≤ ~30%`, and guarantee **no val/test input appears in train**. Corrections from the feedback store are de-duplicated against the synthetic pool so a single recurring error does not flood the corpus.

### 3.2 Training configuration

Fine-tuning reuses the established recipe (HF `Seq2SeqTrainer`, `predict_with_generate`, task prefix `"normalize: "`):

- `learning_rate` 5e-4 (byt5) / 3e-4 (t5-small), effective batch 256, cosine schedule, `warmup_ratio` 0.05, `weight_decay` 0.01, `label_smoothing` 0.1, dropout 0.1, `max_grad_norm` 1.0.
- Early stopping (patience 4 on `sentence_accuracy`), `load_best_model_at_end`, resume via `get_last_checkpoint`.
- bf16 + tf32 on H100/A100; held-out **hard slice** is the canary metric for ambiguity handling.
- Rough cost: byt5-small ~3–6 h on one H100 for the full corpus (cap `max_steps` to bound it); t5-small ~3–4× faster, fits T4.

### 3.3 Registry versioning and canary / AB rollout

Every promoted model is versioned through the **model registry** (`tn_meta.json` + a `latest` pointer, with `repo@revision` pins). A new candidate is **not** swapped in globally:

1. **Offline gate** — the candidate must beat the current production model on the held-out test and hard slices, and must not regress macro per-class accuracy (the metric that exposes rare-class failures). Recall the validated **baseline** on the synthetic test distribution: sentence EM easy = 0.945, hard = 0.006, overall = 0.712 — a candidate that does not clear the trained model's bar on the hard slice is rejected.
2. **Canary** — promote behind the `latest` pointer for a small fraction of books; compare live monitoring metrics (§5) against the incumbent.
3. **AB** — if the canary holds, run an A/B split so the new revision and the previous revision serve in parallel and we can attribute any metric movement to the model change rather than to input mix.
4. **Promote or roll back** — because the registry pins `repo@revision`, rollback is a one-line pointer change; no redeploy of the API/worker images is required.

---

## 4. How performance degradation is detected

Degradation is detected by **comparing live production signals against a fixed reference**, not by waiting for user complaints. Three reference anchors are used:

- **A golden set** — a frozen, human-curated set of `(written, spoken)` pairs spanning all 16 classes. We compute per-class accuracy on it for every model revision; a drop on any class is a regression signal even if overall accuracy looks fine.
- **The shipped offline test/hard slices** — the numbers a model was promoted on; live behavior should not diverge from them.
- **The incumbent model** (during canary/AB) — relative movement isolates model-caused change from input-caused change.

The dedicated entry point for this comparison is **`src/audiobook_ai/monitoring/drift_report.py`**, exposed via the CLI as `audiobook-ai monitor` / `generate-report`. It aggregates the production signals below over a window, diffs them against the reference anchors, and emits a drift report (flag-rate trend, per-class golden-set accuracy, RTF distribution, confidence histogram). When a metric crosses its threshold, the report marks the model as a retraining candidate and the harvested flagged segments (§2) are exactly the data needed to fix it.

---

## 5. Proposed monitoring metrics

These are the signals `drift_report.py` tracks. Each has a direction-of-concern and ties back to a concrete pipeline mechanism.

| Metric | Definition | Why it matters | Concern direction |
|---|---|---|---|
| **TN flag-rate** | Fraction of segments flagged by D2 (confidence `< 0.55` or neural-vs-baseline disagreement) | Rising flag-rate = model increasingly unsure = distribution shift | ↑ bad |
| **Audio-QA fail-rate** | Fraction of clips that fail D3 checks (empty/NaN, duration ratio, peak/clipping, silence fraction) and need re-synth | Spikes often originate in bad normalization producing unspeakable text | ↑ bad |
| **RTF drift** | Real-Time Factor reported per run in `manifest.json` (validated: ~2.28 CPU, ~0.1 GPU) | Sudden RTF growth signals longer generations (e.g. model emitting verbose/looping output) or infra regression | ↑ bad |
| **Confidence distribution** | Histogram of length-normalized sequence probabilities across all segments | A left-shifting distribution warns of drift *before* the flag-rate threshold trips | shift left = bad |
| **Per-class accuracy on the golden set** | Sentence/class exact-match per semiotic class on the frozen golden set | Catches rare-class regressions (e.g. MONEY, DATE, ROMAN) that overall accuracy hides | ↓ bad |

Two reference numbers are always carried alongside live metrics for context: the validated baseline EM (easy 0.945 / hard 0.006 / overall 0.712) and the project's loudness conformance target (ACX **−18 LUFS / −3 dBTP**). We also keep the Sproat & Jaitly notion of **unrecoverable / "silly" errors** (semantically catastrophic mis-reads) as a tracked count whose target is →0 — a single rise here is treated as more serious than a small dip in benign exact-match.

---

## 6. Model drift risks and mitigation

Drift here is overwhelmingly **input drift**: the world keeps producing written forms our synthetic generator never modeled. The character-level ByT5 backbone is inherently robust to novel numbers, symbols, and OOV (a deliberate model choice), which softens — but does not eliminate — these risks.

| Drift risk | Example | Mitigation |
|---|---|---|
| **New number / currency formats** | A new currency symbol, thousands-separator convention, or `$5.2M`-style shorthand the corpus lacks | Extend `tn_corpus.py` generators to emit the new shapes; rule guardrail in `baseline_rules.py` (num2words / pure-python expanders) provides a safe fallback; ByT5 char-level robustness handles unseen glyphs gracefully |
| **New domains** | A medical, legal, or scientific book with unfamiliar ABBREV / MEASURE jargon | Harvest flagged D2 segments from that domain, add human-verified corrections to the augmented corpus, re-train; per-class golden-set accuracy flags the ABBREV/MEASURE regression early |
| **OOD documents** | Scanned PDFs producing garbled flat text; unusual structure | Agent **D1** parse-quality routing (`parse_score`) sends low-quality input down the **degraded flat-text path** instead of forcing structured assumptions; garbled text yields a low `parse_score` and is contained, not silently mis-read |
| **Confidence miscalibration over time** | Model becomes over- or under-confident as input mix shifts, skewing the D2 flag-rate | Monitor the confidence distribution (§5); re-tune `norm_confidence_min` (0.55) during periodic retraining so the flag-rate stays in a useful operating range |
| **Catastrophic mis-reads** | A confident-but-wrong normalization that TTS will read aloud | Rule guardrail + D3 audio-QA gate + the unrecoverable-error count act as defense-in-depth; D3's bounded re-synth (max 2 attempts: reseed → split → fallback backend) prevents a single bad segment from failing the whole book |

### Closing the loop

The end-to-end mechanism is intentionally circular: production **flags** (D2/D3) → **feedback store** (§2) → **augmented corpus** (§3.1) → **periodic fine-tune** (§3.2) → **registry-versioned canary/AB** (§3.3) → **drift monitoring** (`drift_report.py`, §4–5) → new flags. Each turn of the loop replaces guesses in the synthetic generator with verified real corrections, so the TN model's accuracy on the *served* distribution — not just the synthetic one — improves over time while the monitoring metrics give an early, quantitative warning whenever it does not.
