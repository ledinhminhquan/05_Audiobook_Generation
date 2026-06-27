# Ethics & Responsible AI Statement — P05 Audiobook Generation System

**Project:** Audiobook Generation System — turning long documents (EPUB/PDF/TXT/MD) into mastered, chaptered audiobooks (`.wav` / `.mp3` / `.m4b` + `.srt`).
**Author:** Le Dinh Minh Quan (student 23127460) — NLP in Industry, final assignment.
**Scope of this document:** who benefits, who could be harmed, bias and fairness risks, explainability for non-technical stakeholders, foreseeable misuse, and the concrete safeguards built into this system.

This statement is deliberately critical and balanced. The system's trainable ML heart is a **Text-Normalization (TN) seq2seq model** (`google/byt5-small`, fallback `google-t5/t5-small`); the audio is produced by **pretrained neural TTS backends** (primary `microsoft/speecht5_tts`); and a deterministic agent FSM orchestrates the full document-to-audiobook pipeline. The ethical surface area is shaped by all three: text understanding, synthetic voice, and automation at catalog scale.

---

## 1. Who benefits

| Beneficiary | Benefit, grounded in the system design |
| --- | --- |
| **Readers with disabilities** | Sight-impaired and dyslexic readers gain spoken access to documents that exist only as text. Accessibility is an explicit business success metric. |
| **Commuters / time-constrained learners** | Listening replaces reading; "time-to-first-audio" is optimized by streaming chapter 1 while later chapters render. |
| **Independent authors** | The cost per finished audio-hour is far below human narration, putting audiobook production within reach of authors who could never afford a studio. |
| **Publishers** | Faster catalog conversion of back-catalog titles into audio; chaptered `.m4b` with markers and `.srt` subtitles are produced automatically. |
| **Learners / non-native speakers** | Subtitles (`.srt`) synchronized to audio support follow-along reading and language learning. |

The honest framing: the value is **scale and cost**, not artistic performance. This system does not claim to match a skilled human narrator's interpretive reading. It claims to make *more text audible to more people, more cheaply* — which is precisely why the harms below must be taken seriously.

---

## 2. Who could be harmed

### 2.1 Human narrators' livelihoods
A cost per audio-hour that is "much less than human narration" is a feature for publishers and a threat to professional narrators. Cheap synthetic narration can displace paid voice work, especially for mid-list and back-catalog titles where margins are thin. This is a genuine labor-displacement risk that no technical safeguard fully resolves. The system mitigates *quality-based* false equivalence (see §3 and §5) but cannot, by itself, protect a profession; that requires policy and procurement choices by the people deploying it.

### 2.2 Voice-clone misuse and impersonation
The optional high-quality voice-cloning backends can reproduce a *specific* person's voice. Without controls, this enables impersonation, fraudulent endorsements, and deepfake narration attributed to a real person. This is the single most dangerous capability in the system and is treated as such: cloning backends are **non-commercial-licensed, consent-gated, and OFF by default** (see §5).

### 2.3 Copyright holders
Converting a book into an audiobook is a **derivative work**. Running an in-copyright text through this pipeline without authorization infringes the rights holder. At catalog scale, the same automation that helps a publisher convert its own catalog could mass-produce *pirated* audiobooks from texts the operator has no rights to.

### 2.4 People named or described inside the books
Books contain PII — phone numbers, addresses, emails, and personal names. Pipeline logging of normalized segments could inadvertently persist this data.

---

## 3. Bias & fairness risks

A text-to-speech system encodes whose language and whose voice are treated as "default." This system carries several concrete biases, which we name rather than hide:

- **Voice accent/dialect coverage.** The primary voice bank is `Matthijs/cmu-arctic-xvectors` (7931 embeddings, **7 speakers**). Seven speakers is a narrow slice of human vocal diversity. Multi-voice routing (narrator / dialogue / heading) selects among these few x-vector indices, so the range of accents, ages, and dialects a listener hears is structurally limited. A book set in a specific region may be read in a voice that does not represent its characters.
- **English-/US-centric normalization.** The TN model is trained on an English-centric synthetic corpus across 16 semiotic classes (CARDINAL, MONEY, DATE, TIME, MEASURE, ABBREV, etc.). Normalization conventions — currency read as dollars, US date ordering, imperial measure habits — bias toward US English. Text in other locales, or non-English passages, can be normalized incorrectly.
- **Name pronunciation.** Proper names are a known failure mode for any TTS. The normalizer is byte/char-level (ByT5), which is robust to OOV tokens *as strings* but does not guarantee culturally correct pronunciation of names. Mispronouncing a person's or place's name is a real, dignity-affecting error, not a cosmetic one. The system's "unrecoverable / silly error" metric (after Sproat & Jaitly) is designed to catch *catastrophic* mis-reads; subtle name mispronunciations may still slip through.

The fairness posture is therefore: **honest about coverage limits, instrumented to catch the worst errors, and explicit that benign quality gaps remain.** The baseline-vs-trained comparison is reported honestly — the context-blind rule baseline scores sentence exact-match EM easy = 0.945, hard = 0.006, overall = 0.712 on the synthetic test distribution; the trained model is expected to beat it on both slices, especially the hard one. We do not overclaim parity with human reading.

---

## 4. Explainability for non-technical stakeholders

A core ethical strength of this design is that **every reading is auditable** without reading code. Three artifacts make a non-technical stakeholder — an editor, a rights manager, an author — able to inspect exactly what the system decided and why:

1. **The per-segment normalized script.** For each text segment, the written form and its spoken normalization are recorded. A reviewer can see that `"$5.2M"` became "five point two million dollars" *before* any audio was generated, and correct it if wrong.
2. **The decision log.** The agent is a deterministic FSM with four recorded decision points, each captured as a `Decision` record:
   - **D1 — parse-quality routing** (`parse_score` → structured ≥ 0.85 / assisted 0.5–0.85 / degraded < 0.5).
   - **D2 — normalization-confidence escalation** (per-segment confidence below `norm_confidence_min` = 0.55 is flagged; neural-vs-baseline disagreement is recorded).
   - **D3 — audio-QA re-synthesis gate** (empty/NaN, duration ratio, peak/clipping, silence fraction; bounded re-synth, max 2 attempts).
   - **D4 — voice routing** (heading / dialogue / narration → distinct, book-stable voices).
3. **The content-hash manifest (`manifest.json`).** Every step is timed and traced (`ToolTrace`); the manifest records the full provenance of the run, including the input's **SHA-256** content hash and per-clip QA outcomes.

```text
document → PARSE → CHAPTER → SEGMENT+CLASSIFY → NORMALIZE (per-segment, confidence)
         → VOICE-ROUTE → SYNTHESIZE → AUDIO-QA → STITCH → EXPORT + manifest.json
```

Because the normalized script, decision log, and manifest are produced for *every* run, a complaint such as "the narrator read my name wrong in chapter 4" can be traced to the exact segment, the exact normalization, and the exact voice index — and fixed. This is a far higher standard of accountability than an opaque end-to-end model that emits audio with no inspectable intermediate.

---

## 5. Potential misuse and safeguards

We treat three misuse scenarios as foreseeable and design against each:

| Misuse | Safeguard built into the system |
| --- | --- |
| **Cloning a real person's voice without consent** (impersonation, deepfake narration) | Voice-cloning backends (`coqui/XTTS-v2` CPML, `SWivid/F5-TTS` CC-BY-NC, `facebook/mms-tts-eng` CC-BY-NC) are **non-commercial-licensed, consent-gated, and OFF by default**. The default voices are the permissively licensed, generic SpeechT5 x-vectors — not a clone of any identifiable individual. |
| **Mass-producing pirated audiobooks** from texts the operator has no rights to | Rights / provenance checks: converting a book is treated as a derivative work; the operator is expected to verify rights or public-domain status. The input's **SHA-256 is recorded in the manifest**, creating a tamper-evident record of *what* was converted. |
| **Passing synthetic narration off as a real person, or laundering provenance** | The **content-hash manifest** plus full decision/trace logging make every reading attributable to its inputs and parameters. Audio is auditable end-to-end rather than anonymous. |

Additional safeguards that reduce harm in normal operation:

- **Permissive-only commercial path.** Any commercial deployment is restricted to **MIT / Apache-2.0** model IDs only — `microsoft/speecht5_tts` + `speecht5_hifigan` (MIT), `Matthijs/cmu-arctic-xvectors` (MIT), `hexgrad/Kokoro-82M` (Apache-2.0), `parler-tts/parler-tts-mini-v1` (Apache-2.0), and the Apache-2.0 TN models. The non-commercial cloning models are excluded from that path by license hygiene, not just by policy.
- **Guardrails against hallucinated pronunciations.** A rule-based normalizer guardrail, the D3 audio-QA gate, and explicit **"unrecoverable-error" counting** (driven toward zero) reduce the chance of confident, wrong, or fabricated readings reaching the listener.
- **PII minimization.** Books contain phones, addresses, emails, and names; the system minimizes logging of normalized content, applies TTL cleanup, and uses NER only for attribution — not for retention or profiling.
- **Graceful, non-deceptive degradation.** When inputs are out-of-distribution (scanned PDFs route to a degraded flat-text path; garbled text yields a low `parse_score`), the system flags reduced quality rather than silently emitting confident-sounding but wrong audio. A rule normalizer, `pyttsx3`, and a deterministic `PlaceholderTTS` floor mean the pipeline never hard-fails — but it also never pretends a degraded run is a clean one.

---

## 6. Residual risks we do not claim to have solved

Responsible disclosure means naming what the safeguards *do not* cover:

- **Labor displacement** of human narrators is an economic harm that the deployer, not the model, must weigh.
- **Consent gating is procedural.** The system can refuse to enable cloning by default and require an attestation, but it cannot independently verify that a claimed consent is genuine. The honesty of the operator remains a dependency.
- **Rights verification is the operator's responsibility.** Recording an input SHA-256 proves *what* was converted; it does not prove the operator *had the right* to convert it.
- **Voice and locale coverage is narrow** (7 default speakers; US-English normalization). Listeners outside that envelope receive a less representative experience.
- **Name pronunciation** can still be subtly wrong below the "catastrophic error" threshold.

We consider it more ethical to ship these limits in writing — alongside an auditable manifest and decision log — than to imply the system is neutral, complete, or risk-free. It is a cost-reducing accessibility tool with real benefits and real, named hazards, and it is built to be inspected.
