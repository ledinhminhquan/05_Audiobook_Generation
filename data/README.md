# data/

This project's **primary training corpus is generated, not downloaded** — there is
no permissively-licensed English Text-Normalization Challenge mirror on the HF Hub.

- The synthetic `(written, spoken)` TN corpus is built by
  `src/audiobook_ai/data/tn_corpus.py` and cached under
  `${AUDIOBOOK_AI_DATA_DIR}/tn_corpus/{train,val,test,hard}.jsonl`.
- Build it with: `audiobook-ai data --task corpus`
  (or it is auto-built on first `train` / `evaluate`).

Large artifacts (the cached corpus, downloaded models, generated audio) are
**git-ignored** — only this README is committed. Nothing here is hard-coded:
all paths come from environment variables (`AUDIOBOOK_AI_DATA_DIR`,
`AUDIOBOOK_AI_ARTIFACTS_DIR`, `HF_HOME`).

Optional external data (auto-handled, all degrade gracefully if offline):
- `Matthijs/cmu-arctic-xvectors` — SpeechT5 speaker embeddings (MIT).
- `DigitalUmuganda/Text_Normalization_Challenge_Unittests_Eng_Fra` — small English eval set.
