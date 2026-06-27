# models/

Trained model checkpoints live here (git-ignored — only this README is committed).

The fine-tuned Text-Normalization model is written by the trainer to
`${AUDIOBOOK_AI_MODEL_DIR}/tn_normalizer/<version>/` with a `latest` pointer:

```
models/tn_normalizer/
├── byt5-small-YYYYMMDD-HHMMSS/    # the versioned checkpoint (weights + tokenizer + tn_meta.json)
└── latest/                        # pointer to the most recent version (symlink or marker)
```

- Train: `audiobook-ai --config configs/train.yaml train`
- The agent / API resolve `latest` automatically; if no checkpoint exists they fall
  back to the **rule baseline** (so the system always runs).
- Override the location with `AUDIOBOOK_AI_MODEL_DIR`; on Colab point it at Google Drive
  so checkpoints survive disconnects (resume-safe via `get_last_checkpoint`).

Pretrained models (downloaded to `HF_HOME`, never committed):
`google/byt5-small`, `microsoft/speecht5_tts`, `microsoft/speecht5_hifigan`,
`Matthijs/cmu-arctic-xvectors` (+ optional `hexgrad/Kokoro-82M`).
