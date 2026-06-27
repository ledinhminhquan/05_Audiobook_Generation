# ☁️ Colab Training Guide — Audiobook Generation System

Step-by-step to train the **ByT5 Text-Normalization** model on Colab (Pro/Pro+),
then test it and collect the deliverables. The notebook auto-adapts to **H100 / A100 / L4 / T4**
and **resumes** after a disconnect.

---

## 0. What you need
- A Google account with **Colab** (Pro+ recommended for H100/A100; the notebook also runs on T4/L4).
- (Recommended) A **public GitHub repo** containing this project (`05_Audiobook_Generation`).
  No GitHub? You can instead upload the project folder to Google Drive (see step 2).

## 1. Get the project onto Colab
**Option A — GitHub (recommended).** Push this folder to a public repo, e.g.
`https://github.com/<you>/05_Audiobook_Generation`.

**Option B — Google Drive.** Upload the whole `05_Audiobook_Generation/` folder to
`MyDrive/05_Audiobook_Generation/` (the notebook will copy it).

## 2. Drive layout (artifacts persist here, so training survives disconnects)
You do **not** need to pre-create anything except (Option B) the project folder.
The notebook creates and uses:
```
MyDrive/
├── 05_Audiobook_Generation/        # (Option B only) the uploaded project source
└── audiobook_ai/                   # auto-created — all artifacts live here
    ├── data/        # cached synthetic TN corpus
    ├── models/      # trained checkpoints + tn_normalizer/latest
    ├── runs/        # eval / benchmark / analysis JSON
    ├── outputs/     # generated audiobooks (wav/mp3/m4b/srt)
    ├── submission/  # report.pdf + slides.pptx + bundle.zip
    └── hf_cache/    # HuggingFace model cache
```

## 3. Open & configure the notebook
1. Open `notebooks/Audiobook_AI_Colab_Training_H100_AUTOPILOT.ipynb` in Colab.
2. `Runtime → Change runtime type → GPU` (H100 if available; A100/L4/T4 all work).
3. In **cell 0 (Controls)** set:
   - `GIT_REPO_URL` → your repo URL (Option A), or leave blank for Option B.
   - `BASE_MODEL` → `auto` (recommended: ByT5 on big GPUs, t5-small on T4).
   - `TRAIN_SIZE` / `EPOCHS` → defaults (60000 / 3) are good; lower `TRAIN_SIZE`
     (e.g. 15000) for a quick run.
   - `TTS_BACKEND` → `speecht5` (default) or `kokoro` (needs `pip install kokoro`).

## 4. Run
`Runtime → Run all`. The notebook will:
1. mount Drive + set artifact paths, install Colab-safe deps (never touches torch),
2. auto-profile the GPU (batch + precision, **effective batch held at 256**),
3. build the synthetic corpus,
4. **autopilot** (cell 11): train → evaluate → analysis → `report.pdf` + `slides.pptx`.

**Disconnected?** Just re-open and **re-run cell 11** — it resumes from the last
checkpoint on Drive (`get_last_checkpoint`).

## 5. Verify it worked
- **Cell 12b / 13** — `evaluate` should show the trained model **beating the rule baseline**,
  especially on the **hard** (ambiguous) slice (baseline ≈ 0.0 there).
- **Cell 14a** — the model normalizes tricky inputs correctly, e.g.
  `Dr. Vance lives on Oak Dr.` → *Doctor Vance lives on Oak Drive* (the baseline says *Doctor … Oak doctor*).
- **Cell 14b** — listen to the generated audiobook; check RTF in the printout.
- **Cell 15** — find `report.pdf` + `slides.pptx` in `…/submission/`.

## 6. Test the model later (anywhere)
```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
m = AutoModelForSeq2SeqLM.from_pretrained("…/models/tn_normalizer/latest")
tok = AutoTokenizer.from_pretrained("…/models/tn_normalizer/latest")
ids = tok("normalize: He paid $5.2M in 1984.", return_tensors="pt").input_ids
print(tok.decode(m.generate(ids, max_new_tokens=128)[0], skip_special_tokens=True))
```
or simply: `audiobook-ai normalize --text "..."` / `audiobook-ai synthesize --file book.epub`.

## Troubleshooting
- **OOM** → lower `TRAIN_SIZE`, or the profile already enables gradient checkpointing on
  A100-40/L4/T4; you can also set `BASE_MODEL = google-t5/t5-small`.
- **H100 not available** → pick any GPU; the profile downshifts automatically (T4 → fp16 + t5-small).
- **m4b not produced** → ffmpeg missing; cell 3 installs it. mp3/m4b are optional (wav always works).
- **Slow on CPU** → SpeechT5 is ~RTF 1–2 on CPU; use a GPU runtime for real-time synthesis.
