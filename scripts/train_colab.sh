#!/usr/bin/env bash
# Train the ByT5 Text-Normalization model (Colab/GPU). The notebook does this for
# you with an auto GPU profile; this script is the plain CLI equivalent.
set -euo pipefail
cd "$(dirname "$0")/.."

export AUDIOBOOK_AI_ARTIFACTS_DIR="${AUDIOBOOK_AI_ARTIFACTS_DIR:-/content/drive/MyDrive/audiobook_ai}"

audiobook-ai data --task corpus
audiobook-ai --config configs/train.yaml train
audiobook-ai evaluate --which test
audiobook-ai evaluate --which hard
audiobook-ai error-analysis
audiobook-ai autopilot --no-train
