#!/usr/bin/env bash
# Quickstart: install, build the corpus, run the offline demo, and self-grade.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> install (core + extras)"
pip install -e ".[ml,tts,api,report]"

echo "==> build the synthetic TN corpus"
audiobook-ai data --task corpus

echo "==> evaluate the rule baseline (train the model to beat it)"
audiobook-ai evaluate --which test
audiobook-ai evaluate --which hard

echo "==> run the agent on the built-in sample book (offline, placeholder TTS)"
audiobook-ai demo-agent --rule --backend placeholder

echo "==> generate report.pdf + slides.pptx and self-grade"
audiobook-ai generate-report
audiobook-ai generate-slides
audiobook-ai grade
