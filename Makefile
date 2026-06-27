.PHONY: help install install-all data train evaluate demo serve ui test grade report slides autopilot clean

help:
	@echo "Audiobook Generation System — common tasks"
	@echo "  make install       core install (pip install -e .)"
	@echo "  make install-all   full install (.[all])"
	@echo "  make data          build/cache the synthetic TN corpus"
	@echo "  make train         fine-tune the ByT5 normalizer"
	@echo "  make evaluate      neural vs baseline (overall + per-class)"
	@echo "  make demo          run the agent on the built-in sample book"
	@echo "  make serve         start the FastAPI server (REST)"
	@echo "  make ui            start FastAPI + Gradio UI at /ui"
	@echo "  make test          run the CPU-only test suite"
	@echo "  make report slides autopilot grade"

install:
	pip install -e .

install-all:
	pip install -e ".[all]"

data:
	audiobook-ai data --task corpus

train:
	audiobook-ai --config configs/train.yaml train

evaluate:
	audiobook-ai evaluate --which test
	audiobook-ai evaluate --which hard

demo:
	audiobook-ai demo-agent --rule --backend placeholder

serve:
	audiobook-ai --config configs/infer.yaml serve --host 0.0.0.0 --port 8000

ui:
	audiobook-ai serve --ui --host 0.0.0.0 --port 7860

test:
	pytest -q

grade:
	audiobook-ai grade

report:
	audiobook-ai generate-report

slides:
	audiobook-ai generate-slides

autopilot:
	audiobook-ai autopilot --no-train

clean:
	rm -rf artifacts __pycache__ .pytest_cache src/*.egg-info build dist
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
