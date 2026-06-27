"""Command-line interface — the single entrypoint for the Audiobook AI system.

    audiobook-ai <command> [options]

Commands: data, train, tune, evaluate, normalize, synthesize, demo-agent, serve,
benchmark, error-analysis, audio-quality, monitor, generate-report,
generate-slides, autopilot, grade.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from .config import AppConfig, ensure_dirs, load_config
from .logging_utils import get_logger

logger = get_logger(__name__)

TITLE = "Audiobook Generation System"
AUTHOR = "Le Dinh Minh Quan"


def _load(args) -> AppConfig:
    cfg = load_config(args.config) if getattr(args, "config", None) else AppConfig()
    ensure_dirs()
    return cfg


def cmd_data(args):
    from .data.download_dataset import download_all, download_task
    cfg = _load(args)
    res = download_all(cfg) if args.task == "all" else download_task(args.task, cfg)
    print(json.dumps(res, indent=2))


def cmd_train(args):
    from .training.train_normalizer import train_normalizer
    print(json.dumps(train_normalizer(_load(args), limit=args.limit, base_model=args.base_model), indent=2))


def cmd_tune(args):
    from .training.tune import tune_normalizer
    print(json.dumps(tune_normalizer(_load(args), n_trials=args.n_trials, limit=args.limit), indent=2))


def cmd_evaluate(args):
    from .training.evaluate import evaluate
    print(json.dumps(evaluate(_load(args), which=args.which, limit=args.limit).get("summary", {}), indent=2))


def cmd_normalize(args):
    from .agent.narrator_agent import NarratorAgent
    text = args.text or Path(args.file).read_text(encoding="utf-8")
    job = NarratorAgent(_load(args), load_model=not args.rule).normalize_preview(text)
    print(json.dumps({"normalized": " ".join(s.normalized for s in job.segments),
                      "source": job.metrics.get("normalization", {}).get("source"),
                      "segments": [s.to_dict() for s in job.segments]}, indent=2, ensure_ascii=False))


def cmd_synthesize(args):
    from .agent.narrator_agent import NarratorAgent
    agent = NarratorAgent(_load(args), tts_backend=args.backend)
    job = agent.process(path=args.file, text=args.text, title=args.title or TITLE, synth=True)
    print(json.dumps(job.to_dict(), indent=2, ensure_ascii=False))


def cmd_demo_agent(args):
    from .agent.narrator_agent import NarratorAgent
    from .data.samples import write_sample_book
    from .config import data_dir
    cfg = _load(args)
    book = write_sample_book(data_dir() / "samples")
    job = NarratorAgent(cfg, load_model=not args.rule, tts_backend=args.backend).process(
        path=str(book), title="The Clockmaker's Ledger", synth=not args.no_audio)
    sd = job.to_dict()
    print(f"\nstatus      : {sd['status']}")
    print(f"chapters    : {sd['n_chapters']} | spoken segments: {sd['n_spoken_segments']} | flagged: {sd['n_flagged']}")
    print(f"decisions   : {[(d['id'], d['branch']) for d in sd['decisions']]}")
    print(f"metrics     : {json.dumps({k: sd['metrics'].get(k) for k in ('audio_duration','rtf','audio_qa')})}")
    print(f"outputs     : {list(sd['outputs'].keys())}")
    print("\nnormalized (first 6):")
    for s in sd["segments"][:6]:
        print(f"  [{s['kind']:9}|{s['voice']:8}] {s['text'][:46]!r} -> {s['normalized'][:56]!r}")


def cmd_serve(args):
    import os
    import uvicorn
    if args.config:
        os.environ["AUDIOBOOK_AI_INFER_CONFIG"] = str(args.config)
    if args.backend:
        os.environ["AUDIOBOOK_AI_TTS_BACKEND"] = args.backend
    target = "audiobook_ai.api.app_combined:app" if args.ui else "audiobook_ai.api.main:app"
    uvicorn.run(target, host=args.host, port=args.port, reload=False)


def cmd_benchmark(args):
    from .analysis.latency import benchmark
    print(json.dumps(benchmark(_load(args), n=args.n, warmup=args.warmup), indent=2))


def cmd_error_analysis(args):
    from .analysis.error_analysis import error_analysis
    print(json.dumps(error_analysis(_load(args), limit=args.limit), indent=2))


def cmd_audio_quality(args):
    from .analysis.audio_quality import audio_quality_report
    print(json.dumps(audio_quality_report(_load(args)), indent=2))


def cmd_monitor(args):
    from .monitoring.drift_report import monitoring_report
    print(json.dumps(monitoring_report(_load(args), log_path=args.log), indent=2))


def cmd_generate_report(args):
    from .autoreport.report_pdf import generate_report
    print("Report ->", generate_report(_load(args), title=args.title, author=args.author))


def cmd_generate_slides(args):
    from .autoreport.slides_pptx import generate_slides
    print("Slides ->", generate_slides(_load(args), title=args.title, author=args.author))


def cmd_autopilot(args):
    from .automation.autopilot import run_autopilot
    print(json.dumps(run_autopilot(_load(args), title=args.title, author=args.author,
                                   train=not args.no_train, limit=args.limit), indent=2))


def cmd_grade(args):
    from .grading.checklist import build_checklist
    repo = Path(args.repo) if args.repo else Path(__file__).resolve().parents[2]
    print(json.dumps(build_checklist(repo), indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="audiobook-ai", description=TITLE)
    p.add_argument("--config", help="Path to a YAML config")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("data", help="build/cache the TN corpus (+ optional prefetch)")
    sp.add_argument("--task", choices=["all", "corpus", "speakers"], default="corpus"); sp.set_defaults(func=cmd_data)
    sp = sub.add_parser("train", help="fine-tune the ByT5/T5 normalizer")
    sp.add_argument("--limit", type=int, default=None); sp.add_argument("--base-model", default=None); sp.set_defaults(func=cmd_train)
    sp = sub.add_parser("tune", help="basic LR hyperparameter search")
    sp.add_argument("--n-trials", type=int, default=3); sp.add_argument("--limit", type=int, default=4000); sp.set_defaults(func=cmd_tune)
    sp = sub.add_parser("evaluate", help="neural vs baseline, overall + per-class")
    sp.add_argument("--which", choices=["test", "hard", "val"], default="test"); sp.add_argument("--limit", type=int, default=None); sp.set_defaults(func=cmd_evaluate)
    sp = sub.add_parser("normalize", help="normalize text (spoken-form preview)")
    sp.add_argument("--text"); sp.add_argument("--file"); sp.add_argument("--rule", action="store_true"); sp.set_defaults(func=cmd_normalize)
    sp = sub.add_parser("synthesize", help="make an audiobook from a file/text")
    sp.add_argument("--file"); sp.add_argument("--text"); sp.add_argument("--title", default=None); sp.add_argument("--backend", default=None); sp.set_defaults(func=cmd_synthesize)
    sp = sub.add_parser("demo-agent", help="run the agent on the built-in sample book")
    sp.add_argument("--rule", action="store_true"); sp.add_argument("--backend", default=None); sp.add_argument("--no-audio", action="store_true"); sp.set_defaults(func=cmd_demo_agent)
    sp = sub.add_parser("serve", help="start the FastAPI server")
    sp.add_argument("--host", default="0.0.0.0"); sp.add_argument("--port", type=int, default=8000); sp.add_argument("--ui", action="store_true"); sp.add_argument("--backend", default=None); sp.set_defaults(func=cmd_serve)
    sp = sub.add_parser("benchmark", help="latency benchmark (normalize + synth RTF)")
    sp.add_argument("--n", type=int, default=30); sp.add_argument("--warmup", type=int, default=3); sp.set_defaults(func=cmd_benchmark)
    sp = sub.add_parser("error-analysis", help="per-class error analysis vs baseline")
    sp.add_argument("--limit", type=int, default=None); sp.set_defaults(func=cmd_error_analysis)
    sp = sub.add_parser("audio-quality", help="audio QA report over the sample book")
    sp.set_defaults(func=cmd_audio_quality)
    sp = sub.add_parser("monitor", help="monitoring report from job logs")
    sp.add_argument("--log", default=None); sp.set_defaults(func=cmd_monitor)
    sp = sub.add_parser("generate-report", help="generate the PDF report")
    sp.add_argument("--title", default=TITLE); sp.add_argument("--author", default=AUTHOR); sp.set_defaults(func=cmd_generate_report)
    sp = sub.add_parser("generate-slides", help="generate the PPTX slides")
    sp.add_argument("--title", default=TITLE); sp.add_argument("--author", default=AUTHOR); sp.set_defaults(func=cmd_generate_slides)
    sp = sub.add_parser("autopilot", help="one-button: train -> eval -> analysis -> report+slides")
    sp.add_argument("--title", default=TITLE); sp.add_argument("--author", default=AUTHOR); sp.add_argument("--no-train", action="store_true"); sp.add_argument("--limit", type=int, default=None); sp.set_defaults(func=cmd_autopilot)
    sp = sub.add_parser("grade", help="rubric completeness self-check")
    sp.add_argument("--repo", default=None); sp.set_defaults(func=cmd_grade)
    return p


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
