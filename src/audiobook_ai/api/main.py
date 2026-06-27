"""FastAPI service for the Audiobook Generation system.

Endpoints
---------
* ``GET  /healthz`` / ``GET /readyz`` – liveness / readiness
* ``POST /normalize``                 – text -> spoken-form preview (fast, no audio)
* ``POST /synthesize``                – JSON {text,title,...} -> audiobook job
* ``POST /synthesize/file``           – upload epub/pdf/txt/md -> audiobook job
* ``GET  /artifacts/...``             – download generated audio/srt/manifest
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..config import output_dir
from ..logging_utils import get_logger
from .dependencies import get_agent, get_config
from .schemas import (HealthResponse, JobResponse, NormalizeRequest, NormalizeResponse,
                      SegmentView, SynthesizeRequest)

logger = get_logger(__name__)
cfg = get_config()
app = FastAPI(title=cfg.serving.api_title, version=cfg.serving.api_version)

_ART = output_dir()
_ART.mkdir(parents=True, exist_ok=True)
app.mount("/artifacts", StaticFiles(directory=str(_ART)), name="artifacts")


def _job_response(job) -> JobResponse:
    sd = job.to_dict()
    return JobResponse(status=sd["status"], title=sd["title"], source_format=sd["source_format"],
                       n_chapters=sd["n_chapters"], n_segments=sd["n_segments"], n_flagged=sd["n_flagged"],
                       parse_score=sd["parse_score"], metrics=sd["metrics"], outputs=sd["outputs"],
                       decisions=sd["decisions"], model_versions=sd["model_versions"])


@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    agent = get_agent()
    return HealthResponse(status="ok", normalizer=getattr(agent.normalizer, "name", "rule"),
                          version=__version__)


@app.get("/readyz")
def readyz() -> dict:
    get_agent()
    return {"status": "ready"}


@app.post("/normalize", response_model=NormalizeResponse)
def normalize(req: NormalizeRequest) -> NormalizeResponse:
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="empty text")
    job = get_agent().normalize_preview(req.text)
    segs = [SegmentView(text=s.text, normalized=s.normalized, kind=s.kind, voice=s.voice,
                        confidence=s.confidence, source=s.source, flagged=s.flagged)
            for s in job.segments]
    full = " ".join(s.normalized for s in job.segments)
    src = job.metrics.get("normalization", {}).get("source", "rule")
    return NormalizeResponse(normalized=full, n_segments=len(segs), source=src, segments=segs)


@app.post("/synthesize", response_model=JobResponse)
def synthesize(req: SynthesizeRequest) -> JobResponse:
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="empty text")
    agent = get_agent()
    if req.backend:
        agent._tts_backend = req.backend
        agent._tts = None
    job = agent.process(text=req.text, title=req.title, author=req.author, synth=True)
    return _job_response(job)


def _multipart_available() -> bool:
    for mod in ("multipart", "python_multipart"):
        try:
            __import__(mod)
            return True
        except Exception:
            continue
    return False


# The file-upload route needs python-multipart; register it only when available so
# the module imports cleanly in minimal environments (the JSON /synthesize still works).
if _multipart_available():
    @app.post("/synthesize/file", response_model=JobResponse)
    async def synthesize_file(file: UploadFile = File(...), title: str = Form("Audiobook"),
                              author: str = Form("Unknown")) -> JobResponse:
        suffix = Path(file.filename or "doc.txt").suffix or ".txt"
        data = await file.read()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(data)
        tmp.close()
        try:
            job = get_agent().process(path=tmp.name, title=title or Path(file.filename or "book").stem,
                                      author=author)
        finally:
            try:
                Path(tmp.name).unlink()
            except OSError:
                pass
        return _job_response(job)
else:
    logger.warning("python-multipart not installed; POST /synthesize/file is disabled")


@app.get("/download")
def download(path: str) -> FileResponse:
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(str(p), filename=p.name)


__all__ = ["app"]
