"""Lightweight model registry / versioning for the TN normalizer.

Records training provenance (base model, dataset signature, metrics, git-ish
stamp) next to each checkpoint and maintains a ``latest`` pointer so inference
and the agent always resolve a single, reproducible model version.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from ..logging_utils import get_logger, utc_now_iso, utc_stamp

logger = get_logger(__name__)


def make_version(base_model: str) -> str:
    short = base_model.split("/")[-1]
    return f"{short}-{utc_stamp()}"


def write_metadata(model_path: str | os.PathLike, *, version: str, base_model: str,
                   dataset_signature: Dict[str, Any], metrics: Optional[Dict[str, Any]] = None,
                   extra: Optional[Dict[str, Any]] = None) -> Path:
    p = Path(model_path)
    p.mkdir(parents=True, exist_ok=True)
    meta = {
        "version": version,
        "base_model": base_model,
        "created_at": utc_now_iso(),
        "dataset_signature": dataset_signature,
        "metrics": metrics or {},
    }
    if extra:
        meta.update(extra)
    out = p / "tn_meta.json"
    out.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def read_metadata(model_path: str | os.PathLike) -> Dict[str, Any]:
    p = Path(model_path) / "tn_meta.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def update_latest_pointer(output_dir: str | os.PathLike, model_path: str | os.PathLike) -> Path:
    """Point ``output_dir/latest`` at the freshly trained checkpoint.

    Copies metadata + a ``LATEST`` marker; the heavy weights stay in place and
    ``latest`` is created as a directory symlink when the OS allows, else a marker
    file pointing at the real path (Windows-safe).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    latest = out / "latest"
    target = Path(model_path).resolve()
    try:
        if latest.exists() or latest.is_symlink():
            if latest.is_symlink() or latest.is_file():
                latest.unlink()
            else:
                shutil.rmtree(latest)
        os.symlink(target, latest, target_is_directory=True)
        logger.info("latest -> %s (symlink)", target)
    except Exception:
        latest.mkdir(parents=True, exist_ok=True)
        (latest / "LATEST").write_text(str(target), encoding="utf-8")
        meta = target / "tn_meta.json"
        if meta.exists():
            shutil.copy2(meta, latest / "tn_meta.json")
        logger.info("latest -> %s (marker, symlink unavailable)", target)
    return latest


def resolve_latest(output_dir: str | os.PathLike) -> Optional[Path]:
    latest = Path(output_dir) / "latest"
    if not latest.exists():
        return None
    marker = latest / "LATEST"
    if marker.exists():
        p = Path(marker.read_text(encoding="utf-8").strip())
        return p if p.exists() else None
    return latest


__all__ = ["make_version", "write_metadata", "read_metadata",
           "update_latest_pointer", "resolve_latest"]
