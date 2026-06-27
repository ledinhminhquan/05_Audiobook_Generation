"""Monitoring & drift report from production job logs (JSONL).

Aggregates job outcomes (status mix, flag-rate, audio-QA fail-rate, RTF) and
computes a simple drift signal by comparing a recent window against an earlier
baseline window — the operational early-warning for the system.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional

from ..config import AppConfig, run_dir
from ..logging_utils import get_logger, utc_stamp

logger = get_logger(__name__)


def _read_logs(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _window_stats(rows: List[Dict]) -> Dict:
    if not rows:
        return {"n": 0}
    statuses: Dict[str, int] = {}
    flags, segs, rtfs, qa_fail = [], [], [], []
    for r in rows:
        statuses[r.get("status", "?")] = statuses.get(r.get("status", "?"), 0) + 1
        m = r.get("metrics", {}) or {}
        n_seg = r.get("n_segments", 0) or 0
        segs.append(n_seg)
        flags.append((r.get("n_flagged", 0) or 0) / max(1, n_seg))
        if isinstance(m.get("rtf"), (int, float)):
            rtfs.append(m["rtf"])
        qa = m.get("audio_qa", {}) or {}
        tot = (qa.get("pass", 0) + qa.get("fail", 0)) or 0
        if tot:
            qa_fail.append(qa.get("fail", 0) / tot)
    return {"n": len(rows), "statuses": statuses,
            "flag_rate": round(mean(flags), 4) if flags else 0.0,
            "qa_fail_rate": round(mean(qa_fail), 4) if qa_fail else 0.0,
            "mean_rtf": round(mean(rtfs), 4) if rtfs else None,
            "mean_segments": round(mean(segs), 1) if segs else 0.0}


def monitoring_report(cfg: AppConfig, log_path: Optional[str] = None, save: bool = True) -> Dict:
    path = Path(log_path) if log_path else cfg.serving.job_log_path
    rows = _read_logs(path)
    overall = _window_stats(rows)

    drift = {}
    if len(rows) >= 6:
        half = len(rows) // 2
        base, recent = _window_stats(rows[:half]), _window_stats(rows[half:])
        def delta(k):
            a, b = base.get(k), recent.get(k)
            return round(b - a, 4) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else None
        drift = {"baseline_window": base, "recent_window": recent,
                 "delta_flag_rate": delta("flag_rate"),
                 "delta_qa_fail_rate": delta("qa_fail_rate"),
                 "delta_mean_rtf": delta("mean_rtf"),
                 "alert": bool((delta("flag_rate") or 0) > 0.1 or (delta("qa_fail_rate") or 0) > 0.1)}

    result = {"log_path": str(path), "n_jobs": len(rows), "overall": overall, "drift": drift,
              "note": "no job logs found yet" if not rows else ""}
    if save:
        d = run_dir() / "monitoring"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"monitor-{utc_stamp()}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        (d / "latest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


__all__ = ["monitoring_report"]
