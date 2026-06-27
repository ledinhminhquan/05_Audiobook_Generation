"""Generate the submission report.pdf from the docs + live artifacts (reportlab).

Assembles a 10-15 page report: title page, the core Section-I documents rendered
from ``docs/*.md``, a live results section (TN model-vs-baseline table + charts),
and conclusions. Degrades to a Markdown file if reportlab is unavailable.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import AppConfig, artifacts_dir
from ..logging_utils import get_logger, utc_now_iso
from . import charts as charts_mod
from .artifact_loader import load_artifacts, read_doc

logger = get_logger(__name__)

_SECTIONS = [
    ("1. Problem Definition", "problem_definition.md"),
    ("2. Data Description", "data_description.md"),
    ("3. Model Selection & Optimization", "model_selection.md"),
    ("4. Text-Normalization Quality Evaluation", "tn_quality_evaluation.md"),
    ("5. Agent Architecture", "agent_architecture.md"),
    ("6. Deployment", "deployment.md"),
    ("7. Continual Learning & Monitoring", "continual_learning_monitoring.md"),
    ("8. Data Privacy & Model Robustness", "privacy_robustness.md"),
    ("9. Project Plan & Teamwork", "project_plan.md"),
    ("10. Ethics & Responsible AI", "ethics_statement.md"),
]


def _md_to_flowables(md: str, styles, max_lines: int = 240):
    from reportlab.platypus import Paragraph, Spacer, Preformatted
    flow = []
    lines = md.splitlines()[:max_lines]
    in_code = False
    code: List[str] = []
    bullet: List[str] = []

    def flush_bullet():
        nonlocal bullet
        for b in bullet:
            flow.append(Paragraph("• " + _esc(b), styles["Body"]))
        bullet = []

    for ln in lines:
        if ln.strip().startswith("```"):
            if in_code:
                flow.append(Preformatted("\n".join(code), styles["Code"]))
                code = []
            in_code = not in_code
            continue
        if in_code:
            code.append(ln)
            continue
        s = ln.rstrip()
        if not s:
            flush_bullet()
            flow.append(Spacer(1, 5))
            continue
        if s.startswith("#"):
            flush_bullet()
            level = len(s) - len(s.lstrip("#"))
            txt = _esc(s.lstrip("#").strip())
            flow.append(Paragraph(txt, styles["H2" if level <= 2 else "H3"]))
        elif s.lstrip().startswith(("- ", "* ")):
            bullet.append(s.lstrip()[2:])
        elif s.lstrip().startswith("|") and "|" in s[1:]:
            flush_bullet()
            if not re.match(r"^\s*\|[\s:|-]+\|\s*$", s):  # skip separator rows
                cells = [c.strip() for c in s.strip().strip("|").split("|")]
                flow.append(Paragraph(_esc(" — ".join(cells)), styles["Body"]))
        else:
            flush_bullet()
            flow.append(Paragraph(_esc(s), styles["Body"]))
    flush_bullet()
    return flow


def _esc(s: str) -> str:
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"`(.+?)`", r"<font face='Courier'>\1</font>", s)
    s = s.replace("&", "&amp;").replace("<b>", "\x00b\x00").replace("</b>", "\x00/b\x00")
    s = s.replace("<font face='Courier'>", "\x00f\x00").replace("</font>", "\x00/f\x00")
    s = s.replace("<", "&lt;").replace(">", "&gt;")
    s = (s.replace("\x00b\x00", "<b>").replace("\x00/b\x00", "</b>")
          .replace("\x00f\x00", "<font face='Courier'>").replace("\x00/f\x00", "</font>"))
    return s


def _results_table(arts: Dict[str, Any], styles):
    from reportlab.lib import colors
    from reportlab.platypus import Paragraph, Spacer, Table, TableStyle
    flow = [Paragraph("Results — Text Normalization (sentence exact-match accuracy)", styles["H3"])]
    ev = arts.get("eval")
    rows = [["Slice", "Rule baseline", "ByT5 (trained)", "Δ"]]
    if ev:
        b = ev.get("baseline", {})
        n = ev.get("neural", {})
        for slice_name in ("test", "hard"):
            ba = b.get(slice_name, {}).get("sentence_accuracy")
            na = n.get(slice_name, {}).get("sentence_accuracy") if n else None
            rows.append([slice_name.upper(),
                         f"{ba:.3f}" if ba is not None else "—",
                         f"{na:.3f}" if na is not None else "(not trained yet)",
                         f"{na - ba:+.3f}" if (na is not None and ba is not None) else "—"])
    else:
        rows.append(["—", "run `evaluate`", "—", "—"])
    t = Table(rows, hAlign="LEFT", colWidths=[90, 110, 130, 60])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2b6cb0")),
                           ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                           ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                           ("FONTSIZE", (0, 0), (-1, -1), 9),
                           ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef3f8")])]))
    flow += [t, Spacer(1, 8)]
    return flow


def generate_report(cfg: AppConfig, title: Optional[str] = None, author: Optional[str] = None,
                    out_path: Optional[str] = None) -> str:
    title = title or cfg.project_title
    author = author or cfg.author
    arts = load_artifacts(cfg)
    out_path = Path(out_path) if out_path else artifacts_dir() / "report.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import cm
        from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer)
    except Exception as exc:
        logger.warning("reportlab unavailable (%s); writing markdown report", exc)
        md = f"# {title}\n\n{author} ({cfg.student_id})\n\n"
        for hd, fn in _SECTIONS:
            md += f"\n\n# {hd}\n\n" + read_doc(fn)
        alt = out_path.with_suffix(".md")
        alt.write_text(md, encoding="utf-8")
        return str(alt)

    base = getSampleStyleSheet()
    styles = {
        "Title": ParagraphStyle("T", parent=base["Title"], fontSize=24, leading=28),
        "H2": ParagraphStyle("H2", parent=base["Heading2"], textColor="#1a365d", spaceBefore=10),
        "H3": ParagraphStyle("H3", parent=base["Heading3"], textColor="#2b6cb0"),
        "Body": ParagraphStyle("B", parent=base["BodyText"], fontSize=9.5, leading=13),
        "Code": ParagraphStyle("C", parent=base["Code"], fontSize=7.5, leading=9, backColor="#f4f6f8"),
        "Meta": ParagraphStyle("M", parent=base["BodyText"], fontSize=11, leading=15),
    }
    chart_dir = out_path.parent / "charts"
    built = dict(charts_mod.build_all(arts, chart_dir))

    story: List[Any] = []
    story += [Spacer(1, 5 * cm), Paragraph(title, styles["Title"]), Spacer(1, 1 * cm),
              Paragraph(f"<b>{author}</b> — Student {cfg.student_id}", styles["Meta"]),
              Paragraph("NLP in Industry — Final Assignment", styles["Meta"]),
              Paragraph("End-to-end production NLP system: trainable text normalization + "
                        "neural TTS + an agentic document→audiobook pipeline.", styles["Meta"]),
              Paragraph(f"Generated {utc_now_iso()}", styles["Body"]), PageBreak()]

    mv = arts.get("model_meta", {}) or {}
    if mv:
        story.append(Paragraph(f"Trained model: <b>{mv.get('version','?')}</b> "
                               f"(base {mv.get('base_model','?')})", styles["Body"]))
    story += _results_table(arts, styles)
    for name in ("baseline_vs_neural", "per_class", "qa"):
        if name in built:
            story += [Image(str(built[name]), width=15 * cm, height=8 * cm), Spacer(1, 6)]
    story.append(PageBreak())

    for heading, fname in _SECTIONS:
        story.append(Paragraph(heading, styles["H2"]))
        story += _md_to_flowables(read_doc(fname), styles)
        story.append(PageBreak())

    SimpleDocTemplate(str(out_path), pagesize=A4, topMargin=1.6 * cm, bottomMargin=1.6 * cm,
                      leftMargin=1.8 * cm, rightMargin=1.8 * cm,
                      title=title, author=author).build(story)
    logger.info("Report -> %s", out_path)
    return str(out_path)


__all__ = ["generate_report"]
