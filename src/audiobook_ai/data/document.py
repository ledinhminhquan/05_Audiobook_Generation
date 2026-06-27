"""Document parsing & structuring: epub / pdf / txt / md -> Document.

A ``Document`` is a list of ``Chapter`` s, each a list of typed ``Segment`` s
(``narration | dialogue | heading | skippable``) already chunked to TTS-friendly
sizes (never splitting mid-sentence, tiny fragments merged). Parsers for EPUB
(ebooklib+bs4), PDF (PyMuPDF, pdfplumber fallback) and txt/markdown are imported
lazily and degrade to a plain-text + regex path when libraries are missing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from ..config import ParseConfig
from ..logging_utils import get_logger

logger = get_logger(__name__)

SegmentKind = str  # "narration" | "dialogue" | "heading" | "skippable"


@dataclass
class Segment:
    text: str
    kind: SegmentKind = "narration"
    chapter_index: int = 0
    order: int = 0

    def to_dict(self) -> Dict:
        return {"text": self.text, "kind": self.kind,
                "chapter_index": self.chapter_index, "order": self.order}


@dataclass
class Chapter:
    index: int
    title: str
    segments: List[Segment] = field(default_factory=list)


@dataclass
class Document:
    title: str = "Untitled"
    author: str = "Unknown"
    language: str = "en"
    source_format: str = "txt"
    chapters: List[Chapter] = field(default_factory=list)

    def iter_segments(self) -> Iterator[Segment]:
        for ch in self.chapters:
            for seg in ch.segments:
                yield seg

    def spoken_segments(self) -> List[Segment]:
        return [s for s in self.iter_segments() if s.kind != "skippable"]

    @property
    def n_chapters(self) -> int:
        return len(self.chapters)

    def n_segments(self) -> int:
        return sum(len(c.segments) for c in self.chapters)

    def total_chars(self) -> int:
        return sum(len(s.text) for s in self.iter_segments())

    def to_dict(self) -> Dict:
        return {"title": self.title, "author": self.author, "language": self.language,
                "source_format": self.source_format,
                "chapters": [{"index": c.index, "title": c.title,
                              "segments": [s.to_dict() for s in c.segments]} for c in self.chapters]}


# ─────────────────────────────────────────────────────────────────────────────
# Sentence segmentation + TTS chunking
# ─────────────────────────────────────────────────────────────────────────────
# split only at a sentence-ender followed by whitespace + a capital/quote/paren
# (so it never splits inside "$5.2M" or "1984." mid-token)
_SENT_BOUNDARY = re.compile(r'(?<=[.!?])["\')\]]?\s+(?=[A-Z"\'(\[])')
_ABBR_END = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|Prof|St|Jr|Sr|vs|etc|Inc|Ltd|Corp|Dept|No|Vol|Ch|Fig|Ave|Blvd|Rd|Mt|e\.g|i\.e)\.$",
    re.I)


def _regex_sentences(text: str) -> List[str]:
    parts = _SENT_BOUNDARY.split(text.strip())
    merged: List[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if merged and _ABBR_END.search(merged[-1]):   # split landed after an abbreviation -> rejoin
            merged[-1] = merged[-1] + " " + p
        else:
            merged.append(p)
    return merged


def split_sentences(text: str, segmenter: str = "auto") -> List[str]:
    text = text.strip()
    if not text:
        return []
    if segmenter in ("auto", "pysbd"):
        try:
            import pysbd  # lazy
            seg = pysbd.Segmenter(language="en", clean=False)
            return [s.strip() for s in seg.segment(text) if s.strip()]
        except Exception:
            if segmenter == "pysbd":
                logger.info("pysbd unavailable; falling back to regex segmenter")
    if segmenter == "nltk":
        try:
            import nltk  # lazy
            return [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]
        except Exception:
            pass
    return _regex_sentences(text)


def chunk_sentences(sentences: List[str], max_chars: int, min_chars: int) -> List[str]:
    """Merge sentences into chunks <= max_chars, never splitting a sentence; merge tiny tails."""
    chunks: List[str] = []
    cur = ""
    for s in sentences:
        if len(s) > max_chars:                       # hard-split an over-long sentence on clauses
            if cur:
                chunks.append(cur.strip())
                cur = ""
            chunks.extend(_split_long(s, max_chars))
            continue
        if not cur:
            cur = s
        elif len(cur) + 1 + len(s) <= max_chars:
            cur += " " + s
        else:
            chunks.append(cur.strip())
            cur = s
    if cur:
        chunks.append(cur.strip())
    # merge fragments shorter than min_chars into the previous chunk
    merged: List[str] = []
    for c in chunks:
        if merged and len(c) < min_chars:
            merged[-1] = (merged[-1] + " " + c).strip()
        else:
            merged.append(c)
    return merged


def _split_long(sentence: str, max_chars: int) -> List[str]:
    parts = re.split(r"(?<=[,;:])\s+", sentence)
    out: List[str] = []
    cur = ""
    for p in parts:
        if not cur:
            cur = p
        elif len(cur) + 1 + len(p) <= max_chars:
            cur += " " + p
        else:
            out.append(cur.strip())
            cur = p
    if cur:
        out.append(cur.strip())
    # last resort: hard slice anything still too long
    final: List[str] = []
    for c in out:
        while len(c) > max_chars:
            final.append(c[:max_chars])
            c = c[max_chars:]
        if c:
            final.append(c)
    return final


# ─────────────────────────────────────────────────────────────────────────────
# Segment classification
# ─────────────────────────────────────────────────────────────────────────────
def classify_segment(text: str, cfg: ParseConfig, is_heading: bool = False) -> SegmentKind:
    t = text.strip()
    if not t:
        return "skippable"
    for pat in cfg.drop_patterns:
        if re.match(pat, t, re.I):
            return "skippable"
    if is_heading:
        return "heading"
    # a short, terminal-punctuation-free line is likely a heading/title
    if len(t) <= 64 and not re.search(r"[.!?]\"?$", t) and "\n" not in t:
        if re.match(cfg.chapter_regex, t, re.I) or t.isupper() or t.istitle():
            return "heading"
    quote_chars = sum(t.count(q) for q in "\"“”")
    if t[0] in "\"“" and quote_chars >= 2:
        return "dialogue"
    return "narration"


# ─────────────────────────────────────────────────────────────────────────────
# Chaptering from a flat block list
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class _Block:
    text: str
    is_heading: bool = False


def _blocks_to_chapters(blocks: List[_Block], cfg: ParseConfig) -> List[Chapter]:
    chapters: List[Chapter] = []
    cur = Chapter(index=0, title="Chapter 1")
    order = 0
    started = False
    for b in blocks:
        text = b.text.strip()
        if not text:
            continue
        is_heading = b.is_heading or bool(re.match(cfg.chapter_regex, text, re.I))
        is_chapter_start = is_heading and (b.is_heading or re.match(cfg.chapter_regex, text, re.I))
        if is_chapter_start and started and cur.segments:
            chapters.append(cur)
            cur = Chapter(index=len(chapters), title=text[:80])
            order = 0
        started = True
        kind = classify_segment(text, cfg, is_heading=is_heading)
        if kind == "heading" and not cur.segments and cur.title.startswith("Chapter "):
            cur.title = text[:80]
        if kind == "skippable":
            continue
        if kind == "heading":
            cur.segments.append(Segment(text=text, kind="heading", chapter_index=cur.index, order=order))
            order += 1
            continue
        for chunk in chunk_sentences(split_sentences(text, cfg.segmenter),
                                     cfg.max_chars_per_segment, cfg.min_chars_per_segment):
            ck = classify_segment(chunk, cfg)
            cur.segments.append(Segment(text=chunk, kind=ck, chapter_index=cur.index, order=order))
            order += 1
    if cur.segments:
        chapters.append(cur)
    return chapters or [Chapter(index=0, title="Chapter 1", segments=[])]


# ─────────────────────────────────────────────────────────────────────────────
# Format parsers
# ─────────────────────────────────────────────────────────────────────────────
def _read_text_file(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except Exception:
            continue
    return path.read_bytes().decode("utf-8", errors="ignore")


def parse_txt(path: Path, cfg: ParseConfig) -> List[_Block]:
    raw = _read_text_file(path)
    return [_Block(text=p) for p in re.split(r"\n\s*\n", raw) if p.strip()]


def parse_markdown(path: Path, cfg: ParseConfig) -> List[_Block]:
    raw = _read_text_file(path)
    blocks: List[_Block] = []
    for para in re.split(r"\n\s*\n", raw):
        para = para.strip()
        if not para:
            continue
        m = re.match(r"^(#{1,6})\s+(.*)$", para)
        if m:
            blocks.append(_Block(text=m.group(2).strip(), is_heading=True))
        else:
            blocks.append(_Block(text=re.sub(r"[*_`>#]", "", para)))
    return blocks


def parse_epub(path: Path, cfg: ParseConfig) -> List[_Block]:
    import ebooklib  # lazy
    from ebooklib import epub
    from bs4 import BeautifulSoup
    book = epub.read_epub(str(path))
    blocks: List[_Block] = []
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), "html.parser")
        for el in soup.find_all(["h1", "h2", "h3", "h4", "p", "blockquote"]):
            text = el.get_text(" ", strip=True)
            if text:
                blocks.append(_Block(text=text, is_heading=el.name in ("h1", "h2", "h3", "h4")))
    return blocks


def parse_pdf(path: Path, cfg: ParseConfig) -> List[_Block]:
    try:
        import fitz  # PyMuPDF, lazy
        doc = fitz.open(str(path))
        sizes: List[float] = []
        page_blocks = []
        for page in doc:
            d = page.get_text("dict")
            for blk in d.get("blocks", []):
                for line in blk.get("lines", []):
                    spans = line.get("spans", [])
                    text = "".join(s.get("text", "") for s in spans).strip()
                    if len(text) >= cfg.pdf_min_line_chars:
                        size = max((s.get("size", 0) for s in spans), default=0)
                        sizes.append(size)
                        page_blocks.append((text, size))
        body = sorted(sizes)[len(sizes) // 2] if sizes else 0
        blocks: List[_Block] = []
        for text, size in page_blocks:
            blocks.append(_Block(text=text, is_heading=size >= body * 1.25 and len(text) <= 80))
        return blocks
    except Exception as exc:
        logger.info("PyMuPDF path failed (%s); trying pdfplumber", exc)
    import pdfplumber  # lazy fallback
    blocks = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            txt = page.extract_text() or ""
            for para in re.split(r"\n\s*\n", txt):
                if para.strip():
                    blocks.append(_Block(text=para.strip()))
    return blocks


_PARSERS = {".txt": parse_txt, ".text": parse_txt, ".md": parse_markdown,
            ".markdown": parse_markdown, ".epub": parse_epub, ".pdf": parse_pdf}


def parse_document(path: str, cfg: Optional[ParseConfig] = None,
                   title: Optional[str] = None, author: Optional[str] = None) -> Document:
    """Parse a document file into a structured ``Document``."""
    cfg = cfg or ParseConfig()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    ext = p.suffix.lower()
    parser = _PARSERS.get(ext)
    fmt = ext.lstrip(".") or "txt"
    if parser is None:
        logger.info("Unknown extension %s; reading as plain text", ext)
        parser, fmt = parse_txt, "txt"
    try:
        blocks = parser(p, cfg)
    except Exception as exc:
        logger.warning("Parser for %s failed (%s); falling back to plain text", ext, exc)
        blocks, fmt = parse_txt(p, cfg), "txt"
    chapters = _blocks_to_chapters(blocks, cfg)
    return Document(title=title or p.stem, author=author or "Unknown",
                    source_format=fmt, chapters=chapters)


def document_from_text(text: str, cfg: Optional[ParseConfig] = None,
                       title: str = "Pasted Text") -> Document:
    """Build a Document directly from a raw text string (used by the API /synthesize)."""
    cfg = cfg or ParseConfig()
    blocks = [_Block(text=p) for p in re.split(r"\n\s*\n", text) if p.strip()] or [_Block(text=text)]
    return Document(title=title, source_format="text", chapters=_blocks_to_chapters(blocks, cfg))


__all__ = ["Segment", "Chapter", "Document", "parse_document", "document_from_text",
           "split_sentences", "chunk_sentences", "classify_segment"]
