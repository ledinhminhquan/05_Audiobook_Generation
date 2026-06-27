"""Document parsing, chaptering, segmentation and classification."""

from __future__ import annotations

from audiobook_ai.config import ParseConfig
from audiobook_ai.data.document import (chunk_sentences, document_from_text, parse_document,
                                        split_sentences)


def test_parse_sample_book(sample_book):
    doc = parse_document(sample_book, ParseConfig())
    assert doc.n_chapters >= 2
    kinds = {s.kind for s in doc.iter_segments()}
    assert "heading" in kinds
    assert any(s.kind == "narration" for s in doc.iter_segments())
    assert doc.total_chars() > 100


def test_parse_markdown(sample_md):
    doc = parse_document(sample_md, ParseConfig())
    assert any(s.kind == "heading" for s in doc.iter_segments())


def test_chunking_respects_max_chars():
    cfg = ParseConfig(max_chars_per_segment=60, min_chars_per_segment=10)
    text = " ".join(f"Sentence number {i} here." for i in range(20))
    chunks = chunk_sentences(split_sentences(text), cfg.max_chars_per_segment, cfg.min_chars_per_segment)
    assert chunks
    assert all(len(c) <= 60 for c in chunks)


def test_document_from_text():
    doc = document_from_text("It cost $5 in 1999.\n\nThe end.", ParseConfig())
    segs = doc.spoken_segments()
    assert segs
    assert sum(len(s.text) for s in segs) > 0
