"""Synthetic corpus generation, leakage-free splits, and baseline evaluation."""

from __future__ import annotations

import re

from audiobook_ai.data.tn_corpus import TNCorpusGenerator
from audiobook_ai.data.dataset import build_corpus
from audiobook_ai.models import baseline_rules as br


def _ws(s):
    return re.sub(r"\s+", " ", s).strip()


def test_generator_covers_classes_and_hard_slice():
    gen = TNCorpusGenerator(seed=7)
    ex = gen.generate(1500)
    classes = {c for e in ex for c in e.classes}
    for required in ("CARDINAL", "MONEY", "MEASURE", "DATE", "ORDINAL", "ABBREV", "ROMAN", "TIME"):
        assert required in classes, required
    assert sum(e.hard for e in ex) > 50          # some ambiguous examples


def test_generator_is_deterministic():
    a = TNCorpusGenerator(seed=1).generate(200)
    b = TNCorpusGenerator(seed=1).generate(200)
    assert [e.before for e in a] == [e.before for e in b]


def test_baseline_fails_hard_slice(cfg):
    splits = build_corpus(cfg, save=False)
    hard = splits["hard"]
    assert hard
    acc = sum(_ws(br.normalize(e["before"])) == _ws(e["after"]) for e in hard) / len(hard)
    assert acc < 0.2                              # the trained model exists to beat this


def test_splits_are_leakage_free(cfg):
    splits = build_corpus(cfg, save=False)
    train_before = {r["before"] for r in splits["train"]}
    assert not (train_before & {r["before"] for r in splits["test"]})
    assert not (train_before & {r["before"] for r in splits["hard"]})


def test_evaluate_baseline_summary(cfg):
    from audiobook_ai.training.evaluate import evaluate
    res = evaluate(cfg, which="test", save=False)
    assert "baseline" in res
    assert 0.0 <= res["baseline"]["test"]["sentence_accuracy"] <= 1.0
    assert res["baseline"]["test"]["sentence_accuracy"] > res["baseline"]["hard"]["sentence_accuracy"]
