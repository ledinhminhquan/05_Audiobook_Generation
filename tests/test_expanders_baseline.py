"""Expanders + the context-blind rule baseline."""

from __future__ import annotations

from audiobook_ai.models import expanders as ex
from audiobook_ai.models import baseline_rules as br


def test_expanders_core():
    assert ex.cardinal(1234) == "one thousand two hundred thirty four"
    assert ex.year(1984) == "nineteen eighty four"
    assert ex.year(2000) == "two thousand"
    assert ex.ordinal(21) == "twenty first"
    assert ex.decimal("3.14") == "three point one four"
    assert ex.money("$5.2M") == "five point two million dollars"
    assert "cents" in ex.money("$1,234.50")
    assert ex.measure("5km") == "five kilometers"
    assert ex.measure("20%") == "twenty percent"
    assert ex.fraction("3/4") == "three quarters"
    assert ex.roman("XIV") == "fourteen"
    assert ex.roman_to_int("IIII") is None      # invalid roman rejected


def test_baseline_normalizes_numbers():
    out = br.normalize("He paid $5.2M for 1,204 shares.")
    assert "five point two million dollars" in out
    assert "one thousand two hundred four" in out


def test_baseline_is_context_blind_on_ambiguity():
    # "St." -> always Saint; in "Main St." that is the wrong reading the model fixes.
    out = br.normalize("She walked down Main St.")
    assert "saint" in out.lower()
    # percent and url handled
    assert "twenty percent" in br.normalize("It was 20% done.")
    assert "dot" in br.normalize("Visit www.example.com today.")


def test_baseline_year_heuristic_and_roman_context():
    assert "nineteen eighty four" in br.normalize("It happened in 1984.")
    assert "the eighth" in br.normalize("King Henry VIII reigned.")
    assert "chapter four" in br.normalize("Chapter IV is long.").lower()


def test_rule_normalizer_object():
    rn = br.RuleNormalizer()
    outs = rn.normalize_batch(["12 cats", "$3"])
    assert outs == ["twelve cats", "three dollars"]
