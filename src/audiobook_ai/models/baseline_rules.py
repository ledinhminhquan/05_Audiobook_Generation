"""Rule-based text normalizer — the **baseline** the neural model must beat.

A real-world-style baseline: an ordered set of regexes finds non-overlapping
semiotic spans (money, url, phone, time, measure, fraction, ordinal, decimal,
roman, cardinal, abbreviation) and each is expanded with a *single,
context-blind* rule. It deliberately cannot resolve context-dependent cases:

* "St." -> always *Saint* (so "Main St." comes out wrong),
* "Dr." -> always *Doctor* (so "Elm Dr." comes out wrong),
* a 4-digit number in [1500, 2099] -> always a *year* (so "1984 people" is wrong),
* a date day with no ordinal suffix -> a *cardinal* ("March 3" -> "march three").

These are exactly the gaps a context-aware seq2seq model closes. Runs with
**zero heavy dependencies**.
"""

from __future__ import annotations

import re
from typing import Callable, List, Tuple

from . import expanders as ex

_ROMAN_CTX_ORDINAL = {"king", "queen", "pope", "louis", "henry", "george",
                      "edward", "william", "elizabeth", "charles", "richard",
                      "philip", "napoleon", "peter", "catherine"}
_ROMAN_CTX_CARDINAL = {"chapter", "part", "section", "book", "act", "scene",
                       "volume", "vol", "appendix", "world war", "super bowl"}


def _h_money(m: re.Match) -> str:
    return ex.money(m.group(0))


def _h_time(m: re.Match) -> str:
    return ex.clock_time(m.group(0))


def _h_measure(m: re.Match) -> str:
    return ex.measure(m.group(0))


def _h_fraction(m: re.Match) -> str:
    return ex.fraction(m.group(0))


def _h_ordinal(m: re.Match) -> str:
    return ex.ordinal(int(m.group(1)))


def _h_decimal(m: re.Match) -> str:
    return ex.decimal(m.group(0))


def _h_phone(m: re.Match) -> str:
    return ex.telephone(m.group(0))


def _h_electronic(m: re.Match) -> str:
    return ex.electronic(m.group(0))


def _h_grouped(m: re.Match) -> str:
    return ex.cardinal(int(m.group(0).replace(",", "")))


def _h_cardinal(m: re.Match) -> str:
    n = int(m.group(0))
    if 1500 <= n <= 2099 and len(m.group(0)) == 4:   # context-blind year heuristic
        return ex.year(n)
    return ex.cardinal(n)


def _h_roman(m: re.Match) -> str:
    ctx, num = m.group(1), m.group(2)
    val = ex.roman_to_int(num)
    if val is None:
        return m.group(0)
    if ctx.lower() in _ROMAN_CTX_ORDINAL:
        return f"{ctx} the {ex.ordinal(val)}"
    return f"{ctx} {ex.cardinal(val)}"


def _h_abbrev(m: re.Match) -> str:
    key = m.group(0).lower().rstrip(".")
    if key in ex.AMBIGUOUS:
        return ex.AMBIGUOUS[key][0]                  # context-blind: first reading
    return ex.abbrev(m.group(0))


# (class, compiled-regex, handler) in PRIORITY order (earlier = wins overlaps).
_RULES: List[Tuple[str, re.Pattern, Callable[[re.Match], str]]] = [
    ("ELECTRONIC", re.compile(r"(?:https?://|www\.)[^\s]*[\w/]|\b[\w.+-]+@[\w-]+\.[\w.-]*[\w]\b"), _h_electronic),
    ("MONEY", re.compile(r"[$£€¥]\s?\d[\d,]*(?:\.\d+)?\s?(?:bn|tr|[kKmMbB])?\b"), _h_money),
    ("TELEPHONE", re.compile(r"\(?\d{3}\)?[\s.\-]\d{3}[\s.\-]\d{4}\b"), _h_phone),
    ("TIME", re.compile(r"\b\d{1,2}:\d{2}(?:\s?[ap]\.?m\.?)?\b", re.I), _h_time),
    ("MEASURE", re.compile(r"\b\d[\d,]*(?:\.\d+)?\s?(?:%|°[CF]|km/h|mph|kg|km|cm|mm|mg|ml|GB|MB|KB|TB|kHz|MHz|GHz|kW|lbs?|oz|ft|mi|hrs?|min|sec)(?!\w)"), _h_measure),
    ("ROMAN", re.compile(r"\b(king|queen|pope|louis|henry|george|edward|william|elizabeth|charles|richard|philip|napoleon|peter|catherine|chapter|part|section|book|act|scene|volume|vol|appendix)\s+([MDCLXVI]{1,4})\b", re.I), _h_roman),
    ("FRACTION", re.compile(r"\b\d+/\d+\b"), _h_fraction),
    ("ORDINAL", re.compile(r"\b(\d+)(?:st|nd|rd|th)\b", re.I), _h_ordinal),
    ("DECIMAL", re.compile(r"\b\d*\.\d+\b"), _h_decimal),
    ("CARDINAL_G", re.compile(r"\b\d{1,3}(?:,\d{3})+\b"), _h_grouped),
    ("CARDINAL", re.compile(r"\b\d+\b"), _h_cardinal),
    ("ABBREV", re.compile(r"\b(?:Mr|Mrs|Ms|Dr|Prof|St|Jr|Sr|Ave|Blvd|Rd|Ln|Mt|Vol|Ch|Fig|Dept|Govt|Inc|Ltd|Corp|approx|vs|etc|e\.g|i\.e)\.(?=\s|$|[^\w])|\b(?:St|Ave|Blvd|Rd|Dr|Ln|Mt)\b(?=\s+[A-Z])"), _h_abbrev),
]

_PRIORITY = {name: i for i, (name, _, _) in enumerate(_RULES)}


def _scan(text: str):
    """Return ordered non-overlapping spans ``(start, end, cls, spoken)``."""
    cand = []
    for cls, rx, h in _RULES:
        for m in rx.finditer(text):
            if m.start() == m.end():
                continue
            cand.append((m.start(), m.end(), _PRIORITY[cls], cls, h(m)))
    cand.sort(key=lambda c: (c[0], c[2], -(c[1] - c[0])))
    chosen = []
    last = -1
    for start, end, _prio, cls, spoken in cand:
        if start >= last:
            chosen.append((start, end, cls, spoken))
            last = end
    return chosen


def normalize(text: str) -> str:
    """Normalize a written string into its spoken form (baseline)."""
    spans = _scan(text)
    out: List[str] = []
    cur = 0
    for start, end, _cls, spoken in spans:
        out.append(text[cur:start])
        out.append(spoken)
        cur = end
    out.append(text[cur:])
    return re.sub(r"\s+", " ", "".join(out)).strip()


def normalize_tokens(text: str) -> List[Tuple[str, str, str]]:
    """Return ``[(surface, class, spoken)]`` for non-plain spans (per-class analysis)."""
    return [(text[s:e], cls, sp) for s, e, cls, sp in _scan(text)]


class RuleNormalizer:
    """Thin object wrapper so the agent / registry can treat it like a model."""

    version = "baseline-rules-1.0"
    name = "rule_baseline"

    def normalize(self, text: str) -> str:
        return normalize(text)

    def normalize_batch(self, texts: List[str]) -> List[str]:
        return [normalize(t) for t in texts]


__all__ = ["normalize", "normalize_tokens", "RuleNormalizer"]
