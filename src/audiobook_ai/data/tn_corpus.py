"""Synthetic Text-Normalization corpus generator (the PRIMARY training data).

There is no permissively-licensed English TN-Challenge mirror on the HF Hub
(the canonical ids 404), so we generate a reproducible, context-rich corpus of
``(written, spoken)`` pairs covering all 16 semiotic classes. Crucially we
**inject context-dependent ambiguity** the rule baseline cannot resolve:

* "Main St." -> *Street* but "St. Peter" -> *Saint*,
* "1984 people" -> *one thousand nine hundred eighty four* but "in 1984" -> *year*,
* "March 3, 2020" -> day as an *ordinal* (*third*),
* "Episode IV" -> *four* (a Roman context the baseline does not cover).

Those examples form the ``hard`` evaluation slice where the trained, context-aware
seq2seq model beats the baseline. Fully deterministic given a seed; **zero heavy
dependencies** (reuses ``models.expanders``).
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from ..models import expanders as ex

# ─────────────────────────────────────────────────────────────────────────────
# Vocabulary pools (kept plain so carriers normalize to themselves)
# ─────────────────────────────────────────────────────────────────────────────
_NAMES = ["Adams", "Bennett", "Clarke", "Dawson", "Ellis", "Foster", "Grant",
          "Hughes", "Irwin", "Jensen", "Keller", "Lawson", "Mercer", "Newton",
          "Owens", "Porter", "Quinn", "Reeves", "Sutton", "Turner", "Vance",
          "Walsh", "Mary", "Peter", "John", "Anne", "Paul", "James"]
_STREET_NAMES = ["Elm", "Baker", "Maple", "Oak", "Pine", "Cedar", "Park",
                 "Church", "Market", "King", "Queen", "River", "Hill", "Lake"]
_STREET_SUFFIX = [("St", "Street"), ("Dr", "Drive"), ("Ln", "Lane"),
                  ("Ave", "Avenue"), ("Rd", "Road"), ("Blvd", "Boulevard")]
_TITLE = [("Dr", "Doctor"), ("Mr", "Mister"), ("Mrs", "Missus"), ("Prof", "Professor"),
          ("St", "Saint")]
_MONTHS = ["January", "February", "March", "April", "May", "June", "July",
           "August", "September", "October", "November", "December"]
_REGNAL = ["Henry", "Louis", "George", "Edward", "William", "Charles", "Richard",
           "Elizabeth", "Catherine"]
_ROMAN_UNCOVERED = ["Episode", "Grade", "Level", "Type", "Mark", "Form", "Phase",
                    "Series", "Edition"]   # contexts the baseline ROMAN rule omits

_GENERIC = [
    "The total came to {X} in the end.",
    "She said {X} would be enough for now.",
    "He counted {X} before he left the room.",
    "According to the ledger, {X} remained.",
    "They reported {X} without any comment.",
    "Only {X} could be found on the shelf.",
    "The manuscript clearly states {X}.",
    "We settled on {X} after the meeting.",
    "It was {X}, give or take.",
    "Everyone agreed that {X} was fair.",
]
_COUNT_CARRIER = [
    "{X} people attended the rally that evening.",
    "There were {X} votes counted by midnight.",
    "The warehouse held {X} crates in total.",
    "More than {X} readers signed the petition.",
    "A crowd of {X} gathered outside the hall.",
]
_YEAR_CARRIER = [
    "In {X}, everything began to change.",
    "By {X} the project was complete.",
    "The treaty was signed in {X}.",
    "She was born in {X} near the coast.",
]
_TIME_CARRIER = ["We arrived at {X} sharp.", "The train departs at {X} every day.",
                 "Dinner is served at {X} in the hall."]
_DATE_CARRIER = ["The letter was dated {X}.", "On {X} the festival opened.",
                 "Their anniversary falls on {X}."]


def _wrap(rng: random.Random, written: str, spoken: str, carriers: List[str]) -> Tuple[str, str]:
    c = rng.choice(carriers)
    return c.format(X=written), c.format(X=spoken)


def _decade_words(d: int) -> str:
    base = ex.year(d)                       # "1980" -> "nineteen eighty"
    last = base.split()[-1]
    if last.endswith("y"):
        plural = last[:-1] + "ies"
    elif last in ("hundred", "thousand"):
        plural = last + "s"
    else:
        plural = last + "s"
    return " ".join(base.split()[:-1] + [plural])


# ─────────────────────────────────────────────────────────────────────────────
# Builders — each returns (before, after, classes, hard)
# ─────────────────────────────────────────────────────────────────────────────
def _b_plain(rng):
    s = rng.choice([
        "The morning light fell softly across the quiet valley.",
        "She closed the book and listened to the rain outside.",
        "He had never seen the harbor so calm before.",
        "They spoke of old friends and distant summers.",
        "A gentle wind carried the scent of pine through the camp.",
        "Nothing about the letter seemed unusual at first.",
    ])
    return s, s, ["PLAIN"], False


def _b_cardinal(rng):
    n = rng.choice([rng.randint(0, 99), rng.randint(100, 999), rng.randint(1000, 99999),
                    rng.randint(100000, 9999999)])
    written = f"{n:,}" if (n >= 1000 and rng.random() < 0.6) else str(n)
    cls = "CARDINAL"
    if rng.random() < 0.25:                 # token-level atomic example
        return written, ex.cardinal(n), [cls], False
    b, a = _wrap(rng, written, ex.cardinal(n), _GENERIC)
    return b, a, [cls], False


def _b_count_in_year_range(rng):            # HARD: looks like a year, read as count
    n = rng.randint(1500, 2099)
    b, a = _wrap(rng, str(n), ex.cardinal(n), _COUNT_CARRIER)
    return b, a, ["CARDINAL"], True


def _b_year(rng):
    y = rng.randint(1500, 2025)
    b, a = _wrap(rng, str(y), ex.year(y), _YEAR_CARRIER)
    return b, a, ["DATE"], False


def _b_decade(rng):                         # HARD: "1980s" -> "nineteen eighties"
    d = rng.choice([1820, 1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010])
    b, a = _wrap(rng, f"{d}s", _decade_words(d), _GENERIC)
    return b, a, ["DATE"], True


def _b_ordinal(rng):
    n = rng.randint(1, 120)
    suf = {1: "st", 2: "nd", 3: "rd"}.get(n % 10 if n % 100 not in (11, 12, 13) else 0, "th")
    written = f"{n}{suf}"
    if rng.random() < 0.25:
        return written, ex.ordinal(n), ["ORDINAL"], False
    b, a = _wrap(rng, written, ex.ordinal(n), _GENERIC)
    return b, a, ["ORDINAL"], False


def _b_decimal(rng):
    x = round(rng.uniform(0, 999), rng.choice([1, 2, 3]))
    written = str(x)
    b, a = _wrap(rng, written, ex.decimal(written), _GENERIC)
    return b, a, ["DECIMAL"], False


def _b_money(rng):
    if rng.random() < 0.4:
        amt = round(rng.uniform(1.0, 9.99), 1)
        mag = rng.choice(["M", "B", "k"])
        written = f"${amt}{mag}"
    else:
        n = rng.randint(1, 999999)
        cents = rng.choice(["", f".{rng.randint(0,99):02d}"])
        written = f"${n:,}{cents}"
    b, a = _wrap(rng, written, ex.money(written), _GENERIC)
    return b, a, ["MONEY"], False


def _b_measure(rng):
    unit = rng.choice(["km", "kg", "cm", "mm", "ml", "GB", "MB", "mph", "kW", "%", "ft", "mi", "oz"])
    num = rng.choice([str(rng.randint(1, 999)), f"{rng.uniform(0.1, 99.9):.1f}"])
    written = f"{num}{unit}" if rng.random() < 0.5 else f"{num} {unit}"
    if rng.random() < 0.3:
        return written, ex.measure(written), ["MEASURE"], False
    b, a = _wrap(rng, written, ex.measure(written), _GENERIC)
    return b, a, ["MEASURE"], False


def _b_time(rng):
    h, m = rng.randint(1, 12), rng.choice([0, 0, 5, 15, 30, 45, rng.randint(1, 59)])
    ampm = rng.choice(["", " AM", " PM"])
    written = f"{h}:{m:02d}{ampm}"
    b, a = _wrap(rng, written, ex.clock_time(written), _TIME_CARRIER)
    return b, a, ["TIME"], False


def _b_fraction(rng):
    den = rng.choice([2, 3, 4, 5, 6, 8, 10])
    num = rng.randint(1, den - 1)
    written = f"{num}/{den}"
    b, a = _wrap(rng, written, ex.fraction(written), _GENERIC)
    return b, a, ["FRACTION"], False


def _b_telephone(rng):
    a3, b3, c4 = rng.randint(200, 999), rng.randint(200, 999), rng.randint(0, 9999)
    written = rng.choice([f"{a3}-{b3}-{c4:04d}", f"({a3}) {b3}-{c4:04d}"])
    b, a = _wrap(rng, written, ex.telephone(written), _GENERIC)
    return b, a, ["TELEPHONE"], False


def _b_electronic(rng):
    host = rng.choice(["example", "books", "audio", "openlib", "readit"])
    tld = rng.choice(["com", "org", "net", "io"])
    if rng.random() < 0.5:
        written = f"www.{host}.{tld}"
    else:
        written = f"{rng.choice(['info', 'hello', 'contact'])}@{host}.{tld}"
    b, a = _wrap(rng, written, ex.electronic(written), _GENERIC)
    return b, a, ["ELECTRONIC"], False


def _b_date(rng):                           # HARD: day read as an ordinal
    mon = rng.choice(_MONTHS)
    day = rng.randint(1, 28)
    year = rng.randint(1700, 2025)
    written = f"{mon} {day}, {year}"
    spoken = f"{mon} {ex.ordinal(day)}, {ex.year(year)}"
    b, a = _wrap(rng, written, spoken, _DATE_CARRIER)
    return b, a, ["DATE"], True


def _b_ambiguous_street(rng):               # HARD: St./Dr. = title vs street suffix
    title_abbr, title_sp = rng.choice(_TITLE)
    name = rng.choice(_NAMES)
    num = rng.randint(2, 990)
    street = rng.choice(_STREET_NAMES)
    suf_abbr, suf_sp = rng.choice(_STREET_SUFFIX)
    written = f"{title_abbr}. {name} lived at {num} {street} {suf_abbr}."
    spoken = f"{title_sp} {name} lived at {ex.cardinal(num)} {street} {suf_sp}."
    return written, re.sub(r"\s+", " ", spoken).strip(), ["ABBREV"], True


def _b_regnal_roman(rng):                   # baseline-covered (both right): good coverage
    name = rng.choice(_REGNAL)
    n = rng.randint(1, 20)
    roman = ex._int_to_roman(n)
    written = f"{name} {roman}"
    spoken = f"{name} the {ex.ordinal(n)}"
    b, a = _wrap(rng, written, spoken, _GENERIC)
    return b, a, ["ROMAN"], False


def _b_uncovered_roman(rng):                # HARD: Roman context the baseline omits
    ctx = rng.choice(_ROMAN_UNCOVERED)
    n = rng.randint(1, 12)
    roman = ex._int_to_roman(n)
    written = f"{ctx} {roman}"
    spoken = f"{ctx} {ex.cardinal(n)}"
    b, a = _wrap(rng, written, spoken, _GENERIC)
    return b, a, ["ROMAN"], True


def _b_abbrev(rng):
    key = rng.choice(["Mr", "Mrs", "Prof", "etc", "vs", "Inc", "Ltd", "Dept", "Ave"])
    name = rng.choice(_NAMES)
    written = f"{key}. {name} agreed to the terms."
    spoken = f"{ex.abbrev(key)} {name} agreed to the terms."
    return written, spoken, ["ABBREV"], False


def _b_mixed(rng):                          # two semiotic spans for realism
    n = rng.randint(2, 99)
    pct = rng.randint(1, 99)
    written = f"About {n} of the {pct}% surveyed agreed."
    spoken = f"About {ex.cardinal(n)} of the {ex.measure(str(pct) + '%')} surveyed agreed."
    return written, spoken, ["CARDINAL", "MEASURE"], False


# (builder, weight) — rare/ambiguous classes are over-sampled; PLAIN capped low.
_BUILDERS: List[Tuple[Callable, float]] = [
    (_b_plain, 0.06), (_b_cardinal, 0.12), (_b_count_in_year_range, 0.06),
    (_b_year, 0.06), (_b_decade, 0.05), (_b_ordinal, 0.07), (_b_decimal, 0.05),
    (_b_money, 0.08), (_b_measure, 0.08), (_b_time, 0.06), (_b_fraction, 0.05),
    (_b_telephone, 0.04), (_b_electronic, 0.04), (_b_date, 0.06),
    (_b_ambiguous_street, 0.06), (_b_regnal_roman, 0.02), (_b_uncovered_roman, 0.02),
    (_b_abbrev, 0.03), (_b_mixed, 0.02),
]


@dataclass
class TNExample:
    before: str
    after: str
    classes: List[str]
    hard: bool

    def as_dict(self) -> Dict:
        return {"before": self.before, "after": self.after,
                "classes": self.classes, "hard": self.hard}


class TNCorpusGenerator:
    """Deterministic (written, spoken) example generator."""

    def __init__(self, seed: int = 42):
        self._builders = [b for b, _ in _BUILDERS]
        self._weights = [w for _, w in _BUILDERS]
        self.seed = seed

    def generate(self, n: int, seed: Optional[int] = None, dedup: bool = True) -> List[TNExample]:
        rng = random.Random(self.seed if seed is None else seed)
        out: List[TNExample] = []
        seen = set()
        guard = 0
        while len(out) < n and guard < n * 50:
            guard += 1
            builder = rng.choices(self._builders, weights=self._weights, k=1)[0]
            try:
                before, after, classes, hard = builder(rng)
            except Exception:
                continue
            after = re.sub(r"\s+", " ", after).strip()
            before = before.strip()
            if dedup:
                key = (before, after)
                if key in seen:
                    continue
                seen.add(key)
            out.append(TNExample(before, after, classes, hard))
        return out

    def generate_hard(self, n: int, seed: Optional[int] = None) -> List[TNExample]:
        """Generate ``n`` examples drawn only from the ambiguous (hard) builders."""
        hard_builders = [(_b_count_in_year_range, 1), (_b_decade, 1), (_b_date, 1),
                         (_b_ambiguous_street, 1), (_b_uncovered_roman, 1)]
        rng = random.Random((self.seed if seed is None else seed) + 999)
        out: List[TNExample] = []
        seen = set()
        guard = 0
        bs = [b for b, _ in hard_builders]
        while len(out) < n and guard < n * 50:
            guard += 1
            before, after, classes, hard = rng.choice(bs)(rng)
            after = re.sub(r"\s+", " ", after).strip()
            if (before, after) in seen:
                continue
            seen.add((before, after))
            out.append(TNExample(before, after, classes, True))
        return out


__all__ = ["TNCorpusGenerator", "TNExample", "SEMIOTIC_BUILDERS"]
SEMIOTIC_BUILDERS = [b.__name__ for b, _ in _BUILDERS]
