"""Low-level text-normalization *expansion primitives* (written -> spoken).

These deterministic functions convert individual semiotic tokens (numbers,
money, dates, measures, ...) into their spoken form. They are deliberately
**shared** by two consumers:

* ``data/tn_corpus.py`` — generates the (written, spoken) training corpus, where
  the spoken side is built with *context awareness* (e.g. "St." -> Saint vs
  Street depending on neighbours).
* ``models/baseline_rules.py`` — the *context-blind* rule baseline that the
  trained neural model must beat.

Everything works with **zero heavy dependencies**: ``num2words`` is used when
installed (better currency/ordinal coverage) but a self-contained pure-Python
fallback keeps the package importable and testable on core deps only.
"""

from __future__ import annotations

import re
from typing import List, Optional

try:  # optional, improves coverage; never required
    from num2words import num2words as _n2w  # type: ignore
    _HAVE_N2W = True
except Exception:  # pragma: no cover - exercised when dep missing
    _HAVE_N2W = False


# ─────────────────────────────────────────────────────────────────────────────
# Pure-python cardinal fallback (covers 0 .. 10^15)
# ─────────────────────────────────────────────────────────────────────────────
_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
         "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
         "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
         "eighty", "ninety"]
_SCALES = [(10 ** 12, "trillion"), (10 ** 9, "billion"), (10 ** 6, "million"),
           (10 ** 3, "thousand"), (100, "hundred")]


def _under_1000(n: int) -> str:
    parts: List[str] = []
    if n >= 100:
        parts.append(_ONES[n // 100])
        parts.append("hundred")
        n %= 100
    if n >= 20:
        parts.append(_TENS[n // 10])
        if n % 10:
            parts.append(_ONES[n % 10])
    elif n > 0:
        parts.append(_ONES[n])
    return " ".join(parts)


def _int_to_words_fallback(n: int) -> str:
    if n == 0:
        return "zero"
    neg = n < 0
    n = abs(n)
    chunks: List[str] = []
    for scale, name in _SCALES[:-1]:  # trillion .. thousand
        if n >= scale:
            chunks.append(_under_1000(n // scale) + " " + name)
            n %= scale
    if n > 0:
        chunks.append(_under_1000(n))
    out = " ".join(chunks).strip()
    return ("negative " + out) if neg else out


def _canon(s: str) -> str:
    """Canonicalize verbaliser output to the US, no-'and', no-hyphen style.

    Ensures the corpus is byte-identical whether or not ``num2words`` is installed
    (num2words emits "one thousand, two hundred and thirty-four"; we drop the
    comma, the hyphen and the British "and").
    """
    s = s.replace("-", " ").replace(",", " ")
    s = re.sub(r"\band\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def cardinal(n: int) -> str:
    """Integer -> spoken cardinal ("1234" -> "one thousand two hundred thirty four")."""
    if _HAVE_N2W:
        try:
            return _canon(_n2w(int(n)))
        except Exception:
            pass
    return _int_to_words_fallback(int(n))


_ORD_SUFFIX = {"one": "first", "two": "second", "three": "third", "five": "fifth",
               "eight": "eighth", "nine": "ninth", "twelve": "twelfth"}


def ordinal(n: int) -> str:
    """Integer -> spoken ordinal ("21" -> "twenty first")."""
    if _HAVE_N2W:
        try:
            return _canon(_n2w(int(n), to="ordinal"))
        except Exception:
            pass
    words = cardinal(n).split()
    last = words[-1]
    if last in _ORD_SUFFIX:
        words[-1] = _ORD_SUFFIX[last]
    elif last.endswith("y"):
        words[-1] = last[:-1] + "ieth"
    else:
        words[-1] = last + "th"
    return " ".join(words)


def decimal(text: str) -> str:
    """"3.14" -> "three point one four"; "0.5" -> "zero point five"."""
    text = text.replace(",", "")
    neg = text.startswith("-")
    text = text.lstrip("+-")
    if "." not in text:
        return cardinal(int(text)) if text.isdigit() else text
    whole, frac = text.split(".", 1)
    whole_words = cardinal(int(whole)) if whole.isdigit() else "zero"
    frac_words = " ".join(_ONES[int(d)] for d in frac if d.isdigit())
    out = f"{whole_words} point {frac_words}".strip()
    return ("negative " + out) if neg else out


def digits(text: str, oh_for_zero: bool = True) -> str:
    """Speak each digit ("007" -> "oh oh seven")."""
    names = {**{str(i): _ONES[i] for i in range(10)}}
    if oh_for_zero:
        names["0"] = "oh"
    return " ".join(names[c] for c in text if c.isdigit())


def year(n: int) -> str:
    """Spoken *year* form ("1984" -> "nineteen eighty four", "2007" -> "two thousand seven")."""
    n = int(n)
    if n < 1000 or n > 9999:
        return cardinal(n)
    hi, lo = divmod(n, 100)
    if 2000 <= n <= 2009:
        return "two thousand" if lo == 0 else f"two thousand {cardinal(lo)}"
    if lo == 0:
        return f"{cardinal(hi)} hundred"
    if lo < 10:
        return f"{cardinal(hi)} oh {cardinal(lo)}"
    return f"{cardinal(hi)} {cardinal(lo)}"


# ── money ────────────────────────────────────────────────────────────────────
_CURRENCY = {"$": "dollars", "£": "pounds", "€": "euros", "¥": "yen",
             "USD": "dollars", "GBP": "pounds", "EUR": "euros"}
_CENTS = {"$": "cents", "£": "pence", "€": "cents"}
_MAGNITUDE = {"k": "thousand", "K": "thousand", "m": "million", "M": "million",
              "b": "billion", "B": "billion", "bn": "billion", "tr": "trillion"}


def money(text: str) -> str:
    """"$5.2M" -> "five point two million dollars"; "$1,234.50" -> "... dollars and fifty cents"."""
    t = text.strip()
    sym = "$"
    for s in _CURRENCY:
        if t.startswith(s):
            sym = s
            t = t[len(s):]
            break
    mag = ""
    m = re.search(r"([kKmMbB]|bn|tr)\b", t)
    if m and m.group(1) in _MAGNITUDE:
        mag = _MAGNITUDE[m.group(1)]
        t = t[:m.start()] + t[m.end():]
    t = t.strip().replace(",", "")
    unit = _CURRENCY.get(sym, "dollars")
    if mag:
        amount = decimal(t) if "." in t else (cardinal(int(t)) if t.isdigit() else t)
        return f"{amount} {mag} {unit}".strip()
    if "." in t:
        whole, frac = t.split(".", 1)
        frac = (frac + "00")[:2]
        words = f"{cardinal(int(whole) if whole.isdigit() else 0)} {unit}"
        if int(frac):
            words += f" and {cardinal(int(frac))} {_CENTS.get(sym, 'cents')}"
        return words
    return f"{cardinal(int(t)) if t.isdigit() else t} {unit}"


# ── measures ───────────────────────────────────────────────────────────────--
_UNITS = {
    "km": "kilometers", "m": "meters", "cm": "centimeters", "mm": "millimeters",
    "kg": "kilograms", "g": "grams", "mg": "milligrams", "lb": "pounds", "oz": "ounces",
    "ml": "milliliters", "l": "liters", "km/h": "kilometers per hour", "mph": "miles per hour",
    "%": "percent", "°c": "degrees celsius", "°f": "degrees fahrenheit", "ft": "feet",
    "in": "inches", "mi": "miles", "kb": "kilobytes", "mb": "megabytes", "gb": "gigabytes",
    "tb": "terabytes", "hz": "hertz", "khz": "kilohertz", "ghz": "gigahertz", "w": "watts",
    "kw": "kilowatts", "s": "seconds", "min": "minutes", "hr": "hours",
}


def measure(text: str) -> str:
    """"5km" / "3.5 kg" / "20%" -> spoken form."""
    t = text.strip()
    m = re.match(r"^([+-]?\d[\d,]*\.?\d*)\s*([a-zA-Z°%/]+)$", t)
    if not m:
        if t.endswith("%"):
            num = t[:-1].strip()
            return f"{decimal(num) if '.' in num else cardinal(int(num)) if num.isdigit() else num} percent"
        return t
    num, unit = m.group(1), m.group(2).lower()
    num_words = decimal(num) if "." in num else (cardinal(int(num.replace(',', ''))) if num.replace(',', '').isdigit() else num)
    unit_words = _UNITS.get(unit, unit)
    return f"{num_words} {unit_words}"


# ── time ─────────────────────────────────────────────────────────────────────
def clock_time(text: str) -> str:
    """"3:30 PM" -> "three thirty p m"; "9:00" -> "nine o'clock"; "12:05" -> "twelve oh five"."""
    t = text.strip()
    ampm = ""
    m = re.search(r"\b([ap])\.?m\.?$", t, re.I)
    if m:
        ampm = " " + " ".join(list(m.group(1).lower())) + " m"
        t = t[:m.start()].strip()
    m = re.match(r"^(\d{1,2}):(\d{2})$", t)
    if not m:
        return text
    h, mn = int(m.group(1)), int(m.group(2))
    if mn == 0:
        body = f"{cardinal(h)} o'clock" if not ampm else cardinal(h)
    elif mn < 10:
        body = f"{cardinal(h)} oh {cardinal(mn)}"
    else:
        body = f"{cardinal(h)} {cardinal(mn)}"
    return (body + ampm).strip()


# ── fraction ───────────────────────────────────────────────────────────────--
_FRAC_DENOM = {2: "half", 3: "third", 4: "quarter", 5: "fifth", 6: "sixth",
               7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth"}


def fraction(text: str) -> str:
    """"3/4" -> "three quarters"; "1/2" -> "one half"."""
    m = re.match(r"^(\d+)\s*/\s*(\d+)$", text.strip())
    if not m:
        return text
    num, den = int(m.group(1)), int(m.group(2))
    den_word = _FRAC_DENOM.get(den, ordinal(den))
    num_word = cardinal(num)
    if num != 1:
        den_word += "s"
    return f"{num_word} {den_word}"


# ── roman numerals ───────────────────────────────────────────────────────────
_ROMAN_MAP = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"),
              (90, "XC"), (50, "L"), (40, "XL"), (10, "X"), (9, "IX"),
              (5, "V"), (4, "IV"), (1, "I")]


def roman_to_int(s: str) -> Optional[int]:
    s = s.upper()
    if not re.fullmatch(r"[MDCLXVI]+", s):
        return None
    vals = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total, prev = 0, 0
    for ch in reversed(s):
        v = vals[ch]
        total += -v if v < prev else v
        prev = max(prev, v)
    # round-trip check rejects invalid sequences like "IIII"
    return total if _int_to_roman(total) == s else None


def _int_to_roman(n: int) -> str:
    out = []
    for val, sym in _ROMAN_MAP:
        while n >= val:
            out.append(sym)
            n -= val
    return "".join(out)


def roman(text: str, ordinal_form: bool = False) -> str:
    n = roman_to_int(text)
    if n is None:
        return text
    return ordinal(n) if ordinal_form else cardinal(n)


# ── telephone ─────────────────────────────────────────────────────────────---
def telephone(text: str) -> str:
    """"555-1234" / "(800) 273-8255" -> digit-by-digit (sil between groups)."""
    out = []
    for ch in text:
        if ch.isdigit():
            out.append(digits(ch))
        elif ch in "-).(":
            out.append("sil")
    cleaned: List[str] = []
    for w in " ".join(out).split():
        if w == "sil" and (not cleaned or cleaned[-1] == "sil"):
            continue
        cleaned.append(w)
    return " ".join(w for w in cleaned if w != "sil").strip()


# ── electronic (URLs / emails / hashtags) ────────────────────────────────────
_SYMBOL_WORDS = {".": "dot", "/": "slash", ":": "colon", "@": "at", "_": "underscore",
                 "-": "dash", "#": "hash", "&": "and", "=": "equals", "?": "question mark"}


def electronic(text: str) -> str:
    """"http://example.com" / "a@b.com" -> spoken symbols + lettered domain pieces."""
    t = text.strip().rstrip(".,;:!?")
    t = re.sub(r"^https?://", "", t)
    t = t.rstrip("/")
    out: List[str] = []
    for tok in re.split(r"([./:@_#&=\-])", t):
        if not tok:
            continue
        if tok in _SYMBOL_WORDS:
            out.append(_SYMBOL_WORDS[tok])
        elif tok.isdigit():
            out.append(digits(tok))
        elif tok in ("www",):
            out.append("w w w")
        elif len(tok) <= 3 and tok.lower() in ("com", "org", "net", "io", "gov", "edu", "co", "uk"):
            out.append(tok.lower())
        else:
            out.append(tok)
    return " ".join(out)


# ── abbreviations / titles ───────────────────────────────────────────────────
# Unambiguous abbreviations expanded the same way regardless of context.
ABBREV = {
    "mr": "mister", "mrs": "missus", "ms": "miz", "prof": "professor",
    "jr": "junior", "sr": "senior", "vs": "versus", "etc": "et cetera",
    "e.g": "for example", "i.e": "that is", "approx": "approximately",
    "dept": "department", "govt": "government", "inc": "incorporated",
    "ltd": "limited", "corp": "corporation", "no": "number", "vol": "volume",
    "ch": "chapter", "fig": "figure", "p": "page", "pp": "pages", "ave": "avenue",
    "blvd": "boulevard", "rd": "road", "mt": "mount", "ft": "fort",
    "phd": "p h d", "ceo": "c e o", "usa": "u s a", "uk": "u k",
    "am": "a m", "pm": "p m", "lb": "pounds", "oz": "ounces",
}

# Context-dependent abbreviations: spoken form depends on neighbouring words.
# (A context-blind baseline must pick ONE of the readings and thus errs ~half
#  the time — exactly where the trained neural model wins.)
AMBIGUOUS = {
    "st": ("saint", "street"),    # St. Peter  vs  Main St.
    "dr": ("doctor", "drive"),    # Dr. Smith  vs  Elm Dr.
    "ln": ("lane", "line"),
    "sq": ("square", "square"),
}


def abbrev(token: str) -> str:
    key = token.lower().rstrip(".")
    return ABBREV.get(key, token)


__all__ = [
    "cardinal", "ordinal", "decimal", "digits", "year", "money", "measure",
    "clock_time", "fraction", "roman", "roman_to_int", "telephone", "electronic",
    "abbrev", "ABBREV", "AMBIGUOUS",
]
