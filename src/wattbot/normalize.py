"""Turn a model's free-form value into the form the scorer compares.

The scorer (data/Score.py) treats an answer_value as one of:
  * a plain number          -> numeric compare, +-0.1 % relative
  * "[lo,hi]"  (range)      -> a single number inside the range is correct
  * "(a,b)"    (list)       -> every element is required
  * text                    -> compared after lower-casing, punctuation and leading-article removal
  * "is_blank"              -> the question is unanswerable

Design rule: normalisation is idempotent and conservative. A value that is already clean is
returned untouched (so "7.0" stays "7.0" and "Flan-T5-xxl" stays as is); we only rewrite when
there is something to strip (commas, %, ~, currency, words like "million", unit echoes).
"""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

from .config import BLANK

_BLANK_TOKENS = {
    "", "na", "n/a", "none", "null", "nan", "nil", "is_blank", "unanswerable",
    "not available", "not specified", "not stated", "unknown", "cannot be determined",
}
_FALLBACK_PREFIX = "unable to answer with confidence"

_SCALES = {
    "thousand": Decimal(10) ** 3,
    "million": Decimal(10) ** 6,
    "billion": Decimal(10) ** 9,
    "trillion": Decimal(10) ** 12,
}
_FILLER = {"approximately", "about", "around", "roughly", "of", "per", "times", "x", "fold", "percent", "approx"}

_PLAIN_NUMBER = re.compile(r"^-?\d+(\.\d+)?$")
_NUM = r"-?\d[\d,]*(?:\.\d+)?|-?\.\d+"
_NUMBER = re.compile(
    rf"""^\s*[~≈<>≤≥]*\s*[$€£]?\s*
         (?P<num>{_NUM})
         (?:\s*[eE]\s*(?P<exp>[+-]?\d+))?
         \s*(?P<pct>%)?
         \s*(?P<scale>thousand|million|billion|trillion)?
         \s*(?P<rest>.*?)\s*$""",
    re.VERBOSE | re.IGNORECASE,
)
_RANGE = re.compile(rf"^\s*(?:between\s+)?(?P<a>{_NUM})\s*(?:-|–|—|to|and)\s*(?P<b>{_NUM})\s*(?P<rest>.*?)\s*$", re.IGNORECASE)


def is_blank_value(v: str | None) -> bool:
    s = (v or "").strip().lower()
    return s in _BLANK_TOKENS or s.startswith(_FALLBACK_PREFIX)


def _unit_tokens(unit: str | None) -> set[str]:
    u = (unit or "").lower()
    if u in ("", BLANK):
        return set()
    return {t for t in re.split(r"[^a-z0-9%]+", u) if t}


def _fmt(d: Decimal) -> str:
    d = d.normalize()
    s = format(d, "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return "0" if s in ("", "-0") else s


def _to_decimal(num: str) -> Decimal | None:
    try:
        return Decimal(num.replace(",", ""))
    except InvalidOperation:
        return None


def _rest_is_unit_echo(rest: str, unit: str | None) -> bool:
    if not rest:
        return True
    allowed = _unit_tokens(unit) | _FILLER
    toks = [t for t in re.split(r"[^a-z0-9%]+", rest.lower()) if t]
    # singular/plural tolerant
    allowed |= {t.rstrip("s") for t in allowed}
    return all(t in allowed or t.rstrip("s") in allowed for t in toks)


def _clean_number(s: str, unit: str | None) -> str | None:
    """Return the canonical number string, or None when `s` is not purely a number."""
    if _PLAIN_NUMBER.match(s.strip()):
        return s.strip()
    m = _NUMBER.match(s)
    if not m:
        return None
    if not _rest_is_unit_echo(m["rest"], unit):
        return None
    d = _to_decimal(m["num"])
    if d is None:
        return None
    if m["exp"]:
        d = d * (Decimal(10) ** int(m["exp"]))
    scale = (m["scale"] or "").lower()
    if scale and scale not in _unit_tokens(unit):
        d = d * _SCALES[scale]  # "1.2 million" -> 1200000 unless the unit itself is in millions
    return _fmt(d)


def _normalise_collection(s: str, unit: str | None) -> str | None:
    if not ((s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")"))):
        return None
    items = [t.strip() for t in s[1:-1].split(",")]
    if not items or any(not t for t in items):
        return None
    cleaned = []
    for t in items:
        c = _clean_number(t, unit)
        cleaned.append(c if c is not None else t.strip("'\" "))
    return s[0] + ",".join(cleaned) + s[-1]


def normalise_value(raw: str | None, unit: str | None = None) -> str:
    """Normalise one answer_value. `unit` is the expected unit (given in test_Q.csv)."""
    if raw is None:
        return BLANK
    s = str(raw).strip()
    if is_blank_value(s):
        return BLANK
    # strip wrapping quotes / trailing period on text
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        s = s[1:-1].strip()

    coll = _normalise_collection(s, unit)
    if coll is not None:
        return coll

    num = _clean_number(s, unit)
    if num is not None:
        return num

    m = _RANGE.match(s)
    if m and _rest_is_unit_echo(m["rest"], unit):
        a, b = _to_decimal(m["a"]), _to_decimal(m["b"])
        if a is not None and b is not None and a <= b:
            return f"[{_fmt(a)},{_fmt(b)}]"

    return s.rstrip(".").strip() if s.endswith(".") and not re.search(r"\b[A-Z]\.$", s) else s
