from __future__ import annotations
import hashlib
import regex as re
from typing import Iterable, Tuple, Dict
from ..config import settings

# PHI patterns (balanced precision/recall for clinical notes; tune for your data)
PATTERNS: Dict[str, re.Pattern] = {
    "email": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    "phone": re.compile(r"\b(?:\+?\d{1,3})?[-. (]*\d{2,4}[-. )]*\d{3,4}[-. ]*\d{3,4}\b"),
    "mrn": re.compile(r"\b(MRN[:\s]*\d{5,12}|\d{7,12})\b", re.I),
    "ssn_like": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "date": re.compile(r"\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s+\d{4})\b", re.I),
    "name_hint": re.compile(r"\b(?:Patient|Pt\.?|Mr\.|Ms\.|Mrs\.|Dr\.|MD|RN)\s+[A-Z][a-z]+(?:\s[A-Z][a-z]+)?\b"),
    "address_like": re.compile(r"\b\d{1,5}\s+[A-Z][A-Za-z]+\s(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Ln|Lane)\b", re.I),
}

def _hash_token(token: str) -> str:
    h = hashlib.sha256((settings.hash_salt + token).encode("utf-8")).hexdigest()[:10]
    return f"<ID:{h}>"

def _mask_token(token: str) -> str:
    return settings.mask_char * max(6, len(token))

def _replace(text: str, pattern: re.Pattern, strategy: str) -> str:
    def repl(m: re.Match) -> str:
        token = m.group(0)
        if strategy == "hash":
            return _hash_token(token)
        return _mask_token(token)
    return pattern.sub(repl, text)

def anonymize(text: str) -> Tuple[str, Dict[str, int]]:
    """
    Apply configurable PHI anonymization. Returns (text, counts).
    """
    counts: Dict[str, int] = {}
    strategy = settings.anonymize_strategy.lower().strip()
    keep_dates = bool(settings.keep_dates)

    out = text

    for name, pattern in PATTERNS.items():
        if name == "date" and keep_dates:
            continue
        before = out
        out = _replace(out, pattern, strategy=strategy)
        if out != before:
            counts[name] = counts.get(name, 0) + len(list(pattern.finditer(before)))

    if settings.reduce_whitespace:
        out = re.sub(r"[ \t]+", " ", out)
        out = re.sub(r"\s*\n\s*", "\n", out).strip()

    return out, counts
