from typing import List
import re
import nltk

# Ensure punkt is available (one-time download if missing).
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:  # pragma: no cover
    nltk.download("punkt")

_SENT_SPLIT_FALLBACK = re.compile(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s")

def split_sentences(text: str) -> List[str]:
    """
    Robust sentence splitter with NLTK punkt (fallback to regex).
    Keeps short clinical notes readable without over-segmentation.
    """
    if not text.strip():
        return []
    try:
        from nltk.tokenize import sent_tokenize
        sents = sent_tokenize(text)
        # Filter very short artifacts
        return [s.strip() for s in sents if len(s.strip()) > 1]
    except Exception:
        # Conservative regex fallback
        sents = _SENT_SPLIT_FALLBACK.split(text)
        return [s.strip() for s in sents if s.strip()]
