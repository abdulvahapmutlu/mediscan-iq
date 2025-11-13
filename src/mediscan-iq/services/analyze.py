from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
from ..logging import get_logger
from ..preprocess.anonymizer import anonymize
from ..preprocess.segmenter import split_sentences
from ..nlp.summarizer import Summarizer
from ..nlp.risk_tagger import RiskTagger

logger = get_logger("mediscan.services.analyze")

@dataclass
class AnalysisOutput:
    summary: str
    risk_level: str
    risk_probs: Dict[str, float]
    sentences: List[str]
    anonymized: str
    meta: Dict[str, str]

class Analyzer:
    def __init__(self) -> None:
        self.summarizer = Summarizer()
        self.risk = RiskTagger()

    def run(self, text: str, report_type: str) -> AnalysisOutput:
        # Step-1 reuse: anonymize + sentence split
        anonym, counts = anonymize(text)
        sents = split_sentences(anonym)
        logger.info("Analyze: anonymized=%d chars | sents=%d | type=%s",
                    len(anonym), len(sents), report_type)

        # Step-2: summarization + risk
        summary, s_meta = self.summarizer.summarize(anonym, report_type=report_type)
        risk_label, probs, r_meta = self.risk.tag(anonym)

        meta = {}
        meta.update({f"phi_{k}": str(v) for k, v in counts.items()})
        meta.update({f"summ_{k}": v for k, v in s_meta.items()})
        meta.update({f"risk_{k}": v for k, v in r_meta.items()})

        return AnalysisOutput(
            summary=summary,
            risk_level=risk_label,
            risk_probs=probs,
            sentences=sents,
            anonymized=anonym,
            meta=meta,
        )
