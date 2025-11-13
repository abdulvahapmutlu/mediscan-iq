from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import re
import torch
import numpy as np
from transformers import PreTrainedTokenizerBase, PreTrainedModel
from ..config import settings
from ..logging import get_logger
from .registry import select_device, set_seed, load_nli

logger = get_logger("mediscan.nlp.risk")

HEURISTIC_PATTERNS = {
    "high": [
        r"\b(subarachnoid|intracranial)\s+hemorrhage\b",
        r"\bmass\s+(?:with|and)?\s*invasion\b",
        r"\bpulmonary\s+embolism\b",
        r"\bstemi\b|\bnstemi\b|\bmyocardial\s+infarction\b",
        r"\bmalignan\w+\b",
        r"\bperforation\b",
    ],
    "moderate": [
        r"\bcardiomegaly\b",
        r"\bconsolidation\b",
        r"\beffusion\b",
        r"\bpneumonia\b",
        r"\bfracture\b",
        r"\bischemi\w+\b",
    ],
}

@dataclass
class RiskConfig:
    model_name: str
    labels: List[str]
    th_high: float
    th_mod: float
    device: torch.device
    use_heuristics: bool

class RiskTagger:
    def __init__(self, cfg: Optional[RiskConfig] = None) -> None:
        set_seed(settings.seed)
        device = select_device(settings.device_preference)
        label_list = [l.strip() for l in settings.risk_labels_csv.split(",") if l.strip()]
        self.cfg = cfg or RiskConfig(
            model_name=settings.risk_nli_model,
            labels=label_list or ["low risk", "moderate risk", "high risk"],
            th_high=float(settings.risk_threshold_high),
            th_mod=float(settings.risk_threshold_moderate),
            device=device,
            use_heuristics=bool(settings.risk_heuristics_enabled),
        )
        try:
            self.tokenizer, self.model = load_nli(self.cfg.model_name, self.cfg.device)
            logger.info("Loaded NLI model: %s on %s", self.cfg.model_name, self.cfg.device)
            self.nli_ok = True
        except Exception as e:
            logger.warning("RiskTagger falling back to heuristics-only: %s", e)
            self.tokenizer, self.model = None, None
            self.nli_ok = False

    def _heuristic_score(self, text: str) -> Dict[str, float]:
        text_l = text.lower()
        high = any(re.search(p, text_l) for p in HEURISTIC_PATTERNS["high"])
        moderate = any(re.search(p, text_l) for p in HEURISTIC_PATTERNS["moderate"])
        if high:
            return {"high risk": 0.85, "moderate risk": 0.1, "low risk": 0.05}
        if moderate:
            return {"high risk": 0.2, "moderate risk": 0.6, "low risk": 0.2}
        return {"high risk": 0.05, "moderate risk": 0.25, "low risk": 0.70}

    @torch.inference_mode()
    def tag(self, text: str) -> Tuple[str, Dict[str, float], Dict[str, str]]:
        base_probs = self._heuristic_score(text) if self.cfg.use_heuristics else None

        if self.nli_ok:
            # entailment scoring for each label using hypothesis templates
            probs = {}
            for label in self.cfg.labels:
                premise = text[:2000]  # keep it short for stability
                hypothesis = f"The clinical case is {label}."
                inputs = self.tokenizer(
                    premise, hypothesis, return_tensors="pt", truncation=True
                ).to(self.cfg.device)
                logits = self.model(**inputs).logits[0]
                # BART MNLI label mapping: [contradiction, neutral, entailment]
                # We want P(entailment)
                entail = torch.softmax(logits, dim=-1)[2].item()
                probs[label] = float(entail)

            # normalize
            s = sum(probs.values()) or 1.0
            probs = {k: v / s for k, v in probs.items()}
        else:
            probs = base_probs or {"low risk": 1.0}

        # fuse heuristics if both available
        if base_probs and self.nli_ok:
            probs = {k: 0.7 * probs.get(k, 0.0) + 0.3 * base_probs.get(k, 0.0) for k in set(probs) | set(base_probs)}

        # choose label w/ thresholds
        top_label = max(probs.items(), key=lambda kv: kv[1])[0]
        if probs.get("high risk", 0.0) >= self.cfg.th_high:
            top_label = "high risk"
        elif probs.get("moderate risk", 0.0) >= self.cfg.th_mod and top_label != "high risk":
            top_label = "moderate risk"
        else:
            top_label = top_label  # may be “low risk”

        meta = {
            "model": self.cfg.model_name if self.nli_ok else "heuristics-only",
            "heuristics": str(self.cfg.use_heuristics),
            "labels": ",".join(self.cfg.labels),
        }
        return top_label, probs, meta
