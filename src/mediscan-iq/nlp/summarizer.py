from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import math
import re

import torch
from transformers import PreTrainedTokenizerBase, PreTrainedModel
from ..config import settings
from ..logging import get_logger
from .registry import select_device, set_seed, load_seq2seq

logger = get_logger("mediscan.nlp.summarizer")

PROMPTS: Dict[str, str] = {
    "radiology_brief": (
        "You are a senior radiologist.\n"
        "Summarize the FINDINGS/IMPRESSION in 2-3 crisp sentences with clinical precision.\n"
        "Avoid PHI, avoid speculation, no recommendations, just the key findings.\n\n"
        "REPORT:\n{input}\n\nSUMMARY:"
    ),
    "pathology_brief": (
        "You are a senior pathologist. Provide a concise 2-3 sentence summary of key pathology findings.\n"
        "Stick to facts present in the text. Avoid PHI.\n\nREPORT:\n{input}\n\nSUMMARY:"
    ),
    "generic_clinical": (
        "Provide a concise medical summary (2-3 sentences) of the following clinical note.\n"
        "Avoid PHI and avoid recommendations.\n\nNOTE:\n{input}\n\nSUMMARY:"
    ),
}

@dataclass
class SummarizationConfig:
    model_name: str
    max_input_tokens: int
    max_output_tokens: int
    num_beams: int
    temperature: float
    prompt_style: str
    device: torch.device

class Summarizer:
    def __init__(self, cfg: Optional[SummarizationConfig] = None) -> None:
        set_seed(settings.seed)
        device = select_device(settings.device_preference)
        self.cfg = cfg or SummarizationConfig(
            model_name=settings.summarizer_model,
            max_input_tokens=settings.summarizer_max_input_tokens,
            max_output_tokens=settings.summarizer_max_output_tokens,
            num_beams=settings.summarizer_num_beams,
            temperature=settings.summarizer_temperature,
            prompt_style=settings.summarizer_prompt_style,
            device=device,
        )
        try:
            self.tokenizer, self.model = load_seq2seq(self.cfg.model_name, self.cfg.device)
            logger.info("Loaded summarizer: %s on %s", self.cfg.model_name, self.cfg.device)
            self.abstractive_ok = True
        except Exception as e:
            logger.warning("Falling back to extractive summarizer: %s", e)
            self.tokenizer, self.model = None, None
            self.abstractive_ok = False

    def _prompt(self, text: str) -> str:
        tpl = PROMPTS.get(self.cfg.prompt_style, PROMPTS["generic_clinical"])
        return tpl.format(input=text)

    def _truncate_by_tokens(self, text: str) -> str:
        if not self.abstractive_ok:
            return text
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) <= self.cfg.max_input_tokens:
            return text
        # keep head + tail windows
        head = math.floor(self.cfg.max_input_tokens * 0.6)
        tail = self.cfg.max_input_tokens - head
        trunc = tokens[:head] + tokens[-tail:]
        return self.tokenizer.decode(trunc)

    @torch.inference_mode()
    def summarize(self, text: str, report_type: str) -> Tuple[str, Dict[str, str]]:
        clean = text.strip()
        clean = re.sub(r"\s+", " ", clean)
        if len(clean) < 20:
            return clean, {"mode": "passthrough", "reason": "short_text"}

        if self.abstractive_ok:
            prompt = self._prompt(clean)
            trunc = self._truncate_by_tokens(prompt)
            inputs = self.tokenizer(trunc, return_tensors="pt", truncation=True).to(self.cfg.device)

            gen = self.model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_output_tokens,
                num_beams=self.cfg.num_beams,
                do_sample=self.cfg.temperature > 0.0,
                temperature=max(1e-6, self.cfg.temperature),
                length_penalty=1.0,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )
            out = self.tokenizer.decode(gen[0], skip_special_tokens=True).strip()
            out = _post_summarize(out)
            return out, {
                "mode": "abstractive",
                "model": self.cfg.model_name,
                "report_type": report_type,
            }

        # Fallback: simple extractive scoring (position + length + keyword boosts)
        out = extractive_summary(clean)
        return out, {"mode": "extractive", "report_type": report_type, "model": "fallback"}

def _post_summarize(text: str) -> str:
    # Normalize whitespace and cut trailing labels
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^(SUMMARY|FINDINGS|IMPRESSION)[:\-]\s*", "", text, flags=re.I)
    # Ensure 2-3 sentences tops
    sents = re.split(r"(?<=[.!?])\s+", text)
    sents = [s.strip() for s in sents if s.strip()]
    return " ".join(sents[:3])

KEY_BOOSTS = {
    "pneumonia": 1.6, "hemorrhage": 1.8, "malignant": 1.7, "mass": 1.5,
    "effusion": 1.4, "cardiomegaly": 1.4, "consolidation": 1.5,
    "fracture": 1.6, "ischemia": 1.7, "embolism": 1.8,
}

def extractive_summary(text: str, max_sents: int = 3) -> str:
    # Naive extractor: split sentences, score by tf-ish + keyword boosts + position decay
    sents = re.split(r"(?<=[.!?])\s+", text)
    sents = [s.strip() for s in sents if len(s.strip()) > 3]
    if not sents:
        return text

    def score(idx: int, s: str) -> float:
        base = math.log(len(s.split()) + 1)
        pos = 1.0 / (1.0 + idx * 0.15)
        boost = 1.0 + sum(wt for k, wt in KEY_BOOSTS.items() if k in s.lower()) * 0.15
        return base * pos * boost

    scored = sorted(((score(i, s), s) for i, s in enumerate(sents)), reverse=True)
    top = [s for _, s in scored[:max_sents]]
    return " ".join(top)
