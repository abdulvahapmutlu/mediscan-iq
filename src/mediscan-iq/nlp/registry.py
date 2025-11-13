from __future__ import annotations
import os
import random
import torch
import numpy as np
from typing import Literal
from ..config import settings
from ..logging import get_logger

logger = get_logger("mediscan.nlp.registry")

def select_device(pref: Literal["auto", "cpu", "cuda", "mps"] = "auto") -> torch.device:
    pref = (pref or "auto").lower()
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if pref == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Small HF loader helpers with safe failover
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification

def load_seq2seq(model_name: str, device: torch.device):
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    mdl.to(device)
    mdl.eval()
    return tok, mdl

def load_nli(model_name: str, device: torch.device):
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_name)
    mdl.to(device)
    mdl.eval()
    return tok, mdl
