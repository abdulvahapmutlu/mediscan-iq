from fastapi import FastAPI, HTTPException
from langdetect import detect, DetectorFactory

from .config import settings
from .logging import get_logger
from .schemas import (
    IngestRequest,
    IngestResponse,
    AnalyzeRequest,
    AnalyzeResponse,
)
from .preprocess.anonymizer import anonymize
from .preprocess.segmenter import split_sentences
from .services.analyze import Analyzer

DetectorFactory.seed = 42

logger = get_logger("mediscan.api")
app = FastAPI(title="MediScan-IQ — Ingest, Summarize, Risk Tag")

# Initialize heavy components once
analyzer = Analyzer()


@app.get("/health")
def health():
    return {"status": "ok", "version": "0.2.0"}


@app.post("/ingest", response_model=IngestResponse)
def ingest(payload: IngestRequest):
    text = payload.text.strip()
    if len(text) > settings.max_chars:
        raise HTTPException(
            status_code=413,
            detail=f"Text too long ({len(text)} chars). Max allowed is {settings.max_chars}.",
        )

    if payload.report_type not in settings.accepted_report_types:
        raise HTTPException(status_code=400, detail="Unsupported report_type.")

    try:
        lang = detect(text)
    except Exception:
        lang = None

    anon, counts = anonymize(text)
    sents = split_sentences(anon)

    meta = {f"phi_{k}": str(v) for k, v in counts.items()}
    logger.info(
        "Ingested report | type=%s chars=%d lang=%s sentences=%d",
        payload.report_type,
        len(text),
        lang,
        len(sents),
    )

    return IngestResponse(
        ok=True,
        char_count=len(text),
        report_type=payload.report_type,
        language=lang,
        anonymized=anon,
        sentences=sents,
        meta=meta,
    )


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(payload: AnalyzeRequest):
    text = payload.text.strip()
    if len(text) > settings.max_chars:
        raise HTTPException(
            status_code=413,
            detail=f"Text too long ({len(text)} chars). Max allowed is {settings.max_chars}.",
        )
    if payload.report_type not in settings.accepted_report_types:
        raise HTTPException(status_code=400, detail="Unsupported report_type.")

    out = analyzer.run(text, report_type=payload.report_type)

    logger.info(
        "Analyzed report | type=%s len=%d sents=%d risk=%s",
        payload.report_type,
        len(out.anonymized),
        len(out.sentences),
        out.risk_level,
    )

    return AnalyzeResponse(
        ok=True,
        report_type=payload.report_type,
        summary=out.summary,
        risk_level=out.risk_level,
        risk_probs=out.risk_probs,
        sentences=out.sentences,
        anonymized=out.anonymized,
        meta=out.meta,
    )
