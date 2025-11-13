from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field, constr

ReportType = Literal["radiology", "pathology", "discharge", "ecg", "echo", "others"]
RiskLevel = Literal["low risk", "moderate risk", "high risk"]


class IngestRequest(BaseModel):
    text: constr(strip_whitespace=True, min_length=3) = Field(..., description="Raw clinical text")
    report_type: ReportType = Field(..., description="Type of report (validated)")


class IngestResponse(BaseModel):
    ok: bool
    char_count: int
    report_type: ReportType
    language: Optional[str]
    anonymized: str
    sentences: List[str]
    meta: Dict[str, str] = {}


class AnalyzeRequest(IngestRequest):
    """Same payload as ingest; we just run the full pipeline."""
    pass


class AnalyzeResponse(BaseModel):
    ok: bool = True
    report_type: ReportType
    summary: str
    risk_level: RiskLevel
    risk_probs: Dict[str, float]
    sentences: List[str]
    anonymized: str
    meta: Dict[str, str] = {}
