# app_streamlit.py
from __future__ import annotations

import html
import os
import re
from dataclasses import asdict
from typing import Dict, List, Tuple

import altair as alt
import pandas as pd
import streamlit as st

from mediscan_iq.config import settings
from mediscan_iq.services.analyze import Analyzer
from mediscan_iq.nlp.risk_tagger import HEURISTIC_PATTERNS


# ---------------------------
# Streamlit page config / CSS
# ---------------------------
st.set_page_config(
    page_title="MediScan-IQ — Clinical Report Summarizer & Risk Tagger",
    layout="wide",
    page_icon="🩺",
)

CUSTOM_CSS = """
<style>
.badge {
  display:inline-block; padding:4px 10px; border-radius:999px; font-weight:600; font-size:0.9rem;
}
.badge.low { background:#e6f4ea; color:#116149; border:1px solid #b7e1c2; }
.badge.moderate { background:#fff7e6; color:#8a5a00; border:1px solid #ffe0a3; }
.badge.high { background:#fdeaea; color:#8a1f1f; border:1px solid #f5b5b5; }

.hl-high { background:#fdeaea; padding:0 .15rem; border-radius:.25rem; }
.hl-moderate { background:#fff7e6; padding:0 .15rem; border-radius:.25rem; }

.codeblock {
  background: var(--background-color, #0e1117);
  border-radius: 8px; padding: 12px; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  white-space: pre-wrap; word-break: break-word; border:1px solid #2a2a2a;
}
.smallmeta { font-size: 0.9rem; color: #6b7280; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------
# Caching for speed
# ---------------------------
@st.cache_resource(show_spinner=False)
def get_analyzer() -> Analyzer:
    # Respect device preference if user toggled env before launch
    os.environ.setdefault("DEVICE_PREFERENCE", settings.device_preference)
    return Analyzer()


@st.cache_data(show_spinner=False)
def analyze_cached(text: str, report_type: str) -> Dict:
    """Run analysis and return a plain dict for Streamlit caching."""
    a = get_analyzer()
    out = a.run(text, report_type=report_type)
    d = asdict(out)
    # Prob keys to stable order
    d["risk_probs"] = dict(sorted(d["risk_probs"].items(), key=lambda kv: kv[0]))
    return d


# ---------------------------
# Sample de-identified notes
# ---------------------------
SAMPLES: Dict[str, str] = {
    "CXR — Cardiomegaly (Moderate risk)": """FINDINGS: Mild cardiomegaly. No acute pulmonary edema or pleural effusion.
IMPRESSION: Cardiomegaly without acute cardiopulmonary process.""",
    "Head CT — Subarachnoid Hemorrhage (High risk)": """FINDINGS: Acute subarachnoid hemorrhage within the suprasellar cisterns and anterior interhemispheric fissure.
No midline shift. Ventricles normal in size.
IMPRESSION: Acute subarachnoid hemorrhage. Urgent neurosurgical evaluation recommended.""",
    "CXR — No acute disease (Low risk)": """FINDINGS: Cardiomediastinal silhouette within normal limits. No focal consolidation, effusion, or pneumothorax.
IMPRESSION: No acute cardiopulmonary disease.""",
}


# ---------------------------
# Helpers: risk badge & highlighting
# ---------------------------
def risk_badge(level: str) -> str:
    css = {"low risk": "low", "moderate risk": "moderate", "high risk": "high"}.get(level, "low")
    return f'<span class="badge {css}">{html.escape(level)}</span>'


def _collect_matches(text: str) -> List[Tuple[int, int, str]]:
    """Collect regex spans for heuristic keywords, tagged with risk level."""
    spans: List[Tuple[int, int, str]] = []
    t = text
    for severity, patterns in HEURISTIC_PATTERNS.items():
        for pat in patterns:
            for m in re.finditer(pat, t, flags=re.I):
                spans.append((m.start(), m.end(), severity))
    # sort, and drop overlaps (keep earliest)
    spans.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    non_overlap: List[Tuple[int, int, str]] = []
    last_end = -1
    for s, e, sev in spans:
        if s >= last_end:
            non_overlap.append((s, e, sev))
            last_end = e
    return non_overlap


def highlight_text(text: str) -> str:
    if not text.strip():
        return ""
    spans = _collect_matches(text)
    if not spans:
        return html.escape(text)

    parts: List[str] = []
    cur = 0
    for s, e, sev in spans:
        if s > cur:
            parts.append(html.escape(text[cur:s]))
        frag = html.escape(text[s:e])
        cls = "hl-high" if sev == "high" else "hl-moderate"
        parts.append(f'<span class="{cls}"><b>{frag}</b></span>')
        cur = e
    if cur < len(text):
        parts.append(html.escape(text[cur:]))
    return "".join(parts)


def probs_chart(probs: Dict[str, float]):
    df = pd.DataFrame({"label": list(probs.keys()), "prob": list(probs.values())})
    order = ["low risk", "moderate risk", "high risk"]
    df["label"] = pd.Categorical(df["label"], categories=order, ordered=True)
    c = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("label:N", sort=order, title=None),
            y=alt.Y("prob:Q", title="Probability", scale=alt.Scale(domain=[0, 1])),
            tooltip=["label", alt.Tooltip("prob:Q", format=".3f")],
        )
        .properties(height=200)
    )
    st.altair_chart(c, use_container_width=True)


# ---------------------------
# UI
# ---------------------------
st.title("🩺 MediScan-IQ")
st.caption("Secure ingestion → summarization → risk tagging for clinical text (de-identified).")

with st.sidebar:
    st.header("Settings")
    st.write(f"**Summarizer**: `{settings.summarizer_model}`")
    st.write(f"**NLI Risk Model**: `{settings.risk_nli_model}`")
    st.write(f"**Device**: `{settings.device_preference}`")
    keep_dates = st.toggle("Keep dates (no date redaction)", value=bool(settings.keep_dates), disabled=True)
    st.caption("Env-driven settings (editable via `.env` before launch).")

    st.markdown("---")
    st.subheader("Try a sample")
    sample_key = st.selectbox("Prefill text", ["(none)"] + list(SAMPLES.keys()))
    st.markdown("---")
    st.subheader("Upload file")
    uploaded = st.file_uploader("TXT file with a clinical report", type=["txt"], accept_multiple_files=False)

col1, col2 = st.columns([1.1, 0.9], gap="large")

with col1:
    st.subheader("Input")
    default_text = SAMPLES.get(sample_key, "")
    if uploaded is not None:
        default_text = uploaded.read().decode("utf-8", errors="ignore")
    text = st.text_area(
        "Paste clinical text (de-identified preferred)",
        value=default_text,
        height=220,
        placeholder="FINDINGS: ...\nIMPRESSION: ...",
    )
    report_type = st.selectbox("Report type", options=settings.accepted_report_types, index=0)
    go = st.button("Analyze", type="primary", use_container_width=False)

with col2:
    st.subheader("Output")

    if go:
        if not text.strip():
            st.warning("Please paste or upload a report first.")
        else:
            with st.spinner("Analyzing…"):
                result = analyze_cached(text, report_type=report_type)

            # Summary + risk
            st.markdown("**Summary**")
            st.success(result["summary"])

            st.markdown("**Risk**")
            st.markdown(risk_badge(result["risk_level"]), unsafe_allow_html=True)
            probs_chart(result["risk_probs"])

            # Sentences
            with st.expander("Sentences (anonymized, split)"):
                st.write(result["sentences"])

            # Anonymized text with highlights
            st.markdown("**Anonymized input (keyword highlights)**")
            st.markdown(
                f'<div class="codeblock">{highlight_text(result["anonymized"])}</div>',
                unsafe_allow_html=True,
            )

            # Meta
            with st.expander("Meta"):
                md = "\n".join(f"- **{k}**: {v}" for k, v in result["meta"].items())
                st.markdown(md or "_none_", unsafe_allow_html=False)

    else:
        st.info("Choose a sample or paste your own report, then click **Analyze**.")

st.markdown("---")
st.caption(
    "MediScan-IQ demo • Abstractive summarization + NLI-based risk • Heuristic keyword highlights for transparency."
)
