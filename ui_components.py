"""Reusable presentation components for the Streamlit interface."""

import html

import altair as alt
import pandas as pd
import streamlit as st


PALETTE = {
    "blue": "#486f8f",
    "powder": "#a9c8dc",
    "peach": "#e9b8a4",
    "ink": "#1d3343",
    "muted": "#647783",
    "surface": "#f8fbfc",
}


def inject_styles():
    st.markdown(
        """
        <style>
        :root { --ink:#1d3343; --muted:#647783; --blue:#486f8f; --powder:#a9c8dc; --peach:#e9b8a4; }
        .stApp { background:#f7f7f3; color:var(--ink); }
        html, body, [class*="css"] { font-family:Inter,Aptos,"Segoe UI","Helvetica Neue",Arial,sans-serif; }
        h1,h2,h3 { font-family:Inter,Aptos,"Segoe UI","Helvetica Neue",Arial,sans-serif !important; letter-spacing:-.025em; color:var(--ink) !important; font-weight:560 !important; }
        h1 { font-size:2rem !important; line-height:1.18 !important; margin-bottom:.55rem !important; }
        h2 { font-size:1.32rem !important; margin-top:.15rem !important; }
        h3 { font-size:1.02rem !important; }
        [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] { display:none; }
        [data-testid="stHeader"] { background:rgba(247,247,243,.92); }
        .block-container { max-width:1060px; padding-top:6.35rem; padding-bottom:6rem; }
        .product-header { display:flex; align-items:flex-end; justify-content:space-between; gap:1.5rem; padding-bottom:1rem; border-bottom:1px solid #dde5e7; margin-bottom:2.2rem; }
        .product-name { font-size:1.05rem; font-weight:580; letter-spacing:.005em; color:#29495d; }
        .product-descriptor { color:#7a8b94; font-size:.66rem; letter-spacing:.105em; text-transform:uppercase; margin-top:.2rem; }
        .overview-aside { border-left:1px solid #d6e2e7; padding:1rem 0 1rem 1.5rem; color:#627783; }
        .overview-aside strong { color:#294b60; font-weight:560; }
        .overview-trace { width:100%; height:48px; margin-top:1.1rem; opacity:.72; }
        div[data-testid="stSegmentedControl"] { margin:0 0 2.8rem auto; width:fit-content; }
        div[data-testid="stSegmentedControl"] button { border:0 !important; border-bottom:1px solid transparent !important; border-radius:0 !important; background:transparent !important; padding:.35rem .15rem .5rem !important; margin-left:1.35rem; color:#6b7e89 !important; min-height:auto; }
        div[data-testid="stSegmentedControl"] button[aria-pressed="true"] { color:#294c62 !important; border-bottom-color:#6288a1 !important; }
        .eyebrow { color:#718793; font-size:.66rem; font-weight:650; letter-spacing:.15em; text-transform:uppercase; margin-bottom:.65rem; }
        .lede { color:var(--muted); font-size:1rem; line-height:1.68; max-width:720px; }
        .surface { background:transparent; border-top:1px solid #d9e4e7; border-bottom:1px solid #d9e4e7; padding:1.25rem 0; border-radius:0; }
        .band-card { background:transparent; border-top:1px solid #c9dbe3; padding:.85rem .15rem; min-height:100px; }
        .band-card strong { display:block; font:600 1rem 'Manrope'; color:var(--ink); }
        .band-card span { color:var(--muted); font-size:.82rem; line-height:1.45; }
        .workflow { display:flex; gap:.65rem; align-items:center; flex-wrap:wrap; margin:1.25rem 0; }
        .workflow-step { background:transparent; border:0; padding:.55rem .2rem; color:#37586e; font-weight:600; }
        .workflow-arrow { color:#91a7b4; }
        .notice { background:#fffaf6; border-left:3px solid var(--peach); padding:.9rem 1rem; color:#6c5a52; font-size:.88rem; line-height:1.5; }
        .empty-state { background:#f7fafb; border:1px dashed #c8d9e1; padding:1rem 1.1rem; color:#667c89; font-size:.88rem; line-height:1.55; }
        .error-state { background:#fff7f3; border-left:3px solid #dba58e; padding:.9rem 1rem; color:#6e5147; font-size:.88rem; line-height:1.55; }
        .editorial-rule { border:0; border-top:1px solid #dbe4e7; margin:4rem 0; }
        .editorial-copy { color:#526a78; font-size:1rem; line-height:1.75; max-width:780px; }
        .science-grid { display:grid; grid-template-columns:repeat(3,1fr); column-gap:2rem; border-top:1px solid #d7e4e9; }
        .science-item { padding:1.35rem 0; border-bottom:1px solid #d7e4e9; background:transparent; min-height:145px; }
        .science-item strong { display:block; color:var(--ink); font:580 .94rem Inter,Aptos,"Segoe UI",sans-serif; margin-bottom:.45rem; }
        .science-item span { color:var(--muted); font-size:.87rem; line-height:1.55; }
        .band-table { width:100%; border-collapse:collapse; background:transparent; }
        .band-table th { color:#6b8596; font-size:.7rem; letter-spacing:.12em; text-transform:uppercase; font-weight:700; text-align:left; padding:.8rem 1rem; border-bottom:1px solid #ccdce3; }
        .band-table td { padding:.9rem 1rem; border-bottom:1px solid #e1eaee; color:#526a78; font-size:.9rem; }
        .band-table td:first-child { color:var(--ink); font-weight:580; }
        .vertical-flow { margin:.5rem 0; }
        .vertical-step { display:grid; grid-template-columns:36px 1fr; gap:.8rem; align-items:center; background:transparent; border-bottom:1px solid #dbe5e8; padding:.8rem 0; }
        .vertical-step .num { color:#7895a6; font:580 .72rem Inter,Aptos,"Segoe UI",sans-serif; }
        .vertical-step strong { color:var(--ink); font:580 .9rem Inter,Aptos,"Segoe UI",sans-serif; }
        .vertical-arrow { color:#94aab6; margin-left:17px; height:22px; line-height:22px; }
        .assessment-list { display:grid; grid-template-columns:repeat(2,1fr); gap:0; border-top:1px solid #dae6eb; }
        .assessment-item { padding:.9rem 0; border-bottom:1px solid #dae6eb; color:#405b6b; }
        .assessment-item:nth-child(odd) { padding-right:1rem; }
        .assessment-item:nth-child(even) { padding-left:1rem; border-left:1px solid #dae6eb; }
        .peach-panel { background:#fff5ef; border:1px solid #f0d4c8; padding:1.35rem 1.5rem; color:#654f47; line-height:1.65; }
        .peach-panel .eyebrow { color:#9b6f5e; }
        .source-note { color:#718591; font-size:.78rem; line-height:1.6; }
        .source-note a { color:#486f8f; text-decoration:none; }
        .status-badge { display:inline-block; background:transparent; border:0; border-left:2px solid #9bb9c8; color:#587384; padding:.2rem 0 .2rem .65rem; font-size:.68rem; font-weight:650; letter-spacing:.11em; text-transform:uppercase; }
        .docs-block { border-left:2px solid var(--powder); padding-left:1.2rem; color:#526a78; line-height:1.7; }
        .pipeline-strip { display:flex; align-items:stretch; flex-wrap:wrap; gap:.4rem; margin:1rem 0; }
        .pipeline-node { display:flex; align-items:center; background:transparent; border-bottom:1px solid #bcd0d9; padding:.65rem .2rem; color:#38596d; font:580 .8rem Inter,Aptos,"Segoe UI",sans-serif; }
        .pipeline-link { align-self:center; color:#8da4b1; }
        .metadata-grid { display:grid; grid-template-columns:repeat(2,1fr); border-top:1px solid #d9e5ea; }
        .metadata-row { padding:.9rem 0; border-bottom:1px solid #d9e5ea; }
        .metadata-row:nth-child(odd) { padding-right:1rem; }
        .metadata-row:nth-child(even) { padding-left:1rem; border-left:1px solid #d9e5ea; }
        .metadata-label { color:#748995; font-size:.7rem; letter-spacing:.1em; text-transform:uppercase; }
        .metadata-value { color:var(--ink); font:580 .9rem Inter,Aptos,"Segoe UI",sans-serif; margin-top:.25rem; overflow-wrap:anywhere; }
        .feature-list { display:grid; grid-template-columns:repeat(4,1fr); column-gap:1.2rem; margin-top:.8rem; border-top:1px solid #dce5e8; }
        .feature-chip { background:transparent; border:0; border-bottom:1px solid #dce5e8; color:#526a78; padding:.52rem 0; font:500 .72rem Inter,Aptos,"Segoe UI",sans-serif; overflow-wrap:anywhere; }
        .limitations-panel { background:#fff6f0; border-left:3px solid var(--peach); padding:1.25rem 1.4rem; }
        .limitations-panel ul { color:#66564f; line-height:1.65; margin-bottom:0; }
        @media (max-width:800px) { .science-grid { grid-template-columns:1fr; } .assessment-list { grid-template-columns:1fr; } .assessment-item:nth-child(even) { padding-left:0; border-left:0; } }
        @media (max-width:800px) { .metadata-grid { grid-template-columns:1fr; } .metadata-row:nth-child(even) { padding-left:0; border-left:0; } .feature-list { grid-template-columns:repeat(2,1fr); } }
        .prob-wrap { background:transparent; border:0; border-top:1px solid #d8e3e7; border-bottom:1px solid #d8e3e7; padding:1.05rem 0; }
        .prob-label { display:flex; justify-content:space-between; color:var(--ink); font-weight:600; }
        .prob-track { height:5px; background:#e3eaed; margin-top:.7rem; overflow:hidden; }
        .prob-fill { height:100%; background:#5f8298; }
        .result-hero { padding:2.8rem 0 1.35rem; border-top:1px solid #d8e5ea; }
        .result-value { color:var(--ink); font:580 3.15rem/1 Inter,Aptos,"Segoe UI",sans-serif; letter-spacing:-.05em; margin:.4rem 0 .5rem; }
        .result-name { color:#35566b; font:560 1.15rem Inter,Aptos,"Segoe UI",sans-serif; margin-bottom:.5rem; }
        .result-caption { color:var(--muted); font-size:.92rem; line-height:1.55; }
        .prob-scale { display:flex; justify-content:space-between; color:#8296a1; font-size:.7rem; margin-top:.4rem; }
        .result-section { margin-top:3.3rem; }
        .assistant-cta { background:#edf5f8; border-left:3px solid var(--powder); padding:1rem 1.15rem; color:#526a78; margin-top:.75rem; }
        div.stButton > button { border-radius:3px; font-weight:560; min-height:2.35rem; padding:.35rem .85rem; }
        div.stButton > button[kind="primary"] { background:#496d84; border-color:#496d84; border-radius:3px; font-weight:560; }
        div.stButton > button[kind="secondary"] { border-color:#c9d9e1; color:#456579; background:rgba(255,255,255,.68); }
        div[data-testid="stFileUploader"] { background:#f9fbfc; border:1px dashed #b9ccd6; padding:.4rem; }
        [data-testid="stMetric"] { background:transparent; border:0; border-top:1px solid #dce5e8; padding:.8rem 0; border-radius:0; }
        [data-testid="stMetricValue"] { color:var(--ink); font-family:Inter,Aptos,"Segoe UI",sans-serif; font-size:1.4rem; }
        [data-testid="stAlert"] { border-radius:2px; box-shadow:none; }
        [data-testid="stExpander"] { background:transparent; border-color:#d8e5ea; border-radius:2px; }
        [data-testid="stDataFrame"] { border:1px solid #dce7eb; border-radius:4px; overflow:hidden; }
        footer { visibility:hidden; }
        @media (max-width:760px) {
            .block-container { padding-top:5.2rem; padding-left:1.15rem; padding-right:1.15rem; }
            .product-header { align-items:flex-start; margin-bottom:1.25rem; }
            div[data-testid="stSegmentedControl"] { width:100%; margin-bottom:2rem; overflow-x:auto; }
            div[data-testid="stSegmentedControl"] button { margin-left:0; margin-right:1rem; font-size:.78rem; white-space:nowrap; }
            h1 { font-size:1.8rem !important; }
            .overview-aside { border-left:0; border-top:1px solid #d6e2e7; padding:1rem 0 0; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_intro(label: str, title: str, description: str):
    st.markdown(f'<div class="eyebrow">{html.escape(label)}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="lede">{html.escape(description)}</div>', unsafe_allow_html=True)


def disclaimer():
    st.markdown(
        '<div class="notice"><strong>Research use only.</strong> This experimental '
        'software does not diagnose Alzheimer\'s disease, estimate an individual\'s '
        'clinical risk, or replace evaluation by a qualified clinician.</div>',
        unsafe_allow_html=True,
    )


def probability_bar(label: str, value: float):
    percent = max(0.0, min(1.0, value)) * 100
    st.markdown(
        f'<div class="prob-wrap"><div class="prob-label"><span>{html.escape(label)}</span>'
        f'<span>{percent:.1f}%</span></div><div class="prob-track">'
        f'<div class="prob-fill" style="width:{percent:.1f}%"></div></div></div>',
        unsafe_allow_html=True,
    )


def research_probability(value: float):
    """Render a neutral, single-score research probability scale."""
    percent = max(0.0, min(1.0, value)) * 100
    st.markdown(
        f'<div class="prob-wrap"><div class="prob-label"><span>Model classification scale</span>'
        f'<span>{percent:.1f}%</span></div><div class="prob-track">'
        f'<div class="prob-fill" style="width:{percent:.1f}%"></div></div>'
        '<div class="prob-scale"><span>0%</span><span>50%</span><span>100%</span></div></div>',
        unsafe_allow_html=True,
    )


def band_chart(bands: dict[str, float]):
    frame = pd.DataFrame({"Band": list(bands), "Power": list(bands.values())})
    chart = (
        alt.Chart(frame)
        .mark_bar(color="#63879c", opacity=0.82, size=34)
        .encode(
            x=alt.X(
                "Band:N",
                sort=list(bands),
                axis=alt.Axis(labelAngle=0, title=None, ticks=False, domain=False),
            ),
            y=alt.Y(
                "Power:Q",
                title="Relative / entered power",
                axis=alt.Axis(tickCount=4, domain=False, tickSize=0),
            ),
            tooltip=["Band", alt.Tooltip("Power", format=".3f")],
        )
        .properties(height=210)
        .configure_view(strokeWidth=0)
        .configure_axis(
            gridColor="#dfe7ea",
            gridOpacity=0.65,
            labelColor="#6c7f89",
            titleColor="#6c7f89",
            labelFont="Inter",
            titleFont="Inter",
            labelFontSize=11,
            titleFontSize=11,
            titleFontWeight="normal",
        )
    )
    st.altair_chart(chart, width="stretch")
