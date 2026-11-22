from pathlib import Path

import streamlit as st

from ui import research, studio

st.set_page_config(page_title="AUREL | Model Intelligence", page_icon="◈", layout="wide")
ROOT = Path(__file__).resolve().parent
st.html(ROOT / "ui" / "style.css")

CHAPTERS = {
    "Research Overview": research.overview,
    "Model Intelligence": research.intelligence,
    "Live Inference Studio": studio.render,
    "Performance Lab": research.performance,
    "Knowledge Transfer Explorer": research.transfer,
    "Research Findings": research.findings,
}
with st.sidebar:
    st.html('<div class="wordmark">◈ AUREL</div><p class="identity">AI Model Intelligence &amp; Research Environment</p>')
    st.divider()
    chapter = st.radio("Research chapters", list(CHAPTERS),
                       format_func=lambda name: f"{list(CHAPTERS).index(name)+1:02}   {name}")
    st.divider()
    st.caption("Study index  /  CIFAR-10")
    st.caption("Knowledge Distillation — Teacher and Student Model Evaluation")

st.html(f'<div class="folio">AUREL &nbsp;/&nbsp; RESEARCH SERIES <span>CHAPTER {list(CHAPTERS).index(chapter)+1:02} OF 06</span></div>')
CHAPTERS[chapter]()
