import plotly.graph_objects as go
import streamlit as st

MODELS = ("Teacher", "Baseline", "Distilled")
COLORS = ("#91ACCB", "#B5C49F", "#73B8AC")
SYMBOLS = ("circle", "square", "diamond")


def show(fig: go.Figure, x_label: str, height: int, x_range=None) -> None:
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#171B1A", plot_bgcolor="#171B1A",
        height=height, margin=dict(l=8, r=18, t=24, b=12),
        font=dict(family="Arial, sans-serif", size=16, color="#F1EEE6"),
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0),
        hoverlabel=dict(font_size=16),
    )
    fig.update_xaxes(title=x_label, range=x_range, gridcolor="#3C4541", zeroline=False)
    fig.update_yaxes(automargin=True, showgrid=False)
    st.plotly_chart(fig, theme=None, width="stretch", config={"displayModeBar": False})
