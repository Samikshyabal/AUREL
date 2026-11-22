import pickle

import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from inference import load_models, predict_all
from preprocessing import CLASSES
from ui.figures import COLORS, MODELS, SYMBOLS, show

cached_models = st.cache_resource(show_spinner=False)(load_models)


def render():
    st.title("Live Inference Studio")
    st.html('<p class="lead">One image, three readings. Compare what each trained checkpoint predicts from the same pixels.</p>')
    upload = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"],
                              help="PNG, JPEG or WebP · up to 10 MB and 16 megapixels.")
    if upload is None:
        st.caption("Awaiting an image. Predictions and class probabilities appear only after successful inference.")
        return
    try:
        with Image.open(upload) as source:
            if source.format not in {"PNG", "JPEG", "WEBP"}:
                raise ValueError("Choose a PNG, JPEG or WebP image.")
            if source.width * source.height > 16_000_000:
                raise ValueError("Choose an image no larger than 16 megapixels.")
            source.load()
            image = source.copy()
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
        st.error(f"Cannot read this upload. Choose a valid PNG, JPEG or WebP. Details: {exc}")
        return

    with st.spinner("Evaluating the three checkpoints…"):
        try:
            models = cached_models()
            results = predict_all(models, image)
        except (OSError, ValueError, RuntimeError, EOFError, pickle.UnpicklingError) as exc:
            st.error(f"Inference failed. No predictions are available: {exc}")
            return
    preview, comparison = st.columns([1, 1.45], gap="large")
    with preview:
        st.subheader("Input image")
        # Use a bounded preview, not the resized tensor fed to the models.
        thumb = image.convert("RGB")
        thumb.thumbnail((800, 420))
        st.image(thumb, width="stretch")
        st.caption(f"{image.width} × {image.height} px original · 32 × 32 RGB model input")
    with comparison:
        st.subheader("Model predictions")
        for name, color, key in zip(MODELS, COLORS, ("teacher", "baseline", "distilled")):
            prediction = results[key]
            st.html(f'<div class="prediction"><span style="color:{color}">{name}</span>'
                    f'<strong>{prediction.predicted_class}</strong><b>{prediction.confidence_percent:.2f}%</b></div>')
        device = next(models["teacher"].parameters()).device
        st.caption(f"Live on {device} · Confidence is the largest softmax probability, not a guarantee of correctness.")
    st.subheader("Probability across all ten classes")
    fig = go.Figure()
    table = [{"Class": c} for c in CLASSES]
    for key, name, color, symbol in zip(("teacher", "baseline", "distilled"), MODELS, COLORS, SYMBOLS):
        values = [100*results[key].probabilities[c] for c in CLASSES]
        fig.add_scatter(x=values, y=CLASSES, mode="markers", name=name,
                        marker=dict(color=color, symbol=symbol, size=11, opacity=.85),
                        hovertemplate="%{y}: %{x:.2f}%<extra>%{fullData.name}</extra>")
        for row, value in zip(table, values):
            row[name] = f"{value:.2f}%"
    fig.update_yaxes(autorange="reversed")
    show(fig, "Class probability (%)", 470, [-1, 101])
    st.caption("Source: live outputs from the three uploaded project checkpoints. Each model’s ten probabilities total 100% before rounding.")
    with st.expander("View class probabilities"):
        st.table(table)
