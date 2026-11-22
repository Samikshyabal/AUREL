import base64
import csv
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from ui.figures import COLORS, MODELS, SYMBOLS, show

ROOT = Path(__file__).resolve().parents[1]
TEST_CSV = ROOT / "03_Results_Charts" / "accuracy_test (1).csv"


def overview():
    st.title("What does the teacher transfer to a smaller model?")
    st.html('<p class="lead">Two students share an architecture. One learns from labels alone; the other also learns from the teacher’s distribution over all ten classes.</p>')
    diagram = base64.b64encode(Path(__file__).with_name("pipeline.svg").read_bytes()).decode("ascii")
    st.html('<div class="plate"><div class="figure-label">FIG. 01 &nbsp; / &nbsp; TWO PATHS TO A STUDENT</div>'
            f'<img class="pipeline" src="data:image/svg+xml;base64,{diagram}" alt="Teacher soft targets flow only to the distilled student. Hard labels flow to both students.">'
            '<div class="small-flow"><b>Teacher · ResNet-18</b><br>↓ soft targets at T = 4<br>'
            '<b>Distilled student</b> ← hard labels<hr><b>Baseline student</b> ← hard labels only</div></div>')
    st.caption("Training paths · Both students see the same labeled images. The teacher is frozen during distillation; its predictions are not used by the baseline.")
    st.subheader("The experiment")
    st.html('<div class="band"><strong>CIFAR-10</strong><span>32 × 32 RGB images in ten object classes.</span></div>'
            '<div class="band"><strong>45,000 / 5,000 / 10,000</strong><span>Training / validation / test images. Validation selects checkpoints; the separate test set measures their performance.</span></div>')
    st.caption("Source: Teacher_training.ipynb and Role3_Student_and_Distillation edited.ipynb.")


def intelligence():
    st.title("Model Intelligence")
    st.html('<p class="lead">One reference architecture. One compact architecture. Three distinct sets of trained weights.</p>')
    st.subheader("01 / Teacher architecture")
    st.html('<div class="plate"><div class="figure-label">RESNET-18 · CIFAR-10 STEM</div>'
            '3×3 convolution, stride 1 → BatchNorm + ReLU → four residual stages → global average pool → 10 logits</div>')
    st.write("The original ImageNet stem is replaced: no stride-2 convolution and no initial max-pool. Residual stages use 64, 128, 256 and 512 channels.")
    st.subheader("02 / Shared student architecture")
    st.html('<div class="plate"><div class="figure-label">STUDENTCNN · BASELINE AND DISTILLED</div>'
            '3×3 convolution blocks (32 → 64 → 64 channels) → adaptive average pool (4×4) → 1,024 → 128 → 10 logits</div>')
    st.write("Each block uses batch normalization and ReLU. Only the first two use max-pooling. The classifier uses dropout 0.2 during training, disabled for inference.")
    st.table([{"Model": m, "Parameters": f"{p:,}", "Supervision": s} for m, p, s in zip(
        MODELS, (11173962, 188970, 188970), ("Hard labels", "Hard labels", "Hard labels + soft targets"))])
    st.caption("Source: original teacher and Role 3 training definitions; counts checked against strict-loaded checkpoints.")


def recorded_history():
    st.subheader("Recorded epoch history")
    measure = st.radio("Epoch measurement", ["Training accuracy", "Validation accuracy",
                                             "Training loss", "Validation loss"], horizontal=True)
    with TEST_CSV.with_name("recorded_epochs.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    accuracy = measure.endswith("accuracy")
    unit = "% correct" if accuracy else "Cross-entropy loss"
    fig = go.Figure()
    for model, color, symbol in zip(MODELS, COLORS, SYMBOLS):
        observed = [r for r in rows if r["Model"] == model and r[measure]]
        if observed:
            fig.add_scatter(x=[int(r["Epoch"]) for r in observed],
                            y=[float(r[measure]) for r in observed], name=model,
                            mode="markers", marker=dict(color=color, symbol=symbol, size=10))
    fig.update_yaxes(title=unit, range=[0, 100] if accuracy else None, rangemode="tozero")
    fig.update_xaxes(tickmode="array", tickvals=[1, 5, 10, 15, 20])
    show(fig, "Epoch", 350, [0.5, 20.5])
    st.caption("Source: saved epoch outputs, Teacher_training (cell 20) and Role 3 (cells 13 and 19; one-based). "
               "Markers only: no interpolated epochs. Student training losses and distilled validation losses were not printed.")
    st.write("Best validation is a separate checkpoint-selection statistic. The baseline summary reports 80.84%, but does not identify its epoch.")
    with st.expander("View recorded observations"):
        st.table([{"Model": r["Model"], "Epoch": r["Epoch"], f"{measure} ({unit})": r[measure] or "Not logged"} for r in rows])


def performance():
    st.title("Performance Lab")
    st.write("Compare like with like. Each figure uses one metric, one unit and one source.")
    options = {
        "Test accuracy": ("Accuracy (%)", "% correct", ".2f"),
        "Test loss": ("Test loss", "Cross-entropy loss", ".4f"),
        "Best validation accuracy": (None, "% correct", ".2f"),
        "Parameters": ("Params (learnable)", "Parameters (count)", ",.0f"),
        "Recorded latency": ("Latency (ms)", "Milliseconds per image", ".3f"),
    }
    metric = st.selectbox("Comparison metric", [*options, "Recorded epoch history"])
    if metric == "Recorded epoch history":
        recorded_history()
        return
    field, unit, fmt = options[metric]
    if field is not None:
        with TEST_CSV.open(encoding="utf-8-sig", newline="") as stream:
            rows = {r["Model"]: r for r in csv.DictReader(stream)}
        values = [float(rows[m][field]) for m in ("Teacher", "Baseline", "KD")]
        note = "Source: accuracy_test (1).csv · uploaded comparative evaluation."
    else:
        values = [90.28, 80.84, 81.66]
        note = "Source: best validation outputs, teacher and Role 3 training notebooks · 5,000 validation images; not test accuracy."
    st.subheader(metric)
    fig = go.Figure(go.Bar(x=values, y=MODELS, orientation="h", marker_color=COLORS,
                           text=[format(v, fmt) for v in values], textposition="auto",
                           insidetextfont_color="#171B1A", outsidetextfont_color="#F1EEE6"))
    fig.update_yaxes(autorange="reversed")
    show(fig, unit, 290, [0, 100 if "%" in unit else max(values)*1.15])
    st.caption(note)
    if metric == "Recorded latency":
        st.write("Historical batch-one forward-pass times: 20 warm-ups, then 100 timed passes. "
                 "The CSV does not identify its device. Other saved runs give different times, "
                 "so these measurements do not establish which student is faster in deployment.")
    elif metric.startswith("Test"):
        st.write("Official test split · 10,000 images. The comparative evaluation reports the teacher at 90.38%; its original notebook reports 90.30%.")
    with st.expander("View figure data"):
        st.table([{"Model": m, unit: format(v, fmt)} for m, v in zip(MODELS, values)])


def transfer():
    st.title("Knowledge Transfer Explorer")
    st.html('<p class="lead">A hard label names one class. A soft target describes how the teacher distributes probability across all ten.</p>')
    st.subheader("What the student minimizes")
    st.latex(r"\mathcal L=(1-\alpha)\,\mathrm{CE}(z_s,y)+\alpha T^2\,\mathrm{KL}(p_t^T\,\|\,p_s^T)")
    st.latex(r"p^T=\mathrm{softmax}(z/T)")
    st.html('<div class="band"><strong>T = 4</strong><span>Temperature divides teacher and student logits before softmax. Higher temperature makes the distribution less concentrated.</span></div>'
            '<div class="band"><strong>α = 0.7</strong><span>The trained objective assigns weight 0.7 to the scaled KL term and 0.3 to ordinary cross-entropy against the hard label.</span></div>')
    st.subheader("Inspect the two signals")
    signal = st.radio("Training signal", ["Soft targets", "Hard targets"], horizontal=True)
    if signal == "Soft targets":
        st.write("For each labeled training image, the frozen teacher supplies a distribution. KL divergence penalizes the student for departing from it, including on classes other than the correct class.")
    else:
        st.write("The ground-truth class supplies the label. Cross-entropy rewards probability assigned to that class. Both students receive this signal; the baseline receives only this signal.")
    st.caption("Source: executed kd_loss implementation in Role 3. Its Markdown equation reverses the KL arguments; "
               "the code uses teacher targets and student log-probabilities. These are trained settings, not a measured sweep.")


def findings():
    st.title("Research Findings")
    st.html('<div class="insight">Distillation led on validation, but not on the held-out test set.</div>')
    st.subheader("Read the two splits separately")
    st.table([
        {"Split": "Best validation", "Baseline": "80.84%", "Distilled": "81.66%", "KD − baseline": "+0.82 pp"},
        {"Split": "Test", "Baseline": "81.52%", "Distilled": "80.88%", "KD − baseline": "−0.64 pp"},
    ])
    st.write("The student architecture uses about 59.1× fewer parameters than the teacher. That reduction applies to both students; it is not a benefit unique to distillation.")
    st.subheader("Two teacher test records")
    st.html('<div class="band"><strong>90.30% · loss 0.2917</strong><span>Original teacher training notebook, after reloading the best-validation weights.</span></div>'
            '<div class="band"><strong>90.38% · loss 0.2920</strong><span>Role 3 comparison and accuracy-test CSV. The cause of the discrepancy has not been established.</span></div>')
    st.subheader("Limits of the evidence")
    st.write("These student results come from one training seed. They do not establish a general advantage or a statistically significant difference. Placeholder sweeps and results from a mismatched student pooling layer are excluded.")
    st.caption("Sources: Teacher_training.ipynb · Role3_Student_and_Distillation edited.ipynb · accuracy_test (1).csv. pp = percentage points.")
