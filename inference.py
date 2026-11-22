"""Load once with load_models(), then call predict_all(models, uploaded_file)."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from models import StudentCNN, make_teacher
from preprocessing import CLASSES, ImageSource, preprocess_image

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
CHECKPOINTS = {
    "teacher": ("teacher_model.pth", make_teacher),
    "baseline": ("student_baseline.pt", StudentCNN),
    "distilled": ("student_kd.pt", StudentCNN),
}


@dataclass(frozen=True)
class Prediction:
    predicted_class: str
    confidence_percent: float
    probabilities: dict[str, float]


def load_model(name: str, device: str | torch.device | None = None,
               checkpoint_dir: Path = CHECKPOINT_DIR) -> nn.Module:
    if name not in CHECKPOINTS:
        raise ValueError(f"Unknown model {name!r}; choose from {tuple(CHECKPOINTS)}")
    target = torch.device(device if device is not None else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable in this PyTorch runtime.")
    filename, build = CHECKPOINTS[name]
    path = checkpoint_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing {name} checkpoint: {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = build()
    try:
        model.load_state_dict(state, strict=True)
    except (TypeError, RuntimeError) as exc:
        raise ValueError(f"Incompatible {name} checkpoint {path}: {exc}") from exc
    return model.to(target).eval()


def load_models(device: str | torch.device | None = None) -> dict[str, nn.Module]:
    return {name: load_model(name, device) for name in CHECKPOINTS}


@torch.inference_mode()
def _predict(model: nn.Module, batch: Tensor) -> Prediction:
    if model.training:
        raise ValueError("Prediction requires a model in evaluation mode.")
    logits = model(batch.to(next(model.parameters()).device))
    if logits.shape != (1, len(CLASSES)) or not torch.isfinite(logits).all():
        raise ValueError("Expected one finite logit for each of the ten CIFAR-10 classes.")
    values = logits.softmax(dim=1)[0].cpu()
    best = int(values.argmax())
    return Prediction(CLASSES[best], 100 * values[best].item(),
                      dict(zip(CLASSES, values.tolist(), strict=True)))


def predict(model: nn.Module, source: ImageSource) -> Prediction:
    return _predict(model, preprocess_image(source))


def predict_all(models: Mapping[str, nn.Module], source: ImageSource) -> dict[str, Prediction]:
    """Use identical pixels for all three models, decoding the upload only once."""
    if models.keys() != CHECKPOINTS.keys():
        raise ValueError(f"Expected models named {tuple(CHECKPOINTS)}.")
    batch = preprocess_image(source)
    return {name: _predict(models[name], batch) for name in CHECKPOINTS}
