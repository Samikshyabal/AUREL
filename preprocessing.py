"""Convert one uploaded image into a CIFAR-10 evaluation batch."""

from pathlib import Path
from typing import BinaryIO

from PIL import Image
from torch import Tensor
from torchvision.transforms.functional import normalize, to_tensor

ImageSource = Image.Image | str | Path | BinaryIO
CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


def preprocess_image(source: ImageSource) -> Tensor:
    """Return float32 [1, 3, 32, 32]; no training-time crops or flips."""
    if isinstance(source, Image.Image):
        rgb = source.convert("RGB")
    else:
        with Image.open(source) as image:
            rgb = image.convert("RGB")
    if rgb.size != (32, 32):
        # Upload adaptation only: native CIFAR-10 images are never resampled.
        rgb = rgb.resize((32, 32), Image.Resampling.BILINEAR)
    return normalize(
        to_tensor(rgb), (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    ).unsqueeze(0)
