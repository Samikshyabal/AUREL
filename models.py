"""Architectures from Teacher_training and Role3_Student_and_Distillation edited."""

from torch import Tensor, nn
from torchvision.models import resnet18
from torchvision.models.resnet import ResNet


def make_teacher() -> ResNet:
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, 10)
    return model


class StudentCNN(nn.Module):
    """The baseline and KD checkpoints differ in weights, not architecture."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(1024, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.2), nn.Linear(128, 10),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.features(x))
