"""Run Pleural Effusion inference with the NIH real-data pilot checkpoint."""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


class ResNet50Binary(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet50(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.classifier(self.backbone(x))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path(__file__).with_name("best_checkpoint.pt"))
    args = parser.parse_args()

    model = ResNet50Binary()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    with Image.open(args.image) as image:
        tensor = transform(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probability = torch.sigmoid(model(tensor)).item()
    prediction = "Effusion" if probability >= 0.5 else "No Effusion"
    print(f"prediction={prediction}")
    print(f"effusion_probability={probability:.6f}")
    print("warning=200-image research pilot; not for clinical use")


if __name__ == "__main__":
    main()
