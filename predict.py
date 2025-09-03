
"""Predict script for single images."""

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from model import build_resnet18


def load_image(path: Path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0)  # [1, C, H, W]


def main():
    parser = argparse.ArgumentParser(description="Predict lens presence for an image")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--model", type=str, required=True, help="Path to trained .pt model")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_resnet18()
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)
    model.eval()

    x = load_image(Path(args.image)).to(device)
    with torch.no_grad():
        prob_lensed = torch.softmax(model(x), dim=1)[0, 1].item()

    print(f"Probability of *lensed* class: {prob_lensed:.4f}")


if __name__ == "__main__":
    main()
