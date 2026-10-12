"""Run a small Grad-CAM proof of concept with torchvision's ViT-B/16."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision import transforms
from torchvision.models import ViT_B_16_Weights, vit_b_16

from gradcam_vit import ViTGradCAM, torchvision_vit_b_16_target_layer


IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, help="Input image; omit for a generated demo image")
    parser.add_argument("--output", type=Path, default=Path("gradcam_overlay.png"))
    parser.add_argument(
        "--weights",
        choices=("default", "none"),
        default="default",
        help="Use pretrained ImageNet weights (downloaded once) or random weights",
    )
    parser.add_argument("--target-class", type=int, help="ImageNet class index; default: top prediction")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def demo_image() -> Image.Image:
    image = Image.new("RGB", (360, 280), "#d9e8ef")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((60, 65, 300, 230), radius=28, fill="#e28a38", outline="#713f18", width=6)
    draw.ellipse((135, 90, 225, 180), fill="#f7cf72", outline="#713f18", width=5)
    draw.ellipse((158, 116, 172, 130), fill="#1e293b")
    draw.ellipse((190, 116, 204, 130), fill="#1e293b")
    draw.arc((158, 122, 204, 154), 10, 170, fill="#713f18", width=4)
    return image


def prepare_image(image: Image.Image) -> tuple[Image.Image, torch.Tensor]:
    crop = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(IMAGE_SIZE),
        ]
    )(image.convert("RGB"))
    tensor = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )(crop)
    return crop, tensor.unsqueeze(0)


def colorize(cam: np.ndarray) -> np.ndarray:
    """Small dependency-free approximation of the Turbo/Jet color scale."""

    red = np.clip(1.5 - np.abs(4.0 * cam - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * cam - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * cam - 1.0), 0.0, 1.0)
    return np.stack((red, green, blue), axis=-1)


def overlay(image: Image.Image, cam: torch.Tensor, alpha: float = 0.45) -> Image.Image:
    base = np.asarray(image, dtype=np.float32) / 255.0
    heatmap = colorize(cam.detach().cpu().numpy())
    blended = np.clip((1.0 - alpha) * base + alpha * heatmap, 0.0, 1.0)
    return Image.fromarray((blended * 255).astype(np.uint8))


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    weights = ViT_B_16_Weights.DEFAULT if args.weights == "default" else None
    if weights is None:
        torch.manual_seed(0)
    model = vit_b_16(weights=weights).to(device).eval()
    if weights is None:
        # torchvision deliberately zero-initializes the ViT classifier head.
        # Randomize it for an offline gradient-flow smoke test.
        torch.nn.init.trunc_normal_(model.heads.head.weight, std=0.02)

    source = Image.open(args.image) if args.image else demo_image()
    display_image, input_tensor = prepare_image(source)
    input_tensor = input_tensor.to(device)

    target_layer = torchvision_vit_b_16_target_layer(model)
    with ViTGradCAM(model, target_layer) as gradcam:
        cams, logits, targets = gradcam(input_tensor, args.target_class)

    probabilities = logits.softmax(dim=1)
    target_index = targets.item()
    confidence = probabilities[0, target_index].item()
    label = weights.meta["categories"][target_index] if weights is not None else f"class_{target_index}"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    overlay(display_image, cams[0]).save(args.output)
    print(f"target={target_index} label={label!r} probability={confidence:.4f}")
    print(f"saved={args.output.resolve()}")


if __name__ == "__main__":
    main()
