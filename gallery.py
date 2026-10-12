"""Generate a real-photo food Grad-CAM gallery with ViT-B/16."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torchvision.models import ViT_B_16_Weights, vit_b_16

from demo import overlay, prepare_image
from gradcam_vit import ViTGradCAM, torchvision_vit_b_16_target_layer


FOOD_IMAGES = (
    ("pizza", "pizza.jpg"),
    ("cheeseburger", "cheeseburger.jpg"),
    ("hot dog", "hot_dog.jpg"),
    ("ice cream", "ice_cream.jpg"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("assets/food"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/food_gallery"))
    parser.add_argument("--weights", choices=("default", "none"), default="default")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_sources(input_dir: Path) -> list[tuple[str, Image.Image]]:
    sources: list[tuple[str, Image.Image]] = []
    for expected_class, filename in FOOD_IMAGES:
        path = input_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing {expected_class} photo: {path}")
        with Image.open(path) as image:
            sources.append((expected_class, image.convert("RGB")))
    return sources


def make_contact_sheet(
    items: list[tuple[str, Image.Image, Image.Image, str]], output: Path, subtitle: str
) -> None:
    margin, image_gap, pair_gap, header, row_height = 20, 14, 30, 62, 278
    pair_width = 2 * 224 + image_gap
    sheet = Image.new(
        "RGB",
        (2 * pair_width + pair_gap + 2 * margin, header + 2 * row_height),
        "#f8fafc",
    )
    draw = ImageDraw.Draw(sheet)
    draw.text((margin, 12), subtitle, fill="#166534" if "Pretrained" in subtitle else "#9a3412")
    draw.text((margin, 34), "Left: photograph    Right: class-conditioned Grad-CAM", fill="#475569")

    for index, (expected_class, source, result, prediction) in enumerate(items):
        row, column = divmod(index, 2)
        x = margin + column * (pair_width + pair_gap)
        y = header + row * row_height
        sheet.paste(source, (x, y))
        sheet.paste(result, (x + 224 + image_gap, y))
        draw.text((x, y + 230), f"expected: {expected_class}", fill="#0f172a")
        draw.text((x + 224 + image_gap, y + 230), prediction, fill="#0f172a")
    sheet.save(output)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    weights = ViT_B_16_Weights.DEFAULT if args.weights == "default" else None

    if weights is None:
        torch.manual_seed(0)
    model = vit_b_16(weights=weights).to(device).eval()
    if weights is None:
        torch.nn.init.trunc_normal_(model.heads.head.weight, std=0.02)

    sources = load_sources(args.input_dir)
    prepared = [(name, *prepare_image(image)) for name, image in sources]
    batch = torch.cat([tensor for _, _, tensor in prepared]).to(device)

    with ViTGradCAM(model, torchvision_vit_b_16_target_layer(model)) as gradcam:
        cams, logits, targets = gradcam(batch)
    probabilities = logits.softmax(dim=1)

    gallery_items: list[tuple[str, Image.Image, Image.Image, str]] = []
    for index, (expected_class, source, _) in enumerate(prepared):
        result = overlay(source, cams[index])
        target = targets[index].item()
        confidence = probabilities[index, target].item()
        label = weights.meta["categories"][target] if weights else f"random class {target}"
        result.save(args.output_dir / f"{index + 1}_{expected_class.replace(' ', '_')}_gradcam.png")
        gallery_items.append(
            (expected_class, source, result, f"predicted: {label} ({confidence:.1%})")
        )

    contact_sheet = args.output_dir / "gradcam_food_gallery.png"
    subtitle = (
        "Pretrained ImageNet ViT-B/16: real food photographs"
        if weights
        else "Random weights: spatial smoke test only"
    )
    make_contact_sheet(gallery_items, contact_sheet, subtitle)
    print(f"saved={contact_sheet.resolve()}")


if __name__ == "__main__":
    main()
