"""Visual QA for rendered samples: mask outline overlays, label statistics, contact sheets.

    python3 -m factory.qa overlay RGB.png MASK.png OUT.png
    python3 -m factory.qa sheet OUT.png IMAGE.png [IMAGE.png ...]   # grid, titled by parent dir
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image


def load_mask(path: str | Path, threshold: int | None = None) -> np.ndarray:
    """Binary whale mask from a rendered label image (0/255, any channel) or a dataset
    class mask (0/1). With threshold None, class-id images (max value 1) use >= 1."""
    image = Image.open(path)
    array = np.asarray(image) if image.mode in ("P", "L", "I;16", "I") else np.asarray(image.convert("RGB")).max(axis=2)
    if threshold is None:
        threshold = 1 if array.max() <= 1 else 128
    return array >= threshold


def mask_outline(mask: np.ndarray) -> np.ndarray:
    """Pixels on the mask boundary (4-neighbourhood)."""
    padded = np.pad(mask, 1, mode="edge")
    interior = (padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:])
    return mask & ~interior


def overlay(rgb_path: str | Path, mask_path: str | Path, out_path: str | Path,
            tint=(255, 64, 32), alpha: float = 0.35) -> dict:
    rgb = np.asarray(Image.open(rgb_path).convert("RGB")).astype(np.float32)
    mask = load_mask(mask_path)
    if mask.shape != rgb.shape[:2]:
        raise ValueError(f"mask {mask.shape} and image {rgb.shape[:2]} differ in size")
    out = rgb.copy()
    out[mask] = (1 - alpha) * out[mask] + alpha * np.asarray(tint, np.float32)
    out[mask_outline(mask)] = (255, 255, 0)
    Image.fromarray(out.clip(0, 255).astype(np.uint8)).save(out_path)
    stats = {"whale_pixels": int(mask.sum()), "whale_fraction": float(mask.mean())}
    if mask.any():
        rows, cols = np.nonzero(mask)
        stats["bbox_xyxy"] = [int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max())]
    return stats


def contact_sheet(out_path: str | Path, image_paths: list, columns: int = 3, width: int = 640) -> None:
    """Grid of images, each titled with its parent directory name (the shot name)."""
    from PIL import ImageDraw

    paths = [Path(p) for p in image_paths]
    if not paths:
        raise ValueError("no images")
    thumbs = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        height = round(image.height * width / image.width)
        thumbs.append((path.parent.name, image.resize((width, height), Image.LANCZOS)))
    cell_h = max(t.height for _, t in thumbs) + 24
    rows = (len(thumbs) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * width, rows * cell_h), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    for index, (title, thumb) in enumerate(thumbs):
        x, y = (index % columns) * width, (index // columns) * cell_h
        draw.text((x + 6, y + 5), title, fill=(255, 255, 255))
        sheet.paste(thumb, (x, y + 24))
    sheet.save(out_path)


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "overlay":
        print(overlay(*sys.argv[2:]))
    elif len(sys.argv) >= 4 and sys.argv[1] == "sheet":
        contact_sheet(sys.argv[2], sys.argv[3:])
    else:
        sys.exit(__doc__)
