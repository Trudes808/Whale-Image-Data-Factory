"""Contact sheet of post-processed shots with the run03 labels drawn on: visible mask edge, the
keypoints coloured by state (green visible, blue under water, orange hidden, red occluded, grey
out of frame drawn at the edge), the water row, and the posed mesh as a sparse wire (from
pose_in_camera + the deformed per-pixel map is not needed: the shot's own deformation is applied).

    python3 -m factory.qa_sheet RUN_DIR [--count 24] [--out sheet.png]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from factory import deform, project

MODEL = None

STATE_COLOUR = {"visible": (60, 220, 60), "under_water": (70, 130, 255), "hidden": (255, 170, 40), "occluded": (255, 60, 60), "out_of_frame": (160, 160, 160)}


def draw_shot(meta: dict, run_dir: Path, vertices: np.ndarray, faces: np.ndarray, size: int = 480) -> Image.Image:
    ds = run_dir / "dataset"
    img = Image.open(ds / meta["paths"]["image"]).convert("RGB")
    mask = np.array(Image.open(ds / meta["paths"]["mask"]).convert("P")) == 1
    w, h = img.size
    draw = ImageDraw.Draw(img, "RGBA")
    # visible mask edge
    edge = mask ^ np.roll(mask, 1, 0) | mask ^ np.roll(mask, 1, 1)
    ys, xs = np.nonzero(edge)
    for x, y in zip(xs[::2], ys[::2]):
        draw.point((int(x), int(y)), fill=(255, 255, 255, 200))
    # posed mesh: sparse edges of the (scaled, bent) whale
    camera = project.Camera(meta["camera"]["location"], meta["camera"]["rotation"], meta["camera"]["focal_mm"], w, h)
    scale = float(meta.get("body_scale", 1.0))
    if MODEL is not None:
        geometry = MODEL.posed(meta)
    else:
        params = deform.params_from_dict(meta.get("deform"))
        base = vertices * scale if abs(scale - 1.0) > 1e-6 else vertices
        geometry = base if params.is_identity() else deform.deform(base, params, deform.fin_regions(base))
    world = project.transform_mesh(geometry, meta["whale"]); uv, depth = camera.project(world)
    for f in faces[::40]:
        if (depth[f] <= 1).any():
            continue
        pts = [tuple(uv[i]) for i in f]
        if all(-w < p[0] < 2 * w and -h < p[1] < 2 * h for p in pts):
            draw.polygon(pts, outline=(80, 220, 255, 90))
    # water row
    if meta.get("water_row_px") is not None:
        y = meta["water_row_px"]; draw.line([(0, y), (w, y)], fill=(0, 200, 255, 120), width=1)
    # keypoints
    for name, k in meta["keypoints"].items():
        u, v = k["uv"]; st = k.get("state", "unknown"); col = STATE_COLOUR.get(st, (255, 255, 255))
        if not k["in_frame"]:
            u = min(max(u, 4), w - 5); v = min(max(v, 4), h - 5)
        r = 5
        draw.ellipse([u - r, v - r, u + r, v + r], fill=col + (230,), outline=(0, 0, 0, 255))
        draw.text((u + 6, v - 7), name.replace("_", " "), fill=(255, 255, 255, 230))
    img = img.resize((size, int(size * h / w)))
    d2 = ImageDraw.Draw(img, "RGBA")
    o = meta["orientation"]
    cap = (f"{meta['id']} {meta['category']} {meta['sea_state']} | vis {meta['visible_fraction_of_whale']:.2f} ext {meta['camera'].get('extent_frame_fraction', 0):.2f} "
           f"| {o['distance_m']:.0f} m f{meta['camera']['focal_mm']:.0f} | roll {o['roll_deg']:.0f} pitch {o['pitch_deg']:.0f} | scale {scale:.2f} | kp {meta.get('keypoint_states', {})}")
    d2.rectangle([0, 0, size, 26], fill=(0, 0, 0, 170)); d2.text((4, 2), cap[:120], fill=(255, 255, 255, 255))
    d2.text((4, 14), cap[120:240], fill=(255, 255, 255, 255))
    return img


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path); p.add_argument("--count", type=int, default=24); p.add_argument("--out", type=Path, default=None); p.add_argument("--cols", type=int, default=4)
    a = p.parse_args(argv)
    run_config = json.loads((a.run_dir / "run_config.json").read_text())
    from factory.whalemodel import WhaleModel
    global MODEL
    m = WhaleModel.from_config(run_config.get("config", {})); MODEL = m if m.kind != "k9239" else None
    vertices, faces, _ = (MODEL.canonical, MODEL.faces, None) if MODEL is not None else project.load_obj(run_config["mesh"], with_groups=True)
    metas = [json.loads(l) for l in open(a.run_dir / "dataset" / "manifest.jsonl")]
    metas = [json.loads((a.run_dir / "dataset" / "meta" / f"{m['id']}.json").read_text()) for m in metas if not m["negative"]][: a.count]
    tiles = [draw_shot(m, a.run_dir, vertices, faces) for m in metas]
    tw, th = tiles[0].size; cols = a.cols; rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw, rows * th), (20, 20, 20))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * tw, (i // cols) * th))
    out = a.out or (a.run_dir / "dataset" / "qa_sheet.png"); sheet.save(out); print("wrote", out, f"({len(tiles)} shots)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
