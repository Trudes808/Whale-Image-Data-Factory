"""QA for shots with several whales (run06 scenes, also run04 / run05 mom + calf after the per-animal postprocess):
automatic checks and contact sheets with image | instances | parts | keypoints of every animal.

    python3 -m factory.qa_multi check RUN_DIR                     # -> RUN_DIR/dataset/qa_multi.json, exit 1 on failures
    python3 -m factory.qa_multi sheet RUN_DIR [--out-dir DIR] [--per-page 6] [--ids 000001,000007]
    python3 -m factory.qa_multi preview SHOTLIST.jsonl OUT.png [--count 12]   # projected instances of a shot list, no render

Checks per shot (thresholds below):
  mesh_vs_instance   every animal with >= MIN_PIXELS visible pixels: >= INSIDE_LIMIT of its instance pixels inside its own
                     projected mesh grown by 1 px (render and projection differ by about that on silhouettes; the
                     primary's meta check, for every animal)
  stray_ids          pixels of id k farther than STRAY_PX from animal k's projection (a blended id at another whale's
                     silhouette would show up here): <= STRAY_LIMIT of the id's pixels
  invalid_ids        decoded ids beyond the shot's animal count (label_decode.invalid_ids == 0)
  parts_in_instance  parts/ > 0 exactly where instances/ > 0 (no part label outside an instance, none missing)
  mask_matches       masks/ class 1 exactly where instances/ > 0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from factory import project

MIN_PIXELS = 50
INSIDE_LIMIT = 0.95
STRAY_PX = 3
STRAY_LIMIT = 0.01
INSTANCE_COLOURS = [(0, 0, 0), (255, 255, 255), (255, 60, 60), (60, 220, 60), (70, 140, 255), (255, 220, 40), (230, 60, 230), (40, 220, 230)]
PART_COLOURS = [(0, 0, 0), (60, 120, 255), (255, 60, 60), (255, 210, 40), (230, 80, 230), (120, 200, 120)]
PART_NAMES = ["", "fluke", "dorsal", "pectoral", "head", "body"]
STATE_COLOUR = {"visible": (60, 220, 60), "under_water": (70, 130, 255), "hidden": (255, 170, 40), "occluded": (255, 60, 60), "out_of_frame": (160, 160, 160),
                "under_wave": (0, 220, 255), "label_edge": (255, 255, 0), "behind_splash": (255, 255, 255)}   # round 5 states
MAIN_KEYPOINTS = {"snout", "blowhole", "dorsal_fin_tip", "dorsal_fin_front", "dorsal_fin_back", "fluke_notch", "fluke_tip_left", "fluke_tip_right",
                  "pectoral_tip_left", "pectoral_tip_right", "spine_end", "eye_left", "eye_right", "tail_base_top"}


def _model(run_dir: Path):
    from factory.whalemodel import WhaleModel
    run_config = json.loads((run_dir / "run_config.json").read_text())
    m = WhaleModel.from_config(run_config.get("config", {}))
    return m if m.kind != "k9239" else None


def check_shot(ds: Path, meta: dict, model) -> dict:
    from scipy import ndimage
    from factory import postprocess as pp
    sid = meta["id"]
    inst = np.asarray(Image.open(ds / "instances" / f"{sid}.png"))
    mask = np.asarray(Image.open(ds / "masks" / f"{sid}.png"))
    parts_path = ds / "parts" / f"{sid}.png"
    parts = np.asarray(Image.open(parts_path)) if parts_path.exists() else None
    w, h = meta["camera_intrinsics"]["width"], meta["camera_intrinsics"]["height"]
    camera = project.Camera(meta["camera"]["location"], meta["camera"]["rotation"], meta["camera"]["focal_mm"], w, h)
    res = {"id": sid, "n_animals": meta.get("n_animals", 1), "scene": (meta.get("scene") or {}).get("type"), "animals": [], "failures": []}
    for k, a in enumerate(pp.animals_of(meta), start=1):
        geometry = model.posed(a) if model is not None else None
        amodal = project.rasterize(*camera.project(project.transform_mesh(geometry, a["whale"])), model.faces, w, h) > 0
        vis = inst == k
        n_vis = int(vis.sum())
        inside = float((vis & ndimage.binary_dilation(amodal, iterations=1)).sum() / max(n_vis, 1)) if amodal.any() else 0.0   # 1 px: render vs projection
        near = ndimage.binary_dilation(amodal, iterations=STRAY_PX) if amodal.any() else amodal
        stray = int((vis & ~near).sum())
        row = {"instance_id": k, "visible_pixels": n_vis, "inside_own_mesh": round(inside, 4), "stray_pixels": stray}
        res["animals"].append(row)
        if n_vis >= MIN_PIXELS and inside < INSIDE_LIMIT:
            res["failures"].append(f"mesh_vs_instance:{k}")
        if n_vis >= MIN_PIXELS and stray > STRAY_LIMIT * n_vis:
            res["failures"].append(f"stray_ids:{k}")
    if (inst > res["n_animals"]).any() or (meta.get("label_decode") or {}).get("invalid_ids"):
        res["failures"].append("invalid_ids")
    if parts is not None:
        res["parts_outside_instance"] = int(((parts > 0) & (inst == 0)).sum())
        res["instance_without_part"] = int(((parts == 0) & (inst > 0)).sum())
        if res["parts_outside_instance"] or res["instance_without_part"]:
            res["failures"].append("parts_in_instance")
    if not np.array_equal(mask == 1, inst > 0):
        res["failures"].append("mask_matches")
    res["label_decode"] = meta.get("label_decode")
    occl = sum(1 for a in meta.get("animals") or [] for kp in a["keypoints"].values() if kp.get("occluded_by") not in (None, a["instance_id"]))
    res["keypoints_occluded_by_other_whale"] = occl
    return res


_CHECK = {}


def _check_init(run_dir: str):
    _CHECK["ds"] = Path(run_dir) / "dataset"; _CHECK["model"] = _model(Path(run_dir))


def _check_one(path: str) -> dict:
    return check_shot(_CHECK["ds"], json.loads(Path(path).read_text()), _CHECK["model"])


def check_run(run_dir: Path, workers: int = 1) -> dict:
    ds = run_dir / "dataset"
    paths = [str(p) for p in sorted((ds / "meta").glob("*.json"))]
    if workers > 1:
        from multiprocessing import Pool
        with Pool(workers, initializer=_check_init, initargs=(str(run_dir),)) as pool:
            rows = list(pool.imap(_check_one, paths, chunksize=8))
    else:
        _check_init(str(run_dir))
        rows = [_check_one(p) for p in paths]
    fails = [r for r in rows if r["failures"]]
    from collections import Counter
    summary = {"shots": len(rows), "shots_failing": len(fails), "failures": dict(Counter(f.split(":")[0] for r in rows for f in r["failures"])),
               "animals_checked": sum(1 for r in rows for a in r["animals"] if a["visible_pixels"] >= MIN_PIXELS),
               "min_inside_own_mesh": min((a["inside_own_mesh"] for r in rows for a in r["animals"] if a["visible_pixels"] >= MIN_PIXELS), default=None),
               "stray_pixels_total": sum(a["stray_pixels"] for r in rows for a in r["animals"]),
               "visible_pixels_total": sum(a["visible_pixels"] for r in rows for a in r["animals"]),
               "parts_outside_instance_total": sum(r.get("parts_outside_instance", 0) for r in rows),
               "keypoints_occluded_by_other_whale": sum(r["keypoints_occluded_by_other_whale"] for r in rows),
               "blended_pixels_total": sum((r["label_decode"] or {}).get("blended_pixels", 0) for r in rows),
               "poorly_explained_total": sum((r["label_decode"] or {}).get("poorly_explained", 0) for r in rows),
               "scenes": dict(Counter(r["scene"] for r in rows)), "animals_per_shot": dict(Counter(r["n_animals"] for r in rows))}
    out = {"summary": summary, "shots": rows}
    (ds / "qa_multi.json").write_text(json.dumps(out, indent=1))
    return out


def _tint(rgb: np.ndarray, labels: np.ndarray, colours, alpha: float = 0.55) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    for k in range(1, len(colours)):
        sel = labels == k
        if sel.any():
            out[sel] = (1 - alpha) * out[sel] + alpha * np.array(colours[k], np.float32)
    # outlines between labels
    edge = (labels != np.roll(labels, 1, 0)) | (labels != np.roll(labels, 1, 1))
    out[edge & (labels > 0)] = 255
    return out.clip(0, 255).astype(np.uint8)


def _crop_box(inst: np.ndarray, w: int, h: int) -> tuple[int, int, int, int]:
    """The region of the image the whales occupy, with margin, at 16:9 (the whole frame when they fill most of it)."""
    if not (inst > 0).any():
        return 0, 0, w, h
    rows, cols = np.nonzero(inst > 0)
    x0, x1, y0, y1 = cols.min(), cols.max(), rows.min(), rows.max()
    cw = max((x1 - x0) * 1.5, (y1 - y0) * 1.5 * 16 / 9, 320); ch = cw * 9 / 16
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    cw = min(cw, w); ch = min(ch, h)
    a = int(np.clip(cx - cw / 2, 0, w - cw)); b = int(np.clip(cy - ch / 2, 0, h - ch))
    return a, b, int(a + cw), int(b + ch)


def draw_shot(ds: Path, meta: dict, panel_w: int = 440, zoom: bool = True) -> Image.Image:
    sid = meta["id"]
    img = np.asarray(Image.open(ds / "images" / f"{sid}.png").convert("RGB"))
    inst = np.asarray(Image.open(ds / "instances" / f"{sid}.png"))
    pp = ds / "parts" / f"{sid}.png"
    parts = np.asarray(Image.open(pp)) if pp.exists() else np.zeros_like(inst)
    h, w = inst.shape
    box = _crop_box(inst, w, h) if zoom else (0, 0, w, h)
    sx = panel_w / (box[2] - box[0]); ph = int(round((box[3] - box[1]) * sx))

    def panel(arr):
        return Image.fromarray(arr).crop(box).resize((panel_w, ph), Image.NEAREST if arr is not img else Image.LANCZOS)
    p_img = panel(img)
    p_inst = panel(_tint(img, inst, INSTANCE_COLOURS))
    p_part = panel(_tint(img, parts, PART_COLOURS, 0.6))
    p_kp = p_img.copy(); d = ImageDraw.Draw(p_kp, "RGBA")
    for a in meta.get("animals") or []:
        col_id = INSTANCE_COLOURS[min(a["instance_id"], len(INSTANCE_COLOURS) - 1)]
        for name, k in a["keypoints"].items():
            if name not in MAIN_KEYPOINTS or not k["in_frame"]:
                continue
            u, v = (k["uv"][0] - box[0]) * sx, (k["uv"][1] - box[1]) * sx
            c = STATE_COLOUR.get(k.get("state"), (255, 255, 255))
            r = 3 if name not in ("dorsal_fin_tip", "fluke_notch", "snout") else 4
            d.ellipse([u - r, v - r, u + r, v + r], fill=c + (230,), outline=col_id + (255,), width=2)
            if name in ("dorsal_fin_tip", "fluke_notch", "snout"):
                d.text((u + 5, v - 6), f"{a['instance_id']}{name[0]}", fill=col_id + (255,))
    tile = Image.new("RGB", (4 * panel_w + 6, ph + 30), (25, 25, 25))
    for i, p in enumerate((p_img, p_inst, p_part, p_kp)):
        tile.paste(p, (i * (panel_w + 2), 30))
    dr = ImageDraw.Draw(tile)
    sc = meta.get("scene") or {}
    vis = " ".join(f"{a['instance_id']}:{a['role'][:3]} {a['category'][:9]} {a['visible']['visible_pixels']}px {sum(1 for q in a['keypoints'].values() if q.get('state') == 'occluded' and q.get('occluded_by') not in (None, a['instance_id']))}occ"
                   for a in meta.get("animals") or [])
    dr.text((4, 2), f"{sid} {sc.get('type', 'legacy')} {sc.get('framing', '')} {meta['sea_state']} f{meta['camera']['focal_mm']:.0f} {meta['camera']['distance_m']:.0f}m "
                    f"blend {meta.get('label_decode', {}).get('blended_pixels', '?')} | crop {box}", fill=(255, 255, 255))
    dr.text((4, 15), vis[:230], fill=(200, 220, 255))
    return tile


def sheet(run_dir: Path, out_dir: Path, per_page: int = 6, ids: list[str] | None = None, panel_w: int = 440, prefix: str = "qa_multi") -> list[Path]:
    ds = run_dir / "dataset"
    metas = [json.loads(p.read_text()) for p in sorted((ds / "meta").glob("*.json"))]
    if ids:
        metas = [m for m in metas if m["id"] in ids]
    out_dir.mkdir(parents=True, exist_ok=True)
    pages = []
    legend = Image.new("RGB", (4 * panel_w + 6, 22), (40, 40, 40)); dl = ImageDraw.Draw(legend)
    x = 4
    for k in range(1, 8):
        dl.rectangle([x, 5, x + 12, 17], fill=INSTANCE_COLOURS[k]); dl.text((x + 15, 5), f"id {k}", fill=(255, 255, 255)); x += 60
    x += 20
    for k in range(1, 6):
        dl.rectangle([x, 5, x + 12, 17], fill=PART_COLOURS[k]); dl.text((x + 15, 5), PART_NAMES[k], fill=(255, 255, 255)); x += 80
    x += 20
    for st, c in STATE_COLOUR.items():
        dl.ellipse([x, 6, x + 10, 16], fill=c); dl.text((x + 13, 5), st, fill=(255, 255, 255)); x += 95
    for start in range(0, len(metas), per_page):
        tiles = [draw_shot(ds, m, panel_w) for m in metas[start:start + per_page]]
        H = legend.height + sum(t.height for t in tiles)
        page = Image.new("RGB", (legend.width, H), (15, 15, 15)); page.paste(legend, (0, 0))
        y = legend.height
        for t in tiles:
            page.paste(t, (0, y)); y += t.height
        path = out_dir / f"{prefix}_p{start // per_page + 1:02d}.jpg"
        page.save(path, quality=90); pages.append(path)
    return pages


def preview(shotlist: Path, out: Path, count: int = 12, width: int = 640) -> None:
    """Projected instances (nearest whale per pixel, water ignored) of the first shots of a shot list: framing and
    overlap at a glance, before any render. The mean water plane is drawn as the horizon line's level."""
    from factory.whalemodel import WhaleModel
    from factory import postprocess as pp
    shots = [json.loads(l) for l in open(shotlist)][:count]
    model = WhaleModel("humpback4k")
    tiles = []
    for s in shots:
        W, H = 1280, 720
        cam = project.Camera(s["camera"]["location"], s["camera"]["rotation"], s["camera"]["focal_mm"], W, H)
        zbest = np.full((H, W), np.inf); ids = np.zeros((H, W), np.uint8); above = np.zeros((H, W), bool)
        for k, a in enumerate(pp.animals_of(s), start=1):
            g = model.posed(a); world = project.transform_mesh(g, a["whale"])
            uv, dep = cam.project(world)
            img, z, attrs = project.rasterize(uv, dep, model.faces, W, H, None, world[:, 2:3] - s.get("water_surface_z_cm", {}).get("whale", 0.0), return_extra=True)
            nearer = (img > 0) & (z < zbest)
            zbest[nearer] = z[nearer]; ids[nearer] = k; above[nearer] = attrs[..., 0][nearer] > 0
        rgb = np.zeros((H, W, 3), np.uint8) + np.array([20, 40, 70], np.uint8)
        for k in range(1, 8):
            sel = ids == k
            rgb[sel] = (np.array(INSTANCE_COLOURS[k]) * np.where(above[sel], 1.0, 0.45)[:, None]).astype(np.uint8)
        t = Image.fromarray(rgb).resize((width, width * 9 // 16))
        dr = ImageDraw.Draw(t)
        sc = s.get("scene") or {}
        dr.text((4, 2), f"{s['id']} {sc.get('type')} {sc.get('framing')} n={sc.get('animals')} {s['category']} f{s['camera']['focal_mm']:.0f} {s['camera']['distance_m']:.0f}m", fill=(255, 255, 255))
        dr.text((4, 14), " ".join(f"{x['role'][:3]}:{x['arrangement']}" for x in sc.get("arrangements", [])), fill=(255, 255, 160))
        tiles.append(t)
    cols = 3; rows = (len(tiles) + cols - 1) // cols
    sheet_img = Image.new("RGB", (cols * width, rows * width * 9 // 16), (0, 0, 0))
    for i, t in enumerate(tiles):
        sheet_img.paste(t, ((i % cols) * width, (i // cols) * (width * 9 // 16)))
    sheet_img.save(out)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check"); c.add_argument("run_dir", type=Path); c.add_argument("--workers", type=int, default=1)
    s = sub.add_parser("sheet"); s.add_argument("run_dir", type=Path); s.add_argument("--out-dir", type=Path); s.add_argument("--per-page", type=int, default=6)
    s.add_argument("--ids", default=None); s.add_argument("--prefix", default="qa_multi")
    v = sub.add_parser("preview"); v.add_argument("shotlist", type=Path); v.add_argument("out", type=Path); v.add_argument("--count", type=int, default=12)
    a = p.parse_args(argv)
    if a.cmd == "check":
        out = check_run(a.run_dir, a.workers)
        print(json.dumps(out["summary"], indent=1))
        for r in out["shots"]:
            if r["failures"]:
                print(r["id"], r["failures"], r["animals"])
        return 1 if out["summary"]["shots_failing"] else 0
    if a.cmd == "sheet":
        pages = sheet(a.run_dir, a.out_dir or (a.run_dir / "dataset"), a.per_page, a.ids.split(",") if a.ids else None, prefix=a.prefix)
        for pg in pages:
            print("wrote", pg)
        return 0
    preview(a.shotlist, a.out, a.count)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
