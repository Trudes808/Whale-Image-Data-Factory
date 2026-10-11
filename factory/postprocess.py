"""Raw batch renders -> dataset: images, label masks, amodal body-part masks, metadata.

    python3 -m factory.postprocess data/renders/run01 [--batch batch_003]

Reads RUN_DIR/raw/batch_NNN/ (from tools/render_batch.sh) and writes:

    RUN_DIR/dataset/images/<id>.png    RGB, sRGB (the last frame of the shot)
    RUN_DIR/dataset/masks/<id>.png     uint8 palette PNG: 0 background, 1 whale (visible above
                                       water), 2 whale just below the surface (not in the
                                       rendered visible mask, but the whale surface is within
                                       labels.through_water_depth_m of the mean water plane, so
                                       it may show through the water: an "ignore" class for
                                       segmentation). Pixel values are class ids (read with PIL
                                       or cv2.IMREAD_UNCHANGED); the palette shows 1 white, 2 grey.
    RUN_DIR/dataset/amodal/<id>.png    uint8: body-part id of the whole whale, water ignored
                                       (0 none; see factory.project.PART_IDS)
    RUN_DIR/dataset/bodycoords/<id>.png RGB8 over the amodal whale: R = fraction along the body
                                       (tail 0 -> snout 255), G = angle around the body axis
                                       (0..255 for 0..360 deg; 0 back, 64 right side, 128 belly),
                                       B = 255 where defined. Same body location in every photo.
    RUN_DIR/dataset/depth/<id>.png     uint16 depth of the amodal whale surface in cm (0 = none)
    RUN_DIR/dataset/meta/<id>.json     the shot + orientation relative to the camera, the full
                                       6-DoF pose in the camera frame, camera intrinsics and
                                       pixels-per-metre at the whale, keypoints (pixel position,
                                       depth, occluded by the body, inside the label), visible
                                       fraction, bounding boxes, per-part visible fractions, QA
    RUN_DIR/dataset/manifest.jsonl     one line per shot (paths + the key fields)

QA flags per shot: ``visible_outside_amodal`` (label pixels the projection says are not
whale; anti-aliasing gives about 1 %), ``empty_mask`` for a non-negative shot, and
``mask_touches_border``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from factory import deform, project, qa

VISIBLE_OUTSIDE_AMODAL_LIMIT = 0.05


def frame_files(raw_dir: Path, pass_name: str, frame: int) -> Path:
    return raw_dir / f"{pass_name}.{frame:04d}.png"


MASK_PALETTE = [0, 0, 0, 255, 255, 255, 128, 128, 128]
WATER_Z_CM = 0.0            # shots are placed against the mean water plane at z = 0 (factory.poses.place)
THROUGH_WATER_DEPTH_M = 1.5  # default for runs whose config has no "labels" section (run01)
PART_PALETTE = [0, 0, 0, 60, 120, 255, 40, 200, 200, 255, 220, 40, 255, 120, 40, 230, 60, 200]


def save_palette_png(values: np.ndarray, path: Path, palette: list) -> None:
    """uint8 class ids as a palette PNG: np.asarray(Image.open(path)) returns the ids."""
    image = Image.fromarray(values.astype(np.uint8), mode="P")
    image.putpalette(palette + [0] * (768 - len(palette)))
    image.save(path)


def through_water(amodal: np.ndarray, visible: np.ndarray, height_cm: np.ndarray, depth_m: float) -> np.ndarray:
    """Whale pixels not in the visible label whose surface lies between the mean water plane and
    depth_m below it. Deeper whale stays background; whale above the plane but not visible is
    behind spray or a crest and stays background too."""
    with np.errstate(invalid="ignore"):
        shallow = (height_cm <= WATER_Z_CM) & (height_cm >= WATER_Z_CM - depth_m * 100.0)
    return (amodal > 0) & ~visible & shallow


def mask_stats(mask: np.ndarray) -> dict:
    stats = {"visible_pixels": int(mask.sum()), "visible_fraction_of_image": round(float(mask.mean()), 6)}
    if mask.any():
        rows, cols = np.nonzero(mask)
        stats["bbox_xyxy"] = [int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max())]
        h, w = mask.shape
        stats["mask_touches_border"] = bool(rows.min() == 0 or cols.min() == 0 or rows.max() == h - 1 or cols.max() == w - 1)
    return stats


def process_shot(shot: dict, raw_dir: Path, frame: int, out: Path, vertices: np.ndarray, faces: np.ndarray,
                 resolution, points: dict | None = None, through_water_depth_m: float = THROUGH_WATER_DEPTH_M) -> dict:
    width, height = resolution
    rgb_path = frame_files(raw_dir, "FinalImage", frame)
    label_path = frame_files(raw_dir, "FinalImageM_WhaleVisibleMask", frame)
    if not rgb_path.exists() or not label_path.exists():
        raise FileNotFoundError(f"missing frame {frame} for shot {shot['id']} in {raw_dir}")
    mask = qa.load_mask(label_path)
    camera = project.Camera(shot["camera"]["location"], shot["camera"]["rotation"], shot["camera"]["focal_mm"], width, height)
    params = deform.params_from_dict(shot.get("deform"))
    geometry = vertices if params.is_identity() else deform.deform(vertices, params, deform.fin_regions(vertices))
    amodal, zbuf, body, surface_z = project.amodal_and_parts(geometry, faces, shot["whale"], camera, extra=True,
                                                             canonical_cm=vertices, world_z=True)
    check = project.check_against_mask(amodal, mask)
    shallow = through_water(amodal, mask, surface_z, through_water_depth_m)
    label = mask.astype(np.uint8)
    label[shallow] = 2
    parts = {}
    for name, part_id in project.PART_IDS.items():
        region = amodal == part_id
        parts[name] = {"amodal_pixels": int(region.sum()),
                       "visible_fraction": round(float((mask & region).sum() / max(region.sum(), 1)), 4)}
    shot_id = shot["id"]
    shutil.copyfile(rgb_path, out / "images" / f"{shot_id}.png")
    save_palette_png(label, out / "masks" / f"{shot_id}.png", MASK_PALETTE)
    save_palette_png(np.clip(amodal, 0, 255), out / "amodal" / f"{shot_id}.png", PART_PALETTE)
    defined = amodal > 0
    coords = np.zeros(amodal.shape + (3,), np.uint8)
    coords[..., 0] = np.clip(np.round(body[..., 0] * 255), 0, 255)
    coords[..., 1] = np.clip(np.round(body[..., 1] / 360.0 * 255), 0, 255)
    coords[..., 2] = 255
    coords[~defined] = 0
    Image.fromarray(coords).save(out / "bodycoords" / f"{shot_id}.png")
    depth_cm = np.where(defined, np.clip(zbuf, 0, 65535), 0).astype(np.uint16)
    Image.fromarray(depth_cm, mode="I;16").save(out / "depth" / f"{shot_id}.png")
    keypoints = project.project_landmarks(project.landmark_positions(points or {}, geometry), shot["whale"], camera, zbuf, mask)
    whale_depth = project.pose_in_camera(shot["whale"], camera)
    stats = mask_stats(mask)
    stats["through_water_pixels"] = int(shallow.sum())
    flags = []
    if 1.0 - check["visible_inside_amodal"] > VISIBLE_OUTSIDE_AMODAL_LIMIT and mask.any():
        flags.append("visible_outside_amodal")
    if not mask.any() and not shot.get("negative"):
        flags.append("empty_mask")
    if stats.get("mask_touches_border"):
        flags.append("mask_touches_border")
    meta = dict(shot)
    meta.update({
        "frame": frame,
        "orientation": project.orientation(shot["whale"], camera),
        "pose_in_camera": whale_depth,
        "camera_intrinsics": {"fx_px": round(camera.fx, 3), "fy_px": round(camera.fy, 3), "cx_px": camera.cx, "cy_px": camera.cy,
                              "sensor_width_mm": project.SENSOR_WIDTH_MM, "width": width, "height": height},
        "pixels_per_metre_at_whale": round(camera.fx / max(whale_depth["translation_cm"][0] / 100.0, 1e-3), 3),
        "whale_length_m": round(float(vertices[:, 0].max() - vertices[:, 0].min()) / 100.0, 3),
        "keypoints": keypoints,
        "visible": stats,
        "amodal_pixels": check["amodal_pixels"],
        "visible_fraction_of_whale": check["visible_fraction"],
        "visible_inside_amodal": check["visible_inside_amodal"],
        "parts": parts,
        "qa_flags": flags,
        "paths": {"image": f"images/{shot_id}.png", "mask": f"masks/{shot_id}.png", "amodal": f"amodal/{shot_id}.png",
                  "bodycoords": f"bodycoords/{shot_id}.png", "depth": f"depth/{shot_id}.png"},
    })
    (out / "meta" / f"{shot_id}.json").write_text(json.dumps(meta, indent=1))
    return meta


def process_batch(run_dir: Path, batch_name: str, vertices: np.ndarray, faces: np.ndarray, points: dict | None = None,
                  through_water_depth_m: float = THROUGH_WATER_DEPTH_M) -> list[dict]:
    raw_dir = run_dir / "raw" / batch_name
    batch = json.loads((raw_dir / "batch.json").read_text())
    out = run_dir / "dataset"
    for sub in ("images", "masks", "amodal", "bodycoords", "depth", "meta"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    per_shot = int(batch.get("frames_per_shot", 3))
    metas = []
    for index, shot in enumerate(batch["shots"]):
        frame = index * per_shot + per_shot - 1
        metas.append(process_shot(shot, raw_dir, frame, out, vertices, faces, batch["resolution"], points, through_water_depth_m))
    return metas


def write_manifest(run_dir: Path) -> int:
    meta_dir = run_dir / "dataset" / "meta"
    lines = []
    for path in sorted(meta_dir.glob("*.json")):
        meta = json.loads(path.read_text())
        lines.append(json.dumps({
            "id": meta["id"], "image": meta["paths"]["image"], "mask": meta["paths"]["mask"], "amodal": meta["paths"]["amodal"],
            "category": meta["category"], "negative": meta["negative"], "sea_state": meta["sea_state"], "look": meta["look"],
            "visible_pixels": meta["visible"]["visible_pixels"], "visible_fraction_of_whale": meta["visible_fraction_of_whale"],
            "through_water_pixels": meta["visible"].get("through_water_pixels", 0),
            "orientation": meta["orientation"], "qa_flags": meta["qa_flags"],
        }))
    (run_dir / "dataset" / "manifest.jsonl").write_text("\n".join(lines) + ("\n" if lines else ""))
    return len(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--batch", action="append", help="batch name(s) to process (default: all rendered)")
    parser.add_argument("--mesh", type=Path, default=None, help="whale OBJ (default: from run_config.json)")
    args = parser.parse_args(argv)
    run_config = json.loads((args.run_dir / "run_config.json").read_text())
    mesh = args.mesh or Path(run_config["mesh"])
    depth_m = float(run_config.get("config", {}).get("labels", {}).get("through_water_depth_m", THROUGH_WATER_DEPTH_M))
    vertices, faces, groups = project.load_obj(mesh, with_groups=True)
    points = project.landmarks(vertices, groups)
    (args.run_dir / "dataset").mkdir(parents=True, exist_ok=True)
    (args.run_dir / "dataset" / "landmarks_mesh_frame_cm.json").write_text(
        json.dumps({k: np.round(v, 2).tolist() for k, v in project.landmark_positions(points, vertices).items()}, indent=1))
    raw = args.run_dir / "raw"
    names = args.batch or sorted(p.name for p in raw.iterdir() if (p / "done").exists())
    flagged = 0
    for name in names:
        metas = process_batch(args.run_dir, name, vertices, faces, points, depth_m)
        flagged += sum(bool(m["qa_flags"]) for m in metas)
        print(f"{name}: {len(metas)} shots, {sum(bool(m['qa_flags']) for m in metas)} flagged")
    total = write_manifest(args.run_dir)
    print(f"manifest: {total} shots ({flagged} flagged in this pass) -> {args.run_dir / 'dataset'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
