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
    RUN_DIR/dataset/canon/<id>.npz     key "xyz_mm": uint16 (H, W, 3), the point on the *undeformed*
                                       mesh seen at each pixel, mesh frame (x tail -> snout, y
                                       right, z up), in mm + 32768; 0 where there is no whale
                                       (decode_canon). Same body location in every photo and
                                       non-degenerate on the flukes and fins, unlike the angle
                                       in bodycoords.
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
from collections import Counter
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from factory import deform, project, qa

VISIBLE_OUTSIDE_AMODAL_LIMIT = 0.05
PROJECTION_COVERED_LIMIT = 0.85       # projection above the water row covered by the rendered mask (render vs label scale)
CANON_OFFSET = 32768          # canon npz value = round(mm) + CANON_OFFSET


def encode_canon(canon_cm: np.ndarray) -> np.ndarray:
    """(H, W, 3) cm with nan outside the whale -> uint16 (H, W, 3) mm + offset, 0 outside."""
    defined = np.isfinite(canon_cm[..., 0])
    mm = np.clip(np.round(np.nan_to_num(canon_cm) * 10.0) + CANON_OFFSET, 1, 65535).astype(np.uint16)
    mm[~defined] = 0
    return mm


def decode_canon(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """uint16 (H, W, 3) -> (cm float32 (H, W, 3), defined bool)."""
    defined = values[..., 0] > 0
    cm = (values.astype(np.float32) - CANON_OFFSET) / 10.0
    cm[~defined] = np.nan
    return cm, defined


def write_canon(canon_cm: np.ndarray, path: Path) -> None:
    np.savez_compressed(path, xyz_mm=encode_canon(canon_cm))


def read_canon(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        return decode_canon(data["xyz_mm"])


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
    # label pass: 1.0 the shot's whale, 0.6 a companion whale (mom + calf shots; ~204 after the pass's
    # output encoding), 0 elsewhere
    mask_any = qa.load_mask(label_path, threshold=100)
    mask = qa.load_mask(label_path, threshold=240)
    companion_mask = mask_any & ~mask
    camera = project.Camera(shot["camera"]["location"], shot["camera"]["rotation"], shot["camera"]["focal_mm"], width, height)
    scale = float(shot.get("body_scale", 1.0))
    if MODEL is not None and MODEL.kind != "k9239":
        geometry = MODEL.posed(shot); vertices = MODEL.canonical; faces = MODEL.faces; base = MODEL.canonical * scale
    else:
        params = deform.params_from_dict(shot.get("deform"))
        base = vertices * scale if abs(scale - 1.0) > 1e-6 else vertices
        geometry = base if params.is_identity() else deform.deform(base, params, deform.fin_regions(base))
    amodal, zbuf, body, surface_z, canon, canon_deformed = project.amodal_and_parts(
        geometry, faces, shot["whale"], camera, extra=True, canonical_cm=vertices, world_z=True, canonical_xyz=True, deformed_xyz=True)
    check = project.check_against_mask(amodal, mask)
    shallow = through_water(amodal, mask, surface_z, through_water_depth_m)
    if companion_mask.any():                       # the calf occludes the mother's labels where it is in front
        amodal = np.where(companion_mask, 0, amodal); shallow &= ~companion_mask
    label = mask.astype(np.uint8)
    label[shallow] = 2
    label[companion_mask] = 1                      # any whale is "whale" for the segmentation
    parts = {}
    for name, part_id in project.PART_IDS.items():
        region = amodal == part_id
        parts[name] = {"amodal_pixels": int(region.sum()),
                       "visible_fraction": round(float((mask & region).sum() / max(region.sum(), 1)), 4)}
    shot_id = shot["id"]
    fx = None
    if CAMERA_FX:
        from factory import camera_fx

        rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
        rgb, fx = camera_fx.apply(rgb, int(shot_id) if str(shot_id).isdigit() else abs(hash(shot_id)) % (1 << 31), CAMERA_FX)
        Image.fromarray(rgb).save(out / "images" / f"{shot_id}.png")
    else:
        shutil.copyfile(rgb_path, out / "images" / f"{shot_id}.png")
    save_palette_png(label, out / "masks" / f"{shot_id}.png", MASK_PALETTE)
    if companion_mask.any() or shot.get("companion"):
        instances = mask.astype(np.uint8); instances[companion_mask] = 2
        (out / "instances").mkdir(exist_ok=True)
        save_palette_png(instances, out / "instances" / f"{shot_id}.png", MASK_PALETTE)
    save_palette_png(np.clip(amodal, 0, 255), out / "amodal" / f"{shot_id}.png", PART_PALETTE)
    defined = amodal > 0                            # amodal already cleared under the companion
    coords = np.zeros(amodal.shape + (3,), np.uint8)
    coords[..., 0] = np.clip(np.round(body[..., 0] * 255), 0, 255)
    coords[..., 1] = np.clip(np.round(body[..., 1] / 360.0 * 255), 0, 255)
    coords[..., 2] = 255
    coords[~defined] = 0
    Image.fromarray(coords).save(out / "bodycoords" / f"{shot_id}.png")
    depth_cm = np.where(defined, np.clip(zbuf, 0, 65535), 0).astype(np.uint16)
    Image.fromarray(depth_cm, mode="I;16").save(out / "depth" / f"{shot_id}.png")
    write_canon(canon, out / "canon" / f"{shot_id}.npz")
    write_canon(canon_deformed, out / "canon_deformed" / f"{shot_id}.npz")
    keypoints = project.project_landmarks(project.landmark_positions(points or {}, geometry), shot["whale"], camera, zbuf, mask,
                                          water_z_cm=WATER_Z_CM, canonical=project.landmark_positions(points or {}, vertices))
    rows_visible = np.nonzero(mask.any(axis=1))[0]
    water_row = int(np.percentile(rows_visible, 99)) if len(rows_visible) else None
    whale_depth = project.pose_in_camera(shot["whale"], camera)
    stats = mask_stats(mask)
    stats["through_water_pixels"] = int(shallow.sum())
    stats["companion_pixels"] = int(companion_mask.sum())
    flags = []
    if 1.0 - check["visible_inside_amodal"] > VISIBLE_OUTSIDE_AMODAL_LIMIT and mask.any():
        flags.append("visible_outside_amodal")
    # The converse check: the projection well above the water row must be covered by the rendered mask,
    # otherwise the render is smaller than the labels (e.g. a focal length the camera clamped).
    if water_row is not None and mask.any():
        proj_above = (amodal > 0) & (np.arange(amodal.shape[0])[:, None] <= water_row - 8)
        projection_covered = float((mask & proj_above).sum() / max(proj_above.sum(), 1)) if proj_above.sum() > 500 else None
        if projection_covered is not None and projection_covered < PROJECTION_COVERED_LIMIT:
            flags.append("projection_not_covered")
    else:
        projection_covered = None
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
        "whale_length_m": round(float(base[:, 0].max() - base[:, 0].min()) / 100.0, 3),
        "body_scale": scale,
        "keypoints": keypoints,
        "keypoint_states": dict(Counter(k.get("state", "unknown") for k in keypoints.values())),
        "water_row_px": water_row,
        "camera_fx": fx,
        "visible": stats,
        "amodal_pixels": check["amodal_pixels"],
        "visible_fraction_of_whale": check["visible_fraction"],
        "visible_inside_amodal": check["visible_inside_amodal"],
        "projection_covered_by_mask": None if projection_covered is None else round(projection_covered, 4),
        "parts": parts,
        "qa_flags": flags,
        "paths": {"image": f"images/{shot_id}.png", "mask": f"masks/{shot_id}.png", "amodal": f"amodal/{shot_id}.png",
                  "bodycoords": f"bodycoords/{shot_id}.png", "depth": f"depth/{shot_id}.png", "canon": f"canon/{shot_id}.npz",
                  "canon_deformed": f"canon_deformed/{shot_id}.npz"},
    })
    (out / "meta" / f"{shot_id}.json").write_text(json.dumps(meta, indent=1))
    return meta


def canon_only_shot(shot: dict, out: Path, vertices: np.ndarray, faces: np.ndarray, resolution) -> str:
    """Add canon/<id>.npz (and its path in meta) to a shot that was post-processed before."""
    width, height = resolution
    camera = project.Camera(shot["camera"]["location"], shot["camera"]["rotation"], shot["camera"]["focal_mm"], width, height)
    scale = float(shot.get("body_scale", 1.0))
    if MODEL is not None and MODEL.kind != "k9239":
        geometry = MODEL.posed(shot); vertices = MODEL.canonical; faces = MODEL.faces
    else:
        params = deform.params_from_dict(shot.get("deform"))
        base = vertices * scale if abs(scale - 1.0) > 1e-6 else vertices
        geometry = base if params.is_identity() else deform.deform(base, params, deform.fin_regions(base))
    _, _, _, canon = project.amodal_and_parts(geometry, faces, shot["whale"], camera, extra=True, canonical_cm=vertices, canonical_xyz=True)
    shot_id = shot["id"]
    write_canon(canon, out / "canon" / f"{shot_id}.npz")
    meta_path = out / "meta" / f"{shot_id}.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        meta.setdefault("paths", {})["canon"] = f"canon/{shot_id}.npz"
        meta_path.write_text(json.dumps(meta, indent=1))
    return shot_id


def _canon_batch_worker(args) -> int:
    run_dir, batch_name, mesh = args
    global MODEL
    MODEL = _load_model(json.loads((Path(run_dir) / "run_config.json").read_text()))
    vertices, faces, _ = (MODEL.canonical, MODEL.faces, None) if MODEL is not None else project.load_obj(mesh, with_groups=True)
    raw_dir = Path(run_dir) / "raw" / batch_name
    batch = json.loads((raw_dir / "batch.json").read_text())
    out = Path(run_dir) / "dataset"
    (out / "canon").mkdir(parents=True, exist_ok=True)
    for shot in batch["shots"]:
        canon_only_shot(shot, out, vertices, faces, batch["resolution"])
    return len(batch["shots"])


_WORKER: dict = {}
CAMERA_FX: dict | None = None     # run config "camera_fx": noise / chromatic aberration / vignette / JPEG on the RGB (factory.camera_fx)
MODEL = None                      # factory.whalemodel.WhaleModel of the run (None / k9239: the static mesh path below)


def _load_model(run_config: dict):
    from factory.whalemodel import WhaleModel

    model = WhaleModel.from_config(run_config.get("config", {}))
    return model if model.kind != "k9239" else None


def _init_worker(run_dir: str, mesh: str, depth_m: float, camera_fx: dict | None = None, run_config: dict | None = None) -> None:
    global CAMERA_FX, MODEL
    CAMERA_FX = camera_fx
    MODEL = _load_model(run_config or {})
    if MODEL is not None:
        vertices, faces, groups = MODEL.canonical, MODEL.faces, MODEL.groups
    else:
        vertices, faces, groups = project.load_obj(mesh, with_groups=True)
    points = MODEL.landmark_indices() if MODEL is not None else project.landmarks(vertices, groups)
    _WORKER.update(run_dir=Path(run_dir), vertices=vertices, faces=faces, points=points, depth_m=depth_m)


def _batch_worker(name: str) -> tuple:
    w = _WORKER
    metas = process_batch(w["run_dir"], name, w["vertices"], w["faces"], w["points"], w["depth_m"])
    return name, len(metas), sum(bool(m["qa_flags"]) for m in metas)


def process_batch(run_dir: Path, batch_name: str, vertices: np.ndarray, faces: np.ndarray, points: dict | None = None,
                  through_water_depth_m: float = THROUGH_WATER_DEPTH_M) -> list[dict]:
    raw_dir = run_dir / "raw" / batch_name
    batch = json.loads((raw_dir / "batch.json").read_text())
    out = run_dir / "dataset"
    for sub in ("images", "masks", "amodal", "bodycoords", "depth", "canon", "canon_deformed", "meta"):
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
    parser.add_argument("--canon-only", action="store_true", help="only add canon/<id>.npz to an already processed run")
    parser.add_argument("--workers", type=int, default=1, help="processes (batches are independent)")
    args = parser.parse_args(argv)
    run_config = json.loads((args.run_dir / "run_config.json").read_text())
    mesh = args.mesh or Path(run_config["mesh"])
    depth_m = float(run_config.get("config", {}).get("labels", {}).get("through_water_depth_m", THROUGH_WATER_DEPTH_M))
    global CAMERA_FX
    CAMERA_FX = run_config.get("config", {}).get("camera_fx") or None
    global MODEL
    MODEL = _load_model(run_config)
    if MODEL is not None:
        vertices, faces, groups = MODEL.canonical, MODEL.faces, MODEL.groups
    else:
        vertices, faces, groups = project.load_obj(mesh, with_groups=True)
    points = MODEL.landmark_indices() if MODEL is not None else project.landmarks(vertices, groups)
    (args.run_dir / "dataset").mkdir(parents=True, exist_ok=True)
    (args.run_dir / "dataset" / "landmarks_mesh_frame_cm.json").write_text(
        json.dumps({k: np.round(v, 2).tolist() for k, v in project.landmark_positions(points, vertices).items()}, indent=1))
    raw = args.run_dir / "raw"
    names = args.batch or sorted(p.name for p in raw.iterdir() if (p / "done").exists())
    if args.canon_only:
        from multiprocessing import Pool

        jobs = [(str(args.run_dir), name, str(mesh)) for name in names]
        total = 0
        with Pool(max(1, args.workers)) as pool:
            for name, n in zip(names, pool.imap(_canon_batch_worker, jobs)):
                total += n
                print(f"{name}: canon for {n} shots ({total} so far)", flush=True)
        print(f"canon labels: {total} shots -> {args.run_dir / 'dataset' / 'canon'}")
        return 0
    flagged = 0
    if args.workers > 1:
        from multiprocessing import Pool

        with Pool(args.workers, initializer=_init_worker, initargs=(str(args.run_dir), str(mesh), depth_m, CAMERA_FX, run_config)) as pool:
            for name, n, f in pool.imap_unordered(_batch_worker, names):
                flagged += f
                print(f"{name}: {n} shots, {f} flagged", flush=True)
    else:
        for name in names:
            metas = process_batch(args.run_dir, name, vertices, faces, points, depth_m)
            flagged += sum(bool(m["qa_flags"]) for m in metas)
            print(f"{name}: {len(metas)} shots, {sum(bool(m['qa_flags']) for m in metas)} flagged", flush=True)
    total = write_manifest(args.run_dir)
    print(f"manifest: {total} shots ({flagged} flagged in this pass) -> {args.run_dir / 'dataset'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
