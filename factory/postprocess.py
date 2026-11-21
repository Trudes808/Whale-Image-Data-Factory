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
    RUN_DIR/dataset/instances/<id>.png palette uint8 per visible whale pixel: 0 background, 1 the shot's whale (primary),
                                       2..N the other whales (meta "animals"; run04 / run05: 2 = the calf)
    RUN_DIR/dataset/parts/<id>.png     palette uint8 per visible whale pixel, every whale: 1 fluke, 2 dorsal fin (blade
                                       and hump), 3 pectoral fin, 4 head, 5 body, 0 background (WhaleModel.part_groups)
    RUN_DIR/dataset/animals/<id>.npz   per whale k its whole (amodal) projection, cropped to a{k}_box = [x0, y0, x1, y1]:
                                       a{k}_parts, a{k}_canon_mm (encode_canon), a{k}_depth_cm, a{k}_front (k is the nearest
                                       whale), a{k}_ignore (its through-water zone); front_id (H, W) = nearest whale

Several whales (run06 scenes, run04 / run05 mom + calf): the primary's masks-independent files (amodal, bodycoords,
depth, canon, canon_deformed) and the top-level meta fields describe the primary where it is the nearest whale; canon
npz also holds xyz_mm_all / front_id (every whale) when the shot has more than one. masks/ class 1 is any whale, class 2
the through-water zone of any whale. Contours class 3 = edge between two whales. Keypoints hidden behind another whale
are "occluded" with "occluded_by" = its instance id. The label pass is decoded by decode_label (exact ids, blended
silhouette pixels resolved by weight; see LABEL_ENCODING_*).

QA flags per shot: ``visible_outside_amodal`` (label pixels the projection says are not
whale; anti-aliasing gives about 1 %), ``empty_mask`` for a non-negative shot, and
``mask_touches_border``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from factory import deform, project, qa

VISIBLE_OUTSIDE_AMODAL_LIMIT = 0.05
PROJECTION_COVERED_LIMIT = 0.5        # the projection's clearly above-water part must be this covered by the rendered mask
PROJECTION_MIN_HEIGHT_CM = 25.0       # "clearly above water": the whale surface this far above the local (frozen Gerstner) surface
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
CONTOUR_PALETTE = [0, 0, 0, 0, 255, 0, 255, 128, 0, 255, 0, 255]     # none, true silhouette, waterline / foam, calf edge
CONTOUR_TOLERANCE_PX = 4          # the body must continue at least this far past the visible edge to count as hidden by foam
CONTOUR_WATER_CM = 45.0           # a visible edge whose whale surface is this close to the local water (and whose body continues
                                  # past it) is the seam into the water: foam and the nearest crest hide the lowest band of skin


def contour_classes(visible: np.ndarray, amodal: np.ndarray, companion: np.ndarray | None, height_above_water_cm: np.ndarray) -> np.ndarray:
    """Classes of the visible mask's boundary pixels (see process_shot). 1 = the whale's own outline (the edge
    coincides with the whole-whale projection's outline: open air against sky or water). 2 = any other edge:
    the seam into the water, the edge of foam or spray lying over the body. 3 = edge against the calf."""
    from scipy import ndimage
    # border_value=1: a whale cut by the frame has no contour along the frame edge (it is neither silhouette nor seam)
    inner = visible & ~ndimage.binary_erosion(visible, iterations=1, border_value=1)
    outside = ~visible
    # The render's own answer is the ground truth of what water (surface, crests in front of the whale, foam)
    # hides: the whole-whale projection minus the visible mask. A visible edge is the whale's own outline only
    # where it coincides (within CONTOUR_TOLERANCE_PX, the pixel or two by which the render and the factory
    # projection differ) with the outline of the whole-whale projection; every other edge, whether the straight
    # seam into the water or the ragged edge of foam lying over the body, is a seam and not to be trusted.
    amodal_outline = amodal & ~ndimage.binary_erosion(amodal, iterations=1, border_value=1)
    near_outline = ndimage.binary_dilation(amodal_outline, iterations=CONTOUR_TOLERANCE_PX)
    calf = ndimage.binary_dilation(outside & companion, iterations=2) & inner if companion is not None and companion.any() else np.zeros_like(inner)
    out = np.zeros(visible.shape, np.uint8)
    out[inner] = 2
    out[inner & near_outline] = 1
    out[calf] = 3
    return out


# ---- label pass decoding --------------------------------------------------------------------------------------------
# The label pass (M_WhaleVisibleMask) is rendered with the RGB pass, i.e. with its spatial supersampling and
# reconstruction filter, so silhouette pixels hold blends of the ids on either side, and PNG quantisation adds +-1 LSB
# of dither to every value strictly between 0 and 1.
# * rgb_id_v1 (run06 on: M_WhaleInstanceMask, wf_build_scene.INSTANCE_LABEL_SHADER, frames FinalImageM_WhaleInstanceMask.*): whale instance k is the colour ID_CODES[k], every channel exactly
#   0 or 1. A pixel all of whose samples saw the same whale is exact (each channel 0 or 255: the dither never moves 0 or
#   1); every blend has a channel strictly in between, and no code is a blend of others (cube corners), so a blend can
#   never pass for another id. Blended pixels go to the id with the larger weight in the best two-code explanation
#   among the exact ids within LABEL_RESOLVE_RADIUS px (all codes when none is near).
# * grey_v0 (run04 / run05): 1.0 = the shot's whale, 0.6 = the calf (stencil 3). A primary / background blend of weight
#   0.6 looks exactly like the calf: thresholding at 100 / 240 (the old decoder) put a ring of false "companion" pixels
#   round every whale, in shots without a calf too. Now: no calf code at all unless the shot has a companion; with one,
#   calf-valued pixels count only in the calf's core (a 3x3 opening), blended pixels within 2 px of the core may go to the
#   calf, all others are split between the primary and background by weight (>= 0.5 primary).
LABEL_ENCODING_LEGACY = "grey_v0"
LABEL_ENCODING_RGB = "rgb_id_v1"
INSTANCE_PASS = "FinalImageM_WhaleInstanceMask"       # the label pass frames of the run06 instance material
ID_CODES = {1: (1, 1, 1), 2: (1, 0, 0), 3: (0, 1, 0), 4: (0, 0, 1), 5: (1, 1, 0), 6: (1, 0, 1), 7: (0, 1, 1)}
MAX_ANIMALS = len(ID_CODES)
LABEL_RESOLVE_RADIUS = 2
RESOLVE_TOLERANCE = 0.08     # linear-light distance a blend may sit from its best explanation (dither, 3-way blends)
LEGACY_CALF_LINEAR = 0.6
INSTANCE_PALETTE = [0, 0, 0, 255, 255, 255, 255, 60, 60, 60, 220, 60, 70, 120, 255, 255, 220, 40, 230, 60, 230, 40, 220, 230]
PART6_IDS = {"fluke": 1, "dorsal": 2, "pectoral": 3, "head": 4, "body": 5}
PART6_PALETTE = [0, 0, 0, 60, 120, 255, 255, 60, 60, 255, 210, 40, 230, 80, 230, 120, 200, 120]


def srgb_to_linear(values: np.ndarray) -> np.ndarray:
    s = np.asarray(values, np.float32) / 255.0
    return np.where(s <= 0.04045, s / 12.92, ((s + 0.055) / 1.055) ** 2.4).astype(np.float32)


def _resolve(lin: np.ndarray, mixed: np.ndarray, exact_ids: np.ndarray, codes: dict, radius: int) -> tuple[np.ndarray, np.ndarray]:
    """Blended pixels -> (ids, residual): the best explanation as w * code(a) + (1 - w) * code(b) (a single code
    included) over the exact ids present within radius px; the id with weight >= 0.5 wins."""
    from scipy import ndimage
    ids = sorted(codes)
    P = np.array([codes[k] for k in ids], np.float32)                      # (K, C)
    ys, xs = np.nonzero(mixed)
    v = lin[ys, xs]                                                         # (M, C)
    size = 2 * radius + 1
    present = np.stack([ndimage.maximum_filter((exact_ids == k).astype(np.uint8), size=size)[ys, xs] > 0 for k in ids], 1)   # (M, K)
    none = ~present.any(1)
    present[none] = True                                                    # nothing exact near: any code may explain it
    best = np.full(len(v), np.inf, np.float32); label = np.zeros(len(v), np.uint8)

    def fit(rows, allowed):
        for i in range(len(ids)):
            for j in range(i, len(ids)):
                a, b = P[i], P[j]
                d = a - b
                dd = float((d * d).sum())
                vv = v[rows]
                w = np.clip(((vv - b) @ d) / dd, 0.0, 1.0) if dd > 0 else np.ones(len(vv), np.float32)
                r = np.linalg.norm(vv - (w[:, None] * a + (1.0 - w[:, None]) * b), axis=1)
                ok = allowed[rows, i] & allowed[rows, j] & (r < best[rows] - 1e-6)
                idx = rows[ok]
                best[idx] = r[ok]
                label[idx] = np.where(w[ok] >= 0.5, ids[i], ids[j])
    fit(np.arange(len(v)), present)
    # a thin part (fin tip, distant fluke) can be all blended pixels with no exact pixel of its own id nearby: explain
    # what the near ids cannot with every code
    poor = np.nonzero(best > RESOLVE_TOLERANCE)[0]
    if len(poor):
        best[poor] = np.inf
        fit(poor, np.ones_like(present))
    out = np.zeros(mixed.shape, np.uint8); res = np.zeros(mixed.shape, np.float32)
    out[ys, xs] = label; res[ys, xs] = best
    return out, res


def decode_label(path: Path, encoding: str, n_animals: int) -> tuple[np.ndarray, dict]:
    """Label-pass PNG -> (instance ids uint8 (H, W): 0 background, 1 the shot's whale, 2..n the others; stats)."""
    from scipy import ndimage
    image = Image.open(path)
    rgb = np.asarray(image.convert("RGB")).astype(np.int16)
    if encoding == LABEL_ENCODING_RGB:
        exact = ((rgb == 0) | (rgb == 255)).all(axis=2)
        bits = (rgb[..., 0] == 255).astype(np.int16) * 4 + (rgb[..., 1] == 255) * 2 + (rgb[..., 2] == 255)
        lut = np.zeros(8, np.uint8)
        for k, (r, g, b) in ID_CODES.items():
            lut[4 * r + 2 * g + b] = k
        ids = np.where(exact, lut[bits], 0).astype(np.uint8)
        invalid = exact & (ids > n_animals)
        ids[invalid] = 0
        codes = {0: (0, 0, 0)} | {k: ID_CODES[k] for k in range(1, min(n_animals, MAX_ANIMALS) + 1)}
        mixed = ~exact | invalid
        lin = srgb_to_linear(rgb)
        res = np.zeros(ids.shape, np.float32)
        if mixed.any():
            resolved, res = _resolve(lin, mixed, np.where(mixed, 255, ids), codes, LABEL_RESOLVE_RADIUS)
            ids[mixed] = resolved[mixed]
        stats = {"exact_pixels": int(((ids > 0) & ~mixed).sum()), "blended_pixels": int(mixed.sum()),
                 "blended_to_whale": int((mixed & (ids > 0)).sum()), "invalid_ids": int(invalid.sum()),
                 "max_residual": round(float(res.max()), 4), "poorly_explained": int((res > RESOLVE_TOLERANCE).sum())}
        return ids, stats
    # grey_v0
    v = rgb.max(axis=2)
    lin = srgb_to_linear(v)[..., None]
    ids = np.zeros(v.shape, np.uint8)
    ids[v == 255] = 1
    exact = (v == 0) | (v == 255)
    calf_core = np.zeros(v.shape, bool)
    codes = {0: (0.0,), 1: (1.0,)}
    if n_animals >= 2:
        calf_val = np.abs(lin[..., 0] - LEGACY_CALF_LINEAR) < 0.012           # 203 / 204 (+-1 LSB dither) and its neighbours
        calf_core = ndimage.binary_opening(calf_val, structure=np.ones((3, 3), bool))
        ids[calf_core] = 2
        exact = exact | calf_core
    mixed = ~exact
    stats = {"exact_pixels": int(((ids > 0) & exact).sum()), "blended_pixels": int(mixed.sum()), "invalid_ids": 0}
    if mixed.any():
        exact_ids = np.where(mixed, 255, ids)
        near_calf = ndimage.binary_dilation(calf_core, iterations=2) if calf_core.any() else np.zeros(v.shape, bool)
        resolved, res = _resolve(lin, mixed & ~near_calf, exact_ids, codes, LABEL_RESOLVE_RADIUS)
        ids[mixed & ~near_calf] = resolved[mixed & ~near_calf]
        if (mixed & near_calf).any():
            r2, res2 = _resolve(lin, mixed & near_calf, exact_ids, codes | {2: (LEGACY_CALF_LINEAR,)}, LABEL_RESOLVE_RADIUS)
            ids[mixed & near_calf] = r2[mixed & near_calf]
            res = np.maximum(res, res2)
        stats.update(blended_to_whale=int((mixed & (ids > 0)).sum()), max_residual=round(float(res.max()), 4), poorly_explained=int((res > RESOLVE_TOLERANCE).sum()))
    return ids, stats


def legacy_threshold_decode(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """The run04 / run05 decoder (thresholds 240 and 100), kept for before / after measurements: (primary, companion)."""
    any_ = qa.load_mask(path, threshold=100)
    primary = qa.load_mask(path, threshold=240)
    return primary, any_ & ~primary



def height_above_local_water(deformed_xyz: np.ndarray, whale: dict, waves, wave_time_s: float) -> np.ndarray:
    """Per pixel: the whale surface's height above the Gerstner water surface at its own world position (cm),
    nan where there is no whale. The water is the same frozen surface the render used."""
    from factory import gerstner
    H, W, _ = deformed_xyz.shape
    defined = ~np.isnan(deformed_xyz[..., 0])
    out = np.full((H, W), np.nan, np.float32)
    if not defined.any():
        return out
    pts = deformed_xyz[defined].reshape(-1, 3).astype(np.float64)
    world = project.transform_mesh(pts, whale)
    water = gerstner.surface_height(waves, world[:, 0], world[:, 1], wave_time_s)
    out[defined] = (world[:, 2] - water).astype(np.float32)
    return out
WATER_Z_CM = 0.0            # shots are placed against the mean water plane at z = 0 (factory.poses.place)
# Revision of the keypoint definitions, written to every meta. 2026-11-02: dorsal_fin_front = the front of the whole fin /
# hump (WhaleModel.DORSAL_FRONT_RATIO), no longer the blade's root; runs labelled before it are refreshed in place with
# --keypoints-only (RELABEL_MOVED = the keypoints that revision moves; every other one must reproduce exactly).
KEYPOINT_REVISION = "2026-11-02-fin-front"
RELABEL_MOVED = {"dorsal_fin_front"}
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


def shot_geometry(shot: dict, vertices: np.ndarray, faces: np.ndarray) -> tuple:
    """The shot's whale in its own frame: (posed geometry, canonical vertices, faces, scaled undeformed base)."""
    scale = float(shot.get("body_scale", 1.0))
    if MODEL is not None and MODEL.kind != "k9239":
        if getattr(MODEL, "weld", None) is not None:           # run06 round 12: the welded patch differs per animal
            can = MODEL.canonical_for(shot.get("individual"))
            return MODEL.posed(shot), can, MODEL.faces, can * scale
        return MODEL.posed(shot), MODEL.canonical, MODEL.faces, MODEL.canonical * scale
    params = deform.params_from_dict(shot.get("deform"))
    base = vertices * scale if abs(scale - 1.0) > 1e-6 else vertices
    geometry = base if params.is_identity() else deform.deform(base, params, deform.fin_regions(base))
    return geometry, vertices, faces, base


def relabel_keypoints(run_dir: Path, shot_id: str, vertices: np.ndarray, faces: np.ndarray, points: dict | None,
                      write: bool = True) -> dict:
    """Recompute an already processed shot's keypoints the way process_shot did: the posed mesh from the meta (the
    shot) rasterized again for the depth buffer (the saved depth map is no substitute: it is zeroed on the anti-aliased
    silhouette pixels and truncated to cm, which flips edge keypoints' states), the visible mask from the dataset
    (instances == 1 where there is a companion, else masks == 1); images and maps stay. Returns {name: px moved} for
    the keypoints whose pixel position (> 0.15 px) or state changed. Only RELABEL_MOVED keypoints are meant to move:
    any other one moving means the geometry was not reproduced (wrong individual library / fin code for the run:
    individuals.LEGACY_DORSAL, WF_INDIVIDUALS), and the shot is then left untouched."""
    ds = run_dir / "dataset"; meta_path = ds / "meta" / f"{shot_id}.json"
    meta = json.loads(meta_path.read_text())
    cam = meta["camera_intrinsics"]; width, height = int(cam["width"]), int(cam["height"])
    camera = project.Camera(meta["camera"]["location"], meta["camera"]["rotation"], meta["camera"]["focal_mm"], width, height)
    geometry, vertices, faces, _ = shot_geometry(meta, vertices, faces)
    _, zbuf, _ = project.amodal_and_parts(geometry, faces, meta["whale"], camera, extra=True)
    inst = ds / "instances" / f"{shot_id}.png"
    mask = (np.asarray(Image.open(inst)) == 1) if inst.exists() else (np.asarray(Image.open(ds / "masks" / f"{shot_id}.png")) == 1)
    keypoints = project.project_landmarks(project.landmark_positions(points or {}, geometry), meta["whale"], camera, zbuf, mask,
                                          water_z_cm=WATER_Z_CM, canonical=project.landmark_positions(points or {}, vertices))
    old = meta.get("keypoints") or {}
    moved = {}
    for name, k in keypoints.items():
        o = old.get(name)
        if o is None:
            moved[name] = float("nan")
            continue
        d = float(np.hypot(k["uv"][0] - o["uv"][0], k["uv"][1] - o["uv"][1]))
        if d > 0.15 or k.get("state") != o.get("state"):
            moved[name] = d
    if write and not (set(moved) - RELABEL_MOVED):
        meta["keypoints"] = keypoints
        meta["keypoint_states"] = dict(Counter(k.get("state", "unknown") for k in keypoints.values()))
        meta["keypoint_revision"] = KEYPOINT_REVISION
        meta_path.write_text(json.dumps(meta, indent=1))
    return moved


def _relabel_worker(job: tuple) -> dict:
    """One chunk of shot ids -> {"n", "moved": Counter per keypoint, "unexpected": [(id, name, px)]}."""
    ids, write = job
    w = _WORKER
    moved_count = Counter(); unexpected = []
    for shot_id in ids:
        moved = relabel_keypoints(w["run_dir"], shot_id, w["vertices"], w["faces"], w["points"], write=write)
        moved_count.update(moved.keys())
        unexpected += [(shot_id, name, round(px, 1)) for name, px in moved.items() if name not in RELABEL_MOVED]
    return {"n": len(ids), "moved": moved_count, "unexpected": unexpected}


QUALITY_GATE: dict | None = None   # config "quality_gate" (run06 realism): reject shots a photographer would not keep
DROP_ISOLATED = False   # config labels.drop_isolated_instance_pixels (run06, round 11): drop single instance pixels with no 8-neighbour of their id


def drop_isolated_pixels(inst: np.ndarray) -> np.ndarray:
    """Instance pixels with no 8-neighbour of the same id (a lone decode artefact, e.g. on the frame's last row) -> 0."""
    from scipy import ndimage
    out = inst.copy()
    for k in np.unique(inst):
        if k == 0:
            continue
        m = inst == k
        nb = ndimage.convolve(m.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant") - m
        out[m & (nb == 0)] = 0
    return out


def quality_gate(image_path: Path, inst: np.ndarray, projection_covered, shot: dict) -> dict:
    """run06 realism: a shot is rejected (meta quality_gate.rejected, left out of the splits by factory.dataset) when more
    than max_clipped of the frame is blown out, more than max_white_background of it is bright white that is not whale
    (splash / foam walls), or the whale's clearly-above-water body is almost all hidden (projection_covered_by_mask under
    min_covered) in a category that does not ask for it."""
    g = QUALITY_GATE or {}
    rgb = np.asarray(Image.open(image_path).convert("RGB"))
    clipped = float((rgb.max(axis=2) >= 254).mean())
    white_bg = float(((rgb.min(axis=2) >= 225) & (inst == 0)).mean())
    reasons = []
    if clipped > float(g.get("max_clipped", 0.2)):
        reasons.append("clipped_frame")
    if white_bg > float(g.get("max_white_background", 0.3)):
        reasons.append("white_water_over_frame")
    hidden_ok = set(g.get("hidden_ok_categories", ["fluke_sinking", "submerged", "awash", "blow_sliver"]))
    if projection_covered is not None and projection_covered < float(g.get("min_covered", 0.25)) and shot.get("category") not in hidden_ok \
            and not shot.get("negative"):
        reasons.append("whale_hidden")
    out = {"rejected": False, "reasons": reasons, "clipped_frame": round(clipped, 4), "white_background": round(white_bg, 4)}
    if g.get("frame_L_mean") or g.get("frame_L_p99"):
        # run06 realism round 2: exposure inside the field photos' range (their 1st..99th percentiles of frame mean L* and
        # frame 99th-percentile L*, training days): a near-black or washed-out frame is rejected
        k = max(1, int(round(rgb.shape[1] / 1280)))         # the score card's 1280-wide frame (area average)
        h, w = rgb.shape[0] // k * k, rgb.shape[1] // k * k
        lin = rgb[:h, :w].astype(np.float32).reshape(h // k, k, w // k, k, 3).mean(axis=(1, 3)) / 255.0
        lin = np.where(lin <= 0.04045, lin / 12.92, ((lin + 0.055) / 1.055) ** 2.4)
        y = lin @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        lstar = np.where(y > 0.008856, 116.0 * np.cbrt(y) - 16.0, 903.3 * y)
        mean, p99 = float(lstar.mean()), float(np.percentile(lstar, 99))
        out.update(frame_L_mean=round(mean, 2), frame_L_p99=round(p99, 2))
        for key, value in (("frame_L_mean", mean), ("frame_L_p99", p99)):
            lo, hi = g.get(key) or (-1e9, 1e9)
            if value < lo:
                reasons.append(f"{key}_low")
            elif value > hi:
                reasons.append(f"{key}_high")
        if g.get("water_L_min") is not None:
            # run06 round 8: the water's own L* (median of the pixels that are not whale, not white water and below the
            # horizon line's brightness), which white water on a near-black sea cannot lift the way it lifts the frame mean
            ik = inst[:h:k, :w:k][: lstar.shape[0], : lstar.shape[1]]
            water = (ik == 0) & (lstar < 75.0)
            if water.sum() > 500:
                wl = float(np.median(lstar[water]))
                out["water_L"] = round(wl, 2)
                if wl < float(g["water_L_min"]):
                    reasons.append("water_L_low")
    out["rejected"] = bool(reasons)
    return out


def enrich_keypoints(kps: dict, whale: dict, waves, t_wave: float, vis_k: np.ndarray, whale_len_px: float) -> None:
    """run06 round 5, in place on one animal's keypoints:
    height_above_local_water_m: above the frozen Gerstner surface at the point (the render's water), next to the mean-plane
    height; the state "hidden" (above the mean plane, in frame, unoccluded, not in the label) split into under_wave (below
    the local surface), label_edge (visible label within 6 px: a thin tip the mask misses) and behind_splash (anything else:
    white water, splash or stream geometry in front of the point); soft_visibility: the share of a disc of radius 0.02 whale
    lengths round the point that is in this animal's visible label."""
    from factory import gerstner
    H, W = vis_k.shape
    r = max(2.0, 0.02 * whale_len_px)
    for kp in kps.values():
        if kp.get("xyz_whale_cm") is None:
            continue
        world = project.transform_mesh(np.asarray(kp["xyz_whale_cm"], float)[None, :], whale)[0]
        if waves is not None:
            surf = float(gerstner.surface_height(waves, np.array([world[0]]), np.array([world[1]]), t_wave)[0])
            kp["height_above_local_water_m"] = round((float(world[2]) - surf) / 100.0, 3)
        u, v = kp["uv"]
        if kp.get("in_frame"):
            c0, c1, r0, r1 = int(max(u - r, 0)), int(min(u + r + 1, W)), int(max(v - r, 0)), int(min(v + r + 1, H))
            yy, xx = np.mgrid[r0:r1, c0:c1]
            disc = (xx - u) ** 2 + (yy - v) ** 2 <= r * r
            kp["soft_visibility"] = round(float(vis_k[r0:r1, c0:c1][disc].mean()), 3) if disc.any() else 0.0
        if kp.get("state") == "hidden":
            if kp.get("height_above_local_water_m") is not None and kp["height_above_local_water_m"] < 0:
                kp["state"] = "under_wave"
            else:
                c0, c1, r0, r1 = int(max(u - 6, 0)), int(min(u + 7, W)), int(max(v - 6, 0)), int(min(v + 7, H))
                kp["state"] = "label_edge" if vis_k[r0:r1, c0:c1].any() else "behind_splash"


def part_planes(geometry: np.ndarray, vparts: np.ndarray | None, rest: np.ndarray | None, pinc: dict, fluke_curl_deg) -> dict:
    """run06 round 5: the fluke and dorsal fin planes per animal in camera coordinates (x forward, y right, z up), from a
    plane fit to the part's posed vertices (the tail stock bends, so the body's angles do not give them).
    fluke: normal (pointing out of the dorsal surface), facing (dorsal / ventral side toward the camera), view_deg (angle
    between the normal and the line of sight: 0 face-on, 90 edge-on), span_roll_deg (tip-to-tip axis against the image
    horizontal), normal azimuth / elevation in the camera frame, and the shot's curl. fin: normal (pointing out of the
    right flank), facing flank, sight_deg (angle between the fin's normal and the line of sight: 0 side-on, 90 end-on)."""
    out = {}
    if vparts is None or rest is None or len(vparts) != len(geometry):
        return out
    R = np.asarray(pinc["rotation"], float)
    sight = np.array([1.0, 0.0, 0.0])
    for name, pid in (("fluke", 1), ("fin", 2)):
        sel = vparts == pid
        if name == "fin" and sel.any():                   # the blade only (the fin label also holds the hump's slope)
            zr = rest[sel, 2]
            sel = sel & (rest[:, 2] > zr.min() + 0.6 * (zr.max() - zr.min()))
        if sel.sum() < 20:
            continue
        P = geometry[sel] @ R
        c = P.mean(0)
        _, _, vt = np.linalg.svd(P - c, full_matrices=False)
        n = vt[2]; major = vt[0]
        rp = rest[sel]
        if name == "fluke":
            # dorsal side: the posed image of the rest blade's upper half (rest z above the blade's mid plane)
            upper = rp[:, 2] > np.median(rp[:, 2])
            ref = P[upper].mean(0) - P[~upper].mean(0)
            if n @ ref < 0:
                n = -n
            span = (geometry[sel][np.argmin(rp[:, 1])] - geometry[sel][np.argmax(rp[:, 1])]) @ R    # left tip - right tip
            out["fluke"] = {"normal": [round(float(v), 4) for v in n], "facing": "dorsal" if n @ -sight > 0 else "ventral",
                            "view_deg": round(float(np.degrees(np.arccos(min(1.0, abs(n @ sight))))), 1),
                            "span_roll_deg": round(float(np.degrees(np.arctan2(span[2], -span[1]))), 1),
                            "normal_azimuth_deg": round(float(np.degrees(np.arctan2(n[1], -n[0]))), 1),
                            "normal_elevation_deg": round(float(np.degrees(np.arcsin(max(-1.0, min(1.0, n[2]))))), 1),
                            "curl_deg": fluke_curl_deg}
        else:
            # a fin is too small and thick for a plane fit to pick its side: the normal is the posed direction from the
            # blade's left half to its right half (whale-frame +y, the fluke_tip_right side)
            # the posed left-to-right direction across the back at the fin (rest vertices 15-45 cm either side of the
            # midline, upper half, within 25 cm of the blade's x): the fin plane is the body's local midplane there
            xf = float(np.median(rp[:, 0])); zf = float(rp[:, 2].min())
            near = (np.abs(rest[:, 0] - xf) < 25.0) & (rest[:, 2] > zf - 60.0)
            rs = near & (rest[:, 1] > 15.0) & (rest[:, 1] < 45.0); ls = near & (rest[:, 1] < -15.0) & (rest[:, 1] > -45.0)
            if rs.sum() < 3 or ls.sum() < 3:
                continue
            ref = (geometry[rs].mean(0) - geometry[ls].mean(0)) @ R
            n = ref / max(float(np.linalg.norm(ref)), 1e-9)
            out["fin"] = {"normal": [round(float(v), 4) for v in n], "facing": "right" if n @ -sight > 0 else "left",
                          "sight_deg": round(float(np.degrees(np.arccos(min(1.0, abs(n @ sight))))), 1)}
    return out


def _fin_spec(individual) -> dict | None:
    """run06 round 6: the animal's dorsal fin specification from its individual (height scale, chord multiplier, sweep and
    rake in cm, tip roundness 0..1, hump height in cm, trailing-edge wear), for computing fin coverage from labels."""
    if not _individual_spec(individual):
        return None
    spec = (_INDIVIDUALS or {}).get(individual) or {}           # the full library entry (blade fields included)
    keys = ("dorsal_height", "dorsal_chord", "dorsal_sweep_cm", "dorsal_rake_cm", "dorsal_round", "dorsal_hump_cm", "dorsal_edge_cm", "dorsal_nicks",
            "blade_template", "blade_H_cm", "blade_nicks", "blade_thickness")
    out = {k: spec.get(k) for k in keys if k in spec}
    if MODEL is not None and getattr(MODEL, "n_rig", None) is not None:
        # round 7: the generated blade's own outline in its fin frame (a along the base from the back base, in units of
        # the height H; levels h = 0..1) and its shape numbers, from the blade mesh itself
        from factory import blade as bl
        loc = MODEL._blade_local(individual)
        V = loc["vertices"]
        H = float(spec.get("blade_H_cm", 45.0)) if spec.get("blade_template", -1) >= 0 else 45.0
        R2 = 2 * bl.NC
        if "outline" in loc:          # round 12 (weld): the template outline at the standard levels (the same numbers)
            o = loc["outline"]
            lv = [(float(h), float(t), float(l)) for h, t, l in zip(o["levels"], o["te"], o["le"]) if h >= 0.0]
        else:
            lv = [(float(h), float(V[k * R2:(k + 1) * R2, 0].min() / H), float(V[k * R2:(k + 1) * R2, 0].max() / H))
                  for k, h in enumerate(bl.LEVELS) if h >= 0.0]
        hs = np.array([x[0] for x in lv]); tr = np.array([x[1] for x in lv]); le = np.array([x[2] for x in lv])
        c75, c90 = float(np.interp(0.75, hs, le - tr)), float(np.interp(0.9, hs, le - tr))
        base_oh = float(le[0] - tr[0])
        if getattr(MODEL, "weld", None) is not None:
            # round 13: the welded mesh's own base (front-base to back-base landmark along the fin frame, / H): the
            # keypoints and the picture show the mesh, whose lifted hump reaches beyond the template's base points
            base_oh = MODEL.weld_base_over_height(individual)
        out["blade"] = {"template": spec.get("blade_template"), "H_cm": round(H, 2), "base_over_height": round(base_oh, 3),
                        "base_over_height_template": round(float(le[0] - tr[0]), 3),
                        "broad": round(c75, 3), "sharp": round(c90 / max(c75, 1e-6), 3), "overhang": round(float(np.interp(0.75, hs, tr)), 3),
                        "hump_front_slope_over_height": round(float(le[0] - np.interp(0.5, hs, le)), 3),
                        "outline": {"h": [round(x, 3) for x in hs], "trail": [round(x, 3) for x in tr], "lead": [round(x, 3) for x in le]}}
    return out


def pose_summary(kps: dict, pinc: dict) -> dict:
    """run06 round 5: compact pose numbers per animal for measuring (and the pose-estimation side experiment):
    heading of the body in the image (degrees, 0 = travelling right, 90 = up the frame), heights above the local water of
    the fluke tips, notch and dorsal fin tip (m), the fluke plane against the camera (fluke_view_deg 0 = blade face-on,
    90 = edge-on; fluke_span_tilt_deg = tip-to-tip line against the water, 0 level, 90 one blade straight up; the normal
    of the blade's back surface in camera coordinates x forward, y right, z up)."""
    def g(n):
        return kps.get(n) or {}
    out = {}
    s, t = g("snout"), g("spine_end") or g("fluke_notch")
    if s.get("uv") and t.get("uv"):
        out["heading_image_deg"] = round(math.degrees(math.atan2(-(s["uv"][1] - t["uv"][1]), s["uv"][0] - t["uv"][0])), 1)
    for n in ("fluke_tip_left", "fluke_tip_right", "fluke_notch", "dorsal_fin_tip"):
        if g(n).get("height_above_water_m") is not None:
            out[f"height_m_{n}"] = g(n)["height_above_water_m"]
    tl, tr, no = g("fluke_tip_left").get("xyz_whale_cm"), g("fluke_tip_right").get("xyz_whale_cm"), g("fluke_notch").get("xyz_whale_cm")
    if tl and tr and no:
        R = np.asarray(pinc["rotation"], float)               # rows: the whale's axes in camera coordinates
        a, b, c = (np.asarray(v, float) @ R for v in (tl, tr, no))
        span = b - a
        nrm = np.cross(span, c - 0.5 * (a + b))
        if np.linalg.norm(nrm) > 1e-6:
            nrm /= np.linalg.norm(nrm)
            out["fluke_view_deg"] = round(math.degrees(math.acos(min(1.0, abs(float(nrm[0]))))), 1)
            out["fluke_normal_camera"] = [round(float(v), 4) for v in nrm]
        hl, hr = g("fluke_tip_left").get("height_above_water_m"), g("fluke_tip_right").get("height_above_water_m")
        if hl is not None and hr is not None:
            out["fluke_span_tilt_deg"] = round(math.degrees(math.asin(max(-1.0, min(1.0, (hl - hr) * 100.0 / max(float(np.linalg.norm(span)), 1e-3))))), 1)
    return out


def animals_of(shot: dict) -> list[dict]:
    """The shot's whales, primary first: [shot] + shot["companions"] (run06 scenes) or + [shot["companion"]] (run04 /
    run05 mom + calf, instance 2). Each further animal carries "instance_id" (2..N) and "role"."""
    comps = shot.get("companions")
    if comps is None and shot.get("companion"):
        comps = [dict(shot["companion"], instance_id=2, role=shot["companion"].get("role", "calf"))]
    out = [shot]
    for k, c in enumerate(comps or []):
        out.append(c if c.get("instance_id") else dict(c, instance_id=k + 2))
    return out


def _parts_fill(parts: np.ndarray, region: np.ndarray) -> np.ndarray:
    """parts where they are defined, else the nearest defined part, inside region (the animal's visible pixels whose
    pixel the projection misses by a pixel: render and projection differ by about that on silhouettes)."""
    from scipy import ndimage
    out = np.zeros(parts.shape, np.uint8)
    if not region.any():
        return out
    missing = region & (parts == 0)
    out[region] = parts[region]
    if missing.any() and (parts > 0).any():
        rows, cols = np.nonzero(region | (parts > 0))
        r0, r1, c0, c1 = max(rows.min() - 1, 0), rows.max() + 2, max(cols.min() - 1, 0), cols.max() + 2
        sub = parts[r0:r1, c0:c1]
        _, (ii, jj) = ndimage.distance_transform_edt(sub == 0, return_indices=True)
        filled = sub[ii, jj]
        m = missing[r0:r1, c0:c1]
        out[r0:r1, c0:c1][m] = filled[m]
    return out


def _crop(region: np.ndarray) -> list:
    if not region.any():
        return [-1, -1, -1, -1]
    rows, cols = np.nonzero(region)
    return [int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max())]


def process_shot(shot: dict, raw_dir: Path, frame: int, out: Path, vertices: np.ndarray, faces: np.ndarray,
                 resolution, points: dict | None = None, through_water_depth_m: float = THROUGH_WATER_DEPTH_M,
                 label_encoding: str | None = None) -> dict:
    """One shot: every whale in it (animals_of) gets its own projection and labels. The primary's files and top-level
    meta fields keep their run03-run05 meaning (cleared where another animal is in front of it); the per-animal labels
    are instances/, parts/, animals/ and meta["animals"]."""
    from factory import gerstner as _gerstner
    width, height = resolution
    rgb_path = frame_files(raw_dir, "FinalImage", frame)
    label_path = frame_files(raw_dir, "FinalImageM_WhaleVisibleMask", frame)
    if not label_path.exists() and frame_files(raw_dir, INSTANCE_PASS, frame).exists():
        label_path = frame_files(raw_dir, INSTANCE_PASS, frame)          # run06 instance label pass (MRQ_BatchInst*)
        label_encoding = label_encoding or LABEL_ENCODING_RGB
    if not rgb_path.exists() or not label_path.exists():
        raise FileNotFoundError(f"missing frame {frame} for shot {shot['id']} in {raw_dir}")
    animals = animals_of(shot)
    n = len(animals)
    inst, decode_stats = decode_label(label_path, label_encoding or LABEL_ENCODING_LEGACY, n)
    if DROP_ISOLATED:
        inst = drop_isolated_pixels(inst)
    mask = inst == 1
    camera = project.Camera(shot["camera"]["location"], shot["camera"]["rotation"], shot["camera"]["focal_mm"], width, height)
    scale = float(shot.get("body_scale", 1.0))
    vparts = MODEL.vertex_part_ids() if (MODEL is not None and MODEL.kind != "k9239") else None
    waves = _gerstner.generate_waves(shot["waves"]) if shot.get("waves") else None
    t_wave = float(shot.get("wave_time_s") or 0.0)
    geoms, maps = [], []
    for a in animals:
        geometry, canon_v, faces_a, base = shot_geometry(a, vertices, faces)
        geoms.append((geometry, canon_v, base))
        blade = None
        vparts_a = vparts
        if vparts is not None and getattr(MODEL, "weld", None) is not None:        # round 12: per animal (hump patch)
            vparts_a = MODEL.vertex_part_ids(a.get("individual"))
        if MODEL is not None and getattr(MODEL, "landmarks_v2", False) and vparts is not None:
            if getattr(MODEL, "weld", None) is not None:
                blade = MODEL.blade_weight(a.get("individual"))
            elif getattr(MODEL, "n_rig", None) is not None and len(vparts) == len(geometry):     # round 7: the parametric blade
                blade = (np.arange(len(geometry)) >= MODEL.n_rig).astype(float)
            else:
                fin = MODEL.dorsal_fin()                 # run06 round 5: the blade alone, inside the fin + hump part
                if fin and len(fin["w"]) == len(geometry):
                    blade = ((np.asarray(fin["w"]) > 0.5) & (vparts == 2)).astype(float)
        soft = MODEL.weld_dorsal_soft(a.get("individual")) if (vparts is not None and getattr(MODEL, "weld", None) is not None) else None
        maps.append(project.animal_maps(geometry, faces_a, a["whale"], camera, canon_v, vparts_a, blade_weight=blade, dorsal_soft=soft))
    zstack = np.stack([m["zbuf"] for m in maps])
    zall = zstack.min(0)
    front = np.where(np.isfinite(zall), np.argmin(zstack, axis=0) + 1, 0).astype(np.uint8)    # nearest whale surface, water ignored
    geometry, vertices_c, base = geoms[0]
    m1 = maps[0]
    amodal_full, zbuf, body, surface_z, canon, canon_deformed = m1["amodal"], m1["zbuf"], m1["body"], m1["height"], m1["canon"], m1["deformed"]
    check = project.check_against_mask(amodal_full, mask)
    heights, shallows = [], []
    for k, (a, m) in enumerate(zip(animals, maps), start=1):
        # through-water (class 2): whale just under the actual (frozen Gerstner) surface, not the mean plane; only where this
        # whale is the nearest one and no whale is visible
        h = height_above_local_water(m["deformed"], a["whale"], waves, t_wave) if waves is not None else m["height"] - WATER_Z_CM
        heights.append(h)
        own = np.where(front == k, m["amodal"], 0)
        shallows.append(through_water(own, inst == k, h + WATER_Z_CM, through_water_depth_m) & (inst == 0))
    height_above = heights[0]
    amodal = np.where(front == 1, amodal_full, 0)       # the primary's labels end where another whale is in front of it
    shallow = shallows[0]
    shallow_any = np.logical_or.reduce(shallows)
    label = (inst > 0).astype(np.uint8)                 # any whale is "whale" for the segmentation
    label[shallow_any & (inst == 0)] = 2
    others = (inst > 0) & ~mask
    parts = {}
    for name, part_id in project.PART_IDS.items():
        region = amodal == part_id
        parts[name] = {"amodal_pixels": int(region.sum()),
                       "visible_fraction": round(float((mask & region).sum() / max(region.sum(), 1)), 4)}
    shot_id = shot["id"]
    fx = None
    if IMAGES_FROM is not None and (IMAGES_FROM / f"{shot_id}.png").exists():
        # relabelling an existing run into a new folder (--images-from): link its finished images, do not redo them
        dst = out / "images" / f"{shot_id}.png"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to((IMAGES_FROM / f"{shot_id}.png").resolve())
        prev = IMAGES_FROM.parent / "meta" / f"{shot_id}.json"
        fx = json.loads(prev.read_text()).get("camera_fx") if prev.exists() else None
    elif CAMERA_FX:
        from factory import camera_fx

        rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
        rgb, fx = camera_fx.apply(rgb, int(shot_id) if str(shot_id).isdigit() else abs(hash(shot_id)) % (1 << 31), CAMERA_FX)
        Image.fromarray(rgb).save(out / "images" / f"{shot_id}.png")
    else:
        shutil.copyfile(rgb_path, out / "images" / f"{shot_id}.png")
    save_palette_png(label, out / "masks" / f"{shot_id}.png", MASK_PALETTE)
    # Contour classes along every whale's visible boundary: 1 = true silhouette (the whale ends there against sky or water
    # background: its amodal projection ends too), 2 = waterline / foam edge (the body continues under the surface or
    # behind white water: unreliable in real photos), 3 = edge between two whales (from the instance map). The top contour
    # of a surfacing whale is class 1; a fit should match it and ignore class 2.
    contours = np.zeros(label.shape, np.uint8)
    for k, m in enumerate(maps, start=1):
        vis_k = inst == k
        if vis_k.any():
            c_k = contour_classes(vis_k, (m["amodal"] > 0) & (front == k), (inst > 0) & ~vis_k, heights[k - 1])
            contours[c_k > 0] = c_k[c_k > 0]
    (out / "contours").mkdir(exist_ok=True)
    save_palette_png(contours, out / "contours" / f"{shot_id}.png", CONTOUR_PALETTE)
    (out / "instances").mkdir(exist_ok=True)
    save_palette_png(inst, out / "instances" / f"{shot_id}.png", INSTANCE_PALETTE)
    part_map = None
    if vparts is not None:
        part_map = np.zeros(label.shape, np.uint8)
        for k, m in enumerate(maps, start=1):
            vis_k = inst == k
            if vis_k.any():
                part_map[vis_k] = _parts_fill(m["parts"], vis_k)[vis_k]
        (out / "parts").mkdir(exist_ok=True)
        save_palette_png(part_map, out / "parts" / f"{shot_id}.png", PART6_PALETTE)
    save_palette_png(np.clip(amodal, 0, 255), out / "amodal" / f"{shot_id}.png", PART_PALETTE)
    defined = amodal > 0                            # the primary where it is the nearest whale
    coords = np.zeros(amodal.shape + (3,), np.uint8)
    coords[..., 0] = np.clip(np.round(body[..., 0] * 255), 0, 255)
    coords[..., 1] = np.clip(np.round(body[..., 1] / 360.0 * 255), 0, 255)
    coords[..., 2] = 255
    coords[~defined] = 0
    Image.fromarray(coords).save(out / "bodycoords" / f"{shot_id}.png")
    depth_cm = np.where(defined, np.clip(zbuf, 0, 65535), 0).astype(np.uint16)
    Image.fromarray(depth_cm, mode="I;16").save(out / "depth" / f"{shot_id}.png")
    canon_p = np.where(defined[..., None], canon, np.nan)
    canon_d = np.where(defined[..., None], canon_deformed, np.nan)
    if not DENSE_BODY_MAPS:
        pass                                            # storage.dense_body_maps = false (run06 realism): no canon files
    elif n > 1:
        # every whale's canonical point where it is the nearest one (front_id says which): for dense models over all whales
        canon_all = np.full(canon.shape, np.nan, np.float32)
        for k, m in enumerate(maps, start=1):
            sel = front == k
            canon_all[sel] = m["canon"][sel]
        np.savez_compressed(out / "canon" / f"{shot_id}.npz", xyz_mm=encode_canon(canon_p), xyz_mm_all=encode_canon(canon_all), front_id=front)
    else:
        write_canon(canon_p, out / "canon" / f"{shot_id}.npz")
    if DENSE_BODY_MAPS:
        write_canon(canon_d, out / "canon_deformed" / f"{shot_id}.npz")
    # per-animal amodal labels (the whole whale, hidden parts included), cropped to its projection's box
    crops = {"front_id": front, "n_animals": np.int32(n)}
    for k, m in enumerate(maps, start=1):
        box = _crop(m["amodal"] > 0)
        crops[f"a{k}_box"] = np.array(box, np.int32)
        if box[0] >= 0:
            x0, y0, x1, y1 = box; sl = (slice(y0, y1 + 1), slice(x0, x1 + 1))
            if "parts" in m:
                crops[f"a{k}_parts"] = m["parts"][sl]
            if "blade" in m:
                crops[f"a{k}_blade"] = m["blade"][sl]
            if DENSE_BODY_MAPS:
                crops[f"a{k}_canon_mm"] = encode_canon(m["canon"][sl])
                crops[f"a{k}_depth_cm"] = np.where(m["amodal"][sl] > 0, np.clip(m["zbuf"][sl], 0, 65535), 0).astype(np.uint16)
            crops[f"a{k}_front"] = (front[sl] == k)
            crops[f"a{k}_ignore"] = shallows[k - 1][sl]
    (out / "animals").mkdir(exist_ok=True)
    np.savez_compressed(out / "animals" / f"{shot_id}.npz", **crops)
    animal_meta, animal_flags = [], {}
    keypoints = None
    for k, (a, m) in enumerate(zip(animals, maps), start=1):
        g_k, cv_k, base_k = geoms[k - 1]
        vis_k = inst == k
        kps = project.project_landmarks(project.landmark_positions(points or {}, g_k), a["whale"], camera, zall, vis_k,
                                        water_z_cm=WATER_Z_CM, canonical=project.landmark_positions(points or {}, cv_k))
        for kp in kps.values():                         # which whale's surface is in front of an occluded keypoint (k itself: self-occlusion)
            if kp.get("occluded"):
                col = min(max(int(kp["uv"][0]), 0), front.shape[1] - 1); row = min(max(int(kp["uv"][1]), 0), front.shape[0] - 1)
                kp["occluded_by"] = int(front[row, col])          # (uv is rounded: 719.96 -> 720.0 on a 720-row frame)
        if k == 1:
            keypoints = kps
        chk = project.check_against_mask(m["amodal"], vis_k)
        st = mask_stats(vis_k)
        st["through_water_pixels"] = int(shallows[k - 1].sum())
        flags = []
        if vis_k.any() and 1.0 - chk["visible_inside_amodal"] > VISIBLE_OUTSIDE_AMODAL_LIMIT:
            flags.append("visible_outside_amodal")
        if st.get("mask_touches_border"):
            flags.append("mask_touches_border")
        own_front = (m["amodal"] > 0) & (front == k)
        proj_above = own_front & (np.nan_to_num(heights[k - 1], nan=-1e9) > PROJECTION_MIN_HEIGHT_CM)
        cov = float((vis_k & proj_above).sum() / max(proj_above.sum(), 1)) if proj_above.sum() > 500 else None
        if cov is not None and cov < PROJECTION_COVERED_LIMIT:
            flags.append("projection_not_covered")
        if flags:
            animal_flags[str(k)] = flags
        pinc = project.pose_in_camera(a["whale"], camera)
        if MODEL is not None and getattr(MODEL, "landmarks_v2", False):      # run06 round 5 fields (config whale.landmarks_v2)
            whale_len_px = float(np.ptp(base_k[:, 0])) / 100.0 * camera.fx / max(pinc["translation_cm"][0] / 100.0, 1e-3)
            enrich_keypoints(kps, a["whale"], waves, t_wave, vis_k, whale_len_px)
            planes = part_planes(g_k, vparts, MODEL.rest if vparts is not None else None, pinc, a.get("fluke_curl_deg"))
        else:
            planes = None
        animal_meta.append({
            "instance_id": k, "role": a.get("role") or (("mother" if any(c.get("role") == "calf" for c in animals[1:]) else "primary") if k == 1 else "adult"), "individual": a.get("individual"), "skin": a.get("skin"),
            "category": a.get("category"), "body_scale": float(a.get("body_scale", 1.0)),
            "whale_length_m": round(float(np.ptp(base_k[:, 0])) / 100.0, 3),
            "pose": a.get("pose"), "whale": a.get("whale"), "pec_droop_deg": a.get("pec_droop_deg"), "fluke_curl_deg": a.get("fluke_curl_deg"),
            "deform": a.get("deform"), "orientation": project.orientation(a["whale"], camera), "pose_in_camera": pinc,
            "pixels_per_metre_at_whale": round(camera.fx / max(pinc["translation_cm"][0] / 100.0, 1e-3), 3),
            "keypoints": kps, "keypoint_states": dict(Counter(q.get("state", "unknown") for q in kps.values())),
            "visible": st, "amodal_pixels": chk["amodal_pixels"], "amodal_front_pixels": int(own_front.sum()),
            "visible_fraction_of_whale": chk["visible_fraction"], "visible_inside_amodal": chk["visible_inside_amodal"],
            "projection_covered_by_mask": None if cov is None else round(cov, 4),
            "parts_visible": ({name: int(((part_map == pid) & vis_k).sum()) for name, pid in PART6_IDS.items()} if part_map is not None else None),
            "parts_amodal": ({name: int(((m["parts"] == pid) & own_front).sum()) for name, pid in PART6_IDS.items()}
                             if part_map is not None and m.get("parts") is not None else None),
            "pose_summary": pose_summary(kps, pinc),
            "fin_spec": _fin_spec(a.get("individual")),
            **({"part_planes": planes} if planes is not None else {}),
            "qa_flags": flags,
        })
        if "blade" in m:                                  # run06 round 5: the blade sub-label's pixels (amodal front / visible)
            animal_meta[-1]["dorsal_blade_pixels"] = {"amodal": int((m["blade"] & own_front).sum()), "visible": int((m["blade"] & vis_k).sum())}
        pa, pv = animal_meta[-1]["parts_amodal"], animal_meta[-1]["parts_visible"]
        if pa and pv:                                     # run06 round 5: visible fraction per part (visible / its unoccluded projection)
            animal_meta[-1]["parts_visible_fraction"] = {n: (round(pv[n] / pa[n], 4) if pa[n] > 0 else None) for n in pa}
    rows_visible = np.nonzero(mask.any(axis=1))[0]
    water_row = int(np.percentile(rows_visible, 99)) if len(rows_visible) else None
    whale_depth = project.pose_in_camera(shot["whale"], camera)
    stats = mask_stats(mask)
    stats["through_water_pixels"] = int(shallow.sum())
    stats["companion_pixels"] = int(others.sum())
    n_contour = int((contours > 0).sum())
    stats["contour_pixels"] = n_contour
    stats["contour_true_fraction"] = round(float((contours == 1).sum() / max(n_contour, 1)), 4)
    flags = []
    if 1.0 - check["visible_inside_amodal"] > VISIBLE_OUTSIDE_AMODAL_LIMIT and mask.any():
        flags.append("visible_outside_amodal")
    # The converse check: the part of the projection that is clearly above the water (the whale surface more than
    # PROJECTION_MIN_HEIGHT_CM above the local frozen surface, per pixel) must be covered by the rendered mask, otherwise
    # the render is smaller than the labels (e.g. a focal length the camera clamped). A 3D criterion: with an elevated
    # camera or an arched / pitched whale the submerged far body projects above the waterline row, so the image row is
    # no test of being above water. What stays uncovered legitimately is the body behind wave crests or foam, hence the
    # low limit: the flag marks a render / label mismatch, not partial occlusion.
    projection_covered = None
    if mask.any():
        proj_above = (amodal > 0) & (np.nan_to_num(height_above, nan=-1e9) > PROJECTION_MIN_HEIGHT_CM)
        projection_covered = float((mask & proj_above).sum() / max(proj_above.sum(), 1)) if proj_above.sum() > 500 else None
        if projection_covered is not None and projection_covered < PROJECTION_COVERED_LIMIT:
            flags.append("projection_not_covered")
    if not mask.any() and not shot.get("negative"):
        flags.append("empty_mask")              # the primary is not visible (another whale may be)
    if stats.get("mask_touches_border"):
        flags.append("mask_touches_border")
    if decode_stats.get("invalid_ids"):
        flags.append("label_invalid_ids")
    gate = quality_gate(out / "images" / f"{shot_id}.png", inst, projection_covered, shot) if QUALITY_GATE else None
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
        # the bend of the body as the picture shows it (whale frame), and the individual's proportions:
        # labels for a shape + pose head that scales and bends the mesh to the photo
        "spine": project.spine_curve(geometry, vertices_c),
        "individual_spec": _individual_spec(shot.get("individual")),
        "keypoints": keypoints,
        "keypoint_states": dict(Counter(k.get("state", "unknown") for k in keypoints.values())),
        "keypoint_revision": KEYPOINT_REVISION,
        "water_row_px": water_row,
        "camera_fx": fx,
        "visible": stats,
        "amodal_pixels": check["amodal_pixels"],
        "visible_fraction_of_whale": check["visible_fraction"],
        "visible_inside_amodal": check["visible_inside_amodal"],
        "projection_covered_by_mask": None if projection_covered is None else round(projection_covered, 4),
        "parts": parts,
        "parts_visible": ({name: int(((part_map == pid) & mask).sum()) for name, pid in PART6_IDS.items()} if part_map is not None else None),
        "qa_flags": flags,
        "label_encoding": label_encoding or LABEL_ENCODING_LEGACY,
        "label_decode": decode_stats,
        "n_animals": n,
        "animals": animal_meta,
        "animal_qa_flags": animal_flags,
        **({"quality_gate": gate} if gate is not None else {}),
        "paths": {"image": f"images/{shot_id}.png", "mask": f"masks/{shot_id}.png", "amodal": f"amodal/{shot_id}.png", "contours": f"contours/{shot_id}.png",
                  "bodycoords": f"bodycoords/{shot_id}.png", "depth": f"depth/{shot_id}.png", "canon": f"canon/{shot_id}.npz",
                  "canon_deformed": f"canon_deformed/{shot_id}.npz", "instances": f"instances/{shot_id}.png", "animals": f"animals/{shot_id}.npz"}
                 | ({"parts": f"parts/{shot_id}.png"} if part_map is not None else {})
                 if DENSE_BODY_MAPS else
                 {"image": f"images/{shot_id}.png", "mask": f"masks/{shot_id}.png", "amodal": f"amodal/{shot_id}.png", "contours": f"contours/{shot_id}.png",
                  "bodycoords": f"bodycoords/{shot_id}.png", "depth": f"depth/{shot_id}.png", "instances": f"instances/{shot_id}.png",
                  "animals": f"animals/{shot_id}.npz"} | ({"parts": f"parts/{shot_id}.png"} if part_map is not None else {}),
    })
    if shot.get("boat20") is not None:          # round 20f: scene classes (boat rail / tape / deck / cleat / pontoon, land), additive
        from factory import boat_scene
        boat_scene.add_scene_labels(meta, shot, raw_dir, inst, label, camera, out)
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
IMAGES_FROM: Path | None = None   # --images-from: link these finished images instead of writing new ones (relabelling a run)
DENSE_BODY_MAPS = True            # config storage.dense_body_maps (factory.storage): canon / canon_deformed / dense animal crops


_INDIVIDUALS: dict | None = None


def _individual_spec(ind) -> dict | None:
    """Proportions / fluke shape / colouring class of the shot's individual (factory.individuals), or None."""
    global _INDIVIDUALS
    if not ind:
        return None
    if _INDIVIDUALS is None:
        try:
            from factory import individuals as indiv
            lib = getattr(MODEL, "library_path", None) if MODEL is not None else None      # the run's own library (run06)
            _INDIVIDUALS = {k: v.to_dict() for k, v in (indiv.load_library(lib) if lib else indiv.load_library()).items()}
        except Exception:
            _INDIVIDUALS = {}
    spec = _INDIVIDUALS.get(ind)
    if not spec:
        return None
    keep = ("calf", "head_len", "girth", "peduncle", "fluke_span", "fluke_sweep_cm", "fluke_notch_cm", "fluke_edge_cm", "pec_len",
            "dorsal_height", "dorsal_sweep_cm", "fluke_class")
    return {k: spec[k] for k in keep if k in spec}


def _load_model(run_config: dict):
    from factory.whalemodel import WhaleModel

    model = WhaleModel.from_config(run_config.get("config", {}))
    return model if model.kind != "k9239" else None


def _init_worker(run_dir: str, mesh: str, depth_m: float, camera_fx: dict | None = None, run_config: dict | None = None) -> None:
    global CAMERA_FX, MODEL, DENSE_BODY_MAPS, IMAGES_FROM, QUALITY_GATE, DROP_ISOLATED
    CAMERA_FX = camera_fx
    QUALITY_GATE = (run_config or {}).get("config", {}).get("quality_gate")
    DROP_ISOLATED = bool(((run_config or {}).get("config", {}).get("labels") or {}).get("drop_isolated_instance_pixels", False))
    IMAGES_FROM = Path(os.environ["WF_IMAGES_FROM"]) if os.environ.get("WF_IMAGES_FROM") else None
    DENSE_BODY_MAPS = bool(((run_config or {}).get("config", {}).get("storage") or {}).get("dense_body_maps", True))
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
    for sub in ("images", "masks", "amodal", "bodycoords", "depth", "meta") + (("canon", "canon_deformed") if DENSE_BODY_MAPS else ()):
        (out / sub).mkdir(parents=True, exist_ok=True)
    per_shot = int(batch.get("frames_per_shot", 3))
    encoding = batch.get("label_encoding")             # None: from the label pass's file name (process_shot)
    metas = []
    for index, shot in enumerate(batch["shots"]):
        frame = index * per_shot + per_shot - 1
        metas.append(process_shot(shot, raw_dir, frame, out, vertices, faces, batch["resolution"], points, through_water_depth_m, encoding))
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
            "n_animals": meta.get("n_animals", 1), "scene_type": (meta.get("scene") or {}).get("type"),
            **({"rejected": meta["quality_gate"]["rejected"]} if meta.get("quality_gate") else {}),
        }))
    (run_dir / "dataset" / "manifest.jsonl").write_text("\n".join(lines) + ("\n" if lines else ""))
    return len(lines)


def relabel_run(args, mesh: Path, depth_m: float, run_config: dict) -> int:
    """--keypoints-only: every processed shot's keypoints recomputed in place (see relabel_keypoints); a summary in
    dataset/keypoint_revision.json. Exit 1 when some shots did not reproduce (they are left as they were)."""
    from multiprocessing import Pool
    import datetime
    ids = sorted(p.stem for p in (args.run_dir / "dataset" / "meta").glob("*.json"))
    if args.limit:
        ids = ids[: args.limit]
    chunk = 100
    jobs = [(ids[i: i + chunk], not args.dry_run) for i in range(0, len(ids), chunk)]
    init = (str(args.run_dir), str(mesh), depth_m, CAMERA_FX, run_config)
    moved = Counter(); unexpected = []; done = 0
    if args.workers > 1:
        with Pool(args.workers, initializer=_init_worker, initargs=init) as pool:
            for r in pool.imap_unordered(_relabel_worker, jobs):
                moved.update(r["moved"]); unexpected += r["unexpected"]; done += r["n"]
                if done % 1000 < chunk:
                    print(f"{done} / {len(ids)} shots, {len(unexpected)} not reproduced", flush=True)
    else:
        _init_worker(*init)
        for job in jobs:
            r = _relabel_worker(job)
            moved.update(r["moved"]); unexpected += r["unexpected"]; done += r["n"]
    bad_shots = sorted({u[0] for u in unexpected})
    summary = {"revision": KEYPOINT_REVISION, "date": datetime.date.today().isoformat(), "dry_run": bool(args.dry_run),
               "shots": len(ids), "moved": dict(moved), "shots_not_reproduced": len(bad_shots),
               "not_reproduced_examples": unexpected[:20]}
    print(json.dumps(summary, indent=1))
    if not args.dry_run:
        (args.run_dir / "dataset" / "keypoint_revision.json").write_text(json.dumps(summary, indent=1))
    return 1 if bad_shots else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--batch", action="append", help="batch name(s) to process (default: all rendered)")
    parser.add_argument("--mesh", type=Path, default=None, help="whale OBJ (default: from run_config.json)")
    parser.add_argument("--canon-only", action="store_true", help="only add canon/<id>.npz to an already processed run")
    parser.add_argument("--keypoints-only", action="store_true",
                        help="recompute the keypoints of the processed shots in place (revision KEYPOINT_REVISION) from the saved "
                             "depth maps and masks; needs no raw frames. Shots whose other keypoints do not reproduce are skipped")
    parser.add_argument("--dry-run", action="store_true", help="with --keypoints-only: compute and compare, write nothing")
    parser.add_argument("--limit", type=int, default=0, help="with --keypoints-only: only the first N shots (by id)")
    parser.add_argument("--workers", type=int, default=1, help="processes (batches are independent)")
    parser.add_argument("--images-from", type=Path, default=None,
                        help="relabel into a new run folder: symlink the finished images of this folder (e.g. run05/dataset/images) instead of writing them")
    parser.add_argument("--dense-body-maps", choices=("config", "on", "off"), default="config",
                        help="override storage.dense_body_maps of the run config (canon, canon_deformed, dense animal crops)")
    args = parser.parse_args(argv)
    run_config = json.loads((args.run_dir / "run_config.json").read_text())
    mesh = args.mesh or Path(run_config["mesh"])
    depth_m = float(run_config.get("config", {}).get("labels", {}).get("through_water_depth_m", THROUGH_WATER_DEPTH_M))
    global CAMERA_FX, DENSE_BODY_MAPS, IMAGES_FROM, QUALITY_GATE, DROP_ISOLATED
    CAMERA_FX = run_config.get("config", {}).get("camera_fx") or None
    QUALITY_GATE = run_config.get("config", {}).get("quality_gate")
    DROP_ISOLATED = bool((run_config.get("config", {}).get("labels") or {}).get("drop_isolated_instance_pixels", False))
    DENSE_BODY_MAPS = bool((run_config.get("config", {}).get("storage") or {}).get("dense_body_maps", True))
    if args.dense_body_maps != "config":
        DENSE_BODY_MAPS = args.dense_body_maps == "on"
        run_config.setdefault("config", {}).setdefault("storage", {})["dense_body_maps"] = DENSE_BODY_MAPS     # for the workers
    if args.images_from:
        IMAGES_FROM = args.images_from.resolve()
        os.environ["WF_IMAGES_FROM"] = str(IMAGES_FROM)                                                     # for the workers
    global MODEL
    MODEL = _load_model(run_config)
    if MODEL is not None:
        vertices, faces, groups = MODEL.canonical, MODEL.faces, MODEL.groups
    else:
        vertices, faces, groups = project.load_obj(mesh, with_groups=True)
    points = MODEL.landmark_indices() if MODEL is not None else project.landmarks(vertices, groups)
    (args.run_dir / "dataset").mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        (args.run_dir / "dataset" / "landmarks_mesh_frame_cm.json").write_text(
            json.dumps({k: np.round(v, 2).tolist() for k, v in project.landmark_positions(points, vertices).items()}, indent=1))
    if args.keypoints_only:
        return relabel_run(args, mesh, depth_m, run_config)
    if MODEL is not None and MODEL.vertex_part_ids() is not None and not args.canon_only:
        # the run06 part taxonomy as vertex groups on the rest mesh (WhaleModel.part_groups): ids per vertex, for loaders
        # that want to map canon / vertex labels to parts
        np.savez_compressed(args.run_dir / "dataset" / "part_vertex_ids.npz", ids=MODEL.vertex_part_ids(),
                            names=np.array(list(PART6_IDS)), counts=np.bincount(MODEL.vertex_part_ids(), minlength=6))
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
