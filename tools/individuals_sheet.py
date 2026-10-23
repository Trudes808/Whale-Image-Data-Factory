"""Review sheet: one tile per rendered shot, grouped by individual, with the fin / fluke keypoints and edge chains drawn.

    python3 tools/individuals_sheet.py RUN_DIR [--out RUN_DIR/individuals_sheet.png] [--cols 6] [--tile 420]

Reads RUN_DIR/dataset/{images,meta}. Each tile: the whale crop (amodal box + margin) with the keypoints coloured by
state (green visible, blue under water, orange hidden, red occluded), the dorsal fin trailing edge (magenta) and
leading edge (cyan) and the fluke trailing edge (yellow) as polylines over their visible points, and a caption with
the individual id, pose category and the individual's fin / fluke / skin parameters (from the library).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

STATE_COL = {"visible": (80, 220, 80), "under_water": (230, 150, 60), "hidden": (60, 170, 255), "occluded": (60, 60, 230), "out_of_frame": (120, 120, 120), "missing": (120, 120, 120)}
POINTS = ["dorsal_fin_tip", "dorsal_fin_front", "dorsal_fin_back", "fluke_tip_left", "fluke_tip_right", "fluke_notch", "tail_base_top", "tail_base_bottom", "snout", "blowhole"]
CHAINS = {"fin_te": (255, 0, 255), "fin_le": (255, 255, 0), "fluke_te": (0, 220, 255)}


def chain_points(kps: dict, prefix: str) -> list[tuple[np.ndarray, str]]:
    names = sorted(k for k in kps if k.startswith(prefix + "_"))
    return [(np.array(kps[n]["uv"], float), kps[n].get("state", "?")) for n in names]


def tile_for(run: Path, meta: dict, size: int, lib: dict) -> np.ndarray:
    img = cv2.imread(str(run / "dataset" / "images" / f"{meta['id']}.png"))
    if img is None:
        img = cv2.imread(str(run / "dataset" / "images" / f"{meta['id']}.jpg"))
    H, W = img.shape[:2]
    kps = meta.get("keypoints", {})
    uv = np.array([k["uv"] for k in kps.values() if k.get("in_frame")]) if kps else np.zeros((0, 2))
    am = meta.get("amodal_bbox") or meta.get("bbox")
    if am:
        x0, y0, x1, y1 = am
    elif len(uv):
        x0, y0 = uv.min(0); x1, y1 = uv.max(0)
    else:
        x0, y0, x1, y1 = 0, 0, W, H
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2; side = max(x1 - x0, y1 - y0) * 1.3 + 40
    X0, Y0 = int(max(cx - side / 2, 0)), int(max(cy - side / 2, 0)); X1, Y1 = int(min(cx + side / 2, W)), int(min(cy + side / 2, H))
    crop = img[Y0:Y1, X0:X1].copy(); sc = size / max(crop.shape[:2], default=[1])
    crop = cv2.resize(crop, (int(crop.shape[1] * sc), int(crop.shape[0] * sc)))

    def P(p):
        return (int((p[0] - X0) * sc), int((p[1] - Y0) * sc))
    for prefix, col in CHAINS.items():
        pts = chain_points(kps, prefix); vis = [(p, s) for p, s in pts if s in ("visible", "hidden")]
        for (p, s), (q, t) in zip(pts[:-1], pts[1:]):
            if s in ("visible", "hidden") and t in ("visible", "hidden"):
                cv2.line(crop, P(p), P(q), col, 2)
        for p, s in pts:
            if s in ("visible", "hidden"):
                cv2.circle(crop, P(p), 2, col, -1)
    for name in POINTS:
        k = kps.get(name)
        if not k or not k.get("in_frame"):
            continue
        c = STATE_COL.get(k.get("state", "?"), (200, 200, 200)); cv2.circle(crop, P(k["uv"]), 6, c, 2)
        cv2.putText(crop, name.replace("dorsal_fin_", "df_").replace("fluke_", "fl_").replace("tail_base_", "tb_"), (P(k["uv"])[0] + 7, P(k["uv"])[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, c, 1)
    canvas = np.zeros((size + 54, size, 3), np.uint8); canvas[:crop.shape[0], :crop.shape[1]] = crop[:size, :size]
    ind = meta.get("individual") or "?"; spec = lib.get(ind, {})
    cap1 = f"{ind}  {meta.get('category', '?')}  {meta['id']}"
    cap2 = f"fin h{spec.get('dorsal_height', 1):.2f} sw{spec.get('dorsal_sweep_cm', 0):+.0f} edge{spec.get('dorsal_edge_cm', 0):.0f} n{spec.get('dorsal_nicks', 0)}"
    cap3 = f"fluke sp{spec.get('fluke_span', 1):.2f} sw{spec.get('fluke_sweep_cm', 0):+.0f} notch{spec.get('fluke_notch_cm', 0):.0f} edge{spec.get('fluke_edge_cm', 0):.0f} n{spec.get('fluke_nicks', 0)} tip{spec.get('fluke_tip_loss_cm', 0):.0f} cls{spec.get('fluke_class', '?')}"
    for i, c in enumerate((cap1, cap2, cap3)):
        cv2.putText(canvas, c, (4, size + 15 + 17 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1)
    return canvas


PART_NAMES = {"fin": ("dorsal_fin_tip", "dorsal_fin_front", "dorsal_fin_back"), "fluke": ("fluke_tip_left", "fluke_tip_right", "fluke_notch")}
PART_CHAINS = {"fin": ("fin_te", "fin_le"), "fluke": ("fluke_te",)}


def closeup(run: Path, meta: dict, part: str, size: int) -> np.ndarray:
    """Zoom on the fin or the flukes with the chain points drawn one by one (state-coloured dots, the chain as a line)."""
    img = cv2.imread(str(run / "dataset" / "images" / f"{meta['id']}.png")); H, W = img.shape[:2]
    kps = meta.get("keypoints", {})
    names = [n for n in kps if (n in PART_NAMES[part] or n.startswith(PART_CHAINS[part])) and kps[n].get("in_frame")]
    canvas = np.zeros((size + 54, size, 3), np.uint8)
    if not names:
        return canvas
    pts = np.array([kps[n]["uv"] for n in names], float); c = pts.mean(0); r = max(np.ptp(pts, axis=0).max(), 24) * 0.75 + 16
    x0, y0 = int(max(c[0] - r, 0)), int(max(c[1] - r, 0)); x1, y1 = int(min(c[0] + r, W)), int(min(c[1] + r, H))
    crop = img[y0:y1, x0:x1]; sc = size / max(crop.shape[:2]); crop = cv2.resize(crop, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)

    def P(p):
        return (int((p[0] - x0) * sc), int((p[1] - y0) * sc))
    for prefix in PART_CHAINS[part]:
        col = CHAINS[prefix]; prev = None
        for p, st in chain_points(kps, prefix):
            q = P(p); dot = STATE_COL.get(st, (120, 120, 120))
            if prev is not None and st == "visible":
                cv2.line(crop, prev, q, col, 1)                        # the line follows visible points only
            if st == "visible":
                cv2.circle(crop, q, 3, dot, -1); prev = q
            elif st in ("hidden", "under_water"):
                cv2.circle(crop, q, 2, dot, 1)
    for n in PART_NAMES[part]:
        k = kps.get(n)
        if k and k.get("in_frame"):
            cv2.circle(crop, P(k["uv"]), 7, STATE_COL.get(k.get("state"), (200, 200, 200)), 2)
    canvas[:min(crop.shape[0], size), :min(crop.shape[1], size)] = crop[:size, :size]
    cv2.putText(canvas, f"{meta.get('individual')} {part} close-up  {meta['id']}  x{sc:.1f}", (4, size + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1)
    cv2.putText(canvas, "dots: green visible, orange hidden, blue under water, red occluded", (4, size + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 170, 170), 1)
    return canvas


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path); p.add_argument("--out", type=Path, default=None); p.add_argument("--cols", type=int, default=6); p.add_argument("--tile", type=int, default=420)
    p.add_argument("--per-individual", type=int, default=3); p.add_argument("--library", type=Path, default=Path("third_party_assets/humpback_4k/individuals.json"))
    p.add_argument("--per-page", type=int, default=0, help="also write pages of this many individuals (individuals_sheet_pNN.png)")
    a = p.parse_args(argv)
    lib = json.loads(a.library.read_text()) if a.library.exists() else {}
    metas = [json.loads(f.read_text()) for f in sorted((a.run_dir / "dataset" / "meta").glob("*.json"))]
    by_ind = defaultdict(list)
    for m in metas:
        if not m.get("no_whale_in_frame"):
            by_ind[m.get("individual") or "?"].append(m)
    tiles = []; tiles_by_ind = {}
    for ind in sorted(by_ind):
        shots = by_ind[ind]
        # one row per individual: each shot (fluke poses first) with a close-up of its part next to it
        shots.sort(key=lambda m: (0 if m.get("category") in ("fluke_up", "fluke_edge", "fluke_down") else 1, m["id"]))
        picked = shots[: a.per_individual]
        row = []
        for m in picked:
            kind = "fluke" if m.get("category") in ("fluke_up", "fluke_edge", "fluke_down") else "fin"
            row.append(tile_for(a.run_dir, m, a.tile, lib)); row.append(closeup(a.run_dir, m, kind, a.tile))
        tiles_by_ind[ind] = row
        tiles += row
    if a.per_page:
        inds = sorted(tiles_by_ind)
        for page, start in enumerate(range(0, len(inds), a.per_page), 1):
            pt = [t for i in inds[start:start + a.per_page] for t in tiles_by_ind[i]]
            while len(pt) % a.cols:
                pt.append(np.zeros_like(pt[0]))
            pg = np.concatenate([np.concatenate(pt[i:i + a.cols], 1) for i in range(0, len(pt), a.cols)], 0)
            out_p = (a.out or (a.run_dir / "individuals_sheet.png")).with_name(f"individuals_sheet_p{page:02d}.png")
            cv2.imwrite(str(out_p), pg); print(f"page {page}: {inds[start]}..{inds[min(start + a.per_page, len(inds)) - 1]} -> {out_p}")
    while len(tiles) % a.cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.concatenate(tiles[i:i + a.cols], 1) for i in range(0, len(tiles), a.cols)]
    sheet = np.concatenate(rows, 0)
    out = a.out or (a.run_dir / "individuals_sheet.png")
    cv2.imwrite(str(out), sheet); print(f"{len(tiles)} tiles, {len(by_ind)} individuals -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
