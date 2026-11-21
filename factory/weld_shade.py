"""run06 round 12: offline flat-grey look of the posed dorsal region (no textures, no engine): each shot's camera, one
directional light from the shot's sun, Lambert + a little ambient, per-pixel interpolated smooth vertex normals (the
engine also recomputes smooth normals over positions). Old design (round 11, the blade lying on the back) and the new
welded design, cropped around each fin like closeups_pair.py, side by side with real fins.

    python3 -m factory.weld_shade RUN_DIR OUT.jpg ID:ANIMAL ... [--real SHEET.jpg]

RUN_DIR gives the shot list and run config (the same shots render in both designs); a sheet goes to OUT (keep it out of
the repository: photo / render sheets live in the scratchpad)."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

T = 400


def _animal(shots: dict, sid: str, k: int) -> dict:
    s = shots[sid]
    if int(k) == 1:
        return s
    return [c for c in s.get("companions") or [] if int(c.get("instance_id")) == int(k)][0]


def vertex_normals(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    tri = V[F]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    _, pid = np.unique(np.round(V, 3), axis=0, return_inverse=True); pid = pid.ravel()
    acc = np.zeros((int(pid.max()) + 1, 3))
    for k in range(3):
        np.add.at(acc, pid[F[:, k]], fn)
    N = acc[pid]
    used = np.zeros(len(V), bool); used[F.ravel()] = True
    c = V[used].mean(0)
    if (np.einsum("ij,ij->i", fn, tri.mean(1) - c) < 0).mean() > 0.5:
        N = -N
    return N / (np.linalg.norm(N, axis=1, keepdims=True) + 1e-12)


def shade_crop(model, animal: dict, shot: dict, res, landmarks=("dorsal_fin_tip", "dorsal_fin_front", "dorsal_fin_back")):
    """(T, T) grey crop of the animal's fin, the shot's camera and sun."""
    from factory import project
    W_, H_ = res
    V = np.asarray(model.posed(animal), float)
    world = project.transform_mesh(V, animal["whale"])
    cam = project.Camera(shot["camera"]["location"], shot["camera"]["rotation"], shot["camera"]["focal_mm"], W_, H_)
    idx = model.landmark_indices()
    P = np.array([world[idx[n]].mean(0) if isinstance(idx[n], list) else world[idx[n]] for n in landmarks])
    uvk, _ = cam.project(P)
    c = uvk.mean(0); side = max(np.ptp(uvk[:, 0]), np.ptp(uvk[:, 1])) * 1.5 + 40
    x0, y0 = c[0] - side / 2, c[1] - side / 2
    F = np.asarray(model.faces)
    uv, depth = cam.project(world)
    # faces inside the crop window
    tu = uv[F]
    inwin = (tu[:, :, 0].max(1) >= x0) & (tu[:, :, 0].min(1) <= x0 + side) & (tu[:, :, 1].max(1) >= y0) & (tu[:, :, 1].min(1) <= y0 + side) & (depth[F] > 1).all(1)
    Fs = F[inwin]
    N = vertex_normals(world, F)
    n_px = int(math.ceil(side))
    sc = T / side
    uv2 = (uv - [x0, y0]) * sc
    img, zb, att = project.rasterize(uv2, depth, Fs, T, T, vertex_attributes=N, return_extra=True)
    sun = shot.get("sun") or {}
    p, y = math.radians(sun.get("pitch", -45.0)), math.radians(sun.get("yaw", 0.0))
    L = -np.array([math.cos(p) * math.cos(y), math.cos(p) * math.sin(y), math.sin(p)])
    n = att / (np.linalg.norm(att, axis=-1, keepdims=True) + 1e-12)
    lam = np.clip((n * L).sum(-1), 0, 1)
    g = 0.12 + 0.8 * lam
    out = np.where(img > 0, g, 0.0)
    rgb = np.stack([out * 0.95 + (img == 0) * 0.20, out * 0.95 + (img == 0) * 0.32, out + (img == 0) * 0.5], -1)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def main(argv=None) -> int:
    from PIL import Image, ImageDraw
    from factory.whalemodel import WhaleModel
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir"); ap.add_argument("out"); ap.add_argument("items", nargs="+")
    ap.add_argument("--real", help="a fins-vs-real sheet whose first row of real fins is shown as a reference row")
    ap.add_argument("--res", default="2560,1707")
    a = ap.parse_args(argv)
    run = Path(a.run_dir)
    cfg = json.loads((run / "run_config.json").read_text())["config"]
    shots = {json.loads(l)["id"]: json.loads(l) for l in (run / "shotlist.jsonl").read_text().splitlines() if l.strip()}
    res = tuple(int(x) for x in a.res.split(","))
    old_cfg = json.loads(json.dumps(cfg)); old_cfg.setdefault("whale", {}).pop("blade_weld", None)
    new_cfg = json.loads(json.dumps(cfg)); new_cfg.setdefault("whale", {})["blade_weld"] = True
    models = [("round 11 (blade on the back)", WhaleModel.from_config(old_cfg)), ("round 12 (welded)", WhaleModel.from_config(new_cfg))]
    items = [s.split(":") for s in a.items]
    cols = 4
    nrow_items = (len(items) + cols - 1) // cols
    extra = 1 if a.real else 0
    sheet = Image.new("RGB", (cols * T, (nrow_items * 2 + extra) * (T + 14)), (0, 0, 0)); d = ImageDraw.Draw(sheet)
    for j, (sid, k) in enumerate(items):
        x, y = (j % cols) * T, (j // cols) * 2 * (T + 14)
        an = _animal(shots, sid, int(k))
        for r, (lab, m) in enumerate(models):
            try:
                sheet.paste(Image.fromarray(shade_crop(m, an, shots[sid], res)), (x, y + r * (T + 14) + 14))
            except Exception as exc:
                d.text((x + 4, y + r * (T + 14) + 30), str(exc)[:60], fill=(255, 80, 80))
            d.text((x + 3, y + r * (T + 14)), f"{sid} a{k} {an.get('individual')} {lab}", fill=(255, 255, 0))
    if a.real:
        real = Image.open(a.real).convert("RGB")
        # the reference sheet's first row: 8 real fins, 249 px tiles under a 12 px label
        row = real.crop((0, 0, real.width, 260))
        row = row.resize((cols * T, int(260 * cols * T / real.width)))
        yy = nrow_items * 2 * (T + 14)
        sheet.paste(row.crop((0, 0, cols * T, min(row.height, T + 14))), (0, yy))
        d.text((3, yy + 2), "REAL fins (reference row of the round-11 sheet)", fill=(255, 120, 120))
    if max(sheet.size) > 3000:
        s = 3000 / max(sheet.size); sheet = sheet.resize((int(sheet.width * s), int(sheet.height * s)))
    sheet.save(a.out, quality=92)
    print("->", a.out, sheet.size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
