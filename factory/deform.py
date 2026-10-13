"""Procedural body deformation of the whale mesh: the "animation" a rig would give.

The humpback mesh is rigid, so this module bends it. Along the body axis (x, tail at
-L/2 to snout at +L/2) a smooth bending profile turns the spine: an arch over the back, a
curl of the tail stock and flukes (up or down), a head pitch, a lateral flex (swim stroke)
and a fluke twist. The pectoral fins rotate about their roots (raise, sweep). Vertices
keep their indices, so canonical body coordinates, part ids and landmarks carry over.

    params = DeformParams(arch_deg=12, tail_curl_deg=-35, head_pitch_deg=8, lateral_deg=10, ...)
    deformed = deform(vertices_cm, params)     # same shape as vertices_cm

Angles are degrees; positive arch bends the back upward (nose and tail down), positive
tail curl lifts the flukes, positive head pitch lifts the snout, positive lateral bends
the tail toward +y (the whale's right), positive fin raise lifts the fins toward the back.
Each shot stores its parameters; the post-processing step regenerates the deformed mesh
from them (deterministic), so no per-shot mesh is kept in the dataset.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields

import numpy as np

# Body axis milestones on the K9239 humpback mesh, as fractions of length from the tail.
TAIL_START, TAIL_END = 0.0, 0.32      # flukes and peduncle
BODY_START, BODY_END = 0.2, 0.85      # arch region
HEAD_START = 0.75                     # head pitch region (hinge at HEAD_START, moves the snout)


@dataclass
class DeformParams:
    arch_deg: float = 0.0        # back arched up (+) or sagging (-), total bend across the body
    tail_curl_deg: float = 0.0   # flukes curled up (+) or down (-)
    head_pitch_deg: float = 0.0  # snout up (+) or down (-)
    lateral_deg: float = 0.0     # tail swept toward +y (+): the swim stroke
    fluke_twist_deg: float = 0.0 # roll of the flukes about the body axis
    fin_raise_deg: float = 0.0   # pectoral fins rotated up toward the back (+) or down (-)
    fin_sweep_deg: float = 0.0   # pectoral fins swept back (+) or forward (-)
    arch_centre: float = 0.0     # body fraction (0 tail .. 1 snout) at the arch's apex; 0 = the default BODY_START..BODY_END region
    arch_span: float = 0.0       # length of the arch region (body fraction) when arch_centre is set

    def is_identity(self) -> bool:
        return all(abs(getattr(self, f.name)) < 1e-9 for f in fields(self))


def sample_params(rng: np.random.Generator, category: str, strength: float = 1.0) -> DeformParams:
    """Plausible deformation for a pose category (degrees), scaled by strength."""
    u = lambda lo, hi: float(rng.uniform(lo, hi)) * strength  # noqa: E731
    category = {"fin_tip": "pec_up", "fluke_edge": "fluke_up", "peduncle": "arch", "back_arch": "arch", "spyhop_low": "spyhop", "roll_belly": "breach", "blow_sliver": "blow"}.get(category, category)
    if category == "fluke_up":
        p = DeformParams(arch_deg=u(0, 10), tail_curl_deg=u(-25, 25), lateral_deg=u(-15, 15), fluke_twist_deg=u(-15, 15))
    elif category == "arch":
        p = DeformParams(arch_deg=u(10, 30), tail_curl_deg=u(-20, 10), head_pitch_deg=u(-10, 5), lateral_deg=u(-8, 8))
    elif category == "breach":
        p = DeformParams(arch_deg=u(-15, 15), tail_curl_deg=u(-30, 30), lateral_deg=u(-25, 25), fluke_twist_deg=u(-25, 25),
                         fin_raise_deg=u(-20, 60), fin_sweep_deg=u(-20, 20))
    elif category == "pec_up":
        p = DeformParams(arch_deg=u(-5, 10), lateral_deg=u(-15, 15), fin_raise_deg=u(20, 70), fin_sweep_deg=u(-15, 15))
    elif category == "spyhop":
        p = DeformParams(arch_deg=u(-10, 10), head_pitch_deg=u(-5, 10), fin_raise_deg=u(-10, 40))
    else:  # logging, blow, awash, submerged: gentle
        p = DeformParams(arch_deg=u(-6, 8), tail_curl_deg=u(-12, 12), head_pitch_deg=u(-5, 5), lateral_deg=u(-10, 10),
                         fin_raise_deg=u(-10, 25), fin_sweep_deg=u(-10, 10))
    return p


def _smooth(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _bend_profile(frac: np.ndarray, params: DeformParams) -> tuple[np.ndarray, np.ndarray]:
    """Pitch (about y) and yaw (about z) of the spine, in radians, per body fraction."""
    pitch = np.zeros_like(frac)
    # Arch: a bump of curvature over the body, so the angle goes from -A/2 at the tail end of
    # the region to +A/2 at the head end (back curves upward in between).
    if params.arch_deg:
        lo, hi = BODY_START, BODY_END
        if params.arch_centre:
            span = params.arch_span or (BODY_END - BODY_START)
            lo, hi = params.arch_centre - span / 2, params.arch_centre + span / 2
        t = _smooth((frac - lo) / (hi - lo))
        pitch += np.radians(params.arch_deg) * (t - 0.5)
    # Tail curl: angle grows toward the fluke tips (frac -> 0).
    if params.tail_curl_deg:
        t = _smooth((TAIL_END - frac) / (TAIL_END - TAIL_START))
        pitch -= np.radians(params.tail_curl_deg) * t      # minus: lifting the tail means the spine points up toward -x
    # Head pitch: angle grows toward the snout.
    if params.head_pitch_deg:
        t = _smooth((frac - HEAD_START) / (1.0 - HEAD_START))
        pitch += np.radians(params.head_pitch_deg) * t
    yaw = np.zeros_like(frac)
    if params.lateral_deg:
        t = _smooth((TAIL_END + 0.15 - frac) / (TAIL_END + 0.15))
        yaw -= np.radians(params.lateral_deg) * t
    return pitch, yaw


def deform(vertices_cm: np.ndarray, params: DeformParams, fin_regions: dict | None = None) -> np.ndarray:
    """Bend the mesh. Returns a new (N, 3) array."""
    v = np.asarray(vertices_cm, float)
    if params.is_identity():
        return v.copy()
    x = v[:, 0]
    x_min, x_max = x.min(), x.max()
    length = x_max - x_min
    frac = (x - x_min) / length
    # Integrate the spine from the body's mid-length outward on a fine grid, then map vertices.
    grid = np.linspace(0.0, 1.0, 513)
    pitch_g, yaw_g = _bend_profile(grid, params)
    mid = 0.5
    mid_i = int(np.argmin(np.abs(grid - mid)))
    pos = np.zeros((len(grid), 3))
    dx = length / (len(grid) - 1)
    for i in range(mid_i + 1, len(grid)):
        pitch_m, yaw_m = 0.5 * (pitch_g[i] + pitch_g[i - 1]), 0.5 * (yaw_g[i] + yaw_g[i - 1])
        pos[i] = pos[i - 1] + dx * np.array([np.cos(pitch_m) * np.cos(yaw_m), np.cos(pitch_m) * np.sin(yaw_m), np.sin(pitch_m)])
    for i in range(mid_i - 1, -1, -1):
        pitch_m, yaw_m = 0.5 * (pitch_g[i] + pitch_g[i + 1]), 0.5 * (yaw_g[i] + yaw_g[i + 1])
        pos[i] = pos[i + 1] - dx * np.array([np.cos(pitch_m) * np.cos(yaw_m), np.cos(pitch_m) * np.sin(yaw_m), np.sin(pitch_m)])
    # Per vertex: spine point and local frame at its body fraction.
    spine = np.stack([np.interp(frac, grid, pos[:, k]) for k in range(3)], axis=1)
    pitch = np.interp(frac, grid, pitch_g)
    yaw = np.interp(frac, grid, yaw_g)
    roll = np.zeros_like(frac)
    if params.fluke_twist_deg:
        roll = np.radians(params.fluke_twist_deg) * _smooth((TAIL_END - frac) / (TAIL_END - TAIL_START))
    # Local offsets from the straight spine (y, z), rotated by the local frame.
    offset = np.stack([np.zeros_like(x), v[:, 1], v[:, 2]], axis=1)
    cp, sp, cy, sy, cr, sr = np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw), np.cos(roll), np.sin(roll)
    # roll about x, then pitch about y, then yaw about z (applied to the offset vector)
    oy = offset[:, 1] * cr - offset[:, 2] * sr
    oz = offset[:, 1] * sr + offset[:, 2] * cr
    ox = np.zeros_like(oy)
    # pitch about y: x' = x cos + z sin ; z' = -x sin + z cos
    px = ox * cp + oz * sp
    pz = -ox * sp + oz * cp
    py = oy
    # yaw about z
    qx = px * cy - py * sy
    qy = px * sy + py * cy
    qz = pz
    out = spine + (x_min + mid * length) * np.array([1.0, 0.0, 0.0]) + np.stack([qx, qy, qz], axis=1)
    # Re-anchor so the mid-body point stays where it was (spine integration started at 0).
    out[:, 0] += 0.0
    if fin_regions and (params.fin_raise_deg or params.fin_sweep_deg):
        out = rotate_fins(out, fin_regions, params)
    return out


def fin_regions(vertices_cm: np.ndarray) -> dict:
    """Vertex indices of the left and right pectoral fins and their root hinge points.

    On this model the fins hang down from the body between 10 % and 45 % of the length from
    the snout; fin vertices are those well outside the body's half-width or well below the
    belly in that range."""
    v = np.asarray(vertices_cm, float)
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    x_min, x_max = x.min(), x.max()
    frac = (x - x_min) / (x_max - x_min)
    in_range = (frac > 0.45) & (frac < 0.78)
    body_core = in_range & (np.abs(y) < 60)
    belly = np.percentile(z[body_core], 5) if body_core.any() else z.min()
    regions = {}
    for name, side in (("left", -1), ("right", 1)):
        sel = in_range & (side * y > 90) & (z < belly + 60)
        idx = np.where(sel)[0]
        if len(idx) < 20:
            continue
        root = np.array([x[idx].mean(), side * 100.0, belly + 40.0])
        regions[name] = {"indices": idx, "root": root, "side": side}
    return regions


def rotate_fins(vertices: np.ndarray, regions: dict, params: DeformParams) -> np.ndarray:
    out = vertices.copy()
    for name, region in regions.items():
        idx, root, side = region["indices"], region["root"], region["side"]
        rel = out[idx] - root
        # Raise: rotate about the body axis (x) so the fin tip moves up toward the back.
        a = np.radians(params.fin_raise_deg) * side
        y, z = rel[:, 1], rel[:, 2]
        rel[:, 1], rel[:, 2] = y * np.cos(a) - z * np.sin(a), y * np.sin(a) + z * np.cos(a)
        # Sweep: rotate about the vertical axis so the tip moves back toward the tail.
        b = np.radians(params.fin_sweep_deg) * side
        x, y = rel[:, 0], rel[:, 1]
        rel[:, 0], rel[:, 1] = x * np.cos(b) - y * np.sin(b), x * np.sin(b) + y * np.cos(b)
        # Blend near the root so the fin does not tear off the body.
        dist = np.linalg.norm(vertices[idx] - root, axis=1)
        w = _smooth(dist / 150.0)[:, None]
        out[idx] = root + rel * w + (vertices[idx] - root) * (1 - w)
    return out


def params_to_dict(params: DeformParams) -> dict:
    return {k: round(float(v), 3) for k, v in asdict(params).items()}


def params_from_dict(d: dict | None) -> DeformParams:
    return DeformParams(**{k: float(v) for k, v in (d or {}).items() if k in {f.name for f in fields(DeformParams)}})


def write_obj_with_vertices(source_obj, vertices_cm: np.ndarray, out_path) -> None:
    """Copy an OBJ, replacing its vertex positions (same order), keeping uv/normals/faces."""
    out = []
    index = 0
    with open(source_obj) as handle:
        for line in handle:
            if line.startswith("v "):
                x, y, z = vertices_cm[index]
                out.append(f"v {x:.4f} {-y:.4f} {z:.4f}\n")   # Unreal's OBJ import mirrors Y
                index += 1
            elif line.startswith("vn "):
                continue  # normals no longer match; Unreal recomputes them
            else:
                out.append(line)
    if index != len(vertices_cm):
        raise ValueError(f"{source_obj} has {index} vertices, got {len(vertices_cm)}")
    # faces reference v/vt/vn; drop the vn index so the importer computes normals
    fixed = []
    for line in out:
        if line.startswith("f "):
            parts = [t.split("/") for t in line.split()[1:]]
            line = "f " + " ".join(f"{p[0]}/{p[1]}" if len(p) > 1 else p[0] for p in parts) + "\n"
        fixed.append(line)
    with open(out_path, "w") as handle:
        handle.writelines(fixed)


def write_batch_meshes(batch_path, base_obj, out_dir) -> dict:
    """Write the deformed whale OBJ of every shot in a batch that has a deformation, and
    return a copy of the batch with ``whale.mesh_obj`` filled in (saved as batch.json in
    out_dir). Used by tools/render_batch.sh right before building the batch."""
    import json
    from pathlib import Path

    from factory import project

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    batch = json.loads(Path(batch_path).read_text())
    if batch.get("whale_kind", "k9239") != "k9239":
        from factory.whalemodel import WhaleModel

        model = WhaleModel(batch["whale_kind"])
        for shot in batch["shots"]:
            path = out_dir / f"{shot['id']}.obj"
            model.write_obj(model.posed(shot), path, f"{batch['whale_kind']} posed for shot {shot['id']} (clip t {(shot.get('pose') or {}).get('clip_t')})")
            shot["whale"]["mesh_obj"] = str(path.resolve())
            if shot.get("companion"):
                cpath = out_dir / f"{shot['id']}c.obj"
                model.write_obj(model.posed(shot["companion"]), cpath, f"{batch['whale_kind']} companion of shot {shot['id']}")
                shot["companion"]["whale"]["mesh_obj"] = str(cpath.resolve())
        (out_dir / "batch.json").write_text(json.dumps(batch))
        return {"shots": len(batch["shots"]), "meshes": len(batch["shots"])}
    vertices, _ = project.load_obj(base_obj)
    regions = fin_regions(vertices)
    count = 0
    for shot in batch["shots"]:
        params = params_from_dict(shot.get("deform"))
        scale = float(shot.get("body_scale", 1.0))
        if params.is_identity() and abs(scale - 1.0) < 1e-6:
            continue
        path = out_dir / f"{shot['id']}.obj"
        base = vertices * scale if abs(scale - 1.0) > 1e-6 else vertices
        reg = regions if abs(scale - 1.0) < 1e-6 else fin_regions(base)
        write_obj_with_vertices(base_obj, deform(base, params, reg) if not params.is_identity() else base, path)
        shot["whale"]["mesh_obj"] = str(path.resolve())
        count += 1
    (out_dir / "batch.json").write_text(json.dumps(batch))
    return {"shots": len(batch["shots"]), "meshes": count}


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 4:
        sys.exit("usage: python3 -m factory.deform BATCH.json BASE_WHALE.obj OUT_DIR")
    print(write_batch_meshes(sys.argv[1], sys.argv[2], sys.argv[3]))
