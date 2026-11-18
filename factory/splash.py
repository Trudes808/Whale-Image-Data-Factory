"""Calculated splash geometry: crown sheets, ballistic droplets, foam and spout mist.

A splash is described by a kind, a centre on the water, a radius, a height and a seed;
``build_mesh`` turns that into one triangle mesh (cm, world space) and ``write_obj`` saves
it for Unreal. The editor scripts import it and render it with M_Splash (translucent
water with a noise-torn, height-faded alpha) plus a hidden coverage twin that writes
custom stencil 2, so dense white water in front of the whale labels as background.

Kinds:
    crown   impact splash (breach, slap): a ragged cluster of water plumes, each thrown on
            its own ballistic arc from the impact ring, droplets, foam patches
    wash    white water along the waterline: foam patches, a few low plumes, droplets

``amount`` scales how much white water there is (plume count, droplets).

The mesh UV's v coordinate is "height fraction" (0 at the base, 1 at the top): the
material fades alpha with it, so sheet tops and mist look thin and torn.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

KINDS = ("crown", "wash", "blow", "spray", "streams")      # blow / spray: run06 realism (factory.realism)
GRAVITY_CM_S2 = 981.0


@dataclass
class Mesh:
    vertices: np.ndarray  # (N, 3) cm
    faces: np.ndarray     # (M, 3) int
    uvs: np.ndarray       # (N, 2)

    @staticmethod
    def concat(meshes: list["Mesh"]) -> "Mesh":
        meshes = [m for m in meshes if len(m.faces)]
        if not meshes:
            return Mesh(np.zeros((0, 3)), np.zeros((0, 3), int), np.zeros((0, 2)))
        offsets = np.cumsum([0] + [len(m.vertices) for m in meshes[:-1]])
        return Mesh(np.concatenate([m.vertices for m in meshes]),
                    np.concatenate([m.faces + o for m, o in zip(meshes, offsets)]),
                    np.concatenate([m.uvs for m in meshes]))


def _fourier_noise(rng: np.random.Generator, theta: np.ndarray, harmonics=(2, 3, 5, 7), amount: float = 1.0) -> np.ndarray:
    out = np.zeros_like(theta)
    for h in harmonics:
        out += rng.uniform(0.3, 1.0) / len(harmonics) * np.sin(h * theta + rng.uniform(0, 2 * math.pi))
    return 1.0 + amount * out


def grid_faces(n_rows: int, n_cols: int, wrap: bool) -> np.ndarray:
    faces = []
    cols = n_cols if wrap else n_cols - 1
    for r in range(n_rows - 1):
        for c in range(cols):
            a, b = r * n_cols + c, r * n_cols + (c + 1) % n_cols
            d, e = (r + 1) * n_cols + c, (r + 1) * n_cols + (c + 1) % n_cols
            faces += [(a, d, e), (a, e, b)]
    return np.asarray(faces, int)


def _sheet_faces(n_theta: int, n_s: int) -> np.ndarray:
    faces = []
    for i in range(n_theta):
        i2 = (i + 1) % n_theta
        for j in range(n_s - 1):
            a, b, c, d = i * n_s + j, i2 * n_s + j, i * n_s + j + 1, i2 * n_s + j + 1
            faces += [(a, b, d), (a, d, c)]
    return np.asarray(faces, int)


def ellipsoid(centre: np.ndarray, axis: np.ndarray, length_cm: float, width_cm: float, v_uv: float,
              segments: int = 8, rings: int = 5) -> Mesh:
    """A droplet: ellipsoid elongated along ``axis``."""
    axis = axis / max(np.linalg.norm(axis), 1e-6)
    helper = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(axis, helper)
    u /= np.linalg.norm(u)
    w = np.cross(axis, u)
    verts, uvs = [], []
    for i in range(rings + 1):
        phi = math.pi * i / rings
        for j in range(segments):
            th = 2 * math.pi * j / segments
            p = centre + axis * (length_cm / 2) * math.cos(phi) + (u * math.cos(th) + w * math.sin(th)) * (width_cm / 2) * math.sin(phi)
            verts.append(p)
            uvs.append([0.5, v_uv])   # constant u: droplets are solid, no side fade
    return Mesh(np.asarray(verts), grid_faces(rings + 1, segments, wrap=True), np.asarray(uvs))


def droplets(rng: np.random.Generator, centre: np.ndarray, radius_cm: float, height_cm: float, count: int,
             elevation_deg=(45.0, 85.0), size_cm=(2.5, 10.0), ring=(0.4, 1.0), spread_deg: float = 30.0,
             age=(0.15, 1.0), v_uv: float = 0.1) -> Mesh:
    """Droplets on ballistic arcs launched from a ring around the centre."""
    parts = []
    v0_max = math.sqrt(2 * GRAVITY_CM_S2 * max(height_cm, 1.0))
    for _ in range(count):
        a = rng.uniform(0, 2 * math.pi)
        r0 = radius_cm * rng.uniform(*ring)
        p0 = centre + np.array([r0 * math.cos(a), r0 * math.sin(a), 0.0])
        elev = math.radians(rng.uniform(*elevation_deg))
        az = a + math.radians(rng.uniform(-spread_deg, spread_deg))
        speed = v0_max * rng.uniform(0.55, 1.25)
        velocity = speed * np.array([math.cos(elev) * math.cos(az), math.cos(elev) * math.sin(az), math.sin(elev)])
        flight = 2 * velocity[2] / GRAVITY_CM_S2
        t = flight * rng.uniform(*age)
        pos = p0 + velocity * t + np.array([0.0, 0.0, -0.5 * GRAVITY_CM_S2 * t * t])
        if pos[2] < centre[2] + 2.0:
            continue
        vel_now = velocity + np.array([0.0, 0.0, -GRAVITY_CM_S2 * t])
        size = rng.uniform(*size_cm)
        parts.append(ellipsoid(pos, vel_now, size * rng.uniform(1.4, 2.8), size, v_uv))
    return Mesh.concat(parts)


def foam_disc(rng: np.random.Generator, centre: np.ndarray, inner_cm: float, outer_cm: float, z_cm: float = 18.0,
              n_theta: int = 48, v_uv: float = 0.3) -> Mesh:
    theta = np.linspace(0, 2 * math.pi, n_theta, endpoint=False)
    edge = _fourier_noise(rng, theta, harmonics=(3, 5, 9, 14), amount=0.3)
    verts, uvs = [], []
    for i, t in enumerate(theta):
        for j, f in enumerate((0.0, 0.5, 1.0)):
            r = inner_cm + (outer_cm * edge[i] - inner_cm) * f
            jx, jy = rng.normal(0, 0.03 * outer_cm, size=2)
            verts.append([centre[0] + r * math.cos(t) + jx, centre[1] + r * math.sin(t) + jy, centre[2] + z_cm])
            uvs.append([(3.0 * t / (2 * math.pi)) % 1.0, v_uv + 0.5 * f])
    return Mesh(np.asarray(verts), _sheet_faces(n_theta, 3), np.asarray(uvs))


def plume(rng: np.random.Generator, centre: np.ndarray, angle: float, width_deg: float, r0_cm: float,
          v_r: float, v_z: float, n_cols: int = 9, n_rows: int = 14, jitter_cm: float = 10.0) -> Mesh:
    """One sheet of water thrown outward from the impact ring: each column of the sheet
    follows its own ballistic arc (rises, curls over, breaks up), so no two look alike."""
    verts, uvs = [], []
    for j in range(n_cols):
        a = math.radians(angle + (j / max(n_cols - 1, 1) - 0.5) * width_deg)
        r0 = r0_cm * rng.uniform(0.85, 1.15)
        vz = v_z * rng.uniform(0.7, 1.3)
        vr = v_r * rng.uniform(0.6, 1.4)
        t_end = (2 * vz / GRAVITY_CM_S2) * rng.uniform(0.45, 0.9)   # the sheet breaks before landing
        for i in range(n_rows):
            f = i / (n_rows - 1)
            t = f * t_end
            r = r0 + vr * t
            z = vz * t - 0.5 * GRAVITY_CM_S2 * t * t
            jitter = rng.normal(0, jitter_cm, size=3) * f
            verts.append([centre[0] + r * math.cos(a) + jitter[0], centre[1] + r * math.sin(a) + jitter[1],
                          centre[2] + max(z, 0.0) + jitter[2]])
            uvs.append([0.02 + 0.96 * j / max(n_cols - 1, 1), f])   # u across the sheet: sides fade out
    return Mesh(np.asarray(verts), _sheet_faces(n_cols, n_rows), np.asarray(uvs))


def plumes(rng: np.random.Generator, centre: np.ndarray, radius_cm: float, height_cm: float, count: int,
           height_range=(0.3, 1.3)) -> list[Mesh]:
    """A cluster of plumes around the ring; angles cluster rather than spread evenly."""
    parts = []
    base_angle = rng.uniform(0, 360)
    for _ in range(count):
        angle = base_angle + rng.normal(0, 70) if rng.uniform() < 0.6 else rng.uniform(0, 360)
        h = height_cm * rng.uniform(*height_range)
        v_z = math.sqrt(2 * GRAVITY_CM_S2 * max(h, 5.0))
        parts.append(plume(rng, centre, angle, rng.uniform(20, 80), radius_cm * rng.uniform(0.45, 1.0),
                           v_z * rng.uniform(0.15, 0.9), v_z, n_cols=int(rng.integers(6, 11)), jitter_cm=0.05 * radius_cm))
    return parts


def build_mesh(spec: dict) -> Mesh:
    if spec.get("style") == "soft":
        return _soft(spec)
    return _build_mesh(spec)


def _soft(spec: dict) -> Mesh:
    """run06 realism: the same white water with droplets of varied size, stretched along their flight (motion blur),
    and finer sheets."""
    global droplets
    plain = droplets

    def varied(rng, centre, radius_cm, height_cm, count, elevation_deg=(45.0, 85.0), size_cm=(2.5, 10.0), ring=(0.4, 1.0),
               spread_deg=30.0, age=(0.15, 1.0), v_uv=0.1):
        parts = []
        for k in range(3):                                  # three size classes: mist, drops, gobs
            lo, hi = [(0.6, 2.0), (2.0, 6.0), (6.0, 16.0)][k]
            parts.append(plain(rng, centre, radius_cm, height_cm, max(1, int(count * (0.55, 0.35, 0.1)[k] * 1.5)), elevation_deg,
                               (lo, hi), ring, spread_deg, age, v_uv))
        m = Mesh.concat(parts)
        return m
    droplets = varied
    try:
        return _build_mesh(spec)
    finally:
        droplets = plain


def _build_mesh(spec: dict) -> Mesh:
    kind = spec["kind"]
    if kind not in KINDS:
        raise ValueError(f"unknown splash kind {kind!r}; choose from {KINDS}")
    rng = np.random.default_rng(int(spec.get("seed", 0)))
    centre = np.asarray(spec["centre_cm"], float)
    radius, height = float(spec["radius_cm"]), float(spec["height_cm"])
    amount = float(spec.get("amount", 1.0))  # 0.3 = a few low plumes, 1.5 = a big breach crown
    if kind == "crown":
        n_plumes = max(3, int(round(rng.integers(6, 12) * amount)))
        parts = plumes(rng, centre, radius, height, n_plumes)
        parts += plumes(rng, centre, radius * 0.8, height * 0.25, max(2, n_plumes // 2), height_range=(0.5, 1.0))  # low skirt
        parts.append(droplets(rng, centre, radius, height * 1.2, count=int(90 * amount)))
        for _ in range(int(rng.integers(2, 5))):
            offset = np.array([rng.uniform(-0.5, 0.5) * radius, rng.uniform(-0.5, 0.5) * radius, 0.0])
            parts.append(foam_disc(rng, centre + offset, 0.0, radius * rng.uniform(0.5, 1.2)))
        return Mesh.concat(parts)
    if kind == "blow":
        # the breath: a bushy column of mist from the blowhole, droplets thinning out toward the top, drifting downwind
        parts = []
        drift = np.asarray(spec.get("drift_cm", [0.0, 0.0]), float)
        for k in range(int(rng.integers(5, 9))):
            h = height * rng.uniform(0.5, 1.05)
            v_z = math.sqrt(2 * GRAVITY_CM_S2 * max(h, 5.0))
            parts.append(plume(rng, centre, rng.uniform(0, 360), rng.uniform(40, 120), radius * rng.uniform(0.3, 1.0),
                               v_z * rng.uniform(0.05, 0.18), v_z, n_cols=int(rng.integers(5, 9)), n_rows=16, jitter_cm=0.25 * radius))
        parts.append(droplets(rng, centre, radius, height, count=int(160 * amount), elevation_deg=(70.0, 89.0), size_cm=(1.5, 6.0),
                              ring=(0.0, 1.0), spread_deg=60.0, age=(0.2, 0.75)))
        mesh = Mesh.concat(parts)
        z = np.clip((mesh.vertices[:, 2] - centre[2]) / max(height, 1.0), 0.0, 1.5)
        mesh.vertices[:, :2] += z[:, None] * drift[None, :]               # wind leans the column
        return mesh
    if kind == "streams":
        # water pouring off a raised fluke's trailing edge: from each edge point (spec points_rel_cm, relative to the centre)
        # a thin crossed ribbon falls to the water (centre z = the local water level), wavering, thinning and breaking into
        # drops; v runs 0 at the edge -> 1 at the bottom, so the material's fade makes the stream thin out as it falls
        if spec.get("strands"):
            return _strands(rng, spec, centre)
        parts = []
        water_z = centre[2]
        w0, w1 = spec.get("width_cm", [2.0, 7.0])
        keep = float(spec.get("keep_fraction", 0.75))
        f0, f1 = spec.get("fall_fraction", [1.0, 1.0])
        for rel in spec.get("points_rel_cm", []):
            top = centre + np.asarray(rel, float)
            fall = (top[2] - water_z) * float(rng.uniform(f0, f1))            # a stream may break up before the water
            if fall < 5.0 or rng.uniform() > keep:
                continue
            width = float(rng.uniform(w0, w1)); n = 10
            lean = rng.normal(0, 0.08, size=2) * fall
            for ang in (0.0, math.pi / 2):
                dx, dy = math.cos(ang) * width / 2, math.sin(ang) * width / 2
                verts, uvs = [], []
                for i in range(n):
                    f = i / (n - 1)
                    wob = rng.normal(0, 1.5 + 4.0 * f, size=2)
                    c = top + np.array([lean[0] * f * f + wob[0], lean[1] * f * f + wob[1], -fall * f])
                    w = 1.0 - 0.6 * f
                    verts += [c + np.array([-dx * w, -dy * w, 0.0]), c + np.array([dx * w, dy * w, 0.0])]
                    uvs += [[0.05, f], [0.95, f]]
                parts.append(Mesh(np.asarray(verts), _sheet_faces(2, n), np.asarray(uvs)))
            parts.append(droplets(rng, np.array([top[0], top[1], water_z]), 15.0, max(fall, 10.0), count=4, elevation_deg=(80.0, 90.0),
                                  size_cm=(1.5, 4.0), ring=(0.0, 0.5), spread_deg=20.0, age=(0.55, 0.95)))
        return Mesh.concat(parts)
    if kind == "spray":
        # a curtain of water thrown toward one side (azimuth_deg, e.g. toward the camera): it stands in front of the body
        az = float(spec.get("azimuth_deg", rng.uniform(0, 360)))
        parts = []
        for _ in range(max(3, int(round(rng.integers(5, 10) * amount)))):
            h = height * rng.uniform(0.4, 1.2)
            v_z = math.sqrt(2 * GRAVITY_CM_S2 * max(h, 5.0))
            parts.append(plume(rng, centre, az + rng.normal(0, 35), rng.uniform(30, 90), radius * rng.uniform(0.2, 0.8),
                               v_z * rng.uniform(0.2, 0.7), v_z, n_cols=int(rng.integers(7, 12)), jitter_cm=0.06 * radius))
        parts.append(droplets(rng, centre, radius, height * 1.1, count=int(70 * amount), size_cm=(2.0, 8.0)))
        parts.append(foam_disc(rng, centre, 0.0, radius * rng.uniform(0.6, 1.1)))
        return Mesh.concat(parts)
    # wash: white water along the waterline, a few low plumes, low droplets
    parts = []
    for _ in range(int(rng.integers(3, 6))):
        offset = np.array([rng.uniform(-1.0, 1.0) * radius, rng.uniform(-0.6, 0.6) * radius, 0.0])
        parts.append(foam_disc(rng, centre + offset, 0.0, radius * rng.uniform(0.3, 0.7)))
    parts += plumes(rng, centre, radius * 0.7, max(height, 40.0), max(2, int(round(4 * amount))), height_range=(0.4, 1.0))
    parts.append(droplets(rng, centre, radius, max(height, 40.0), count=int(40 * amount),
                          elevation_deg=(30.0, 70.0), size_cm=(2.0, 7.0)))
    return Mesh.concat(parts)


def _strands(rng: np.random.Generator, spec: dict, centre: np.ndarray) -> Mesh:
    """run06 realism round 2 (spec "strands"): water sheeting off a raised fluke as the field photos show it: many thin
    strands spaced along the whole trailing edge (the edge points are joined into a polyline), each a narrow crossed
    ribbon for its first part that then breaks into a train of falling drops, spaced wider as they fall. strands =
    {"spacing_cm": [a, b], "width_cm": [a, b], "break_fraction": [a, b], "keep": p}."""
    st = spec["strands"]
    water_z = centre[2]
    pts = [centre + np.asarray(r, float) for r in spec.get("points_rel_cm", [])]
    s0, s1 = st.get("spacing_cm", [5.0, 12.0])
    w0, w1 = st.get("width_cm", [0.4, 1.2])
    b0, b1 = st.get("break_fraction", [0.25, 0.7])
    keep = float(st.get("keep", 0.7))
    # strand roots along the polyline (segments longer than 80 cm are gaps between edge pieces, not edge)
    roots = []
    for a, b in zip(pts[:-1], pts[1:]):
        seg = b - a
        L = float(np.linalg.norm(seg[:2]))
        if L > 80.0 or L < 1e-3:
            continue
        t = float(rng.uniform(0, s0))
        while t < L:
            roots.append(a + seg * (t / L))
            t += float(rng.uniform(s0, s1))
    parts = []
    for top in roots:
        fall = top[2] - water_z
        if fall < 8.0 or rng.uniform() > keep:
            continue
        width = float(rng.uniform(w0, w1))
        brk = fall * float(rng.uniform(b0, b1))                # the ribbon part, then drops
        n = 6
        lean = rng.normal(0, 0.05, size=2) * fall
        for ang in (0.0, math.pi / 2):
            dx, dy = math.cos(ang) * width / 2, math.sin(ang) * width / 2
            verts, uvs = [], []
            for i in range(n):
                f = i / (n - 1)
                wob = rng.normal(0, 0.4 + 1.2 * f, size=2)
                c = top + np.array([lean[0] * f * f * brk / fall + wob[0], lean[1] * f * f * brk / fall + wob[1], -brk * f])
                w = 1.0 - 0.5 * f
                verts += [c + np.array([-dx * w, -dy * w, 0.0]), c + np.array([dx * w, dy * w, 0.0])]
                uvs += [[0.05, 0.3 * f], [0.95, 0.3 * f]]
            parts.append(Mesh(np.asarray(verts), _sheet_faces(2, n), np.asarray(uvs)))
        z = brk
        gap = float(rng.uniform(2.0, 5.0))
        while z < fall - 2.0:                                  # the drop train: longer gaps lower down (they accelerate)
            f = z / fall
            size = width * float(rng.uniform(0.6, 1.3))
            pos = top + np.array([lean[0] * f * f + rng.normal(0, 1.0 + 2.0 * f), lean[1] * f * f + rng.normal(0, 1.0 + 2.0 * f), -z])
            parts.append(ellipsoid(pos, np.array([0.0, 0.0, 1.0]), size * float(rng.uniform(2.0, 4.0)), size, 0.3 + 0.5 * f,
                                   segments=6, rings=3))
            z += gap
            gap *= float(rng.uniform(1.15, 1.45))
    if not parts:
        return Mesh(np.zeros((0, 3)), np.zeros((0, 3), int), np.zeros((0, 2)))
    return Mesh.concat(parts)


def write_obj(spec: dict, path) -> dict:
    mesh = build_mesh(spec)
    lines = [f"# {spec['kind']} splash, seed {spec.get('seed', 0)}, generated by factory.splash"]
    lines.extend(f"v {x:.2f} {-y:.2f} {z:.2f}" for x, y, z in mesh.vertices)   # Unreal's OBJ import mirrors Y
    # Unreal's OBJ import flips V (top-left UV origin), so write 1 - v to keep "v = height".
    lines.extend(f"vt {u:.4f} {1.0 - v:.4f}" for u, v in mesh.uvs)
    lines.extend(f"f {a}/{a} {b}/{b} {c}/{c}" for a, b, c in mesh.faces + 1)
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    return {"vertices": int(len(mesh.vertices)), "faces": int(len(mesh.faces))}


def splash_spec(kind: str, centre_cm, radius_cm: float, height_cm: float, seed: int = 0, amount: float = 1.0, **extra) -> dict:
    """The JSON-ready description used in a shot file; geometry comes from build_mesh."""
    if kind not in KINDS:
        raise ValueError(f"unknown splash kind {kind!r}; choose from {KINDS}")
    return {
        "kind": kind,
        "centre_cm": [round(float(v), 2) for v in centre_cm],
        "radius_cm": round(float(radius_cm), 2),
        "height_cm": round(float(height_cm), 2),
        "seed": int(seed),
        "amount": round(float(amount), 3),
    } | {k: v for k, v in extra.items()}


def ensure_obj(spec: dict | None, directory, name: str) -> dict | None:
    """Write the splash mesh for a shot (if any) and record its path in the spec."""
    if not spec:
        return spec
    from pathlib import Path

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.obj"
    spec["mesh"] = write_obj(spec, path)
    spec["mesh_obj"] = str(path.resolve())
    return spec
