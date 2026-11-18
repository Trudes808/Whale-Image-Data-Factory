"""run06 round 9: per-animal UVs for the parametric dorsal blade (factory.blade), replacing WhaleModel._blade_back_uv.

The round-8 UVs were computed once from the default blade and clipped to a patch of the back: big blades were stretched up
to 8x and the hump's front collapsed into a line in UV (62 of 1969 vertices pinned), so the skin's normal and roughness
maps were sampled along a sliver with bad tangents (pale, crumpled plates and parallel ridges in the renders).

Here each animal's blade gets two planar side charts from its own blade-local cm (a along the base, h up):
  right side (ring points i < NC):  (s, t) = ( a, h)
  left side  (ring points i >= NC): (s, t) = (-a, h)   (mirrored in a, so both sides keep the same tangent handedness)
scaled to the back's texel density (DENSITY: 2.6 texels per cm on the 4096 map), turned onto the long axis of the skin
maps' always-dark region (the upper back, where factory.skin_run06 never keeps white) and placed inside it, the two
charts next to each other; a very large blade steps its scale down to 0.7 x. A planar projection of the near-vertical
sides keeps the stretch even (a cylinder unwrap is sheared by the fin's rake: tried, anisotropy 3-8x). The seams are the
leading and trailing edges and the tip: faces across them belong to the right chart, and seam_split gives the left side's
seam vertices (and the tip, for the left half of the top fan) duplicate copies at the OBJ level, so no face spans both
charts.

    uv, info = blade_uv(local_vertices)                    # local blade cm (factory.blade.make()["vertices"])
    V2, F2, UV2 = seam_split(V, F, uv, info, n0)           # n0 = index of the blade's first vertex in V
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from factory import blade as bl

REPO = Path(__file__).resolve().parents[1]
MASK_PNG = REPO / "third_party_assets" / "humpback_4k" / "blade_uv_mask.png"   # the dark region, eroded (make_mask)
DENSITY = 2.6 / 4096.0                 # UV per cm: the back's texel density beside the fin (2.6 texels / cm at 4096)
CENTRE = np.array([0.4014, 0.4629])    # the dark region's inscribed centre (UV)
AXIS = np.array([0.854, 0.520])        # its long axis (UV, unit)
ACROSS = np.array([-AXIS[1], AXIS[0]])
GAP_CM = 6.0                           # between the two charts
_MASK = None


def _mask():
    global _MASK
    if _MASK is None:
        from PIL import Image
        _MASK = np.asarray(Image.open(MASK_PNG).convert("L")) > 127 if MASK_PNG.exists() else None
    return _MASK


def _inside(uv: np.ndarray) -> bool:
    m = _mask()
    if m is None:
        return bool(np.linalg.norm(uv - CENTRE, axis=1).max() < 0.14)
    n = m.shape[0]
    x = np.clip((uv[:, 0] * n).astype(int), 0, n - 1); y = np.clip(((1 - uv[:, 1]) * n).astype(int), 0, n - 1)
    return bool(m[y, x].all())


def _side(n_verts: int) -> np.ndarray:
    """0 = right chart, 1 = left chart, per blade-local vertex (the tip is right)."""
    R = 2 * bl.NC
    i = np.arange(n_verts - 1) % R
    return np.concatenate([(i >= bl.NC).astype(int), [0]])


def _frames(st: np.ndarray, side: np.ndarray, arrangement: str) -> list:
    """Per chart: (centre of its (s, t) extent, its offset in cm in the layout)."""
    out = []
    for k in (0, 1):
        sel = side == k
        c = np.array([0.5 * (st[sel, 0].max() + st[sel, 0].min()), 0.5 * (st[sel, 1].max() + st[sel, 1].min())])
        span = np.ptp(st[sel], axis=0)
        off = (k - 0.5) * ((span[1] + GAP_CM) * ACROSS if arrangement == "stack" else (span[0] + GAP_CM) * AXIS)
        out.append((c, off))
    return out


def _to_uv(st: np.ndarray, side: np.ndarray, frames: list, scale: float, shift) -> np.ndarray:
    d = DENSITY * scale
    out = np.zeros((len(st), 2))
    for k in (0, 1):
        sel = side == k
        c, off = frames[k]
        q = st[sel] - c[None, :]
        out[sel] = CENTRE[None, :] + np.asarray(shift)[None, :] + d * (q[:, 0:1] * AXIS[None, :] + q[:, 1:2] * ACROSS[None, :] + off[None, :])
    return out


def _seam(L: np.ndarray) -> dict:
    """Blade-local indices of the left side's seam columns (leading and trailing edge) and of the tip, the chart they are
    copied into (right for the edge columns, left for the tip) and their chart coordinates there."""
    R = 2 * bl.NC
    cols = np.concatenate([np.arange(bl.NL) * R + bl.NC, np.arange(bl.NL) * R + (R - 1)])
    tip = len(L) - 1
    st = np.concatenate([np.stack([L[cols, 0], L[cols, 2]], 1), [[-L[tip, 0], L[tip, 2]]]])
    return {"cols": cols, "tip": tip, "st": st, "side": np.concatenate([np.zeros(len(cols), int), [1]])}


def blade_uv(L: np.ndarray) -> tuple[np.ndarray, dict]:
    """UVs (N_VERTS, 2) of one blade (local cm) in its two charts, and the layout info seam_split needs."""
    L = np.asarray(L, float)
    side = _side(len(L))
    st = np.stack([np.where(side == 1, -L[:, 0], L[:, 0]), L[:, 2]], 1)
    seam = _seam(L)
    shifts = [(0.0, 0.0)] + [(dx, dy) for dx in (-0.06, -0.03, 0.0, 0.03, 0.06) for dy in (-0.06, -0.03, 0.0, 0.03, 0.06) if dx or dy]
    for scale in (1.0, 0.9, 0.8, 0.7):
        for arrangement in ("stack", "row"):
            fr = _frames(st, side, arrangement)
            for sh in shifts:
                uv = _to_uv(st, side, fr, scale, sh)
                if _inside(uv) and _inside(_to_uv(seam["st"], seam["side"], fr, scale, sh)):
                    return uv, {"scale": scale, "arrangement": arrangement, "shift": sh, "frames": fr, "seam": seam, "side": side}
    fr = _frames(st, side, "stack")
    return _to_uv(st, side, fr, 0.7, (0.0, 0.0)), {"scale": 0.7, "arrangement": "stack", "shift": (0.0, 0.0), "frames": fr,
                                                   "seam": seam, "side": side, "outside": True}


def seam_split(V: np.ndarray, F: np.ndarray, uv: np.ndarray, info: dict, n0: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """OBJ-level seams: every face with vertices of both charts uses copies of its left-side seam vertices placed in the
    right chart; the left half of the top fan uses a copy of the tip placed in the left chart. Positions are exact copies
    (render geometry and labels unchanged). Returns the extended vertices, faces and uvs."""
    side, seam = info["side"], info["seam"]
    uv_copies = _to_uv(seam["st"], seam["side"], info["frames"], info["scale"], info["shift"])
    cols = seam["cols"] + n0; tip = seam["tip"] + n0
    base = len(V)
    V2 = np.concatenate([V, V[cols], V[[tip]]]); UV2 = np.concatenate([uv, uv_copies])
    remap = {int(c): base + j for j, c in enumerate(cols)}
    tip_copy = base + len(cols)
    loc = F - n0
    inb = (loc >= 0) & (loc < len(side))
    sides = np.where(inb, side[np.clip(loc, 0, len(side) - 1)], -1)
    is_tip = F == tip
    sides = np.where(is_tip, -1, sides)
    has_r, has_l = (sides == 0).any(1), (sides == 1).any(1)
    F2 = F.copy()
    for fi in np.nonzero(has_r & has_l)[0]:                     # across the leading / trailing edge: right chart
        F2[fi] = [remap.get(int(x), int(x)) for x in F[fi]]
    for fi in np.nonzero(is_tip.any(1) & has_l & ~has_r)[0]:     # left half of the top fan: the tip's left copy
        F2[fi] = [tip_copy if int(x) == tip else int(x) for x in F[fi]]
    return V2, F2, UV2


def make_mask(out: Path = MASK_PNG, size: int = 1024, erode_px: int = 8) -> Path:
    """The skin maps' always-dark region (factory.skin_run06.dark_uv_mask: the fin, the upper back and flanks), its largest
    connected piece, eroded by erode_px texels at `size`: where blade UVs may go."""
    from PIL import Image
    from scipy import ndimage
    from factory import skin_run06 as sk
    m = sk.dark_uv_mask(size) > 0.5
    lab, n = ndimage.label(m)
    big = lab == (int(np.argmax(ndimage.sum(m, lab, range(1, n + 1)))) + 1)
    er = ndimage.binary_erosion(big, iterations=erode_px)
    Image.fromarray((er * 255).astype(np.uint8)).save(out)
    return out


if __name__ == "__main__":
    print(make_mask())
