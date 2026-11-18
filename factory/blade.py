"""run06 round 7: the parametric dorsal blade (fin + hump silhouette) built from real fin outlines.

A blade is its own closed-sided mesh (open at the bottom, sunk into the back): NL levels of height h (fractions of
the fin height H; negative levels are inside the body) x a ring of 2 * NC points round an airfoil-like section from
the trailing edge to the leading edge, plus a tip vertex. The outline at each height comes from a real fin template
(factory.blade_templates: leading / trailing edge in units of H from the back base, normalised to the researcher's
triangle frame) with a little smoothing and jitter, optional trailing-edge nicks, and a thickness that is broad and
low over the hump and thin at the top. Topology is the same for every blade, so a model can append one blade to its
vertex arrays and give each individual its own shape.

Frame (blade-local, cm): a = along the back toward the head from the back base, y = to the whale's left / right
(whale frame y), h = up from the base line. Landmarks (local indices): tip, back_base (the trailing edge's foot at
h = 0) and front_base (the leading edge's foot at h = 0: where the hump's front slope leaves the back line).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

NC = 24                                       # chord samples per side (round 9: 12 -> 24, smooth sides)
LEVELS = np.array([-0.35, -0.15, 0.0, 0.05] + list(np.linspace(0.1, 1.0, 37)))   # round 9: 19 -> 37 levels above the foot
NL = len(LEVELS)
N_VERTS = NL * 2 * NC + 1


def _edges_at(levels: np.ndarray, heights: np.ndarray, lead: np.ndarray, trail: np.ndarray, base_over_height: float | None = None):
    """Leading / trailing edge at every level: the template inside [0.1, 1]; below it the base row extended (the
    leading edge runs on along the back line, the trailing edge drops straight into the body)."""
    le = np.interp(levels, heights, lead)
    te = np.interp(levels, heights, trail)
    slope = (lead[1] - lead[0]) / max(heights[1] - heights[0], 1e-6)       # d lead / d h at the base (negative)
    low = levels < heights[0]
    le[low] = lead[0] + slope * (levels[low] - heights[0]) * 0.6
    if base_over_height is not None and base_over_height > lead[0]:
        # the leading edge reaches the clicked front base at h = 0 (where the hump's slope leaves the back line)
        f = np.clip((heights[0] - levels[low]) / heights[0], 0.0, None)
        le[low] = lead[0] + (base_over_height - lead[0]) * np.minimum(f, 1.0) + slope * np.clip(-levels[low], 0, None) * 0.3
    te[low] = trail[0] - 0.05 * (heights[0] - levels[low])
    return le, te


def make(template: dict, heights, H_cm: float, rng: np.random.Generator, nicks: int = 0, thickness: float = 1.0) -> dict:
    """Blade-local vertices (N_VERTS, 3) in cm, faces (int), and landmark indices, for one individual."""
    heights = np.asarray(heights, float)
    lead, trail = np.asarray(template["lead"], float).copy(), np.asarray(template["trail"], float).copy()
    # jitter and smoothing: a few percent of H, smooth along the height (never outside what real fins look like)
    for arr in (lead, trail):
        arr += np.convolve(rng.normal(0, 0.03, len(arr)), np.ones(5) / 5, mode="same")
        arr[:] = np.convolve(np.pad(arr, 2, mode="edge"), np.ones(5) / 5, mode="valid")
    for _ in range(int(nicks)):                    # trailing-edge nicks: a small bite into the rear edge
        hn = float(rng.uniform(0.35, 0.95)); wn = float(rng.uniform(0.03, 0.07)); dp = float(rng.uniform(0.03, 0.09))
        trail += dp * np.exp(-((heights - hn) / wn) ** 2)
    trail = np.minimum(trail, lead - 0.04)
    le, te = _edges_at(LEVELS, heights, lead, trail, template.get("base_over_height"))
    le = np.maximum(le, te + 0.04)
    # thickness (half-width over H): broad and low over the hump, thin near the top
    # round 9: thicker sections (half-width 0.20 H at mid height, 0.10 H near the top; round 8 0.12 / 0.05): a thin blade
    # leaked light through its shadow (back-lit blades showed pale, blotchy, crumpled-looking plates)
    tmax = thickness * (0.35 * np.clip(1.0 - LEVELS, 0.0, 1.35) ** 1.2 + 0.05)
    s = (1 - np.cos(np.linspace(0, np.pi, NC))) / 2                      # 0 trailing edge .. 1 leading edge
    # round 8: rounded leading edge (s = 1: square-root nose), thin trailing edge (s = 0); round 7 had them the other way
    prof = np.sqrt(np.clip(1.0 - s, 0, 1)) * np.clip(s, 0, 1) ** 0.7
    prof = prof / prof.max()
    prof = np.clip(prof, 0.02, None)
    V = []
    for k in range(NL):
        a = te[k] + (le[k] - te[k]) * s
        w = tmax[k] * prof * min(1.0, (le[k] - te[k]) / 0.25)                # very short chords stay thin
        right = np.stack([a, w, np.full(NC, LEVELS[k])], 1)
        left = np.stack([a[::-1], -w[::-1], np.full(NC, LEVELS[k])], 1)
        V.append(np.concatenate([right, left]))
    V = np.concatenate(V)
    top = V[-2 * NC:]
    tip = np.array([[top[:, 0].mean(), 0.0, LEVELS[-1] + 0.02]])
    V = np.concatenate([V, tip]) * H_cm
    R = 2 * NC
    F = []
    for k in range(NL - 1):
        for i in range(R):
            a0, a1 = k * R + i, k * R + (i + 1) % R
            b0, b1 = (k + 1) * R + i, (k + 1) * R + (i + 1) % R
            F += [(a0, a1, b1), (a0, b1, b0)]
    t = N_VERTS - 1
    for i in range(R):
        F.append(((NL - 1) * R + i, (NL - 1) * R + (i + 1) % R, t))
    k0 = int(np.argmin(np.abs(LEVELS - 0.0)))
    return {"vertices": V, "faces": np.asarray(F, np.int64),
            "landmarks": {"tip": t, "back_base": k0 * R + 0, "front_base": k0 * R + NC - 1}}


def load_templates(path) -> dict:
    return json.loads(Path(path).read_text())


def place(local: np.ndarray, B: np.ndarray, along: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Blade-local (a, y, h) cm -> the model's rest frame: origin at the back base B, a along `along`, h along `up`
    (both unit vectors in the x-z plane), y the frame's y."""
    return B[None, :] + local[:, 0:1] * along[None, :] + local[:, 2:3] * up[None, :] + local[:, 1:2] * np.array([[0.0, 1.0, 0.0]])


def kabsch(P: np.ndarray, Q: np.ndarray):
    """Rotation R and translation t with R @ p + t ~ q for the rows of P, Q (least squares)."""
    pc, qc = P.mean(0), Q.mean(0)
    Hm = (P - pc).T @ (Q - qc)
    U, _, Vt = np.linalg.svd(Hm)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    Rm = Vt.T @ D @ U.T
    return Rm, qc - Rm @ pc
