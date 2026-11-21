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
LEVELS = np.array([-0.6, -0.25, 0.0, 0.05] + list(np.linspace(0.1, 1.0, 37)))   # round 9: 19 -> 37 levels above the foot; round 11: sunk rings deeper (-0.35/-0.15 left the bottom outside the skin on arched backs)
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
    # round 11: the sunk rings (below the foot, h < 0) keep the foot's footprint: extrapolating the edges down to
    # -0.6 H ran the leading edge metres ahead (up to 6.6 H) and out of the body
    sunk = levels < 0.0
    if sunk.any() and (~sunk).any():
        k0 = int(np.argmin(np.where(sunk, np.inf, levels)))
        le[sunk] = np.minimum(le[sunk], le[k0]); te[sunk] = np.maximum(te[sunk], te[k0] - 0.05)
    return le, te


def make(template: dict, heights, H_cm: float, rng: np.random.Generator, nicks: int = 0, thickness: float = 1.0,
         levels=None) -> dict:
    """Blade-local vertices (N_VERTS, 3) in cm, faces (int), and landmark indices, for one individual.
    round 12 (factory.weld): levels = the welded blade's own levels (from its foot ring up, no sunk rings); the result
    then also carries "levels" and "outline" (te / le in units of H at the standard LEVELS, the same numbers the
    round-7..11 blade had, for the hump below the foot and the fin statistics)."""
    if levels is not None:
        return _make_levels(template, heights, H_cm, rng, nicks, thickness, np.asarray(levels, float))
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
    bo = template.get("base_over_height")
    if bo is not None:
        bo = min(float(bo), 4.5)          # round 11: the few outlier templates (up to 6.6 H) put the hump's foot 5 m ahead
    le, te = _edges_at(LEVELS, heights, lead, trail, bo)
    le = np.minimum(le, 4.5)              # the same cap on the leading edge at every height
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


def _make_levels(template: dict, heights, H_cm: float, rng: np.random.Generator, nicks: int, thickness: float,
                 levels: np.ndarray) -> dict:
    """make() on the given levels (same jitter draws, same edges, same sections as make() at those heights); no bottom."""
    heights = np.asarray(heights, float)
    lead, trail = np.asarray(template["lead"], float).copy(), np.asarray(template["trail"], float).copy()
    for arr in (lead, trail):
        arr += np.convolve(rng.normal(0, 0.03, len(arr)), np.ones(5) / 5, mode="same")
        arr[:] = np.convolve(np.pad(arr, 2, mode="edge"), np.ones(5) / 5, mode="valid")
    nk = []
    for _ in range(int(nicks)):
        hn = float(rng.uniform(0.35, 0.95)); wn = float(rng.uniform(0.03, 0.07)); dp = float(rng.uniform(0.03, 0.09))
        nk.append((hn, max(wn, NICK_W_MIN), dp))
    # round 13: a nick is a bite into the trailing edge only (round 12 moved the trailing edge and with it every point of
    # the section, a groove across the whole side); its own depth profile, applied near the trailing edge below
    nick_dp = lambda h: sum(dp * np.exp(-((np.asarray(h, float) - hn) / wn) ** 2) for hn, wn, dp in nk) if nk else 0.0 * np.asarray(h, float)
    trail = np.minimum(trail, lead - 0.04)
    bo = template.get("base_over_height")
    if bo is not None:
        bo = min(float(bo), 4.5)
    # the standard levels' edges (identical to make()), then the welded levels interpolated on them where they fall
    # between standard levels (they are a subset of the standard 0.1..1 grid when H_FOOT is on it)
    le_s, te_s = _edges_at(LEVELS, heights, lead, trail, bo)
    le_s = np.minimum(le_s, 4.5); le_s = np.maximum(le_s, te_s + 0.04)
    # round 13: no forward bulge of the leading edge above the hump (a jittered template could step the nose forward
    # and back within two levels, folding the thin nose): the leading edge never moves ahead going up from 0.25 H
    up_ = LEVELS >= 0.25
    le_s[up_] = np.maximum(np.minimum.accumulate(le_s[up_]), te_s[up_] + 0.04)
    # round 13: the top closes round / pointed (the template's last sample at h = 1 is a flat chord), its levels denser
    # towards the top; the outline (fin statistics) carries the same capped edges
    levels, cap = _top_cap(levels, LEVELS, le_s, te_s, template.get("tip_along"))
    le, te = cap(levels)
    le_s, te_s = cap(LEVELS)
    te_s = te_s + np.minimum(nick_dp(LEVELS), (le_s - te_s) / (NICK_S_POW + 1.0))
    nl = len(levels)
    tmax = thickness * (0.35 * np.clip(1.0 - levels, 0.0, 1.35) ** 1.2 + 0.05)
    s = (1 - np.cos(np.linspace(0, np.pi, NC))) / 2
    prof = np.sqrt(np.clip(1.0 - s, 0, 1)) * np.clip(s, 0, 1) ** 0.7
    prof = prof / prof.max()
    prof = np.clip(prof, 0.02, None)
    V = []
    ndp = np.minimum(nick_dp(levels), (le - te) / (NICK_S_POW + 1.0))     # the section's points stay in order along s
    for k in range(nl):
        a = te[k] + (le[k] - te[k]) * s + ndp[k] * (1.0 - s) ** NICK_S_POW
        w = tmax[k] * prof * min(1.0, max(le[k] - te[k], TOP_CHORD_T) / 0.25)
        right = np.stack([a, w, np.full(NC, levels[k])], 1)
        left = np.stack([a[::-1], -w[::-1], np.full(NC, levels[k])], 1)
        V.append(np.concatenate([right, left]))
    V = np.concatenate(V)
    top = V[-2 * NC:]
    # the top is closed by a strip across the ring (right point i to its left mirror), not a fan to one vertex: a fan
    # over a long top chord is slivers. The tip vertex (landmark) sits on that cap at the top's middle, in no face.
    tip = np.array([[0.5 * (top[:, 0].max() + top[:, 0].min()), 0.0, levels[-1]]])
    V = np.concatenate([V, tip]) * H_cm
    R = 2 * NC
    F = []
    for k in range(nl - 1):
        for i in range(R):
            a0, a1 = k * R + i, k * R + (i + 1) % R
            b0, b1 = (k + 1) * R + i, (k + 1) * R + (i + 1) % R
            F += [(a0, a1, b1), (a0, b1, b0)]
    t = nl * R
    o = (nl - 1) * R
    for i in range(NC - 1):
        r0, r1 = o + i, o + i + 1
        l0, l1 = o + R - 1 - i, o + R - 2 - i
        F += [(r0, r1, l1), (r0, l1, l0)]
    return {"vertices": V, "faces": np.asarray(F, np.int64), "levels": levels,
            "landmarks": {"tip": t, "back_base": 0, "front_base": NC - 1},
            "outline": {"levels": LEVELS.copy(), "te": te_s, "le": le_s}}


NICK_W_MIN = 0.045         # round 13: a nick spans at least this (x H; the welded levels are 0.025 H apart)
NICK_S_POW = 6.0           # round 13: the nick's bite fades along the chord as (1 - s)^NICK_S_POW (s = 0 at the trailing edge)
TOP_GU_MIN = 0.2           # round 13: the top ring keeps this fraction of the cap's half chord (no zero-area faces)
TOP_CHORD_T = 0.08         # round 13: chords under this (x H) keep the thickness of this chord (a round, not knife, top)


def _top_cap(levels: np.ndarray, L: np.ndarray, le_s: np.ndarray, te_s: np.ndarray, tip_along=None):
    """round 13: close the blade's top following the outline. Over the top hc = clip(chord(1) / 2, 0.05, 0.2) of the
    height the half chord shrinks as sqrt(1 - u^2) (u = 0 at 1 - hc, 1 at the top: a round end whose edges keep the
    template's slope where the cap starts) and the centre moves to the template's tip position (tip_along, inside the
    top chord) as u^2. The welded levels above 1 - hc are remapped to h0 + hc sin(pi/2 u) (dense where the chord turns).
    Returns (levels, cap) with cap(h) -> (le, te)."""
    le1, te1 = float(np.interp(1.0, L, le_s)), float(np.interp(1.0, L, te_s))
    hc = float(np.clip(0.5 * (le1 - te1), 0.05, 0.2)); h0 = 1.0 - hc
    tip = 0.5 * (le1 + te1) if tip_along is None else float(np.clip(tip_along, te1, le1))
    lv = np.asarray(levels, float).copy()
    top = lv > h0
    lv[top] = h0 + hc * np.sin(0.5 * np.pi * (lv[top] - h0) / hc)

    def cap(h):
        h = np.asarray(h, float)
        le = np.interp(h, L, le_s); te = np.interp(h, L, te_s)
        u = np.clip((h - h0) / hc, 0.0, 1.0)
        c = 0.5 * (le + te); half = 0.5 * (le - te)
        c = c + (tip - 0.5 * (le1 + te1)) * u * u
        gu = np.maximum(np.sqrt(np.clip(1.0 - u * u, 0.0, 1.0)), TOP_GU_MIN)
        return c + half * gu, c - half * gu
    return lv, cap


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
