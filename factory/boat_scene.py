"""Round 20f (run06): the research boat's bow and land as labelled content.

Real field days hold marker photos the researcher takes on purpose: the bow rail with a band of coloured tape (the role of
the animal just photographed: red female, green male / escort, blue calf, yellow unknown), land with no whale (the boat
moves to a new group) and a straight-down view of the bow deck and its bitt. Whale photos also show a rail or deck corner
in front and land on the horizon. This module builds that content procedurally per shot (no assets bought or downloaded)
and labels it offline by projecting the same geometry with the shot's camera (factory.project), like the whale's amodal
labels; the engine side is unreal/.../wf_boat20.py (spawn, materials) via one hook in wf_build_batch.py.

Config (all dormant when "boat_scene" is absent; a run without it is byte-identical to before):

    "boat_scene": {
      "marker_fraction": {"rail": 0.025, "deck": 0.01, "land": 0.01},  # share of ALL shots turned into marker shots
      "corner_fraction": 0.10,        # share of whale shots with a rail (or rail + deck) corner in front
      "corner_deck_share": 0.3,       # of those, with the bow deck / pontoon in the corner too
      "horizon_land_fraction": 0.10,  # share of whale shots with a ridge on the horizon (only where the horizon is in frame)
      "tape_colour_weights": {"red": 0.27, "green": 0.27, "blue": 0.27, "yellow": 0.19},
      "force_plan": null              # look renders only: {"rail:red": 2, "deck": 3, "land": 3, "corner": 4, "horizon": 4}
    }

Per shot (every shot of a run with the key): shot["boat20"] = {"version", "shot_type" (whale | marker_rail | marker_deck |
marker_land), "marker" ({"type", "colour"} or null), "boat" (mesh + placement or null), "land" (mesh + placement or null),
"tape" (or null), "content" (what was added)}. Marker shots are negatives (negative = no_whale_in_frame = true) with the
camera moved onto the boat and turned away from the whale; every other shot field is kept.

Labels (factory.postprocess hook, shots with "boat20"): dataset/scene/<id>.png, palette uint8 SCENE_CLASSES:
0 water / sky, 1 whale (any instance, = masks class 1), 2 whale through water (= masks class 2, an ignore class), 3 rail,
4 tape, 5 deck (bow structure), 6 cleat (bitt with base plate and horns), 7 pontoon, 8 land (this module's ridge and the
batch's distant islands). Boat > whale > land > water where they overlap (the boat is always nearest; the engine's whale
mask already ends behind it). meta: "shot_type", "marker", "scene_pixels" (per class), "scene_in_frame",
"boat20_label_check" and paths.scene; masks / instances / parts are unchanged.

    python3 -m factory.boat_scene apply RUN_DIR CONFIG.json     # add the keys to an existing shot list (rewrites batches)
    python3 -m factory.boat_scene preview RUN_DIR [--ids ...]   # offline label previews (no render)
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from factory import project
from factory.poses import rotation_matrix

VERSION = "20h"
SCENE_CLASSES = {"water_sky": 0, "whale": 1, "whale_through_water": 2, "rail": 3, "tape": 4, "deck": 5, "cleat": 6,
                 "pontoon": 7, "land": 8}
SCENE_PALETTE = [(0, 0, 0), (255, 255, 255), (128, 128, 128), (230, 120, 40), (255, 0, 255), (60, 140, 255),
                 (255, 230, 0), (0, 200, 120), (140, 90, 40)]
BOAT_CLASSES = ("rail", "tape", "deck", "cleat", "pontoon")
TAPE_COLOURS = ("red", "green", "blue", "yellow")
TAPE_MEANING = {"red": "female", "green": "male / escort", "blue": "calf", "yellow": "unknown"}
TUBE_R_CM = 1.9                      # 1.5" painted stainless tube


# ----------------------------------------------------------------------------------------------- noise and geometry

def _hash01(i: np.ndarray, j: np.ndarray, seed: int) -> np.ndarray:
    h = (i.astype(np.int64) * 73856093) ^ (j.astype(np.int64) * 19349663) ^ seed
    h = (h * 2654435761) & 0xFFFFFFFF
    return h / 0xFFFFFFFF


def vnoise(x: np.ndarray, y: np.ndarray, cell: float, seed: int) -> np.ndarray:
    """Smooth value noise in [0, 1]."""
    u, v = np.asarray(x, float) / cell, np.asarray(y, float) / cell
    i0, j0 = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
    fu, fv = u - i0, v - j0
    fu = fu * fu * (3 - 2 * fu); fv = fv * fv * (3 - 2 * fv)
    a, b = _hash01(i0, j0, seed), _hash01(i0 + 1, j0, seed)
    c, d = _hash01(i0, j0 + 1, seed), _hash01(i0 + 1, j0 + 1, seed)
    return (a * (1 - fu) + b * fu) * (1 - fv) + (c * (1 - fu) + d * fu) * fv


def fbm(x, y, cell: float, seed: int, octaves: int = 5, gain: float = 0.5) -> np.ndarray:
    total, amp, norm = 0.0, 1.0, 0.0
    for o in range(octaves):
        total = total + amp * vnoise(x, y, cell / 2 ** o, seed + 1013 * o)
        norm += amp; amp *= gain
    return total / norm


class Part:
    """One mesh piece: vertices (N, 3) cm in its frame, faces (M, 3) wound outward, uv (N, 2), a label class, a material."""

    def __init__(self, cls: str, mat: str, V, F, UV=None):
        self.cls, self.mat = cls, mat
        self.V = np.asarray(V, float); self.F = np.asarray(F, np.int64)
        self.UV = np.zeros((len(self.V), 2)) if UV is None else np.asarray(UV, float)

    def moved(self, R: np.ndarray, t) -> "Part":
        return Part(self.cls, self.mat, self.V @ R + np.asarray(t, float), self.F, self.UV)


def _orient(V: np.ndarray, F: np.ndarray, inside: np.ndarray) -> np.ndarray:
    """Wind each face so its normal points away from the given inside point (one per face)."""
    tri = V[F]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    flip = np.einsum("ij,ij->i", n, tri.mean(1) - inside) < 0
    F = F.copy(); F[flip] = F[flip][:, [0, 2, 1]]
    return F


def _grid_faces(rows: int, cols: int, base: int = 0) -> np.ndarray:
    i = (np.arange(rows - 1)[:, None] * cols + np.arange(cols - 1)[None, :]).ravel() + base
    return np.concatenate([np.stack([i, i + cols, i + cols + 1], 1), np.stack([i, i + cols + 1, i + 1], 1)])


def _frames(P: np.ndarray):
    T = np.gradient(P, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-12
    a = np.array([0.0, 0.0, 1.0]) if abs(T[0, 2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    N = [np.cross(T[0], a) / np.linalg.norm(np.cross(T[0], a))]
    for k in range(1, len(P)):
        v = N[-1] - np.dot(N[-1], T[k]) * T[k]
        N.append(v / (np.linalg.norm(v) + 1e-12))
    N = np.array(N)
    return T, N, np.cross(T, N)


def tube(path, radius, nseg: int = 28, caps: bool = True, cls: str = "rail", mat: str = "bfpaint") -> Part:
    """A tube along a polyline (radius scalar or per point); u around, v along (cm / 10)."""
    P = np.asarray(path, float)
    r = np.broadcast_to(np.asarray(radius, float), (len(P),))
    T, N, B = _frames(P)
    th = np.linspace(0, 2 * np.pi, nseg + 1)
    ring = np.cos(th)[None, :, None] * N[:, None, :] + np.sin(th)[None, :, None] * B[:, None, :]
    V = (P[:, None, :] + r[:, None, None] * ring).reshape(-1, 3)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    UV = np.stack(np.meshgrid(th / (2 * np.pi), s / 10.0), -1).reshape(-1, 2)
    F = _grid_faces(len(P), nseg + 1)
    inside = np.repeat(P, nseg + 1, axis=0)[F[:, 0]]
    F = _orient(V, F, inside)
    parts_V, parts_F, parts_UV = [V], [F], [UV]
    if caps:
        base = len(V)
        for end, sign in ((0, -1.0), (len(P) - 1, 1.0)):
            c = P[end]; rimv = c + r[end] * ring[end, :-1]
            cv = np.vstack([c[None], rimv]); idx = np.arange(nseg)
            f = np.stack([np.zeros(nseg, int), 1 + idx, 1 + (idx + 1) % nseg], 1) + base
            allv = np.vstack(parts_V)
            f = _orient(np.vstack([allv, cv]), f, np.repeat((c - sign * T[end])[None], nseg, 0))
            parts_V.append(cv); parts_F.append(f); parts_UV.append(np.zeros((nseg + 1, 2)))
            base += nseg + 1
    return Part(cls, mat, np.vstack(parts_V), np.vstack(parts_F), np.vstack(parts_UV))


def lathe(profile, nseg: int = 40, cls: str = "cleat", mat: str = "bfpaint", centre=(0.0, 0.0, 0.0), outward_sign=1.0) -> Part:
    """Surface of revolution about the z axis through `centre` from a profile [(r, z), ...] (one smooth strip)."""
    pr = np.asarray(profile, float)
    th = np.linspace(0, 2 * np.pi, nseg + 1)
    V = np.stack([pr[:, 0][:, None] * np.cos(th)[None], pr[:, 0][:, None] * np.sin(th)[None],
                  np.repeat(pr[:, 1][:, None], nseg + 1, 1)], -1).reshape(-1, 3)
    UV = np.stack(np.meshgrid(th / (2 * np.pi), np.linspace(0, 1, len(pr))), -1).reshape(-1, 2)
    F = _grid_faces(len(pr), nseg + 1)
    axis_pts = np.stack([np.zeros(len(F)), np.zeros(len(F)), V[F].mean(1)[:, 2]], 1)
    F = _orient(V, F, axis_pts)
    if outward_sign < 0:
        F = F[:, [0, 2, 1]]
    return Part(cls, mat, V + np.asarray(centre, float), F, UV)


def loft(rings, cls: str, mat: str, cap_top: bool = True, cap_bottom: bool = False) -> Part:
    """Side walls between closed polygons of equal vertex count (each its own quad strip, hard edges) and caps."""
    Vs, Fs = [], []
    base = 0
    centre = np.mean(np.vstack(rings), axis=0)
    for a, b in zip(rings[:-1], rings[1:]):
        m = len(a)
        for k in range(m):
            q = np.array([a[k], a[(k + 1) % m], b[(k + 1) % m], b[k]])
            Vs.append(q); f = np.array([[0, 1, 2], [0, 2, 3]]) + base
            Fs.append(_orient(np.vstack(Vs), f, np.repeat(centre[None], 2, 0))); base += 4
    for ring, on in ((rings[-1], cap_top), (rings[0], cap_bottom)):
        if not on:
            continue
        c = ring.mean(0); m = len(ring)
        v = np.vstack([c[None], ring]); idx = np.arange(m)
        f = np.stack([np.zeros(m, int), 1 + idx, 1 + (idx + 1) % m], 1) + base
        Vs.append(v); Fs.append(_orient(np.vstack(Vs), f, np.repeat(centre[None], m, 0))); base += m + 1
    V = np.vstack(Vs)
    return Part(cls, mat, V, np.vstack(Fs), V[:, :2] / 50.0)


def sweep(xs, profile_fn, cls: str, mat: str, caps: bool = True) -> Part:
    """Round 20h: a closed cross-section [(y, z), ...] (same point count at every x, from profile_fn(x)) swept along x with
    SHARED vertices, so the engine shades the walls smoothly (loft gives every quad its own vertices: faceted sides); end
    caps fanned from the section's centre. uv = (x, y) / 50 cm like loft."""
    xs = np.asarray(xs, float)
    secs = [np.asarray(profile_fn(x), float) for x in xs]
    m = len(secs[0])
    V = np.concatenate([np.column_stack([np.full(m, x), s]) for x, s in zip(xs, secs)])
    i = (np.arange(len(xs) - 1)[:, None] * m + np.arange(m)[None, :]).ravel()
    j = (np.arange(len(xs) - 1)[:, None] * m + (np.arange(m)[None, :] + 1) % m).ravel()
    F = np.concatenate([np.stack([i, i + m, j + m], 1), np.stack([i, j + m, j], 1)])
    rows = np.concatenate([np.arange(len(xs) - 1).repeat(m), np.arange(len(xs) - 1).repeat(m)])
    centres = np.column_stack([xs, [s.mean(0) for s in secs]])
    F = _orient(V, F, 0.5 * (centres[rows] + centres[rows + 1]))
    Vs, Fs = [V], [F]
    base = len(V)
    if caps:
        for k, out in ((0, -1.0), (len(xs) - 1, 1.0)):
            c = np.array([xs[k], *secs[k].mean(0)])
            v = np.vstack([c[None], V[k * m:(k + 1) * m]]); idx = np.arange(m)
            f = np.stack([np.zeros(m, int), 1 + idx, 1 + (idx + 1) % m], 1) + base
            f = _orient(np.vstack(Vs + [v]), f, np.repeat((c - np.array([out, 0, 0]))[None], m, 0))
            Vs.append(v); Fs.append(f); base += m + 1
    VV = np.vstack(Vs)
    return Part(cls, mat, VV, np.vstack(Fs), VV[:, :2] / 50.0)


def disc(centre, radius: float, cls: str, mat: str, nr: int = 6, nseg: int = 48) -> Part:
    """A flat horizontal disc (normals up), rings of vertices; uv = xy / 100 cm."""
    c = np.asarray(centre, float)
    rr = np.linspace(0, radius, nr + 1)[1:]
    th = np.linspace(0, 2 * np.pi, nseg, endpoint=False)
    V = [c[None]] + [np.column_stack([c[0] + r * np.cos(th), c[1] + r * np.sin(th), np.full(nseg, c[2])]) for r in rr]
    V = np.vstack(V)
    F = [np.stack([np.zeros(nseg, int), 1 + np.arange(nseg), 1 + (np.arange(nseg) + 1) % nseg], 1)]
    for k in range(nr - 1):
        a = 1 + k * nseg + np.arange(nseg); b = 1 + k * nseg + (np.arange(nseg) + 1) % nseg
        F.append(np.stack([a, a + nseg, b + nseg], 1)); F.append(np.stack([a, b + nseg, b], 1))
    F = np.vstack(F)
    F = _orient(V, F, V[F].mean(1) - np.array([0.0, 0.0, 100.0]))
    return Part(cls, mat, V, F, V[:, :2] / 100.0)


def chaikin(P: np.ndarray, n: int = 3) -> np.ndarray:
    for _ in range(n):
        Q = 0.75 * P[:-1] + 0.25 * P[1:]; R = 0.25 * P[:-1] + 0.75 * P[1:]
        P = np.vstack([P[:1], np.stack([Q, R], 1).reshape(-1, 3), P[-1:]])
    return P


def resample(P: np.ndarray, step: float) -> np.ndarray:
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    t = np.linspace(0, s[-1], max(2, int(s[-1] / step) + 1))
    return np.stack([np.interp(t, s, P[:, k]) for k in range(3)], 1)


# ------------------------------------------------------------------------------------------------------- the bow

def bow_model(rng: np.random.Generator, with_hull: bool = True, mid_rail: bool | None = None) -> dict:
    """A RIB bow in the boat frame (cm; x forward, bow tip near x = 0; y right; z up, 0 = mean water plane): grey pontoons,
    a white bow box with a bitt, stanchions and a top rail that runs round the bow and bends down to the deck aft.
    Returns {"parts": [Part], "stanchions": [(side, k, base, top)], "deck_z", "bitt_top", "rail_points": [...]}"""
    R = float(rng.uniform(22, 28)); beam = float(rng.uniform(100, 120)); pz = 14.0 + R
    deck_z = float(rng.uniform(92, 108))

    def half_beam(x):
        t = np.clip(-np.asarray(x, float) / 280.0, 0, 1)
        return 26.0 + (beam - 26.0) * (1 - (1 - t) ** 2)
    parts = []
    if with_hull:
        xs = np.linspace(-820, -18, 90)
        for s in (-1, 1):
            path = np.stack([xs, s * half_beam(xs), np.full_like(xs, pz)], 1)
            rr = R * np.clip(0.75 + 0.25 * np.clip((-xs - 18) / 120, 0, 1), 0, 1)
            parts.append(tube(path, rr, nseg=36, cls="pontoon", mat="bfpontoon"))
        # bow box (round 20h): a smooth swept section with a rounded top edge (shared vertices: no facets); the top is
        # ~2.5 flange widths wide at the bitt (the real bow plate), narrowing to the stem; walls flare onto the pontoons
        bx = -float(rng.uniform(70, 105)); sc = float(rng.uniform(0.9, 1.15))
        w_tip, w_bitt = float(rng.uniform(13, 17)), float(rng.uniform(24, 31))
        z_box = pz + 0.55 * R

        def top_w(x):
            t = np.clip((-x - 38.0) / max(-bx - 38.0, 1.0), 0, 1)
            return w_tip + (w_bitt - w_tip) * np.sqrt(t) + 4.0 * np.clip((-x + bx) / 200.0, 0, 1)

        def box_section(x):
            tw = float(top_w(x)); bw = max(float(half_beam(x)) - 0.55 * R, tw + 6.0)
            bev = [(0.0, 3.0), (0.4, 1.6), (1.2, 0.6), (2.6, 0.0)]          # rounded top edge (cm in from the side, down)
            left = [(-bw, z_box)] + [(-(tw - a), deck_z - b) for a, b in bev]
            mid = [(-tw + 2.6 + (2 * tw - 5.2) * s, deck_z) for s in np.linspace(0, 1, 7)[1:-1]]
            right = [(tw - a, deck_z - b) for a, b in bev[::-1]] + [(bw, z_box)]
            return left + mid + right
        parts.append(sweep(np.linspace(-320, -38, 48), box_section, "deck", "bfdeck"))
        # two chocks (rounded bars) beside the bitt base, the horns' ends above them; a few bolt heads on the plate
        for s in (-1, 1):
            yc = s * (10.5 * sc + 6.0)
            prof = [(yc + 4.0 * math.cos(a), deck_z + 0.2 + 5.5 * max(math.sin(a), 0.0)) for a in np.linspace(0, math.pi, 9)] + \
                   [(yc - 4.0, deck_z), (yc + 4.0, deck_z)]
            parts.append(sweep(np.linspace(bx - 24 * sc, bx + 20 * sc, 8), lambda x, p=prof: p, "deck", "bfdeck"))
        for k in range(int(rng.integers(4, 9))):
            bxk = bx + float(rng.uniform(-60, 45)); byk = float(rng.choice([-1, 1])) * float(rng.uniform(14, max(float(top_w(bxk)) - 4, 15)))
            parts.append(lathe([(1.3, 0.0), (1.2, 0.5), (0.8, 0.9), (0.01, 1.05)], nseg=12, cls="deck", mat="bfdeck",
                               centre=(bxk, byk, deck_z - 0.05)))
        # the bitt (hollow-topped post, flange, cross horns) on a base plate, on the box's centreline
        c = np.array([bx, 0.0, deck_z])
        plate = np.array([[-13, -9], [13, -9], [13, 9], [-13, 9]], float) * sc
        p0 = np.column_stack([plate + c[:2], np.full(4, deck_z)]); p1 = np.column_stack([plate * 0.94 + c[:2], np.full(4, deck_z + 1.6 * sc)])
        parts.append(loft([p0, p1], "cleat", "bfpaint", cap_top=True))
        r0, r1, h = 7.0 * sc, 10.5 * sc, 21.0 * sc
        outer = [(r0 + 0.6, 1.4), (r0, 2.4), (r0, h - 3.2), (r0 + 1.2, h - 2.4), (r1, h - 1.6), (r1, h), (r1 - 0.6, h + 0.5), (r0 - 0.4, h + 0.5)]
        parts.append(lathe([(r * 1.0, z * 1.0) for r, z in outer], cls="cleat", mat="bfpaint", centre=c))
        parts.append(lathe([(r0 - 0.4, h + 0.5), (r0 - 1.0, h - 1.0), (r0 - 1.2, h - 7.0), (0.01, h - 7.2)], cls="cleat", mat="bfdark",
                           centre=c, outward_sign=-1.0))
        bead_th = np.linspace(0, 2 * np.pi, 49)
        for zb in (2.2, 0.45 * h):
            ring = np.stack([c[0] + (r0 + 0.15) * np.cos(bead_th), c[1] + (r0 + 0.15) * np.sin(bead_th), np.full(49, deck_z + zb)], 1)
            parts.append(tube(ring, 0.45 * sc, nseg=10, caps=False, cls="cleat", mat="bfpaint"))
        span = 17.5 * sc; zh = deck_z + 0.42 * h
        hy = np.linspace(-span, span, 30)
        hr = 2.6 * sc * np.sqrt(np.clip(1 - (np.abs(hy) / span) ** 8, 0.02, 1))
        parts.append(tube(np.stack([np.full(30, bx), hy, np.full(30, zh)], 1), hr, nseg=24, caps=True, cls="cleat", mat="bfpaint"))
        bitt_top = c + np.array([0, 0, h])
    else:
        bitt_top = None
    # rail: stanchions on the pontoon tops, a top rail round the bow (pulpit) bending down aft
    sp = float(rng.uniform(70, 105)); x0 = -float(rng.uniform(55, 80)); n_st = 4
    base_z = pz + 0.85 * R; top_z = deck_z + float(rng.uniform(55, 72)); lean = math.radians(float(rng.uniform(2, 9)))
    stanchions, tops, stanchion_parts = [], {-1: [], 1: []}, []
    for s in (-1, 1):
        for k in range(n_st):
            x = x0 - k * sp; y = s * (float(half_beam(x)) - 0.25 * R)
            b = np.array([x, y, base_z]); t = np.array([x, y + s * (top_z - base_z) * math.tan(lean), top_z])
            stanchions.append((s, k, b, t)); tops[s].append(t)
            stanchion_parts.append(len(parts))
            parts.append(tube(resample(np.stack([b, t]), 4.0), TUBE_R_CM, cls="rail"))
    nose = np.array([x0 + float(rng.uniform(22, 40)), 0.0, top_z + float(rng.uniform(-4, 6))])
    aft = {}
    for s in (-1, 1):
        last = tops[s][-1]
        a1 = last + np.array([-30.0, 0, 0]); a2 = np.array([last[0] - 52, last[1], top_z - 25]); a3 = np.array([last[0] - 57, last[1], base_z])
        aft[s] = [a3, a2, a1]
    ctrl = np.vstack(aft[-1] + tops[-1][::-1] + [np.array([nose[0] - 12, -0.6 * abs(tops[-1][0][1]), nose[2]]), nose,
                                                    np.array([nose[0] - 12, 0.6 * abs(tops[1][0][1]), nose[2]])] + tops[1] + aft[1][::-1])
    top_rail = resample(chaikin(ctrl, 4), 3.0)
    parts.append(tube(top_rail, TUBE_R_CM, nseg=28, cls="rail"))
    if mid_rail is None:
        mid_rail = rng.uniform() < 0.4
    if mid_rail:
        zm = 0.5 * (base_z + top_z)
        for s in (-1, 1):
            pts = np.array([[t[0], t[1] - s * (top_z - zm) * math.tan(lean), zm] for t in tops[s]])
            parts.append(tube(resample(pts, 4.0), 0.85 * TUBE_R_CM, cls="rail"))
    return {"parts": parts, "stanchions": stanchions, "stanchion_parts": stanchion_parts, "deck_z": deck_z, "bitt_top": bitt_top,
            "top_rail": top_rail, "bitt_flange_cm": (21.0 * sc if with_hull else None),
            "pontoon_z": pz, "pontoon_r": R, "half_beam_fn": half_beam}


def tape_band(stanchion, frac: float, width_cm: float, colour: str, shade: int, seed: int, layered: bool = False) -> list:
    """Vinyl tape round a stanchion at `frac` of its length (round 20h): a ring 0.12 cm proud of the tube with a seam
    overlap, wavy edges (+-0.12 cm), fine wrinkles (radial ridges running obliquely across the band) and, when `layered`,
    a second narrower wrap offset along the tube (the real bands are often two or three turns, not quite on top of each
    other). width_cm is the whole band. uv: u around, v across each wrap (0..1) for the material's worn edges.
    Returns a list of Parts (class tape)."""
    _, _, b, t = stanchion
    ax = (t - b) / np.linalg.norm(t - b)
    c = b + frac * (t - b)
    rng = np.random.default_rng(seed)
    wraps = [(0.0, width_cm)]
    if layered:
        w1 = width_cm * float(rng.uniform(0.62, 0.8)); off = width_cm - w1
        wraps = [(-0.5 * off, w1), (0.5 * off, w1)]
    out = []
    for k, (shift, w) in enumerate(wraps):
        n = 20
        nseg = 48
        th = np.linspace(0, 2 * np.pi, nseg + 1)
        seam = float(rng.uniform(0, 2 * np.pi))
        vv = np.linspace(-0.5, 0.5, n)
        path = np.array([c + ax * (shift + v * w) for v in vv])
        _, N, B = _frames(path)
        ring = np.cos(th)[None, :, None] * N[:, None, :] + np.sin(th)[None, :, None] * B[:, None, :]
        # edge waviness: the first / last row moved along the tube by a smooth function of the angle
        ph = rng.uniform(0, 2 * np.pi, 3)
        wav = 0.06 * np.sin(th + ph[0]) + 0.04 * np.sin(2 * th + ph[1]) + 0.03 * np.sin(5 * th + ph[2])
        along = (vv[:, None] * w) + np.where(np.abs(vv)[:, None] > 0.49, np.sign(vv)[:, None] * wav[None, :], 0.0)
        # wrinkles: thin oblique ridges, 0.02-0.06 cm high
        wr = np.zeros((n, nseg + 1))
        for _ in range(int(rng.integers(3, 8))):
            th0 = float(rng.uniform(0, 2 * np.pi)); slope = float(rng.uniform(-1.2, 1.2)); amp = float(rng.uniform(0.02, 0.06))
            d = np.angle(np.exp(1j * (th[None, :] - th0 - slope * vv[:, None])))
            wr += amp * np.exp(-(d / 0.06) ** 2)
        proud = 0.12 + 0.12 * k + 0.06 * (np.mod(th - seam, 2 * np.pi) < 0.35)[None, :] + wr
        rr = TUBE_R_CM + proud
        V = (c[None, None, :] + (shift + along)[..., None] * ax[None, None, :] + rr[..., None] * ring).reshape(-1, 3)
        UV = np.stack(np.meshgrid(th / (2 * np.pi), np.linspace(0, 1, n)), -1).reshape(-1, 2)
        F = _grid_faces(n, nseg + 1)
        F = _orient(V, F, np.repeat(path, nseg + 1, axis=0)[F[:, 0]])
        out.append(Part("tape", f"bftape_{colour}_{shade}", V, F, UV))
    return out


# --------------------------------------------------------------------------------------------------------- land

def land_mesh(seed: int, length_m: float, height_m: float, slope_deg: float, cell_m: float, max_vertices: int = 200000,
              v2: bool = False) -> Part:
    """A ridge seen from the sea (land frame, cm): x away from the viewer (shore at x = 0), y along the coast. Crest height
    varies along the coast with peaks and saddles; spurs and gullies run down the slope (ridged noise warped along x);
    fine relief and a rough tree line near the crest. uv = (gully greenness 0..1, slope 0..1) for M_BF20_Land."""
    H = float(height_m); xc0 = H / math.tan(math.radians(slope_deg))
    depth = 1.9 * xc0 + 200.0
    nx = int(depth / cell_m) + 1; ny = int(length_m / cell_m) + 1
    while nx * ny > max_vertices:
        cell_m *= 1.15; nx = int(depth / cell_m) + 1; ny = int(length_m / cell_m) + 1
    xs = np.linspace(-0.04 * depth, depth, nx); ys = np.linspace(-length_m / 2, length_m / 2, ny)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    s = seed
    rng = np.random.default_rng(seed)
    L = float(length_m)
    c1 = 0.45 + 0.5 * fbm(ys, 0 * ys, max(L / 2.5, 50.0), s + 1, octaves=3)          # broad undulation of the crest line
    for _ in range(int(rng.integers(2, 5))):                                          # and a few peaks / saddles
        c1 = c1 + float(rng.uniform(0.25, 0.8)) * np.exp(-((ys - float(rng.uniform(-0.45, 0.45)) * L) / (float(rng.uniform(0.07, 0.3)) * L)) ** 2)
    crest = (H * c1 / c1.max())[None, :]
    xc = (xc0 * (0.75 + 0.5 * fbm(ys, 0 * ys, max(L / 4.0, 50.0), s + 2, octaves=3)))[None, :]
    r = gx / xc
    front = np.where(r < 1, 1 - (1 - np.clip(r, 0, 1)) ** 1.6, 0.0)
    behind = np.where(r >= 1, np.maximum(0.0, 1 - 0.55 * (r - 1)), 0.0)
    h = crest * np.where(r < 1, front, behind)
    mid = np.sin(np.pi * (0.12 + 0.88 * np.clip(r, 0, 1))) ** 0.8                     # valleys cut deepest mid-slope
    # major valleys: the 0.5-isoline of a 1D noise along the coast, warped by a 2D noise so they meander down-slope;
    # strength varies along the coast (some stretches smooth); minor gullies the same at a sixth of the spacing
    s1 = max(float(rng.uniform(0.5, 1.1)) * H, 6 * cell_m)
    strength = (0.35 + 0.65 * fbm(ys, 0 * ys, max(L / 4.0, 50.0), s + 8, octaves=2))[None, :]
    if v2:
        # round 20h (marker land only): valleys less parallel: a stronger warp that changes faster down the slope and a
        # valley field that varies along x too (valleys drift, merge and branch)
        warp1 = (fbm(gx, gy, max(0.3 * xc0, 20.0), s + 3, octaves=3) - 0.5) * 2.4 * s1
        v1 = np.exp(-((fbm(gy + warp1, 0.35 * gx, s1, s + 4, octaves=2) - 0.5) / 0.10) ** 2)
        s2 = max(s1 / 4.0, 4 * cell_m)
        warp2 = (fbm(gx, gy, max(0.12 * xc0, 10.0), s + 9, octaves=3) - 0.5) * 2.0 * s2
        v2n = np.exp(-((fbm(gy + warp2, 0.5 * gx, s2, s + 10, octaves=2) - 0.5) / 0.14) ** 2)
        minor = 0.09
    else:                                   # round 20f as it was (horizon ridges)
        warp1 = (fbm(gx, gy, max(0.45 * xc0, 20.0), s + 3, octaves=3) - 0.5) * 1.4 * s1
        v1 = np.exp(-((fbm(gy + warp1, 0 * gy, s1, s + 4, octaves=2) - 0.5) / 0.10) ** 2)
        s2 = max(s1 / 6.0, 4 * cell_m)
        warp2 = (fbm(gx, gy, max(0.15 * xc0, 10.0), s + 9, octaves=3) - 0.5) * 1.2 * s2
        v2n = np.exp(-((fbm(gy + warp2, 0 * gy, s2, s + 10, octaves=2) - 0.5) / 0.14) ** 2)
        minor = 0.06
    h = h - crest * mid * (0.30 * strength * v1 + minor * v2n)
    gully = np.clip(v1 * strength + 0.4 * v2n, 0, 1)
    h = h + H * 0.035 * (fbm(gx, gy, max(H * 0.25, 10.0), s + 5, octaves=5) - 0.5)
    trees = np.clip((h / np.maximum(crest, 1) - 0.8) / 0.2, 0, 1) * (vnoise(gx, gy, max(2.5 * cell_m, 6.0), s + 6) - 0.3) * min(14.0, 0.05 * H)
    h = h + np.maximum(trees, 0)
    shore = np.clip(gx / max(0.04 * depth, cell_m), -1, 0)
    h = np.where(gx < 0, shore * 20.0, h) - 1.0
    gz_x, gz_y = np.gradient(h, xs, ys)
    slope = np.clip(np.hypot(gz_x, gz_y) / 1.2, 0, 1)
    green = np.clip(1.4 * gully * mid + 0.25 * (fbm(gx, gy, max(H * 0.5, 20.0), s + 7, octaves=3) - 0.5), 0, 1)
    V = np.stack([gx, gy, h], -1).reshape(-1, 3) * 100.0
    # uv = (gully greenness, slope) for M_BF20_Land_v1 (horizon ridges); v2 (round 20h, marker land): (gully greenness,
    # height as a fraction of the crest) for M_BF20_Land_v2, which takes the slope from the normal
    if v2:
        UV = np.stack([green, np.clip(h / np.maximum(crest, 1.0), 0, 1)], -1).reshape(-1, 2)
    else:
        UV = np.stack([green, slope], -1).reshape(-1, 2)
    F = _grid_faces(nx, ny)
    F = _orient(V, F, V[F].mean(1) - np.array([0, 0, 1e5]))
    return Part("land", "bfland2" if v2 else "bfland", V, F, UV)


# ----------------------------------------------------------------------------------------------- camera helpers

def _fb(shot: dict) -> float:
    return float((shot["camera"].get("filmback_mm") or [project.SENSOR_WIDTH_MM])[0])


def _cam(shot: dict, res) -> project.Camera:
    return project.Camera(shot["camera"]["location"], shot["camera"]["rotation"], shot["camera"]["focal_mm"], res[0], res[1],
                          sensor_width_mm=_fb(shot))


def look_rotation(d: np.ndarray, roll: float = 0.0) -> dict:
    d = d / np.linalg.norm(d)
    return {"roll": round(float(roll), 3), "pitch": round(math.degrees(math.asin(np.clip(d[2], -1, 1))), 4),
            "yaw": round(math.degrees(math.atan2(d[1], d[0])), 4)}


def ray_dir(cam: project.Camera, u: float, v: float) -> np.ndarray:
    local = np.array([1.0, (u - cam.cx) / cam.fx, -(v - cam.cy) / cam.fy])
    d = local @ cam.basis
    return d / np.linalg.norm(d)


def horizon_row(cam: project.Camera) -> float | None:
    """Image row of the horizon (level, no curvature) at the frame's centre column; None when the camera rolls so far the
    horizon leaves the centre column."""
    fwd = cam.basis[0]
    h = np.array([fwd[0], fwd[1], 0.0])
    if np.linalg.norm(h) < 1e-6:
        return None
    uv, depth = cam.project(cam.location[None] + 1e7 * h[None] / np.linalg.norm(h))
    return float(uv[0, 1]) if depth[0] > 0 else None


def clip_near(V_cam: np.ndarray, F: np.ndarray, near: float = 5.0):
    """Triangles in camera space clipped against depth >= near: returns new vertices and faces (with a source face index)."""
    d = V_cam[:, 0]
    inside = d[F] >= near
    keep = inside.all(1)
    outV = [V_cam]; outF = [F[keep]]; src = [np.nonzero(keep)[0]]
    base = len(V_cam)
    for fi in np.nonzero(inside.any(1) & ~keep)[0]:
        poly = []
        tri = V_cam[F[fi]]
        for a in range(3):
            p, q = tri[a], tri[(a + 1) % 3]
            pin, qin = p[0] >= near, q[0] >= near
            if pin:
                poly.append(p)
            if pin != qin:
                t = (near - p[0]) / (q[0] - p[0]); poly.append(p + t * (q - p))
        if len(poly) < 3:
            continue
        poly = np.array(poly); outV.append(poly)
        for k in range(1, len(poly) - 1):
            outF.append(np.array([[base, base + k, base + k + 1]])); src.append(np.array([fi]))
        base += len(poly)
    return np.vstack(outV), np.vstack(outF), np.concatenate(src)


def raster_parts(parts: list, cam: project.Camera, width: int, height: int, scale: float = 1.0):
    """Class image (SCENE_CLASSES ids) and depth (cm) of world-space parts, z-buffered, near-clipped at 5 cm."""
    if not parts:
        return np.zeros((height, width), np.uint8), np.full((height, width), np.inf)
    V = np.vstack([p.V for p in parts]); offs = np.cumsum([0] + [len(p.V) for p in parts])
    F = np.vstack([p.F + o for p, o in zip(parts, offs[:-1])])
    cls = np.concatenate([np.full(len(p.F), SCENE_CLASSES[p.cls]) for p in parts])
    Vc = cam.to_camera(V)
    Vc2, F2, src = clip_near(Vc, F)
    depth = Vc2[:, 0]
    safe = np.maximum(depth, 1e-3)
    u = (cam.cx + cam.fx * Vc2[:, 1] / safe) * scale; v = (cam.cy - cam.fy * Vc2[:, 2] / safe) * scale
    uv = np.stack([u, v], 1)
    w2, h2 = int(round(width * scale)), int(round(height * scale))
    tri = uv[F2]
    vis = ~((tri[:, :, 0] < 0).all(1) | (tri[:, :, 0] > w2).all(1) | (tri[:, :, 1] < 0).all(1) | (tri[:, :, 1] > h2).all(1))
    img, zbuf, _ = project.rasterize(uv, depth, F2[vis], w2, h2, face_values=cls[src[vis]], return_extra=True)
    return img.astype(np.uint8), zbuf


def fill_mask(parts: list, cam: project.Camera, width: int, height: int, min_z_cm: float = 0.0) -> np.ndarray:
    """Union of the projected faces of far parts (land): PIL polygon fill, faces wholly below min_z_cm skipped."""
    img = Image.new("L", (width, height), 0); draw = ImageDraw.Draw(img)
    for p in parts:
        top = p.V[p.F][:, :, 2].max(1)
        F = p.F[top > min_z_cm]
        if not len(F):
            continue
        uv, depth = cam.project(p.V)
        ok = (depth[F] > 100.0).all(1)
        tri = uv[F[ok]]
        inview = ~((tri[:, :, 0] < 0).all(1) | (tri[:, :, 0] > width).all(1) | (tri[:, :, 1] < 0).all(1) | (tri[:, :, 1] > height).all(1))
        for t in tri[inview]:
            draw.polygon([tuple(x) for x in t], fill=1)
    return np.asarray(img, bool)


# ----------------------------------------------------------------------------------------------------- placement

def _boat_world(model_parts, yaw: float, origin) -> list:
    R = rotation_matrix(0.0, 0.0, yaw)
    return [p.moved(R, origin) for p in model_parts]


def _coverage(parts_w, cam, res, scale=0.125):
    img, _ = raster_parts(parts_w, cam, res[0], res[1], scale)
    return img


def _set_camera(shot: dict, loc, rot: dict, focal: float, focus_m: float, aperture: float, note: str) -> None:
    cam = shot["camera"]
    orig = {k: cam.get(k) for k in ("location", "rotation", "focal_mm", "focus_distance_m", "aperture_f", "height_m")}
    ratio = float(cam.get("focal_mm_field_equivalent") or cam["focal_mm"]) / float(cam["focal_mm"])
    cam.update({"location": [round(float(x), 2) for x in loc], "rotation": rot, "focal_mm": round(float(focal), 2),
                "focus_distance_m": round(float(focus_m), 3), "aperture_f": round(float(aperture), 2),
                "height_m": round(float(loc[2]) / 100.0, 3), "focal_mm_field_equivalent": round(float(focal) * ratio, 2),
                "boat20_moved": note, "boat20_original": orig})


def _away_yaw(rng, shot: dict, lo: float = 60.0, hi: float = 170.0) -> float:
    return float(shot["camera"]["rotation"]["yaw"]) + float(rng.choice([-1, 1])) * float(rng.uniform(lo, hi))


def _wrap180(a: float) -> float:
    return (float(a) + 180.0) % 360.0 - 180.0


def _sun_azimuth(shot: dict) -> float:
    """Compass yaw (deg) toward the sun: the sun actor's yaw is the direction its light travels."""
    return float(shot["sun"]["yaw"]) + 180.0


def _choose_yaw(rng, shot: dict, sun_lo: float, sun_hi: float, away_lo: float = 60.0, away_hi: float = 180.0, tries: int = 24):
    """Round 20h: a camera yaw turned away from the whale (|yaw - original| in [away_lo, away_hi]) with the sun between
    sun_lo and sun_hi degrees off the view direction (rail markers: 80-165, the sun beside / behind the photographer, so
    no glints from the water in front and side light across the tube, as in the real shots)."""
    y0 = float(shot["camera"]["rotation"]["yaw"]); sa = _sun_azimuth(shot)
    for _ in range(tries):
        y = sa + float(rng.choice([-1, 1])) * float(rng.uniform(sun_lo, sun_hi))
        if away_lo <= abs(_wrap180(y - y0)) <= away_hi:
            return y
    return None


def _whale_points(shot: dict) -> list:
    pts = [shot["whale"]["location"]]
    for c in shot.get("companions") or []:
        loc = c.get("location") or (c.get("whale") or {}).get("location")
        if loc:
            pts.append(loc)
    return [np.asarray(p, float) for p in pts]


def _whales_clear(shot: dict, cam: project.Camera, res, body_cm: float = 1000.0) -> bool:
    """No whale near the frame: every whale centre is outside the half-diagonal field of view by its own angular size."""
    half_diag = math.atan(0.5 * math.hypot(res[0], res[1]) / cam.fx)
    fwd = cam.basis[0]
    for p in _whale_points(shot):
        d = p - cam.location; dist = float(np.linalg.norm(d))
        if dist < 1.0:
            return False
        ang = math.acos(float(np.clip(np.dot(d / dist, fwd), -1, 1)))
        if ang < half_diag + math.atan(body_cm / dist):
            return False
    return True


def _waves(shot: dict):
    try:
        from factory import gerstner
        return gerstner.generate_waves(shot["waves"]) if shot.get("waves") else None
    except Exception:
        return None


def _float_z(shot: dict, bm: dict, yaw: float, origin_xy, waves) -> float:
    """Round 20h: the boat's z (cm) so it floats on the shot's frozen waves: the pontoon bottoms (pontoon_z - R above the
    boat origin) just touch the highest crest under the visible bow (x -450..0). At z = 0 a rough sea flooded the pontoons
    (000015 of r06r_boat20_look: the water over the grey tubes rendered bright cyan)."""
    if not waves:
        return 0.0
    xs = np.linspace(-450.0, -20.0, 24)
    hb = np.array([float(bm["half_beam_fn"](x)) for x in xs])
    pts = np.concatenate([np.column_stack([xs, s * hb, np.zeros_like(xs)]) for s in (-1, 1)]
                         + [np.column_stack([xs, np.zeros_like(xs), np.zeros_like(xs)])])
    R = rotation_matrix(0.0, 0.0, yaw)
    w = pts @ R + np.array([origin_xy[0], origin_xy[1], 0.0])
    from factory import gerstner
    h = gerstner.surface_height(waves, w[:, 0], w[:, 1], float(shot.get("wave_time_s") or 0.0))
    bottom = float(bm["pontoon_z"] - bm["pontoon_r"])
    return round(float(np.max(h)) - bottom + 4.0, 2)


def _marker_ev(opts: dict, key: str, counter: dict, default: float) -> float:
    """EV100 offset for a marker type (negative = brighter); a list in the config cycles over the markers of that type
    (look renders: an exposure ladder)."""
    v = (opts.get("marker_exposure_ev100") or {}).get(key, default)
    if isinstance(v, list):
        k = counter.get(key, 0); counter[key] = k + 1
        return float(v[k % len(v)])
    return float(v)


def _bokeh_cvars(bg_radius_frac: float | None = None, fg_radius_frac: float | None = None) -> dict:
    """Per-shot console variables (keyed in the shot's own CVar section by wf_build_batch, restored after it): round
    bokeh (the diaphragm blade shape off) and, where the physical blur is larger, the gather radius cap raised."""
    cv = {"r.DOF.Gather.EnableBokehSettings": 0, "r.DOF.Scatter.EnableBokehSettings": 0, "r.DOF.Recombine.EnableBokehSettings": 0}
    if bg_radius_frac is not None and bg_radius_frac > 0.025:           # only where the physical blur exceeds the default cap
        cv["r.DOF.Kernel.MaxBackgroundRadius"] = round(float(np.clip(bg_radius_frac, 0.025, 0.08)), 4)
    if fg_radius_frac is not None and fg_radius_frac > 0.025:
        cv["r.DOF.Kernel.MaxForegroundRadius"] = round(float(np.clip(fg_radius_frac, 0.025, 0.08)), 4)
    return cv


def _floor_part(cam_w: np.ndarray, dir_w: np.ndarray, depth_cm: float, radius_cm: float = 2500.0) -> Part:
    """A dark matte plane under the water where the view meets the sea (marker shots only, not labelled): stands in for the
    light the water column scatters back up, which the ocean material leaves near black when looked at steeply."""
    t = cam_w[2] / max(-dir_w[2], 0.05)
    hit = cam_w + t * dir_w
    return disc([hit[0], hit[1], -depth_cm], radius_cm, "water_sky", "bfseafloor")


def place_marker_rail(rng, shot, res, colour: str, opts: dict | None = None, tries: int = 400):
    """Round 20h: camera standing in the boat, looking 28-58 deg down past a stanchion at the sea a few metres off the hull;
    the sun 80-165 deg off the view (no glints in front, side light across the tube); most shots (rail_second_tube_share)
    with a second tube (top rail / another stanchion) in frame, some (rail_wedge_share) with a hull wedge (bow box side or
    pontoon, 2-25 % of the frame) in a corner; the boat floats on the shot's waves."""
    opts = opts or {}
    fb = _fb(shot); waves = _waves(shot)
    want_second = rng.uniform() < float(opts.get("rail_second_tube_share", 0.8))
    want_wedge = rng.uniform() < float(opts.get("rail_wedge_share", 0.45))
    for attempt in range(tries):
        if attempt == tries // 2:                     # relax the composition wishes rather than fail the shot: wedge first
            want_wedge = None
        if attempt == (9 * tries) // 10:
            want_second = None
        bm = bow_model(rng, with_hull=True, mid_rail=False)     # the real bow rail has no mid rail (a crossing tube read wrong)
        idx = [k for k, st in enumerate(bm["stanchions"]) if st[1] <= 2]
        k = idx[int(rng.integers(len(idx)))]
        st = bm["stanchions"][k]; st_part = bm["stanchion_parts"][k]
        frac = float(rng.uniform(0.3, 0.75)); shade = int(rng.integers(3))
        width = 2 * TUBE_R_CM * float(rng.uniform(0.4, 0.9)); layered = bool(rng.uniform() < 0.45)   # 20h test read 0.9-1.4 tube widths
        tape = tape_band(st, frac, width, colour, shade, int(rng.integers(1 << 30)), layered=layered)
        centre = st[2] + frac * (st[3] - st[2])
        f = float(rng.uniform(170, 300)); q = float(rng.uniform(0.05, 0.13))
        d = 2 * TUBE_R_CM * f / (q * fb)
        el = math.radians(float(rng.uniform(-58, -28))); az = math.radians(float(rng.uniform(-20, 100))); s = st[0]
        dir_b = np.array([math.cos(el) * math.cos(az), s * math.cos(el) * math.sin(az), math.sin(el)])
        cam_b = centre - d * dir_b
        if abs(cam_b[1]) > float(bm["half_beam_fn"](min(cam_b[0], 0.0))) + 40 or cam_b[0] > 30:
            continue
        if not bm["deck_z"] + 60.0 <= cam_b[2] <= bm["deck_z"] + 330.0:
            continue
        yaw_cam_world = _choose_yaw(rng, shot, 80.0, 165.0)
        if yaw_cam_world is None:
            continue
        yaw_boat = yaw_cam_world - math.degrees(math.atan2(dir_b[1], dir_b[0]))
        Rb = rotation_matrix(0.0, 0.0, yaw_boat)
        cam_xy = np.asarray(shot["camera"]["location"][:2], float)
        oxy = cam_xy - (cam_b @ Rb)[:2]
        z0 = _float_z(shot, bm, yaw_boat, oxy, waves)
        origin = np.array([oxy[0], oxy[1], z0])
        cam_w = cam_b @ Rb + origin
        dir_w = dir_b @ Rb
        rot = look_rotation(dir_w, roll=float(np.clip(rng.normal(0, 5), -12, 12)))
        cam = project.Camera(cam_w, rot, f, res[0], res[1], sensor_width_mm=fb)
        if not _whales_clear(shot, cam, res):
            continue
        # coverage, the tape's own stanchion counted apart from the other rails (a second tube in frame or not)
        marked = [Part("cleat" if i == st_part else p.cls, p.mat, p.V, p.F) for i, p in enumerate(bm["parts"])] + tape
        img = _coverage(_boat_world(marked, yaw_boat, origin), cam, res)
        tape_px = (img == SCENE_CLASSES["tape"]).sum(); own = (img == SCENE_CLASSES["cleat"]).mean()
        other = (img == SCENE_CLASSES["rail"]).mean()
        hull = np.isin(img, [SCENE_CLASSES["deck"], SCENE_CLASSES["pontoon"]]).mean()
        if tape_px < 3 or own < 0.03 or (img > 0).mean() > 0.5 or hull > 0.25:
            continue
        if want_second is True and other < 0.015 or want_second is False and other > 0.004:
            continue
        if want_wedge is True and hull < 0.02 or want_wedge is False and hull > 0.004:
            continue
        aperture = math.exp(rng.uniform(math.log(4.0), math.log(8.0)))      # 20h b: wider, the sea behind softer
        _set_camera(shot, cam_w, rot, f, d / 100.0, aperture, "marker_rail")
        sea_cm = cam_w[2] / max(-dir_w[2], 0.05)
        extra = {"bg_blur_frac": coc_radius_px(shot, sea_cm, res[0]) / res[0], "dir_w": dir_w, "cam_w": cam_w}
        return bm["parts"] + tape, yaw_boat, origin, {"colour": colour, "meaning": TAPE_MEANING[colour], "shade": shade,
                                                       "width_cm": round(width, 2), "width_tube_d": round(width / (2 * TUBE_R_CM), 3),
                                                       "layered": layered, "height_frac": round(frac, 3),
                                                       "stanchion": {"side": int(st[0]), "index": int(st[1])},
                                                       "distance_m": round(d / 100.0, 3), "look_down_deg": round(-math.degrees(el), 1),
                                                       "sun_off_view_deg": round(abs(_wrap180(yaw_cam_world - _sun_azimuth(shot))), 1),
                                                       "second_tube_frac": round(float(other), 4), "hull_frac": round(float(hull), 4),
                                                       "float_z_cm": z0}, extra
    return None


def place_marker_deck(rng, shot, res, opts: dict | None = None, tries: int = 80):
    """Round 20h: the bitt from close above like the real ones (camera 1.4-3 m from the bitt top, 48-78 deg down from aft,
    a normal-ish lens 24-135 mm so the flange is 17-32 % of the frame width), sharp (f/8-13, focus on the bitt); the sun
    50-180 deg off the view (no glints in the sea corners); the boat floats on the shot's waves."""
    fb = _fb(shot); waves = _waves(shot)
    for _ in range(tries):
        bm = bow_model(rng, with_hull=True)
        target = bm["bitt_top"]
        d = float(rng.uniform(140, 300)); q = float(rng.uniform(0.17, 0.32))
        f = q * d * fb / float(bm["bitt_flange_cm"])
        if not 24.0 <= f <= 135.0:
            continue
        el = math.radians(float(rng.uniform(-78, -48))); az = math.radians(float(rng.uniform(-20, 20)))
        dir_b = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        cam_b = target - d * dir_b
        if not bm["deck_z"] + 100.0 <= cam_b[2] <= bm["deck_z"] + 240.0:
            continue
        yaw_cam_world = _choose_yaw(rng, shot, 50.0, 180.0)
        if yaw_cam_world is None:
            continue
        yaw_boat = yaw_cam_world - math.degrees(math.atan2(dir_b[1], dir_b[0]))
        Rb = rotation_matrix(0.0, 0.0, yaw_boat)
        cam_xy = np.asarray(shot["camera"]["location"][:2], float)
        oxy = cam_xy - (cam_b @ Rb)[:2]
        z0 = _float_z(shot, bm, yaw_boat, oxy, waves)
        origin = np.array([oxy[0], oxy[1], z0])
        cam_w = cam_b @ Rb + origin
        dir_w = dir_b @ Rb
        rot = look_rotation(dir_w, roll=float(np.clip(rng.normal(0, 5), -10, 10)))
        cam = project.Camera(cam_w, rot, f, res[0], res[1], sensor_width_mm=fb)
        if not _whales_clear(shot, cam, res):
            continue
        img = _coverage(_boat_world(bm["parts"], yaw_boat, origin), cam, res)
        sea = (img == 0).mean()
        if (img == SCENE_CLASSES["cleat"]).mean() < 0.04 or not 0.02 <= sea <= 0.35:
            continue
        _set_camera(shot, cam_w, rot, f, d / 100.0, math.exp(rng.uniform(math.log(8.0), math.log(13.0))), "marker_deck")
        return bm["parts"], yaw_boat, origin, {"distance_m": round(d / 100.0, 3), "look_down_deg": round(-math.degrees(el), 1),
                                               "sun_off_view_deg": round(abs(_wrap180(yaw_cam_world - _sun_azimuth(shot))), 1),
                                               "float_z_cm": z0}, {"dir_w": dir_w, "cam_w": cam_w}
    return None


def _land_for(rng, cam: project.Camera, res, distance_m: float, top_row: float, fb: float, focal: float, seed: int,
              v2: bool = False):
    """A ridge at distance_m along the camera's heading whose crest reaches image row top_row (at the centre column)."""
    hz = horizon_row(cam)
    ang = math.atan((hz - top_row) / cam.fy)
    H = distance_m * math.tan(ang) + cam.location[2] / 100.0
    if not 25.0 <= H <= 1900.0:
        return None
    width_m = distance_m * fb / focal * 1.7 * res[0] / max(res[0], 1) + 2.5 * H
    cell = max(1.6 * distance_m * (fb / focal) / res[0], 0.8)
    slope = float(rng.uniform(14, 32))
    land = land_mesh(seed, width_m, H / 0.85, slope, cell, v2=v2)
    fwd = cam.basis[0].copy(); fwd[2] = 0; fwd /= np.linalg.norm(fwd)
    yaw = math.degrees(math.atan2(fwd[1], fwd[0])) + float(rng.uniform(-8, 8))
    origin = np.array([cam.location[0] + 100 * distance_m * fwd[0], cam.location[1] + 100 * distance_m * fwd[1], 0.0])
    return land, yaw, origin, {"distance_m": round(distance_m, 1), "crest_m": round(H, 1), "slope_deg": round(slope, 1),
                               "cell_m": round(cell, 4), "length_m": round(width_m, 3), "seed": int(seed)}


def cloud_card(rng, cam: project.Camera, res, land_distance_m: float, crest_m: float):
    """Round 20h: a vertical card of trade-wind cumulus behind the ridge (2-7 km beyond it, facing the camera), drawn by
    M_BF20_Cloud_v1 from its uv: v = 0 at the flat cloud base, 1 at the tops; u in the same units along the card (plus a
    random offset). World-space Part (not labelled: clouds are sky). None when the cloud layer would sit above the frame."""
    fwd = cam.basis[0].copy(); fwd[2] = 0; fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, np.array([0.0, 0.0, 1.0])); right /= np.linalg.norm(right)
    Dc = 100.0 * (land_distance_m + float(rng.uniform(2000, 7000)))
    hfov = math.atan(0.5 * res[0] / cam.fx); vhalf = math.atan(0.5 * res[1] / cam.fy)
    pitch = math.asin(float(np.clip(cam.basis[0][2], -1, 1)))
    z_top_frame = cam.location[2] + Dc * math.tan(pitch + vhalf)
    z_b = max(0.8 * 100.0 * crest_m, 45000.0) + float(rng.uniform(-5000, 30000))
    if z_b > 0.92 * z_top_frame:
        z_b = 0.92 * z_top_frame
        if z_b < 0.6 * 100.0 * crest_m:
            return None
    T = float(rng.uniform(60000, 160000))
    W = 2.0 * Dc * math.tan(hfov) * 1.6
    nu, nv = 41, 13
    us = np.linspace(-0.5, 0.5, nu) * W; vs = np.linspace(-0.08, 1.0, nv)
    gu, gv = np.meshgrid(us, vs, indexing="ij")
    c0 = cam.location[:2] + Dc * fwd[:2]
    V = np.stack([c0[0] + gu * right[0], c0[1] + gu * right[1], z_b + gv * T], -1).reshape(-1, 3)
    UV = np.stack([gu / T + float(rng.uniform(0, 20)), gv], -1).reshape(-1, 2)
    F = _grid_faces(nu, nv)
    F = _orient(V, F, V[F].mean(1) + 1e4 * np.array([fwd[0], fwd[1], 0.0]))       # facing the camera
    return Part("water_sky", "bfcloud", V, F, UV), {"distance_m": round(Dc / 100.0, 1), "base_m": round(z_b / 100.0, 1),
                                                     "thickness_m": round(T / 100.0, 1)}


def place_marker_land(rng, shot, res, opts: dict | None = None, tries: int = 30):
    """Round 20h: two looks, like the real land photos: "clear" (green / brown ridges, the shot's height fog thinned to
    5-15 %, cumulus over the ridge; sunny shots only) or "hazy" (fog x0.5-1, a pale ridge, no clouds); share
    land_clear_share."""
    opts = opts or {}
    fb = _fb(shot)
    clear = bool(rng.uniform() < float(opts.get("land_clear_share", 0.6))) and not shot.get("overcast")   # no cumulus look on overcast
    for _ in range(tries):
        f = float(math.exp(rng.uniform(math.log(28), math.log(240))))
        cam_z = float(rng.uniform(250, 330))
        fy = f / fb * res[0]
        hz_frac = float(rng.uniform(0.75, 1.04))
        pitch = math.degrees(math.atan((hz_frac - 0.5) * res[1] / fy))
        yaw = _choose_yaw(rng, shot, 45.0, 180.0)
        if yaw is None:
            continue
        loc = [float(shot["camera"]["location"][0]), float(shot["camera"]["location"][1]), cam_z]
        rot = {"roll": round(float(np.clip(rng.normal(0, 1.5), -4, 4)), 3), "pitch": round(pitch, 4), "yaw": round(yaw, 4)}
        cam = project.Camera(loc, rot, f, res[0], res[1], sensor_width_mm=fb)
        if not _whales_clear(shot, cam, res):
            continue
        D = float(math.exp(rng.uniform(math.log(4000), math.log(15000))))
        top = float(rng.uniform(0.12, 0.5)) * res[1]
        got = _land_for(rng, cam, res, D, top, fb, f, int(rng.integers(1 << 30)), v2=True)
        if got is None:
            continue
        land, lyaw, origin, info = got
        info["v2"] = True
        cloud = cloud_card(rng, cam, res, D, info["crest_m"]) if clear and opts.get("land_clouds", True) else None
        fog = float(rng.uniform(0.05, 0.15)) if clear else float(rng.uniform(0.5, 1.0))     # 20h test: x0.15-0.4 still fog-blue, x1-1.8 white
        _set_camera(shot, loc, rot, f, D, math.exp(rng.uniform(math.log(8.0), math.log(11.0))), "marker_land")
        info.update(look="clear" if clear else "hazy", fog_factor=round(fog, 3))
        return land, lyaw, origin, info, {"cloud": cloud, "fog_factor": fog, "clear": clear}
    return None


def place_corner(rng, shot, res, with_deck: bool, tries: int = 40):
    """A rail (and maybe deck / pontoon) corner in front of a whale shot. The camera, whale and everything else stay; the bow
    is placed in the camera's frame (run06 cameras stand 4.5-10 m above the water, higher than a real bow: the rail is not
    tied to the water level and its hull stays out of frame)."""
    cam = _cam(shot, res); fb = _fb(shot); f = float(shot["camera"]["focal_mm"])
    for _ in range(tries):
        bm = bow_model(rng, with_hull=with_deck)
        left = rng.uniform() < 0.5
        u = float(rng.uniform(0.0, 0.22)) * res[0]; u = u if left else res[0] - u
        v = float(rng.uniform(0.82, 1.0)) * res[1]
        q = float(rng.uniform(0.012, 0.07))
        d = float(np.clip(2 * TUBE_R_CM * f / (q * fb), 80, 1800))
        Pw = cam.location + d * ray_dir(cam, u, v)
        yaw_boat = float(shot["camera"]["rotation"]["yaw"]) + float(rng.uniform(-55, 55))
        pick = rng.uniform()
        if pick < 0.5:
            st = bm["stanchions"][int(rng.integers(len(bm["stanchions"])))]
            Pb = st[2] + float(rng.uniform(0.4, 1.0)) * (st[3] - st[2])
        else:
            Pb = bm["top_rail"][int(rng.integers(len(bm["top_rail"])))]
        Rb = rotation_matrix(0.0, 0.0, yaw_boat)
        origin = Pw - Pb @ Rb
        parts_w = _boat_world(bm["parts"], yaw_boat, origin)
        # drop faces closer than 60 cm to the camera (the boat around the photographer; out of frame or a blur)
        keep_parts, keep_local = [], []
        for pl, pw in zip(bm["parts"], parts_w):
            dep = cam.to_camera(pw.V)[:, 0]
            ok = (dep[pw.F] > 60.0).all(1)
            if ok.any():
                keep_parts.append(Part(pw.cls, pw.mat, pw.V, pw.F[ok], pw.UV)); keep_local.append(Part(pl.cls, pl.mat, pl.V, pl.F[ok], pl.UV))
        img = _coverage(keep_parts, cam, res)
        cov = (img > 0).mean()
        h8, w8 = img.shape
        central = img[int(0.2 * h8):int(0.8 * h8), int(0.25 * w8):int(0.75 * w8)]
        if not 0.01 <= cov <= 0.22 or (central > 0).mean() > 0.02 or (img[: int(0.35 * h8)] > 0).any():
            continue
        return keep_local, yaw_boat, origin, {"corner": "left" if left else "right", "distance_m": round(d / 100.0, 3),
                                             "coverage_est": round(float(cov), 4), "with_deck": bool(with_deck)}
    return None


def place_horizon_land(rng, shot, res):
    cam = _cam(shot, res); hz = horizon_row(cam)
    if hz is None or not 0.02 * res[1] <= hz <= 0.85 * res[1]:
        return None
    fb = _fb(shot); f = float(shot["camera"]["focal_mm"])
    for _ in range(12):
        D = float(math.exp(rng.uniform(math.log(3000), math.log(20000))))
        rise = float(rng.uniform(0.01, 0.15)) * res[1]
        got = _land_for(rng, cam, res, D, hz - rise, fb, f, int(rng.integers(1 << 30)))
        if got is not None:
            return got
    return None


def horizon_in_frame(shot: dict, res) -> bool:
    hz = horizon_row(_cam(shot, res))
    return hz is not None and 0.02 * res[1] <= hz <= 0.85 * res[1]


# ------------------------------------------------------------------------------------------------------ files

def write_obj(path: Path, parts: list, comment: str) -> list:
    """One OBJ with a usemtl group per material, Y mirrored for Unreal's importer and each face reversed once more so the
    engine sees outward faces (as factory.humpback4k.write_obj with winding_fix)."""
    lines = [f"# {comment}", f"mtllib {path.stem}.mtl"]
    base = 0; mats = []
    by_mat: dict[str, list] = {}
    for p in parts:
        by_mat.setdefault(p.mat, []).append(p)
    for mat, group in by_mat.items():
        lines.append(f"usemtl {mat}"); mats.append(mat)
        for p in group:
            lines.extend(f"v {x:.3f} {-y:.3f} {z:.3f}" for x, y, z in p.V)
            lines.extend(f"vt {u:.5f} {1.0 - t:.5f}" for u, t in p.UV)
            for a, b, c in p.F[:, [0, 2, 1]] + base + 1:
                lines.append(f"f {a}/{a} {b}/{b} {c}/{c}")
            base += len(p.V)
    path.write_text("\n".join(lines) + "\n")
    path.with_suffix(".mtl").write_text("".join(f"newmtl {m}\n" for m in mats))
    return mats


def save_parts(path: Path, parts: list) -> None:
    np.savez_compressed(path, n=np.int32(len(parts)), **{f"V{k}": p.V.astype(np.float32) for k, p in enumerate(parts)},
                        **{f"F{k}": p.F.astype(np.int32) for k, p in enumerate(parts)},
                        cls=np.array([p.cls for p in parts]), mat=np.array([p.mat for p in parts]))


def load_parts(path) -> list:
    z = np.load(path)
    return [Part(str(z["cls"][k]), str(z["mat"][k]), z[f"V{k}"].astype(float), z[f"F{k}"]) for k in range(int(z["n"]))]


def _spec(parts, yaw, origin, out_dir: Path, name: str, extra: dict) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    obj = out_dir / f"{name}.obj"
    mats = write_obj(obj, parts, f"{name} (factory.boat_scene {VERSION})")
    save_parts(out_dir / f"{name}.npz", parts)
    return {"obj": str(obj.resolve()), "npz": str((out_dir / f"{name}.npz").resolve()), "materials": mats,
            "location_cm": [round(float(x), 2) for x in origin], "yaw_deg": round(float(yaw), 4),
            "faces": int(sum(len(p.F) for p in parts)), **extra}


# ----------------------------------------------------------------------------------------------------- apply

def _plan(rng, shots, cfg, res) -> dict:
    """shot index -> content type: marker_rail:<colour>, marker_deck, marker_land, corner, corner_deck, horizon, or none."""
    n = len(shots)
    plan = {}
    weights = cfg.get("tape_colour_weights") or {"red": 0.27, "green": 0.27, "blue": 0.27, "yellow": 0.19}
    names = list(weights); p = np.array([weights[k] for k in names], float); p /= p.sum()
    forced = cfg.get("force_plan")
    order = list(rng.permutation(n))
    horizon_ok = [i for i in order if not shots[i].get("negative") and horizon_in_frame(shots[i], res)]
    sunny = [i for i in order if not shots[i].get("overcast")]
    seas = cfg.get("rail_sea_states")                      # round 20h: rail markers on these sea states (None = any)
    rail_ok = [i for i in sunny if not seas or shots[i].get("sea_state") in seas]
    if forced:
        free = [i for i in order]
        def take(pool):
            for i in pool:
                if i in free:
                    free.remove(i); return i
            return None
        for key in ("horizon",):
            for _ in range(int(forced.get(key, 0))):
                i = take(horizon_ok)
                if i is not None:
                    plan[i] = "horizon"
        whale_pool = [i for i in order if not shots[i].get("negative")]
        for key in ("corner", "corner_deck"):
            for _ in range(int(forced.get(key, 0))):
                i = take(whale_pool)
                if i is not None:
                    plan[i] = key
        for key, count in forced.items():
            if key.startswith("rail") or key in ("deck", "land"):
                for _ in range(int(count)):
                    i = take(rail_ok + sunny if key.startswith("rail") else sunny if key == "deck" else order)  # round 20h: rail / deck in sun
                    if key == "land" and len(plan) % 2 == 0:                                   # every other land marker sunny if any
                        j = take(sunny)
                        if j is not None:
                            free.append(i); i = j
                    if i is None:
                        continue
                    if key.startswith("rail"):
                        colour = key.split(":")[1] if ":" in key else names[int(rng.choice(len(names), p=p))]
                        plan[i] = f"marker_rail:{colour}"
                    else:
                        plan[i] = f"marker_{key}"
        return plan
    mf = cfg.get("marker_fraction") or {}
    # round 20h: rail markers only on sunny shots of rail_sea_states, deck markers only on sunny shots (overcast ones read
    # black); the probability on the allowed shots is raised so the share of all shots stays marker_fraction
    rail_set, sunny_set = set(rail_ok), set(sunny)
    k_rail = n / max(len(rail_set), 1); k_deck = n / max(len(sunny_set), 1)
    for i in order:
        x = rng.uniform()
        r = float(mf.get("rail", 0)) * (k_rail if i in rail_set else 0.0)
        dk = float(mf.get("deck", 0)) * (k_deck if i in sunny_set else 0.0)
        ld = float(mf.get("land", 0))
        if x < r:
            plan[i] = f"marker_rail:{names[int(rng.choice(len(names), p=p))]}"
        elif x < r + dk:
            plan[i] = "marker_deck"
        elif x < r + dk + ld:
            plan[i] = "marker_land"
    whale = [i for i in order if i not in plan and not shots[i].get("negative")]
    share_h = len([i for i in horizon_ok if i not in plan]) / max(len(whale), 1)
    p_h = min(1.0, float(cfg.get("horizon_land_fraction", 0.0)) / max(share_h, 1e-6)) if share_h > 0 else 0.0
    for i in whale:
        if i in horizon_ok and rng.uniform() < p_h:
            plan[i] = "horizon"
    for i in whale:
        if rng.uniform() < float(cfg.get("corner_fraction", 0.0)):
            deck = rng.uniform() < float(cfg.get("corner_deck_share", 0.3))
            plan[i] = (plan[i] + "+" if i in plan else "") + ("corner_deck" if deck else "corner")
    return plan


def _local(part: Part):
    """A world-space part moved to its own centre (spawned there, yaw 0): keeps the OBJ's numbers small."""
    c = part.V.mean(0)
    return Part(part.cls, part.mat, part.V - c, part.F, part.UV), c


def apply(shots: list, config: dict, out_dir: Path, seed: int) -> tuple[list, dict]:
    """Add shot["boat20"] to every shot (dormant: returns shots unchanged without config["boat_scene"]). Uses its own random
    stream, so every other shot field and every shot without new content is exactly as without the key. Marker shots
    (round 20h) also change exposure_ev100 (marker_exposure_ev100), land markers the fog density, and carry per-shot
    console variables in boat20.cvars (round bokeh, blur cap) that wf_build_batch keys into the shot's own CVar section."""
    cfg = config.get("boat_scene")
    if not cfg:
        return shots, {}
    res = tuple(config["resolution"])
    rng = np.random.default_rng([int(seed), 20206])
    plan = _plan(rng, shots, cfg, res)
    stats = {"planned": {}, "placed": {}, "failed": {}}
    gdir = Path(out_dir) / "boat20"
    ev_count: dict = {}
    for i, shot in enumerate(shots):
        srng = np.random.default_rng([int(seed), 20206, i])
        kinds = (plan.get(i) or "").split("+") if plan.get(i) else []
        b = {"version": VERSION, "shot_type": "whale", "marker": None, "boat": None, "land": None, "tape": None, "content": []}
        for kind in kinds:
            stats["planned"][kind.split(":")[0]] = stats["planned"].get(kind.split(":")[0], 0) + 1
            got = None
            if kind.startswith("marker_rail"):
                colour = kind.split(":")[1]
                got = place_marker_rail(srng, shot, res, colour, cfg)
                if got:
                    parts, yaw, origin, info, extra = got
                    b["boat"] = _spec(parts, yaw, origin, gdir, f"{shot['id']}_boat", {"variant": "marker_rail"})
                    b["tape"] = info; b["marker"] = {"type": "rail", "colour": colour, "meaning": TAPE_MEANING[colour]}
                    b["cvars"] = _bokeh_cvars(bg_radius_frac=extra["bg_blur_frac"])
                    if cfg.get("marker_saturation_shadows") is not None:
                        b["grade"] = {"saturation_shadows": float(cfg["marker_saturation_shadows"])}
                    ev = _marker_ev(cfg, "rail", ev_count, -1.2)
            elif kind == "marker_deck":
                got = place_marker_deck(srng, shot, res, cfg)
                if got:
                    parts, yaw, origin, info, extra = got
                    b["boat"] = _spec(parts, yaw, origin, gdir, f"{shot['id']}_boat", {"variant": "marker_deck", **info})
                    b["marker"] = {"type": "deck", "colour": None, "meaning": "unknown"}
                    b["cvars"] = _bokeh_cvars()
                    if cfg.get("marker_saturation_shadows") is not None:
                        b["grade"] = {"saturation_shadows": float(cfg["marker_saturation_shadows"])}
                    ev = _marker_ev(cfg, "deck", ev_count, -0.9)
            elif kind == "marker_land":
                got = place_marker_land(srng, shot, res, cfg)
                if got:
                    land, yaw, origin, info, extra = got
                    b["land"] = _spec([land], yaw, origin, gdir, f"{shot['id']}_land", info)
                    b["marker"] = {"type": "land", "colour": None, "meaning": "boat moves to a new group"}
                    if extra["cloud"] is not None:
                        cpart, cinfo = extra["cloud"]; cl, c0 = _local(cpart)
                        b["clouds"] = _spec([cl], 0.0, c0, gdir, f"{shot['id']}_clouds", cinfo)
                    fog = shot.get("fog") or {}
                    if fog.get("density") is not None:
                        b["fog_density_original"] = fog["density"]
                        fog["density"] = round(float(fog["density"]) * float(extra["fog_factor"]), 6)
                    ev = _marker_ev(cfg, "land_clear" if extra["clear"] else "land_hazy", ev_count, 0.2 if extra["clear"] else 0.5)
            elif kind in ("corner", "corner_deck"):
                got = place_corner(srng, shot, res, with_deck=kind == "corner_deck")
                if got:
                    parts, yaw, origin, info = got
                    b["boat"] = _spec(parts, yaw, origin, gdir, f"{shot['id']}_boat", {"variant": kind, **info})
                    if cfg.get("corner_dof_cap"):          # round 20h option: foreground blur cap at the rail's physical blur
                        r = coc_radius_px(shot, 100.0 * float(info["distance_m"]), res[0]) / res[0]
                        if r > 0.025:
                            b["cvars"] = {"r.DOF.Kernel.MaxForegroundRadius": round(float(min(r, 0.08)), 4)}
            elif kind == "horizon":
                got = place_horizon_land(srng, shot, res)
                if got:
                    land, yaw, origin, info = got
                    b["land"] = _spec([land], yaw, origin, gdir, f"{shot['id']}_land", {"variant": "horizon", **info})
            key = "placed" if got else "failed"
            stats[key][kind.split(":")[0]] = stats[key].get(kind.split(":")[0], 0) + 1
            if got:
                b["content"].append(kind)
                if kind.startswith("marker"):
                    b["shot_type"] = kind.split(":")[0]
                    shot["negative"] = True; shot["no_whale_in_frame"] = True
                    if shot.get("exposure_ev100") is not None and ev:
                        b["exposure_ev100_original"] = shot["exposure_ev100"]
                        shot["exposure_ev100"] = round(float(shot["exposure_ev100"]) + ev, 3)
                    b["exposure_offset_ev"] = ev
                    floor = cfg.get("marker_sea_floor") or {}
                    use = srng.uniform() < float(floor.get("share", 0.0)) if floor else False
                    if floor.get("pattern") and kind != "marker_land":      # look renders: on / off in turn per marker type
                        fk = "floor_" + kind.split(":")[0]
                        kk = ev_count.get(fk, 0); ev_count[fk] = kk + 1
                        use = bool(floor["pattern"][kk % len(floor["pattern"])])
                    if kind != "marker_land" and use:
                        dep = 100.0 * float(srng.uniform(*floor.get("depth_m", [2.0, 4.0])))
                        if floor.get("depth_ladder"):                     # look renders: fixed depths in turn
                            dk = ev_count.get("depth_" + kind.split(":")[0], 0); ev_count["depth_" + kind.split(":")[0]] = dk + 1
                            dep = 100.0 * float(floor["depth_ladder"][dk % len(floor["depth_ladder"])])
                        fl, c0 = _local(_floor_part(extra["cam_w"], extra["dir_w"], dep))
                        b["floor"] = _spec([fl], 0.0, c0, gdir, f"{shot['id']}_floor", {"depth_m": round(dep / 100.0, 2)})
                    break
        shot["boat20"] = b
    return shots, stats


# ------------------------------------------------------------------------------------------------- labels

def _batch_islands(raw_dir: Path) -> list:
    try:
        batch = json.loads((Path(raw_dir) / "batch.json").read_text())
    except Exception:
        return []
    out = []
    for spec in batch.get("islands") or []:
        try:
            V, F = [], []
            for line in Path(spec["mesh_obj"]).read_text().splitlines():
                if line.startswith("v "):
                    x, y, z = (float(t) for t in line.split()[1:4]); V.append((x, -y, z))
                elif line.startswith("f "):
                    F.append([int(t.split("/")[0]) - 1 for t in line.split()[1:4]])
            sc = float(spec.get("scale", 1.0))
            R = rotation_matrix(0.0, 0.0, float(spec.get("yaw_deg", 0.0)))
            out.append(Part("land", "island", (np.asarray(V) * sc) @ R + np.asarray(spec["location_cm"], float), np.asarray(F)))
        except Exception:
            continue
    return out


def world_parts(spec: dict) -> list:
    R = rotation_matrix(0.0, 0.0, float(spec["yaw_deg"]))
    return [p.moved(R, spec["location_cm"]) for p in load_parts(spec["npz"])]


def scene_map(shot: dict, raw_dir, inst: np.ndarray, mask: np.ndarray, camera: project.Camera):
    """The scene class image and the boat's own class image (before the whale is laid over it)."""
    h, w = inst.shape
    b = shot.get("boat20") or {}
    land_parts = _batch_islands(raw_dir) + (world_parts(b["land"]) if b.get("land") else [])
    scene = np.zeros((h, w), np.uint8)
    if land_parts:
        scene[fill_mask(land_parts, camera, w, h)] = SCENE_CLASSES["land"]
    scene[inst > 0] = SCENE_CLASSES["whale"]
    scene[(mask == 2) & (inst == 0)] = SCENE_CLASSES["whale_through_water"]
    boat = np.zeros((h, w), np.uint8); zb = np.full((h, w), np.inf)
    if b.get("boat"):
        boat, zb = raster_parts(world_parts(b["boat"]), camera, w, h)
        scene[boat > 0] = boat[boat > 0]
    return scene, boat, zb


def coc_radius_px(shot: dict, depth_cm: float, width: int) -> float:
    """Thin-lens blur circle radius (px) of a point at depth_cm with the shot's focus and f-number (the engine's diaphragm DOF
    caps it near r.DOF.Kernel.MaxForegroundRadius = 2.5 % of the frame width by default)."""
    cam = shot["camera"]; f = float(cam["focal_mm"]); n = float(cam.get("aperture_f", 8.0))
    s = 1000.0 * float(cam.get("focus_distance_m") or cam.get("distance_m") or 100.0); d = 10.0 * depth_cm
    c = (f * f / (n * max(s - f, 1.0))) * abs(d - s) / max(d, 1.0)
    return c / float((cam.get("filmback_mm") or [project.SENSOR_WIDTH_MM])[0]) * width / 2.0


def add_scene_labels(meta: dict, shot: dict, raw_dir, inst: np.ndarray, mask: np.ndarray, camera: project.Camera, out: Path) -> None:
    """factory.postprocess hook (shots with "boat20"): dataset/scene/<id>.png and the meta fields; the quality gate's
    white-background share is recomputed without the boat's own (white) pixels."""
    scene, boat, zb = scene_map(shot, raw_dir, inst, mask, camera)
    (out / "scene").mkdir(exist_ok=True)
    img = Image.fromarray(scene, mode="P")
    pal = [c for rgb in SCENE_PALETTE for c in rgb]; img.putpalette(pal + [0] * (768 - len(pal)))
    img.save(out / "scene" / f"{shot['id']}.png")
    counts = {name: int((scene == k).sum()) for name, k in SCENE_CLASSES.items()}
    b = shot["boat20"]
    meta["shot_type"] = b["shot_type"]
    meta["marker"] = b["marker"]
    meta["scene_classes"] = SCENE_CLASSES
    meta["scene_pixels"] = counts
    meta["scene_in_frame"] = [k for k, v in counts.items() if v > 0 and k != "water_sky"]
    meta["boat20_label_check"] = {"whale_label_on_boat_px": int(((inst > 0) & (boat > 0)).sum()),
                                  "boat_px": int((boat > 0).sum())}
    if (boat > 0).any():
        dm = float(np.median(zb[boat > 0]))
        meta["boat20_label_check"].update(boat_depth_m=round(dm / 100.0, 3),
                                          coc_radius_px_est=round(coc_radius_px(shot, dm, inst.shape[1]), 1))
    meta.setdefault("paths", {})["scene"] = f"scene/{shot['id']}.png"
    g = meta.get("quality_gate")
    if g and (boat > 0).any():
        rgb = np.asarray(Image.open(out / "images" / f"{shot['id']}.png").convert("RGB"))
        white_bg = float(((rgb.min(axis=2) >= 225) & (inst == 0) & (boat == 0)).mean())
        g["white_background_boat_excluded"] = round(white_bg, 4)
        if "white_water_over_frame" in g["reasons"] and white_bg <= 0.3:
            g["reasons"] = [r for r in g["reasons"] if r != "white_water_over_frame"]
            g["rejected"] = bool(g["reasons"]); g["boat20_adjusted"] = True
    if g is not None and str(meta.get("shot_type", "whale")).startswith("marker_"):
        # round 20h: marker shots get their own gate (the whale gate's brightness band and water floor were made for whale
        # photos; its reasons are kept in whale_gate_reasons); whale shots are gated exactly as before
        rgb = np.asarray(Image.open(out / "images" / f"{shot['id']}.png").convert("RGB"))
        m = marker_gate(rgb, scene, inst, b["shot_type"], (b.get("marker") or {}).get("colour"))
        g["whale_gate_reasons"] = list(g.get("reasons") or [])
        g["marker_gate"] = m
        g["reasons"] = list(m["reasons"]); g["rejected"] = bool(m["reasons"])


TAPE_HUE_SECTORS = {"red": (330.0, 60.0), "yellow": (60.0, 105.0), "green": (105.0, 215.0), "blue": (215.0, 330.0)}


def _lab(rgb: np.ndarray) -> np.ndarray:
    c = np.asarray(rgb, np.float32) / 255.0
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]], np.float32)
    xyz = c @ M.T / np.array([0.95047, 1.0, 1.08883], np.float32)
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def marker_gate(rgb: np.ndarray, scene: np.ndarray, inst: np.ndarray, shot_type: str, colour: str | None) -> dict:
    """Round 20h: what a usable marker photo needs, from first principles (not fitted to any real day):
    every type: no whale in frame; not blown out (<= 25 % of the frame clipped); not black (frame mean L* >= 12).
    rail: the tape big enough to read (>= 0.04 % of the frame), not clipped (<= 15 % of its pixels), its colour nameable
    (median chroma >= 12 and the median hue inside the colour's sector: red / pink / coral 330-60, yellow 60-105,
    green 105-215, blue 215-330 deg) and apart from the tube (dE >= 15); the tube reads white but keeps shading (median
    L* >= 55, <= 70 % of its pixels clipped); the background neither black nor washed out (median L* 6-75).
    deck: the bitt >= 3 % of the frame; bitt + deck not mostly clipped (<= 50 %).
    land: land >= 5 % of the frame, its median L* 12-92 and set off from the sky / sea (|dL*| >= 3 or its own L* spread
    >= 3)."""
    small = slice(None, None, 2)
    rgb2, sc, ins = rgb[small, small], scene[small, small], inst[small, small]
    lab = _lab(rgb2); L = lab[..., 0]
    clip = rgb2.max(axis=2) >= 250
    reasons = []; m = {}
    m["frame_clipped"] = round(float(clip.mean()), 4); m["frame_L_mean"] = round(float(L.mean()), 2)
    if (ins > 0).any():
        reasons.append("whale_in_marker")
    if clip.mean() > 0.25:
        reasons.append("clipped_frame")
    if L.mean() < 12.0:
        reasons.append("frame_black")
    n = sc.size
    if shot_type == "marker_rail":
        tape = sc == SCENE_CLASSES["tape"]; tube = sc == SCENE_CLASSES["rail"]; bg = sc == 0
        m["tape_frac"] = round(float(tape.sum() / n), 5)
        if tape.sum() / n < 0.0004:
            reasons.append("tape_small")
        if tape.any():
            tl = np.median(lab[tape], axis=0); chroma = float(np.hypot(tl[1], tl[2])); hue = float(np.degrees(np.arctan2(tl[2], tl[1])) % 360)
            m.update(tape_lab=[round(float(x), 1) for x in tl], tape_chroma=round(chroma, 1), tape_hue=round(hue, 1),
                     tape_clipped=round(float(clip[tape].mean()), 4))
            lo, hi = TAPE_HUE_SECTORS.get(colour, (0.0, 360.0))
            inside = (lo <= hue < hi) if lo < hi else (hue >= lo or hue < hi)
            if chroma < 12.0 or not inside:
                reasons.append("tape_colour_unreadable")
            if clip[tape].mean() > 0.15:
                reasons.append("tape_clipped")
            if tube.any():
                ul = np.median(lab[tube], axis=0)
                m["tape_tube_dE"] = round(float(np.linalg.norm(tl - ul)), 1)
                if np.linalg.norm(tl - ul) < 15.0:
                    reasons.append("tape_low_contrast")
        if tube.any():
            m.update(tube_L_median=round(float(np.median(L[tube])), 1), tube_clipped=round(float(clip[tube].mean()), 4))
            if np.median(L[tube]) < 55.0:
                reasons.append("tube_dark")
            if clip[tube].mean() > 0.7:
                reasons.append("tube_clipped")
        if bg.sum() > 100:
            bl = float(np.median(L[bg])); m["background_L_median"] = round(bl, 1)
            m["background_lab"] = [round(float(x), 1) for x in np.median(lab[bg], axis=0)]
            if bl < 6.0:
                reasons.append("sea_black")
            elif bl > 75.0:
                reasons.append("sea_washed")
    elif shot_type == "marker_deck":
        bitt = sc == SCENE_CLASSES["cleat"]; white = bitt | (sc == SCENE_CLASSES["deck"])
        m["bitt_frac"] = round(float(bitt.sum() / n), 4)
        if bitt.sum() / n < 0.03:
            reasons.append("bitt_small")
        if white.any():
            m["deck_clipped"] = round(float(clip[white].mean()), 4); m["deck_L_median"] = round(float(np.median(L[white])), 1)
            if clip[white].mean() > 0.5:
                reasons.append("deck_clipped")
        bg = sc == 0
        if bg.sum() > 100:
            m["background_lab"] = [round(float(x), 1) for x in np.median(lab[bg], axis=0)]
    elif shot_type == "marker_land":
        land = sc == SCENE_CLASSES["land"]; rest = sc == 0
        m["land_frac"] = round(float(land.sum() / n), 4)
        if land.sum() / n < 0.05:
            reasons.append("land_small")
        if land.any():
            ll = float(np.median(L[land])); spread = float(np.percentile(L[land], 90) - np.percentile(L[land], 10))
            m.update(land_L_median=round(ll, 1), land_L_spread=round(spread, 1), land_lab=[round(float(x), 1) for x in np.median(lab[land], axis=0)])
            if not 12.0 <= ll <= 92.0:
                reasons.append("land_too_dark" if ll < 12.0 else "land_washed")
            dl = abs(ll - float(np.median(L[rest]))) if rest.sum() > 100 else 99.0
            m["land_vs_rest_dL"] = round(dl, 1)
            if dl < 3.0 and spread < 3.0:
                reasons.append("land_invisible")
    m["reasons"] = reasons
    return m


# --------------------------------------------------------------------------------------------------------- CLI

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("apply"); a.add_argument("run_dir", type=Path); a.add_argument("config", type=Path)
    a.add_argument("--seed", type=int, default=None)
    rl = sub.add_parser("rebuild-land"); rl.add_argument("run_dir", type=Path)
    p = sub.add_parser("preview"); p.add_argument("run_dir", type=Path); p.add_argument("--ids", default=None)
    p.add_argument("--scale", type=float, default=0.25)
    args = ap.parse_args(argv)
    if args.cmd == "apply":
        config = json.loads(args.config.read_text())
        rc = json.loads((args.run_dir / "run_config.json").read_text())
        seed = args.seed if args.seed is not None else int(rc["seed"])
        shots = [json.loads(l) for l in (args.run_dir / "shotlist.jsonl").read_text().splitlines() if l.strip()]
        shots, stats = apply(shots, config, args.run_dir, seed)
        by_id = {s["id"]: s for s in shots}
        (args.run_dir / "shotlist.jsonl").write_text("".join(json.dumps(s) + "\n" for s in shots))
        for bp in sorted((args.run_dir / "batches").glob("batch_*.json")):
            batch = json.loads(bp.read_text())
            batch["shots"] = [by_id[s["id"]] for s in batch["shots"]]
            bp.write_text(json.dumps(batch))
        rc["config"]["boat_scene"] = config["boat_scene"]
        (args.run_dir / "run_config.json").write_text(json.dumps(rc, indent=2))
        (args.run_dir / "boat20_apply.json").write_text(json.dumps(stats, indent=1))
        print(json.dumps(stats))
        return 0
    if args.cmd == "rebuild-land":
        # the land meshes of an existing run re-made with the current land_mesh (same placement, files overwritten)
        for line in (args.run_dir / "shotlist.jsonl").read_text().splitlines():
            s = json.loads(line); L = (s.get("boat20") or {}).get("land")
            if not L:
                continue
            seed = int(L.get("seed", int(s["id"]) * 7919 + 11))
            land = land_mesh(seed, float(L["length_m"]), float(L["crest_m"]) / 0.85, float(L["slope_deg"]), float(L["cell_m"]),
                             v2=bool(L.get("v2")))
            write_obj(Path(L["obj"]), [land], f"{s['id']}_land (factory.boat_scene {VERSION}, rebuilt)")
            save_parts(Path(L["npz"]), [land])
            print(s["id"], "land rebuilt", len(land.V), "vertices")
        return 0
    if args.cmd == "preview":
        shots = [json.loads(l) for l in (args.run_dir / "shotlist.jsonl").read_text().splitlines() if l.strip()]
        res = tuple(json.loads((args.run_dir / "run_config.json").read_text())["config"]["resolution"])
        want = set(args.ids.split(",")) if args.ids else None
        od = args.run_dir / "boat20_preview"; od.mkdir(exist_ok=True)
        pal = [c for rgb in SCENE_PALETTE for c in rgb]
        for s in shots:
            b = s.get("boat20") or {}
            if (want and s["id"] not in want) or not (b.get("boat") or b.get("land")):
                continue
            cam = _cam(s, res); w, h = int(res[0] * args.scale), int(res[1] * args.scale)
            cam_s = project.Camera(cam.location, s["camera"]["rotation"], s["camera"]["focal_mm"], w, h, sensor_width_mm=_fb(s))
            sc = np.zeros((h, w), np.uint8)
            if b.get("land"):
                sc[fill_mask(world_parts(b["land"]), cam_s, w, h)] = SCENE_CLASSES["land"]
            if b.get("boat"):
                bi, _ = raster_parts(world_parts(b["boat"]), cam_s, w, h); sc[bi > 0] = bi[bi > 0]
            im = Image.fromarray(sc, mode="P"); im.putpalette(pal + [0] * (768 - len(pal)))
            im.convert("RGB").save(od / f"{s['id']}_{b['shot_type']}.png")
            print(s["id"], b["shot_type"], b["content"], {k: int((sc == v).sum()) for k, v in SCENE_CLASSES.items() if (sc == v).any()})
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
