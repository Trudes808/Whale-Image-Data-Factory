"""run06 round 12: acceptance of the welded dorsal fin (factory.weld), offline, before any engine render. Gates
tools/build_dataset.sh like factory.blade_check (which it calls for runs without the weld).

Per posed whale (every whale of a run's shot list; with --sweep K also every library animal in K poses of that list),
on the dorsal region (the blade, the patch, and the rig faces near the region):
  intersect   pairs of non-adjacent triangles that cross each other (Moller's test; pairs sharing a position skipped)
  fold        edges whose two faces' normals differ by more than FOLD_DEG (a flipped / folded triangle)
  degenerate  faces with area < DEGEN_CM2 (x scale^2) or a corner under DEGEN_DEG
  stretch     edges longer than STRETCH x (or shorter than 1 / STRETCH x) the same edge on the animal's rest mesh x scale
  normal_jump the largest angle between the smooth vertex normals (over positions, as the importer computes them) of the
              two ends of an edge at the junctions: edges at the blade's foot ring, at the hump's edge (where the lift
              ends) and at the region's loop (patch to rig); fails above NORMAL_JUMP_DEG
Once per model: manifold (every edge touching a weld vertex shared by exactly two faces, opposite directions).
Outline (--outline): per library animal the welded rest mesh's side silhouette (midline top per along position and
leading / trailing edge per height) against its template, and the library's statistics.
Exit status 1 if any count is above zero.

    python3 -m factory.weld_check RUN_DIR [--sweep 40] [--workers 8] [--json OUT] [--outline]
    python3 -m factory.weld_check RUN_DIR --old ...     (the same measures on the round-11 design, for comparison)
    python3 -m factory.weld_check RUN_DIR --design 15 ...  (round 15 on a run made with another design, same shots)
The design checked is the run's own whale.blade_weld (15 = round 15, true = round 13) unless --design is given.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

FOLD_DEG = 100.0
DEGEN_CM2 = 1e-4
DEGEN_DEG = 0.5
STRETCH = 1.6
NORMAL_JUMP_DEG = 12.0
NORMAL_JUMP_MARGIN = 5.0   # round 13: a junction's jump fails above max(NORMAL_JUMP_DEG, the rig's own largest step among
                           # its edges near the loop, same whale and pose) + this (the bare rig's back is already at ~43 deg)
STRETCH_MIN_CM = 1.0       # ... and by more than this (mm-scale edges by the foot change by fractions of a mm)
RING_CM = 40.0             # rig faces within this of the loop belong to the checked region
_M = None


# ---- geometry tests ------------------------------------------------------------------------------------------------
def _pairs_aabb(lo: np.ndarray, hi: np.ndarray, chunk: int = 400000):
    """Candidate pairs (i < j) of boxes overlapping in x, y and z (sort and sweep on x)."""
    order = np.argsort(lo[:, 0])
    los, his = lo[order], hi[order]
    end = np.searchsorted(los[:, 0], his[:, 0], side="right")
    cnt = end - np.arange(len(order)) - 1
    cnt = np.maximum(cnt, 0)
    out_i, out_j = [], []
    starts = np.concatenate([[0], np.cumsum(cnt)])
    total = int(starts[-1])
    for s0 in range(0, total, chunk):
        s1 = min(total, s0 + chunk)
        k = np.arange(s0, s1)
        i = np.searchsorted(starts, k, side="right") - 1
        j = i + 1 + (k - starts[i])
        ok = (los[j, 1] <= his[i, 1]) & (his[j, 1] >= los[i, 1]) & (los[j, 2] <= his[i, 2]) & (his[j, 2] >= los[i, 2])
        out_i.append(order[i[ok]]); out_j.append(order[j[ok]])
    if not out_i:
        return np.zeros(0, int), np.zeros(0, int)
    return np.concatenate(out_i), np.concatenate(out_j)


def _tri_tri(V1: np.ndarray, V2: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Moller's triangle-triangle intersection, vectorised; V1, V2 (n, 3, 3). Coplanar pairs count as not crossing;
    touching (a vertex on the other plane) does not count."""
    def plane_d(T, P):
        N = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
        n = np.linalg.norm(N, axis=1, keepdims=True) + 1e-30
        N = N / n
        return N, ((P - T[:, None, 0]) * N[:, None, :]).sum(-1)
    N2, dv = plane_d(V2, V1)
    N1, du = plane_d(V1, V2)
    sc = np.maximum(np.abs(V1).max((1, 2)), 1.0)
    e = eps * sc[:, None]
    dv = np.where(np.abs(dv) < e, 0.0, dv); du = np.where(np.abs(du) < e, 0.0, du)
    sep = ((dv > 0).all(1) | (dv < 0).all(1) | (du > 0).all(1) | (du < 0).all(1))
    cop = (dv == 0).all(1) | (du == 0).all(1)
    D = np.cross(N1, N2)
    ax = np.argmax(np.abs(D), 1)
    idx = np.arange(len(V1))
    p1 = V1[idx, :, ax] if False else np.take_along_axis(V1, ax[:, None, None].repeat(3, 1), 2)[..., 0]
    p2 = np.take_along_axis(V2, ax[:, None, None].repeat(3, 1), 2)[..., 0]

    def interval(p, d):
        # the vertex alone on its side (or on the plane with the two others on one side)
        s = np.sign(d)
        lone = np.zeros(len(p), int)
        for k in range(3):
            o1, o2 = (k + 1) % 3, (k + 2) % 3
            m = (s[:, o1] == s[:, o2]) & (s[:, k] != s[:, o1])
            lone = np.where(m & (lone == 0) | m, k, lone)
        # degenerate sign patterns (two zeros etc.): pick the vertex with the largest |d| difference
        k = lone; o1 = (k + 1) % 3; o2 = (k + 2) % 3
        pk, po1, po2 = p[idx, k], p[idx, o1], p[idx, o2]
        dk, do1, do2 = d[idx, k], d[idx, o1], d[idx, o2]
        t1 = np.where(np.abs(dk - do1) > 1e-30, pk + (po1 - pk) * dk / np.where(np.abs(dk - do1) > 1e-30, dk - do1, 1.0), pk)
        t2 = np.where(np.abs(dk - do2) > 1e-30, pk + (po2 - pk) * dk / np.where(np.abs(dk - do2) > 1e-30, dk - do2, 1.0), pk)
        return np.minimum(t1, t2), np.maximum(t1, t2)
    a0, a1 = interval(p1, dv)
    b0, b1 = interval(p2, du)
    hit = ~sep & ~cop & (a1 > b0 + 1e-9 * sc) & (b1 > a0 + 1e-9 * sc)
    return hit


def self_intersections(V: np.ndarray, F: np.ndarray, pid: np.ndarray) -> int:
    T = V[F]
    lo, hi = T.min(1), T.max(1)
    i, j = _pairs_aabb(lo, hi)
    if not len(i):
        return 0
    Pi, Pj = pid[F[i]], pid[F[j]]
    share = (Pi[:, :, None] == Pj[:, None, :]).any((1, 2))
    i, j = i[~share], j[~share]
    n = 0
    for s in range(0, len(i), 200000):
        n += int(_tri_tri(T[i[s:s + 200000]], T[j[s:s + 200000]]).sum())
    return n


def _edges(F: np.ndarray):
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    fid = np.tile(np.arange(len(F)), 3)
    return E, fid


def vertex_normals_pos(V, F, pid):
    tri = V[F]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    acc = np.zeros((int(pid.max()) + 1, 3))
    for k in range(3):
        np.add.at(acc, pid[F[:, k]], fn)
    N = acc[pid]
    return N / (np.linalg.norm(N, axis=1, keepdims=True) + 1e-12), fn


# ---- per-model setup -----------------------------------------------------------------------------------------------
class Ctx:
    """Faces of the dorsal region and the junction edges of a welded model (fixed topology)."""

    def __init__(self, m):
        from factory import blade as bl
        self.m = m
        W = m.weld; n0 = m.n_rig; R2 = 2 * bl.NC
        from scipy.spatial import cKDTree
        F = np.asarray(m.faces)
        loopv0 = np.zeros(len(m.rest), bool)
        for k in W.loop_weld:
            for e in W.loop_copies[k].values():
                loopv0[int(m._ext2model(np.array([e]))[0])] = True
        d, _ = cKDTree(m.rest[loopv0]).query(m.rest[F].mean(1))
        near = (F >= n0).any(1) | (d < RING_CM)                         # every weld face and the rig faces near the loop
        self.fidx = np.nonzero(near)[0]
        self.F = F[self.fidx]
        # junction edges: at the foot ring, at the loop (patch <-> rig)
        E, _ = _edges(self.F)
        Es = np.unique(np.sort(E, 1), axis=0)
        foot = np.zeros(len(m.rest), bool); foot[n0:n0 + R2] = True
        loopv = np.zeros(len(m.rest), bool)
        for k in W.loop_weld:
            for e in W.loop_copies[k].values():
                loopv[int(m._ext2model(np.array([e]))[0])] = True
        self.loopv = loopv
        self.E = Es
        # foot junction: edges from a foot-ring vertex to a vertex off the ring (blade ring 1 or patch row 1), without
        # the trailing edge's knife (its two sides meet at a sharp edge by design, above and below the foot)
        te = np.zeros(len(m.rest), bool)
        te[[n0 + k * R2 + c for k in range(m.n_blade // R2) for c in (0, 1, R2 - 2, R2 - 1)]] = True
        self.e_foot = (foot[Es].sum(1) == 1) & ~(te[Es].all(1))
        # loop junction: patch-to-loop edges; the rig's own neighbour edges among the kept faces near the loop are the
        # baseline (the rig's coarse back has its own normal steps)
        newv = np.zeros(len(m.rest), bool); newv[n0:] = True
        self.e_loop = loopv[Es].any(1) & newv[Es].any(1) & ~(newv[Es].all(1) & (Es >= m._off_cut).all(1))
        kept_v = np.zeros(len(m.rest), bool); kept_v[np.unique(F[self.fidx][self.fidx < m._face_ranges["kept"][1]])] = True
        self.e_rig = (Es < n0).all(1) & kept_v[Es].all(1)
        self.e_weld = newv[Es].any(1)
        patch = np.zeros(len(m.rest), bool); patch[m._off_patch:m._off_cut] = True; patch[n0:n0 + R2] = True
        self.patchv = patch


def manifold(m) -> dict:
    """Edges touching weld vertices (blade, patch, cut, loop) on the position-welded mesh: used by exactly two faces in
    opposite directions."""
    F = np.asarray(m.faces)
    _, pid = np.unique(np.round(m.rest, 4), axis=0, return_inverse=True); pid = pid.ravel()
    ctx = Ctx(m)
    touch = np.zeros(len(m.rest), bool); touch[m.n_rig:] = True; touch |= ctx.loopv
    tp = np.zeros(int(pid.max()) + 1, bool); tp[pid[touch]] = True
    Fw = pid[F]
    E = np.concatenate([Fw[:, [0, 1]], Fw[:, [1, 2]], Fw[:, [2, 0]]])
    sel = tp[E].any(1)
    E = E[sel]
    key = np.sort(E, 1)
    u, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    bad_count = int((cnt != 2).sum())
    d, dc = np.unique(E, axis=0, return_counts=True)
    bad_dir = int((dc > 1).sum())
    return {"edges": int(len(u)), "not_two_faces": bad_count, "same_direction": bad_dir}


def check_animal(m, ctx: Ctx, animal: dict, details: bool = False) -> dict:
    ind = animal.get("individual"); scale = float(animal.get("body_scale", 1.0))
    V = np.asarray(m.posed(animal), float)
    F = ctx.F
    _, pid = np.unique(np.round(V, 4), axis=0, return_inverse=True); pid = pid.ravel()
    inter = self_intersections(V, F, pid)
    N, fn = vertex_normals_pos(V, F, pid)
    area = 0.5 * np.linalg.norm(fn, axis=1)
    T = V[F]
    ang = []
    for k in range(3):
        a = T[:, (k + 1) % 3] - T[:, k]; b = T[:, (k + 2) % 3] - T[:, k]
        ang.append(np.degrees(np.arccos(np.clip((a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12), -1, 1))))
    min_ang = np.min(ang, 0)
    degen = int(((area < DEGEN_CM2 * scale * scale) | (min_ang < DEGEN_DEG)).sum())
    # folds: dihedral across edges (faces welded by position)
    Fw = pid[F]
    E = np.concatenate([Fw[:, [0, 1]], Fw[:, [1, 2]], Fw[:, [2, 0]]]); fid = np.tile(np.arange(len(F)), 3)
    key = np.sort(E, 1)
    o = np.lexsort((key[:, 1], key[:, 0])); ks = key[o]
    same = (ks[1:] == ks[:-1]).all(1)
    fa, fb = fid[o][:-1][same], fid[o][1:][same]
    un = fn / (np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12)
    dih = np.degrees(np.arccos(np.clip((un[fa] * un[fb]).sum(1), -1, 1)))
    # flipped faces: posed weld faces against their rest normals carried by the back's motion (the bin rotations of
    # factory.weld); at rest the patch is a height field along its lift direction and the blade rings must keep their
    # order (round 11's ring tests)
    from factory import weld as wd
    A0 = m._weld_animal(ind)
    Rr_ = np.asarray(m.posed({"individual": ind, "body_scale": scale}), float) if ind is not None else m.rest * scale
    Tr = Rr_[F]; fr = np.cross(Tr[:, 1] - Tr[:, 0], Tr[:, 2] - Tr[:, 0])
    bins = m.__dict__.get("_weld_bins") or wd.bend_bins(m.weld)
    W_ = m.weld
    Ex_p = W_.ext(np.asarray(V[: m.n_rig]))
    tr = wd.bin_transforms(bins, A0["Ex_rest"] * scale, W_.base_positions(np.asarray(V[: m.n_rig])) if False else Ex_p)
    ac = W_.chart(Rr_[F].mean(1) / scale)[:, 0]
    fr_t = wd.rotate_by_bins(fr, ac, (tr[0], tr[1]))
    weldf = (F >= m.n_rig).any(1)
    flip_pose = int((((fn * fr_t).sum(1) <= 0) & weldf & (area > DEGEN_CM2)).sum())
    pf = (F >= m._off_patch) | ((F >= m.n_rig) & (F < m.n_rig + 48))
    patchf = pf.all(1) | ((F >= m.n_rig) & ~((F >= m.n_rig + 48) & (F < m._off_patch))).all(1)
    dn = A0["lift_dir"] if "lift_dir" in A0 else W_.e_n
    sgn = np.sign(np.median((fr[patchf] @ dn)))
    flip_rest = int(((fr[patchf] @ dn) * sgn <= 0).sum())
    from factory import blade as bl
    R2 = 2 * bl.NC; nl = m.n_blade // R2
    ring = Rr_[m.n_rig: m.n_rig + nl * R2].reshape(nl, R2, 3)
    B_, al_, up_ = A0["frame_i"] if "frame_i" in A0 else m._blade_frame_for(A0["local"])
    x, h, y = ring @ al_, ring @ up_, ring[..., 1]
    tol = 0.3 * scale
    backwards = int(((np.diff(x[:, : bl.NC], axis=1) < -tol).any(1) | (np.diff(x[:, bl.NC:], axis=1) > tol).any(1)).sum())
    column_down = int((np.diff(h, axis=0) < -tol).sum())
    sides_cross = int(((y[:, : bl.NC] - y[:, bl.NC:][:, ::-1]) < -tol).any(1).sum())
    fold = flip_pose + flip_rest + backwards + column_down + sides_cross
    # stretch against the animal's rest weld x scale
    Rr = np.asarray(m.posed({"individual": ind, "body_scale": scale}), float) if ind is not None else m.rest * scale
    Es = ctx.E
    lp = np.linalg.norm(V[Es[:, 0]] - V[Es[:, 1]], axis=1); lr = np.linalg.norm(Rr[Es[:, 0]] - Rr[Es[:, 1]], axis=1)
    r = lp / np.maximum(lr, 1e-6)
    big = np.abs(lp - lr) > STRETCH_MIN_CM * scale
    bad_r = ((r > STRETCH) | (r < 1.0 / STRETCH)) & big
    stretch = int((bad_r & ctx.e_weld).sum())
    rw = r[ctx.e_weld & big] if (ctx.e_weld & big).any() else np.ones(1)
    rr = r[ctx.e_rig & big] if (ctx.e_rig & big).any() else np.ones(1)
    # normal continuity at the junctions
    jn = np.degrees(np.arccos(np.clip((N[Es[:, 0]] * N[Es[:, 1]]).sum(1), -1, 1)))
    A = m._weld_animal(ind)
    from factory import blade as bl
    R2 = 2 * bl.NC
    Dv = np.zeros(len(V)); Dv[m.n_rig:m.n_rig + R2] = A["g"]["D"][:R2]; Dv[m._off_patch:m._off_cut] = A["g"]["D"][R2:R2 + m.weld.n_int]
    thr = 0.02 * A["local"]["H"]
    edge_h = ctx.patchv[Es].all(1) & ((Dv[Es[:, 0]] > thr) != (Dv[Es[:, 1]] > thr))
    j_foot = float(jn[ctx.e_foot].max()); j_loop = float(jn[ctx.e_loop].max()); j_hump = float(jn[edge_h].max()) if edge_h.any() else 0.0
    j_rig = float(jn[ctx.e_rig].max())
    j_max = max(j_foot, j_hump, j_loop)
    if details:
        return {"flip_pose_faces": F[((fn * fr_t).sum(1) <= 0) & weldf & (area > DEGEN_CM2)],
                "flip_rest_faces": F[patchf][((fr[patchf] @ dn) * sgn <= 0)],
                "degen_faces": F[(area < DEGEN_CM2 * scale * scale) | (min_ang < DEGEN_DEG)],
                "stretch_edges": Es[bad_r & ctx.e_weld], "stretch_ratio": r[bad_r & ctx.e_weld],
                "backwards_rings": np.nonzero((np.diff(x[:, : bl.NC], axis=1) < -tol).any(1) | (np.diff(x[:, bl.NC:], axis=1) > tol).any(1))[0]}
    return {"intersect": inter, "fold": fold, "_fold_parts": [flip_pose, flip_rest, backwards, column_down, sides_cross],
            "_dihedral_over_100": int((dih > FOLD_DEG).sum()), "degenerate": degen, "stretch": stretch,
            "normal_jump": int(j_max > max(NORMAL_JUMP_DEG, j_rig) + NORMAL_JUMP_MARGIN),
            "_jump_deg": [round(j_foot, 2), round(j_hump, 2), round(j_loop, 2), round(j_rig, 2)], "_min_angle": round(float(min_ang.min()), 2),
            "_stretch_max": round(float(rw.max()), 3), "_stretch_min": round(float(rw.min()), 3),
            "_rig_stretch": [round(float(rr.min()), 3), round(float(rr.max()), 3)], "_dih_max": round(float(dih.max()), 1)}


def old_junction(m, animal: dict) -> dict:
    """The round-11 design (blade on the back): intersecting blade x back face pairs and the angle between their
    smooth normals there (the shading jump at the visible junction), and the posed blade's own self-intersections."""
    V = np.asarray(m.posed(animal), float)
    n0 = m.n_rig
    F = np.asarray(m.faces)
    _, pid = np.unique(np.round(V, 4), axis=0, return_inverse=True); pid = pid.ravel()
    N, _ = vertex_normals_pos(V, F, pid)
    fb = F[(F >= n0).all(1)]
    c = V[n0:].mean(0); rad = np.linalg.norm(V[n0:] - c, axis=1).max() + 20
    fr = F[(F < n0).all(1) & (np.linalg.norm(V[F].mean(1) - c, axis=1) < rad)]
    allF = np.concatenate([fb, fr]); nb = len(fb)
    T = V[allF]; i, j = _pairs_aabb(T.min(1), T.max(1))
    cross = (i < nb) != (j < nb)
    i, j = i[cross], j[cross]
    hit = _tri_tri(T[i], T[j]) if len(i) else np.zeros(0, bool)
    i, j = i[hit], j[hit]
    na = N[allF[i]].mean(1); nbv = N[allF[j]].mean(1)
    na /= np.linalg.norm(na, axis=1, keepdims=True) + 1e-12; nbv /= np.linalg.norm(nbv, axis=1, keepdims=True) + 1e-12
    angd = np.degrees(np.arccos(np.clip((na * nbv).sum(1), -1, 1)))
    own = self_intersections(V, fb, pid)
    return {"blade_back_crossings": int(hit.sum()), "jump_deg_median": round(float(np.median(angd)), 1) if len(angd) else 0.0,
            "jump_deg_max": round(float(angd.max()), 1) if len(angd) else 0.0, "blade_self_intersect": own}


# ---- outline -------------------------------------------------------------------------------------------------------
def outline(m) -> dict:
    """Per library animal: the welded rest mesh's side silhouette in the blade frame (units of H) against the template:
    leading / trailing edge at template heights 0.35 .. 0.95 (blade) and the midline top at along positions on the hump,
    and the statistics (base / H at the back line, broad = chord at 0.75 over H, sharp = chord 0.9 / chord 0.75,
    overhang = trailing edge at 0.75) from the mesh."""
    from factory import blade as bl
    rows = {}
    for k in sorted(m.individuals()):
        loc = m._blade_local(k); H = loc["H"]
        B, al, up = m._weld_animal(k)["frame_i"]
        V = np.asarray(m.posed({"individual": k, "body_scale": 1.0}), float)[m.n_rig:]
        a = ((V - B) @ al) / H; h = ((V - B) @ up) / H; y = V[:, 1] / H
        o = loc["outline"]; L, te_t, le_t = o["levels"], o["te"], o["le"]
        # silhouette from the mesh: for heights hq, the extreme along positions of points at that height (side view:
        # all y). Points of the blade rings and the patch.
        nb = m.n_blade

        def edges_at(hq, tol=0.012):
            s = (np.abs(h - hq) < tol) & (np.arange(len(h)) < nb)          # the blade (the fin above its foot)
            return (float(a[s].min()), float(a[s].max())) if s.sum() >= 3 else (np.nan, np.nan)
        hs = np.round(np.arange(0.35, 0.96, 0.05), 3)
        err_le, err_te = [], []
        for hq in hs:
            t0, l0 = edges_at(hq)
            err_le.append(abs(l0 - float(np.interp(hq, L, le_t)))); err_te.append(abs(t0 - float(np.interp(hq, L, te_t))))
        # midline top (|y| small) per along position over the hump vs template height (min(H_FOOT, outline))
        from factory import weld as wd
        mid = np.abs(y) < 0.03
        te0, le0 = float(np.interp(0.0, L, te_t)), float(np.interp(0.0, L, le_t))
        # where the mesh's midline rises 0.02 H above the back's (the lift field): front / back base from the mesh
        A = m._weld_animal(k); R2 = 2 * bl.NC
        P2 = np.concatenate([A["g"]["P2"][R2:R2 + m.weld.n_int]])
        Dp = A["g"]["D"][R2:R2 + m.weld.n_int]
        Pp = V[m._off_patch - m.n_rig: m._off_cut - m.n_rig]
        ap = ((Pp - B) @ al) / H
        # round 13: base from the front / back base landmarks (the keypoints and fin_spec carry the same number); the
        # lifted extent (round 12's measure) is kept as base_over_H_lifted
        lifted = (Dp > 0.02 * H) & (np.abs(Pp[:, 1]) < 0.05 * H)
        lift_front, lift_back = float(ap[lifted].max()), float(ap[lifted].min())
        bo_mesh = m.weld_base_over_height(k)
        c75 = edges_at(0.75); c90 = edges_at(0.9)
        rows[k] = {"H": round(H, 1), "le_err_max": round(float(np.nanmax(err_le)), 3), "te_err_max": round(float(np.nanmax(err_te)), 3),
                   "base_over_H_mesh": round(bo_mesh, 3), "base_over_H_lifted": round(lift_front - lift_back, 3), "base_over_H_template": round(le0 - te0, 3),
                   "broad": round(c75[1] - c75[0], 3), "sharp": round((c90[1] - c90[0]) / max(c75[1] - c75[0], 1e-6), 3),
                   "overhang": round(c75[0], 3)}
    lib = {s: round(float(np.median([r[s] for r in rows.values()])), 3) for s in ("base_over_H_mesh", "base_over_H_template", "broad", "sharp", "overhang")}
    return {"animals": rows, "library_median": lib,
            "worst_le_err": max(r["le_err_max"] for r in rows.values()), "worst_te_err": max(r["te_err_max"] for r in rows.values())}


# ---- driver --------------------------------------------------------------------------------------------------------
DESIGN = None              # round 15: --design 13 / 15 overrides the run's whale.blade_weld (None: the run's own design)


def _model(config: dict, old: bool):
    global _M
    if _M is None:
        from factory.whalemodel import WhaleModel
        from factory import weld as wd
        c = json.loads(json.dumps(config))
        if old:
            c.setdefault("whale", {}).pop("blade_weld", None)
        else:
            d = DESIGN if DESIGN is not None else wd.design_of(c.get("whale", {}).get("blade_weld"))
            c.setdefault("whale", {})["blade_weld"] = 15 if d == 15 else True
        m = WhaleModel.from_config(c)
        _M = (m, None if old else Ctx(m))
    return _M


def _job(args):
    global DESIGN
    config, label, animal, old = args[:4]
    if len(args) > 4:
        DESIGN = args[4]
    m, ctx = _model(config, old)
    try:
        r = old_junction(m, animal) if old else check_animal(m, ctx, animal)
    except Exception as exc:
        import traceback
        r = {"error": f"{exc} {traceback.format_exc()[-300:]}"}
    return label, r


def jobs_for(run: Path, config: dict, sweep: int, limit: int = 0):
    shots = [json.loads(l) for l in (run / "shotlist.jsonl").read_text().splitlines() if l.strip()]
    if limit:
        shots = shots[:limit]
    jobs = []
    for s in shots:
        if s.get("individual"):
            jobs.append((f"{s['id']}a1 {s['individual']}", s))
        for c in s.get("companions") or []:
            if c.get("individual"):
                jobs.append((f"{s['id']}a{c.get('instance_id')} {c['individual']}", c))
    if sweep:
        from factory import individuals as indiv
        from factory.whalemodel import REPO_ROOT
        lp = (config.get("whale") or {}).get("individuals")
        lib = indiv.load_library(Path(lp) if Path(lp).is_absolute() else REPO_ROOT / lp)
        poses = [s for s in shots if s.get("individual")]
        step = max(1, len(poses) // sweep)
        for s in poses[::step][:sweep]:
            for k in sorted(lib):
                jobs.append((f"sweep {s['id']} {k}", {**s, "individual": k}))
    return jobs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir"); ap.add_argument("--sweep", type=int, default=0); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--json"); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--old", action="store_true")
    ap.add_argument("--outline", action="store_true")
    ap.add_argument("--force-weld", action="store_true", help="check the welded design on a run made without it (same shots)")
    ap.add_argument("--design", type=int, choices=(13, 15), help="round 15: check this weld design instead of the run's own")
    a = ap.parse_args(argv)
    global DESIGN
    DESIGN = a.design
    run = Path(a.run_dir)
    config = json.loads((run / "run_config.json").read_text())["config"]
    w = config.get("whale") or {}
    if not w.get("blade_templates"):
        print("weld_check: no parametric blade in this run"); return 0
    if not w.get("blade_weld") and not a.old and not a.force_weld and a.design is None:
        from factory import blade_check
        return blade_check.main([str(run)] + (["--sweep", str(a.sweep)] if a.sweep else []) + ["--workers", str(a.workers)] + (["--json", a.json] if a.json else []))
    from factory import weld as wd
    d15 = not a.old and (a.design if a.design is not None else wd.design_of(w.get("blade_weld"))) == 15
    out = {"run": str(run), "design": "round 11 (blade on the back)" if a.old else ("round 15 (implicit surface, gate posing)" if d15 else "round 12 (welded)")}
    if not a.old:
        m, ctx = _model(config, False)
        out["manifold"] = manifold(m)
        if a.outline:
            out["outline"] = outline(m)
    jobs = [(config, l, s, a.old) + ((a.design,) if a.design is not None else ()) for l, s in jobs_for(run, config, a.sweep, a.limit)]
    from multiprocessing import Pool
    with Pool(a.workers) as pool:
        res = pool.map(_job, jobs, chunksize=4)
    errors = [(l, r["error"]) for l, r in res if "error" in r]
    ok = [(l, r) for l, r in res if "error" not in r]
    if a.old:
        keys = ("blade_back_crossings", "blade_self_intersect")
        tot = {k: int(sum(r[k] for _, r in ok)) for k in keys}
        tot["jump_deg_max"] = max((r["jump_deg_max"] for _, r in ok), default=0.0)
        tot["jump_deg_median_of_medians"] = float(np.median([r["jump_deg_median"] for _, r in ok if r["blade_back_crossings"]])) if ok else 0.0
        bad = [(l, r) for l, r in ok if r["blade_back_crossings"] or r["blade_self_intersect"]]
    else:
        keys = ("intersect", "fold", "degenerate", "stretch", "normal_jump")
        tot = {k: int(sum(r[k] for _, r in ok)) for k in keys}
        tot["jump_deg_max_foot_hump_loop_rigbaseline"] = [max((r["_jump_deg"][i] for _, r in ok), default=0.0) for i in range(4)]
        tot["rig_stretch_range"] = [min((r["_rig_stretch"][0] for _, r in ok), default=1.0), max((r["_rig_stretch"][1] for _, r in ok), default=1.0)]
        tot["min_angle_deg"] = min((r["_min_angle"] for _, r in ok), default=0.0)
        tot["stretch_range"] = [min((r["_stretch_min"] for _, r in ok), default=1.0), max((r["_stretch_max"] for _, r in ok), default=1.0)]
        tot["dihedral_max_deg"] = max((r["_dih_max"] for _, r in ok), default=0.0)
        bad = [(l, r) for l, r in ok if any(r[k] for k in keys)]
    out.update({"whales_checked": len(res), "totals": tot, "whales_failing": len(bad), "errors": len(errors),
                "failing": [{"whale": l, **r} for l, r in bad][:5000], "error_list": errors[:20]})
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in out if k not in ("failing", "error_list", "outline")}))
    if "outline" in out:
        print(json.dumps({k: v for k, v in out["outline"].items() if k != "animals"}))
    for l, r in bad[:25]:
        print("  ", l, r)
    for l, e in errors[:5]:
        print("  ERROR", l, e)
    failed = bool(bad or errors) or (not a.old and (out["manifold"]["not_two_faces"] or out["manifold"]["same_direction"]))
    if d15:
        # round 17b: design 15 gates on self-intersections, errors and manifoldness (its folds / slivers / stretch counts
        # are reported above; they are the rig's and the patch's known small faults, not a reason to stop a run)
        failed = bool(errors) or any(r["intersect"] for _, r in bad) or bool(out["manifold"]["not_two_faces"] or out["manifold"]["same_direction"])
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
