"""run06 round 12 (route B): the dorsal fin and its hump welded into the whale's own surface.

Rounds 7-11 laid a separate blade mesh ON the back (sunk rings, a draped hump sheet): two overlapping surfaces with
different normals at the same place, which rendered as pale / dark plates, a sawtooth where the back's triangles poke
through and a lighting step between fin and back. Here the fin is part of the body surface, one mesh with shared
vertices:

  * a fixed region of the back around the fin (the same rig faces for every animal, big enough for the largest fin and
    hump of the library) is cut out; its boundary loop (rig vertices) stays where it is;
  * the generated blade (factory.blade, the animal's template outline) starts at its foot ring at h = H_FOOT x H; there
    are no sunk rings;
  * between the foot ring and the loop lies a refined patch with a fixed topology (rows from the foot ring out to the
    loop, laid out per animal in the elliptic coordinates of the foot's chord: a conformal, fold-free layout that
    shrinks with small fins and stretches with long ones). Its vertices sit on the back's own surface (barycentric in
    the cut-out rig triangles) and are lifted along the back's smooth normal by the hump field D: the template's
    outline below the foot (the midline ramp ahead of the fin and behind it) and a concave flank around the foot whose
    slope at the foot matches the blade's sides, falling to zero with zero slope before the loop;
  * posing: patch vertices follow the posed rig triangles they lie in (their skin is the back's), the blade above the
    foot is rigid (round 11's transform at the fin base) and its lowest rings blend into the welded foot.

The importer recomputes normals and tangents; shared vertices give smooth shading across the foot and the hump's edge.
UV seams (the back's midline seam, blade charts against back UVs at the foot ring) are split at OBJ level only.
"""
from __future__ import annotations

import numpy as np

from factory import blade as bl

H_FOOT = 0.30                  # the blade's foot ring (fraction of H); below it the hump is the back's own surface
NL_W = 29                      # blade levels H_FOOT..1 (the round-9 spacing of 0.025 H)
LEVELS_W = np.linspace(H_FOOT, 1.0, NL_W)
SKIRT_R = 0.36                 # lateral reach of the hump's flank (x H)
CREST = 0.12                   # rounding of the midline ramp's crest (x H)
SLOPE_MAX = 8.0                # the blade's side slope at its foot counts up to this (near-vertical knife edges)
PHONG_T0 = 0.95                # the base's Phong correction fades out over the last rows (the loop lies on the rig's faces)
BASE_SMOOTH_IT = 80             # smoothing passes of the base under the patch
SMAX_K = 16.0                  # smooth maximum of the flank around the foot and the midline ramp
DORSAL_D_MIN = 0.03             # patch points lifted by more than this fraction of the foot height are dorsal (label)
DORSAL_SOFT_BAND = 4.0          # round 13: the soft dorsal label is linear in the lift over +- this x the threshold
UV_JUMP = 0.02                 # a patch face whose corners' back UVs differ by more is across the back's UV seam
FADE = 0.25                    # blade levels H_FOOT .. H_FOOT + FADE blend from the welded foot to the rigid fin
FOCUS = 0.80                   # foci of the elliptic layout at this fraction of the foot's half chord
N_ROWS = 80                    # patch rows: 0 = the foot ring, N_ROWS = the outer row
MARGIN_CM = 12.0               # region beyond every animal's hump support
SE_N = 3.0                     # the region's super-ellipse exponent (rounder corners stay off the steep flanks)
SNAP = 0.03                    # a cut nearer than this (edge fraction) to a rig vertex moves onto it
ROWJ_CLEAR_CM = 5.0            # the patch's outer row this far inside the rig loop
N_ROWJ = 160                   # points on the patch's outer row
LM_MIN = 0.50                  # the front / back base landmark at least this far (x R) outside the foot
DELTA0 = 0.12                  # the patch's first row step at the foot (x R) ... (round 13: 0.075 -> 0.12)
DELTA_MIN_CM = 2.0             # ... at least this (round 13: 1.2 -> 2.0; rows 0.3-1 cm apart crossed in bent poses)
BLADE_BINS = 0                 # round 13: 1 = blade carried per 15 cm bin (round 12; the largest blades span 2 m of back and
                               # broke as one rigid piece in bent poses), 0 = one rigid fit at its base
RIGID_D0 = 0.0                 # round 13 (tried, off): patch points within RIGID_D0 x R of the foot moving with the blade's
RIGID_D1 = 0.0                 # rigid fit, blending to the back's own motion at RIGID_D1 x R (folded in the blend zone)
RAISE_MAX = 0.2               # round 13: largest raise of a blade whose nominal foot lies under the animal's back (x H)
JUNCTION_H = 0.15             # round 13: the junction band smoothed at rest: blade rings up to this (x H) above the foot ...
JUNCTION_R = 0.6               # ... and patch points within this (x R) of it
JUNCTION_IT = 25               # Taubin passes (lambda / mu pairs)
DELTA_DECAY = 1.0              # round 13: the foot's offset to the nominal blade ring fades out over this x R


def design_of(value) -> int:
    """The weld design selected by the config's whale.blade_weld: 15 = round 15 (factory.weld15, implicit rest surface,
    'foot' posing); anything else that is true (true, 13) = round 13 (this module + WhaleModel's own weld code)."""
    if isinstance(value, bool) or value is None:
        return 13
    try:
        return 15 if int(value) == 15 else 13
    except (TypeError, ValueError):
        return 13


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


# ---- elliptic coordinates: z = c + f cosh(mu + i nu), z = a + i y --------------------------------------------------
def to_ell(P2: np.ndarray, c: float, f: float):
    z = ((P2[:, 0] - c) + 1j * P2[:, 1]) / f
    w = np.arccosh(z.astype(complex))
    mu, nu = w.real.copy(), w.imag.copy()
    neg = mu < 0
    mu[neg] = -mu[neg]; nu[neg] = -nu[neg]
    nu = np.mod(nu, 2 * np.pi)
    # arccosh's branch cut: make sure the point maps back
    back = f * np.cosh(mu + 1j * nu)
    bad = np.abs(back - (z * f)) > 1e-6 * max(f, 1.0)
    nu[bad] = np.mod(2 * np.pi - nu[bad], 2 * np.pi)
    return mu, nu


def from_ell(mu, nu, c: float, f: float) -> np.ndarray:
    z = c + f * np.cosh(mu + 1j * nu)
    return np.stack([z.real, z.imag], 1)


def periodic_interp(x, xp, fp, period=2 * np.pi):
    """Linear interpolation of a periodic function sampled at increasing xp (within one period)."""
    xp = np.asarray(xp, float); fp = np.asarray(fp, float)
    xe = np.concatenate([xp[-1:] - period, xp, xp[:1] + period]); fe = np.concatenate([fp[-1:], fp, fp[:1]])
    return np.interp(np.mod(x, period), xe, fe)


def periodic_map(x, xp, yp, period=2 * np.pi):
    """Monotone periodic map sending xp -> yp (both increasing within one period, x in [0, period))."""
    xp = np.asarray(xp, float); yp = np.asarray(yp, float)
    xe = np.concatenate([xp[-1:] - period, xp, xp[:1] + period]); ye = np.concatenate([yp[-1:] - period, yp, yp[:1] + period])
    return np.interp(np.mod(x, period), xe, ye)


def zipper(ida, na, idb, nb):
    """Triangles between two closed rows (ids ida with canonical angles na, idb with nb; both increasing from ~0),
    row a inside row b; counter-clockwise in (t, nu)."""
    F = []
    i = j = 0
    la, lb = len(ida), len(idb)
    # start pair: both rows start at the smallest angle; walk both around once
    while i < la or j < lb:
        a0, a1 = ida[i % la], ida[(i + 1) % la]
        b0, b1 = idb[j % lb], idb[(j + 1) % lb]
        na1 = na[(i + 1) % la] + (2 * np.pi if i + 1 >= la else 0.0)
        nb1 = nb[(j + 1) % lb] + (2 * np.pi if j + 1 >= lb else 0.0)
        if j >= lb or (i < la and na1 <= nb1):
            F.append((a0, a1, b0)); i += 1
        else:
            F.append((a0, b1, b0)); j += 1
    return F


class Weld:
    """Model-level weld data (fixed for every animal): the cut region and its boundary loop, the base surface and the
    canonical patch topology. Per-animal layouts come from layout().

    The region is a super-ellipse in the chart (asymmetric along the back), the smallest that holds every animal's hump
    support + MARGIN_CM. Rig faces inside it are removed; faces it crosses are cut along it (new vertices on the rig's
    edges, posed and textured by linear interpolation of the edge's ends, or the end itself when the crossing is near
    it), so the loop is a smooth convex curve: the patch's outer row. Extended index space: rig vertices 0..n_rig-1, cut
    vertices n_rig..n_rig+n_cut-1."""

    def __init__(self, model, fin: dict):
        R = np.asarray(model.rest[: model.n_rig], float)
        self.n_rig = model.n_rig
        B0, e_a, e_n = model._blade_frame
        self.B0, self.e_a, self.e_n = np.asarray(B0, float), np.asarray(e_a, float), np.asarray(e_n, float)
        _, inv = np.unique(np.round(R, 3), axis=0, return_inverse=True)
        self.weld_id = inv.ravel()                    # position-welded ids of the rig (UV seams duplicate vertices)

    # ---- chart --------------------------------------------------------------------------------------------------
    def chart(self, P: np.ndarray) -> np.ndarray:
        rel = np.asarray(P, float) - self.B0
        return np.stack([rel @ self.e_a, rel[:, 1]], 1)

    def phi(self, P2: np.ndarray) -> np.ndarray:
        ac, har, haf, hy = self.shape
        dx = P2[:, 0] - ac
        ha = np.where(dx < 0, har, haf)
        return np.abs(dx / ha) ** SE_N + np.abs(P2[:, 1] / hy) ** SE_N - 1.0

    @staticmethod
    def fit_shape(pts: np.ndarray):
        """Smallest-area asymmetric super-ellipse (centre ac on the midline, rear / front half axes, half width) holding
        the points."""
        best = None
        y = np.abs(pts[:, 1]); ymax = float(y.max())
        for ac in np.linspace(np.percentile(pts[:, 0], 20), np.percentile(pts[:, 0], 80), 41):
            dx = pts[:, 0] - ac
            for hy in np.linspace(1.02 * ymax, 2.0 * ymax, 60):
                k = (1.0 - (y / hy) ** SE_N) ** (1.0 / SE_N)
                need = np.abs(dx) / k
                har = float(need[dx < 0].max()) if (dx < 0).any() else 1.0
                haf = float(need[dx >= 0].max()) if (dx >= 0).any() else 1.0
                area = (har + haf) * hy
                if best is None or area < best[0]:
                    best = (area, (float(ac), har, haf, float(hy)))
        return best[1]

    def build(self, model, fin: dict, need_pts: np.ndarray) -> None:
        R = np.asarray(model.rest[: model.n_rig], float)
        F = np.asarray(model.faces[: model.n_rig_faces])
        w = np.asarray(fin["w"], float)
        C = self.chart(R)
        self.shape = self.fit_shape(need_pts)
        ph = self.phi(C)
        # the top surface: faces connected to the one nearest the back base, all vertices within the curve grown 1.5 x
        Fw = self.weld_id[F]
        tri3 = R[F]; fn = np.cross(tri3[:, 1] - tri3[:, 0], tri3[:, 2] - tri3[:, 0])
        fz = (fn @ self.e_n) / (np.linalg.norm(fn, axis=1) + 1e-12)
        i0 = int(np.argmin(np.linalg.norm(tri3.mean(1) - self.B0, axis=1) + 1e3 * (ph[F].max(1) > 0)))
        up = np.sign(fz[i0]) * fz > 0.15                                # facing up like the back at the fin (not the belly)
        cand = np.nonzero((ph[F] < 1.5 ** SE_N - 1.0).all(1) & up)[0]   # (about 1.5 x the half axes)
        edge_faces = {}
        for fi in cand:
            a, b, c = Fw[fi]
            for e in ((a, b), (b, c), (c, a)):
                edge_faces.setdefault((min(e), max(e)), []).append(int(fi))
        cent = R[F[cand]].mean(1)
        seed = int(cand[np.argmin(np.linalg.norm(cent - self.B0, axis=1))])
        top_faces = np.array(sorted(_flood(seed, Fw, cand, edge_faces)))
        top = np.zeros(len(R), bool); top[np.unique(F[top_faces])] = True
        phv = np.where(top, ph, 1.0)
        # snap crossings near an edge's end onto that end (no slivers), on welded ids
        E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
        for _ in range(4):
            pa, pb = phv[E[:, 0]], phv[E[:, 1]]
            cross = (pa * pb < 0)
            s_ = np.where(cross, pa / np.where(cross, pa - pb, 1.0), 0.5)
            snap = np.zeros(len(R), bool)
            snap[E[cross & (s_ < SNAP), 0]] = True; snap[E[cross & (s_ > 1 - SNAP), 1]] = True
            if not snap.any():
                break
            wsn = np.unique(self.weld_id[snap])
            phv[np.isin(self.weld_id, wsn)] = 0.0
        self.phv = phv
        # cut vertices on crossing edges (per rig edge with its own copies)
        cuts = {}
        cut_list = []

        def cut(u, v):
            key = (min(u, v), max(u, v))
            if key not in cuts:
                a_, b_ = key
                # the exact crossing of the curve along the edge (bisection; phi is not linear along it)
                lo, hi = 0.0, 1.0
                sa = np.sign(phv[a_])
                for _ in range(40):
                    mid = 0.5 * (lo + hi)
                    pm = C[a_] + (C[b_] - C[a_]) * mid
                    val = self.phi(pm[None])[0] if (top[a_] and top[b_]) else phv[a_] + (phv[b_] - phv[a_]) * mid
                    if np.sign(val) == sa:
                        lo = mid
                    else:
                        hi = mid
                cuts[key] = self.n_rig + len(cut_list); cut_list.append((a_, b_, float(0.5 * (lo + hi))))
            return cuts[key]
        kept, base = [], []
        for fi in range(len(F)):
            tri = [int(x) for x in F[fi]]
            p = [phv[x] for x in tri]
            if all(q <= 0 for q in p):
                base.append(tri); continue
            if all(q >= 0 for q in p):
                kept.append(tri); continue
            ins, out = [], []
            for k in range(3):
                u, v = tri[k], tri[(k + 1) % 3]
                pu, pv = phv[u], phv[v]
                if pu <= 0:
                    ins.append(u)
                if pu >= 0:
                    out.append(u)
                if pu * pv < 0:
                    c_ = cut(u, v); ins.append(c_); out.append(c_)
            for poly, dst in ((ins, base), (out, kept)):
                for k in range(1, len(poly) - 1):
                    dst.append([poly[0], poly[k], poly[k + 1]])
        self.n_cut = len(cut_list)
        self.cut_uvs = np.array(cut_list, dtype=float).reshape(-1, 3)
        self.cut_u = self.cut_uvs[:, 0].astype(int); self.cut_v = self.cut_uvs[:, 1].astype(int); self.cut_s = self.cut_uvs[:, 2]
        self.base_faces = np.asarray(base, np.int64)
        self.kept_faces = np.asarray(kept, np.int64)
        # extended welded ids (cut vertices: by their welded edge)
        wcut = {}
        ext_w = list(self.weld_id)
        nW = int(self.weld_id.max()) + 1
        for k in range(self.n_cut):
            key = (min(self.weld_id[self.cut_u[k]], self.weld_id[self.cut_v[k]]), max(self.weld_id[self.cut_u[k]], self.weld_id[self.cut_v[k]]))
            if key not in wcut:
                wcut[key] = nW + len(wcut)
            ext_w.append(wcut[key])
        self.ext_weld = np.asarray(ext_w)
        Rext = self.ext(R)
        Cx = self.chart(Rext)
        self.R_rest = Rext
        loop_w, copies = self._loop(Cx)
        self.loop_weld, self.loop_copies = loop_w, copies
        self.loop_rep = np.array([next(iter(copies[k].values())) for k in loop_w])
        self.loop2d = Cx[self.loop_rep]
        # seam copies to average (base vertices and loop copies, rig only)
        bv = np.unique(np.concatenate([self.base_faces[self.base_faces < self.n_rig].ravel(),
                                       [v for k in loop_w for v in copies[k].values() if v < self.n_rig]]))
        wid = self.weld_id
        allm = np.nonzero(np.isin(wid, wid[bv]))[0]
        u, gid, cnt = np.unique(wid[allm], return_inverse=True, return_counts=True)
        multi = cnt[gid] > 1
        members = allm[multi]
        u2, gid2 = np.unique(wid[members], return_inverse=True)
        self._seam_groups = (members, gid2.ravel(), len(u2))
        # base surface fill (old asset fin vertices) and UVs of the extended vertices
        fill = np.unique(self.base_faces[self.base_faces < self.n_rig])
        fill = fill[w[fill] > 0.01]
        self.fill = fill
        self.fill_op = self._fill_operator(fill, self.base_faces, Rext)
        self._phong_cuts = False         # (cut vertices stay on the rig's flat edges, like the snapped rig vertices)
        self.R_rest = self.ext(R)
        uv = np.asarray(model.rig.uv, float)
        self.uv_ext = np.concatenate([uv, uv[self.cut_u] * (1 - self.cut_s[:, None]) + uv[self.cut_v] * self.cut_s[:, None]])

    def _top_set(self, top_faces):
        ts = self.__dict__.get("_ts")
        if ts is None:
            ts = self._ts = set(int(x) for x in top_faces)
        return ts

    def ext(self, Rig: np.ndarray) -> np.ndarray:
        """Extended positions: the rig's (old fin filled) and the cut vertices: on the rig's edge, lifted onto the edge's
        Phong curve (alpha 0.75, along the normal) like every patch point, so the patch's outer row lies on the same
        smooth surface as the rows inside it (a flat outer band would show the rig's long edges as grooves)."""
        Rb = self.base_positions(Rig)
        s_ = self.cut_s[:, None]
        cutp = Rb[self.cut_u] * (1 - s_) + Rb[self.cut_v] * s_
        X = np.concatenate([Rb, cutp])
        if self.__dict__.get("_phong_cuts", False) and len(cutp):
            N = self.base_normals(X)
            nu, nv = N[self.cut_u], N[self.cut_v]
            pu = cutp - ((cutp - Rb[self.cut_u]) * nu).sum(1, keepdims=True) * nu
            pv = cutp - ((cutp - Rb[self.cut_v]) * nv).sum(1, keepdims=True) * nv
            ps = pu * (1 - s_) + pv * s_
            n = nu * (1 - s_) + nv * s_
            n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
            X[self.n_rig:] = cutp + 0.75 * ((ps - cutp) * n).sum(1, keepdims=True) * n
        return X

    def _loop(self, Cx):
        """The boundary between kept and base faces (welded), counter-clockwise in the chart, starting at the point with
        the smallest polar angle about the region's centre; per welded id the extended copies by side (0: y >= 0 face)."""
        Wx = self.ext_weld
        cnt_b, cnt_k = {}, {}
        for Fs, cnt in ((self.base_faces, cnt_b), (self.kept_faces, cnt_k)):
            for a, b, c in Wx[Fs]:
                for e in ((a, b), (b, c), (c, a)):
                    k = (min(e), max(e)); cnt[k] = cnt.get(k, 0) + 1
        bedges = [e for e in cnt_b if e in cnt_k and cnt_b[e] == 1]
        adj = {}
        for a, b in bedges:
            adj.setdefault(a, []).append(b); adj.setdefault(b, []).append(a)
        if any(len(v) != 2 for v in adj.values()):
            raise RuntimeError(f"weld: the region's boundary is not a simple loop ({sum(len(v) != 2 for v in adj.values())} pinch vertices)")
        start = next(iter(adj)); loop = [start]; prev = None
        while True:
            nxt = [v for v in adj[loop[-1]] if v != prev][0]
            if nxt == start:
                break
            prev = loop[-1]; loop.append(nxt)
        if len(loop) != len(adj):
            raise RuntimeError("weld: the region's boundary has more than one loop")
        rep_any = {}
        for v in range(len(Wx)):
            rep_any.setdefault(int(Wx[v]), v)
        P = Cx[[rep_any[k] for k in loop]]
        area = 0.5 * np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1])
        if area < 0:
            loop = loop[::-1]
        P = Cx[[rep_any[k] for k in loop]]
        th = np.mod(np.arctan2(P[:, 1], P[:, 0] - self.shape[0]), 2 * np.pi)
        k0 = int(np.argmin(th)); loop = loop[k0:] + loop[:k0]
        loopset = set(loop)
        copies = {k: {} for k in loop}
        for fv in self.kept_faces:
            cy = Cx[fv, 1].mean()
            for v in fv:
                k = int(Wx[v])
                if k in loopset:
                    copies[k].setdefault(0 if cy >= 0 else 1, int(v))
        for k in loop:
            if not copies[k]:
                copies[k][0] = rep_any[k]
        return [int(k) for k in loop], copies

    def _fill_operator(self, fill, faces, Rx):
        """X_fill = M @ X_known: harmonic (uniform Laplacian) fill of the old fin's vertices from their neighbours in the
        base faces (on welded positions)."""
        if len(fill) == 0:
            return None
        wid = self.ext_weld
        fw = np.unique(wid[fill])
        verts = np.unique(faces)
        vw = np.unique(wid[verts])
        known_w = np.setdiff1d(vw, fw)
        pos = {int(k): i for i, k in enumerate(fw)}; kpos = {int(k): i for i, k in enumerate(known_w)}
        n, m = len(fw), len(known_w)
        A = np.zeros((n, n)); Bk = np.zeros((n, m))
        nb = {}
        for a, b, c in wid[faces]:
            for u, v in ((a, b), (b, c), (c, a)):
                nb.setdefault(int(u), set()).add(int(v)); nb.setdefault(int(v), set()).add(int(u))
        for k, i in pos.items():
            A[i, i] = len(nb[k])
            for v in nb[k]:
                if v in pos:
                    A[i, pos[v]] -= 1.0
                else:
                    Bk[i, kpos[v]] += 1.0
        Mw = np.linalg.solve(A, Bk)
        rep = {}
        for v in verts:
            rep.setdefault(int(wid[v]), int(v))
        known_v = np.array([rep[int(k)] for k in known_w])
        used = np.nonzero(np.abs(Mw).max(0) > 1e-12)[0]
        if (known_v[used] >= self.n_rig).any():
            raise RuntimeError("weld: the old fin's fill reaches the cut")
        rows = np.array([pos[int(wid[v])] for v in fill])
        return (fill, rows, Mw[:, used], known_v[used])

    def base_positions(self, Rig: np.ndarray) -> np.ndarray:
        """Rig positions with the old fin's vertices replaced by the harmonic fill and the rig's UV-seam copies in the
        region (and on its loop) averaged: the skin moves them apart by up to ~4 cm, a crack the patch would copy."""
        out = np.array(Rig, float, copy=True)
        sg = self.__dict__.get("_seam_groups")
        if sg is not None:
            members, gid, ng = sg
            acc = np.zeros((ng, 3)); np.add.at(acc, gid, out[members])
            cnt = np.bincount(gid, minlength=ng)[:, None]
            out[members] = (acc / cnt)[gid]
        op = self.__dict__.get("fill_op")
        if op is not None:
            fill, rows, Mw, known_v = op
            X = Mw @ out[known_v]
            out[fill] = X[rows]
        return out

    def base_normals(self, Rx: np.ndarray) -> np.ndarray:
        """Area-weighted vertex normals of the base faces on the extended vertices (welded by position); zero off the
        base. Oriented like the rig's faces (outward)."""
        Fb = self.base_faces
        wid = self.ext_weld
        tri = Rx[Fb]
        fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        acc = np.zeros((int(wid.max()) + 1, 3))
        for k in range(3):
            np.add.at(acc, wid[Fb[:, k]], fn)
        N = acc[wid]
        return N / (np.linalg.norm(N, axis=1, keepdims=True) + 1e-12)


def _zipper_geom(PA, PB, ta, tb):
    """Strip between an inner closed row PA (angles ta) and an outer one PB (tb), both counter-clockwise from angle ~0:
    advancing front that keeps every triangle the same orientation as the rows' faces (clockwise in the chart, like the
    rig's) and, when both moves are valid, takes the shorter diagonal. Outer entries are -1 - k."""
    la, lb = len(PA), len(PB)
    def area(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    F = []; i = j = 0
    while i < la or j < lb:
        a0, a1 = PA[i % la], PA[(i + 1) % la]; b0, b1 = PB[j % lb], PB[(j + 1) % lb]
        # canonical zipper triangles (a0, a1, b0) / (a0, b1, b0) are counter-clockwise in (nu, t), clockwise in the chart
        okA = i < la and area(a0, a1, b0) < 0
        okB = j < lb and area(a0, b1, b0) < 0
        if okA and okB:
            useA = np.linalg.norm(a1 - b0) <= np.linalg.norm(a0 - b1)
        elif okA or okB:
            useA = okA
        else:
            useA = i < la and (j >= lb or ta[(i + 1) % la] + (2 * np.pi if i + 1 >= la else 0) <= tb[(j + 1) % lb] + (2 * np.pi if j + 1 >= lb else 0))
        if useA:
            F.append((i % la, (i + 1) % la, -1 - (j % lb))); i += 1
        else:
            F.append((i % la, -1 - ((j + 1) % lb), -1 - (j % lb))); j += 1
    return F


def _seg_dist(P, A, B):
    """Distance of points P (n, 2) to the closest of segments A[i] -> B[i]."""
    AB = B - A
    t = np.clip((((P[:, None, :] - A[None]) * AB[None]).sum(-1)) / np.maximum((AB * AB).sum(-1), 1e-12)[None], 0, 1)
    Q = A[None] + t[..., None] * AB[None]
    return np.sqrt(((P[:, None, :] - Q) ** 2).sum(-1)).min(1)


def _flood(seed, Fw, cand, edge_faces):
    region = {int(seed)}; stack = [int(seed)]
    while stack:
        fi = stack.pop()
        a, b, c = Fw[fi]
        for e in ((a, b), (b, c), (c, a)):
            for g in edge_faces.get((min(e), max(e)), []):
                if g not in region:
                    region.add(int(g)); stack.append(int(g))
    return region


# ---- per-animal inputs ---------------------------------------------------------------------------------------------
def animal_inputs(weld: "Weld", local: dict, frame, H: float) -> dict:
    """The animal's foot ring and hump core in the chart. local = factory.blade.make(..., levels=LEVELS_W) (cm), frame =
    (B, along, up) of the animal's blade, H its height (cm). The 2D layout ignores heights: a blade-local point (a, y, h)
    sits at chart(B + a along) + (0, y)."""
    B, along, _ = frame
    B = np.asarray(B, float); along = np.asarray(along, float)
    V = np.asarray(local["vertices"], float)
    R2 = 2 * bl.NC
    ring = V[:R2]
    # the nominal foot ring (blade frame, h = H_FOOT H) seen along the chart's normal: lifting the back there along that
    # normal to the foot height lands exactly on the blade's ring
    ring2d = weld.chart(bl.place(ring, B, along, np.asarray(frame[2], float)))
    # the blade's side slope at its foot, per column: height gained per cm inwards from ring 0 to ring 1 (outward normal
    # of the foot section in the (a, y) plane); the hump's flank starts with that slope
    r0, r1 = V[:R2], V[R2:2 * R2]
    tang = np.roll(r0[:, :2], -1, axis=0) - np.roll(r0[:, :2], 1, axis=0)
    nrm = np.stack([tang[:, 1], -tang[:, 0]], 1)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    cen = r0[:, :2].mean(0)
    flip = ((r0[:, :2] - cen) * nrm).sum(1) < 0
    nrm[flip] *= -1
    d_out = ((r1[:, :2] - r0[:, :2]) * nrm).sum(1)
    slope = (r1[:, 2] - r0[:, 2]) / np.maximum(-d_out, (r1[:, 2] - r0[:, 2]) / SLOPE_MAX)
    out = local["outline"]
    L, te_s, le_s = np.asarray(out["levels"]), np.asarray(out["te"]), np.asarray(out["le"])
    hh = np.linspace(0.0, H_FOOT, 61)
    te_h, le_h = np.interp(hh, L, te_s), np.interp(hh, L, le_s)
    a_lo, a_hi = float(te_h.min()), float(le_h.max())
    a_ft, a_fl = float(te_h[-1]), float(le_h[-1])          # the foot's own chord at H_FOOT (units of H)
    na = max(8, int(np.ceil((a_hi - a_lo) / 0.01)))
    a_s = np.linspace(a_lo, a_hi, na + 1)
    inside = (te_h[None, :] <= a_s[:, None] + 1e-9) & (a_s[:, None] <= le_h[None, :] + 1e-9)
    hm = np.where(inside, hh[None, :], -1.0).max(1)
    hm = np.clip(hm, 0.0, H_FOOT)
    keep = ((a_s < a_ft) | (a_s > a_fl)) & (hm > 0.0)        # the ramps ahead and behind the foot (not under it)
    a_c, c_c = a_s[keep], hm[keep]
    core2d = weld.chart(B[None, :] + (a_c[:, None] * H) * along[None, :])
    Rcm = SKIRT_R * H
    # landmarks: where the hump's midline silhouette ends ahead and behind (template base), at least LM_MIN R off the foot
    fb_a = max(a_hi * H, a_fl * H + LM_MIN * Rcm) if a_hi > a_fl + 1e-6 else a_fl * H + LM_MIN * Rcm
    bb_a = min(a_lo * H, a_ft * H - LM_MIN * Rcm) if a_lo < a_ft - 1e-6 else a_ft * H - LM_MIN * Rcm
    fb2d = weld.chart((B + fb_a * along)[None, :])[0]; bb2d = weld.chart((B + bb_a * along)[None, :])[0]
    fb2d[1] = 0.0; bb2d[1] = 0.0
    core_all = np.concatenate([_densify_ring(ring2d, 2), core2d, fb2d[None], bb2d[None]])
    th = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    needs = (core_all[:, None, :] + (Rcm + MARGIN_CM) * np.stack([np.cos(th), np.sin(th)], 1)[None]).reshape(-1, 2)
    return {"H": H, "ring2d": ring2d, "slope": slope, "core2d": core2d, "core_c": c_c * H, "foot_c": H_FOOT * H, "R": Rcm,
            "fb2d": fb2d, "bb2d": bb2d, "needs": needs, "frame": (B, along, np.asarray(frame[2], float)),
            "c": 0.5 * (ring2d[[0, R2 - 1], 0].mean() + ring2d[[bl.NC - 1, bl.NC], 0].mean()),
            "f": FOCUS * 0.5 * (ring2d[[bl.NC - 1, bl.NC], 0].mean() - ring2d[[0, R2 - 1], 0].mean())}


def _densify_ring(P: np.ndarray, k: int = 4) -> np.ndarray:
    Q = np.roll(P, -1, axis=0)
    t = np.arange(k)[None, :, None] / k
    return (P[:, None, :] * (1 - t) + Q[:, None, :] * t).reshape(-1, 2)


def flank(u: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Flank profile on u = d / R in [0, 1]: 1 at the foot with slope -k (in units of lift / R), 0 with zero slope at
    u = 1. Cubic Hermite for k <= 3 (monotone there), (1 - u) ** k above (the two agree at k = 3)."""
    u = np.clip(u, 0.0, 1.0)
    # round 13: C2 at u = 1 (zero slope AND zero curvature where the flank meets the back: the round-12 cubic left a
    # curvature step there, a visible rim round the hump): (1 - u)^3 (1 + (3 - k) u + c u^2), c = 6 (1 - k / 3)^2
    kk = np.minimum(k, 3.0)
    quint = (1.0 - u) ** 3 * (1.0 + (3.0 - kk) * u + 6.0 * (1.0 - kk / 3.0) ** 2 * u * u)
    powr = (1.0 - u) ** np.maximum(k, 3.0)
    return np.where(k <= 3.0, quint, powr)


def hump_lift(inp: dict, P2: np.ndarray, base_h: np.ndarray, core_ref: np.ndarray, foot_base_h=None) -> np.ndarray:
    """The lift (cm, height in the animal's blade frame) of chart points above the back (heights base_h).
    Core: the foot ring at its nominal height H_FOOT H, and the midline ramps ahead / behind at the template height hm
    over the base line blended onto the back's midline (core_ref: the back's height there) as hm falls (the round-8
    drape): Habs = hm + ref (1 - hm / H_FOOT). Each core point lifts its neighbours to (Habs - base_h) flank(d / R): a
    flank that starts with the blade's own side slope at the nearest foot column (no crease at the foot) and meets the
    back with zero slope at d = R. Families combine by a smooth maximum."""
    R = inp["R"]
    hf = H_FOOT * inp["H"] + float(inp.get("foot_up", 0.0))     # round 13: the blade raised where its foot was buried
    ring = _densify_ring(inp["ring2d"], 6)
    sl = inp["slope"]
    sld = (sl[:, None] * (1 - np.arange(6) / 6)[None, :] + np.roll(sl, -1)[:, None] * (np.arange(6) / 6)[None, :]).ravel()
    d2 = ((P2[:, None, :] - ring[None, :, :]) ** 2).sum(-1)
    j = np.argmin(d2, 1); d = np.sqrt(d2[np.arange(len(P2)), j])
    L0 = np.clip(hf - base_h, 0.0, None)
    # slope matched at the foot: flank'(0) = -k L0 / R = -slope  ->  k = slope R / L0 (L0 of the nearest foot point)
    L0f = np.clip(hf - (foot_base_h if foot_base_h is not None else base_h), 1e-3, None)
    kf = sld[j] * R / L0f[np.clip(j // 6, 0, len(L0f) - 1)] if foot_base_h is not None else sld[j] * R / np.maximum(L0, 1e-3)
    Lf = L0 * flank(d / R, kf)
    Lm = np.zeros(len(P2))
    if len(inp["core2d"]):
        eps = CREST * inp["H"]
        Habs = inp["core_c"] + core_ref * (1.0 - inp["core_c"] / hf)
        le_cols = np.arange(bl.NC - 4, bl.NC + 4)
        te_cols = np.concatenate([np.arange(0, 4), np.arange(2 * bl.NC - 4, 2 * bl.NC)])
        ring_a = inp["ring2d"][:, 0]
        ahead = inp["core2d"][:, 0] > 0.5 * (ring_a.max() + ring_a.min())
        s_core = np.where(ahead, float(np.median(sl[le_cols])), float(np.median(sl[te_cols])))
        for s0 in range(0, len(P2), 2048):
            q = P2[s0:s0 + 2048]
            dd = np.sqrt(((q[:, None, :] - inp["core2d"][None, :, :]) ** 2).sum(-1) + eps * eps) - eps
            Lc = np.clip(Habs[None, :] - base_h[s0:s0 + 2048, None], 0.0, None)
            kc = s_core[None, :] * R / np.maximum(Lc, 1e-3)
            val = Lc * flank(dd / R, kc)
            Lm[s0:s0 + 2048] = val.max(1)
    out = (Lf ** SMAX_K + Lm ** SMAX_K) ** (1.0 / SMAX_K)
    if foot_base_h is not None and len(P2) >= len(foot_base_h):
        n = len(foot_base_h)
        out[:n] = Lf[:n]               # the foot ring itself: exactly its nominal height (no smooth-max overshoot)
    return out


# ---- canonical patch topology and per-animal layout ----------------------------------------------------------------
def _round_even(x):
    return int(2 * round(x / 2.0))


def _quantile_from_knots(u, uk, vk):
    """Monotone periodic quantile: knots (uk in [0, 1), vk in [0, 2 pi)), u in [0, 1]."""
    ue = np.concatenate([uk[-1:] - 1.0, uk, uk[:1] + 1.0]); ve = np.concatenate([vk[-1:] - 2 * np.pi, vk, vk[:1] + 2 * np.pi])
    return np.interp(u, ue, ve)


def _loop_nu(inp: dict, loop2d: np.ndarray) -> np.ndarray:
    """Elliptic angles of the loop's points (in loop order), unwrapped, starting within (-pi, pi]; must increase."""
    _, nu = ell_of(inp, loop2d)
    nu = np.unwrap(nu)
    nu = nu - 2 * np.pi * np.round((nu[0]) / (2 * np.pi))
    if np.any(np.diff(nu) <= 0) or nu[-1] - nu[0] >= 2 * np.pi:
        raise RuntimeError("weld: the loop is not star-shaped in this animal's elliptic coordinates")
    return nu


def foot_order() -> np.ndarray:
    """Blade ring-0 indices in increasing elliptic angle nu (0 = ahead): right side LE -> TE, then left side TE -> LE."""
    return np.concatenate([np.arange(bl.NC - 1, -1, -1), np.arange(2 * bl.NC - 1, bl.NC - 1, -1)])


def ell_of(inp: dict, P2: np.ndarray):
    return to_ell(P2, inp["c"], inp["f"])


def build_canonical(weld: "Weld", ref: dict) -> None:
    """Rows of canonical angles (t_j = j / N_ROWS) from the reference animal's foot ring (row 0) to the loop (last row),
    the patch faces in canonical ids (foot ring 0..2NC-1 = blade ring-0 index, interior 2NC.., loop after), and the
    rows' landmark vertices (front / back base at nu = 0 / pi)."""
    R2 = 2 * bl.NC
    order = foot_order()
    _, nu_f = ell_of(ref, ref["ring2d"][order])
    nu_f = np.sort(np.mod(nu_f, 2 * np.pi))
    nu_f = 0.5 * (nu_f + (2 * np.pi - nu_f[::-1]))           # exactly symmetric
    nu_l = _loop_nu(ref, weld.loop2d)
    nL = len(nu_l)
    u_f = (np.arange(R2) + 0.5) / R2
    # refined distribution (denser near the midline ahead and behind: the ramps)
    g = np.linspace(0, 2 * np.pi, 4001)
    dens = 1.0 + 2.5 * (np.exp(-(g / 0.35) ** 2) + np.exp(-((g - np.pi) / 0.35) ** 2) + np.exp(-((g - 2 * np.pi) / 0.35) ** 2))
    cdf = np.concatenate([[0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(g))]); cdf /= cdf[-1]
    J = N_ROWS
    n_mid = 132
    rows_nu, rows_ids = [], []
    nid = R2
    for j in range(J + 1):
        t = j / J
        if j == 0:
            rows_nu.append(nu_f.copy()); rows_ids.append(order.copy()); continue
        if j == J:
            rows_nu.append(nu_l.copy()); rows_ids.append(None); continue
        if t <= 0.6:
            n = _round_even(R2 + (n_mid - R2) * smoothstep(t / 0.3))
        else:
            n = _round_even(n_mid + (nL - n_mid) * smoothstep((t - 0.6) / 0.4))
        n = max(n, R2)
        if j == J - 1:
            n = nL                     # round 13: the last inner row has the loop's own points (quads, no slivers)
        u = np.arange(n) / n
        q0 = _quantile_from_knots(u, u_f, nu_f); qr = np.interp(u, cdf, g)
        # round 13: the outer rows tend to the loop's own angles (round 12: uniform angles, slivers at the cut)
        lm = np.sort(np.mod(nu_l, 2 * np.pi))
        qL = _quantile_from_knots(u, np.arange(nL) / nL, lm)
        if t <= 0.3:
            lam = smoothstep(t / 0.3); q = (1 - lam) * q0 + lam * qr
        elif t <= 0.7:
            q = qr
        else:
            lam = smoothstep((t - 0.7) / 0.3); q = (1 - lam) * qr + lam * qL
        if j < J - 1:
            q[0] = 0.0; q[n // 2] = np.pi
        q = np.sort(np.mod(q, 2 * np.pi))
        rows_nu.append(q); rows_ids.append(np.arange(nid, nid + n)); nid += n
    n_int = nid - R2
    loop_ids = np.arange(nid, nid + nL)
    rows_ids[-1] = loop_ids
    faces = []
    for j in range(J):
        faces += zipper(rows_ids[j], rows_nu[j], rows_ids[j + 1], rows_nu[j + 1])
    weld.rows_nu, weld.rows_ids, weld.n_int = rows_nu, rows_ids, n_int
    weld.cfaces = np.asarray(faces, np.int64)
    weld.c_t = np.concatenate([np.full(len(r), j / J) for j, r in enumerate(rows_nu)])
    weld.c_nu = np.concatenate(rows_nu)
    weld.c_ids = np.concatenate(rows_ids)
    weld.ref_nu_f, weld.ref_nu_l = nu_f, nu_l
    # neighbour averaging over the patch graph (foot ring + interior; the outer row's points held), for smoothing the
    # base under the patch
    from scipy import sparse
    n_all = R2 + n_int + nL
    E = np.concatenate([weld.cfaces[:, [0, 1]], weld.cfaces[:, [1, 2]], weld.cfaces[:, [2, 0]]])
    A = sparse.coo_matrix((np.ones(2 * len(E)), (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])), shape=(n_all, n_all)).tocsr()
    A.data[:] = 1.0
    deg = np.asarray(A.sum(1)).ravel()
    weld.avg_op = sparse.diags(1.0 / np.maximum(deg, 1)) @ A



def layout(weld: "Weld", inp: dict) -> dict:
    """Chart positions of every canonical vertex for one animal (foot ring and outer row exact): {"P2": (n_c, 2) by
    canonical id}."""
    R2 = 2 * bl.NC
    order = foot_order()
    mu_f, nu_f = ell_of(inp, inp["ring2d"][order])
    nu_f = np.mod(nu_f, 2 * np.pi)
    s = np.argsort(nu_f); 
    if not np.all(s == np.arange(R2)):
        raise RuntimeError("weld: foot ring not ordered in nu")
    mu_l, _ = ell_of(inp, weld.loop2d)
    nu_l = _loop_nu(inp, weld.loop2d)
    # the outer limit of the inner rows follows the loop's straight edges (densified), not a curve through its points
    Ld = _densify_ring(weld.loop2d, 8)
    mu_ld, _ = ell_of(inp, Ld)
    nu_ld = _loop_nu(inp, Ld)

    J = N_ROWS
    t = weld.c_t; nuh = weld.c_nu; ids = weld.c_ids
    nu = (1 - smoothstep(t)) * periodic_map(nuh, weld.ref_nu_f, nu_f) + smoothstep(t) * periodic_map(nuh, weld.ref_nu_l, nu_l)
    muF = periodic_interp(nu, nu_f, mu_f); muL = periodic_interp(nu, nu_ld, mu_ld)
    # rows by arclength along each column (a hyperbola nu = const): geometric, the first step DELTA0 x R (about the
    # blade's own ring step at its foot), growing to fill the column (never shrinking)
    def column_s(mu0, mu1, nuv, n=129):
        g = np.linspace(0.0, 1.0, n)
        M = mu0[:, None] + (mu1 - mu0)[:, None] * g[None, :]
        sp = inp["f"] * np.abs(np.sinh(M + 1j * nuv[:, None]))
        S = np.concatenate([np.zeros((len(M), 1)), np.cumsum(0.5 * (sp[:, 1:] + sp[:, :-1]) * np.diff(M, axis=1), axis=1)], 1)
        return M, S
    M, S = column_s(muF, muL, nu)
    Stot = S[:, -1]
    kk = np.linspace(12.0, 1e-4, 400); qq = kk / np.expm1(kk)
    q = np.clip(J * max(DELTA0 * inp["R"], DELTA_MIN_CM) / np.maximum(Stot, 1e-6), qq[0], 1.0)
    kap = np.interp(q, qq, kk)
    sig = np.where(kap > 1e-3, np.expm1(kap * t) / np.expm1(kap), t)
    target = sig * Stot
    mu = np.array([np.interp(target[i], S[i], M[i]) for i in range(len(t))])
    XY = from_ell(mu, nu, inp["c"], inp["f"])
    # exact rows 0 and J
    r0 = slice(0, R2)
    XY[r0] = inp["ring2d"][order]
    nL = len(weld.loop2d)
    XY[-nL:] = weld.loop2d
    out = np.zeros_like(XY); out[ids] = XY
    return {"P2": out}


# ---- embedding in the base surface and per-animal geometry -----------------------------------------------------------
def locate(weld: "Weld", P2: np.ndarray):
    """For chart points: the base face each lies in (index into weld.base_faces) and its barycentric weights (in the
    chart of the rest base). Points outside every face take the face with the largest minimum weight (clamped)."""
    T = weld.chart(weld.R_rest)[weld.base_faces]                     # (nt, 3, 2)
    cell = 6.0
    lo = T.min(1); hi = T.max(1)
    g0 = np.floor(np.minimum(lo.min(0), P2.min(0)) / cell) - 1
    buckets = {}
    i0 = np.floor(lo / cell - g0).astype(int); i1 = np.floor(hi / cell - g0).astype(int)
    for t in range(len(T)):
        for gx in range(i0[t, 0], i1[t, 0] + 1):
            for gy in range(i0[t, 1], i1[t, 1] + 1):
                buckets.setdefault((gx, gy), []).append(t)
    pc = np.floor(P2 / cell - g0).astype(int)
    tri = np.full(len(P2), -1); bary = np.zeros((len(P2), 3)); best = np.full(len(P2), -np.inf)
    keys = {}
    for i, k in enumerate(map(tuple, pc)):
        keys.setdefault(k, []).append(i)
    for k, idx in keys.items():
        cand = buckets.get(k)
        if not cand:
            continue
        idx = np.asarray(idx); cand = np.asarray(cand)
        A, B, Cc = T[cand, 0], T[cand, 1], T[cand, 2]
        v0, v1 = B - A, Cc - A
        d = v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]
        d = np.where(np.abs(d) < 1e-12, 1e-12, d)
        q = P2[idx][:, None, :] - A[None]
        l1 = (q[..., 0] * v1[None, :, 1] - q[..., 1] * v1[None, :, 0]) / d[None]
        l2 = (v0[None, :, 0] * q[..., 1] - v0[None, :, 1] * q[..., 0]) / d[None]
        l0 = 1 - l1 - l2
        mn = np.minimum(np.minimum(l0, l1), l2)
        j = np.argmax(mn, axis=1)
        sel = mn[np.arange(len(idx)), j] > best[idx]
        ii = idx[sel]; jj = j[sel]
        best[ii] = mn[sel, jj]; tri[ii] = cand[jj]
        bary[ii] = np.stack([l0[sel, jj], l1[sel, jj], l2[sel, jj]], 1)
    if (tri < 0).any():
        raise RuntimeError(f"weld: {(tri < 0).sum()} patch points outside the base surface")
    bary = np.clip(bary, 0.0, None); bary /= bary.sum(1, keepdims=True)
    return tri, bary, best


def base_points(weld: "Weld", Ex: np.ndarray, N: np.ndarray, tri, bary, wphong=None, cdir=None):
    """Points of the base surface at (tri, bary): Phong tessellation (alpha 0.75) of the rig's flat triangles with its
    smooth normals, so the fine patch does not print the coarse rig facets; and the interpolated unit normal."""
    F = weld.base_faces[tri]
    Pi = Ex[F]; ni = N[F]
    P = (Pi * bary[..., None]).sum(1)
    proj = P[:, None, :] - ((P[:, None, :] - Pi) * ni).sum(-1, keepdims=True) * ni
    Ps = (proj * bary[..., None]).sum(1)
    n = (ni * bary[..., None]).sum(1)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    # only the part of the correction along the smooth normal: the points keep their place on the flat triangles
    # (a sideways shift across the back's ridge could swap the two sides of the knife-thin trailing edge)
    w = 0.75 if wphong is None else 0.75 * np.asarray(wphong)[:, None]
    c = n if cdir is None else cdir          # (on the hump: along the lift direction, see animal_geometry)
    return P + w * ((Ps - P) * c).sum(1, keepdims=True) / np.maximum((n * c).sum(1, keepdims=True), 0.3) * c, n


def rotate_min(v: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotate vectors v by the smallest rotations taking unit a to unit b (row-wise)."""
    c = (a * b).sum(1, keepdims=True)
    k = np.cross(a, b)
    # Rodrigues for the rotation a -> b: v' = v c + (k x v) + k (k . v) / (1 + c)
    return v * c + np.cross(k, v) + k * ((k * v).sum(1, keepdims=True) / (1.0 + np.clip(c, -0.999, None)))


def smooth_dirs(n: np.ndarray, d: np.ndarray, b: np.ndarray) -> np.ndarray:
    v = n * (1 - b)[:, None] + (d if d.ndim == 2 else d[None, :]) * b[:, None]
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)


def smooth_base(weld: "Weld", P: np.ndarray, loopP: np.ndarray, n: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Smooth patch base points P (foot ring + interior, canonical order) along their normals n over the patch graph
    (the loop's points loopP held), weight w per point (1 inside, fading to 0 at the outer rows)."""
    X = np.concatenate([P, loopP])
    nI = len(P)
    nn = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
    # Taubin's lambda / mu passes: the creases go, the curved back does not shrink against the held loop
    for k in range(2 * BASE_SMOOTH_IT):
        f = (0.5 if k % 2 == 0 else -0.53) * w[:, None]
        d = weld.avg_op @ X
        step = ((d[:nI] - X[:nI]) * nn).sum(1, keepdims=True) * nn
        X[:nI] = X[:nI] + f * step
    return X[:nI]


def fade(levels: np.ndarray) -> np.ndarray:
    """1 at the blade's foot ring, 0 from H_FOOT + FADE up (smooth)."""
    return 1.0 - smoothstep((np.asarray(levels) - H_FOOT) / FADE)


def bend_bins(weld: "Weld", step: float = 15.0, half: float = 30.0):
    """Bins along the region (chart a): centres and the extended base vertices of each (for a rotation per bin)."""
    Cx = weld.chart(weld.R_rest)
    v = np.unique(weld.base_faces)
    a0, a1 = Cx[v, 0].min(), Cx[v, 0].max()
    cs = np.arange(a0, a1 + step, step)
    sels = [v[np.abs(Cx[v, 0] - c) < half] for c in cs]
    return cs, sels


def bin_rotations(bins, Ex0: np.ndarray, Ex1: np.ndarray):
    cs, sels = bins
    Rs = []
    for s in sels:
        Rm, _ = bl.kabsch(Ex0[s], Ex1[s]) if len(s) >= 4 else (np.eye(3), None)
        Rs.append(Rm)
    return cs, np.asarray(Rs)


def bin_transforms(bins, Ex0: np.ndarray, Ex1: np.ndarray):
    cs, sels = bins
    Rs, ts = [], []
    for s in sels:
        Rm, t = bl.kabsch(Ex0[s], Ex1[s]) if len(s) >= 4 else (np.eye(3), np.zeros(3))
        Rs.append(Rm); ts.append(t)
    return cs, np.asarray(Rs), np.asarray(ts)


def transform_by_bins(P: np.ndarray, a: np.ndarray, tr) -> np.ndarray:
    """Points carried by the back under them: the rigid motion of each bin, blended linearly between bin centres."""
    cs, Rs, ts = tr
    j = np.clip(np.searchsorted(cs, a) - 1, 0, len(cs) - 2)
    f = np.clip((a - cs[j]) / (cs[j + 1] - cs[j]), 0.0, 1.0)[:, None]
    p0 = np.einsum("nij,nj->ni", Rs[j], P) + ts[j]
    p1 = np.einsum("nij,nj->ni", Rs[j + 1], P) + ts[j + 1]
    return p0 * (1 - f) + p1 * f


def rotate_by_bins(v: np.ndarray, a: np.ndarray, rot) -> np.ndarray:
    """Rotate vectors v by the bin rotations interpolated (linearly, re-orthonormalised) at chart positions a."""
    cs, Rs = rot
    j = np.clip(np.searchsorted(cs, a) - 1, 0, len(cs) - 2)
    f = np.clip((a - cs[j]) / (cs[j + 1] - cs[j]), 0.0, 1.0)
    M = Rs[j] * (1 - f)[:, None, None] + Rs[j + 1] * f[:, None, None]
    U, _, Vt = np.linalg.svd(M)
    Rm = U @ Vt
    return np.einsum("nij,nj->ni", Rm, v)


def animal_geometry(weld: "Weld", inp: dict, outward_sign: float, Ex=None, frame=None, lift_dir=None) -> dict:
    """Per-animal patch data: chart layout, embedding, lift above the back, rest positions, back UVs.
    Points: foot ring (blade ring 0, 2NC), patch interior (n_int), the two landmarks (front base, back base)."""
    R2 = 2 * bl.NC
    lay = layout(weld, inp)
    P2 = lay["P2"]
    pts = np.concatenate([P2[:R2 + weld.n_int], inp["fb2d"][None], inp["bb2d"][None]])
    tri, bary, inside = locate(weld, pts)
    Ex = weld.R_rest if Ex is None else Ex
    N = weld.base_normals(Ex) * outward_sign
    # the smoothing of the rig's flat faces fades out over the outer rows: the outer row lies on the rig's own faces
    t_id = np.zeros(len(weld.c_t)); t_id[weld.c_ids] = weld.c_t
    wph = np.concatenate([1.0 - smoothstep((t_id[:R2 + weld.n_int] - PHONG_T0) / (1.0 - PHONG_T0)), [1.0, 1.0]])
    base, nb = base_points(weld, Ex, N, tri, bary, wph)
    B, _, up = inp["frame"] if frame is None else frame
    dn = weld.e_n if lift_dir is None else lift_dir
    # the rig's back carries creases (its dorsal ridge and its posed skin's folds lie on edges between 10-20 cm faces,
    # hidden on the rig by its interpolated normals); the fine patch would show them: the base under the patch is
    # smoothed along its normal (here and in every pose, factory.weld.smooth_base), fading out at the loop
    nI = R2 + weld.n_int
    core_pts = np.concatenate([_densify_ring(inp["ring2d"], 4), inp["core2d"]]) if len(inp["core2d"]) else _densify_ring(inp["ring2d"], 4)
    d_core = np.sqrt(((P2[:nI, None, :] - core_pts[None]) ** 2).sum(-1)).min(1)
    wsm = 1.0 - smoothstep((t_id[:nI] - 0.9) / 0.1)   # every row but the last few (the loop's points are held)
    lt0, lb0, _ = locate(weld, weld.loop2d)
    lbase0 = Ex[weld.loop_rep]
    # smoothing direction: the lift direction on the hump (its thin rows by the foot keep their order), the base's own
    # normal away from it (the rig's creases on the steep flanks lie sideways to the lift direction)
    lift0 = hump_lift(inp, pts, (base - B) @ up, np.zeros(0) if not len(inp["core2d"]) else
                      (base_points(weld, Ex, N, *locate(weld, inp["core2d"])[:2])[0] - B) @ up, foot_base_h=((base - B) @ up)[:R2])
    bmix = smoothstep(lift0[:nI] / (0.03 * inp["H"]))
    bmix_all = np.concatenate([bmix, [1.0, 1.0]])
    base, nb = base_points(weld, Ex, N, tri, bary, wph, smooth_dirs(base_points(weld, Ex, N, tri, bary)[1], dn, bmix_all))
    sdir_rest = smooth_dirs(nb[:nI], dn, bmix)
    base = np.concatenate([smooth_base(weld, base[:nI], lbase0, sdir_rest, wsm), base[nI:]])
    base_h = (base - B) @ up
    if len(inp["core2d"]):
        ct, cb, _ = locate(weld, inp["core2d"])
        cbase, _ = base_points(weld, Ex, N, ct, cb)
        core_ref = (cbase - B) @ up
    else:
        core_ref = np.zeros(0)
    lift = hump_lift(inp, pts, base_h, core_ref, foot_base_h=base_h[:R2])
    # the lift goes along the chart's normal (no sideways shift)
    off = (lift / max(float(up @ dn), 0.3))[:, None] * dn[None, :]
    rest = base + off
    F = weld.base_faces[tri]
    uv = (weld.uv_ext[F] * bary[..., None]).sum(1)
    lt, lb, _ = locate(weld, weld.loop2d)
    lbase, _ = base_points(weld, Ex, N, lt, lb)
    loop_lift = hump_lift(inp, np.concatenate([pts[:R2], weld.loop2d]), np.concatenate([base_h[:R2], (lbase - B) @ up]), core_ref,
                          foot_base_h=base_h[:R2])[R2:]
    ringd = _densify_ring(inp["ring2d"], 6)
    dd = np.sqrt(((pts[:, None, :] - ringd[None]) ** 2).sum(-1))
    d_foot = dd.min(1)
    col_near = np.argmin(dd, 1) // 6                   # nearest foot column (blade ring-0 index order of ring2d)
    # round 13: smooth weights over the foot columns (the foot's offset spread without jumps at the knife-thin TE)
    dcol = np.sqrt(((pts[:, None, :] - inp["ring2d"][None]) ** 2).sum(-1))
    wcol = np.exp(-((dcol - dcol.min(1, keepdims=True)) / (0.08 * inp["R"])) ** 2)
    wcol /= wcol.sum(1, keepdims=True)
    return {"P2": P2, "tri": tri, "bary": bary, "D": lift, "d_foot": d_foot, "col_near": col_near, "wcol": wcol, "off": off, "n_rest": nb, "rest": rest, "uv": uv, "a": pts[:, 0], "wph": wph,
            "wsm": wsm, "loop_tri": (lt0, lb0), "bmix": bmix,
            "inside": inside, "loop_D": loop_lift}
