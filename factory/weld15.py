"""run06 round 15: the welded fin as ONE smooth surface per animal (no foot ring as a geometric feature), and a posing
that is a smooth image of the rest weld.

Selected by the run config's whale.blade_weld = 15 (true = round 13, factory.weld + WhaleModel's own _weld_animal /
_weld_posed); WhaleModel._weld_animal / _weld_posed call animal() / posed() here. Same vertex layout and faces as round
13. Posing mode STIFF_MODE "gate" (round 16b, default: "foot", re-posed with "fit" where "foot" self-intersects; the
caller reads model._weld_pose_info after each posing). See data/renders/r06_morning/HANDOVER.md "Round 15" for the measures; the
notes below the first paragraph describe the prototype's first plan, the code's own comments the final one.

Rest shape (per animal, blade frame a = along, y = side, z = up, cm):
  * the round-13 blade (template sections, round / pointed top, trailing-edge nicks) is DRAPED: its rings are sheared
    vertically by the back's midline height under them, fully at the foot ring (z0 = H_FOOT H above the back
    everywhere along a long chord) and not at all from z0 + H_DRAPE H up (tops, nicks unchanged); a raise u_b only
    where the back stands more than E_MAX H above the template's base line (calves);
  * below the foot ring the fin continues as the foot section extruded down with the blade's own local wall slope
    (draft) at the ring, turning smoothly (C2) to at least SIG_MIN: F_ext = (d0 - off(z0 - z)) / N, d0 the signed 2D
    distance to the foot section;
  * the back's field is the height above the animal's own rest back; fin and back are joined by a cubic smooth minimum
    (C2, radius K_BACK H): the fillet is part of one implicit surface, the foot ring is only a row of vertices on it;
  * the fixed patch topology (rows from the foot ring to the region's loop) is laid out in the blade frame's own plane
    (the 'up-chart'), rows spaced by 3D arclength along the surface (first step FIRST_STEP H, like the blade's rings),
    each patch vertex on its base point lifted along the blade's up onto the iso-surface (one root per vertex: the
    field is monotone along up).
Posing: every weld vertex is moved by a smooth transform field fitted to the posed back (weighted Kabsch at nodes on a
NODE_CM lattice over the region, blended with Gaussian weights like linear blend skinning), evaluated at an anchor that
is the vertex's own base point near the back and moves to the fin base's centre with height (STIFF_H H), plus the
back's residual (posed base minus the field's image of the rest base, so the outer rows meet the rig exactly).
"""
from __future__ import annotations

import numpy as np

from factory import blade as bl
from factory import weld as wd

K_BACK = 0.25          # smooth-min radius between fin and back (x H)
SIG_MIN = 0.25         # the walls below the foot ring lean out by at least this (horizontal per vertical) ...
SIG_MAX = 3.0          # ... and at most this (the blade's low ramps continue no flatter than ~18 degrees)
T_DRAFT = 0.10         # the C2 turn from the blade's own slope at the ring to the clipped slope takes this depth (x H) ...
T_DRAFT_H = 0.10       # ... or less: at most this horizontal run (x H) for flat ramps
H_DRAPE = 0.60         # the drape fades out over this (x H) above the foot ring
DRAPE = True          # round-15 option: shear the blade's lower part by the back's midline height (off: rigid raise)
RAISE_CLEAR = 0.10     # without drape: the foot ring stays at least this (x H) above the back's midline
E_MAX = 0.25           # largest upward drape (x H); the rest is a rigid raise of the whole blade
FIRST_STEP = 0.03      # the patch's first row step at the foot ring (x H, 3D arclength) ...
FIRST_MIN_CM = 0.8     # ... at least this
SAMPLE_CM = 1.5        # base sampling for the blade-frame maps
STIFF_MODE = "gate"    # posing: "gate" (round 16: "foot", re-posed with "fit" only where "foot" self-intersects; see posed),
                       # "fit" (round 16: "foot" below a bend threshold, above it the blade carried by a fit to its posed
                       # foot ring and the back's residual smoothed by the foot), "foot" (round 15), "blend" (anchors to the base
                       # centre with height) or "rigid" (every weld vertex at the centre)
FOOT_D = 0.30          # posing (foot3): patch anchors move from their own base point to the fin base centre over this (x H)
STIFF_H = 0.40         # posing: anchors move from the vertex's own base point to the fin base centre over this (x H)
FIT_H = 0.50           # posing ("fit", round 16): blade vertices this far (x H) above their column's foot are its upper part ...
FIT_SIG = 0.25         # ... the foot ring weighted by a Gaussian of this width (x H) along the chord around them ...
FIT_FLOOR = 0.02       # ... plus this floor (a long low ramp ahead still counts a little)
GATE_M0 = 0.03         # posing ("fit"): the round-16 posing engages from this bend measure (x H; see posed; below it the
                       # result is round 15's "foot" posing exactly) ...
GATE_M1 = 0.08         # ... fully at this one (<= 0: always fully)
LAST = (0.0, 0.0)      # (bend measure, gate) of the last "fit" posing (diagnostics)
SECOND_GATE_OPEN = 1   # posing ("gate"): 1 = the second try uses "fit" with its bend gate fully open
RSM_IT = 40            # posing ("fit"): Jacobi passes smoothing the back's residual over the patch near the foot (0 = off) ...
RSM_LAM = 0.5          # ... step
RSM_D = 0.5            # ... full strength at the foot, none from this distance (x H, chart) on ...
RSM_DMIN = 0.0         # ... or from this distance (cm), whichever is larger (small calf fins)
NODE_CM = 12.0         # posing: transform-field node spacing (chart cm)
NODE_RAD = 30.0        # ... Kabsch neighbourhood radius (cm)
NODE_SIG = 15.0        # ... blending width (cm)
DEBUG = False
LAYOUT_SIG_NU = 0.25   # smoothing of the 3D-arclength row positions across columns (radians of nu)
LAYOUT_W3D = 0.0      # weight of the 3D-arclength rows against the chart-arclength rows
LAYOUT_MIN0 = 0.5      # ... never closer to the foot ring than this fraction of the chart-arclength rows
N_COL = 97             # samples per patch column for the 3D arclength


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def smin3(a, b, k):
    """Cubic smooth minimum (C2): min(a, b) - h^3 k / 6, h = max(k - |a - b|, 0) / k."""
    h = np.clip(k - np.abs(a - b), 0.0, None) / k
    return np.minimum(a, b) - h ** 3 * k / 6.0


def off_draft(t, s0, s1, T):
    """Outward offset of the extruded wall at depth t below the foot ring: slope s0 at t = 0 turning C2 to s1 by T."""
    u = np.asarray(t, float) / T
    Phi = np.where(u <= 0.0, 0.0, np.where(u < 1.0, u ** 3 - 0.5 * u ** 4, u - 0.5))
    return s0 * t + (s1 - s0) * T * Phi


# ---- the animal's rest back seen in the blade frame ------------------------------------------------------------------
class BaseMap:
    def __init__(self, W, Ex, N, axes):
        from matplotlib.path import Path
        from scipy.interpolate import CloughTocher2DInterpolator, LinearNDInterpolator, NearestNDInterpolator
        self.B, self.al, self.ey, self.up = axes
        L2 = W.loop2d
        lo, hi = L2.min(0), L2.max(0)
        gx, gy = np.meshgrid(np.arange(lo[0], hi[0], SAMPLE_CM), np.arange(lo[1], hi[1], SAMPLE_CM), indexing="ij")
        g = np.stack([gx.ravel(), gy.ravel()], 1)
        q = np.concatenate([g[Path(L2).contains_points(g)], wd._densify_ring(L2, 3)])
        tri, bary, _ = wd.locate(W, q)
        X, n = wd.base_points(W, Ex, N, tri, bary)
        abz = self.coords(X)
        self.q, self.U, self.Z, self.c = q, abz[:, :2], abz[:, 2], n @ self.up
        self._z = CloughTocher2DInterpolator(self.U, self.Z)
        self._c = LinearNDInterpolator(self.U, self.c)
        self._inv = LinearNDInterpolator(self.U, self.q)
        self._nn = NearestNDInterpolator(self.U, np.arange(len(self.U)))

    def coords(self, X):
        r = np.asarray(X, float) - self.B
        return np.stack([r @ self.al, r @ self.ey, r @ self.up], 1)

    def _fill(self, v, P2, arr):
        bad = ~np.isfinite(v if v.ndim == 1 else v[:, 0])
        if bad.any():
            v[bad] = arr[self._nn(P2[bad]).astype(int)]
        return v

    def z(self, P2):
        return self._fill(np.asarray(self._z(P2), float).ravel(), P2, self.Z)

    def cfac(self, P2):
        return self._fill(np.asarray(self._c(P2), float).ravel(), P2, self.c)

    def inv(self, P2):
        return self._fill(np.asarray(self._inv(P2), float).reshape(-1, 2), P2, self.q)


# ---- the fin's field below the foot ring: the template's own sections, draped -------------------------------------------
LOW_LEVELS = np.round(np.arange(-0.35, 0.3001, 0.01), 4)     # (x H) sections below / just above the foot ring
LOW_ROUND = 0.0       # the low sections are offset outward by up to this (x H): rounded ridges below the ring
LOW_RATE = 0.10        # the low sections' half-width grows by this per unit depth (x H / H) ...
LOW_T = 0.10           # ... reached this far below the foot ring (C2 from the blade's own rate)


def _poly_dist(Q, P):
    """Signed distance (+ outside) of points Q (n, 2) to closed polygons P (n, m, 2) (one polygon per point) or (m, 2)."""
    if P.ndim == 2:
        P = np.broadcast_to(P[None], (len(Q),) + P.shape)
    A = P; Bq = np.roll(P, -1, axis=1); AB = Bq - A
    q = Q[:, None, :]
    t = np.clip(((q - A) * AB).sum(-1) / np.maximum((AB * AB).sum(-1), 1e-12), 0.0, 1.0)
    d = np.sqrt(((q - (A + t[..., None] * AB)) ** 2).sum(-1)).min(1)
    cr = ((A[..., 1] > q[..., 1]) != (Bq[..., 1] > q[..., 1])) & \
         (q[..., 0] < (Bq[..., 0] - A[..., 0]) * (q[..., 1] - A[..., 1]) / np.where(np.abs(AB[..., 1]) < 1e-12, 1e-12, AB[..., 1]) + A[..., 0])
    inside = (cr.sum(1) % 2) == 1
    return np.where(inside, -d, d)


class Fin:
    """F_fin(a, y, z) = d(a, y; section at the undraped height z - bm(a)) / N: the template's sections (the same edges,
    thickness and nicks as the blade: blade.make on LOW_LEVELS with the blade's own random draws), the drape applied as
    the vertical shear of the blade (full below the foot ring), N = sqrt(1 + (dd/dz)^2) (a first-order distance)."""

    def __init__(self, Vlow, levels_cm, H, ag, bmid):
        self.S = Vlow                    # (nl, 2NC, 2)
        self.lv = levels_cm; self.dl = float(levels_cm[1] - levels_cm[0])
        self.H = H; self.ag, self.bmid = ag, bmid
        self.grid = None

    def bm(self, a):
        return np.interp(a, self.ag, self.bmid)

    def _sec(self, zu):
        f = (zu - self.lv[0]) / self.dl
        k = np.clip(np.floor(f).astype(int), 0, len(self.lv) - 2); t = np.clip(f - k, 0.0, 1.0)
        return self.S[k] * (1 - t)[:, None, None] + self.S[k + 1] * t[:, None, None]

    def F(self, Q, z):
        zu = z - self.bm(Q[:, 0])
        d = _poly_dist(Q, self._sec(zu))
        e = 0.5 * self.dl
        dz = (_poly_dist(Q, self._sec(zu + e)) - _poly_dist(Q, self._sec(zu - e))) / (2 * e)
        return d / np.sqrt(1.0 + dz * dz) - self.rho(zu)

    def rho(self, zu):
        """Rounding of the low sections (an outward offset growing below the foot ring): the knife-thin trailing edge
        continued below the ring would be a sharp wedge ridge."""
        return LOW_ROUND * self.H * smoothstep((self.lv[-1] - zu) / (0.1 * self.H))

    def make_grid(self, lo, hi, step):
        ax = np.arange(lo[0], hi[0] + step, step); ay = np.arange(lo[1], hi[1] + step, step)
        ga, gy = np.meshgrid(ax, ay, indexing="ij"); Q = np.stack([ga.ravel(), gy.ravel()], 1)
        from scipy.spatial import cKDTree
        from matplotlib.path import Path
        D = np.zeros((len(self.lv), len(ax), len(ay)))
        sub = 6
        f = (np.arange(sub) / sub)[None, :, None]
        for k in range(len(self.lv)):
            P = self.S[k]
            Pd = (P[:, None, :] * (1 - f) + np.roll(P, -1, 0)[:, None, :] * f).reshape(-1, 2)
            dist, _ = cKDTree(Pd).query(Q)
            ins = Path(P).contains_points(Q)
            D[k] = np.where(ins, -dist, dist).reshape(len(ax), len(ay))
        self.grid = (ax, ay, step, D)

    def F_grid(self, Q, z):
        from scipy.ndimage import map_coordinates
        ax, ay, step, D = self.grid
        zu = z - self.bm(Q[:, 0])
        fz = (zu - self.lv[0]) / self.dl; fa = (Q[:, 0] - ax[0]) / step; fy = (Q[:, 1] - ay[0]) / step
        d = map_coordinates(D, [fz, fa, fy], order=1, mode="nearest")
        dz = (map_coordinates(D, [fz + 0.5, fa, fy], order=1, mode="nearest") - map_coordinates(D, [fz - 0.5, fa, fy], order=1, mode="nearest")) / self.dl
        return d / np.sqrt(1.0 + dz * dz) - self.rho(zu)

    def lift(self, P2, zX, cX, k, z_top, grid=False):
        """Height L >= 0 above the base point (along up) of the iso-surface smin(back, fin) over chart points P2 (blade
        frame a, y) whose base lies at height zX (base normal's up component cX); z_top: the foot ring's height there."""
        Ff = self.F_grid if grid else self.F
        cX = np.clip(cX, 0.2, 1.0)

        def g(L):
            return smin3(L * cX, Ff(P2, zX + L), k)
        lo = np.zeros(len(P2))
        hi = np.maximum(z_top - zX, 0.0) + 0.05 * self.H
        ok = g(hi) > 0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            pos = g(mid) > 0
            hi = np.where(pos, mid, hi); lo = np.where(pos, lo, mid)
        L = 0.5 * (lo + hi)
        return L, ok


def _nest(Vlow, levels, z0h, Hcm, na=1200):
    """Nested low sections: below the foot ring every section contains the ones above it (half-width profile w(a) on
    a common grid, cumulative maximum going down, chord ends likewise), resampled to the blade's 2NC points per section
    (cosine spacing along the chord, the same layout as blade.make). Above the foot ring: unchanged."""
    NC = bl.NC
    nl = len(levels)
    a_lo = Vlow[..., 0].min() - 1e-6; a_hi = Vlow[..., 0].max() + 1e-6
    A = np.linspace(a_lo, a_hi, na)
    te = Vlow[:, 0, 0].copy(); le = Vlow[:, NC - 1, 0].copy()
    Wp = np.zeros((nl, na))
    for k in range(nl):
        ar, wr = Vlow[k, :NC, 0], np.abs(Vlow[k, :NC, 1])
        o = np.argsort(ar)
        Wp[k] = np.where((A >= ar.min()) & (A <= ar.max()), np.interp(A, ar[o], wr[o]), 0.0)
    k0 = int(np.argmin(np.abs(levels - z0h)))
    out = Vlow.copy()
    s = (1 - np.cos(np.linspace(0, np.pi, NC))) / 2
    dA = A[1] - A[0]
    for k in range(k0 - 1, -1, -1):
        # at least a draft of SIG_MIN (reached C1 within 0.05 H below the ring): no vertical walls under the ring
        dz = (levels[k + 1] - levels[k]) * Hcm
        sm = SIG_MIN * smoothstep((z0h - levels[k]) / 0.05) * dz
        grow = np.maximum(Wp[k + 1] + sm, 0.0) * (Wp[k + 1] > 0)
        # widen the profile along a as well (the chord ends move out by the same draft)
        sh = int(np.ceil(sm / dA)) if sm > 0 else 0
        if sh:
            g2 = grow.copy()
            for q in range(1, sh + 1):
                g2[q:] = np.maximum(g2[q:], grow[:-q] - q * dA); g2[:-q] = np.maximum(g2[:-q], grow[q:] - q * dA)
            grow = g2
        Wp[k] = np.maximum(Wp[k], grow)
        te[k] = min(te[k], te[k + 1] - sm); le[k] = max(le[k], le[k + 1] + sm)
        a = te[k] + (le[k] - te[k]) * s
        w = np.interp(a, A, Wp[k])
        w[0] = max(w[0], abs(Vlow[k, 0, 1])); w[-1] = max(w[-1], abs(Vlow[k, NC - 1, 1]))
        out[k, :NC, 0] = a; out[k, :NC, 1] = w
        out[k, NC:, 0] = a[::-1]; out[k, NC:, 1] = -w[::-1]
    return out


def low_sections(model, ind, levels):
    """blade.make on the given levels with the individual's own template, H, nicks, thickness and random draws (the
    same edges as its welded blade)."""
    spec = None
    if ind is not None:
        try:
            spec = model.individual(ind)["spec"]
        except Exception:
            spec = None
    t = int(getattr(spec, "blade_template", -1)) if spec is not None else -1
    if t < 0:
        t, H, nicks, thick, seed = 0, 45.0, 0, 1.0, 0
    else:
        H, nicks, thick, seed = float(spec.blade_H_cm), int(spec.blade_nicks), float(spec.blade_thickness), int(spec.seed)
    temps = model._blade_lib["templates"]
    loc = bl.make(temps[t % len(temps)], model._blade_lib["heights"], H, np.random.default_rng(seed + 991), nicks, thick,
                  levels=np.asarray(levels, float))
    return loc


# ---- layout of the fixed patch topology in the up-chart, rows by 3D arclength ---------------------------------------
def _push_out(XYu, st, ring_u, H):
    """Chart points of the patch on or inside the foot ring's footprint moved just outside its nearest edge."""
    XYu = XYu.copy()
    dd0 = _poly_dist(XYu[st], ring_u)
    ins = dd0 < 2e-3 * H
    if ins.any():
        Pk = ring_u; Q = XYu[st[ins]]
        A_ = Pk; AB = np.roll(Pk, -1, 0) - Pk
        tt = np.clip((((Q[:, None] - A_[None]) * AB[None]).sum(-1)) / np.maximum((AB * AB).sum(-1), 1e-12)[None], 0, 1)
        C_ = A_[None] + tt[..., None] * AB[None]
        j = np.argmin(((Q[:, None] - C_) ** 2).sum(-1), 1)
        cp = C_[np.arange(len(Q)), j]; ed = AB[j] / (np.linalg.norm(AB[j], axis=1, keepdims=True) + 1e-12)
        nrm = np.stack([ed[:, 1], -ed[:, 0]], 1)
        nrm *= np.where(_poly_dist(cp + 1e-3 * H * nrm, Pk) > 0, 1.0, -1.0)[:, None]
        XYu[st[ins]] = cp + 2e-3 * H * nrm
    return XYu, ins


def layout15(W, ring_u, loop_u, prof, H):
    NC = bl.NC; R2 = 2 * NC
    order = wd.foot_order()
    inp = {"c": 0.5 * (ring_u[[0, R2 - 1], 0].mean() + ring_u[[NC - 1, NC], 0].mean()),
           "f": wd.FOCUS * 0.5 * (ring_u[[NC - 1, NC], 0].mean() - ring_u[[0, R2 - 1], 0].mean())}
    mu_f, nu_f = wd.ell_of(inp, ring_u[order])
    nu_f = np.mod(nu_f, 2 * np.pi)
    if not np.all(np.argsort(nu_f) == np.arange(R2)):
        raise RuntimeError("weld15: foot ring not ordered in nu")
    nu_l = wd._loop_nu(inp, loop_u)
    Ld = wd._densify_ring(loop_u, 8)
    mu_ld, _ = wd.ell_of(inp, Ld)
    nu_ld = wd._loop_nu(inp, Ld)
    t = W.c_t; nuh = W.c_nu; ids = W.c_ids
    nu = (1 - wd.smoothstep(t)) * wd.periodic_map(nuh, W.ref_nu_f, nu_f) + wd.smoothstep(t) * wd.periodic_map(nuh, W.ref_nu_l, nu_l)
    # the foot curve in elliptic coordinates along the ring's straight edges (densified), not linear between its points
    Fd = wd._densify_ring(ring_u[order], 16)
    mu_fd, nu_fd = wd.ell_of(inp, Fd)
    nu_fd = np.mod(nu_fd, 2 * np.pi); of = np.argsort(nu_fd)
    muF = wd.periodic_interp(nu, nu_fd[of], mu_fd[of]); muL = wd.periodic_interp(nu, nu_ld, mu_ld)
    gq = np.linspace(0.0, 1.0, N_COL) ** 2
    M = muF[:, None] + (muL - muF)[:, None] * gq[None, :]
    XY = wd.from_ell(M.ravel(), np.repeat(nu, N_COL), inp["c"], inp["f"]).reshape(len(t), N_COL, 2)
    Zs = prof(XY.reshape(-1, 2)).reshape(len(t), N_COL)
    P3 = np.concatenate([XY, Zs[..., None]], -1)
    seg = np.linalg.norm(np.diff(P3, axis=1), axis=2)
    S = np.concatenate([np.zeros((len(t), 1)), np.cumsum(seg, 1)], 1)
    Stot = np.maximum(S[:, -1], 1e-6)
    J = wd.N_ROWS
    kk = np.linspace(12.0, 1e-4, 400); qq = kk / np.expm1(kk)
    first = max(FIRST_STEP * H, FIRST_MIN_CM)
    qv = np.clip(J * first / Stot, qq[0], 1.0)
    kap = np.interp(qv, qq, kk)
    sig = np.where(kap > 1e-3, np.expm1(kap * t) / np.expm1(kap), t)
    target = sig * Stot
    mu = np.array([np.interp(target[i], S[i], M[i]) for i in range(len(t))])
    # smooth the rows' fraction along their columns across the columns (periodic Gaussian in nu, the same kernel for
    # every row: neighbouring columns with very different profiles - a wall beside a ramp - would fold the chart)
    tau = (mu - muF) / np.maximum(muL - muF, 1e-9)
    # ... and blended with the plain chart-arclength rows (round 13's layout), which never fold
    St = (np.concatenate([np.zeros((len(t), 1)), np.cumsum(np.linalg.norm(np.diff(XY, axis=1), axis=2), 1)], 1))
    qv0 = np.clip(J * max(wd.DELTA0 * 0.36 * H, wd.DELTA_MIN_CM) / np.maximum(St[:, -1], 1e-6), qq[0], 1.0)
    kap0 = np.interp(qv0, qq, kk)
    sig0 = np.where(kap0 > 1e-3, np.expm1(kap0 * t) / np.expm1(kap0), t)
    mu0 = np.array([np.interp(sig0[i] * St[i, -1], St[i], M[i]) for i in range(len(t))])
    tau0 = (mu0 - muF) / np.maximum(muL - muF, 1e-9)
    tau = np.maximum(LAYOUT_W3D * tau + (1 - LAYOUT_W3D) * tau0, LAYOUT_MIN0 * tau0)
    G = 1024; gnu = np.linspace(0, 2 * np.pi, G, endpoint=False)
    kx = np.arange(-G // 2, G // 2); ker = np.exp(-0.5 * (kx * (2 * np.pi / G) / LAYOUT_SIG_NU) ** 2); ker /= ker.sum()
    kf = np.fft.rfft(np.fft.ifftshift(ker))
    rid = np.round(t * J).astype(int)
    for j in range(1, J):
        sel = np.nonzero(rid == j)[0]
        nv = np.mod(nu[sel], 2 * np.pi); o = np.argsort(nv)
        f = wd.periodic_interp(gnu, nv[o], tau[sel][o])
        fs = np.fft.irfft(np.fft.rfft(f) * kf, G)
        tau[sel] = wd.periodic_interp(nv, gnu, fs)
    mu = muF + tau * (muL - muF)
    zt = np.array([np.interp(target[i], S[i], Zs[i]) for i in range(len(t))])
    # local steepness of the profile at the target (rise over chart run between the neighbouring samples)
    jj = np.clip(np.array([np.searchsorted(S[i], target[i]) for i in range(len(t))]), 1, N_COL - 1)
    ar = np.arange(len(t))
    run = np.linalg.norm(XY[ar, jj] - XY[ar, jj - 1], axis=1); rise = np.abs(Zs[ar, jj] - Zs[ar, jj - 1])
    steep = rise > run
    T3 = P3[ar, jj] - P3[ar, jj - 1]
    hn = np.linalg.norm(T3[:, :2], axis=1)
    hd = T3[:, :2] / np.maximum(hn, 1e-9)[:, None]
    nrm3 = np.concatenate([-T3[:, 2:3] * hd, hn[:, None]], 1)
    nrm3 /= np.linalg.norm(nrm3, axis=1, keepdims=True) + 1e-12
    XYo = wd.from_ell(mu, nu, inp["c"], inp["f"])
    XYo[:R2] = ring_u[order]
    nL = len(loop_u)
    XYo[-nL:] = loop_u
    out = np.zeros_like(XYo); out[ids] = XYo
    out, _ = _push_out(out, np.arange(R2, R2 + W.n_int), ring_u, H)
    # untangle what is still folded: local averaging of the chart positions around folded faces (foot ring and loop held)
    nI = R2 + W.n_int
    for _ in range(200):
        Tq = out[W.cfaces]
        ar2 = (Tq[:, 1, 0] - Tq[:, 0, 0]) * (Tq[:, 2, 1] - Tq[:, 0, 1]) - (Tq[:, 1, 1] - Tq[:, 0, 1]) * (Tq[:, 2, 0] - Tq[:, 0, 0])
        bad = (np.sign(ar2) != np.sign(np.median(ar2))) & (np.abs(ar2) > 0.02)
        if not bad.any():
            break
        mv = np.zeros(len(out), bool); mv[np.unique(W.cfaces[bad])] = True
        mv = (W.avg_op @ mv.astype(float)) > 0
        mv[:R2] = False; mv[nI:] = False
        # average over the neighbours off the foot ring (pulling row 1 towards the ring folds the blunt ends)
        wm = np.ones(len(out)); wm[:R2] = 0.0
        den = W.avg_op @ wm
        avg = (W.avg_op @ (out * wm[:, None])) / np.maximum(den, 1e-9)[:, None]
        out[mv] = 0.5 * out[mv] + 0.5 * avg[mv]
        out, _ = _push_out(out, np.arange(R2, nI), ring_u, H)
    extra = {}
    for nm, arr in (("mu", mu), ("nu", nu), ("z", zt), ("steep", steep), ("muF", muF), ("n0", nrm3[:, 0]), ("n1", nrm3[:, 1]), ("n2", nrm3[:, 2])):
        o2 = np.zeros(len(arr), arr.dtype); o2[ids] = arr; extra[nm] = o2
    extra["steep"][:R2] = False; extra["steep"][-nL:] = False
    inp.update(extra)
    return out, inp


# ---- per-animal rest geometry -----------------------------------------------------------------------------------------
def animal(model, ind) -> dict:
    """Per-animal rest data (see the module doc). About 18 s per animal on one core, so it is cached on disk (below)
    on top of WhaleModel._weld_animal's in-memory _weld_cache."""
    d = cache_dir()
    if d is None:
        return _animal(model, ind)
    try:
        key = cache_key(model, ind)
    except Exception:                       # never let the cache decide the geometry: build it
        return _animal(model, ind)
    import os
    import pickle
    p = d / f"{key}.pkl"
    if p.exists():
        try:
            with open(p, "rb") as fh:
                rec = pickle.load(fh)
            if isinstance(rec, dict) and rec.get("key") == key:
                return rec["out"]
        except Exception:
            pass                            # unreadable / partial file: rebuilt and replaced below
    out = _animal(model, ind)
    try:
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / f".{key}.{os.getpid()}.{os.urandom(4).hex()}.tmp"
        with open(tmp, "wb") as fh:
            pickle.dump({"key": key, "out": out}, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, p)                  # atomic: concurrent batch processes never see a partial file
    except OSError:
        pass
    return out


# ---- disk cache of the per-animal rest result ----------------------------------------------------------------------------
# Where the factory already keeps its derived caches (factory.oceanbake: /var/tmp/sat3737-parts-cache/water_engine);
# WF_WELD15_CACHE = a directory overrides it, "off" / "0" / "" disables it. One file per animal and key (~5 MB); a new
# key (code / config / library change) writes new files, old ones are never read again (delete the folder to prune).
CACHE_DIR = "/var/tmp/sat3737-parts-cache/weld15"
CACHE_VERSION = "weld15-rest-v1"
CACHE_CODE = ("weld15.py", "weld.py", "blade.py")      # the code that builds the result from the hashed inputs below


_CODE_DIGEST = None


def _code_digest() -> str:
    """sha256 of CACHE_CODE, read once per process (at the first key, i.e. the code this process runs: a file edited
    while a run is going cannot label old geometry with the new code's key)."""
    global _CODE_DIGEST
    if _CODE_DIGEST is None:
        import hashlib
        from pathlib import Path
        here = Path(__file__).resolve().parent; h = hashlib.sha256()
        for f in CACHE_CODE:
            h.update(f.encode()); h.update((here / f).read_bytes())
        _CODE_DIGEST = h.hexdigest()
    return _CODE_DIGEST


def cache_dir():
    import os
    from pathlib import Path
    v = os.environ.get("WF_WELD15_CACHE")
    if v is not None and v.strip().lower() in ("", "0", "off", "none", "false"):
        return None
    return Path(v) if v else Path(CACHE_DIR)


def _feed(h, obj) -> None:
    """Hash any nesting of dicts / lists / tuples / numpy arrays / scipy sparse matrices / scalars, by content."""
    if isinstance(obj, np.ndarray):
        a = np.ascontiguousarray(obj)
        h.update(f"nd{a.dtype.str}{a.shape}".encode()); h.update(a.tobytes())
    elif isinstance(obj, dict):
        h.update(b"{")
        for k in sorted(obj, key=repr):
            _feed(h, k); _feed(h, obj[k])
        h.update(b"}")
    elif isinstance(obj, (list, tuple)):
        h.update(f"[{len(obj)}".encode())
        for v in obj:
            _feed(h, v)
        h.update(b"]")
    elif hasattr(obj, "tocsr") and hasattr(obj, "shape"):
        c = obj.tocsr()
        h.update(f"sp{c.shape}".encode()); _feed(h, c.data); _feed(h, c.indices); _feed(h, c.indptr)
    elif isinstance(obj, (np.generic, int, float, bool, str, bytes, type(None))):
        h.update(repr(obj.item() if isinstance(obj, np.generic) else obj).encode())
    else:
        raise TypeError(f"weld15 cache: cannot hash {type(obj).__name__}")


def cache_key(model, ind) -> str:
    """Content key of animal(model, ind): the code that builds it, every input it reads (the animal's own rest back,
    its local blade and individual spec, the library's blade templates, the weld region, the blade frame and anchors,
    the module constants) and the numeric library versions. Any change gives a new key, never a stale surface."""
    import hashlib
    import scipy
    from pathlib import Path
    h = hashlib.sha256()
    _feed(h, (CACHE_VERSION, 15, np.__version__, scipy.__version__))
    h.update(_code_digest().encode())
    mk = model.__dict__.get("_w15_model_key")
    if mk is None:
        hm = hashlib.sha256()
        W = model.weld
        _feed(hm, {k: v for k, v in W.__dict__.items()})
        _feed(hm, model._blade_lib)
        _feed(hm, (model.n_rig, float(model._weld_sign), str(model.kind)))
        mk = model._w15_model_key = hm.hexdigest()
    h.update(mk.encode())
    _feed(h, model._weld_key(ind))
    local = model._blade_local(ind)
    _feed(h, {k: v for k, v in local.items()})
    _feed(h, tuple(model._blade_frame_for(local)))
    _feed(h, np.asarray(model._blade_base_anchors(ind, local)))
    if ind is not None:
        _feed(h, np.asarray(model._posed_rig({"individual": ind})[: model.n_rig], float))
        try:
            spec = model.individual(ind)["spec"]
            _feed(h, spec.to_dict() if hasattr(spec, "to_dict") else repr(spec))
        except Exception:
            _feed(h, "no-spec")
    _feed(h, {k: v for k, v in sorted(globals().items()) if k.isupper() and isinstance(v, (int, float, str, bool, np.ndarray))})
    _feed(h, {k: v for k, v in sorted(vars(wd).items()) if k.isupper() and isinstance(v, (int, float, str, bool, np.ndarray))})
    _feed(h, {k: v for k, v in sorted(vars(bl).items()) if k.isupper() and isinstance(v, (int, float, str, bool, np.ndarray))})
    tag = str(ind) if isinstance(ind, str) else ("default" if ind is None else "inline")
    return f"{tag}_{h.hexdigest()[:32]}"


def _animal(model, ind) -> dict:
    W = model.weld; R2 = 2 * bl.NC; NC = bl.NC
    n0 = model.n_rig
    local = model._blade_local(ind)
    H = float(local["H"])
    frame = model._blade_frame_for(local)
    inp_model = wd.animal_inputs(W, local, frame, H)
    anchors = model._blade_base_anchors(ind, local)
    Ex_ind = W.ext(np.asarray(model._posed_rig({"individual": ind})[:n0], float)) if ind is not None else W.R_rest
    Ti, ti = bl.kabsch(W.R_rest[anchors], Ex_ind[anchors])
    B, along, up = frame
    B_i = Ti @ np.asarray(B, float) + ti; al_i = Ti @ along; up_i = Ti @ up
    ey = np.cross(up, along); ey = ey / np.linalg.norm(ey)
    ysgn = 1.0 if ey[1] > 0 else -1.0
    ey_i = Ti @ np.array([0.0, 1.0, 0.0])
    axes = (B_i, al_i, ey_i, up_i)
    N = W.base_normals(Ex_ind) * model._weld_sign
    bm = BaseMap(W, Ex_ind, N, axes)
    Vb = np.asarray(local["vertices"], float).copy()
    z0n = wd.H_FOOT * H
    # drape: the back's midline height under the blade (smoothed), full at the foot ring, none from z0 + H_DRAPE H up
    ag = np.linspace(Vb[:, 0].min() - 0.3 * H, Vb[:, 0].max() + 0.3 * H, 600)
    bmid = bm.z(np.stack([ag, np.zeros_like(ag)], 1))
    sg = 0.05 * H / (ag[1] - ag[0])
    kx = np.arange(-int(4 * sg) - 1, int(4 * sg) + 2)
    ker = np.exp(-0.5 * (kx / sg) ** 2); ker /= ker.sum()
    bmid = np.convolve(np.pad(bmid, len(kx) // 2, mode="edge"), ker, mode="valid")[: len(ag)]
    r0a = Vb[:R2, 0]
    inr = (ag >= r0a.min()) & (ag <= r0a.max())
    if DRAPE:
        u_b = max(0.0, float(bmid[inr].max()) - E_MAX * H)
        e = np.interp(Vb[:, 0], ag, bmid) - u_b
        x_ = np.clip((Vb[:, 2] - z0n) / (H_DRAPE * H), 0.0, 1.0)
        w = 1.0 - x_ ** 3 * (x_ * (6.0 * x_ - 15.0) + 10.0)       # smootherstep (C2): no shoulder where it ends
    else:
        # no shear: the blade keeps its template shape; raised only where the back comes within RAISE_CLEAR H of its
        # foot ring (calves; round 13 raised for burial)
        u_b = max(0.0, float(bmid[inr].max()) - (wd.H_FOOT - RAISE_CLEAR) * H)
        e = np.zeros(len(Vb)); w = np.zeros(len(Vb))
        bmid = np.full_like(bmid, u_b)
    Vd = Vb.copy(); Vd[:, 2] += u_b + e * w
    blade_rest = B_i + Vd[:, 0:1] * al_i + Vd[:, 1:2] * ey_i + Vd[:, 2:3] * up_i
    lv_pos = LOW_LEVELS[LOW_LEVELS >= -1e-9]
    low = low_sections(model, ind, lv_pos)
    Vpos = np.asarray(low["vertices"], float)[:-1].reshape(len(lv_pos), R2, 3)
    # below h = 0 the chords continue linearly (blade.make clamps its sunk rings to the foot's footprint: a cliff)
    neg = LOW_LEVELS[LOW_LEVELS < -1e-9]
    dV = (Vpos[5] - Vpos[0]) / (lv_pos[5] - lv_pos[0])
    Vneg = Vpos[0][None] + (neg[:, None, None]) * dV[None]
    Vneg[..., 2] = neg[:, None] * H
    Vneg[..., 1] = Vpos[0][None, :, 1] * np.ones((len(neg), 1))
    Vlow = np.concatenate([Vneg, Vpos])
    # below the foot ring the template's thickness (built for the old draped hump: half-width 0.4 H at h = 0) grows
    # at the blade's own rate at the ring, turning C2 to LOW_RATE (a fin that stands on the back, not a plinth)
    z0h = wd.H_FOOT
    tm = lambda h: 0.35 * np.clip(1.0 - h, 0.0, 1.35) ** 1.2 + 0.05
    R0 = 0.35 * 1.2 * (1.0 - z0h) ** 0.2
    tlow = tm(z0h) + off_draft(z0h - LOW_LEVELS, R0, LOW_RATE, LOW_T)
    tref = np.where(LOW_LEVELS >= 0.0, tm(LOW_LEVELS), tm(0.0))
    ratio = np.where(LOW_LEVELS < z0h, tlow / tref, 1.0)
    Vlow[..., 1] *= ratio[:, None]
    Vlow = _nest(Vlow, LOW_LEVELS, wd.H_FOOT, H)
    nl = len(LOW_LEVELS)
    if np.abs(Vlow[int(np.argmin(np.abs(LOW_LEVELS - wd.H_FOOT)))] - Vb[:R2]).max() > 1e-6 * H + 1e-6:
        raise RuntimeError("weld15: the low sections do not meet the blade's foot ring")
    fin = Fin(Vlow[..., :2], LOW_LEVELS * H, H, ag, bmid)
    k = K_BACK * H
    loop_u0 = bm.coords(Ex_ind[W.loop_rep])[:, :2]
    ext_lo = np.maximum(Vlow[..., :2].reshape(-1, 2).min(0) - 0.6 * H, loop_u0.min(0) - 2.0)
    ext_hi = np.minimum(Vlow[..., :2].reshape(-1, 2).max(0) + 0.6 * H, loop_u0.max(0) + 2.0)
    fin.make_grid(ext_lo, ext_hi, max(0.4, 0.012 * H))

    # the blade's lowest rings, where the back's smooth union reaches them: each vertex moved along its section's
    # outward normal onto the iso-surface (rings far above the back stay exactly as they are)
    nlb = (len(Vd) - 1) // R2
    Rg = Vd[:-1].reshape(nlb, R2, 3)
    moved = 0
    for kr in range(nlb):
        Pk = Rg[kr, :, :2]
        if Rg[kr, :, 2].min() - bm.z(Pk).max() > 1.2 * k:
            break
        e1 = np.roll(Pk, -1, 0) - Pk; e0 = Pk - np.roll(Pk, 1, 0)
        nn = np.stack([e1[:, 1], -e1[:, 0]], 1) / (np.linalg.norm(e1, axis=1, keepdims=True) + 1e-12) + \
             np.stack([e0[:, 1], -e0[:, 0]], 1) / (np.linalg.norm(e0, axis=1, keepdims=True) + 1e-12)
        nn /= np.linalg.norm(nn, axis=1, keepdims=True) + 1e-12
        if _poly_dist(Pk + 1e-3 * H * nn, Pk).mean() < 0:
            nn = -nn
        up_r = Rg[min(kr + 1, nlb - 1), :, :2] if kr + 1 < nlb else Pk
        dzk = max(float((Rg[min(kr + 1, nlb - 1), :, 2] - Rg[kr, :, 2]).mean()), 1e-3 * H)
        sig = np.clip(_poly_dist(Pk, up_r) / dzk, -3.0, 30.0)
        Nk = np.sqrt(1.0 + sig * sig)
        zk = Rg[kr, :, 2]
        # along the blade's own surface normal (horizontal outward normal tilted up by the wall's slope): a vertex on
        # a nearly flat ramp moves up, one on a steep wall moves out (a horizontal ray was ill-conditioned on ramps)
        uh = nn / Nk[:, None]; uz = sig / Nk

        def gb(tv):
            P2 = Pk + tv[:, None] * uh
            return smin3((zk + tv * uz - bm.z(P2)) * np.clip(bm.cfac(P2), 0.2, 1.0), tv, k)
        need = gb(np.zeros(R2)) < -1e-6 * H
        if not need.any():
            continue
        lo = np.zeros(R2); hi = np.full(R2, 0.6 * H)
        for _ in range(36):
            mid = 0.5 * (lo + hi); pos = gb(mid) > 0
            hi = np.where(pos, mid, hi); lo = np.where(pos, lo, mid)
        tv = np.where(need, 0.5 * (lo + hi), 0.0)
        Rg[kr, :, :2] = Pk + tv[:, None] * uh
        Rg[kr, :, 2] = zk + tv * uz
        moved += int(need.sum())
    Vd[:-1] = Rg.reshape(-1, 3)
    blade_rest = B_i + Vd[:, 0:1] * al_i + Vd[:, 1:2] * ey_i + Vd[:, 2:3] * up_i

    def ztop(P2):
        return z0n + fin.bm(P2[:, 0])

    def prof(P2):
        zX = bm.z(P2); cX = bm.cfac(P2)
        L, _ = fin.lift(P2, zX, cX, k, ztop(P2), grid=True)
        return zX + L
    loop_u = bm.coords(Ex_ind[W.loop_rep])[:, :2]
    ring_u = Vd[:R2, :2]
    XYu, einp = layout15(W, ring_u, loop_u, prof, H)
    nI = R2 + W.n_int
    # landmarks: front / back base on the midline (round 13's template positions)
    o = local["outline"]; Lv, te_s, le_s = np.asarray(o["levels"]), np.asarray(o["te"]), np.asarray(o["le"])
    hh = np.linspace(0.0, wd.H_FOOT, 61)
    te_h, le_h = np.interp(hh, Lv, te_s), np.interp(hh, Lv, le_s)
    a_lo, a_hi = float(te_h.min()), float(le_h.max()); a_ft, a_fl = float(te_h[-1]), float(le_h[-1])
    Rcm = wd.SKIRT_R * H
    fb_a = max(a_hi * H, a_fl * H + wd.LM_MIN * Rcm) if a_hi > a_fl + 1e-6 else a_fl * H + wd.LM_MIN * Rcm
    bb_a = min(a_lo * H, a_ft * H - wd.LM_MIN * Rcm) if a_lo < a_ft - 1e-6 else a_ft * H - wd.LM_MIN * Rcm
    # every patch vertex: on its chart point (pushed just outside the foot ring's footprint where the layout put it on
    # or inside the knife-thin edges), lifted along up onto the iso-surface (a height field: no two rows can cross)
    XYu = XYu.copy()
    st = np.arange(R2, nI)
    XYu, ins = _push_out(XYu, st, ring_u, H)
    n_notok = int(ins.sum())
    Tq = XYu[W.cfaces]
    ar2 = (Tq[:, 1, 0] - Tq[:, 0, 0]) * (Tq[:, 2, 1] - Tq[:, 0, 1]) - (Tq[:, 1, 1] - Tq[:, 0, 1]) * (Tq[:, 2, 0] - Tq[:, 0, 0])
    chart_folds = int((np.sign(ar2) != np.sign(np.median(ar2))).sum())
    pts_u = np.concatenate([XYu[:nI], [[fb_a, 0.0], [bb_a, 0.0]]])
    q = bm.inv(pts_u)
    t_id0 = np.zeros(len(W.c_t)); t_id0[W.c_ids] = W.c_t
    # Newton refinement of the base point under each target (the interpolated inverse is ~mm off; the knife-thin
    # trailing edge is thinner than that)
    for _ in range(3):
        tri, bary, inside = wd.locate(W, q)
        Xq, _n = wd.base_points(W, Ex_ind, N, tri, bary)
        Uq = bm.coords(Xq)[:, :2]
        e = 0.05
        Jc = []
        for dq in ((e, 0.0), (0.0, e)):
            tq, bq, _ = wd.locate(W, q + np.array(dq))
            Jc.append((bm.coords(wd.base_points(W, Ex_ind, N, tq, bq)[0])[:, :2] - Uq) / e)
        Jm = np.stack(Jc, 2)                         # (n, 2 [U], 2 [q])
        res = pts_u - Uq
        det = Jm[:, 0, 0] * Jm[:, 1, 1] - Jm[:, 0, 1] * Jm[:, 1, 0]
        okd = np.abs(det) > 1e-6
        Jm[~okd] = np.eye(2)
        stp = np.linalg.solve(Jm, res[..., None])[..., 0]
        stp[~okd] = 0.0
        stp *= np.minimum(1.0, 2.0 / (np.linalg.norm(stp, axis=1, keepdims=True) + 1e-12))
        stp[np.concatenate([t_id0[:nI] > 0.85, [False, False]])] = 0.0
        q = q + stp
    tri, bary, inside = wd.locate(W, q)
    t_id = np.zeros(len(W.c_t)); t_id[W.c_ids] = W.c_t
    wph = np.concatenate([1.0 - wd.smoothstep((t_id[:nI] - wd.PHONG_T0) / (1.0 - wd.PHONG_T0)), [1.0, 1.0]])
    wsm = 1.0 - wd.smoothstep((t_id[:nI] - 0.9) / 0.1)
    base, nb = wd.base_points(W, Ex_ind, N, tri, bary, wph)
    base = base.copy()
    # smoothed along up only: the patch is a height field over the up-chart and its chart points must stay where the
    # layout put them (smoothing along the base normal moved rows by the foot into the foot's footprint)
    base[:nI] = smooth_up(W, base[:nI], Ex_ind[W.loop_rep], nb[:nI], up_i, wsm)
    abz = bm.coords(base)
    # every patch point exactly over its chart target (where the base map is ill-conditioned - the old fin's harmonic
    # fill under the trailing edge - the Newton step can leave it cm off): the base height carried to the target by the
    # smooth back map's own slope
    tgt = pts_u
    zt = abz[:, 2] + (bm.z(tgt) - bm.z(abz[:, :2]))
    zt[nI:] = abz[nI:, 2]; tgt = tgt.copy(); tgt[nI:] = abz[nI:, :2]
    cX = nb @ up_i
    L, ok = fin.lift(tgt, zt, cX, k, ztop(tgt))
    d = _poly_dist(tgt, Vb[:R2, :2])
    rest = B_i[None] + tgt[:, 0:1] * al_i + tgt[:, 1:2] * ey_i + (zt + L)[:, None] * up_i[None, :]
    far = (L <= 0.0) & (np.linalg.norm(tgt - abz[:, :2], axis=1) < 1e-3)
    rest[far] = base[far]
    rest[:R2] = blade_rest[:R2]
    L = L.copy(); L[:R2] = (blade_rest[:R2] - base[:R2]) @ up_i
    uv = (W.uv_ext[W.base_faces[tri]] * bary[..., None]).sum(1)
    P2 = np.zeros((len(W.c_t), 2)); P2[:nI] = q[:nI]; P2[nI:] = W.loop2d
    ringd = wd._densify_ring(ring_u, 6)
    dd = np.sqrt(((pts_u[:, None, :] - ringd[None]) ** 2).sum(-1))
    d_foot = dd.min(1)
    # posing data: heights above the back, stiffness, anchors (dn-chart) and residual indices
    hgt_p = np.clip(L, 0.0, None)
    s_p = smoothstep(hgt_p / (STIFF_H * H))
    C = q[:R2].mean(0)
    col = np.concatenate([np.arange(len(blade_rest) - 1) % R2, [0]])
    hgt_b = ((blade_rest - base[col]) * up_i).sum(1)
    hgt_b[-1] = max(hgt_b[-1], hgt_b[:-1].max())
    s_b = smoothstep(np.clip(hgt_b, 0.0, None) / (STIFF_H * H))
    anc_p = q * (1 - s_p[:, None]) + C[None] * s_p[:, None]
    anc_b = q[col] * (1 - s_b[:, None]) + C[None] * s_b[:, None]
    nb_ = len(blade_rest)
    k_ = np.arange(nb_ - 1) // R2
    g = {"P2": P2, "tri": tri, "bary": bary, "D": L, "d_foot": d_foot, "col_near": np.argmin(dd, 1) // 6,
         "wcol": None, "off": rest - base, "n_rest": nb, "rest": rest, "uv": uv, "a": q[:, 0], "wph": wph, "wsm": wsm,
         "loop_tri": None, "bmix": np.zeros(nI), "inside": inside, "loop_D": L[nI - len(loop_u):nI], "q": q,
         "base": base, "wr": np.zeros(len(L)), "ok": ok, "n_notok": n_notok, "chart_folds": chart_folds, "blade_moved": moved, "d0": d}
    out = {"local": local, "inp": inp_model, "g": g, "blade_rest": blade_rest, "fade": np.zeros(nb_), "col": col,
           "anchors": anchors, "levels": np.concatenate([wd.LEVELS_W[k_], [wd.LEVELS_W[-1] + 0.02]]), "Ex_rest": Ex_ind,
           "blade_a": W.chart(blade_rest)[:, 0], "lift_dir": up_i, "frame_i": (B_i, al_i, up_i), "ey_i": ey_i,
           "pose15": {"s_p": s_p, "s_b": s_b, "anc_p": anc_p, "anc_b": anc_b, "C": C, "hgt_p": hgt_p, "hgt_b": hgt_b,
                      "qcol": q[col]}, "u_b": u_b,
           "drape": (ag, bmid), "einp": einp, "XYu": XYu, "fin": fin if DEBUG else None, "bm": None}
    return out


def smooth_up(W, P, loopP, n, up, w):
    """round 13's Taubin smoothing of the base along its normals (the rig's creases go), applied as the equivalent move
    along up (the same tangent plane), so every point keeps its place in the up-chart (a pure height smoothing along
    up rippled the rows on the sloped back; a move along the normal pushed rows by the foot into its footprint)."""
    Ps = wd.smooth_base(W, P, loopP, n, w)
    nn = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
    d = ((Ps - P) * nn).sum(1)
    c = np.clip(nn @ up, 0.3, None)
    return P + (d / c)[:, None] * up[None, :]


# ---- posing -----------------------------------------------------------------------------------------------------------
class PoseField:
    """Model-level node lattice over the region (dn-chart of the model's rest), Kabsch neighbourhoods of the extended
    back vertices (the region's base and the rig's top faces near its loop)."""

    def __init__(self, model):
        W = model.weld
        Cx = W.chart(W.R_rest)
        v = np.unique(W.base_faces)
        kept = np.asarray(W.kept_faces)
        ph = W.phi(Cx)
        Rr = W.R_rest
        tri = Rr[kept]; fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        fz = (fn @ W.e_n) / (np.linalg.norm(fn, axis=1) + 1e-12) * model._weld_sign
        near = (ph[kept] < 1.6 ** wd.SE_N - 1.0).all(1) & (fz > 0.2)
        v = np.unique(np.concatenate([v, kept[near].ravel()]))
        self.v = v
        P = Cx[v]
        lo, hi = P.min(0) - 5.0, P.max(0) + 5.0
        gx, gy = np.meshgrid(np.arange(lo[0], hi[0] + NODE_CM, NODE_CM), np.arange(lo[1], hi[1] + NODE_CM, NODE_CM), indexing="ij")
        nodes = np.stack([gx.ravel(), gy.ravel()], 1)
        from scipy.spatial import cKDTree
        tr = cKDTree(P)
        nbrs = tr.query_ball_point(nodes, NODE_RAD)
        keep = [i for i, nn in enumerate(nbrs) if len(nn) >= 12]
        self.nodes = nodes[keep]
        mx = max(len(nbrs[i]) for i in keep)
        idx = np.zeros((len(keep), mx), int); wt = np.zeros((len(keep), mx))
        for r, i in enumerate(keep):
            nn = np.asarray(nbrs[i]); idx[r, :len(nn)] = v[nn]
            wt[r, :len(nn)] = np.exp(-0.5 * (np.linalg.norm(P[nn] - nodes[i], axis=1) / (0.5 * NODE_RAD)) ** 2)
        self.idx, self.wt = idx, wt
        self.ntree = cKDTree(self.nodes)
        self.key = (NODE_CM, NODE_RAD)

    def weights(self, Q):
        """Sparse blend weights (n, K) of chart points Q over the nodes."""
        from scipy import sparse
        nn = self.ntree.query_ball_point(Q, 3.0 * NODE_SIG)
        rows, cols, vals = [], [], []
        for i, ks in enumerate(nn):
            if not ks:
                ks = [int(self.ntree.query(Q[i])[1])]
            ks = np.asarray(ks)
            w = np.exp(-0.5 * (np.linalg.norm(self.nodes[ks] - Q[i], axis=1) / NODE_SIG) ** 2)
            w /= w.sum()
            rows += [i] * len(ks); cols += list(ks); vals += list(w)
        return sparse.csr_matrix((vals, (rows, cols)), shape=(len(Q), len(self.nodes)))

    def transforms(self, X0, X1):
        P = X0[self.idx]; Q = X1[self.idx]; w = self.wt[..., None]
        sw = w.sum(1)
        cp = (w * P).sum(1) / sw; cq = (w * Q).sum(1) / sw
        Hm = np.einsum("kmi,kmj->kij", w * (P - cp[:, None]), Q - cq[:, None])
        U, _, Vt = np.linalg.svd(Hm)
        d = np.sign(np.linalg.det(np.einsum("kji,kjl->kil", Vt, np.transpose(U, (0, 2, 1)))))
        D = np.zeros_like(Hm); D[:, 0, 0] = 1; D[:, 1, 1] = 1; D[:, 2, 2] = d
        Rm = np.einsum("kji,kjl,klm->kim", Vt, D, np.transpose(U, (0, 2, 1)))
        t = cq - np.einsum("kij,kj->ki", Rm, cp)
        return Rm, t


def wkabsch(P, Q, w):
    """Weighted rigid fit Q ~ R P + t."""
    w = (w / w.sum())[:, None]
    cp = (w * P).sum(0); cq = (w * Q).sum(0)
    U, _, Vt = np.linalg.svd(((P - cp) * w).T @ (Q - cq))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, cq - R @ cp


def apply_field(Wsp, Rm, t, P):
    M = (Wsp @ Rm.reshape(len(Rm), 9)).reshape(-1, 3, 3)
    tt = Wsp @ t
    return np.einsum("nij,nj->ni", M, P) + tt


def _smooth_resid(model, A, p15, r, Ex, Rm, t, Wq, scale, H):
    """round 16: the back's residual (posed base minus the rigid carrier's image of the rest base) smoothed over the patch
    graph near the foot (Jacobi passes of the uniform Laplacian, the loop's own residual as the boundary): the posed base
    layer by the foot is uneven at the cm scale (rows 1-2 cm apart over 9 cm rig faces bent by the pose); its smooth part
    (the bend of the back) is kept. Zero at rest."""
    W = model.weld; R2 = 2 * bl.NC; nI = R2 + W.n_int
    key = (RSM_D, RSM_DMIN)
    if p15.get("rsm_key") != key:
        from scipy import sparse
        cf = np.asarray(W.cfaces)
        E = np.concatenate([cf[:, [0, 1]], cf[:, [1, 2]], cf[:, [2, 0]]])
        E = np.unique(np.sort(E, 1), axis=0)
        n = int(cf.max()) + 1
        Adj = sparse.coo_matrix((np.ones(2 * len(E)), (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])), shape=(n, n)).tocsr()
        Adj.data[:] = 1.0
        deg = np.asarray(Adj.sum(1)).ravel()
        Mavg = sparse.diags(1.0 / np.maximum(deg, 1)) @ Adj
        wsm = 1.0 - smoothstep(A["g"]["d_foot"][:nI] / max(RSM_D * H, RSM_DMIN))
        p15["rsm"] = (Mavg.tocsr(), wsm, n)
        p15["rsm_key"] = key
    Mavg, wsm, n = p15["rsm"]
    loop = Ex[W.loop_rep]
    rl = loop - apply_field(Wq[:1], Rm, t, A["Ex_rest"][W.loop_rep] * scale) if Wq.shape[0] else loop * 0
    X = np.zeros((n, 3)); X[:nI] = r[:nI]; X[nI:] = rl[: n - nI]
    lam = (RSM_LAM * wsm)[:, None]
    for _ in range(RSM_IT):
        X[:nI] = X[:nI] + lam * ((Mavg @ X)[:nI] - X[:nI])
    out = r.copy(); out[:nI] = X[:nI]
    return out


def _posed(model, A, rig_posed, scale, mode, gate_open=False):
    global LAST
    W = model.weld; R2 = 2 * bl.NC; nI = R2 + W.n_int
    pf = model.__dict__.get("_pose15")
    if pf is None or getattr(pf, "key", None) != (NODE_CM, NODE_RAD):
        pf = model._pose15 = PoseField(model)
    p15b = A["pose15"]
    H = float(A["local"]["H"])
    key = (STIFF_H, NODE_SIG, NODE_CM, NODE_RAD, mode, FIT_H, FIT_SIG, FIT_FLOOR)
    # per-mode cache of the posing weights (the "gate" posing switches between "foot" and "fit" per whale)
    cc = p15b.setdefault("_bykey", {})
    if key not in cc:
        cc[key] = {k_: v_ for k_, v_ in p15b.items() if k_ not in ("_bykey", "key")}
    p15 = cc[key]
    if p15.get("key") != key:
        g = A["g"]
        if mode == "rigid":
            sp = np.ones(len(p15["hgt_p"])); sb = np.ones(len(p15["hgt_b"]))
        else:
            sp = smoothstep(np.clip(p15["hgt_p"], 0, None) / (STIFF_H * H)); sb = smoothstep(np.clip(p15["hgt_b"], 0, None) / (STIFF_H * H))
        p15["s_p"], p15["s_b"] = sp, sb
        p15["anc_p"] = g["q"] * (1 - sp[:, None]) + p15["C"][None] * sp[:, None]
        p15["anc_b"] = p15["qcol"] * (1 - sb[:, None]) + p15["C"][None] * sb[:, None]
        if mode == "foot":
            # one transform (the fit at the fin base) carries every offset; the patch follows the posed back fully
            # (posed base + that rotation of its rest offset: a height field over the posed back along one direction);
            # the blade from its foot ring follows its column's foot, turning rigid over STIFF_H H above the ring
            R2_ = 2 * bl.NC
            hb = p15["hgt_b"]; col_ = np.concatenate([np.arange(len(hb) - 1) % R2_, [0]])
            sb = smoothstep(np.clip(hb - hb[col_], 0, None) / (STIFF_H * H)); sb[-1] = 1.0
            sp = np.zeros(len(p15["hgt_p"]))
            p15["s_p"], p15["s_b"] = sp, sb
            p15["anc_p"] = np.repeat(p15["C"][None], len(sp), 0); p15["anc_b"] = np.repeat(p15["C"][None], len(sb), 0)
        if mode == "foot3":
            # blade: anchored at its column's foot, moving to the fin base centre over STIFF_H H above the foot ring
            # (rotation and residual of the back there, turning rigid with height); patch: anchored at its own base
            # point by the foot, at the centre beyond FOOT_D H (one direction for the hump's offsets, which keeps it a
            # height field over the posed back) - the two agree at the foot ring
            R2_ = 2 * bl.NC
            hb = p15["hgt_b"]; col_ = np.concatenate([np.arange(len(hb) - 1) % R2_, [0]])
            sb = smoothstep(np.clip(hb - hb[col_], 0, None) / (STIFF_H * H)); sb[-1] = 1.0
            wpa = smoothstep(g["d_foot"] / (FOOT_D * H)); wpa[:R2_] = 0.0
            p15["s_p"], p15["s_b"] = np.zeros(len(wpa)), sb
            p15["anc_p"] = g["q"] * (1 - wpa[:, None]) + p15["C"][None] * wpa[:, None]
            p15["anc_b"] = p15["qcol"] * (1 - sb[:, None]) + p15["C"][None] * sb[:, None]
            p15["Wap"] = pf.weights(p15["anc_p"]); p15["Wab"] = pf.weights(p15["anc_b"])
            p15["Wc"] = pf.weights(np.repeat(p15["C"][None], R2_, 0))
        if mode == "rigidT":
            # one transform for every offset (the fit at the fin base); the back's own motion enters through the residual
            # (posed base minus that transform's image of the rest base), fully at the back, fading with height
            sp = smoothstep(np.clip(p15["hgt_p"], 0, None) / (STIFF_H * H)); sb = smoothstep(np.clip(p15["hgt_b"], 0, None) / (STIFF_H * H))
            p15["s_p"], p15["s_b"] = sp, sb
            p15["anc_p"] = np.repeat(p15["C"][None], len(sp), 0); p15["anc_b"] = np.repeat(p15["C"][None], len(sb), 0)
        if mode == "fit":
            # round 16: as "foot" for the patch; the blade carried by ONE rigid transform fitted to its own posed foot
            # ring (the patch's row 0), weighted by where the blade's upper part stands over the foot (a long low ramp
            # ahead counts little), plus each column's residual fading over STIFF_H H above its foot
            R2_ = 2 * bl.NC
            hb = p15["hgt_b"]; col_ = np.concatenate([np.arange(len(hb) - 1) % R2_, [0]])
            sb = smoothstep(np.clip(hb - hb[col_], 0, None) / (STIFF_H * H)); sb[-1] = 1.0
            p15["s_p"], p15["s_b"] = np.zeros(len(p15["hgt_p"])), sb
            p15["anc_p"] = np.repeat(p15["C"][None], len(p15["hgt_p"]), 0); p15["anc_b"] = p15["anc_p"][:len(sb)]
            Xb = A["blade_rest"]; al_ = A["frame_i"][1]
            upv = (hb - hb[col_]) > FIT_H * H
            d_ = ((Xb[:R2_] @ al_)[:, None] - (Xb[upv] @ al_)[None]) / (FIT_SIG * H)
            wb = np.exp(-0.5 * d_ ** 2).sum(1)
            p15["wfit"] = wb / wb.max() + FIT_FLOOR
        p15["Wq"] = pf.weights(g["q"])
        p15["Wap"] = pf.weights(p15["anc_p"])
        p15["Wab"] = pf.weights(p15["anc_b"])
        p15["key"] = key
    Ex = W.ext(rig_posed)
    N = W.base_normals(Ex) * model._weld_sign
    g = A["g"]
    base_p, n_p = wd.base_points(W, Ex, N, g["tri"], g["bary"], g["wph"])
    base_p = base_p.copy()
    Ra, _ = bl.kabsch(A["Ex_rest"][A["anchors"]] * scale, Ex[A["anchors"]])
    base_p[:nI] = smooth_up(W, base_p[:nI], Ex[W.loop_rep], n_p[:nI], Ra @ A["lift_dir"], g["wsm"])
    Rm, t = pf.transforms(A["Ex_rest"] * scale, Ex)
    Wq = p15["Wap"] if mode in ("rigidT", "foot", "foot3", "fit") else p15["Wq"]
    r = base_p - apply_field(Wq, Rm, t, g["base"] * scale)
    gate = 0.0
    if mode == "fit":
        # the bend measure: how far the posed foot ring leaves the one rigid transform at the base centre (its residual
        # along that transform's up, about the blade-weighted mean, largest over the ring, x H); below GATE_M0 the
        # round-15 "foot" posing is kept exactly, above GATE_M1 the round-16 posing is used, smoothstep between
        M = (p15["Wap"][:1] @ Rm.reshape(len(Rm), 9)).reshape(3, 3)
        u = M @ A["lift_dir"]; u = u / np.linalg.norm(u)
        wf = p15["wfit"]
        ru = r[:R2] @ u
        bend = float(np.abs(ru - (wf * ru).sum() / wf.sum()).max() / (H * scale))
        gate = 1.0 if (gate_open or GATE_M1 <= 0) else float(smoothstep((bend - GATE_M0) / (GATE_M1 - GATE_M0)))
        LAST = (bend, gate)
        if gate > 0.0 and RSM_IT > 0:
            r = r + gate * (_smooth_resid(model, A, p15, r, Ex, Rm, t, Wq, scale, H) - r)
    rbar = r[:R2].mean(0)
    s_p = p15["s_p"][:, None]; s_b = p15["s_b"][:, None]
    P = apply_field(p15["Wap"], Rm, t, g["rest"] * scale) + (1 - s_p) * r + s_p * rbar[None]
    if mode == "foot3":
        rb = base_p[A["col"]] - apply_field(p15["Wab"], Rm, t, g["base"][A["col"]] * scale)
        rbar = (base_p[:R2] - apply_field(p15["Wc"], Rm, t, g["base"][:R2] * scale)).mean(0)
        Bp = apply_field(p15["Wab"], Rm, t, A["blade_rest"] * scale) + (1 - s_b) * rb + s_b * rbar[None]
    elif mode == "fit":
        Bp = apply_field(p15["Wab"], Rm, t, A["blade_rest"] * scale) + (1 - s_b) * r[A["col"]] + s_b * rbar[None]
        if gate > 0.0:
            # the blade carried by one rigid fit to its own posed foot ring (weighted by where its upper part stands),
            # each column's residual to it fading over STIFF_H H above the foot
            Rf, tf = wkabsch(A["blade_rest"][:R2] * scale, P[:R2], p15["wfit"])
            rho = P[:R2] - ((A["blade_rest"][:R2] * scale) @ Rf.T + tf)
            Bfit = (A["blade_rest"] * scale) @ Rf.T + tf + (1 - s_b) * rho[A["col"]]
            Bp = Bp + gate * (Bfit - Bp)
    else:
        Bp = apply_field(p15["Wab"], Rm, t, A["blade_rest"] * scale) + (1 - s_b) * r[A["col"]] + s_b * rbar[None]
    Bp[:R2] = P[:R2]
    return model._weld_assemble(Ex, Bp, P)



def _crossings(model, V):
    """weld_check's intersection count of a posed weld (the same faces: weld + rig faces near the loop, pairs sharing a
    position skipped)."""
    from factory import weld_check as wc
    ctx = model.__dict__.get("_weld15_ctx")
    if ctx is None:
        ctx = model._weld15_ctx = wc.Ctx(model)
    _, pid = np.unique(np.round(V, 4), axis=0, return_inverse=True)
    return wc.self_intersections(V, ctx.F, pid.ravel())


def posed(model, A, rig_posed, scale):
    """The posed weld. STIFF_MODE "gate" (round 16): round 15's "foot" posing; only if that self-intersects (weld_check's
    test) the whale is posed again with "fit" (SECOND_GATE_OPEN: bend gate fully open) and the result with fewer
    crossings is returned. Every posing leaves model._weld_pose_info = {"mode": "foot" | "r16", "intersects": bool,
    "n_foot": crossings of the "foot" try, "n_r16": crossings of the second try or None, "t": seconds (test, re-pose +
    test)}; other modes leave {"mode": STIFF_MODE, "intersects": None, ...} (no test)."""
    import time
    if STIFF_MODE != "gate":
        V = _posed(model, A, rig_posed, scale, STIFF_MODE)
        model._weld_pose_info = {"mode": STIFF_MODE, "intersects": None, "n_foot": None, "n_r16": None, "t": (0.0, 0.0)}
        return V
    V = _posed(model, A, rig_posed, scale, "foot")
    t0 = time.perf_counter()
    n = _crossings(model, V)
    t1 = time.perf_counter()
    info = {"mode": "foot", "intersects": n > 0, "n_foot": int(n), "n_r16": None, "t": (t1 - t0, 0.0)}
    if n > 0:
        V2 = _posed(model, A, rig_posed, scale, "fit", gate_open=bool(SECOND_GATE_OPEN))
        n2 = _crossings(model, V2)
        info["n_r16"] = int(n2)
        if n2 < n:
            V = V2
            info["mode"] = "r16"; info["intersects"] = n2 > 0
        info["t"] = (t1 - t0, time.perf_counter() - t1)
    model._weld_pose_info = info
    return V
