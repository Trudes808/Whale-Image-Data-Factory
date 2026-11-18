"""The whale behind a run: which mesh is rendered and how it is posed, with the canonical body frame
kept on the K9239 mesh so labels, the identity atlas and the review app stay comparable.

    model = WhaleModel.from_config(run_config["config"])
    model.canonical      # (N, 3) cm: every rendered vertex's point in the K9239 canonical frame
    model.faces, model.groups
    geometry = model.posed(shot)      # (N, 3) cm in the whale frame, scaled and bent for the shot

Kinds:
    k9239        the static CC-BY mesh (run01-run04 so far): rest = canonical; bends from factory.deform
    humpback4k   the purchased skinned model: bends from its dive/breach clip (shot["pose"]["clip_t"]),
                 fins still from factory.deform; canonical = rest pose aligned slice by slice (length,
                 height, width along the body) to the K9239 mesh
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

from factory.skinned import MODEL_TO_PROJECT as MODEL_TO_PROJECT_

from factory import deform, poses, project

REPO_ROOT = Path(__file__).resolve().parents[1]
K9239_OBJ = REPO_ROOT / "third_party_assets" / "humpback_k9239" / "converted" / "SM_Humpback.obj"
HUMPBACK4K_GLTF = REPO_ROOT / "third_party_assets" / "humpback_4k" / "src_gltf" / "scene.gltf"
# The dorsal_fin_front label: distance from the fin tip in units of the tip -> back-junction distance, along the dorsal
# midline ahead of the blade. The researcher's definition of the fin front is the front of the whole fin / hump, where
# its leading slope flattens into the back (not the blade's root); on 200 labelled real photos (2024-12-18, 2025-01-02,
# 2026-11-02) that point sits at a median 2.2 x |tip - back junction| from the tip (IQR 1.6-2.8, ~80 deg from the
# tip -> back direction). This asset has no hump (ahead of the blade the ridge runs straight on), so the label is placed
# by the ratio; 2.0 in the curled rest frame = ~2.2 in the levelled straight clip poses the renders use.
DORSAL_FRONT_RATIO = float(os.environ.get("WF_DORSAL_FRONT_RATIO", 2.0))

# Keyframes (clip seconds) per pose category, hand-picked from the clip (see data/renders/h4k_test3/keyframes
# sheet): a few body shapes each, never an arbitrary frame of the breach sequence. The category's own
# pitch/roll/yaw and clearance go on top; for the tail-up categories the pitch is reduced by the frame's own
# fluke lift (shotlist.make_shot), so a lifted fluke plus a nose-down pitch does not fold past vertical.
#   0.0 / 0.3 / 11.0 / 12.0 / 12.6 / 17.0 straight body (breathing, resting, static poses)
#   0.6 / 0.8 / 13.8 / 14.2 / 24.5 / 25.5 back rolling through, tail stock rising
#   1.2 / 1.4 / 1.6 / 1.9 / 2.2 the dive: fluke lifting then vertical
#   2.9 / 7.4 / 15.6 / 16.2 / 16.6 straight or head-up body for breaches and spyhops
STRAIGHT = [0.0, 0.3, 11.0, 12.0, 12.6, 17.0]
BACK = [0.6, 0.8, 13.8, 14.2, 24.5, 25.5]
DIVE = [1.2, 1.4, 1.6, 1.9]
KEYFRAMES = {
    # the surfacing categories share the straight-ish frames of the clip's start; their hump is procedural (shotlist.PROCEDURAL_POSES)
    "logging": [0.2, 0.3, 0.35, 0.45], "blow": [0.2, 0.3, 0.35, 0.45], "blow_sliver": [0.2, 0.3, 0.35, 0.45],
    "awash": [0.2, 0.3, 0.35, 0.45], "submerged": [0.0, 12.0, 13.4],
    "arch": [0.3, 0.6], "peduncle": [0.8, 1.0, 13.8, 25.5], "tail_stock": [0.8, 1.0, 13.8, 25.5],
    "back_arch": [0.2, 0.3, 0.35, 0.45], "surface": [0.2, 0.3, 0.35, 0.45],     # straight-ish frames from the start of the clip; the hump is procedural (shotlist)
    "fluke_up": [0.8, 1.0, 1.2, 1.4, 1.6, 1.9, 2.2], "fluke_edge": [1.6, 1.9, 2.2], "fluke_down": [1.0, 1.2, 1.4, 1.6],   # the whole dive; the fluke curl (shotlist) adds the tips-down phase
    "breach": [2.9, 7.4, 16.2, 16.6],          # 15.6 tears the dorsal-fin base seam in the asset's skinning
    "pec_up": [0.0, 0.3, 12.0], "fin_tip": [0.0, 0.3, 12.0], "roll_belly": [0.0, 0.3, 12.0],
    "spyhop": [0.0, 2.9, 16.6], "spyhop_low": [0.0, 2.9, 16.6],
    # run06 realism categories (their clip time is set by factory.realism; these are the fallbacks)
    "fluke_rollout": [0.8, 1.0], "fluke_sinking": [1.4, 1.6], "pec_slap": [0.0, 0.3, 12.0], "head_rise": [0.0, 0.3, 12.0],
    "tail_slap": [4.0, 5.0], "peduncle_throw": [0.0, 0.3, 12.0],
}
KEYFRAME_JITTER_S = 0.05
# run06 per-pixel part taxonomy (WhaleModel.part_groups; postprocess parts/<id>.png): 0 background
PART_IDS = {"fluke": 1, "dorsal": 2, "pectoral": 3, "head": 4, "body": 5}
DORSAL_HUMP_HALF_WIDTH_CM = 35.0      # the hump under the blade: this far either side of the midline (unit-size rest mesh)
DORSAL_HUMP_DEPTH_CM = 12.0           # ... and down to this far below the back-base -> front-base line
HEAD_BEHIND_BLOWHOLE_CM = 30.0        # the head ends on a cross plane this far behind the blowhole
CLIP_WINDOWS = {k: [(t, t) for t in v] for k, v in KEYFRAMES.items()}     # kept for callers that list windows
REST_FRACTION: dict = {}          # the bind pose is arched: no category samples it


def sample_clip_time(rng: np.random.Generator, category: str) -> float:
    """One of the category's keyframes, with a small jitter."""
    frames = KEYFRAMES.get(category, STRAIGHT)
    t = frames[int(rng.integers(len(frames)))] + float(rng.uniform(-KEYFRAME_JITTER_S, KEYFRAME_JITTER_S))
    return round(max(0.0, t), 3)


def _profile(v: np.ndarray, bins: int = 48, half_width_cm: float = 40.0) -> dict:
    """Top, bottom (near the centre line) and half width per x-bin of a mesh in the project frame."""
    x = v[:, 0]; edges = np.linspace(x.min(), x.max(), bins + 1); centres = 0.5 * (edges[:-1] + edges[1:])
    top = np.full(bins, np.nan); bottom = np.full(bins, np.nan); width = np.full(bins, np.nan)
    for i in range(bins):
        sel = (x >= edges[i]) & (x < edges[i + 1] + (1e-6 if i == bins - 1 else 0))
        if not sel.any():
            continue
        width[i] = np.abs(v[sel, 1]).max()
        mid = sel & (np.abs(v[:, 1]) < half_width_cm)
        if mid.any():
            top[i] = v[mid, 2].max(); bottom[i] = v[mid, 2].min()
    for arr in (top, bottom, width):                           # fill gaps
        ok = ~np.isnan(arr)
        arr[~ok] = np.interp(centres[~ok], centres[ok], arr[ok])
    return {"x": centres, "top": top, "bottom": bottom, "width": width}


def register_to(reference: np.ndarray, mesh: np.ndarray) -> np.ndarray:
    """Map mesh vertices (project frame, same length and centre as the reference) onto the reference's
    canonical frame: per x-slice, height mapped between the two spines, width scaled. Smooth, so the
    labels vary continuously over the surface."""
    pr, pm = _profile(reference), _profile(mesh)
    x = mesh[:, 0]
    r_top, r_bot, r_w = (np.interp(x, pr["x"], pr[k]) for k in ("top", "bottom", "width"))
    m_top, m_bot, m_w = (np.interp(x, pm["x"], pm[k]) for k in ("top", "bottom", "width"))
    hz = (r_top - r_bot) / np.clip(m_top - m_bot, 1e-3, None)
    z = r_bot + (mesh[:, 2] - m_bot) * hz
    y = mesh[:, 1] * (r_w / np.clip(m_w, 1e-3, None))
    return np.stack([x, y, z], 1)


class WhaleModel:
    def __init__(self, kind: str, rig=None):
        self.kind = kind
        self.reference, self.faces_ref, self.groups_ref = project.load_obj(K9239_OBJ, with_groups=True)
        if kind == "k9239":
            self.rest = self.reference; self.faces = self.faces_ref; self.groups = self.groups_ref
            self.canonical = self.reference
            self.eye_vertices = 0
        elif kind == "humpback4k":
            from factory.skinned import Rig
            self.rig = rig or Rig.load(HUMPBACK4K_GLTF)
            rest = self.rig.rest
            self.length_scale = float((self.reference[:, 0].max() - self.reference[:, 0].min()) / (rest[:, 0].max() - rest[:, 0].min()))
            rest = rest * self.length_scale
            self.centre = np.array([(rest[:, 0].max() + rest[:, 0].min()) / 2, 0.0, (rest[:, 2].max() + rest[:, 2].min()) / 2])
            body = rest - self.centre
            # The file's two extra skinned meshes ("jingxu" = baleen) are 72-vertex blocks inside the mouth; they
            # poke through the head in the clip, so the model is the body mesh only.
            eyes = []
            ev = (np.concatenate([e["vertices"] for e in eyes]) * self.length_scale - self.centre) if eyes else np.zeros((0, 3))
            ef = np.concatenate([e["faces"] + sum(len(x["vertices"]) for x in eyes[:k]) for k, e in enumerate(eyes)]) if eyes else np.zeros((0, 3), np.int64)
            self.rest = np.concatenate([body, ev]); self.faces = np.concatenate([self.rig.faces, ef + len(body)])
            self.groups = {"body": np.arange(len(body)), "eye": np.arange(len(body), len(self.rest))}
            self.eye_vertices = len(ev)
            self.canonical = register_to(self.reference, self.rest)
            # the clip's rest axis of the front half, to level each frame the same way
            jp = self.rig.joint_positions(None)
            self._front_axis_rest = self._front_axis(jp)
        else:
            raise ValueError(kind)
        self.fin_regions = deform.fin_regions(self.rest) if kind == "k9239" else deform.fin_regions(self.canonical)

    library_path = None                   # the individual library (None: individuals.LIBRARY or WF_INDIVIDUALS)

    @classmethod
    def from_config(cls, config: dict, rig=None) -> "WhaleModel":
        model = cls((config.get("whale") or {}).get("kind", "k9239"), rig)
        if (config.get("whale") or {}).get("landmarks_v2"):
            model.landmarks_v2 = True
        if (config.get("whale") or {}).get("dorsal_front_ratio"):    # run06 round 5: the fin's front base where the hump's slope starts
            model.dorsal_front_ratio = float(config["whale"]["dorsal_front_ratio"])
        if (config.get("whale") or {}).get("ridge_soften"):           # run06 round 8: ridge knobs behind the fin toned down
            model.ridge_soften = float(config["whale"]["ridge_soften"])
        lib = (config.get("whale") or {}).get("individuals")          # run06 realism: its own library (same ids and textures)
        if lib:
            model.library_path = Path(lib) if Path(lib).is_absolute() else REPO_ROOT / lib
        if (config.get("whale") or {}).get("blade_templates"):          # run06 round 7: the parametric dorsal blade
            bt = config["whale"]["blade_templates"]
            model.enable_blade(Path(bt) if Path(bt).is_absolute() else REPO_ROOT / bt)
        return model

    # ---- run06 round 7: the parametric dorsal blade (factory.blade) ------------------------------------------------
    def enable_blade(self, templates_path) -> None:
        """Append one parametric blade to the model's vertex arrays (rest, faces, canonical, uv, groups "blade"); the
        asset's own blade is flattened by the individuals (dorsal_height near 0). Landmarks dorsal_fin_tip / _back /
        _front and the fin_te_* / fin_le_* chains move onto the blade; its vertices are the dorsal part and the blade
        sub-label. Each individual's blade shape comes from its library entry (blade_template, blade_H_cm, ...), placed
        on the back at the asset's back junction and carried by the pose through a rigid fit to the nearby back."""
        from factory import blade as bl
        fin = self.dorsal_fin()
        self.faces = self._close_old_blade(np.asarray(self.faces), np.asarray(fin["w"]))
        self.n_rig = len(self.rest); self.n_rig_faces = len(self.faces)
        self._blade_lib = bl.load_templates(templates_path)
        R = self.rest
        B = R[fin["back"]].astype(float)
        fb = fin.get("front_base", fin["front"])
        Fp = R[fb].mean(0) if isinstance(fb, (list, tuple)) else R[fb]
        along = Fp - B; along[1] = 0.0; along /= max(np.linalg.norm(along), 1e-6)
        up = np.array([-along[2], 0.0, along[0]])
        if up[2] < 0:
            up = -up
        self._blade_frame = (B, along, up)
        # the back's midline profile (top of the body per 4 cm of x, off the asset blade), so each blade's base line runs
        # from its back base to the point on the back where its own front base lies
        # Only the top surface counts: a 4 cm bin without a top midline vertex would otherwise take a belly vertex, so the
        # bins use a wider window and samples below the running top line are dropped. The midline's top vertex per bin is
        # also kept by index: the posed blade follows the posed back through them.
        w0 = np.asarray(fin["w"]); mid = (np.abs(R[:, 1]) < 10.0) & (w0 < 0.05)
        xs = np.arange(B[0] - 150.0, B[0] + 400.0, 4.0)
        zs = np.full(len(xs), np.nan); bidx = np.full(len(xs), -1)
        for j, xx in enumerate(xs):
            sel = np.nonzero(mid & (np.abs(R[:, 0] - xx) < 5.0))[0]
            if len(sel):
                bidx[j] = int(sel[np.argmax(R[sel, 2])]); zs[j] = R[bidx[j], 2]
        ok = ~np.isnan(zs)
        top = np.array([np.nanmax(zs[max(0, j - 6): j + 7]) if ok[j] else np.nan for j in range(len(xs))])
        ok &= zs > top - 25.0
        self._back_profile = (xs[ok], zs[ok])
        self._back_idx = bidx[ok].astype(int)
        # anchors: back vertices near the base, off the asset blade (for the pose transfer)
        w = np.asarray(fin["w"]); d = np.linalg.norm(R - B, axis=1)
        self._blade_anchors = np.nonzero((d < 90.0) & (d > 10.0) & (w < 0.05))[0]
        default = self._blade_local(None)
        blade_rest = self._blade_place(default)
        nb = len(blade_rest)
        self.rest = np.concatenate([R, blade_rest])
        self.faces = np.concatenate([self.faces, default["faces"] + self.n_rig])
        self.groups = {**self.groups, "blade": np.arange(self.n_rig, self.n_rig + nb)}
        anchor = np.argmin(((R[self._blade_anchors] - B) ** 2).sum(1))
        a_i = self._blade_anchors[anchor]
        self.canonical = np.concatenate([self.canonical, blade_rest + (self.canonical[a_i] - R[a_i])])
        self._blade_uv = self._blade_back_uv(fin, R, B, along, up)
        self._blade_local_lm = default["landmarks"]
        for k in ("_part_groups", "_tiny_edges"):
            self.__dict__.pop(k, None)

    def _blade_back_uv(self, fin: dict, R: np.ndarray, B, along, up) -> np.ndarray:
        """round 8: blade UVs on the dark back skin beside the fin, at the body's own texel density: an affine map from the
        back's (along, across) cm around the fin to its UVs (robust fit), evaluated at the blade's (along, height) cm with
        both sides on the same patch (no seam at the edges). The asset fin's own UV island carries the old fin's creases
        in the normal map, which showed as crumpled pale planes on the larger blade."""
        from factory import blade as bl
        w = np.asarray(fin["w"])
        rel = R - B; aa, hh, yy = rel @ along, rel @ up, rel[:, 1]
        sel = (w < 0.02) & (np.abs(aa) < 120.0) & (np.abs(yy) < 60.0) & (hh > -40.0)
        X = np.stack([aa[sel], yy[sel], np.ones(int(sel.sum()))], 1); U = self.rig.uv[sel]
        best = None
        for sgn in (1.0, -1.0):                     # one side of the back (a UV island each side of the midline)
            s = X[:, 1] * sgn >= 0
            Xs, Us = X[s], U[s]
            keep = np.ones(len(Xs), bool)
            for _ in range(3):
                c = np.linalg.lstsq(Xs[keep], Us[keep], rcond=None)[0]
                r = np.abs(Xs @ c - Us).max(1)
                keep = r <= max(3.0 * float(np.median(r)), 1e-4)
            err = float(np.median(np.abs(Xs[keep] @ c - Us[keep])))
            if best is None or err < best[0]:
                best = (err, c, sgn, Us[keep].min(0), Us[keep].max(0))
        _, c, sgn, lo, hi = best
        local = self._blade_local(None)["vertices"]          # default blade (cm): a along, y across, h up
        a_cm = np.clip(local[:, 0], -100.0, 110.0)
        across = sgn * np.clip(0.55 * np.clip(local[:, 2], 0.0, None) + np.abs(local[:, 1]), 0.0, 55.0)
        uv = np.stack([a_cm, across, np.ones(len(local))], 1) @ c
        return np.clip(uv, lo, hi)

    def _blade_island_uv(self, fin: dict, R: np.ndarray, B, along, up, Fp) -> np.ndarray:
        """Blade UVs spread over the asset fin's own UV island (dark in the run06 skin maps): an affine map from the
        asset fin's (along, up) to its UVs (robust fit on its better-fitting side, used for both sides so the leading and
        trailing edges carry no seam); each blade vertex is put at the same chord fraction and height fraction of the
        asset fin's triangle, so tangents are well defined and the texture detail is the fin's."""
        from factory import blade as bl
        w = np.asarray(fin["w"]); idx = np.nonzero(w > 0.5)[0]
        rel = R[idx] - B; aa, hh, yy = rel @ along, rel @ up, rel[:, 1]
        U = self.rig.uv[idx]
        best = None
        for sgn in (1.0, -1.0):
            s = yy * sgn >= 0
            if s.sum() < 6:
                continue
            X = np.stack([aa[s], hh[s], np.ones(int(s.sum()))], 1); Us = U[s]
            keep = np.ones(len(X), bool)
            for _ in range(2):                                   # drop vertices of another island (seams)
                c = np.linalg.lstsq(X[keep], Us[keep], rcond=None)[0]
                r = np.abs(X @ c - Us).max(1)
                keep = r <= max(3.0 * float(np.median(r)), 1e-4)
            err = float(np.abs(X[keep] @ c - Us[keep]).max())
            if best is None or err < best[0]:
                best = (err, c, Us[keep].min(0), Us[keep].max(0))
        _, c, lo, hi = best
        Ha = max(float(hh.max()), 1.0)
        ta = float((R[fin["tip"]] - B) @ along); La = float((Fp - B) @ along)
        R2 = 2 * bl.NC
        s_c = (1 - np.cos(np.linspace(0, np.pi, bl.NC))) / 2
        k = np.arange(bl.N_VERTS - 1) // R2; i = np.arange(bl.N_VERTS - 1) % R2
        s = np.where(i < bl.NC, s_c[np.minimum(i, bl.NC - 1)], s_c[np.clip(R2 - 1 - i, 0, bl.NC - 1)])
        hn = np.clip(bl.LEVELS[k], 0.0, 1.0)
        te = ta * hn; le = La + (ta - La) * hn
        a_as = te + s * (le - te); h_as = hn * Ha
        uv = np.stack([a_as, h_as, np.ones_like(a_as)], 1) @ c
        tip = np.array([[ta, Ha, 1.0]]) @ c
        return np.clip(np.concatenate([uv, tip]), lo, hi)

    def _blade_local(self, ind) -> dict:
        """The blade (blade-local cm) of an individual (its library entry), or the default (template 0, H 45 cm)."""
        from factory import blade as bl
        spec = None
        if ind is not None:
            try:
                spec = self.individual(ind)["spec"]
            except Exception:
                spec = None
        t = int(getattr(spec, "blade_template", -1)) if spec is not None else -1
        if t < 0:
            t, H, nicks, thick, seed = 0, 45.0, 0, 1.0, 0
        else:
            H, nicks, thick, seed = float(spec.blade_H_cm), int(spec.blade_nicks), float(spec.blade_thickness), int(spec.seed)
        cache = self.__dict__.setdefault("_blade_cache", {})
        key = (t, round(H, 3), nicks, round(thick, 3), seed)
        if key not in cache:
            temps = self._blade_lib["templates"]
            cache[key] = bl.make(temps[t % len(temps)], self._blade_lib["heights"], H, np.random.default_rng(seed + 991), nicks, thick)
        return cache[key]

    def _blade_frame_for(self, local: dict):
        """Back base on the back's midline at the asset's back junction, base line to the point on the back that is the
        blade's own base length ahead (so both feet sit on the skin), up perpendicular to it in the x-z plane."""
        B0, along0, _ = self._blade_frame
        xs, zs = self._back_profile
        V, lm = local["vertices"], local["landmarks"]
        L = float(np.linalg.norm(V[lm["front_base"]] - V[lm["back_base"]]))
        xb = B0[0] - 6.0                 # round 8: 6 cm behind the asset junction, so the old blade's footprint is under the new one
        B = np.array([xb, 0.0, float(np.interp(xb, xs, zs))])
        cand = np.stack([xs, np.zeros_like(xs), zs], 1)
        dist = np.linalg.norm(cand - B, axis=1); ahead = cand[:, 0] > B[0]
        if ahead.any():
            j = np.argmin(np.where(ahead, np.abs(dist - L), np.inf))
            along = cand[j] - B; along /= max(np.linalg.norm(along), 1e-6)
        else:
            along = along0
        up = np.array([-along[2], 0.0, along[0]])
        if up[2] < 0:
            up = -up
        return B, along, up

    def _close_old_blade(self, F: np.ndarray, w: np.ndarray) -> np.ndarray:
        """round 8: the asset blade's faces (any vertex with fin weight > 0.5) are removed and the hole is closed with a fan
        from the core vertex nearest the hole's centre (projected onto the fitted back by the individual's edit), so no
        folded, coincident or sunk faces remain beside the parametric blade."""
        core = w > 0.5
        drop = core[F].any(1)
        if not drop.any():
            return F
        removed = F[drop]
        und = {}
        for a, b, c in removed:
            for e in ((a, b), (b, c), (c, a)):
                und[frozenset(e)] = und.get(frozenset(e), 0) + 1
        kept_edges = set()
        for a, b, c in F[~drop]:
            for e in ((a, b), (b, c), (c, a)):
                kept_edges.add(frozenset(e))
        boundary = [(a, b) for tri in removed for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0]))
                    if und[frozenset((a, b))] == 1 and frozenset((a, b)) in kept_edges]
        if len(boundary) < 3:
            return F
        bverts = np.unique(np.array(boundary).ravel())
        cand = np.nonzero(core)[0]
        centre = int(cand[np.argmin(np.linalg.norm(self.rest[cand] - self.rest[bverts].mean(0), axis=1))])
        fan = np.array([(a, b, centre) for a, b in boundary if centre not in (a, b)], dtype=F.dtype)
        return np.concatenate([F[~drop], fan])

    def _blade_place(self, local: dict) -> np.ndarray:
        """round 8: the blade placed on the back and draped onto it: its lower part (the hump) keeps its height above the
        back's midline profile rather than above the straight base line, and its flanks follow the back's lateral fall
        (z = z0 - 0.019 y^2, measured beside the fin), so the hump merges into the back instead of cutting it as a wedge;
        the top of the blade is left straight."""
        from factory import blade as bl
        B, along, up = self._blade_frame_for(local)
        L = np.asarray(local["vertices"], float)
        V = bl.place(L, B, along, up)
        H = max(float(L[:, 2].max()), 1.0)
        hn = L[:, 2] / H

        def sm(t, a, b):
            q = np.clip((t - a) / (b - a), 0.0, 1.0); return q * q * (3 - 2 * q)
        xs, zs = self._back_profile
        xl = B[0] + L[:, 0] * along[0]; zl = B[2] + L[:, 0] * along[2]
        dz = (np.interp(xl, xs, zs) - zl) * (1.0 - sm(hn, 0.0, 0.25)) - 0.019 * L[:, 1] ** 2 * (1.0 - sm(hn, 0.05, 0.5))
        V[:, 2] += dz
        return V

    def _blade_posed(self, rig_posed: np.ndarray, ind, scale: float) -> np.ndarray:
        from factory import blade as bl
        local = self._blade_local(ind)
        rest_b = self._blade_place(local) * scale
        a = self._blade_anchors
        Rm, t = bl.kabsch(self.rest[a] * scale, rig_posed[a])
        out = rest_b @ Rm.T + t
        # the rigid fit holds near the back base; ahead of it an arched or bent back drops away (and an individual's girth
        # moves it): every blade vertex also takes the posed back's offset from the rigid fit at its own x, so both feet
        # and the hump's front slope stay on the skin
        bi = getattr(self, "_back_idx", None)
        if bi is not None and len(bi) > 4:
            res = rig_posed[bi] - (self.rest[bi] * scale @ Rm.T + t)
            xb = self.rest[bi, 0] * scale; o = np.argsort(xb); xb = xb[o]; res = res[o]
            kern = np.ones(5) / 5.0
            res = np.stack([np.convolve(np.pad(res[:, j], 2, mode="edge"), kern, mode="valid") for j in range(3)], 1)
            xq = rest_b[:, 0]
            out = out + np.stack([np.interp(xq, xb, res[:, j]) for j in range(3)], 1)
        return out

    @staticmethod
    def _front_axis(jp: dict) -> np.ndarray:
        a = jp["joint51_00"] - jp["joint55_04"]
        return a / np.linalg.norm(a)

    def individuals(self) -> dict:
        """The individual library (id -> spec), empty for the static model."""
        if self.kind != "humpback4k":
            return {}
        from factory import individuals as indiv
        lib = getattr(self, "_library", None)
        if lib is None:
            lib = self._library = indiv.load_library(self.library_path)
        return lib

    def dorsal_fin(self) -> dict:
        """The rig's dorsal fin, read off its own rest mesh (self.rest, project frame; the canonical frame is a per-slice
        warp onto the K9239 whale that fabricates a peak at the K9239 fin position and must not be used for this).
        The fin is the thin midline blade standing above the back: protrusion = midline height over the shoulder line
        (15 < |y| < 40, 90th percentile per 8 cm of x). Returns the tip / back-junction / front-junction vertex indices,
        the trailing- and leading-edge chains (tip -> junction, n points each), the per-vertex protrusion (cm) and a
        smooth membership weight for the individual edits. Cached; {} when no blade is found."""
        cached = self.__dict__.get("_dorsal_fin")
        if cached is not None:
            return cached
        out = {}
        if self.kind == "humpback4k":
            R = self.rest[: getattr(self, "n_rig", len(self.rest))]; x, y, z = R[:, 0], R[:, 1], R[:, 2]
            length = x.max() - x.min(); xm = x.min()
            lo, hi = xm + 0.30 * length, xm + 0.55 * length                       # the back from the tail stock to mid-body
            step = 4.0; edges = np.arange(lo, hi + step, step); cx = (edges[:-1] + edges[1:]) / 2
            # dorsal midline only: above the slab's median height (the belly has midline vertices too)
            slab_mid = np.array([np.median(z[(x >= a_ - 20) & (x < b_ + 20)]) if ((x >= a_ - 20) & (x < b_ + 20)).any() else np.nan for a_, b_ in zip(edges[:-1], edges[1:])])
            okm = ~np.isnan(slab_mid); slab_mid = np.interp(x, cx[okm], slab_mid[okm]) if okm.sum() >= 2 else np.full_like(x, np.median(z))
            mid = (np.abs(y) < 8.0) & (z > slab_mid)
            top = np.array([z[mid & (x >= a_) & (x < b_)].max() if (mid & (x >= a_) & (x < b_)).any() else np.nan for a_, b_ in zip(edges[:-1], edges[1:])])
            ok = ~np.isnan(top)
            if ok.sum() >= 10:
                top = np.interp(cx, cx[ok], top[ok])
                # the back ridge without the fin: running median over +-48 cm (the blade is ~20 cm long)
                k = 12; ridge = np.array([np.median(top[max(i - k, 0): i + k + 1]) for i in range(len(top))])
                excess = top - ridge
                k_tip = int(np.argmax(excess))
                if excess[k_tip] > 8.0:
                    # contiguous blade bins around the tip bin
                    k0 = k_tip
                    while k0 > 0 and excess[k0 - 1] > 3.0:
                        k0 -= 1
                    k1 = k_tip
                    while k1 < len(excess) - 1 and excess[k1 + 1] > 3.0:
                        k1 += 1
                    x0, x1 = edges[k0] - 2.0, edges[k1 + 1] + 2.0
                    base = np.interp(x, cx, ridge)
                    blade = mid & (x >= x0) & (x <= x1) & (z > base + 3.0)
                    sel = np.nonzero(blade)[0]
                    if len(sel) >= 6:
                        tip = int(sel[np.argmax((z - base)[sel])])
                        h = np.where(blade, z - base, 0.0); h = np.clip(h, 0.0, None)
                        # The outline is the blade's midline edge loop (|y| < 1 cm: the vertices where the two faces meet);
                        # side-face vertices would zigzag. Trailing edge = loop vertices behind the tip ordered by
                        # descending height, leading edge = those ahead of it; each chain is resampled at n points along
                        # its length, using midpoints of consecutive loop vertices (index pairs -> mean) so the points lie on
                        # the rendered mesh edges. Chain index 00 = tip, n-1 = junction with the back.
                        n_chain = 12
                        # the loop runs down to the base: midline vertices in the blade's x-range from the ridge line up
                        loop = np.nonzero(mid & (x >= x0) & (x <= x1) & (np.abs(y) < 1.0) & (z >= base - 0.5))[0]
                        if tip not in loop:
                            loop = np.append(loop, tip)
                        rear = loop[x[loop] <= x[tip] + 0.5]; fore = loop[x[loop] >= x[tip] - 0.5]
                        rear = rear[np.argsort(-h[rear])]; fore = fore[np.argsort(-h[fore])]        # tip first, base last

                        def monotone(chain, sign):
                            """Drop loop vertices that step back along the body (sign -1: rearward chain, +1: forward)."""
                            keep = [int(tip)]
                            for i in chain:
                                if i == tip:
                                    continue
                                if sign * (x[i] - x[keep[-1]]) >= -1.0:
                                    keep.append(int(i))
                            return keep
                        rear = monotone(rear, -1); fore = monotone(fore, +1)

                        def resample(chain, n):
                            if len(chain) < 2:
                                return [tip] * n
                            pts = R[chain]; seg = np.linalg.norm(np.diff(pts, axis=0), axis=1); arc = np.concatenate([[0.0], np.cumsum(seg)])
                            out_pts = []
                            for tt in np.linspace(0.0, arc[-1], n):
                                k = int(np.clip(np.searchsorted(arc, tt, side="right") - 1, 0, len(chain) - 2))
                                f = (tt - arc[k]) / max(seg[k], 1e-6); q = int(round(f * 4))        # quarter steps along the edge
                                i, j = chain[k], chain[k + 1]
                                out_pts.append(i if q == 0 else j if q == 4 else [i] * (4 - q) + [j] * q)   # a list = mean of its vertices
                            return out_pts
                        trailing, leading = rear, fore
                        back = int(rear[-1]); front = int(fore[np.argmax(x[fore])])
                        tw = np.clip((h - 1.0) / 8.0, 0.0, 1.0)
                        # The dorsal_fin_front label (front_base) is not the blade's root: see DORSAL_FRONT_RATIO. The blade
                        # root (front) stays the base-chord end for dorsal_region() / the individual edits and the fin_le chain.
                        out = {"tip": tip, "back": back, "front": front, "front_base": self._dorsal_front_base(R, mid, tip, back, front, getattr(self, "dorsal_front_ratio", None)),
                               "fin_te": resample(trailing, n_chain), "fin_le": resample(leading, n_chain),
                               "h": h, "w": tw * tw * (3 - 2 * tw), "chord": float(np.linalg.norm(R[front] - R[back])), "n_blade": int(blade.sum()),
                               "excess_cm": float(excess[k_tip]), "x_range": (float(x0), float(x1))}
        self._dorsal_fin = out
        return out

    @staticmethod
    def _dorsal_front_base(R: np.ndarray, mid: np.ndarray, tip: int, back: int, front: int, ratio: float | None = None):
        """The dorsal_fin_front label point: on the dorsal midline ahead of the blade root, where the straight-line
        distance from the tip is ratio x |tip - back junction| (DORSAL_FRONT_RATIO), in quarter steps along the
        mesh edges. A vertex index, or an index list whose mean is the point (as for the chains)."""
        ratio = DORSAL_FRONT_RATIO if ratio is None else ratio
        x, y = R[:, 0], R[:, 1]
        ahead = np.nonzero(mid & (np.abs(y) < 1.0) & (x > x[front] - 0.5) & (x < x[front] + 300.0))[0]
        chain = [int(front)]
        for i in ahead[np.argsort(x[ahead])]:
            if i != front and x[i] - x[chain[-1]] > 0.5:            # one vertex per position (seam duplicates coincide)
                chain.append(int(i))
        target = ratio * float(np.linalg.norm(R[back] - R[tip]))
        best, best_err = int(front), abs(float(np.linalg.norm(R[front] - R[tip])) - target)
        for i, j in zip(chain[:-1], chain[1:]):
            for q in range(1, 5):
                p = R[i] + (R[j] - R[i]) * (q / 4.0)
                err = abs(float(np.linalg.norm(p - R[tip])) - target)
                if err < best_err:
                    best, best_err = (j if q == 4 else [i] * (4 - q) + [j] * q), err
        return best

    def dorsal_region(self) -> dict:
        """Dorsal fin membership for the individual edits (factory.individuals.apply_rest): per-vertex height above the
        fin's base (cm), a smooth weight, the tip / back / front indices and the base chord, all from dorsal_fin()."""
        fin = self.dorsal_fin()
        if not fin:
            return {}
        return {"dorsal_h": fin["h"], "dorsal_w": fin["w"], "dorsal_idx": (fin["tip"], fin["back"], fin["front"]), "dorsal_chord": fin["chord"],
                "dorsal_front_base": fin.get("front_base", fin["front"])}

    def individual(self, ind):
        """Per-individual rig inputs (cached): rest positions per mesh (model frame) and joint offsets."""
        from factory import individuals as indiv
        if ind is None:
            return None
        if isinstance(ind, str):
            lib = getattr(self, "_library", None)
            if lib is None:
                lib = self._library = indiv.load_library(self.library_path)
            if ind not in lib:
                raise KeyError(f"unknown individual {ind!r}: run python3 -m factory.individuals")
            spec = lib[ind]
        else:
            spec = indiv.Individual.from_dict(ind)
        cache = self.__dict__.setdefault("_individual_cache", {})
        if spec.id in cache:
            return cache[spec.id]
        # Edit in the skinned rest pose (project frame, where the joint positions live), then map the edited
        # vertices back to bind space through each vertex's exact blended skin matrix.
        joints = self.rig.joint_positions(None)
        regions = {**self.rig.pec_regions(), **self.dorsal_region()}
        if getattr(self, "n_rig", None) is not None:
            regions["dorsal_flatten"] = True          # round 7: the asset blade is pressed into the back under the parametric one
        if getattr(self, "ridge_soften", 0.0) and "dorsal_idx" in regions:
            # round 8: the ridge knobs between the fin and the flukes (rest frame of the rig, where the edits run)
            rw = self.rig.pose(None, self.rig.mesh_index)
            regions["ridge_soften"] = float(self.ridge_soften)
            regions["ridge_x"] = (-480.0, float(rw[regions["dorsal_idx"][1], 0]) - 15.0)
        positions = {}
        for mi in range(len(self.rig.js["meshes"])):
            rest_world = self.rig.pose(None, mi)
            edited = indiv.apply_rest(rest_world, spec, joints, regions if mi == self.rig.mesh_index else {})
            positions[mi] = self.rig.bind_from_rest(edited, mi)
        entry = {"positions": positions, "spec": spec}
        if len(cache) > 32:
            cache.clear()
        cache[spec.id] = entry
        return entry

    def _pec_membership(self) -> dict:
        """Smooth per-vertex membership of each pectoral fin (chain skin weight, cut off at the shoulder so the
        back never moves): {"pec_left": (N,), "pec_right": (N,)}, cached."""
        mem = self.__dict__.get("_pec_mem")
        if mem is None:
            from factory.individuals import _smooth
            from scipy import sparse
            regions = self.rig.pec_regions(); y = np.abs(self.rig.rest[:, 1])
            # the skin weights jump between neighbours at the armpit: smooth them over the mesh (1-ring mean, 6x)
            F = self.faces[: getattr(self, "n_rig_faces", len(self.faces))]; n = len(self.rig.rest)
            E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
            A = sparse.coo_matrix((np.ones(2 * len(E)), (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])), shape=(n, n)).tocsr()
            deg = np.asarray(A.sum(1)).ravel() + 1.0
            mem = {}
            for side, w in regions.items():
                ws = w.copy()
                for _ in range(6):
                    ws = (A @ ws + ws) / deg
                mem[side] = ws * _smooth(y, 95.0, 150.0)
            self._pec_mem = mem
        return mem

    def droop_pectorals(self, V: np.ndarray, jp: dict, droop_deg: float, side_deg=None) -> np.ndarray:
        """Swing both pectoral fins down by droop_deg (vertex level, membership-weighted) about a hinge along the
        body axis through each fin's root joint; V and jp in the same frame (rig frame, cm)."""
        if abs(droop_deg) < 1e-6 and not side_deg:
            return V
        out = V.copy()
        for k, (side, root_name, sign) in enumerate((("pec_left", "joint64_013", 1.0), ("pec_right", "joint70_019", -1.0))):
            w = self._pec_membership()[side]; idx = np.where(w > 1e-3)[0]
            if len(idx) == 0:
                continue
            # side_deg (run06 realism: one flipper raised for a slap): [left, right] added to the common droop
            d = droop_deg + (float(side_deg[k]) if side_deg else 0.0)
            root = np.asarray(jp[root_name]); ang = np.radians(sign * d) * w[idx]
            rel = out[idx] - root; c, s_ = np.cos(ang), np.sin(ang)
            y, z = rel[:, 1], rel[:, 2]                       # rotation about the x axis through the root
            out[idx, 1] = root[1] + c * y - s_ * z
            out[idx, 2] = root[2] + s_ * y + c * z
        return out

    def _levelling(self, jp: dict) -> np.ndarray:
        """Rotation (rig frame) that levels the front axis of a frame to the rest axis, as clip_pose applies it."""
        a = self._front_axis(jp); b = self._front_axis_rest
        axis = np.cross(a, b); s = np.linalg.norm(axis); c = float(np.dot(a, b))
        if s > 1e-8:
            k = axis / s; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
            return np.eye(3) + s * K + (1 - c) * (K @ K)
        return np.eye(3)

    def body_frame_offsets(self, t: float | None, joint_deg: dict) -> dict:
        """{joint name: 3x3 model-frame rotation} for joint_deg = {joint: [roll, pitch, yaw] degrees about the
        whale's own axes (x forward, y right, z up) in the levelled body frame of clip time t}. The clip pitches
        and turns the whale, so a rotation about a world axis would not be a pitch of the tail; this builds the
        rotation about the body axes carried into the rig frame of that frame."""
        if not joint_deg:
            return {}
        P = MODEL_TO_PROJECT_ / 100.0                                   # model -> project (orthonormal, det -1)
        R_lev = self._levelling(self.rig.joint_positions(t))          # rig/project frame, levelled = R_lev @ rig
        out = {}
        for name, (roll, pitch, yaw) in joint_deg.items():
            if abs(roll) + abs(pitch) + abs(yaw) < 1e-9:
                continue
            R = np.eye(3)
            for axis_proj, deg in ((np.array([1.0, 0, 0]), roll), (np.array([0, 1.0, 0]), pitch), (np.array([0, 0, 1.0]), yaw)):
                if abs(deg) < 1e-9:
                    continue
                k = R_lev.T @ axis_proj                                    # the body axis, in the (unlevelled) rig frame
                k = P.T @ k; k /= np.linalg.norm(k)                       # ... in the model frame
                K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]]); a = np.radians(deg)
                R = (np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * (K @ K)) @ R
            out[name] = R
        return out

    def curl_fluke(self, V: np.ndarray, curl_deg: float) -> np.ndarray:
        """Swing the fluke blade about a lateral hinge at its base by curl_deg (+ = tips up), vertex level in
        the levelled body frame with a smooth membership over the last part of the tail stock. The dive clip
        only ever lifts the blade; the tips-down phase of a whale doubling over needs this on top."""
        if abs(curl_deg) < 1e-6:
            return V
        from factory.individuals import _smooth
        C = self.canonical[: len(V)]                                    # (the rig's vertices; the parametric blade follows later)
        w = _smooth(-C[:, 0], 380.0, 620.0)              # 0 on the tail stock, 1 on the blade (a 2.4 m hinge: a -95 curl without creasing)
        idx = np.where(w > 1e-3)[0]
        hinge = V[(C[:, 0] > -530) & (C[:, 0] < -490)].mean(0)
        ang = np.radians(curl_deg) * w[idx]; c, s_ = np.cos(ang), np.sin(ang)
        rel = V[idx] - hinge; x, z = rel[:, 0], rel[:, 2]
        out = V.copy()
        out[idx, 0] = hinge[0] + c * x + s_ * z                      # rotation about +y through the hinge
        out[idx, 2] = hinge[2] - s_ * x + c * z
        return out

    def clip_pose(self, t: float | None, ind=None, pec_droop_deg: float = 0.0, fluke_curl_deg: float = 0.0, pec_side_deg=None) -> np.ndarray:
        """Clip frame t (None = rest) as a whale-frame mesh: root-relative, the front half levelled to the rest
        axis, scaled and centred like the rest pose (body then eyes); ``ind`` = an individual id / spec."""
        entry = self.individual(ind)
        pos = entry["positions"] if entry else {}
        joint_deg = dict(entry["spec"].joint_deg) if entry else {}
        offs = self.body_frame_offsets(t, joint_deg) or None
        V = self.rig.pose(t, None, pos.get(self.rig.mesh_index), offs); jp = self.rig.joint_positions(t, offs)
        eyes = []                                      # body only (see __init__)
        ev = np.concatenate([e["vertices"] for e in eyes]) if eyes else np.zeros((0, 3))
        allv = np.concatenate([V, ev])
        a = self._front_axis(jp); b = self._front_axis_rest
        # rotation taking a to b (about the axis a x b), then undo the root motion
        axis = np.cross(a, b); s = np.linalg.norm(axis); c = float(np.dot(a, b))
        if s > 1e-8:
            k = axis / s; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
            R = np.eye(3) + s * K + (1 - c) * (K @ K)
        else:
            R = np.eye(3)
        root_rest = self.rig.joint_positions(None)["joint51_00"]
        out = (R @ (allv - jp["joint51_00"]).T).T + root_rest
        if pec_droop_deg or pec_side_deg:
            # the fins swing about the body axis: do it in the levelled frame (the clip's heading and root motion
            # would otherwise tilt the hinge), with the fin root joints carried into that frame
            jp_lev = {k: R @ (np.asarray(v) - jp["joint51_00"]) + root_rest for k, v in jp.items()}
            out = self.droop_pectorals(out, jp_lev, pec_droop_deg, pec_side_deg)
        if fluke_curl_deg:
            out = self.curl_fluke(out, fluke_curl_deg)
        return out * self.length_scale - self.centre

    def fluke_lift_deg(self, t: float | None) -> float:
        """How far the frame itself lifts the fluke: angle (deg, + = up) of the tail-base -> fluke-tips line
        above the levelled front axis, in the clip frame at time t. 0 for the static model."""
        if self.kind != "humpback4k" or t is None:
            return 0.0
        cache = self.__dict__.setdefault("_lift_cache", {})
        key = round(float(t), 2)
        if key not in cache:
            V = self.clip_pose(key)
            x = V[:, 0]
            tail = V[x < x.min() + 0.12 * (x.max() - x.min())]            # the flukes
            base = V[np.abs(x - (x.min() + 0.30 * (x.max() - x.min()))) < 20]   # the tail stock
            d = tail.mean(0) - base.mean(0)
            cache[key] = float(np.degrees(np.arctan2(d[2], -d[0])))
        return cache[key]

    def _stitch(self, V: np.ndarray) -> np.ndarray:
        """Seam stitch: the asset's skinning pulls a few tiny edges (dorsal-fin base, fluke edge) apart by 3-5x in
        some clip frames. Edges under 8 cm at rest that stretch past 2x are closed to their midpoint."""
        tiny = self.__dict__.get("_tiny_edges")
        if tiny is None:
            F = self.faces[: getattr(self, "n_rig_faces", len(self.faces))]
            E = np.unique(np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1), axis=0)
            L0 = np.linalg.norm(self.rest[E[:, 0]] - self.rest[E[:, 1]], axis=1)
            keep = (L0 > 0.5) & (L0 < 8.0)
            tiny = self._tiny_edges = (E[keep], L0[keep])
        E, L0 = tiny
        for _ in range(3):
            L = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)
            bad = L > 2.0 * L0
            if not bad.any():
                break
            a, b = E[bad, 0], E[bad, 1]
            mid = (V[a] + V[b]) / 2
            V[a] = mid; V[b] = mid
        return V

    def posed(self, shot: dict) -> np.ndarray:
        """The shot's whale in its own frame (before the rigid placement): body scale, clip bend, fins (and, with the
        parametric blade, the individual's blade carried by the back)."""
        V = self._posed_rig(shot)
        if getattr(self, "n_rig", None) is not None and self.kind == "humpback4k":
            V = np.concatenate([V[: self.n_rig], self._blade_posed(V[: self.n_rig], shot.get("individual"), float(shot.get("body_scale", 1.0)))])
        return V

    def _posed_rig(self, shot: dict) -> np.ndarray:
        """The shot's whale in its own frame (before the rigid placement): body scale, clip bend, fins."""
        scale = float(shot.get("body_scale", 1.0))
        params = deform.DeformParams(**shot["deform"]) if shot.get("deform") else deform.DeformParams()
        if self.kind == "k9239":
            base = self.rest * scale if abs(scale - 1.0) > 1e-6 else self.rest
            if params.is_identity():
                return base
            regions = self.fin_regions if abs(scale - 1.0) < 1e-6 else deform.fin_regions(base)
            return deform.deform(base, params, regions)
        t = (shot.get("pose") or {}).get("clip_t")
        ind = shot.get("individual")
        droop = float(shot.get("pec_droop_deg") or 0.0); curl = float(shot.get("fluke_curl_deg") or 0.0)
        side = shot.get("pec_side_deg")            # run06 realism only: [left, right] extra flipper angle
        base = (self.clip_pose(float(t) if t is not None else None, ind, droop, curl, side) if (t is not None or ind or droop or curl or side)
                else self.rest[: getattr(self, "n_rig", len(self.rest))].copy())
        base = self._stitch(base) * scale
        # the clip bends the body; keep only the fin parameters of the procedural deformation
        if (shot.get("pose") or {}).get("bend") == "procedural":
            # the back-arch series: a straight keyframe plus the procedural hump (arch < 0 on this model = back up,
            # head and tail down), tail curl and head pitch; see shotlist.make_shot
            params = deform.DeformParams(**{**deform.params_to_dict(params), "fin_raise_deg": 0.0, "fin_sweep_deg": 0.0, "fluke_twist_deg": 0.0})
            if params.is_identity():
                return base
            return deform.deform(base, params, None)
        # The clip bends the body and the pectoral angle is a joint matter (pec_droop_deg, individuals): the
        # procedural fin rotation (deform.rotate_fins) picks its "fin" vertices by a K9239 heuristic that tears
        # this mesh at the shoulder, so none of the procedural parameters apply to clip frames.
        return base

    def landmark_indices(self) -> dict:
        """Landmark vertex indices on this model (factory.project.landmarks semantics). For the rig the
        pectoral bases/tips come from its fin joints (the geometric pick assumes hanging fins) and the eyes
        from the K9239 eye landmarks, mapped through the shared canonical frame (nearest vertex)."""
        if self.kind == "k9239":
            return project.landmarks(self.canonical, self.groups)
        idx = project.landmarks(self.canonical, self.groups)
        jp = self.rig.joint_positions(None)
        for name, joint in (("pectoral_base_left", "joint64_013"), ("pectoral_base_right", "joint70_019"),
                            ("pectoral_tip_left", "joint69_018"), ("pectoral_tip_right", "joint75_024")):
            p = np.asarray(jp[joint]) * self.length_scale - self.centre          # rest frame of self.rest
            idx[name] = int(np.argmin(((self.rest - p) ** 2).sum(1)))
        ref_idx = project.landmarks(self.reference, self.groups_ref)
        for name in ("eye_left", "eye_right"):
            if ref_idx.get(name) is not None:
                p = project.landmark_positions({name: ref_idx[name]}, self.reference)[name]
                idx[name] = int(np.argmin(((self.canonical - p) ** 2).sum(1)))
        # the dorsal fin from the rig's own geometry (the canonical warp puts a phantom fin at the K9239 fin position);
        # dorsal_fin_front = the front of the whole fin / hump (front_base, the researcher's definition since 2026-11-02),
        # fin_le_11 = the blade's root
        fin = self.dorsal_fin()
        if fin:
            idx = {k: v for k, v in idx.items() if not k.startswith(("fin_te_", "fin_le_"))}
            idx["dorsal_fin_tip"], idx["dorsal_fin_back"], idx["dorsal_fin_front"] = fin["tip"], fin["back"], fin.get("front_base", fin["front"])
            for prefix in ("fin_te", "fin_le"):
                for k, i in enumerate(fin[prefix]):
                    idx[f"{prefix}_{k:02d}"] = [int(j) for j in i] if isinstance(i, list) else int(i)   # a pair = the edge's midpoint
        if getattr(self, "n_rig", None) is not None:
            # run06 round 7: the fin landmarks on the parametric blade: tip, the trailing edge's foot (back base), the
            # leading edge's foot (front base = where the hump's slope leaves the back line), and the edge chains on
            # the blade's right flank from the foot (h 0.1) to the top
            from factory import blade as bl
            lm, n0, R2 = self._blade_local_lm, self.n_rig, 2 * bl.NC
            idx = {k: v for k, v in idx.items() if not k.startswith(("fin_te_", "fin_le_"))}
            idx["dorsal_fin_tip"], idx["dorsal_fin_back"], idx["dorsal_fin_front"] = n0 + lm["tip"], n0 + lm["back_base"], n0 + lm["front_base"]
            k1 = int(np.argmin(np.abs(bl.LEVELS - 0.1)))
            ks = [int(round(v)) for v in np.linspace(bl.NL - 1, k1, 10)]    # 10 levels whatever the blade's resolution
            for j, k in enumerate(ks):                                       # tip -> foot, like the asset chains
                idx[f"fin_te_{j:02d}"] = n0 + k * R2 + 0
                idx[f"fin_le_{j:02d}"] = n0 + k * R2 + bl.NC - 1
        if getattr(self, "landmarks_v2", False):
            # run06 round 5 (config whale.landmarks_v2): fluke_notch on the midline (|y| < 3 cm; the |y| < 30 cm pick sat
            # ~25 cm off it), and tail_stock_top = the top of the tail stock at the fluke insertion (the researcher's click,
            # just ahead of the notch: on the midline, a third of the way from the notch to the stock's narrowest point)
            R = self.rest; x, y, z = R[:, 0], R[:, 1], R[:, 2]
            mid = np.abs(y) < 3.0
            tail = x < x.min() + 0.15 * (x.max() - x.min())
            sel = np.nonzero(mid & tail)[0]
            if len(sel):
                notch = int(sel[np.argmin(x[sel])]); idx["fluke_notch"] = notch
                se = idx.get("spine_end")
                if se is not None:
                    xs = float(R[se].mean(0)[0] if isinstance(se, list) else R[se][0])
                    xi = x[notch] + (xs - x[notch]) / 3.0
                    band = np.nonzero(mid & (np.abs(x - xi) < 4.0))[0]
                    if len(band):
                        idx["tail_stock_top"] = int(band[np.argmax(z[band])])
        return {k: v for k, v in idx.items() if v is not None}

    def part_groups(self) -> dict:
        """The run06 per-pixel part taxonomy as vertex groups on the rig's own rest mesh (self.rest, project frame), so it
        is exact for every individual and pose (vertex i stays canonical vertex i) and identical across animals. PART_IDS
        order; a vertex belongs to one group, "body" is all the rest:
          fluke     behind the narrowest point of the tail stock (the spine_end landmark's x): both blades
          dorsal    the fin blade with its hump: between the back base (dorsal_fin_back) and the front base
                    (dorsal_fin_front, the researcher's fin front), above the base line between the two minus
                    DORSAL_HUMP_DEPTH_CM, within DORSAL_HUMP_HALF_WIDTH_CM of the midline
          pectoral  the smoothed pectoral membership (> 0.5) that pec_droop_deg swings
          head      ahead of a cross plane HEAD_BEHIND_BLOWHOLE_CM behind the blowhole (the blowhole, eyes, jaw hinge
                    and rostrum lie on it), pectorals excluded
        Returns {name: vertex indices}; cached. Empty for the static K9239 model."""
        cached = self.__dict__.get("_part_groups")
        if cached is not None:
            return cached
        out: dict = {}
        if self.kind == "humpback4k":
            R = self.rest; x, y, z = R[:, 0], R[:, 1], R[:, 2]
            idx = self.landmark_indices()

            def at(name):
                v = idx[name]
                return R[v].mean(0) if isinstance(v, list) else R[v]
            fluke = x < at("spine_end")[0]
            back, front = at("dorsal_fin_back"), at("dorsal_fin_front")
            base = back[2] + (x - back[0]) * (front[2] - back[2]) / max(front[0] - back[0], 1e-6)
            dorsal = (x >= back[0] - 1.0) & (x <= front[0] + 1.0) & (np.abs(y) < DORSAL_HUMP_HALF_WIDTH_CM) & (z > base - DORSAL_HUMP_DEPTH_CM)
            mem = self._pec_membership()
            pad = len(R) - len(mem["pec_left"])                        # the parametric blade's vertices: never pectoral
            pec = np.concatenate([(mem["pec_left"] > 0.5) | (mem["pec_right"] > 0.5), np.zeros(max(pad, 0), bool)])
            head = (x >= at("blowhole")[0] - HEAD_BEHIND_BLOWHOLE_CM) & ~pec
            taken = np.zeros(len(R), bool)
            for name, sel in (("fluke", fluke), ("dorsal", dorsal), ("pectoral", pec), ("head", head)):
                sel = sel & ~taken; taken |= sel
                out[name] = np.nonzero(sel)[0]
            out["body"] = np.nonzero(~taken)[0]
            if getattr(self, "n_rig", None) is not None:              # the blade is dorsal fin, whole
                bl_idx = np.arange(self.n_rig, len(R))
                for k in list(out):
                    out[k] = np.setdiff1d(out[k], bl_idx)
                out["dorsal"] = np.union1d(out["dorsal"], bl_idx)
        self._part_groups = out
        return out

    def vertex_part_ids(self) -> np.ndarray | None:
        """(N,) uint8 PART_IDS per vertex (part_groups), None for the static model."""
        groups = self.part_groups()
        if not groups:
            return None
        ids = np.zeros(len(self.rest), np.uint8)
        for name, sel in groups.items():
            ids[sel] = PART_IDS[name]
        return ids

    def write_obj(self, vertices: np.ndarray, path: Path, comment: str = "posed whale", individual=None) -> None:
        """Posed OBJ with the model's groups (body, eye...) for the renderer. With the parametric blade, the blade's UVs
        are the animal's own two side charts (factory.blade_uv2, round 9), with OBJ-level copies of its seam vertices."""
        from factory.humpback4k import write_obj as _w
        groups = []
        uv = self._uv()
        blade_info = None
        if getattr(self, "n_rig", None) is not None:
            from factory import blade_uv2
            uv_b, blade_info = blade_uv2.blade_uv(self._blade_local(individual)["vertices"])
            uv = np.concatenate([uv[: self.n_rig], uv_b])
        items = dict(self.groups)
        if getattr(self, "n_rig", None) is not None and "blade" in items and "body" in items and len(items["body"]):
            # round 8: the blade is written in the body's group (one material slot), so it gets the same skin material and
            # values as the back; the indices are contiguous (body 0..n_rig-1, blade n_rig..)
            blade = items.pop("blade")
            if int(items["body"].max()) + 1 == int(blade.min()):
                items["body"] = np.concatenate([items["body"], blade])
            else:
                items["blade"] = blade
        for name, idx in items.items():
            if len(idx) == 0:
                continue
            lo, hi = int(idx.min()), int(idx.max()) + 1
            fmask = (self.faces[:, 0] >= lo) & (self.faces[:, 0] < hi)
            g = (name, vertices[lo:hi], self.faces[fmask] - lo, uv[lo:hi])
            if getattr(self, "n_rig", None) is not None and name == "body" and hi > self.n_rig:
                N = self._smooth_normals(vertices[lo:hi], self.faces[fmask] - lo, self.n_rig - lo)
                if blade_info is not None:
                    from factory import blade_uv2
                    n0 = self.n_rig - lo
                    V2, F2, UV2 = blade_uv2.seam_split(g[1], g[2], g[3], blade_info, n0)
                    sc = blade_info["seam"]
                    N = np.concatenate([N, N[sc["cols"] + n0], N[[sc["tip"] + n0]]])
                    g = (name, V2, F2, UV2)
                g = g + (N,)
            groups.append(g)
        _w(Path(path), groups, comment)

    def _smooth_normals(self, V: np.ndarray, F: np.ndarray, n_body: int) -> np.ndarray:
        """round 8: area-weighted vertex normals (outward) for the body + blade group, the blade's lowest rings blended into
        the normals of the back beside them, so the blade shades as one smooth surface with the hump (no crease)."""
        from factory import blade as bl
        V = np.asarray(V, float)
        tri = V[F]
        fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        if (np.einsum("ij,ij->i", fn, tri.mean(1) - V.mean(0)) < 0).mean() > 0.5:
            fn = -fn
        N = np.zeros_like(V)
        for k in range(3):
            np.add.at(N, F[:, k], fn)
        # weld coincident vertices (UV seams split them) so the normals are smooth across the seams, as the importer's are
        _, inv = np.unique(np.round(V, 2), axis=0, return_inverse=True)
        inv = inv.ravel()
        acc = np.zeros((int(inv.max()) + 1, 3)); np.add.at(acc, inv, N); N = acc[inv]
        N /= np.linalg.norm(N, axis=1, keepdims=True) + 1e-12
        nb = len(V) - n_body
        if nb > 0:
            k = np.arange(nb) // (2 * bl.NC)
            lev = np.where(np.arange(nb) < nb - 1, bl.LEVELS[np.minimum(k, bl.NL - 1)], 1.0)
            w = np.clip((0.05 - lev) / 0.2, 0.0, 1.0)       # only at and under the join
            sel = np.nonzero(w > 0)[0]
            if len(sel):
                Bv = V[n_body + sel]
                near = np.nonzero(np.linalg.norm(V[:n_body] - Bv.mean(0), axis=1) < 400.0)[0]
                if len(near):
                    d = ((Bv[:, None, :] - V[near][None, :, :]) ** 2).sum(-1)
                    nn = near[np.argmin(d, axis=1)]
                    mix = N[n_body + sel] * (1 - w[sel, None]) + N[nn] * w[sel, None]
                    N[n_body + sel] = mix / (np.linalg.norm(mix, axis=1, keepdims=True) + 1e-12)
        return N

    def _uv(self) -> np.ndarray:
        if self.kind == "humpback4k":
            return np.concatenate([self.rig.uv, self._blade_uv]) if getattr(self, "n_rig", None) is not None else self.rig.uv
        raise NotImplementedError
