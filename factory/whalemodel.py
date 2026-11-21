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
        if (config.get("whale") or {}).get("blade_weld"):              # run06 round 12: the blade welded into the back
            model.blade_weld = True
            from factory import weld as _wd                              # round 15: blade_weld = 15 (true = round 13)
            if _wd.design_of(config["whale"]["blade_weld"]) == 15:
                model.weld_design = 15
        if (config.get("whale") or {}).get("arch_bend"):                # run06 round 20c: procedural bends about the centre line
            model.arch_bend = dict(config["whale"]["arch_bend"])
        model.apply_r20d((config.get("whale") or {}))                   # run06 round 20d: weld_rest / weld_smooth
        if (config.get("whale") or {}).get("obj_shared_vertices"):      # run06 round 20g: one OBJ position across UV seams
            model.obj_shared_vertices = dict(config["whale"]["obj_shared_vertices"])
        if (config.get("whale") or {}).get("blade_uv_match"):           # run06 round 20g: blade charts matched to the patch's colour
            model.blade_uv_match = dict(config["whale"]["blade_uv_match"])
        if (config.get("whale") or {}).get("dorsal_hump"):              # run06 round 20e: a convex hump under the welded fin
            model.dorsal_hump = dict(config["whale"]["dorsal_hump"])
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
        if getattr(self, "blade_weld", False):
            return self._enable_weld(fin)
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
        xs = np.arange(B[0] - 150.0, B[0] + 650.0, 4.0)      # round 11: 400 -> 650 cm (long humps ran past it)
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
        spec = None
        if ind is not None:
            try:
                spec = self.individual(ind)["spec"]
            except Exception:
                spec = None
        return self._blade_local_spec(spec)

    def _blade_local_spec(self, spec) -> dict:
        """_blade_local for a library spec (None = the default blade); round 20e: callable before the individual's rest edit."""
        from factory import blade as bl
        t = int(getattr(spec, "blade_template", -1)) if spec is not None else -1
        if t < 0:
            t, H, nicks, thick, seed = 0, 45.0, 0, 1.0, 0
        else:
            H, nicks, thick, seed = float(spec.blade_H_cm), int(spec.blade_nicks), float(spec.blade_thickness), int(spec.seed)
        cache = self.__dict__.setdefault("_blade_cache", {})
        weld = bool(getattr(self, "blade_weld", False))
        key = (t, round(H, 3), nicks, round(thick, 3), seed, weld)
        if key not in cache:
            temps = self._blade_lib["templates"]
            if weld:                     # round 12: the welded blade starts at its foot ring (factory.weld.LEVELS_W)
                from factory import weld as wd
                loc = bl.make(temps[t % len(temps)], self._blade_lib["heights"], H, np.random.default_rng(seed + 991), nicks, thick,
                              levels=wd.LEVELS_W)
                o = loc["outline"]; k0 = int(np.argmin(np.abs(o["levels"])))
                loc["base_len"] = float(o["le"][k0] - o["te"][k0]) * H; loc["H"] = H
                cache[key] = loc
            else:
                cache[key] = bl.make(temps[t % len(temps)], self._blade_lib["heights"], H, np.random.default_rng(seed + 991), nicks, thick)
        return cache[key]

    def _blade_frame_for(self, local: dict):
        """Back base on the back's midline at the asset's back junction, base line to the point on the back that is the
        blade's own base length ahead (so both feet sit on the skin), up perpendicular to it in the x-z plane."""
        B0, along0, _ = self._blade_frame
        xs, zs = self._back_profile
        V, lm = local["vertices"], local["landmarks"]
        L = float(local["base_len"]) if "base_len" in local else float(np.linalg.norm(V[lm["front_base"]] - V[lm["back_base"]]))
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
        # round 11: no column may go down after the drape: every ring stays at least half its nominal step above the ring
        # below it (measured along the blade's up), column by column
        R2 = 2 * bl.NC
        ring = (V[:-1] - B) @ up
        hl = L[:-1, 2]
        ring, hl = ring.reshape(bl.NL, R2), hl.reshape(bl.NL, R2)
        fixed = ring.copy()
        for k in range(1, bl.NL):
            fixed[k] = np.maximum(fixed[k], fixed[k - 1] + 0.5 * (hl[k] - hl[k - 1]))
        V[:-1] += np.outer((fixed - ring).ravel(), up)
        top = (V[-1] - B) @ up
        if top < fixed[-1].max() + 0.25 * (hl[-1].max() - hl[-2].max()):
            V[-1] += up * (fixed[-1].max() + 0.25 * max(hl[-1].max() - hl[-2].max(), 0.1) - top)
        return V

    def _blade_base_anchors(self, ind, local: dict) -> np.ndarray:
        """round 11: the back vertices under the fin itself (within max(40 cm, 0.8 H) of the point 0.6 H ahead of the back
        base, off the old blade), cached per individual: the rigid transform is taken there, so the fin keeps the back's
        position and tilt under it; the long low hump ahead follows the back through the blend instead."""
        cache = self.__dict__.setdefault("_blade_anchor_cache", {})
        key = ind if isinstance(ind, (str, type(None))) else id(ind)
        if key not in cache:
            B, along, _ = self._blade_frame_for(local)
            H = max(float(np.asarray(local["vertices"])[:, 2].max()), 1.0)
            C = B + along * 0.6 * H
            w = np.asarray(self.dorsal_fin()["w"]); R = self.rest[: self.n_rig]
            d = np.linalg.norm(R - C, axis=1)
            sel = np.nonzero((d < max(40.0, 0.8 * H)) & (w < 0.05) & (R[:, 2] > C[2] - 60.0))[0]
            cache[key] = sel if len(sel) >= 12 else self._blade_anchors
        return cache[key]

    def _blade_skin(self, rest_b: np.ndarray, rig_posed: np.ndarray, ind, local: dict, scale: float) -> np.ndarray:
        """round 11: the blade's (scaled) rest vertices carried by the posed back: per 15 cm bin of the blade's rest x, the
        rigid motion (Kabsch) of the back vertices under that bin (within 30 cm in x, off the old blade, the top of the
        body), interpolated linearly between bin centres."""
        from factory import blade as bl
        cache = self.__dict__.setdefault("_blade_skin_cache", {})
        key = ind if isinstance(ind, (str, type(None))) else id(ind)
        R0 = self.rest[: self.n_rig]
        if key not in cache:
            fin = self.dorsal_fin(); w = np.asarray(fin["w"])
            P = self._blade_place(local)
            xs = np.arange(P[:, 0].min() - 15.0, P[:, 0].max() + 30.0, 15.0)
            top = R0[:, 2] > np.interp(R0[:, 0], *self._back_profile) - 70.0
            bins = []
            for cx in xs:
                sel = np.nonzero((np.abs(R0[:, 0] - cx) < 30.0) & (np.abs(R0[:, 1]) < 70.0) & (w < 0.05) & top)[0]
                bins.append(sel if len(sel) >= 8 else None)
            cache[key] = (xs, bins)
        xs, bins = cache[key]
        Ts = [bl.kabsch(R0[sel] * scale, rig_posed[sel]) if sel is not None else None for sel in bins]
        ok = [k for k, T in enumerate(Ts) if T is not None]
        cx = xs[ok] * scale
        out = np.zeros_like(rest_b)
        x = rest_b[:, 0]
        j = np.clip(np.searchsorted(cx, x) - 1, 0, len(ok) - 2)
        f = np.clip((x - cx[j]) / np.maximum(cx[j + 1] - cx[j], 1e-6), 0.0, 1.0)
        for jj in np.unique(j):
            sel = j == jj
            (R1, t1), (R2, t2) = Ts[ok[jj]], Ts[ok[jj + 1]]
            out[sel] = (rest_b[sel] @ R1.T + t1) * (1 - f[sel, None]) + (rest_b[sel] @ R2.T + t2) * f[sel, None]
        return out

    def _blade_near_body(self, ind, local: dict):
        """round 11: for every blade vertex, the nearest back vertex (off the old blade) of the rest model, cached per
        individual; the posed blade's low rings follow the back there."""
        cache = self.__dict__.setdefault("_blade_near_cache", {})
        key = ind if isinstance(ind, (str, type(None))) else id(ind)
        if key not in cache:
            fin = self.dorsal_fin(); w = np.asarray(fin["w"])
            R = self.rest[: self.n_rig]; B = self._blade_frame[0]
            cand = np.nonzero((np.linalg.norm(R - B, axis=1) < 260.0) & (w < 0.05))[0]
            if len(cand) < 10:
                cache[key] = None
            else:
                P = self._blade_place(local)
                d2 = ((P[:, None, :] - R[cand][None, :, :]) ** 2).sum(-1)
                cache[key] = cand[np.argmin(d2, axis=1)]
        return cache[key]

    def _blade_posed(self, rig_posed: np.ndarray, ind, scale: float) -> np.ndarray:
        from factory import blade as bl
        local = self._blade_local(ind)
        rest_b = self._blade_place(local) * scale
        a = self._blade_base_anchors(ind, local)
        Rm, t = bl.kabsch(self.rest[a] * scale, rig_posed[a])
        out = rest_b @ Rm.T + t
        # round 11: the blade is stiff tissue: one rigid transform taken at its base (the back vertices around the back
        # base). Only its lowest rings (sunk, the foot and up to 0.12 H) blend into the posed back, and only along the
        # blade's up, so the junction stays closed while nothing can fold along the chord (round 8-10 moved every blade
        # vertex by the back's offset at its own x, which folded blades on arched backs)
        near = self._blade_near_body(ind, local)
        if near is not None:
            # The low part and the long hump ahead of the fin are the back itself: they are skinned to the posed back,
            # each vertex carried by the rigid motion of the back patch under it (a rotation and a translation per 15 cm
            # of the blade's length, interpolated along it), so the hump bends with an arching back without shearing.
            # The fin above the base stays on the base's rigid transform; the blend weight w is 1 for the sunk rings and
            # the foot (to -0.05 H), falls to 0 at 0.5 H, and is 1 at every height from 1.8 H ahead of the back base
            # (0 up to 0.4 H ahead).
            _, _, up = self._blade_frame_for(local)
            upp = Rm @ up
            Lv = np.asarray(local["vertices"])
            H = max(float(Lv[:, 2].max()), 1.0)
            hn, an = Lv[:, 2] / H, Lv[:, 0] / H
            q = np.clip((hn + 0.05) / 0.55, 0.0, 1.0); w = 1.0 - q * q * (3 - 2 * q)
            qa = np.clip((an - 0.4) / 1.4, 0.0, 1.0); w = np.maximum(w, qa * qa * (3 - 2 * qa))
            skinned = self._blade_skin(rest_b, rig_posed, ind, local, scale)
            out = out * (1.0 - w[:, None]) + skinned * w[:, None]
            # and the blend may not push a low ring past the one above it (columns keep rising along the posed up)
            R2 = 2 * bl.NC
            Lh = np.asarray(local["vertices"])[:-1, 2].reshape(bl.NL, R2) * scale
            ring = (out[:-1] @ upp).reshape(bl.NL, R2); fixed = ring.copy()
            for k in range(1, bl.NL):
                fixed[k] = np.maximum(fixed[k], fixed[k - 1] + 0.5 * (Lh[k] - Lh[k - 1]))
            out[:-1] += np.outer((fixed - ring).ravel(), upp)
        return out

    # ---- run06 round 12: the blade welded into the back (factory.weld) -------------------------------------------
    def _enable_weld(self, fin: dict) -> None:
        """Route B: cut the back around the fin (a fixed region for the whole library), weld the generated blade's foot
        ring to a refined patch whose vertices lie on the back's own surface lifted by the hump field (factory.weld).
        Vertex layout: rig | blade (ring 0 = foot) | patch interior | cut vertices | 2 landmarks (front, back base)."""
        from factory import blade as bl, weld as wd
        R = self.rest
        B = R[fin["back"]].astype(float)
        fb = fin.get("front_base", fin["front"])
        Fp = R[fb].mean(0) if isinstance(fb, (list, tuple)) else R[fb]
        along = Fp - B; along[1] = 0.0; along /= max(np.linalg.norm(along), 1e-6)
        up = np.array([-along[2], 0.0, along[0]])
        if up[2] < 0:
            up = -up
        self._blade_frame = (B, along, up)
        w0 = np.asarray(fin["w"]); mid = (np.abs(R[:, 1]) < 10.0) & (w0 < 0.05)
        xs = np.arange(B[0] - 150.0, B[0] + 650.0, 4.0)
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
        d = np.linalg.norm(R - B, axis=1)
        self._blade_anchors = np.nonzero((d < 90.0) & (d > 10.0) & (w0 < 0.05))[0]
        self._rig_faces_w = np.asarray(self.faces).copy()
        W = wd.Weld(self, fin)
        keys = [None] + sorted(self.individuals())
        inps = {}
        for k in keys:
            loc = self._blade_local(k)
            inps[k] = wd.animal_inputs(W, loc, self._blade_frame_for(loc), loc["H"])
        W.build(self, fin, np.concatenate([v["needs"] for v in inps.values()]))
        wd.build_canonical(W, inps[None])
        self.weld = W
        N0 = W.base_normals(W.R_rest)
        self._weld_sign = 1.0 if float((N0[np.unique(W.base_faces)] @ W.e_n).mean()) > 0 else -1.0
        n0 = self.n_rig; R2 = 2 * bl.NC
        self.n_blade = wd.NL_W * R2 + 1
        self._off_patch = n0 + self.n_blade
        self._off_cut = self._off_patch + W.n_int
        self._off_lm = self._off_cut + W.n_cut
        n_total = self._off_lm + 2
        A0 = self._weld_animal(None, inp=inps[None])
        # faces: kept rig (and cut) faces, the blade without a bottom, the patch (loop corners: the side's copy)
        kept = self._ext2model(W.kept_faces)
        bf = np.asarray(A0["local"]["faces"]) + n0
        P2 = A0["g"]["P2"]
        cf = W.cfaces
        nI = R2 + W.n_int
        loopP = W.loop2d
        cyc = np.where(cf < R2, 0.0, 0.0)
        pos = np.zeros((len(cf), 3, 2))
        isl = cf >= nI
        pos[~isl] = P2[cf[~isl]]
        pos[isl] = loopP[cf[isl] - nI]
        side = np.where(pos[:, :, 1].mean(1) >= 0, 0, 1)
        pf = np.zeros_like(cf)
        pf[cf < R2] = n0 + cf[cf < R2]
        inner = (cf >= R2) & (cf < nI)
        pf[inner] = self._off_patch + (cf[inner] - R2)
        for fi, k in zip(*np.nonzero(isl)):
            cp = W.loop_copies[W.loop_weld[cf[fi, k] - nI]]
            e = cp.get(int(side[fi]), next(iter(cp.values())))
            pf[fi, k] = int(self._ext2model(np.array([e]))[0])
        self._face_ranges = {"kept": (0, len(kept)), "blade": (len(kept), len(kept) + len(bf)),
                             "patch": (len(kept) + len(bf), len(kept) + len(bf) + len(pf))}
        self.faces = np.concatenate([kept, bf, pf]).astype(np.int64)
        rest = np.zeros((n_total, 3))
        rest[:n0] = R[:n0]
        self.rest = rest
        self.rest = self._weld_assemble(W.R_rest, A0["blade_rest"], A0["g"]["rest"])
        self.rest[:n0] = R[:n0]
        self.groups = {**self.groups, "blade": np.arange(n0, n_total)}
        can_ext = np.concatenate([self.canonical[:n0], self.canonical[W.cut_u] * (1 - W.cut_s[:, None]) + self.canonical[W.cut_v] * W.cut_s[:, None]])
        self._can_ext = can_ext
        self.canonical = self._weld_canonical(A0)
        self._blade_local_lm = A0["local"]["landmarks"]
        for k in ("_part_groups", "_tiny_edges"):
            self.__dict__.pop(k, None)

    def _rig_faces(self) -> np.ndarray:
        """The rig's own faces (after the old blade's fan), also when the weld replaced part of them."""
        rf = self.__dict__.get("_rig_faces_w")
        return rf if rf is not None else self.faces[: getattr(self, "n_rig_faces", len(self.faces))]

    def _ext2model(self, e):
        e = np.asarray(e)
        return np.where(e < self.n_rig, e, e - self.n_rig + self._off_cut)

    def _weld_assemble(self, Ex: np.ndarray, blade: np.ndarray, pts: np.ndarray) -> np.ndarray:
        from factory import blade as bl
        R2 = 2 * bl.NC; nI = R2 + self.weld.n_int
        return np.concatenate([Ex[: self.n_rig], blade, pts[R2:nI], Ex[self.n_rig:], pts[nI:]])

    def _weld_key(self, ind):
        if ind is None or isinstance(ind, str):
            return ind
        import json as _json
        return _json.dumps(ind, sort_keys=True, default=str)

    def _weld_animal(self, ind, inp=None) -> dict:
        """Per-animal weld data (cached): the hump / patch geometry on the rest base, the blade's rest (its lowest
        rings blended onto the welded foot), fade weights and anchors."""
        from factory import blade as bl, weld as wd
        cache = self.__dict__.setdefault("_weld_cache", {})
        key = self._weld_key(ind)
        if key in cache:
            return cache[key]
        if getattr(self, "weld_design", 13) == 15:               # round 15: implicit rest surface (factory.weld15)
            from factory import weld15
            wr = getattr(self, "weld_rest", None)
            if wr:
                # round 20d (whale.weld_rest): weld15 module constants for this build only (they are part of the disk
                # cache key, so a changed value never reads an old surface)
                saved = {k: getattr(weld15, k) for k in wr}
                try:
                    for k, v in wr.items():
                        setattr(weld15, k, type(saved[k])(v))
                    out = weld15.animal(self, ind)
                finally:
                    for k, v in saved.items():
                        setattr(weld15, k, v)
            else:
                out = weld15.animal(self, ind)
            if len(cache) > 64:
                cache.clear()
            cache[key] = out
            return out
        W = self.weld; R2 = 2 * bl.NC
        local = self._blade_local(ind)
        frame = self._blade_frame_for(local)
        if inp is None:
            inp = wd.animal_inputs(W, local, frame, local["H"])
        anchors = self._blade_base_anchors(ind, local)
        # the animal's own rest back (its individual edits: flattened old fin, girth, ridge ...): everything per animal is
        # built on it; the blade frame is carried there by the rigid fit at the fin base
        Ex_ind = W.ext(np.asarray(self._posed_rig({"individual": ind})[: self.n_rig], float)) if ind is not None else W.R_rest
        Ti, ti = bl.kabsch(W.R_rest[anchors], Ex_ind[anchors])
        B, along, up = frame
        frame_i = (Ti @ np.asarray(B, float) + ti, Ti @ along, Ti @ up)
        g = wd.animal_geometry(W, inp, self._weld_sign, Ex=Ex_ind, frame=frame_i, lift_dir=Ti @ W.e_n)
        if float(g["loop_D"].max()) > 1e-6 or float(W.phi(inp["needs"]).max()) > 0.0:
            raise RuntimeError(f"weld: the hump of {key!r} reaches the region's loop (the library changed? rebuild)")
        nominal = bl.place(np.asarray(local["vertices"], float), B, along, up) @ Ti.T + ti
        # round 13: where the animal's back stands above the blade's nominal foot ring (small animals: up to 0.27 H) the
        # whole blade is raised along its up by that burial (its largest, at most RAISE_MAX H) and the hump rebuilt to
        # the new foot; a residual goes into the patch (below)
        upi = frame_i[2]
        bur = (g["rest"][:R2] - nominal[:R2]) @ upi
        u_b = min(max(0.0, float(bur.max())), wd.RAISE_MAX * local["H"])
        if u_b > 1e-3 * local["H"]:
            inp = {**inp, "foot_up": u_b}
            g = wd.animal_geometry(W, inp, self._weld_sign, Ex=Ex_ind, frame=frame_i, lift_dir=Ti @ W.e_n)
            nominal = nominal + u_b * upi[None, :]
        nb = len(nominal)
        k = np.arange(nb - 1) // R2
        levels = np.concatenate([wd.LEVELS_W[k], [wd.LEVELS_W[-1] + 0.02]])
        col = np.concatenate([np.arange(nb - 1) % R2, [0]])
        # round 13: the blade keeps its nominal shape (round 12 blended its lowest rings onto the welded foot, shearing
        # them by up to 0.28 H: the arch line low on the fin and the kinked leading edge); the foot ring IS the blade's
        # ring 0, and the patch takes the foot's offset (welded foot -> nominal ring), fading out over DELTA_DECAY x R
        hl_ = np.asarray(local["vertices"])[:, 2] / local["H"]
        fd = wd.fade(np.maximum(hl_, wd.H_FOOT)); fd[-1] = 0.0
        delta = nominal[:R2] - g["rest"][:R2]
        Rcm = inp["R"]
        wdl = 1.0 - wd.smoothstep(g["d_foot"] / (wd.DELTA_DECAY * Rcm))
        sh = (g["wcol"] @ delta) * wdl[:, None]
        sh[:R2] = delta
        g = {**g, "rest": g["rest"] + sh, "off": g["off"] + sh,
             "wr": (1.0 - wd.smoothstep((g["d_foot"] / Rcm - wd.RIGID_D0) / (wd.RIGID_D1 - wd.RIGID_D0))) if wd.RIGID_D1 > wd.RIGID_D0
             else np.zeros(len(g["d_foot"]))}
        g["wr"][:R2] = 1.0
        blade_rest = nominal.copy()
        # round 13: the junction band (blade rings up to JUNCTION_H above the foot, patch points within JUNCTION_R x R)
        # smoothed along its vertex normals (Taubin): the trailing-edge base / leading-edge nose creases and the
        # flank's start become one rounded fillet
        blade_rest, g = self._weld_junction_smooth(np.asarray(local["vertices"])[:, 2] / local["H"], blade_rest, g, Ex_ind, Rcm, Ti @ W.e_n)
        out = {"local": local, "inp": inp, "g": g, "blade_rest": blade_rest, "fade": fd, "col": col, "anchors": anchors,
               "levels": levels, "Ex_rest": Ex_ind, "blade_a": W.chart(blade_rest)[:, 0], "lift_dir": Ti @ W.e_n,
               "frame_i": frame_i}
        if len(cache) > 64:
            cache.clear()
        cache[key] = out
        return out

    def _blade_local_faces(self):
        return self._blade_local(None)["faces"]          # the same topology for every animal

    def _weld_junction_smooth(self, hl, blade_rest, g, Ex_ind, Rcm, dn):
        from factory import blade as bl, weld as wd
        from scipy import sparse
        W = self.weld; R2 = 2 * bl.NC; nI = R2 + W.n_int
        X = self._weld_assemble(Ex_ind, blade_rest, g["rest"]).copy()
        n0 = self.n_rig; nb = len(blade_rest)
        w = np.zeros(len(X))
        hb = np.clip(hl, wd.H_FOOT, None)
        w[n0:n0 + nb] = 1.0 - wd.smoothstep((hb - wd.H_FOOT) / wd.JUNCTION_H)
        w[n0 + nb - 1] = 0.0
        # the knife-thin trailing edge and the thin nose are held (their two sides crossed when moved along normals)
        colb = np.arange(nb) % R2
        ends = (colb == 0) | (colb == R2 - 1) | (colb == bl.NC - 1) | (colb == bl.NC)
        w[n0:n0 + nb][ends] = 0.0
        wp = 1.0 - wd.smoothstep(g["d_foot"][R2:nI] / (wd.JUNCTION_R * Rcm))
        w[self._off_patch:self._off_cut] = wp
        cf = W.cfaces[(W.cfaces < nI).all(1)]
        F = np.concatenate([np.asarray(self._blade_local_faces(), np.int64) + n0,
                            np.where(cf < R2, n0 + cf, self._off_patch + cf - R2)])
        E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
        n = len(X)
        A = sparse.coo_matrix((np.ones(2 * len(E)), (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])), shape=(n, n)).tocsr()
        A.data[:] = 1.0
        deg = np.asarray(A.sum(1)).ravel()
        Lop = sparse.diags(1.0 / np.maximum(deg, 1)) @ A
        mv = w > 0
        X0 = X.copy()
        for it in range(2 * wd.JUNCTION_IT):
            T = X[F]; fn = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
            N = np.zeros_like(X)
            for k in range(3):
                np.add.at(N, F[:, k], fn)
            N /= np.linalg.norm(N, axis=1, keepdims=True) + 1e-12
            # the patch (a height field over the chart along dn) moves along dn only: its rows cannot cross
            N[self._off_patch:self._off_cut] = dn
            f = (0.5 if it % 2 == 0 else -0.53) * w[:, None]
            d = Lop @ X - X
            X[mv] = X[mv] + (f * (d * N).sum(1, keepdims=True) * N)[mv]
        dX = X - X0
        blade_rest = blade_rest + dX[n0:n0 + nb]
        sh = np.zeros_like(g["rest"])
        sh[R2:nI] = dX[self._off_patch:self._off_cut]
        sh[:R2] = dX[n0:n0 + R2]
        return blade_rest, {**g, "rest": g["rest"] + sh, "off": g["off"] + sh}

    def _weld_posed(self, rig_posed: np.ndarray, ind, scale: float) -> np.ndarray:
        """The welded whale posed: patch points on the posed back (filled old fin, cut vertices interpolated) lifted by
        the scaled hump along the posed smooth normal; the blade rigid at its base (round 11), its lowest rings blended
        into the welded foot."""
        from factory import blade as bl, weld as wd
        A = self._weld_animal(ind); W = self.weld; R2 = 2 * bl.NC
        if getattr(self, "weld_design", 13) == 15:               # round 15: posing of factory.weld15 ("foot" mode)
            from factory import weld15
            return weld15.posed(self, A, rig_posed, scale)
        Ex = W.ext(rig_posed)
        N = W.base_normals(Ex) * self._weld_sign
        g = A["g"]
        nI = R2 + W.n_int
        # round 13: ONE rigid fit at the fin base (the back vertices under the fin, round 11's anchors) carries the
        # blade, the patch near its foot, and the direction of every hump offset; the patch's base follows the posed
        # back. Round 12 rotated the offsets per 15 cm bin: in a concave bend lifted rows 0.3-1 cm apart converged and
        # crossed. With one direction the hump stays a height field over the posed back (fold-free while the back under
        # the region turns less than 90 degrees); near the foot the patch blends to the rigid copy of its rest shape
        # between RIGID_D0 and RIGID_D1 x R.
        an = A["anchors"]
        Rm, tm = bl.kabsch(A["Ex_rest"][an] * scale, Ex[an])
        dn_all = np.repeat((Rm @ A["lift_dir"])[None, :], len(g["a"]), 0)
        n_lin = wd.base_points(W, Ex, N, g["tri"], g["bary"], np.zeros(len(g["a"])))[1]
        bm = np.concatenate([g["bmix"], [1.0, 1.0]])
        base, n_p = wd.base_points(W, Ex, N, g["tri"], g["bary"], g["wph"], wd.smooth_dirs(n_lin, dn_all, bm))
        lp = Ex[W.loop_rep]
        base[:nI] = wd.smooth_base(W, base[:nI], lp, wd.smooth_dirs(n_p[:nI], dn_all[:nI], g["bmix"]), g["wsm"])
        P = base + (g["off"] * scale) @ Rm.T
        wr = g["wr"][:, None]
        if wd.RIGID_D1 > 0:
            P = P * (1.0 - wr) + ((g["rest"] * scale) @ Rm.T + tm) * wr
        if wd.BLADE_BINS:
            # a long blade (its foot spans up to 2 m of back) follows a bending back: carried per 15 cm bin (round 12)
            bins = self.__dict__.get("_weld_bins")
            if bins is None:
                bins = self._weld_bins = wd.bend_bins(W)
            Bp = wd.transform_by_bins(A["blade_rest"] * scale, A["blade_a"], wd.bin_transforms(bins, A["Ex_rest"] * scale, Ex))
        else:
            Bp = (A["blade_rest"] * scale) @ Rm.T + tm
        # the blade's lowest rings blend onto the posed foot (zero at rest: the foot is the nominal ring there)
        Bp += A["fade"][:, None] * (P[:R2] - Bp[:R2])[A["col"]]
        Bp[:R2] = P[:R2]
        return self._weld_assemble(Ex, Bp, P)

    def _weld_canonical(self, A: dict) -> np.ndarray:
        from factory import blade as bl
        W = self.weld; R2 = 2 * bl.NC
        F = W.base_faces[A["g"]["tri"]]
        pts = (self._can_ext[F] * A["g"]["bary"][..., None]).sum(1)
        a_i = int(A["anchors"][0]) if len(A["anchors"]) else 0
        blade = A["blade_rest"] + (self._can_ext[a_i] - W.R_rest[a_i])
        out = self._weld_assemble(self._can_ext, blade, pts)
        return out

    def canonical_for(self, ind) -> np.ndarray:
        """round 12: per-animal canonical vertices (the welded patch lies on different parts of the back per animal)."""
        if getattr(self, "weld", None) is None or ind is None:
            return self.canonical
        cache = self.__dict__.setdefault("_weld_can_cache", {})
        key = self._weld_key(ind)
        if key not in cache:
            if len(cache) > 64:
                cache.clear()
            cache[key] = self._weld_canonical(self._weld_animal(ind))
        return cache[key]

    def weld_base_over_height(self, ind) -> float:
        """round 13: the welded fin's base / H at rest: front-base landmark minus back-base landmark along the fin frame
        (the same numbers as factory.weld_check's outline and the exported keypoints)."""
        A = self._weld_animal(ind)
        B, al, _ = A["frame_i"]
        fb, bb = A["g"]["rest"][-2], A["g"]["rest"][-1]
        return float((fb - bb) @ al / A["local"]["H"])

    def blade_weight(self, ind=None) -> np.ndarray | None:
        """The dorsal blade sub-label per vertex: the blade's own vertices (round 12: not the hump patch)."""
        if getattr(self, "weld", None) is None:
            return None
        w = np.zeros(len(self.rest)); w[self.n_rig: self.n_rig + self.n_blade] = 1.0
        return w

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
        hump = getattr(self, "dorsal_hump", None)
        ckey = spec.id if not hump else (spec.id, "hump20", json.dumps(hump, sort_keys=True))
        if ckey in cache:
            return cache[ckey]
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
        spec_e = spec
        if hump and hump.get("mode") == "3c":
            # round 20e variant: the individuals' own 3c hump (asset-fin frame, cm) with a drawn height instead of the library's 0
            hp = self.hump20_params(spec)
            if hp is not None:
                from dataclasses import replace as _replace
                spec_e = _replace(spec, dorsal_hump_cm=float(hp["cm"]))
        for mi in range(len(self.rig.js["meshes"])):
            rest_world = self.rig.pose(None, mi)
            edited = indiv.apply_rest(rest_world, spec_e, joints, regions if mi == self.rig.mesh_index else {})
            if hump and hump.get("mode", "r20e") != "3c" and mi == self.rig.mesh_index and getattr(self, "n_rig", None) is not None:
                # round 20e (whale.dorsal_hump): the hump under the welded fin, on the edited rest back (model frame cm)
                d = self._hump20_disp(spec, edited * self.length_scale - self.centre)
                if d is not None:
                    edited = edited + d / self.length_scale
            positions[mi] = self.rig.bind_from_rest(edited, mi)
        entry = {"positions": positions, "spec": spec}
        if len(cache) > 32:
            cache.clear()
        cache[ckey] = entry
        return entry

    # ---- run06 round 20e: a convex hump under the welded fin (whale.dorsal_hump; dormant when absent) ------------------
    def hump20_params(self, spec) -> dict | None:
        """The individual's hump draws (deterministic per id) from whale.dorsal_hump:
          k [lo, hi]        hump height / the blade's height H (uniform), 0 for a share p_none of the animals
          front [lo, hi]    the plateau's front edge, in chords from the front base landmark (+ ahead)
          front_sig [lo, hi] the front fall's width (normal CDF sigma), chords
          rear [lo, hi]     the plateau's rear edge, chords behind the back base landmark (+ behind)
          rear_sig [lo, hi] the rear fall's width, chords
          half_width_cm     lateral cos^2 fall to 0 at this distance from the midline (default 70)
          depth_cm [d0, d1] full weight down to d0 below the back's top line, none from d1 (default [60, 150])
          max_cm            cap on the height (default 40)
          rear_max_cm, front_max_cm  caps (cm) on rear / rear_sig and front / front_sig times the chord (default none)
          calf_k            factor on k for calves (default 1)
          mode "3c"         instead: the individuals' own 3c hump (factory.individuals, asset-fin frame) of cm [lo, hi] cm
        chord = front base - back base along the blade's base line (the landmarks' places on the template)."""
        cfg = getattr(self, "dorsal_hump", None)
        if not cfg or spec is None or int(getattr(spec, "blade_template", -1)) < 0:
            return None
        import hashlib
        rng = np.random.default_rng(int(hashlib.sha1(f"hump20:{spec.id}".encode()).hexdigest()[:8], 16))
        u = rng.uniform(size=6)
        rng_ = lambda key, dflt, x: float(cfg.get(key, dflt)[0] + (cfg.get(key, dflt)[1] - cfg.get(key, dflt)[0]) * x)
        k = rng_("k", (0.5, 1.0), u[0]) * (float(cfg.get("calf_k", 1.0)) if getattr(spec, "calf", False) else 1.0)
        if u[1] < float(cfg.get("p_none", 0.0)):
            k = 0.0
        if cfg.get("mode") == "3c":
            return {"cm": 0.0 if k == 0.0 else rng_("cm", (10.0, 30.0), u[0])}
        return {"k": k, "front": rng_("front", (0.0, 0.0), u[2]), "front_sig": rng_("front_sig", (0.4, 0.4), u[3]),
                "rear": rng_("rear", (0.0, 0.0), u[4]), "rear_sig": rng_("rear_sig", (1.0, 1.0), u[5]),
                "half_width_cm": float(cfg.get("half_width_cm", 70.0)), "depth_cm": tuple(cfg.get("depth_cm", (60.0, 150.0))),
                "max_cm": float(cfg.get("max_cm", 40.0)), "rear_max_cm": cfg.get("rear_max_cm"), "front_max_cm": cfg.get("front_max_cm")}

    def _hump20_disp(self, spec, P: np.ndarray) -> np.ndarray | None:
        """(n, 3) displacement (model frame cm) of the rest rig vertices P for the individual's hump, or None. Along the
        blade frame's up: A * w_s(s) * w_lat(y) * w_top(depth), w_s = Phi((s - a) / sig_r) * (1 - Phi((s - b) / sig_f)) (a smooth
        plateau from a to b, s along the blade's base line from its back base), A = k * H (capped)."""
        from scipy.special import ndtr
        from factory import weld as wd
        p = self.hump20_params(spec)
        if p is None or p["k"] <= 0.0:
            return None
        local = self._blade_local_spec(spec)
        H = float(local["H"])
        B, along, up = self._blade_frame_for(local)
        o = local["outline"]; Lv, te_s, le_s = np.asarray(o["levels"]), np.asarray(o["te"]), np.asarray(o["le"])
        hh = np.linspace(0.0, wd.H_FOOT, 61)
        te_h, le_h = np.interp(hh, Lv, te_s), np.interp(hh, Lv, le_s)
        a_lo, a_hi, a_ft, a_fl = float(te_h.min()), float(le_h.max()), float(te_h[-1]), float(le_h[-1])
        Rcm = wd.SKIRT_R * H
        fb_a = max(a_hi * H, a_fl * H + wd.LM_MIN * Rcm) if a_hi > a_fl + 1e-6 else a_fl * H + wd.LM_MIN * Rcm
        bb_a = min(a_lo * H, a_ft * H - wd.LM_MIN * Rcm) if a_lo < a_ft - 1e-6 else a_ft * H - wd.LM_MIN * Rcm
        c = max(fb_a - bb_a, 1.0)
        A = min(p["k"] * H, p["max_cm"])
        rel = P - np.asarray(B, float)
        s = rel @ along
        cap = lambda v, key: min(v, float(p[key])) if p.get(key) is not None else v
        a = bb_a - cap(p["rear"] * c, "rear_max_cm"); b = fb_a + cap(p["front"] * c, "front_max_cm")
        sr = cap(p["rear_sig"] * c, "rear_max_cm"); sf = cap(p["front_sig"] * c, "front_max_cm")
        ws = ndtr((s - a) / max(sr, 1e-3)) * (1.0 - ndtr((s - b) / max(sf, 1e-3)))
        yl = np.clip(np.abs(P[:, 1]) / p["half_width_cm"], 0.0, 1.0)
        wl = np.cos(0.5 * np.pi * yl) ** 2
        xs, zs = self._back_profile
        depth = np.interp(P[:, 0], xs, zs) - P[:, 2]
        d0, d1 = (float(v) for v in p["depth_cm"])
        t = np.clip((depth - d0) / max(d1 - d0, 1e-6), 0.0, 1.0)
        wt = 1.0 - t * t * (3.0 - 2.0 * t)
        w = ws * wl * wt
        self.__dict__.setdefault("_hump20_info", {})[spec.id] = {"A_cm": A, "chord_cm": c, "H": H, "w": w, **p}
        return np.outer(A * w, up)

    def _pec_membership(self) -> dict:
        """Smooth per-vertex membership of each pectoral fin (chain skin weight, cut off at the shoulder so the
        back never moves): {"pec_left": (N,), "pec_right": (N,)}, cached."""
        mem = self.__dict__.get("_pec_mem")
        if mem is None:
            from factory.individuals import _smooth
            from scipy import sparse
            regions = self.rig.pec_regions(); y = np.abs(self.rig.rest[:, 1])
            # the skin weights jump between neighbours at the armpit: smooth them over the mesh (1-ring mean, 6x)
            F = self._rig_faces(); n = len(self.rig.rest)
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
            F = self._rig_faces()
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
        if getattr(self, "weld", None) is not None and self.kind == "humpback4k":
            Vw = self._weld_posed(V[: self.n_rig], shot.get("individual"), float(shot.get("body_scale", 1.0)))
            if getattr(self, "weld_smooth", None):         # round 20d: the fin foot / hump smoothed along its normals
                Vw = self._weld_smooth_safe(Vw, shot.get("individual"), self.weld_smooth)
            return Vw
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
            ab = getattr(self, "arch_bend", None)
            if ab:                                       # run06 round 20c (whale.arch_bend): bent about the body's own centre line
                return deform.deform_centreline(base, params, self._arch_exclude(len(base)), neutral=float(ab.get("neutral", 0.0)))
            return deform.deform(base, params, None)
        # The clip bends the body and the pectoral angle is a joint matter (pec_droop_deg, individuals): the
        # procedural fin rotation (deform.rotate_fins) picks its "fin" vertices by a K9239 heuristic that tears
        # this mesh at the shoulder, so none of the procedural parameters apply to clip frames.
        return base

    # ---- run06 round 20d (creases at the fin foot, the keel behind the fin) -------------------------------------------
    R20D_WELD_REST = ("K_BACK", "LOW_RATE", "LOW_T", "SIG_MIN")

    def apply_r20d(self, whale: dict) -> None:
        """Round-20d keys of the config's whale section (each absent = as before): weld_rest {weld15 constant: value}
        (K_BACK, LOW_RATE, LOW_T, SIG_MIN),
        weld_smooth {iters, lam, mu, ring_h, loop_d} (_weld_smooth_posed)."""
        if whale.get("weld_rest"):
            bad = set(whale["weld_rest"]) - set(self.R20D_WELD_REST)
            if bad:
                raise KeyError(f"whale.weld_rest: unknown {sorted(bad)} (allowed {self.R20D_WELD_REST})")
            self.weld_rest = dict(whale["weld_rest"])
        if whale.get("weld_smooth"):
            self.weld_smooth = dict(whale["weld_smooth"])

    def _weld_smooth_safe(self, V: np.ndarray, ind, opts: dict) -> np.ndarray:
        """_weld_smooth_posed with the weld gate's self-crossing test (weld15._crossings): a smoothing that makes the
        whale cross itself (knife-thin trailing-edge feet: 20 of 302 check20 whales at 30 passes) is retried with half
        the passes, then dropped (the unsmoothed posing is returned). model._weld_smooth_info = passes used."""
        from factory import weld15
        it = int(opts.get("iters", 30))
        while it >= 4:
            Vs = self._weld_smooth_posed(V, ind, {**opts, "iters": it})
            if weld15._crossings(self, Vs) == 0:
                self._weld_smooth_info = it
                return Vs
            it //= 2
        self._weld_smooth_info = 0
        return V

    def _weld_smooth_posed(self, V: np.ndarray, ind, opts: dict) -> np.ndarray:
        """round 20d: Taubin smoothing (lam, mu; iters passes) of the posed weld along the vertex normals: the patch (fin
        foot, fillet, hump) with weight 1 out to loop_d[0] x H from the foot ring, falling to 0 at loop_d[1] x H (the
        region's loop and the rig stay), and the blade's rings up to ring_h x H above the foot ring (fading). Moves along
        the normal only, so vertices keep their place on the surface (labels, UVs); the outline of the fin above
        ring_h is untouched. A kink of 6-12 deg at the foot ring and the fillet's tight bend caught the sky / water
        horizon in glossy light as a crisp line (000019, 000077 of r06r_check20)."""
        from factory import blade as bl, weld as wd
        import scipy.sparse as _sp
        V = np.asarray(V, float).copy(); n = len(V); F = np.asarray(self.faces)
        Lm = self.__dict__.get("_r20d_lap")
        if Lm is None or Lm.shape[0] != n:
            i = np.concatenate([F[:, 0], F[:, 1], F[:, 2], F[:, 1], F[:, 2], F[:, 0]])
            j = np.concatenate([F[:, 1], F[:, 2], F[:, 0], F[:, 0], F[:, 1], F[:, 2]])
            Lm = _sp.csr_matrix((np.ones(len(i)), (i, j)), shape=(n, n)); Lm.data[:] = 1.0
            Lm = _sp.diags(1.0 / np.maximum(np.asarray(Lm.sum(1)).ravel(), 1.0)) @ Lm
            self._r20d_lap = Lm
        A = self._weld_animal(ind); R2 = 2 * bl.NC; nI = R2 + self.weld.n_int
        H = float(A["local"]["H"])
        w = np.zeros(n)
        d0, d1 = (float(v) for v in opts.get("loop_d", (0.5, 1.0)))
        t = np.clip((A["g"]["d_foot"][R2:nI] / H - d0) / max(d1 - d0, 1e-6), 0.0, 1.0)
        w[self._off_patch: self._off_cut] = 1.0 - t * t * (3.0 - 2.0 * t)
        nb = len(A["blade_rest"])
        t = np.clip((np.asarray(A["levels"], float) - wd.H_FOOT) / max(float(opts.get("ring_h", 0.1)), 1e-6), 0.0, 1.0)
        wb = 1.0 - t * t * (3.0 - 2.0 * t); wb[-1] = 0.0
        w[self.n_rig: self.n_rig + nb] = wb
        act = np.nonzero(w > 0)[0]; wa = w[act]
        lam, mu, it = float(opts.get("lam", 0.5)), float(opts.get("mu", -0.53)), int(opts.get("iters", 30))

        def vnorm(X):
            fn = np.cross(X[F[:, 1]] - X[F[:, 0]], X[F[:, 2]] - X[F[:, 0]])
            vn = np.zeros_like(X)
            for k in range(3):
                np.add.at(vn, F[:, k], fn)
            return vn / (np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12)
        N = vnorm(V)
        for k in range(it):
            if k and k % 5 == 0:
                N = vnorm(V)
            for f in (lam, mu):
                D = (Lm @ V - V)[act]
                V[act] += (f * wa * (D * N[act]).sum(1))[:, None] * N[act]
        return V

    def _arch_exclude(self, n: int) -> np.ndarray:
        """Vertices left out of the centre line (round 20c): the pectorals and the rig's own dorsal fin."""
        key = ("_arch_excl", n)
        if key not in self.__dict__:
            m = np.zeros(n, bool); pg = self.part_groups()
            for name in ("pectoral", "dorsal"):
                idx = np.asarray(pg.get(name, []), int)
                m[idx[idx < n]] = True
            self.__dict__[key] = m
        return self.__dict__[key]

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
        if getattr(self, "weld", None) is not None:
            # run06 round 12: the tip on the welded blade, the front / back base where the hump's midline meets the back
            # (two landmark vertices of the weld, on the posed surface), the edge chains on the blade from the top to its
            # foot ring
            from factory import blade as bl, weld as wd
            n0, R2 = self.n_rig, 2 * bl.NC
            idx = {k: v for k, v in idx.items() if not k.startswith(("fin_te_", "fin_le_"))}
            idx["dorsal_fin_tip"] = n0 + self.n_blade - 1
            idx["dorsal_fin_front"], idx["dorsal_fin_back"] = self._off_lm, self._off_lm + 1
            ks = [int(round(v)) for v in np.linspace(wd.NL_W - 1, 0, 10)]
            for j, k in enumerate(ks):
                idx[f"fin_te_{j:02d}"] = n0 + k * R2 + 0
                idx[f"fin_le_{j:02d}"] = n0 + k * R2 + bl.NC - 1
        elif getattr(self, "n_rig", None) is not None:
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
            if getattr(self, "weld", None) is not None:               # round 12: blade + hump patch + landmarks (default animal)
                bl_idx = np.arange(self.n_rig, len(R))
                for k in list(out):
                    out[k] = np.setdiff1d(out[k], bl_idx)
                dors = self._weld_dorsal(None)
                out["dorsal"] = np.union1d(out["dorsal"], bl_idx[dors[self.n_rig:]])
                out["body"] = np.union1d(out["body"], bl_idx[~dors[self.n_rig:]])
            elif getattr(self, "n_rig", None) is not None:              # the blade is dorsal fin, whole
                bl_idx = np.arange(self.n_rig, len(R))
                for k in list(out):
                    out[k] = np.setdiff1d(out[k], bl_idx)
                out["dorsal"] = np.union1d(out["dorsal"], bl_idx)
        self._part_groups = out
        return out

    def vertex_part_ids(self, individual=None) -> np.ndarray | None:
        """(N,) uint8 PART_IDS per vertex (part_groups), None for the static model. round 12 (weld): the hump patch's
        vertices are dorsal where that animal's hump lifts them (individual), else body."""
        groups = self.part_groups()
        if not groups:
            return None
        ids = np.zeros(len(self.rest), np.uint8)
        for name, sel in groups.items():
            ids[sel] = PART_IDS[name]
        if getattr(self, "weld", None) is not None and individual is not None:
            dors = self._weld_dorsal(individual)
            new = np.arange(self.n_rig, len(self.rest))
            ids[new] = np.where(dors[new], PART_IDS["dorsal"], PART_IDS["body"])
        return ids

    def _weld_dorsal(self, ind) -> np.ndarray:
        """(N,) bool: the blade, the landmarks and the patch points the animal's hump lifts by more than
        DORSAL_D_MIN of its foot height."""
        from factory import blade as bl, weld as wd
        A = self._weld_animal(ind); R2 = 2 * bl.NC; nI = R2 + self.weld.n_int
        out = np.zeros(len(self.rest), bool)
        out[self.n_rig: self.n_rig + self.n_blade] = True
        out[self._off_lm:] = True
        out[self._off_patch: self._off_cut] = A["g"]["D"][R2:nI] > wd.DORSAL_D_MIN * wd.H_FOOT * A["local"]["H"]
        return out

    def weld_dorsal_soft(self, ind) -> np.ndarray | None:
        """round 13: (N,) soft dorsal weight on the hump patch's vertices (NaN elsewhere): 0.5 where the lift is at the
        dorsal threshold, linear in the lift over +-DORSAL_SOFT_BAND x threshold (factory.project.animal_maps)."""
        if getattr(self, "weld", None) is None or ind is None:
            return None
        from factory import blade as bl, weld as wd
        A = self._weld_animal(ind); R2 = 2 * bl.NC; nI = R2 + self.weld.n_int
        thr = wd.DORSAL_D_MIN * wd.H_FOOT * A["local"]["H"]
        out = np.full(len(self.rest), np.nan)
        D = A["g"]["D"][R2:nI]
        out[self._off_patch: self._off_cut] = np.clip(0.5 + (D - thr) / (2.0 * wd.DORSAL_SOFT_BAND * thr), 0.0, 1.0)
        return out

    def write_obj(self, vertices: np.ndarray, path: Path, comment: str = "posed whale", individual=None) -> None:
        """Posed OBJ with the model's groups (body, eye...) for the renderer. With the parametric blade, the blade's UVs
        are the animal's own two side charts (factory.blade_uv2, round 9), with OBJ-level copies of its seam vertices."""
        if getattr(self, "weld", None) is not None:
            return self._write_obj_weld(vertices, Path(path), comment, individual)
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
        _w(Path(path), groups, comment, winding_fix=bool(getattr(self, "winding_fix", False)))   # round 18b

    def _write_obj_weld(self, vertices: np.ndarray, path: Path, comment: str, individual) -> None:
        """round 12: one body group, one surface. UVs per face corner: the rig's (cut vertices interpolated), the
        blade's own two charts (factory.blade_uv2, seam copies), the back's UVs on the patch (barycentric in the cut-out
        rig faces; a patch face spanning the back's midline UV seam takes its side's island for every corner). OBJ
        vertices are (vertex, uv) pairs, so UV seams split positions only in the file; normals are smooth over
        positions."""
        from factory import blade as bl, blade_uv2, weld as wd
        from factory.humpback4k import write_obj as _w
        V = np.asarray(vertices, float)
        A = self._weld_animal(individual); W = self.weld; g = A["g"]
        n0, R2 = self.n_rig, 2 * bl.NC; nI = R2 + W.n_int
        uv_v = np.zeros((len(V), 2))
        uv_v[:n0] = self.rig.uv
        uv_v[self._off_patch: self._off_cut] = g["uv"][R2:nI]
        uv_v[self._off_cut: self._off_lm] = W.uv_ext[n0:]
        uv_v[self._off_lm:] = g["uv"][nI:]
        F = self.faces
        corner_v = F.copy(); corner_uv = uv_v[F].copy()
        # blade faces: its charts, with the seam copies' UVs
        b0, b1 = self._face_ranges["blade"]
        Lb = np.asarray(A["local"]["vertices"], float)
        uv_b, info = blade_uv2.blade_uv(Lb)
        Fb = F[b0:b1] - n0
        Vb2, Fb2, UVb2 = blade_uv2.seam_split(Lb, Fb, uv_b, info, 0)
        corner_uv[b0:b1] = UVb2[Fb2]
        # (the copies are positions of seam cols / tip: map them back to their blade vertex)
        src = np.concatenate([np.arange(len(Lb)), info["seam"]["cols"], [info["seam"]["tip"]]])
        corner_v[b0:b1] = n0 + src[Fb2]
        # patch faces: foot-ring corners take the back's UV there; a face across a UV seam takes its side's island
        p0, p1 = self._face_ranges["patch"]
        Fp = F[p0:p1]
        foot = (Fp >= n0) & (Fp < n0 + R2)
        cu = corner_uv[p0:p1]
        cu[foot] = g["uv"][:R2][Fp[foot] - n0]
        jump = np.max([np.linalg.norm(cu[:, k] - cu[:, (k + 1) % 3], axis=1) for k in range(3)], axis=0) > wd.UV_JUMP
        if jump.any():
            Cp = self._weld_chart_of(Fp[jump], A)
            for n_, fi in enumerate(np.nonzero(jump)[0]):
                c2 = Cp[n_]; side = 1.0 if c2[:, 1].mean() >= 0 else -1.0
                ref = int(np.argmax(side * c2[:, 1]))
                tri, bary, _ = wd.locate(W, c2[[ref]])
                T = W.chart(W.R_rest)[W.base_faces[tri[0]]]
                M = np.array([[T[0, 0], T[1, 0], T[2, 0]], [T[0, 1], T[1, 1], T[2, 1]], [1.0, 1.0, 1.0]])
                Uf = W.uv_ext[W.base_faces[tri[0]]]
                for k in range(3):
                    if np.linalg.norm(cu[fi, k] - cu[fi, ref]) > wd.UV_JUMP:
                        b = np.linalg.solve(M, np.array([c2[k, 0], c2[k, 1], 1.0]))
                        cu[fi, k] = b @ Uf
        corner_uv[p0:p1] = cu
        if getattr(self, "blade_uv_match", None):        # round 20g: blade charts placed to match the patch's base colour
            self._r20g_blade_uv_match(V, corner_uv, Lb, Fb2, src, info, individual, A)
        # OBJ vertices = unique (vertex, uv)
        key = np.concatenate([corner_v.reshape(-1, 1).astype(float), np.round(corner_uv.reshape(-1, 2), 7)], 1)
        uniq, inv = np.unique(key, axis=0, return_inverse=True)
        inv = inv.ravel()
        ov = uniq[:, 0].astype(int)
        Vo = V[ov]; UVo = uniq[:, 1:]; Fo = inv.reshape(-1, 3)
        # smooth normals over positions (the importer recomputes them the same way)
        tri = V[corner_v]
        fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        _, pid = np.unique(np.round(V, 3), axis=0, return_inverse=True); pid = pid.ravel()
        acc = np.zeros((int(pid.max()) + 1, 3))
        for k in range(3):
            np.add.at(acc, pid[corner_v[:, k]], fn)
        Nv = acc[pid]
        if (np.einsum("ij,ij->i", fn, tri.mean(1) - V[: self.n_rig].mean(0)) < 0).mean() > 0.5:
            Nv = -Nv
        Nv /= np.linalg.norm(Nv, axis=1, keepdims=True) + 1e-12
        sv = getattr(self, "obj_shared_vertices", None)
        if sv:
            # round 20g (whale.obj_shared_vertices {scope}): UV-seam copies written as one OBJ position so the engine's
            # recomputed normal is one across the seam. scope "foot" = the blade-to-patch foot ring only; "model" = every
            # copy of one model vertex (foot ring, blade edge / tip seams, patch faces across the back's midline seam);
            # "position" = also coincident rig vertices (the rig's own UV seams).
            scope = sv.get("scope", "foot")
            pos = np.arange(len(ov)) + len(V)                      # default: every OBJ vertex its own position
            if scope == "foot":
                m = (ov >= n0) & (ov < n0 + R2)
                pos[m] = ov[m]
            elif scope == "model":
                pos = ov.copy()
            elif scope == "position":
                pos = pid[ov].copy()
            else:
                raise ValueError(f"whale.obj_shared_vertices.scope: {scope!r} (foot / model / position)")
            _w(Path(path), [("body", Vo, Fo, UVo, Nv[ov], pos)], comment, winding_fix=bool(getattr(self, "winding_fix", False)))
            return
        _w(Path(path), [("body", Vo, Fo, UVo, Nv[ov])], comment, winding_fix=bool(getattr(self, "winding_fix", False)))   # round 18b

    _R20G_SKIN = {}

    def _r20g_blade_uv_match(self, V, corner_uv, Lb, Fb2, src, info, individual, A=None) -> None:
        """round 20g (whale.blade_uv_match {skin_suffix, band_cm, step, reach}): the blade's two UV charts sample the skin's
        always-dark region (factory.blade_uv2) and the patch beside the foot ring samples the back's own map there, so the
        base colour steps across the ring (40 check20 whales: |blade - patch| luminance q50 / q90 ~6 / 21 %), a crisp line
        in the engine. Among the chart placements blade_uv2 allows (same scale, both arrangements, shifts on a finer grid,
        both charts and their seam copies inside the dark-region mask), take the one whose band within band_cm of the ring
        has the patch band's mean luminance on each side (sum of |difference| over the two sides). Only the blade's face
        corner UVs change (in place); geometry, faces, labels untouched. No skin map file -> nothing changes."""
        from factory import blade as bl, blade_uv2 as U
        cfg = self.blade_uv_match
        skin = f"{individual}{cfg.get('skin_suffix', '')}"
        path = REPO_ROOT / "third_party_assets" / "humpback_4k" / "skins" / f"T_Humpback_BaseColor_{skin}.png"
        if not path.exists():
            self._r20g_uv_info = {"skin": skin, "skipped": "no skin map"}
            return
        if skin not in self._R20G_SKIN:
            from PIL import Image
            a = np.asarray(Image.open(path).convert("RGB"), np.float32)
            self._R20G_SKIN.clear()
            self._R20G_SKIN[skin] = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
        BC = self._R20G_SKIN[skin]
        h, w = BC.shape

        def lum(uv):                                    # model UV (the engine's texel row = t * h); bilinear
            x = (uv[:, 0] % 1.0) * w - 0.5; y = (uv[:, 1] % 1.0) * h - 0.5
            x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int); fx = x - x0; fy = y - y0
            g = lambda yy, xx: BC[np.clip(yy, 0, h - 1), np.clip(xx, 0, w - 1)]
            return ((g(y0, x0) * (1 - fx) + g(y0, x0 + 1) * fx) * (1 - fy) + (g(y0 + 1, x0) * (1 - fx) + g(y0 + 1, x0 + 1) * fx) * fy)
        n0, R2 = self.n_rig, 2 * bl.NC
        band = float(cfg.get("band_cm", 4.0))
        from scipy.spatial import cKDTree
        F = self.faces; b0, b1 = self._face_ranges["blade"]; p0, p1 = self._face_ranges["patch"]
        # band membership from REST geometry only (blade-local cm; the patch's rest chart, cm), so the placement is the
        # same in every shot of an individual (its fin marks stay put)
        A = A if A is not None else self._weld_animal(individual)
        Fbl = np.where(Fb2 < len(Lb), Fb2, 0); Fbl = src[Fb2]
        db, rb = cKDTree(Lb[:R2]).query(Lb[Fbl].mean(1))
        Cp = self._weld_chart_of(F[p0:p1], A)
        dp, rp = cKDTree(np.asarray(A["g"]["P2"], float)[:R2]).query(Cp.mean(1))
        target = []
        for k in (0, 1):
            sel = (dp < band) & ((rp >= bl.NC) == bool(k))
            target.append(float(lum(corner_uv[p0:p1][sel].reshape(-1, 2)).mean()) if sel.any() else None)
        selb = db < band
        if not selb.any() or None in target:
            self._r20g_uv_info = {"skin": skin, "skipped": "empty band"}
            return
        # blade face corners of the band: blade-local vertex, chart (seam copies of the edges: right chart; tip copy: left)
        nL = len(Lb); side = U._side(nL)
        nb_cols = len(info["seam"]["cols"])
        Fsel = Fb2[selb]
        chart = np.where(Fsel < nL, side[np.clip(Fsel, 0, nL - 1)], np.where(Fsel < nL + nb_cols, 0, 1))
        loc = src[Fsel]
        stc = np.stack([np.where(chart == 1, -Lb[loc, 0], Lb[loc, 0]), Lb[loc, 2]], -1)
        fside = (rb[selb] >= bl.NC).astype(int)
        st_all = np.stack([np.where(side == 1, -Lb[:, 0], Lb[:, 0]), Lb[:, 2]], 1); seam = info["seam"]
        step = float(cfg.get("step", 0.01)); reach = float(cfg.get("reach", 0.10))
        grid = np.arange(-reach, reach + 1e-9, step)

        def corners(arr, sh):
            fr = U._frames(st_all, side, arr)
            return U._to_uv(stc.reshape(-1, 2), chart.ravel(), fr, info["scale"], sh).reshape(stc.shape)

        def score(uvc):
            m = lum(uvc.reshape(-1, 2)).reshape(uvc.shape[:2]).mean(1)
            return sum(abs(float(m[fside == k].mean()) - target[k]) for k in (0, 1) if (fside == k).any())
        best = (score(corners(info["arrangement"], info["shift"])), info["arrangement"], tuple(info["shift"]))
        first = best[0]
        for arr in ("stack", "row"):
            fr = U._frames(st_all, side, arr)
            for dx in grid:
                for dy in grid:
                    sh = (float(dx), float(dy))
                    if not (U._inside(U._to_uv(st_all, side, fr, info["scale"], sh)) and U._inside(U._to_uv(seam["st"], seam["side"], fr, info["scale"], sh))):
                        continue
                    s = score(corners(arr, sh))
                    if s < best[0] - 1e-6:
                        best = (s, arr, sh)
        self._r20g_uv_info = {"skin": skin, "score_before": round(first, 2), "score_after": round(best[0], 2), "arrangement": best[1], "shift": best[2]}
        if best[1] == info["arrangement"] and np.allclose(best[2], info["shift"]):
            return
        # every blade face corner with the chosen placement
        Fall = Fb2; chart_all = np.where(Fall < nL, side[np.clip(Fall, 0, nL - 1)], np.where(Fall < nL + nb_cols, 0, 1))
        la = src[Fall]
        st_c = np.stack([np.where(chart_all == 1, -Lb[la, 0], Lb[la, 0]), Lb[la, 2]], -1)
        fr = U._frames(st_all, side, best[1])
        corner_uv[b0:b1] = U._to_uv(st_c.reshape(-1, 2), chart_all.ravel(), fr, info["scale"], best[2]).reshape(st_c.shape)

    def _weld_chart_of(self, faces: np.ndarray, A: dict) -> np.ndarray:
        """Chart positions (n, 3, 2) of model vertices of patch faces for an animal (patch / foot / landmark points from
        its layout, rig and cut vertices from the rest base)."""
        from factory import blade as bl
        W = self.weld; n0, R2 = self.n_rig, 2 * bl.NC; nI = R2 + W.n_int
        Cx = W.chart(W.R_rest)
        out = np.zeros(faces.shape + (2,))
        f = faces
        m_foot = (f >= n0) & (f < n0 + R2)
        m_patch = (f >= self._off_patch) & (f < self._off_cut)
        m_cut = (f >= self._off_cut) & (f < self._off_lm)
        m_rig = f < n0
        out[m_foot] = A["g"]["P2"][f[m_foot] - n0]
        out[m_patch] = A["g"]["P2"][R2 + f[m_patch] - self._off_patch]
        out[m_cut] = Cx[n0 + f[m_cut] - self._off_cut]
        out[m_rig] = Cx[f[m_rig]]
        return out

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
