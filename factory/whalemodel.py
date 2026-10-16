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
from pathlib import Path

import numpy as np

from factory.skinned import MODEL_TO_PROJECT as MODEL_TO_PROJECT_

from factory import deform, poses, project

REPO_ROOT = Path(__file__).resolve().parents[1]
K9239_OBJ = REPO_ROOT / "third_party_assets" / "humpback_k9239" / "converted" / "SM_Humpback.obj"
HUMPBACK4K_GLTF = REPO_ROOT / "third_party_assets" / "humpback_4k" / "src_gltf" / "scene.gltf"

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
    "arch": [0.3, 0.6], "peduncle": [0.8, 1.0, 13.8, 25.5],
    "back_arch": [0.2, 0.3, 0.35, 0.45], "surface": [0.2, 0.3, 0.35, 0.45],     # straight-ish frames from the start of the clip; the hump is procedural (shotlist)
    "fluke_up": [0.8, 1.0, 1.2, 1.4, 1.6, 1.9, 2.2], "fluke_edge": [1.6, 1.9, 2.2], "fluke_down": [1.0, 1.2, 1.4, 1.6],   # the whole dive; the fluke curl (shotlist) adds the tips-down phase
    "breach": [2.9, 7.4, 16.2, 16.6],          # 15.6 tears the dorsal-fin base seam in the asset's skinning
    "pec_up": [0.0, 0.3, 12.0], "fin_tip": [0.0, 0.3, 12.0], "roll_belly": [0.0, 0.3, 12.0],
    "spyhop": [0.0, 2.9, 16.6], "spyhop_low": [0.0, 2.9, 16.6],
}
KEYFRAME_JITTER_S = 0.05
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

    @classmethod
    def from_config(cls, config: dict, rig=None) -> "WhaleModel":
        return cls((config.get("whale") or {}).get("kind", "k9239"), rig)

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
            lib = self._library = indiv.load_library()
        return lib

    def individual(self, ind):
        """Per-individual rig inputs (cached): rest positions per mesh (model frame) and joint offsets."""
        from factory import individuals as indiv
        if ind is None:
            return None
        if isinstance(ind, str):
            lib = getattr(self, "_library", None)
            if lib is None:
                lib = self._library = indiv.load_library()
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
        regions = self.rig.pec_regions()
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
            F = self.faces; n = len(self.rig.rest)
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

    def droop_pectorals(self, V: np.ndarray, jp: dict, droop_deg: float) -> np.ndarray:
        """Swing both pectoral fins down by droop_deg (vertex level, membership-weighted) about a hinge along the
        body axis through each fin's root joint; V and jp in the same frame (rig frame, cm)."""
        if abs(droop_deg) < 1e-6:
            return V
        out = V.copy()
        for side, root_name, sign in (("pec_left", "joint64_013", 1.0), ("pec_right", "joint70_019", -1.0)):
            w = self._pec_membership()[side]; idx = np.where(w > 1e-3)[0]
            if len(idx) == 0:
                continue
            root = np.asarray(jp[root_name]); ang = np.radians(sign * droop_deg) * w[idx]
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
        w = _smooth(-self.canonical[:, 0], 380.0, 620.0)              # 0 on the tail stock, 1 on the blade (a 2.4 m hinge: a -95 curl without creasing)
        idx = np.where(w > 1e-3)[0]
        hinge = V[(self.canonical[:, 0] > -530) & (self.canonical[:, 0] < -490)].mean(0)
        ang = np.radians(curl_deg) * w[idx]; c, s_ = np.cos(ang), np.sin(ang)
        rel = V[idx] - hinge; x, z = rel[:, 0], rel[:, 2]
        out = V.copy()
        out[idx, 0] = hinge[0] + c * x + s_ * z                      # rotation about +y through the hinge
        out[idx, 2] = hinge[2] - s_ * x + c * z
        return out

    def clip_pose(self, t: float | None, ind=None, pec_droop_deg: float = 0.0, fluke_curl_deg: float = 0.0) -> np.ndarray:
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
        if pec_droop_deg:
            # the fins swing about the body axis: do it in the levelled frame (the clip's heading and root motion
            # would otherwise tilt the hinge), with the fin root joints carried into that frame
            jp_lev = {k: R @ (np.asarray(v) - jp["joint51_00"]) + root_rest for k, v in jp.items()}
            out = self.droop_pectorals(out, jp_lev, pec_droop_deg)
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
            F = self.faces
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
        base = (self.clip_pose(float(t) if t is not None else None, ind, droop, curl) if (t is not None or ind or droop or curl) else self.rest.copy())
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
        return {k: v for k, v in idx.items() if v is not None}

    def write_obj(self, vertices: np.ndarray, path: Path, comment: str = "posed whale") -> None:
        """Posed OBJ with the model's groups (body, eye...) for the renderer."""
        from factory.humpback4k import write_obj as _w
        groups = []
        uv = self._uv()
        for name, idx in self.groups.items():
            if len(idx) == 0:
                continue
            lo, hi = int(idx.min()), int(idx.max()) + 1
            fmask = (self.faces[:, 0] >= lo) & (self.faces[:, 0] < hi)
            groups.append((name, vertices[lo:hi], self.faces[fmask] - lo, uv[lo:hi]))
        _w(Path(path), groups, comment)

    def _uv(self) -> np.ndarray:
        if self.kind == "humpback4k":
            return self.rig.uv
        raise NotImplementedError
