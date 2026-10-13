"""The purchased skinned humpback (third_party_assets/humpback_4k, glTF with a 25-joint rig and one
29 s "dive then breach" clip): load it, pose it at any clip time with linear blend skinning, and
express the result in the project frame (+X nose, +Y right, +Z up, cm) like the static K9239 mesh,
so every shot can be written as a posed OBJ for the renderer and the labels.

    rig = Rig.load("third_party_assets/humpback_4k/src_gltf/scene.gltf")
    V = rig.pose(t_s)             # (N, 3) cm, project frame, at clip time t_s (None = rest pose)
    rig.faces, rig.uv, rig.rest   # triangles, UVs, rest-pose vertices (cm, project frame)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# glTF world of this asset (metres, right-handed: +Z nose, +Y up, +X the whale's left) -> project frame
# (+X nose, +Y right, +Z up, cm)
MODEL_TO_PROJECT = np.array([[0.0, 0.0, 100.0], [-100.0, 0.0, 0.0], [0.0, 100.0, 0.0]])


def _quat_to_mat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _slerp(q0, q1, f):
    d = float(np.dot(q0, q1))
    if d < 0:
        q1 = -q1; d = -d
    if d > 0.9995:
        q = q0 + f * (q1 - q0); return q / np.linalg.norm(q)
    th = np.arccos(d); return (np.sin((1 - f) * th) * q0 + np.sin(f * th) * q1) / np.sin(th)


class Rig:
    def __init__(self, gltf: dict, buffer: bytes, mesh_index: int = 2):
        self.js, self.buf, self.mesh_index = gltf, buffer, mesh_index
        nodes = gltf["nodes"]
        self.parent = {}
        for i, n in enumerate(nodes):
            for c in n.get("children", []):
                self.parent[c] = i
        self.mesh_node = [i for i, n in enumerate(nodes) if n.get("mesh") == mesh_index][0]
        prim = gltf["meshes"][mesh_index]["primitives"][0]
        self.positions = self._acc(prim["attributes"]["POSITION"]).astype(np.float64)
        self.uv = self._acc(prim["attributes"]["TEXCOORD_0"]).astype(np.float64)
        self.faces = self._acc(prim["indices"]).astype(np.int64).reshape(-1, 3)
        self.joints_idx = self._acc(prim["attributes"]["JOINTS_0"]).astype(np.int64)
        self.weights = self._acc(prim["attributes"]["WEIGHTS_0"]).astype(np.float64)
        self.weights /= np.clip(self.weights.sum(1, keepdims=True), 1e-9, None)
        skin = gltf["skins"][nodes[self.mesh_node]["skin"]]
        self.joints = list(skin["joints"])
        self.ibm = self._acc(skin["inverseBindMatrices"]).astype(np.float64).reshape(-1, 4, 4).transpose(0, 2, 1)   # column-major -> row-major
        self.joint_names = [nodes[j]["name"] for j in self.joints]
        self.anim = self._load_animation(0) if gltf.get("animations") else {}
        self.clip_end = max((max(v["t"]) for v in self.anim.values()), default=0.0)
        self.rest = self.pose(None)
        self.length_cm = float(self.rest[:, 0].max() - self.rest[:, 0].min())

    def skin_matrices(self, t: float | None = None, mesh_index: int | None = None, offsets: dict | None = None):
        """Per-vertex blended skin transform at time t, model frame: (M (N,3,3), c (N,3)) with world = M p + c.
        Exactly what pose() applies, so an edit made in the posed frame maps back to bind space through it."""
        cache = {}
        if mesh_index is None or mesh_index == self.mesh_index:
            positions, joints_idx, weights, joints, ibm = self.positions, self.joints_idx, self.weights, self.joints, self.ibm
        else:
            positions, joints_idx, weights, joints, ibm = self._skin_data(mesh_index)
        G = np.stack([self._global(j, cache, t, offsets) for j in joints])
        skin = G @ ibm                                                       # (J, 4, 4)
        M = np.zeros((len(positions), 3, 3)); c = np.zeros((len(positions), 3))
        for k in range(4):
            j = joints_idx[:, k]; w = weights[:, k]
            M += w[:, None, None] * skin[j][:, :3, :3]; c += w[:, None] * skin[j][:, :3, 3]
        return M, c

    def bind_from_rest(self, rest_world: np.ndarray, mesh_index: int | None = None) -> np.ndarray:
        """Bind-space positions (model frame) that skin to the given rest-pose vertices (project frame, cm)."""
        M, c = self.skin_matrices(None, mesh_index)
        model = (np.linalg.inv(MODEL_TO_PROJECT) @ rest_world.T).T
        return np.linalg.solve(M, (model - c)[..., None])[..., 0]

    def mesh_positions(self, mesh_index: int) -> np.ndarray:
        """Rest positions (model frame) of a mesh of the file."""
        return self.positions if mesh_index == self.mesh_index else self._skin_data(mesh_index)[0]

    def pec_regions(self) -> dict:
        """Per-vertex skin weight of the body mesh on the left / right pectoral chains (N,) in [0, 1]: a smooth
        membership that edits of the fins can scale by without tearing the shoulder."""
        names = np.array(self.joint_names)
        chains = {"pec_left": ("joint64_013", "joint65_014", "joint66_015", "joint67_016", "joint68_017", "joint69_018"),
                  "pec_right": ("joint70_019", "joint71_020", "joint72_021", "joint73_022", "joint74_023", "joint75_024")}
        out = {}
        for side, chain in chains.items():
            w = np.zeros(len(self.positions))
            for k in range(4):
                w += np.where(np.isin(names[self.joints_idx[:, k]], chain), self.weights[:, k], 0.0)
            out[side] = np.clip(w, 0.0, 1.0)
        return out

    @classmethod
    def load(cls, gltf_path: str | Path, mesh_index: int = 2) -> "Rig":
        gltf_path = Path(gltf_path); js = json.loads(gltf_path.read_text())
        buf = (gltf_path.parent / js["buffers"][0]["uri"]).read_bytes()
        return cls(js, buf, mesh_index)

    def _acc(self, i):
        a = self.js["accessors"][i]; bv = self.js["bufferViews"][a["bufferView"]]
        off = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
        dt = {5126: np.float32, 5123: np.uint16, 5125: np.uint32, 5121: np.uint8}[a["componentType"]]
        n = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}[a["type"]]
        arr = np.frombuffer(self.buf, dt, a["count"] * n, off)
        return arr.reshape(a["count"], n) if n > 1 else arr

    def _load_animation(self, k: int) -> dict:
        anim = self.js["animations"][k]; out = {}
        for ch in anim["channels"]:
            s = anim["samplers"][ch["sampler"]]
            out.setdefault(ch["target"]["node"], {})[ch["target"]["path"]] = (self._acc(s["input"]).astype(np.float64), self._acc(s["output"]).astype(np.float64))
        for node, d in out.items():
            d["t"] = np.unique(np.concatenate([v[0] for v in d.values() if isinstance(v, tuple)]))
        return out

    def _local(self, i: int, t: float | None) -> np.ndarray:
        n = self.js["nodes"][i]
        if "matrix" in n:
            return np.array(n["matrix"]).reshape(4, 4).T
        tr = np.array(n.get("translation", [0, 0, 0]), float); q = np.array(n.get("rotation", [0, 0, 0, 1]), float); sc = np.array(n.get("scale", [1, 1, 1]), float)
        if t is not None and i in self.anim:
            a = self.anim[i]
            if "translation" in a:
                tr = self._sample(a["translation"], t)
            if "rotation" in a:
                q = self._sample(a["rotation"], t, slerp=True)
            if "scale" in a:
                sc = self._sample(a["scale"], t)
        T = np.eye(4); T[:3, :3] = _quat_to_mat(q) * sc[None, :]; T[:3, 3] = tr
        return T

    @staticmethod
    def _sample(curve, t: float, slerp: bool = False):
        times, values = curve
        if t <= times[0]:
            return values[0]
        if t >= times[-1]:
            return values[-1]
        k = int(np.searchsorted(times, t) - 1); f = (t - times[k]) / max(times[k + 1] - times[k], 1e-9)
        if slerp:
            return _slerp(values[k], values[k + 1], f)
        return values[k] * (1 - f) + values[k + 1] * f

    def _global(self, i: int, cache: dict, t: float | None = None, offsets: dict | None = None) -> np.ndarray:
        if i in cache:
            return cache[i]
        M = self._local(i, t)
        G = (self._global(self.parent[i], cache, t, offsets) @ M) if i in self.parent else M
        if offsets:
            R = offsets.get(self.js["nodes"][i]["name"])
            if R is not None:                      # an extra rotation about the joint, in model space; children inherit it
                O = np.eye(4); O[:3, :3] = R
                Tp = np.eye(4); Tp[:3, 3] = G[:3, 3]; Tm = np.eye(4); Tm[:3, 3] = -G[:3, 3]
                G = Tp @ O @ Tm @ G
        cache[i] = G
        return G

    def joint_positions(self, t: float | None = None, offsets: dict | None = None) -> dict:
        cache = {}
        return {name: MODEL_TO_PROJECT @ self._global(j, cache, t, offsets)[:3, 3] for name, j in zip(self.joint_names, self.joints)}

    def pose(self, t: float | None = None, mesh_index: int | None = None, positions=None, offsets: dict | None = None) -> np.ndarray:
        """Skinned vertices at clip time t (seconds; None = bind pose), (N, 3) cm in the project frame.
        glTF rule: a skinned mesh's own node transform is ignored; world = sum_j w_j (G_j . IBM_j) . p.
        ``positions`` replaces the mesh's rest vertices (model frame; e.g. an individual's proportions) and
        ``offsets`` = {joint name: 3x3 model-frame rotation} adds resting joint angles on top of the clip."""
        cache = {}
        if mesh_index is None or mesh_index == self.mesh_index:
            positions_, joints_idx, weights, joints, ibm = self.positions, self.joints_idx, self.weights, self.joints, self.ibm
        else:
            positions_, joints_idx, weights, joints, ibm = self._skin_data(mesh_index)
        positions = positions_ if positions is None else positions
        G = np.stack([self._global(j, cache, t, offsets) for j in joints])                  # (J, 4, 4) joint -> world
        skin_mats = G @ ibm
        P = np.c_[positions, np.ones(len(positions))]
        out = np.zeros((len(P), 3))
        for k in range(4):
            j = joints_idx[:, k]; w = weights[:, k]
            out += w[:, None] * np.einsum("nij,nj->ni", skin_mats[j][:, :3, :], P)
        return (MODEL_TO_PROJECT @ out.T).T

    def _skin_data(self, mesh_index: int):
        prim = self.js["meshes"][mesh_index]["primitives"][0]
        node = [i for i, n in enumerate(self.js["nodes"]) if n.get("mesh") == mesh_index][0]
        skin = self.js["skins"][self.js["nodes"][node]["skin"]]
        positions = self._acc(prim["attributes"]["POSITION"]).astype(np.float64)
        ji = self._acc(prim["attributes"]["JOINTS_0"]).astype(np.int64); w = self._acc(prim["attributes"]["WEIGHTS_0"]).astype(np.float64)
        w /= np.clip(w.sum(1, keepdims=True), 1e-9, None)
        ibm = self._acc(skin["inverseBindMatrices"]).astype(np.float64).reshape(-1, 4, 4).transpose(0, 2, 1)
        return positions, ji, w, list(skin["joints"]), ibm

    def extra_meshes(self, t: float | None = None, positions: dict | None = None, offsets: dict | None = None) -> list[dict]:
        """The other skinned meshes of the file (the eyes), posed: [{name, vertices, faces, uv}].
        ``positions`` = {mesh index: replacement rest positions (model frame)}."""
        out = []
        for mi, m in enumerate(self.js["meshes"]):
            if mi == self.mesh_index:
                continue
            prim = m["primitives"][0]
            out.append({"name": m.get("name", f"mesh{mi}"), "vertices": self.pose(t, mi, (positions or {}).get(mi), offsets), "faces": self._acc(prim["indices"]).astype(np.int64).reshape(-1, 3),
                        "uv": self._acc(prim["attributes"]["TEXCOORD_0"]).astype(np.float64)})
        return out
