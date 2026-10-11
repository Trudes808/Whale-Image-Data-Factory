"""Convert the Sketchfab "High-Poly Humbpack Whale" glTF into Unreal-ready source files.

The asset (CC-BY-4.0, by K9239, see THIRD_PARTY.md) ships as a Sketchfab glTF: a body
mesh, 22 barnacle meshes and two eye spheres, each under its own node transform, with
4K base colour, metallic-roughness and normal textures. Unreal's OBJ import keeps the
file's axes as-is (verified with the procedural stand-in), so this script bakes every
node transform (the importer does mirror Y, see write below), converts the glTF frame
(Y up, right-handed, metres, snout toward -X)
into the project's Unreal frame (+X snout, +Y right, +Z up, centimetres), and writes one
OBJ with a material group per glTF material plus importer-friendly textures. OBJ files are
written with Y negated because Unreal's OBJ importer mirrors Y on import.

    python3 -m factory.humpback_asset <extracted zip dir> <output dir> [--length-m 15.0]

Outputs in <output dir>:
    SM_Humpback.obj          combined mesh, groups "body", "barnacle", "eye"
    T_Humpback_BaseColor.png sRGB base colour (copied)
    T_Humpback_Roughness.png linear greyscale: the G channel of the glTF metallicRoughness
    T_Humpback_Normal.png    tangent-space normal, green flipped (glTF/OpenGL -> Unreal)
    humpback_info.json       bounds and a spine profile used by factory.poses
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Group names in the OBJ (and thus Unreal material slot names), by glTF material name.
MATERIAL_GROUPS = {"body_mat": "body", "hujitubo1": "barnacle", "eye_mat": "eye"}
# glTF (x, y, z) -> Unreal (-x, -z, y): snout to +X, keep handedness (det = -1), Z up.
_GLTF_TO_UNREAL = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
_COMPONENT_TYPES = {5120: "b", 5121: "B", 5122: "h", 5123: "H", 5125: "I", 5126: "f"}
_TYPE_COUNTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


class GltfScene:
    def __init__(self, gltf_path: Path) -> None:
        self.path = gltf_path
        self.gltf = json.loads(gltf_path.read_text())
        self.buffers = [(gltf_path.parent / b["uri"]).read_bytes() for b in self.gltf["buffers"]]
        self.parent: dict[int, int] = {}
        for index, node in enumerate(self.gltf["nodes"]):
            for child in node.get("children", []):
                self.parent[child] = index

    def accessor(self, index: int) -> np.ndarray:
        acc = self.gltf["accessors"][index]
        view = self.gltf["bufferViews"][acc["bufferView"]]
        data = self.buffers[view.get("buffer", 0)]
        fmt = _COMPONENT_TYPES[acc["componentType"]]
        width = _TYPE_COUNTS[acc["type"]]
        item = struct.calcsize(fmt) * width
        stride = view.get("byteStride", item)
        offset = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
        out = np.empty((acc["count"], width), dtype=np.dtype(fmt))
        for i in range(acc["count"]):
            out[i] = struct.unpack_from("<" + fmt * width, data, offset + i * stride)
        return out

    def local_matrix(self, index: int) -> np.ndarray:
        node = self.gltf["nodes"][index]
        if "matrix" in node:
            return np.array(node["matrix"], dtype=float).reshape(4, 4).T  # column-major
        m = np.eye(4)
        if "scale" in node:
            m = m @ np.diag([*node["scale"], 1.0])
        if "rotation" in node:
            x, y, z, w = node["rotation"]
            r = np.array([
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ])
            rm = np.eye(4)
            rm[:3, :3] = r
            m = rm @ m
        if "translation" in node:
            t = np.eye(4)
            t[:3, 3] = node["translation"]
            m = t @ m
        return m

    def world_matrix(self, index: int) -> np.ndarray:
        chain = [index]
        while chain[-1] in self.parent:
            chain.append(self.parent[chain[-1]])
        m = np.eye(4)
        for node in reversed(chain):
            m = m @ self.local_matrix(node)
        return m

    def primitives(self):
        """Yield (group, positions_m, normals, uvs, faces) per primitive in glTF world space."""
        for index, node in enumerate(self.gltf["nodes"]):
            if "mesh" not in node:
                continue
            m = self.world_matrix(index)
            normal_m = np.linalg.inv(m[:3, :3]).T
            for prim in self.gltf["meshes"][node["mesh"]]["primitives"]:
                material = self.gltf["materials"][prim["material"]]["name"]
                group = MATERIAL_GROUPS.get(material, material)
                pos = self.accessor(prim["attributes"]["POSITION"]).astype(float)
                pos = pos @ m[:3, :3].T + m[:3, 3]
                normals = self.accessor(prim["attributes"]["NORMAL"]).astype(float) @ normal_m.T
                normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9)
                uvs = self.accessor(prim["attributes"]["TEXCOORD_0"]).astype(float)
                faces = self.accessor(prim["indices"]).astype(np.int64).reshape(-1, 3)
                yield group, pos, normals, uvs, faces


def to_unreal(points_m: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """glTF world metres -> Unreal centimetres (see _GLTF_TO_UNREAL)."""
    return points_m @ _GLTF_TO_UNREAL.T * (100.0 * scale)


def convert(src_dir: Path, out_dir: Path, length_m: float | None = None) -> dict:
    scene = GltfScene(src_dir / "scene.gltf")
    prims = list(scene.primitives())
    body = [p for p in prims if p[0] == "body"]
    if len(body) != 1:
        raise ValueError(f"expected exactly one body primitive, found {len(body)}")
    body_pos = body[0][1]
    native_length_m = float(body_pos[:, 0].max() - body_pos[:, 0].min())
    scale = 1.0 if length_m is None else length_m / native_length_m
    # Put the origin at the middle of the body's length and width; keep the model's own
    # height origin (its spine line).
    centre = np.array([(body_pos[:, 0].max() + body_pos[:, 0].min()) / 2, 0.0,
                       (body_pos[:, 2].max() + body_pos[:, 2].min()) / 2])

    out_dir.mkdir(parents=True, exist_ok=True)
    lines = ["# High-Poly Humbpack Whale by K9239 (CC-BY-4.0), converted by factory.humpback_asset",
             "mtllib SM_Humpback.mtl"]
    vertex_base = 0
    all_vertices = []
    groups_written: list[str] = []
    for group, pos, normals, uvs, faces in sorted(prims, key=lambda p: list(MATERIAL_GROUPS.values()).index(p[0])):
        v = to_unreal(pos - centre, scale)
        n = normals @ _GLTF_TO_UNREAL.T
        # The transform has det -1, so the index order no longer winds the way Unreal's
        # importer expects (cross(v1-v0, v2-v0) outward, as the stand-in OBJ is wound).
        # Check against the shipped normals and flip if needed.
        tri = v[faces]
        geometric = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        agreement = np.einsum("ij,ij->i", geometric, n[faces[:, 0]])
        if (agreement < 0).mean() > 0.5:
            faces = faces[:, [0, 2, 1]]
        # usemtl only: Unreal's Interchange OBJ importer makes a separate static mesh per
        # "g" group, but material changes within one object become material slots.
        if group not in groups_written:
            lines.append(f"usemtl {group}")
            groups_written.append(group)
        # Unreal's Interchange OBJ importer mirrors Y (right-handed file -> left-handed
        # engine), which a symmetric whale hides; write -y so the engine mesh matches this
        # frame (found 2026-10-11 with a deformed, asymmetric mesh).
        lines.extend(f"v {x:.4f} {-y:.4f} {z:.4f}" for x, y, z in v)
        lines.extend(f"vt {u:.6f} {1.0 - t:.6f}" for u, t in uvs)  # glTF UV origin is top-left
        lines.extend(f"vn {x:.5f} {y:.5f} {z:.5f}" for x, y, z in n)
        for a, b, c in faces + vertex_base + 1:
            lines.append(f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}")
        vertex_base += len(v)
        all_vertices.append(v)
    (out_dir / "SM_Humpback.obj").write_text("\n".join(lines) + "\n")
    (out_dir / "SM_Humpback.mtl").write_text("".join(f"newmtl {g}\n" for g in groups_written))

    textures = src_dir / "textures"
    Image.open(textures / "body_mat_baseColor.png").convert("RGB").save(out_dir / "T_Humpback_BaseColor.png")
    mr = np.asarray(Image.open(textures / "body_mat_metallicRoughness.png").convert("RGB"))
    Image.fromarray(mr[..., 1]).save(out_dir / "T_Humpback_Roughness.png")
    normal = np.asarray(Image.open(textures / "body_mat_normal.png").convert("RGB")).copy()
    normal[..., 1] = 255 - normal[..., 1]
    Image.fromarray(normal).save(out_dir / "T_Humpback_Normal.png")

    vertices = np.concatenate(all_vertices)
    body_v = to_unreal(body_pos - centre, scale)
    info = {
        "source": "High-Poly Humbpack Whale by K9239 (Sketchfab), CC-BY-4.0",
        "scale": scale,
        "length_cm": float(body_v[:, 0].max() - body_v[:, 0].min()),
        "bounds_min_cm": vertices.min(axis=0).tolist(),
        "bounds_max_cm": vertices.max(axis=0).tolist(),
        "groups": groups_written,
        "spine_profile": spine_profile(body_v),
    }
    (out_dir / "humpback_info.json").write_text(json.dumps(info, indent=2))
    return info


def spine_profile(body_cm: np.ndarray, bins: int = 32, half_width_cm: float = 40.0) -> list[dict]:
    """Top and bottom of the body along its length, near the centre line (for posing)."""
    near_centre = body_cm[np.abs(body_cm[:, 1]) < half_width_cm]
    edges = np.linspace(body_cm[:, 0].min(), body_cm[:, 0].max(), bins + 1)
    profile = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sl = near_centre[(near_centre[:, 0] >= lo) & (near_centre[:, 0] < hi)]
        if len(sl):
            profile.append({"x_cm": float((lo + hi) / 2), "top_cm": float(sl[:, 2].max()),
                            "bottom_cm": float(sl[:, 2].min())})
    return profile


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src_dir", type=Path, help="directory with the extracted zip (scene.gltf, scene.bin, textures/)")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--length-m", type=float, default=None, help="rescale the body to this length (default: keep native)")
    args = parser.parse_args(argv)
    info = convert(args.src_dir, args.out_dir, args.length_m)
    print(json.dumps({k: v for k, v in info.items() if k != "spine_profile"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
