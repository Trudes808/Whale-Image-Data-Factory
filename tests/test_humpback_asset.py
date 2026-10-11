import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from factory import humpback_asset


def _write_tiny_gltf(root: Path) -> None:
    """A one-triangle 'body' under a Y-up node scaled by 0.01, like the Sketchfab export."""
    positions = np.array([[-800, 0, 0], [800, 0, 0], [0, 100, 50]], np.float32)  # tail, nose(-X in glTF? no: +X here)
    normals = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], np.float32)
    uvs = np.array([[0, 0], [1, 0], [0.5, 1]], np.float32)
    indices = np.array([0, 1, 2], np.uint16)
    blob = positions.tobytes() + normals.tobytes() + uvs.tobytes() + indices.tobytes()
    (root / "scene.bin").write_bytes(blob)
    offsets = [0, positions.nbytes, positions.nbytes + normals.nbytes, positions.nbytes + normals.nbytes + uvs.nbytes]
    gltf = {
        "asset": {"version": "2.0"},
        "buffers": [{"uri": "scene.bin", "byteLength": len(blob)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": offsets[0], "byteLength": positions.nbytes},
            {"buffer": 0, "byteOffset": offsets[1], "byteLength": normals.nbytes},
            {"buffer": 0, "byteOffset": offsets[2], "byteLength": uvs.nbytes},
            {"buffer": 0, "byteOffset": offsets[3], "byteLength": indices.nbytes},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC3"},
            {"bufferView": 2, "componentType": 5126, "count": 3, "type": "VEC2"},
            {"bufferView": 3, "componentType": 5123, "count": 3, "type": "SCALAR"},
        ],
        "materials": [{"name": "body_mat"}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2}, "indices": 3, "material": 0}]}],
        "nodes": [
            {"name": "root", "children": [1], "matrix": [1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 1]},
            {"name": "unrotate", "children": [2], "matrix": [1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1]},
            {"name": "whale", "children": [3], "scale": [0.01, 0.01, 0.01]},
            {"name": "body", "mesh": 0},
        ],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
    }
    (root / "scene.gltf").write_text(json.dumps(gltf))
    tex = root / "textures"
    tex.mkdir()
    Image.fromarray(np.full((4, 4, 3), 40, np.uint8)).save(tex / "body_mat_baseColor.png")
    mr = np.zeros((4, 4, 3), np.uint8)
    mr[..., 0] = 255
    mr[..., 1] = 70
    mr[..., 2] = 255
    Image.fromarray(mr).save(tex / "body_mat_metallicRoughness.png")
    nm = np.zeros((4, 4, 3), np.uint8)
    nm[..., 0], nm[..., 1], nm[..., 2] = 128, 200, 255
    Image.fromarray(nm).save(tex / "body_mat_normal.png")


class ConvertTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.src = Path(self.tmp.name) / "src"
        self.src.mkdir()
        _write_tiny_gltf(self.src)
        self.out = Path(self.tmp.name) / "out"

    def tearDown(self):
        self.tmp.cleanup()

    def test_axes_units_and_textures(self):
        info = humpback_asset.convert(self.src, self.out)
        self.assertAlmostEqual(info["length_cm"], 1600.0, places=3)  # 16 glTF-m -> 1600 cm
        vertices = {}
        for line in (self.out / "SM_Humpback.obj").read_text().splitlines():
            if line.startswith("v "):
                vertices[len(vertices)] = [float(t) for t in line.split()[1:]]
        # glTF (x, y, z) -> Unreal (-x, -z, y): the +X nose ends at -X... the body is re-centred,
        # and the raised vertex (glTF y = 1 m) is 100 cm up with glTF z = 0.5 m -> Unreal y = -25 after re-centring.
        self.assertEqual(vertices[2], [0.0, 25.0, 100.0])  # width re-centred, Y negated for Unreal's import
        self.assertEqual(vertices[0][0], 800.0)  # glTF -X tail becomes +X
        self.assertEqual(info["groups"], ["body"])
        rough = np.asarray(Image.open(self.out / "T_Humpback_Roughness.png"))
        self.assertEqual(rough.shape, (4, 4))
        self.assertEqual(int(rough[0, 0]), 70)
        normal = np.asarray(Image.open(self.out / "T_Humpback_Normal.png"))
        self.assertEqual(int(normal[0, 0, 1]), 55)  # green flipped
        self.assertEqual(json.loads((self.out / "humpback_info.json").read_text())["length_cm"], info["length_cm"])

    def test_rescale(self):
        info = humpback_asset.convert(self.src, self.out, length_m=14.0)
        self.assertAlmostEqual(info["length_cm"], 1400.0, places=3)
        self.assertAlmostEqual(info["scale"], 14.0 / 16.0)

    def test_winding_matches_shipped_normals(self):
        humpback_asset.convert(self.src, self.out)
        lines = (self.out / "SM_Humpback.obj").read_text().splitlines()
        v = np.array([[float(t) for t in l.split()[1:]] for l in lines if l.startswith("v ")])
        n = np.array([[float(t) for t in l.split()[1:]] for l in lines if l.startswith("vn ")])
        face = [int(t.split("/")[0]) - 1 for l in lines if l.startswith("f ") for t in l.split()[1:]]
        tri = v[face]
        geometric = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        self.assertGreater(float(geometric @ n[face[0]]), 0.0)


if __name__ == "__main__":
    unittest.main()
