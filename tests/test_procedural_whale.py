import tempfile
import unittest
from pathlib import Path

import numpy as np

from factory import procedural_whale


class ProceduralWhaleTests(unittest.TestCase):
    def test_humpback_proportions(self):
        vertices, faces = procedural_whale.build_whale()
        size = vertices.max(axis=0) - vertices.min(axis=0)
        self.assertAlmostEqual(size[0], procedural_whale.LENGTH_CM, delta=50)  # ~14 m long (X)
        self.assertTrue(700 < size[1] < 1000, size)  # pectoral fin span (Y)
        self.assertTrue(faces.min() >= 0 and faces.max() < len(vertices))

    def test_obj_has_uvs_for_every_vertex(self):
        vertices, faces = procedural_whale.build_whale()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "whale.obj"
            procedural_whale.write_obj(path, vertices, faces)
            lines = path.read_text().splitlines()
        self.assertEqual(sum(l.startswith("v ") for l in lines), len(vertices))
        self.assertEqual(sum(l.startswith("vt ") for l in lines), len(vertices))
        self.assertEqual(sum(l.startswith("f ") for l in lines), len(faces))


if __name__ == "__main__":
    unittest.main()
