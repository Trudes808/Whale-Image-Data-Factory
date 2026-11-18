"""run06 multi-whale pieces: label decoding (exact ids, blended silhouettes), capsule collision, animals_of."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from factory import postprocess, scenes


def _linear_to_png(lin: np.ndarray) -> np.ndarray:
    s = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.power(lin, 1 / 2.4) - 0.055)
    return np.clip(np.round(s * 255), 0, 255).astype(np.uint8)


class DecodeTests(unittest.TestCase):
    def _scene(self, codes: dict, background=(0, 0, 0)):
        """Three blocks of ids 1, 2, 3 on background, rendered at 4x and box-filtered (= blended silhouettes)."""
        hi = np.zeros((160, 240), np.uint8)
        hi[22:121, 21:110] = 1
        hi[42:139, 101:181] = 2           # overlaps id 1: id 2 in front
        hi[61:99, 171:229] = 3
        lin = np.zeros(hi.shape + (3,), np.float32)
        for k, c in codes.items():
            lin[hi == k] = c
        lo = lin.reshape(40, 4, 60, 4, 3).mean((1, 3))
        truth = hi[1::4, 1::4]
        return _linear_to_png(lo), truth, hi

    def test_rgb_codes_exact_and_blends_resolved(self):
        png, truth, hi = self._scene({k: postprocess.ID_CODES[k] for k in (1, 2, 3)})
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "l.png"; Image.fromarray(png).save(p)
            ids, st = postprocess.decode_label(p, postprocess.LABEL_ENCODING_RGB, 3)
        self.assertEqual(st["invalid_ids"], 0)
        self.assertGreater(st["blended_pixels"], 50)
        # every decoded id is one the region-majority of its 4x4 footprint agrees with or a neighbour of it
        frac = np.stack([(hi == k).reshape(40, 4, 60, 4).mean((1, 3)) for k in range(4)])
        maj = frac.argmax(0)
        top2 = np.sort(frac, 0)[-2:]
        clear = top2[1] - top2[0] > 0.1                  # exact 50/50 pixels may go either way
        self.assertGreaterEqual((ids == maj)[clear].mean(), 0.995)
        # no id appears where its own region has no coverage at all (a blend never becomes a third id)
        for k in (1, 2, 3):
            self.assertEqual(int(((ids == k) & (frac[k] == 0)).sum()), 0)

    def test_legacy_single_whale_has_no_companion_rim(self):
        hi = np.zeros((160, 240), np.float32); hi[30:130, 30:200] = 1.0
        lo = hi.reshape(40, 4, 60, 4).mean((1, 3))
        png = _linear_to_png(np.repeat(lo[..., None], 3, 2))
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "l.png"; Image.fromarray(png).save(p)
            ids, _ = postprocess.decode_label(p, postprocess.LABEL_ENCODING_LEGACY, 1)
            old_primary, old_comp = postprocess.legacy_threshold_decode(p)
        self.assertGreater(int(old_comp.sum()), 0)                  # the old decoder's false "companion" ring
        self.assertEqual(int((ids == 2).sum()), 0)
        self.assertTrue(((ids == 1) == (lo >= 0.5)).all())


class CollisionTests(unittest.TestCase):
    def test_segments(self):
        A = np.array([[0.0, 0, 0]]); B = np.array([[100.0, 0, 0]])
        C = np.array([[50.0, 30, -50]]); D = np.array([[50.0, 30, 50]])
        self.assertAlmostEqual(float(scenes.segment_distances(A, B, C, D)[0, 0]), 30.0, places=4)
        caps1 = (A, B, np.array([10.0])); caps2 = (C, D, np.array([10.0]))
        self.assertFalse(scenes.collides(caps1, caps2, margin_cm=5.0))
        self.assertTrue(scenes.collides(caps1, (C, D, np.array([25.0])), margin_cm=0.0))


class AnimalsOfTests(unittest.TestCase):
    def test_legacy_and_run06(self):
        shot = {"id": "1", "companion": {"whale": {}, "role": "calf"}}
        a = postprocess.animals_of(shot)
        self.assertEqual([x.get("instance_id") for x in a[1:]], [2])
        shot = {"id": "1", "companion": None, "companions": [{"instance_id": 2}, {"instance_id": 3}]}
        self.assertEqual(len(postprocess.animals_of(shot)), 3)
        self.assertEqual(len(postprocess.animals_of({"id": "1"})), 1)


if __name__ == "__main__":
    unittest.main()
