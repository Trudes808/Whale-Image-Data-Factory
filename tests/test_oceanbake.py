"""The run06 sea: EXR round trip, bake layout, the label height evaluator against the baked displacement,
and the contact layer (python3 -m unittest tests.test_oceanbake)."""
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from factory import contact, exr, fftocean, oceanbake


class TestExr(unittest.TestCase):
    def test_round_trip(self):
        rng = np.random.default_rng(0)
        a = {c: rng.normal(size=(19, 33)).astype(np.float32) for c in "RGBA"}
        with tempfile.TemporaryDirectory() as d:
            for comp in (exr.NO_COMPRESSION, exr.ZIPS_COMPRESSION, exr.ZIP_COMPRESSION):
                for half in (True, False):
                    p = exr.write(Path(d) / "t.exr", a, half=half, compression=comp)
                    b = exr.read(p)
                    for c in "RGBA":
                        ref = a[c].astype(np.float16 if half else np.float32).astype(np.float32)
                        self.assertTrue(np.array_equal(b[c], ref))


class TestSpectrum(unittest.TestCase):
    def test_default_unchanged(self):
        """cap_scale = 0 leaves web-ocean's spectrum untouched."""
        s = fftocean.FFTOcean(seed=2, n=64)
        k = np.linspace(0.01, 50, 200)
        self.assertTrue(np.array_equal(s.spectrum(k, 0 * k), fftocean.FFTOcean(seed=2, n=64, cap_scale=0.0).spectrum(k, 0 * k)))

    def test_cox_munk_and_whitecaps(self):
        self.assertAlmostEqual(fftocean.cox_munk_mss(7.0)["total"], 0.003 + 0.00512 * 7.0)
        self.assertAlmostEqual(fftocean.whitecap_fraction(5.0), 3.84e-6 * 5 ** 3.41)
        self.assertAlmostEqual(fftocean.whitecap_fraction(8.0) * 100, 0.46, places=2)


class TestBake(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = Path(tempfile.mkdtemp())
        spec = oceanbake.draw_state("tw_light", seed=3, rng=np.random.default_rng(3))
        spec.cap_scale = 2.0
        cls.layout = oceanbake.bake(spec, cls.dir / "b", foam=False)
        cls.sea = oceanbake.BakedSea.load(cls.dir / "b", foam=False)

    def test_layout(self):
        lay = json.loads((self.dir / "b" / "layout.json").read_text())
        self.assertEqual(lay["geo_cascades"], 3)
        self.assertEqual(len(lay["cascades_m"]), 6)
        for c in range(6):
            self.assertIn(f"T_Disp_c{c}.exr", lay["files"]); self.assertIn(f"T_Mom_c{c}.exr", lay["files"])

    def test_height_matches_displaced_mesh(self):
        """A displaced mesh vertex lies exactly on the evaluated surface (labels = render)."""
        rng = np.random.default_rng(1)
        qx, qy = rng.uniform(-500, 500, 4000), rng.uniform(-500, 500, 4000)
        D = self.sea.displacement(qx, qy)
        h = self.sea.height(qx + D[:, 0], qy + D[:, 1])
        self.assertLess(np.abs(h - D[:, 2]).max(), 1e-4)       # metres

    def test_periodic(self):
        L0 = self.layout["cascades_m"][0]
        x = np.array([3.3, 100.0]); y = np.array([7.1, -40.0])
        self.assertTrue(np.allclose(self.sea.height(x, y), self.sea.height(x + L0, y - L0), atol=1e-6))

    def test_mip_moments(self):
        """Box-filtered second moments minus squared means = non-negative variance."""
        m = self.sea.mom[4]
        chain_sx = oceanbake.mip_chain(m[..., 0]); chain_sxx = oceanbake.mip_chain(m[..., 2])
        for a, b in zip(chain_sx[1:], chain_sxx[1:]):
            self.assertGreaterEqual(float((b - a * a).min()), -1e-3)


class TestContact(unittest.TestCase):
    def test_moving_box_makes_waves_and_foam(self):
        # a 10 m x 2 m box, 1 m deep, half out of the water, moving at 2 m/s
        xs, ys, zs = np.meshgrid(np.linspace(-5, 5, 41), np.linspace(-1, 1, 9), np.linspace(-1, 0.5, 7))
        V = np.stack([xs.ravel(), ys.ravel(), zs.ravel()], 1)
        from scipy.spatial import ConvexHull
        hull = ConvexHull(V)
        p = contact.ContactParams(size_m=40.0, res=256, history_s=4.0)
        lay = contact.run(V, hull.simplices, "surface", (1.0, 0.0), centre_xy=(0.0, 0.0), params=p, motion=dict(speed=2.0, rise=0.0))
        t = lay["tex"]
        self.assertGreater(float(np.abs(t[..., 0][~lay["pierce_mask"]]).max()), 0.005)
        self.assertGreater(float(t[..., 1].max()), 0.2)
        self.assertGreater(float(t[..., 2].max()), 0.05)
        # height_at agrees with the texture at texel centres
        n = lay["res"]; dx = lay["size_m"] / n
        i, j = 100, 120
        x = lay["origin_m"][0] + (i + 0.5) * dx; y = lay["origin_m"][1] + (j + 0.5) * dx
        self.assertAlmostEqual(float(contact.height_at(lay, np.array([x]), np.array([y]))[0]), float(t[j, i, 0]), places=5)


if __name__ == "__main__":
    unittest.main()
