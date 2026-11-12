"""The dorsal_fin_front label on the rig (keypoint revision 2026-11-02): the front of the whole fin / hump, placed
DORSAL_FRONT_RATIO x |tip - back junction| ahead of the tip along the dorsal midline, while the blade's root stays the
fin_le chain's end and the base-chord end of the individual edits. Needs the humpback_4k asset (skipped without it)."""
import unittest

import numpy as np

from factory import project
from factory.whalemodel import DORSAL_FRONT_RATIO, HUMPBACK4K_GLTF, WhaleModel


@unittest.skipUnless(HUMPBACK4K_GLTF.exists(), "humpback_4k asset not installed")
class FinFrontTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = WhaleModel("humpback4k")
        cls.fin = cls.model.dorsal_fin()

    def test_front_base_sits_ahead_of_the_blade_root_by_the_ratio(self):
        fin, R = self.fin, self.model.rest
        P = project.landmark_positions({"t": fin["tip"], "b": fin["back"], "r": fin["front"], "f": fin["front_base"]}, R)
        self.assertGreater(P["f"][0], P["r"][0] + 10.0)                      # ahead of the blade root (+x = head)
        self.assertLess(abs(P["f"][1]), 1.5)                                 # on the dorsal midline
        ratio = np.linalg.norm(P["f"] - P["t"]) / np.linalg.norm(P["b"] - P["t"])
        self.assertAlmostEqual(ratio, DORSAL_FRONT_RATIO, delta=0.15)

    def test_landmarks_use_it_and_the_blade_root_keeps_its_jobs(self):
        idx = self.model.landmark_indices()
        self.assertEqual(idx["dorsal_fin_front"], self.fin["front_base"])
        self.assertEqual(idx["fin_le_11"], self.fin["front"])                     # the leading-edge chain ends at the blade root
        self.assertEqual(self.model.dorsal_region()["dorsal_idx"][2], self.fin["front"])   # individual edits: unchanged geometry
        self.assertEqual(idx["dorsal_fin_tip"], self.fin["tip"])
        self.assertEqual(idx["dorsal_fin_back"], self.fin["back"])
