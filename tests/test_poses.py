import math
import unittest

import numpy as np

from factory import poses


class RotationTests(unittest.TestCase):
    def test_pitch_raises_nose(self):
        m = poses.rotation_matrix(0, 90, 0)
        np.testing.assert_allclose(np.array([1, 0, 0]) @ m, [0, 0, 1], atol=1e-12)

    def test_yaw_turns_nose_right(self):
        m = poses.rotation_matrix(0, 0, 90)
        np.testing.assert_allclose(np.array([1, 0, 0]) @ m, [0, 1, 0], atol=1e-12)

    def test_roll_drops_right_side(self):
        m = poses.rotation_matrix(90, 0, 0)
        np.testing.assert_allclose(np.array([0, 1, 0]) @ m, [0, 0, -1], atol=1e-12)

    def test_is_rotation(self):
        m = poses.rotation_matrix(33, -20, 140)
        np.testing.assert_allclose(m @ m.T, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(m), 1.0)


class PlaceTests(unittest.TestCase):
    def setUp(self):
        # A stick whale: nose at +800, tail at -800, a dorsal bump 300 up at x=-100.
        self.vertices = np.array([[800, 0, 0], [-800, 0, 0], [-100, 0, 300], [0, 0, -150]], float)

    def test_clearance_sets_highest_point(self):
        for name, pose in poses.POSES.items():
            with self.subTest(pose=name):
                shot = poses.place(self.vertices, pose)
                self.assertAlmostEqual(shot["stats"]["highest_point_cm"], pose.clearance_m * 100, places=6)

    def test_fluke_up_puts_tail_highest(self):
        pose = poses.POSES["fluke_up"]
        rotated = self.vertices @ poses.rotation_matrix(pose.roll_deg, pose.pitch_deg, pose.yaw_deg)
        self.assertEqual(int(np.argmax(rotated[:, 2])), 1)  # the tail vertex

    def test_submerged_has_nothing_above_water(self):
        shot = poses.place(self.vertices, poses.POSES["submerged"])
        self.assertEqual(shot["stats"]["vertex_fraction_above_water"], 0.0)
        self.assertLess(shot["stats"]["highest_point_cm"], 0.0)

    def test_camera_distance_height_and_aim(self):
        pose = poses.Pose(0, 0, 1.0, "test", poses.Camera(bearing_deg=90, distance_m=40, height_m=3), yaw_deg=0)
        shot = poses.place(self.vertices, pose)
        cam = np.array(shot["camera"]["location"])
        aim = np.array(shot["stats"]["aim_point"])
        self.assertAlmostEqual(cam[2], 300.0)
        self.assertAlmostEqual(math.hypot(*(cam - aim)[:2]), 4000.0, places=6)
        self.assertAlmostEqual(cam[1] - aim[1], 4000.0, places=6)  # bearing 90 = off the right side
        # The camera looks at the aim point.
        rot = shot["camera"]["rotation"]
        forward = np.array([1, 0, 0]) @ poses.rotation_matrix(rot["roll"], rot["pitch"], rot["yaw"])
        direction = (aim - cam) / np.linalg.norm(aim - cam)
        np.testing.assert_allclose(forward, direction, atol=1e-6)

    def test_make_shot_round_trips_pose(self):
        shot = poses.make_shot("logging", self.vertices)
        self.assertEqual(shot["name"], "logging")
        self.assertEqual(shot["pose"]["camera"]["focal_mm"], poses.POSES["logging"].camera.focal_mm)
        self.assertEqual(shot["whale"]["rotation"]["yaw"], poses.POSES["logging"].yaw_deg)



class SplashTests(unittest.TestCase):
    def setUp(self):
        self.vertices = np.array([[800, 0, 0], [-800, 0, 0], [-100, 0, 300], [0, 0, -150]] + [[x, 0, 0] for x in range(-700, 800, 10)], float)

    def test_breach_has_crown_at_waterline(self):
        shot = poses.make_shot("breach", self.vertices)
        splash = shot["splash"]
        self.assertEqual(splash["kind"], "crown")
        self.assertAlmostEqual(splash["centre_cm"][2], 0.0)
        from factory import splash as splash_module
        mesh = splash_module.build_mesh(splash)
        self.assertGreater(len(mesh.faces), 1000)
        self.assertGreater(mesh.vertices[:, 2].max(), splash["height_cm"] * 0.8)

    def test_blow_has_light_wash_only(self):
        shot = poses.make_shot("blow", self.vertices)
        splash = shot["splash"]
        self.assertEqual(splash["kind"], "wash")
        self.assertLess(splash["amount"], 1.0)
        from factory import splash as splash_module
        mesh = splash_module.build_mesh(splash)
        self.assertGreater(len(mesh.faces), 100)
        self.assertLess(mesh.vertices[:, 2].max(), splash["centre_cm"][2] + splash["height_cm"] * 1.5 + 60)

    def test_no_splash_by_default(self):
        self.assertIsNone(poses.make_shot("logging", self.vertices)["splash"])

    def test_sampler_is_seeded(self):
        from factory import splash
        a = splash.build_mesh(splash.splash_spec("crown", (0, 0, 0), 300, 200, seed=3))
        b = splash.build_mesh(splash.splash_spec("crown", (0, 0, 0), 300, 200, seed=3))
        np.testing.assert_array_equal(a.vertices, b.vertices)
        with self.assertRaises(ValueError):
            splash.build_mesh({"kind": "tsunami", "centre_cm": [0, 0, 0], "radius_cm": 1, "height_cm": 1})

    def test_obj_round_trip(self):
        import tempfile
        from factory import project, splash
        with tempfile.TemporaryDirectory() as tmp:
            spec = splash.ensure_obj(splash.splash_spec("wash", (10, 20, 0), 300, 60, seed=1), tmp, "x")
            v, f = project.load_obj(spec["mesh_obj"])
            self.assertEqual(len(f), spec["mesh"]["faces"])
            self.assertEqual(len(v), spec["mesh"]["vertices"])



class SeaStateTests(unittest.TestCase):
    def test_presets_apply(self):
        from factory import seastate
        vertices = np.array([[800, 0, 0], [-800, 0, 0], [-100, 0, 300], [0, 0, -150]], float)
        for name, state in seastate.SEA_STATES.items():
            with self.subTest(sea=name):
                shot = seastate.apply_to_shot(poses.make_shot("logging", vertices), name, seed=2)
                self.assertEqual(shot["sea_state"], name)
                self.assertEqual(shot["waves"]["num_waves"], state.waves["num_waves"])
                self.assertIn("Water Roughness", shot["ocean"]["scalars"])
                self.assertIsNone(shot["splash"])
        with self.assertRaises(ValueError):
            seastate.apply_to_shot({}, "hurricane")


if __name__ == "__main__":
    unittest.main()
