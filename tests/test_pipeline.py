import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from factory import dataset, postprocess, project, shotlist

CONFIG = json.loads((Path(__file__).resolve().parents[1] / "factory" / "config" / "poc_boat.json").read_text())


def stick_whale():
    xs = np.arange(-800, 801, 25.0)
    return np.array([[x, 0, 0] for x in xs] + [[x, 0, 300] for x in xs[20:30]] + [[0, 300, -100], [0, -300, -100]], float)


class ShotlistTests(unittest.TestCase):
    def test_shots_are_reproducible_and_complete(self):
        v = stick_whale()
        a = [shotlist.make_shot(i, np.random.default_rng(5), CONFIG, v) for i in range(3)]
        b = [shotlist.make_shot(i, np.random.default_rng(5), CONFIG, v) for i in range(3)]
        self.assertEqual(a, b)
        for shot in a:
            for key in ("id", "whale", "camera", "sun", "fog", "exposure_ev", "waves", "ocean", "sea_state", "look", "stats"):
                self.assertIn(key, shot)
            self.assertGreaterEqual(shot["camera"]["focal_mm"], CONFIG["camera"]["focal_mm"][0])
            self.assertLessEqual(shot["camera"]["focal_mm"], CONFIG["camera"]["focal_mm"][1])
            self.assertIn(shot["sea_state"], CONFIG["sea_state_weights"])

    def test_batches_group_by_sea_and_look(self):
        v = stick_whale()
        rng = np.random.default_rng(9)
        shots = [shotlist.make_shot(i, rng, CONFIG, v) for i in range(40)]
        batches = shotlist.make_batches(shots, batch_size=6)
        self.assertEqual(sum(len(b["shots"]) for b in batches), 40)
        for batch in batches:
            self.assertLessEqual(len(batch["shots"]), 6)
            self.assertEqual({(s["sea_state"], s["look"]) for s in batch["shots"]}, {(batch["sea_state"], batch["look"])})


class ProjectTests(unittest.TestCase):
    def test_projection_centre_and_scale(self):
        cam = project.Camera([0, 0, 0], {"roll": 0, "pitch": 0, "yaw": 0}, 50.0, 1280, 720)
        uv, depth = cam.project(np.array([[1000.0, 0.0, 0.0], [1000.0, 100.0, 0.0], [1000.0, 0.0, 100.0]]))
        np.testing.assert_allclose(uv[0], [640, 360])
        self.assertGreater(uv[1][0], 640)  # +Y is to the right
        self.assertLess(uv[2][1], 360)     # +Z is up (smaller v)
        np.testing.assert_allclose(depth, 1000.0)

    def test_rasterize_fills_triangle(self):
        uv = np.array([[10.0, 10.0], [50.0, 10.0], [10.0, 50.0]])
        image = project.rasterize(uv, np.array([100.0, 100.0, 100.0]), np.array([[0, 1, 2]]), 64, 64)
        self.assertGreater(image.sum(), 700)
        self.assertEqual(image[12, 12], 1)
        self.assertEqual(image[60, 60], 0)

    def test_orientation_bearing(self):
        whale = {"location": [0, 0, 0], "rotation": {"roll": 0, "pitch": 0, "yaw": 0}}
        behind = project.Camera([-5000, 0, 300], {"yaw": 0}, 50, 1280, 720)
        side = project.Camera([0, 5000, 300], {"yaw": -90}, 50, 1280, 720)
        self.assertAlmostEqual(project.orientation(whale, behind)["relative_bearing_deg"], 180.0)
        self.assertAlmostEqual(project.orientation(whale, side)["relative_bearing_deg"], 90.0)
        self.assertLess(project.orientation(whale, behind)["nose_toward_camera"], 0)


class PostprocessTests(unittest.TestCase):
    def test_batch_to_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            v = stick_whale()
            rng = np.random.default_rng(2)
            shots = [shotlist.make_shot(i, rng, CONFIG, v) for i in range(2)]
            batch = {"index": 0, "sea_state": shots[0]["sea_state"], "look": shots[0]["look"], "waves": shots[0]["waves"],
                     "ocean": shots[0]["ocean"], "shots": shots, "resolution": [64, 36], "frames_per_shot": 3}
            raw = run / "raw" / "batch_000"
            raw.mkdir(parents=True)
            (raw / "batch.json").write_text(json.dumps(batch))
            (raw / "done").write_text("x")
            for i in range(2):
                frame = i * 3 + 2
                Image.new("RGB", (64, 36), (10, 20, 30)).save(raw / f"FinalImage.{frame:04d}.png")
                mask = np.zeros((36, 64), np.uint8)
                mask[10:12, 20:24] = 255
                Image.fromarray(mask).convert("RGB").save(raw / f"FinalImageM_WhaleVisibleMask.{frame:04d}.png")
            obj = run / "whale.obj"
            obj.write_text("v -800 0 0\nv 800 0 0\nv 0 0 300\nf 1 2 3\n")
            (run / "run_config.json").write_text(json.dumps({"mesh": str(obj)}))
            self.assertEqual(postprocess.main([str(run)]), 0)
            rows = dataset.load_manifest(run)
            self.assertEqual(len(rows), 2)
            for row in rows:
                self.assertTrue((run / "dataset" / row["mask"]).exists())
                self.assertTrue((run / "dataset" / row["amodal"]).exists())
                self.assertEqual(row["visible_pixels"], 8)
                self.assertIn("relative_bearing_deg", row["orientation"])
                label = np.asarray(Image.open(run / "dataset" / row["mask"]))
                amodal = np.asarray(Image.open(run / "dataset" / row["amodal"]))
                self.assertTrue(set(np.unique(label)) <= {0, 1, 2})
                self.assertEqual(int((label == 1).sum()), 8)
                self.assertTrue(((label == 2) <= (amodal > 0)).all())       # through-water only on the whale
                self.assertEqual(row["through_water_pixels"], int((label == 2).sum()))
                canon_cm, defined = postprocess.read_canon(run / "dataset" / "canon" / f"{row['id']}.npz")
                self.assertEqual(canon_cm.shape, amodal.shape + (3,))
                self.assertTrue((defined == (amodal > 0)).all())
                if defined.any():
                    self.assertTrue((np.abs(canon_cm[defined][:, 0]) <= 800.5).all())    # inside the stick whale's x range
                    self.assertTrue(np.isfinite(canon_cm[defined]).all())
            splits = dataset.split(rows, 0.5, 1)
            self.assertEqual(len(splits["train"]) + len(splits["val"]), 2)
            self.assertIn("categories", dataset.summary(rows))



class CanonTests(unittest.TestCase):
    def test_encode_decode_round_trip(self):
        cm = np.full((2, 3, 3), np.nan, np.float32)
        cm[0, 0] = (-805.0, 314.2, -396.0); cm[1, 2] = (805.0, -0.05, 337.0)
        back, defined = postprocess.decode_canon(postprocess.encode_canon(cm))
        self.assertEqual(defined.sum(), 2)
        np.testing.assert_allclose(back[defined], cm[np.isfinite(cm[..., 0])], atol=0.06)


class ThroughWaterTests(unittest.TestCase):
    def test_band_below_mean_water(self):
        amodal = np.array([[1, 1, 1, 1, 0]])
        visible = np.array([[True, False, False, False, False]])
        height = np.array([[50.0, -50.0, -140.0, -160.0, np.nan]])     # cm above the mean plane
        band = postprocess.through_water(amodal, visible, height, depth_m=1.5)
        self.assertEqual(band.tolist(), [[False, True, True, False, False]])
        above = np.array([[50.0, 20.0, -10.0, -400.0, np.nan]])          # not visible but above water: spray/crest
        self.assertEqual(postprocess.through_water(amodal, visible, above, 1.5).tolist(), [[False, False, True, False, False]])


class SizeFloorTests(unittest.TestCase):
    def test_whale_shots_meet_minimum_extent(self):
        v = stick_whale()
        rng = np.random.default_rng(11)
        shots = [shotlist.make_shot(i, rng, CONFIG, v) for i in range(25)]
        cam = CONFIG["camera"]
        # the stick whale has a handful of vertices; a single one above water has no extent
        whale_shots = [s for s in shots if not s["negative"] and s["stats"]["vertex_fraction_above_water"] > 0.05]
        self.assertGreater(len(whale_shots), 5)
        for s in whale_shots:
            self.assertGreaterEqual(s["camera"]["extent_frame_fraction"], cam["min_extent_frame_fraction"] * 0.999, s["name"])
            self.assertGreaterEqual(s["camera"]["in_frame_fraction"], cam["min_in_frame_fraction"], s["name"])

    def test_visible_extent_projection(self):
        world = np.array([[0.0, -400.0, 100.0], [0.0, 400.0, 100.0], [0.0, 0.0, -50.0]])
        camera = {"location": [-2000.0, 0.0, 100.0], "rotation": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}, "focal_mm": 23.76}
        ext = shotlist.visible_extent(world, camera, [1280, 720])
        self.assertEqual(ext["in_frame_fraction"], 1.0)
        self.assertAlmostEqual(ext["extent_frame_fraction"], 0.4, places=3)   # 8 m wide at 20 m with 1 sensor-width focal


class GroundTruthTests(unittest.TestCase):
    def test_body_coordinates_and_landmarks(self):
        v = stick_whale()
        coords = project.body_coordinates(v)
        self.assertAlmostEqual(coords[:, 0].min(), 0.0)
        self.assertAlmostEqual(coords[:, 0].max(), 1.0)
        self.assertAlmostEqual(coords[np.argmax(v[:, 2]), 1], 0.0)        # top of the back = 0 deg
        # the right-side vertex sits at y=300, z=-100: just below the whale's right flank
        self.assertAlmostEqual(coords[np.argmax(v[:, 1]), 1], np.degrees(np.arctan2(300, -100)))
        points = project.landmark_positions(project.landmarks(v), v)
        self.assertAlmostEqual(points["snout"][0], 800.0)
        self.assertAlmostEqual(points["fluke_notch"][0], -800.0)
        self.assertEqual(points["dorsal_fin_tip"][2], 300.0)

    def test_rasterize_attributes_and_pose(self):
        cam = project.Camera([-6000, 0, 0], {"yaw": 0}, 50, 640, 360)
        whale = {"location": [0, 0, 0], "rotation": {"roll": 0, "pitch": 0, "yaw": 90}}
        v = np.array([[-800, 0, -100], [800, 0, -100], [0, 0, 300]], float)
        f = np.array([[0, 1, 2]])
        image, zbuf, body = project.amodal_and_parts(v, f, whale, cam, extra=True)
        self.assertGreater((image > 0).sum(), 100)
        inside = image > 0
        self.assertTrue(np.isfinite(zbuf[inside]).all())
        self.assertTrue((body[inside][:, 0] >= 0).all() and (body[inside][:, 0] <= 1).all())
        pose = project.pose_in_camera(whale, cam)
        np.testing.assert_allclose(pose["translation_cm"], [6000, 0, 0], atol=1e-6)
        np.testing.assert_allclose(pose["rotation"][0], [0, 1, 0], atol=1e-6)  # nose points to camera-right
        kp = project.project_landmarks({"snout": np.array([800.0, 0, -100])}, whale, cam, zbuf, image > 0)
        self.assertTrue(kp["snout"]["in_frame"])
        self.assertFalse(kp["snout"]["occluded"])



class DeformTests(unittest.TestCase):
    def setUp(self):
        from factory import deform
        self.deform = deform
        xs = np.linspace(-800, 800, 65)
        ring = [(0, 100 * np.cos(a), 100 * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 12, endpoint=False)]
        self.v = np.array([[x, y, z] for x in xs for _, y, z in ring], float)

    def test_identity_and_topology(self):
        out = self.deform.deform(self.v, self.deform.DeformParams())
        np.testing.assert_allclose(out, self.v)
        out = self.deform.deform(self.v, self.deform.DeformParams(arch_deg=20, lateral_deg=15))
        self.assertEqual(out.shape, self.v.shape)

    def test_tail_curl_lifts_flukes_and_keeps_length(self):
        out = self.deform.deform(self.v, self.deform.DeformParams(tail_curl_deg=40))
        tail = self.v[:, 0] < -700
        self.assertGreater(out[tail, 2].mean(), self.v[tail, 2].mean() + 100)
        head = self.v[:, 0] > 700
        np.testing.assert_allclose(out[head], self.v[head], atol=1e-6)  # the head does not move
        # arc length of the spine is preserved (bending, not stretching)
        spine = out[::12]
        self.assertAlmostEqual(float(np.linalg.norm(np.diff(spine, axis=0), axis=1).sum()), 1600.0, delta=20)

    def test_head_pitch_and_lateral(self):
        up = self.deform.deform(self.v, self.deform.DeformParams(head_pitch_deg=15))
        self.assertGreater(up[self.v[:, 0] > 750, 2].mean(), self.v[self.v[:, 0] > 750, 2].mean() + 20)
        side = self.deform.deform(self.v, self.deform.DeformParams(lateral_deg=20))
        self.assertGreater(side[self.v[:, 0] < -700, 1].mean(), 50)  # positive lateral sweeps the tail toward +y

    def test_params_round_trip_and_sampling(self):
        p = self.deform.sample_params(np.random.default_rng(1), "breach")
        d = self.deform.params_to_dict(p)
        self.assertEqual(self.deform.params_from_dict(d), self.deform.DeformParams(**d))
        self.assertTrue(self.deform.params_from_dict(None).is_identity())


class SkinTests(unittest.TestCase):
    def test_variant_changes_dark_skin_only(self):
        from factory import skin
        base = np.full((64, 64, 3), 30, np.uint8)
        base[:16] = 240  # a white patch
        out, recipe = skin.make_variant(base, seed=3)
        self.assertEqual(out.shape, base.shape)
        np.testing.assert_array_equal(out[:16], base[:16])
        self.assertIn("darkness", recipe)


if __name__ == "__main__":
    unittest.main()
