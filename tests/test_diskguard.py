import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

from factory import diskguard


class EnsureFreeTests(unittest.TestCase):
    def test_passes_with_zero_floor(self):
        remaining = diskguard.ensure_free(tempfile.gettempdir(), need_gib=0, min_free_gib=0)
        self.assertGreaterEqual(remaining, 0)

    def test_refuses_when_floor_unreachable(self):
        with self.assertRaises(diskguard.InsufficientDiskSpace):
            diskguard.ensure_free(tempfile.gettempdir(), min_free_gib=1e9)

    def test_need_counts_against_floor(self):
        free = diskguard.free_gib(tempfile.gettempdir())
        with self.assertRaises(diskguard.InsufficientDiskSpace):
            diskguard.ensure_free(tempfile.gettempdir(), need_gib=free + 1, min_free_gib=0)

    def test_nonexistent_output_path_uses_parent_filesystem(self):
        missing = Path(tempfile.gettempdir()) / "does-not-exist" / "renders" / "run_0001"
        self.assertGreater(diskguard.free_bytes(missing), 0)


class EstimateTests(unittest.TestCase):
    def test_scales_with_shots_and_passes(self):
        one = diskguard.estimate_render_bytes(1, 100, 100)
        self.assertEqual(diskguard.estimate_render_bytes(10, 100, 100), 10 * one)
        self.assertGreater(diskguard.estimate_render_bytes(1, 100, 100, n_label_passes=3), one)

    def test_poc_batch_is_bounded(self):
        gib = diskguard.estimate_render_bytes(2000, 1536, 1024) / diskguard.GIB
        self.assertTrue(10 < gib < 20, gib)


class ZipTests(unittest.TestCase):
    def test_uncompressed_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.zip"
            with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("Engine/x.bin", b"\0" * 50_000)
                archive.writestr("Engine/y.txt", "hello")
            self.assertEqual(diskguard.zip_uncompressed_bytes(path), 50_005)


class RunGuardedTests(unittest.TestCase):
    def test_returns_child_exit_code(self):
        code = diskguard.run_guarded(
            [sys.executable, "-c", "import sys; sys.exit(3)"],
            paths=[tempfile.gettempdir()], kill_below_gib=0, warn_below_gib=0, interval_s=0.1,
        )
        self.assertEqual(code, 3)

    def test_kills_when_disk_low(self):
        code = diskguard.run_guarded(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            paths=[tempfile.gettempdir(), Path.home()], kill_below_gib=10, warn_below_gib=20,
            interval_s=0.1, grace_s=5, free_fn=lambda p: 1.0 if Path(p) == Path.home() else 500.0,
        )
        self.assertEqual(code, diskguard.EXIT_LOW_DISK)


if __name__ == "__main__":
    unittest.main()
