import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from factory import qa


class QaTests(unittest.TestCase):
    def test_outline_is_mask_border(self):
        mask = np.zeros((10, 10), bool)
        mask[2:8, 2:8] = True
        outline = qa.mask_outline(mask)
        self.assertEqual(int(outline.sum()), 20)  # perimeter of a 6x6 square
        self.assertFalse(outline[4, 4])

    def test_overlay_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            rgb = Path(tmp) / "rgb.png"; mask = Path(tmp) / "mask.png"; out = Path(tmp) / "o.png"
            Image.new("RGB", (20, 10), (0, 0, 255)).save(rgb)
            m = np.zeros((10, 20), np.uint8); m[3:6, 5:15] = 255
            Image.fromarray(m).save(mask)
            stats = qa.overlay(rgb, mask, out)
            self.assertEqual(stats["whale_pixels"], 30)
            self.assertEqual(stats["bbox_xyxy"], [5, 3, 14, 5])
            self.assertTrue(out.exists())


if __name__ == "__main__":
    unittest.main()


class ContactSheetTests(unittest.TestCase):
    def test_sheet_layout(self):
        import tempfile
        from PIL import Image
        from factory.qa import contact_sheet
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for name in ("a", "b", "c", "d"):
                d = Path(tmp) / name
                d.mkdir()
                Image.new("RGB", (64, 32), (10, 20, 30)).save(d / "overlay.png")
                paths.append(d / "overlay.png")
            out = Path(tmp) / "sheet.png"
            contact_sheet(out, paths, columns=3, width=64)
            sheet = Image.open(out)
            self.assertEqual(sheet.size, (3 * 64, 2 * (32 + 24)))
