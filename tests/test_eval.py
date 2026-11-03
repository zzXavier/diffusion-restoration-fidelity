import unittest
from PIL import Image
from run import score, degrade


class TestEval(unittest.TestCase):
    def test_known_truth(self):
        im = Image.new("RGB", (512, 512), "navy")
        self.assertEqual(score(im, im)["excess_edge_energy"], 0)
        self.assertGreater(score(im, im)["psnr_db"], 100)
        self.assertEqual(degrade(im, "noise").tobytes(), degrade(im, "noise").tobytes())

    def test_mismatch(self):
        with self.assertRaises(ValueError):
            score(Image.new("RGB", (16, 16)), Image.new("RGB", (8, 8)))
