import unittest
from PIL import Image
from src.data import items, prepare, reference_image


class TestOriginalSources(unittest.TestCase):
    def test_all_originals_have_verified_native_pixels(self):
        assets = items()
        self.assertEqual(len(assets), 4)
        self.assertEqual(len({item["photo_sha256"] for item in assets}), 4)
        for item in assets:
            with self.subTest(item=item["id"]):
                self.assertTrue("/images/original/" in item["source_url"])
                self.assertGreater(min(item["source_size"]), 512)
                result, _, digest = prepare(item, 512)
                self.assertEqual(result.size, (512, 512))
                self.assertEqual(digest, item["photo_sha256"])

    def test_undersized_source_is_refused(self):
        with self.assertRaises(ValueError):
            reference_image(Image.new("RGB", (128, 256)), 512)
