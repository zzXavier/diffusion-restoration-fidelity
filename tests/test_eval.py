import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from PIL import Image
from run import score, degrade, evaluate, run, output_lock


class TestEval(unittest.TestCase):
    def test_known_truth(self):
        im = Image.new("RGB", (512, 512), "navy")
        self.assertEqual(score(im, im)["excess_edge_energy"], 0)
        self.assertGreater(score(im, im)["psnr_db"], 100)
        self.assertEqual(degrade(im, "noise").tobytes(), degrade(im, "noise").tobytes())

    def test_mismatch(self):
        with self.assertRaises(ValueError):
            score(Image.new("RGB", (16, 16)), Image.new("RGB", (8, 8)))

    def test_interrupted_rerun_invalidates_existing_results(self):
        example = {"id": "sample", "split": "test", "photo_sha256": "a" * 64}
        image = Image.new("RGB", (512, 512), "navy")
        with TemporaryDirectory() as temp, patch("run.items", return_value=[example]), patch("run.prepare", return_value=(image, None, "a" * 64)):
            out = Path(temp)
            run(out, True, True)
            with patch("run.prepare", side_effect=RuntimeError("interrupted")):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    run(out, True, True)
            with self.assertRaisesRegex(ValueError, "incomplete|completed"):
                evaluate(out, baseline=True)
            run(out, True, True)
            evaluate(out, baseline=True)

    def test_old_images_cannot_complete_a_new_diffusion_run(self):
        example = {"id": "sample", "split": "test", "photo_sha256": "a" * 64}
        image = Image.new("RGB", (512, 512), "navy")
        with TemporaryDirectory() as temp, patch("run.items", return_value=[example]), patch("run.prepare", return_value=(image, None, "a" * 64)):
            out = Path(temp)
            for method in ("stage1", "diffusion"):
                (out/method).mkdir()
                image.save(out/method/"sample-jpeg.png")
            with patch("run.restore", return_value=None):
                with self.assertRaisesRegex(ValueError, "missing|incomplete|Missing"):
                    run(out, True, False)
            with self.assertRaisesRegex(ValueError, "incomplete|completed"):
                evaluate(out)

    def test_completed_run_rejects_wrong_methods_cases_and_changed_pixels(self):
        example = {"id": "sample", "split": "test", "photo_sha256": "a" * 64}
        image = Image.new("RGB", (512, 512), "navy")
        with TemporaryDirectory() as temp, patch("run.items", return_value=[example]), patch("run.prepare", return_value=(image, None, "a" * 64)):
            out = Path(temp)
            run(out, False, True)
            with self.assertRaisesRegex(ValueError, "method|profile"):
                evaluate(out)
            cases = out / "cases.json"
            original = cases.read_bytes()
            cases.write_text(json.dumps(json.loads(original)[:-1]))
            with self.assertRaisesRegex(ValueError, "case"):
                evaluate(out, baseline=True)
            cases.write_bytes(original)
            marker = out / "manifest.json"
            raw = marker.read_bytes()
            forged = json.loads(raw)
            forged["source_sha256"]["sample"] = "b" * 64
            marker.write_text(json.dumps(forged))
            with self.assertRaisesRegex(ValueError, "source|Source"):
                evaluate(out, baseline=True)
            marker.write_bytes(raw)
            Image.new("RGB", (512, 512), "red").save(out/"bicubic/sample-jpeg.png")
            with self.assertRaisesRegex(ValueError, "hash|digest|integrity"):
                evaluate(out, baseline=True)

    def test_concurrent_writer_does_not_change_published_state(self):
        with TemporaryDirectory() as temp:
            out = Path(temp)
            marker = out / "manifest.json"
            marker.write_text('{"status":"completed"}')
            with output_lock(out), self.assertRaisesRegex(RuntimeError, "already in use"):
                run(out, True, True)
            self.assertEqual(json.loads(marker.read_text()), {"status": "completed"})

    def test_unmarked_archive_requires_a_known_identical_reference(self):
        example = {"id": "sample", "split": "test", "photo_sha256": "a" * 64}
        source = Image.new("RGB", (512, 512), "navy")
        with TemporaryDirectory() as temp, patch("run.items", return_value=[example]), patch("run.prepare", return_value=(source, None, "a" * 64)), patch("run.ROOT", Path(temp)):
            out = Path(temp)/"legacy"
            run(out, True, True)
            marker = out/"manifest.json"
            prior = marker.read_bytes()
            marker.unlink()
            with self.assertRaisesRegex(ValueError, "completed"):
                evaluate(out, baseline=True)
            reference = Path(temp)/"results/baselines/manifest.json"
            reference.parent.mkdir(parents=True)
            reference.write_bytes(prior)
            evaluate(out, baseline=True)
            Image.new("RGB", (512, 512), "red").save(out/"bicubic/sample-jpeg.png")
            with self.assertRaisesRegex(ValueError, "hash"):
                evaluate(out, baseline=True)
