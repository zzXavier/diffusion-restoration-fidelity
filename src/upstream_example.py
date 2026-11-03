"""Run the pinned DiffBIR repository's real comic3.png input."""

import argparse
import hashlib
import json
import shutil
from PIL import Image
from src.restore import ROOT, locate, restore


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(stage1):
    repo = locate()
    origin = repo / "inputs" / "demo" / "bsr" / "comic3.png"
    path = ROOT / ".cache" / "upstream-example"
    inp = path / "degraded"
    inp.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(origin, inp / origin.name)
    if stage1:
        from src.stage1 import predict

        predict(path, "cpu")
        keys = ["stage1"]
    else:
        restore(path)
        keys = ["stage1", "diffusion"]
    result = {
        "case": "upstream inputs/demo/bsr/comic3.png",
        "input_sha256": sha(origin),
        "upscale": 2,
        "weights_revision": "fd6263503564bfbfa156665a90832e1b729434f4",
        "mode": "stage1-CPU" if stage1 else "full-diffusion",
        "outputs": {},
    }
    for key in keys:
        file = path / key / "comic3.png"
        result["outputs"][key] = {
            "sha256": sha(file),
            "size": list(Image.open(file).size),
        }
    dest = ROOT / "results" / "upstream"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / ("stage1.json" if stage1 else "diffusion.json")).write_text(
        json.dumps(result, indent=2)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage1-only", action="store_true")
    main(p.parse_args().stage1_only)
