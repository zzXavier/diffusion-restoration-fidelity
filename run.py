import argparse
import csv
import hashlib
import json
import os
import platform
import sqlite3
import tempfile
from contextlib import closing, contextmanager
from io import BytesIO
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy.ndimage import laplace
from src.data import ROOT, items, prepare
from src.restore import restore


def degrade(im, kind):
    out = im.resize((256, 256), Image.Resampling.LANCZOS)
    if kind == "blur":
        return out.filter(ImageFilter.GaussianBlur(1.4))
    if kind == "noise":
        rng = np.random.default_rng(29)
        return Image.fromarray(
            np.clip(np.asarray(out) + rng.normal(0, 9, (*out.size, 3)), 0, 255).astype(
                "uint8"
            )
        )
    if kind == "jpeg":
        buf = BytesIO()
        out.save(buf, format="JPEG", quality=22)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    raise ValueError(kind)


def score(target, pred):
    if target.size != pred.size:
        raise ValueError("Prediction dimensions differ from reference")
    x = np.asarray(target, dtype=float) / 255
    y = np.asarray(pred, dtype=float) / 255
    mse = np.mean((x - y) ** 2)
    high = lambda z: laplace(z.mean(axis=-1))
    # Excess predicted high-frequency magnitude is an edge-based artifact proxy.
    return {
        "psnr_db": float(-10 * np.log10(max(mse, 1e-12))),
        "excess_edge_energy": float(
            np.maximum(0, np.abs(high(y)) - np.abs(high(x))).mean()
        ),
        "label_crop_mae": float(np.abs(x - y)[200:312, 174:338].mean()),
    }


def method_names(baseline, stage1_only):
    return (
        ["bicubic", "unsharp"]
        + ([] if baseline else ["stage1"])
        + ([] if baseline or stage1_only else ["diffusion"])
    )


@contextmanager
def output_lock(out):
    # OS-backed SQLite locks are released even if the process is terminated.
    key = hashlib.sha256(os.path.normcase(str(out.resolve())).encode()).hexdigest()
    path = Path(tempfile.gettempdir()) / f"diffbir-output-{key}.lock"
    with closing(sqlite3.connect(path, timeout=0)) as lock:
        try:
            lock.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as exc:
            raise RuntimeError(f"Output directory is already in use: {out}") from exc
        yield


def save_state(out, state):
    tmp = out / ".run-state.tmp"
    tmp.write_text(json.dumps(state, indent=2, allow_nan=False) + "\n", encoding="utf8")
    tmp.replace(out / "manifest.json")


def checked_files(out, mode, methods):
    subjects = [s for s in items() if s["id"] != "B083V9C8D2"]
    if mode not in ("demo", "formal"):
        raise ValueError("Unknown run profile")
    if mode == "demo":
        subjects = subjects[:1]
    kinds = ["jpeg"] if mode == "demo" else ["blur", "noise", "jpeg"]
    expected = [
        {"id": s["id"], "split": s["split"], "kind": k, "file": f"{s['id']}-{k}.png"}
        for s in subjects for k in kinds
    ]
    if not expected or json.loads((out / "cases.json").read_text()) != expected:
        raise ValueError("Unexpected or incomplete case list")
    files = [f"{folder}/{row['file']}" for row in expected for folder in ("truth", "degraded", *methods)]
    if "diffusion" in methods:
        files += [f"diffusion/{Path(row['file']).stem}.json" for row in expected]
    elif "stage1" in methods:
        files.append("stage1/runtime.json")
    hashes = {}
    for relative in files:
        path = out / relative
        if not path.is_file():
            raise ValueError(f"Missing required output: {relative}")
        if path.suffix == ".json":
            # Ignore platform line endings when verifying tracked JSON.
            value = json.loads(path.read_text())
            if relative == "stage1/runtime.json":
                if set(value["cases"]) != {r["file"] for r in expected}:
                    raise ValueError("Incomplete stage1 runtime cases")
            elif not all(np.isfinite(value[k]) and value[k] > 0 for k in ("seconds", "peak_allocated_mb")):
                raise ValueError(f"Invalid diffusion runtime record: {relative}")
            payload = json.dumps(value, sort_keys=True, allow_nan=False).encode()
        else:
            payload = path.read_bytes()
            size = (256, 256) if relative.startswith("degraded/") else (512, 512)
            with Image.open(BytesIO(payload)) as image:
                if image.size != size:
                    raise ValueError(f"Wrong image dimensions: {relative}")
                image.verify()
        hashes[relative] = hashlib.sha256(payload).hexdigest()
    return expected, hashes


def evaluate(out, baseline=False, stage1_only=False):
    methods = method_names(baseline, stage1_only)
    with output_lock(out):
        marker = out / "manifest.json"
        if marker.is_file():
            state = json.loads(marker.read_text())
        else:
            # Unmarked legacy copies are eligible only if their complete file set
            # exactly matches an independently verified, shipped reference run.
            reference = ROOT / "results" / ("formal" if "diffusion" in methods else "classical" if "stage1" in methods else "baselines") / "manifest.json"
            if not reference.is_file():
                raise ValueError("No completed run manifest; reproduce into a fresh output")
            state = json.loads(reference.read_text())
        if state.get("status") != "completed":
            raise ValueError("Run incomplete; only completed outputs can be evaluated")
        if state["methods"] != methods:
            raise ValueError("Requested methods differ from completed run")
        records, hashes = checked_files(out, state["mode"], methods)
        source_ids = {r["id"] for r in records}
        sources = {s["id"]: s["photo_sha256"] for s in items() if s["id"] in source_ids}
        if state.get("source_sha256") != sources:
            raise ValueError("Recorded source hashes differ from the pinned data configuration")
        if hashes != state["sha256"]:
            raise ValueError("Artifact hash mismatch")
        write_scores(out, records, methods)


def write_scores(out, records, methods):
    rows = []
    for case in records:
        name = case["file"]
        target = Image.open(out / "truth" / name)
        for method in methods:
            pred = Image.open(out / method / name)
            rows.append({**case, "method": method, **score(target, pred)})
    with (out / "scores.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)
    groups = {}
    for m in sorted(set(r["method"] for r in rows)):
        test = [r for r in rows if r["method"] == m and r["split"] == "test"]
        if test:
            groups[m] = {
                k: float(np.mean([r[k] for r in test]))
                for k in ("psnr_db", "excess_edge_energy", "label_crop_mae")
            }
    (out / "summary.json").write_text(json.dumps(groups, indent=2))
    picked = next((c for c in records if c["split"] == "test"), records[0])
    names = ["truth", *methods]
    render_case(out, names, picked["file"], "comparison.jpg")
    test = [r for r in rows if r["method"] == methods[-1] and r["split"] == "test"]
    if test:
        worst = max(test, key=lambda x: x["excess_edge_energy"])
        render_case(out, names, worst["file"], "highest-excess-edge.jpg")
        (out / "highest-excess-edge.json").write_text(json.dumps(worst, indent=2))


def render_case(out, names, name, destination):
    panel = Image.new("RGB", (len(names) * 176, 245), "white")
    d = ImageDraw.Draw(panel)
    for n, model in enumerate(names):
        im = Image.open(out / model / name)
        panel.paste(im.resize((176, 176)), (176 * n, 17))
        panel.paste(im.crop((174, 200, 338, 248)).resize((176, 52)), (176 * n, 193))
        d.text((176 * n + 3, 2), model, fill="black")
    panel.save(out / destination)


def generate(out, demo, baseline, stage1_only):
    for folder in ("truth", "degraded", "bicubic", "unsharp"):
        (out / folder).mkdir(parents=True, exist_ok=True)
    records = []
    sources = {}
    subjects = [x for x in items() if x["id"] != "B083V9C8D2"]
    for item in subjects[:1] if demo else subjects:
        clean, _, digest = prepare(item, size=512)
        if digest != item["photo_sha256"]:
            raise ValueError(f"Source hash mismatch: {item['id']}")
        sources[item["id"]] = digest
        for kind in ["jpeg"] if demo else ["blur", "noise", "jpeg"]:
            p = item["id"] + "-" + kind + ".png"
            low = degrade(clean, kind)
            simple = low.resize(clean.size, Image.Resampling.BICUBIC)
            clean.save(out / "truth" / p)
            low.save(out / "degraded" / p)
            simple.save(out / "bicubic" / p)
            simple.filter(
                ImageFilter.UnsharpMask(radius=1.2, percent=135, threshold=3)
            ).save(out / "unsharp" / p)
            records.append(
                {"id": item["id"], "split": item["split"], "kind": kind, "file": p}
            )
    (out / "cases.json").write_text(json.dumps(records, indent=2))
    if stage1_only:
        from src.stage1 import predict

        predict(out, "cpu")
    elif not baseline:
        restore(out)
    return sources


def run(out, demo, baseline, stage1_only=False):
    methods = method_names(baseline, stage1_only)
    state = {"status": "in_progress", "mode": "demo" if demo else "formal", "methods": methods}
    out.mkdir(parents=True, exist_ok=True)
    with output_lock(out):
        save_state(out, state)
        cache = out / ".cache"
        cache.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as scratch:
            fresh = Path(scratch)
            sources = generate(fresh, demo, baseline, stage1_only)
            records, hashes = checked_files(fresh, state["mode"], methods)
            write_scores(fresh, records, methods)
            if "diffusion" in methods:
                import torch
                from src.restore import PIN
                costs = [json.loads((fresh / "diffusion" / Path(r["file"]).with_suffix(".json")).read_text()) for r in records]
                runtime = {
                    "measurement": "Synchronized per-input GPU timing excludes model loading",
                    "upstream": PIN, "device": torch.cuda.get_device_name(),
                    "torch": str(torch.__version__), "numpy": np.__version__, "python": platform.python_version(),
                    "steps": 10, "upscale": 2, "precision": "fp16", "cases": len(records),
                    "mean_wall_seconds": float(np.mean([c["seconds"] for c in costs])),
                    "max_allocated_mib": max(c["peak_allocated_mb"] for c in costs),
                }
                (fresh / "run-environment.json").write_text(json.dumps(runtime, indent=2))
            for file in fresh.rglob("*"):
                if file.is_file():
                    destination = out / file.relative_to(fresh)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    file.replace(destination)
            save_state(out, state | {"status": "completed", "source_sha256": sources, "sha256": hashes})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["demo", "reproduce", "evaluate"])
    g = p.add_mutually_exclusive_group()
    g.add_argument("--classical-only", action="store_true")
    g.add_argument("--stage1-only", action="store_true")
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    group = (
        "classical" if a.stage1_only else "baselines" if a.classical_only else "formal"
    )
    if a.action == "demo":
        group = "demo" if group == "formal" else f"demo-{group}"
    out = a.output or ROOT / "results" / group
    if a.action == "evaluate":
        evaluate(out, a.classical_only, a.stage1_only)
    else:
        run(out, a.action == "demo", a.classical_only, a.stage1_only)
