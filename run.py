import argparse
import csv
import json
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


def evaluate(out, baseline=False, stage1_only=False):
    records = json.loads((out / "cases.json").read_text())
    rows = []
    methods = (
        ["bicubic", "unsharp"]
        + ([] if baseline else ["stage1"])
        + ([] if baseline or stage1_only else ["diffusion"])
    )
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


def run(out, demo, baseline, stage1_only=False):
    for folder in ("truth", "degraded", "bicubic", "unsharp"):
        (out / folder).mkdir(parents=True, exist_ok=True)
    records = []
    subjects = [x for x in items() if x["id"] != "B083V9C8D2"]
    for item in subjects[:1] if demo else subjects:
        clean, _, _ = prepare(item, size=512)
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
    evaluate(out, baseline, stage1_only)


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
