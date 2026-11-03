# Detail fidelity in diffusion restoration

How well does a restoration keep the letters printed on a package? This project compares four methods against reference images downsampled from full-resolution original photographs. In this held-out blurred coffee-tin case, the complete diffusion pass sharpens the label but invents letters.

![Native-resolution coffee tin reference and measured restoration outputs](results/formal/highest-excess-edge.jpg)

After installing the environment below, run `python run.py demo` for one reference and its four restoration methods. The faster `python run.py demo --classical-only` uses CPU and compares the conventional baselines.

## Reproduce

Use Python 3.11. Install CUDA-enabled PyTorch 2.4.1 and torchvision 0.19.1 first, then `python -m pip install -r requirements-gpu.txt`. The code downloads and checks the official DiffBIR checkout and three SHA-256-pinned weights. To share existing weights, set DIFFBIR_SOURCE to that exact checkout with its `weights/` directory populated.

```sh
python -m unittest discover -s tests
python -m src.upstream_example
python run.py demo
python run.py reproduce
python run.py evaluate
```

The upstream check uses the repository's own `inputs/demo/bsr/comic3.png` with two-times restoration. The reported experiment used an RTX 4060 Laptop GPU (8 GiB), fp16, 10 diffusion steps, a fixed seed and tiled VAE / denoising. [Official-input reproduction](results/upstream/diffusion.json)

## Sources and design

The repository contains four unmodified ABO catalog JPEGs with native dimensions **898×1831, 2006×2560, 700×1000 and 870×2560**. The input loader verifies each file's SHA-256 and dimensions, downsamples it with Lanczos and centers it on a white 512×512 reference canvas. It refuses undersized inputs. Every reference is genuinely downsampled from the corresponding original.

Two products are for development and two are held out. A separate blur, Gaussian noise or JPEG degradation produces a 256×256 low-quality image for each product. Each of the twelve low-quality images receives bicubic, unsharp-mask, actual intermediate SwinIR and full DiffBIR v2.1 restoration. Metrics below average only the **six held-out cases**. [Four original JPEGs and attribution](data.md) · [Source manifest](configs/products.json)

| Method | PSNR (dB) ↑ | Excess edge energy ↓ | Label-crop MAE (unit RGB) ↓ |
|---|---:|---:|---:|
| Bicubic | 26.969 | 0.00586 | 0.04341 |
| Unsharp mask | 26.079 | 0.01244 | 0.04800 |
| SwinIR first stage | 26.896 | 0.00189 | 0.03913 |
| Full DiffBIR v2.1 | 22.758 | 0.02543 | 0.06114 |

Bicubic had the highest average PSNR by 0.073 dB, while SwinIR had lower label-region pixel error and lower excess edge energy. Full diffusion produced the greatest excess edge energy. Its worst held-out edge case was the blurred coffee tin (19.558 dB). Across all twelve inputs the two-stage GPU pipeline averaged **4.25 seconds per image**, with maximum allocated CUDA memory of **4,783 MiB**. Timing excludes model loading. The fixed crop and edge score measure pixel-level differences; visual inspection is needed to assess the changed text. Generalization outside these four products and this short sampling configuration remains untested. [Per-case records](results/formal/scores.csv) · [Primary summary](results/formal/summary.json) · [Runtime](results/formal/run-environment.json)

## Implementation and provenance

- `src/data.py` validates the original JPEGs and creates paired references. `run.py` applies degradations, records the actual source hashes, runs the baselines and verifies the complete set of per-method outputs. Use `--output NEW_DIRECTORY` to create another run.
- `src/restore.py` invokes the authors' BSRInferenceLoop, captures the actual first-stage tensor before diffusion and records synchronized runtime. Fixed checkpoints and resumable byte ranges are in `src/fetch_full.py`.
- `python run.py reproduce --stage1-only` runs the real SwinIR cleaner on CPU. `python run.py evaluate` verifies the formal run before recalculating metrics.
- The old **23.697 dB** diffusion result came from enlarging 256px thumbnails to create nominal references. It is retained solely as a separate [thumbnail pilot](results/pilot-thumbnail/README.md). All primary results and images above use the verified full-resolution originals.

[Upstream implementation](UPSTREAM.md) · [Handoff](HANDOFF.md)

An independent GitHub clean clone regenerated one demo and all twelve formal cases using its own Python environment. All 72 reference, degraded and predicted PNGs were byte-identical, with zero metric differences across all 48 method rows. Nine source, evaluation and state-integrity tests passed. [Clean-clone audit](results/clean-clone.json)
