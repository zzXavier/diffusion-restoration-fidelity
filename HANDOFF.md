# Handoff

The original upstream comic3 sample and the complete held-out experiment ran on an 8 GiB RTX 4060 Laptop using DiffBIR v2.1, CUDA fp16, 2× restoration and 10 steps. The upstream freeze is 5c2d6c177c3f2965e6d33a32931248d242ef7b3d; the three model digests are in src/fetch_full.py.

- Main run: results/formal. Twelve degraded inputs each have a reference, bicubic restoration, unsharp restoration, actual intermediate cleaner prediction and full diffusion prediction. Independent saved-image rescoring: python run.py evaluate. Input-level wall-clock costs and CUDA peaks are in results/formal/diffusion/*.json.
- Primary results: stage1 PSNR 30.428 dB, full diffusion 23.697 dB on six held-out cases. The blurred coffee tin has the highest excess edge proxy in the diffusion outputs. Inspect results/formal/highest-excess-edge.jpg. Per-case model cost excludes initial loading.
- Supporting run: results/classical (CPU SwinIR and both baselines). A separate private GitHub clean clone at bfee3697 reproduced all twelve CPU SwinIR PNGs byte-for-byte. Two automated tests pass.
- To resume from an existing verified model cache, point DIFFBIR_SOURCE at the exact upstream checkout containing its weights directory. Run python run.py demo and python run.py reproduce with a separate --output for an independent full-GPU check.
- Publication gate: rerun the full GPU experiment and demo from the private remote's fresh clone, compare outputs, and record the result.

