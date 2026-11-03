# Handoff

The primary run is results/formal, generated from the four bundled ABO originals under examples/originals. Native widths/heights and SHA-256 are fixed in configs/products.json and validated on load. The latest formal manifest must say completed and list the exact source hashes. The four original URLs were independently downloaded and compared bytewise before this run.

- Profiles: Python 3.11.9, PyTorch 2.4.1+cu121, NumPy 1.26.4, RTX 4060 Laptop 8 GiB, DiffBIR v2.1, 10 steps and 2× restoration. Three exact weight hashes are in src/fetch_full.py.
- Successful entrypoints: python -m src.upstream_example, python run.py reproduce and python run.py evaluate. The 12-case formal run, including six held-out cases, contains all four outputs per input. Synchronized time averaged 4.25 s/image and the maximum peak allocation was 4,783 MiB.
- Held-out primary results: bicubic 26.969 dB, unsharp 26.079, SwinIR 26.896 and full diffusion 22.758. Review the blurred coffee tin and packaging crop in results/formal/highest-excess-edge.jpg.
- Pilot only: results/pilot-thumbnail summarizes the earlier four-product thumbnail experiment. Its 23.697 dB full-diffusion score and old CPU clean-clone record must not be reported as native-resolution results.
- Next: use the newly pushed private remote in a clean clone and a separate GPU environment to run demo, reproduce and evaluate. Record independent fidelity and manifest comparisons before publication.

