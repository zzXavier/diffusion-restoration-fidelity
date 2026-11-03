"""Prepare SHA256-verified v2.1 weights in the upstream cache; supports resume."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
import time
import requests

try:
    from .restore import locate
except ImportError:
    from restore import locate

REV = "fd6263503564bfbfa156665a90832e1b729434f4"
FILES = {
    "realesrgan_s4_swinir_100k.pth": (
        90208287,
        "577b8f84815d324e5ee17ef3c85d8bc9b7ebd05b20a6d61a5452b66e7e24dcaf",
    ),
    "DiffBIR_v2.1.pt": (
        1452744042,
        "49477de737be9028ac2d9352cf45633324a0090afde57a7aa358321dd6a1de53",
    ),
    "sd2.1-base-zsnr-laionaes5.ckpt": (
        5160249169,
        "30bdda1968c2dae28ff23b2b44b0f2a87058d76eea5fc4eadfa3448e40e85cf9",
    ),
}


def local_hash(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def fetch_one(folder, name, size, expected, tail_only=False):
    dest = folder / name
    if dest.exists() and local_hash(dest) == expected:
        return
    cached = folder.parent.parent / name
    if cached.exists() and local_hash(cached) == expected:
        import shutil

        shutil.copyfile(cached, dest)
        return
    parts = folder / (name + ".parts")
    parts.mkdir(exist_ok=True)
    chunk_size = 8 * 1024 * 1024

    def part(start):
        end = min(start + chunk_size, size) - 1
        path = parts / f"{start:012d}"
        target = end - start + 1
        if path.exists() and path.stat().st_size == target:
            return path
        tmp = path.with_suffix(f".{os.getpid()}.pending")
        for i in range(5):
            try:
                with requests.get(
                    f"https://huggingface.co/lxq007/DiffBIR-v2/resolve/{REV}/{name}",
                    params={"segment": start},
                    headers={"Range": f"bytes={start}-{end}"},
                    timeout=(15, 55),
                    stream=True,
                ) as r:
                    r.raise_for_status()
                    if r.headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
                        raise IOError("Unexpected byte range")
                    with tmp.open("wb") as out:
                        for b in r.iter_content(1024 * 1024):
                            out.write(b)
                if tmp.stat().st_size != target:
                    raise IOError("Incomplete fragment")
                tmp.replace(path)
                return path
            except (requests.RequestException, IOError):
                if i == 4:
                    raise
                time.sleep(2 + i)

    starts = range(0, size, chunk_size)
    if tail_only:
        starts = reversed(starts[len(starts) // 2 :])
    with ThreadPoolExecutor(max_workers=5) as executor:
        downloaded = list(executor.map(part, starts))
    if tail_only:
        return
    tmp = dest.with_suffix(".assembling")
    digest = hashlib.sha256()
    with tmp.open("wb") as out:
        for p in downloaded:
            data = p.read_bytes()
            digest.update(data)
            out.write(data)
    if digest.hexdigest() != expected:
        raise IOError(f"Checkpoint hash mismatch: {name}")
    tmp.replace(dest)
    for p in downloaded:
        p.unlink()
    parts.rmdir()
    print(f"Verified {name} ({size:,} bytes)", flush=True)


def fetch():
    source = locate()
    folder = source / "weights"
    folder.mkdir(exist_ok=True)
    for name, (size, digest) in FILES.items():
        fetch_one(folder, name, size, digest)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tail-only",
        action="store_true",
        help="Pre-fetch upper half of the SD backbone while a normal fetch is running.",
    )
    if parser.parse_args().tail_only:
        filename = "sd2.1-base-zsnr-laionaes5.ckpt"
        fetch_one(locate() / "weights", filename, *FILES[filename], tail_only=True)
    else:
        fetch()
