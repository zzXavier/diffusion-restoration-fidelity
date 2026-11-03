"""Load verified native-resolution Amazon Berkeley Objects photographs."""

from io import BytesIO
import hashlib
import json
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def reference_image(source, size):
    if size < 1 or max(source.size) < size:
        raise ValueError(
            f"Source {source.size} must not be enlarged into a {size}px reference"
        )
    image = source.convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    left = (size - image.width) // 2
    top = (size - image.height) // 2
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(image, (left, top))
    return canvas


def prepare(item, size=512):
    """Return the downsampled reference, reserved mask slot and true source SHA-256."""
    name = Path(item["photo"]).name
    original = ROOT / "examples" / "originals" / name
    raw = original.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != item["photo_sha256"]:
        raise ValueError(f"ABO original SHA-256 mismatch: {name}")
    with Image.open(BytesIO(raw)) as im:
        if im.format != "JPEG" or list(im.size) != item["source_size"]:
            raise ValueError(f"Original format or dimensions differ: {name}")
        return reference_image(im, size), None, digest


def items():
    return json.loads(
        (ROOT / "configs" / "products.json").read_text(encoding="utf-8-sig")
    )
