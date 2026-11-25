#!/usr/bin/env python
"""
Download the model checkpoints needed for reproduction into ``checkpoint/``.

Every file is verified against a hard-coded SHA-256 checksum. Warns (and exits
non-zero) if a download link is broken, serves an HTML page instead of the
file, or delivers a file whose size or checksum does not match.

Usage:
    uv run python script/download_checkpoints.py            # download missing checkpoints
    uv run python script/download_checkpoints.py --check    # only verify that links are alive
    uv run python script/download_checkpoints.py --force    # re-download existing files
"""
import argparse
import hashlib
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import NamedTuple

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoint"


class Checkpoint(NamedTuple):
    url: str
    size: int  # bytes
    sha256: str


CHECKPOINTS = {
    "labram-base.pth": Checkpoint(
        url="https://github.com/935963004/LaBraM/raw/main/checkpoints/labram-base.pth",
        size=96_612_769,
        sha256="7c50583826afac76c4ab18f43d958df40496c8229accc09ed6a227c9bb57c37c",
    ),
    "physionet_finetuned_best.ckpt": Checkpoint(
        url="https://keeper.mpdl.mpg.de/seafhttp/f/9ebbb1d2846c48e3ba5d/?op=view",
        size=70_102_401,
        sha256="d54c722a2a58a4868a54f6b896d43b15e141e13ce9b45030183d50b83fff2513",
    ),
}

TIMEOUT = 60  # seconds
CHUNK = 1 << 20


def warn(msg: str) -> None:
    print(f"WARNING: {msg}", file=sys.stderr)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def open_url(name: str, url: str, method: str = "GET"):
    """Open the URL and validate the response; return it or None on failure."""
    request = urllib.request.Request(url, method=method, headers={"User-Agent": "checkpoint-downloader"})
    try:
        response = urllib.request.urlopen(request, timeout=TIMEOUT)
    except urllib.error.HTTPError as e:
        warn(f"{name}: link appears broken (HTTP {e.code} {e.reason}) -> {url}")
        return None
    except (urllib.error.URLError, TimeoutError) as e:
        warn(f"{name}: could not reach link ({e}) -> {url}")
        return None

    content_type = response.headers.get("Content-Type", "")
    if "text/html" in content_type:
        response.close()
        warn(f"{name}: link returns an HTML page instead of a file (expired or moved?) -> {url}")
        return None
    return response


def check_size(name: str, actual: int | None, ckpt: Checkpoint) -> bool:
    if actual is not None and actual != ckpt.size:
        warn(f"{name}: unexpected size {actual:,} bytes (expected {ckpt.size:,}); "
             f"the file behind the link may have changed -> {ckpt.url}")
        return False
    return True


def download(name: str, ckpt: Checkpoint, force: bool) -> bool:
    target = CHECKPOINT_DIR / name
    if target.exists() and not force:
        if sha256_of(target) == ckpt.sha256:
            print(f"{name}: already present and checksum OK, skipping.")
            return True
        warn(f"{name}: existing file has wrong checksum, re-downloading.")

    response = open_url(name, ckpt.url)
    if response is None:
        return False

    length = response.headers.get("Content-Length")
    if not check_size(name, int(length) if length else None, ckpt):
        response.close()
        return False

    print(f"{name}: downloading {ckpt.size / 1e6:.1f} MB ...")
    tmp = target.with_suffix(target.suffix + ".part")
    h = hashlib.sha256()
    try:
        with response, open(tmp, "wb") as f:
            while chunk := response.read(CHUNK):
                h.update(chunk)
                f.write(chunk)
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        tmp.unlink(missing_ok=True)
        warn(f"{name}: download interrupted ({e}) -> {ckpt.url}")
        return False

    if not check_size(name, tmp.stat().st_size, ckpt):
        tmp.unlink(missing_ok=True)
        return False
    if h.hexdigest() != ckpt.sha256:
        tmp.unlink(missing_ok=True)
        warn(f"{name}: checksum mismatch (got {h.hexdigest()}, expected {ckpt.sha256}); "
             f"the file behind the link has changed or is corrupted -> {ckpt.url}")
        return False

    tmp.replace(target)
    print(f"{name}: checksum OK, saved to {target}")
    return True


def check_link(name: str, ckpt: Checkpoint) -> bool:
    """Lightweight liveness check (HEAD request, size only; no checksum)."""
    response = open_url(name, ckpt.url, method="HEAD")
    if response is None:
        return False
    with response:
        length = response.headers.get("Content-Length")
    ok = check_size(name, int(length) if length else None, ckpt)
    if ok:
        print(f"{name}: link OK.")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="only verify that the links are alive, do not download")
    parser.add_argument("--force", action="store_true", help="re-download checkpoints that already exist")
    args = parser.parse_args()

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    failed = []
    for name, ckpt in CHECKPOINTS.items():
        ok = check_link(name, ckpt) if args.check else download(name, ckpt, args.force)
        if not ok:
            failed.append(name)

    if failed:
        warn(f"{len(failed)} checkpoint(s) failed: {', '.join(failed)}. "
             "Please open an issue on the repository so the link can be fixed.")
        return 1
    print("All checkpoints OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
