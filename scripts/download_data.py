"""Download the public-domain Tiny Shakespeare corpus with a SHA-256 check."""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"
DEST = Path(__file__).resolve().parents[1] / "data" / "tinyshakespeare.txt"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.exists() and sha256(DEST) == SHA256:
        print(f"already present and verified: {DEST}")
        return 0
    print(f"downloading {URL}")
    req = urllib.request.Request(URL, headers={"User-Agent": "blockbus-lm/0.1"})
    with urllib.request.urlopen(req, timeout=60) as response:
        payload = response.read()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != SHA256:
        raise SystemExit(f"sha256 mismatch: expected {SHA256}, got {digest}")
    DEST.write_bytes(payload)
    print(f"wrote {DEST} ({len(payload)} bytes), sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
