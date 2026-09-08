"""Download the configured FRED series into data/macro/raw."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen


def download(manifest_path: Path, output_dir: Path, timeout: int = 30) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, object] = {"downloaded_at": datetime.now(timezone.utc).isoformat(), "source": manifest["source"], "series": {}}
    for series_id in manifest["series"]:
        url = manifest["download_endpoint"].format(series_id=series_id)
        destination = output_dir / f"{series_id}.csv"
        request = Request(url, headers={"User-Agent": "btc-perp-research/0.1"})
        with urlopen(request, timeout=timeout) as response:
            payload = response.read()
        destination.write_bytes(payload)
        results["series"][series_id] = {"url": url, "path": str(destination), "bytes": len(payload)}
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/macro/metadata/series_manifest.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/macro/raw"))
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()
    print(json.dumps(download(args.manifest, args.output_dir, args.timeout), indent=2))


if __name__ == "__main__":
    main()
