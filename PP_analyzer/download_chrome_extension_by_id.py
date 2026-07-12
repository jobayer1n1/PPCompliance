#!/usr/bin/env python3
"""Download Chrome extensions by extension ID into raw_data/current.

This uses the Chrome Web Store update service URL documented by Google:
https://clients2.google.com/service/update2/crx
"""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "raw_data" / "zip" /"current"


def build_crx_url(extension_id: str, prodversion: str) -> str:
    x_value = f"id={extension_id}&installsource=ondemand&uc"
    return (
        "https://clients2.google.com/service/update2/crx"
        f"?response=redirect&prodversion={prodversion}&x={quote(x_value)}"
        "&acceptformat=crx3"
    )


def download_extension(extension_id: str, prodversion: str) -> bool:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{extension_id}.crx"

    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"Skipping {extension_id}: already exists at {output_path}")
        return False

    url = build_crx_url(extension_id, prodversion)
    request = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/126.0.0.0 Safari/537.36"
            )
        },
    )

    try:
        with urlopen(request, timeout=60) as response:
            data = response.read()
    except HTTPError as exc:
        print(f"Failed to download {extension_id}: HTTP {exc.code}")
        return False
    except URLError as exc:
        print(f"Failed to download {extension_id}: {exc.reason}")
        return False

    if not data:
        print(f"Failed to download {extension_id}: empty response")
        return False

    output_path.write_bytes(data)
    print(f"Downloaded {extension_id} -> {output_path}")
    return True


def load_ids_from_file(path: Path) -> list[str]:
    ids: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            ids.append(line)
    return ids


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download Chrome extension CRX files by extension ID."
    )
    parser.add_argument(
        "extension_ids",
        nargs="*",
        help="One or more Chrome extension IDs.",
    )
    parser.add_argument(
        "--ids-file",
        type=Path,
        help="Optional text file containing one extension ID per line.",
    )
    parser.add_argument(
        "--request-version",
        default="126.0.0.0",
        help="Chrome version string used by the update service request.",
    )
    args = parser.parse_args()

    extension_ids = list(args.extension_ids)
    if args.ids_file:
        extension_ids.extend(load_ids_from_file(args.ids_file))

    # Preserve order while removing duplicates.
    seen = set()
    unique_ids = []
    for extension_id in extension_ids:
        extension_id = extension_id.strip()
        if extension_id and extension_id not in seen:
            seen.add(extension_id)
            unique_ids.append(extension_id)

    if not unique_ids:
        parser.error("provide at least one extension ID or --ids-file")

    for extension_id in unique_ids:
        download_extension(extension_id, args.request_version)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
