#!/usr/bin/env python3
"""Unzip all extension archives in the expected folder layout."""

import os
import sys
import zipfile
from pathlib import Path

ARCHIVE_EXTENSIONS = ('.crx', '.zip', '.xpi')


def resolve_path(base_dir: Path, path_arg: str) -> Path:
    path = Path(path_arg)
    if path.is_absolute():
        return path
    return base_dir / path


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent

    if len(sys.argv) > 3:
        print("Usage: python unzip.py [from] [to]")
        return 1

    from_arg = sys.argv[1] if len(sys.argv) >= 2 else os.path.join("raw_data", "zip", "current")
    to_arg = sys.argv[2] if len(sys.argv) >= 3 else os.path.join("raw_data", "unzip", "current")

    source_dir = resolve_path(base_dir, from_arg)
    output_root = resolve_path(base_dir, to_arg)
    output_root.mkdir(parents=True, exist_ok=True)

    if not source_dir.exists():
        print(f"Error: Source directory {source_dir} does not exist.")
        return 1
        
    files = sorted(source_dir.iterdir())
    print(f"start to unzip folder {source_dir}")

    for archive_path in files:
        if not archive_path.is_file() or archive_path.suffix.lower() not in ARCHIVE_EXTENSIONS:
            continue
        output_dir = output_root / archive_path.stem
        print(f"Extracting {archive_path.name} to {output_dir}...")
        try:
            import stat
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                for member in zip_ref.infolist():
                    # Resolve destination path for the member
                    target_path = output_dir / member.filename
                    if target_path.exists():
                        # If a read-only file exists, make it writeable
                        if target_path.is_file():
                            mode = target_path.stat().st_mode
                            if not (mode & stat.S_IWRITE):
                                target_path.chmod(stat.S_IWRITE)
                    
                    zip_ref.extract(member, output_dir)
        except Exception as e:
            print(f"Error extracting {archive_path.name}: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
