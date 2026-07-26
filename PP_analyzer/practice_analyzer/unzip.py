#!/usr/bin/env python3
"""Unzip all extension archives in the expected folder layout."""

import os
import sys
import zipfile
from pathlib import Path

def main() -> int:
    script_dir = Path(__file__).resolve().parent
    
    # Default values: resolve "../raw_data" relative to the script's directory
    root = str(script_dir.parent / "raw_data")
    folder = "current"

    # Use command line arguments if specified
    if len(sys.argv) >= 3:
        root = sys.argv[1]
        folder = sys.argv[2]
    elif len(sys.argv) == 2:
        # If only one argument is provided, treat it as the subfolder
        folder = sys.argv[1]

    newpath = os.path.join(root, "unzip", folder)
    os.makedirs(newpath, exist_ok=True)

    path = os.path.join(root, "zip", folder)
    if not os.path.exists(path):
        print(f"Error: Source directory {path} does not exist.")
        return 1
        
    files = sorted(os.listdir(path))
    print(f"start to unzip folder {path}")

    for filename in files:
        if not filename.endswith(('.crx', '.zip', '.xpi')):
            continue
        archive_path = os.path.join(path, filename)
        output_dir = os.path.join(newpath, os.path.splitext(filename)[0])
        print(f"Extracting {filename} to {output_dir}...")
        try:
            import stat
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                for member in zip_ref.infolist():
                    # Resolve destination path for the member
                    target_path = os.path.join(output_dir, member.filename)
                    if os.path.exists(target_path):
                        # If a read-only file exists, make it writeable
                        if os.path.isfile(target_path):
                            mode = os.stat(target_path).st_mode
                            if not (mode & stat.S_IWRITE):
                                os.chmod(target_path, stat.S_IWRITE)
                    
                    zip_ref.extract(member, output_dir)
        except Exception as e:
            print(f"Error extracting {filename}: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
