"""Package the current prototype source for Colab, including uncommitted fixes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from zipfile import ZIP_DEFLATED, ZipFile


def source_files(root: Path) -> list[Path]:
    files = [root / name for name in ("pyproject.toml", "README.md", "requirements.txt",
             ".env.example", "notebooks/prototype_colab.ipynb",
             "docs/guides/prototype_quickstart.md", "docs/guides/prototype_readiness_fixes.md",
             "docs/guides/prototype_recovery_demo.md")]
    for folder in ("src", "tests", "examples/prototype"):
        files.extend(p for p in (root / folder).rglob("*")
                     if p.suffix in (".py", ".json") and "__pycache__" not in p.parts)
    return sorted({p for p in files if p.is_file() and not p.is_symlink()})


def bundle(root: Path, destination: Path) -> dict:
    files = source_files(root)
    hashes = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    try:
        commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                                check=True, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                                    check=True, capture_output=True, text=True).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        commit, dirty = None, None
    manifest = {"bundle_version": 1, "commit": commit, "working_tree_dirty": dirty,
                "source_digest": digest, "files": hashes}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with ZipFile(temporary, "w", ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(root).as_posix())
        archive.writestr("prototype_source_manifest.json", json.dumps(manifest, indent=2))
    temporary.replace(destination)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("outputs/blackboard_prototype_source.zip"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    manifest = bundle(root, args.out)
    print(f"Created {args.out.resolve()} ({len(manifest['files'])} files)")
    print(f"Source digest: {manifest['source_digest']}")


if __name__ == "__main__":
    main()
