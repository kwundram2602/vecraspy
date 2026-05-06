from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _package_name(pyproject: Path) -> str:
    content = pyproject.read_text(encoding="utf-8")
    match = re.search(r'^name\s*=\s*["\']([^"\']+)["\']', content, re.MULTILINE)
    if not match:
        raise ValueError(f"Could not find package name in {pyproject}")
    return match.group(1)


def _latest_wheel(dist_dir: Path, package_name: str) -> Path:
    normalized = package_name.replace("-", "_")
    wheels = sorted(dist_dir.glob(f"{normalized}-*.whl"))
    if not wheels:
        wheels = sorted(dist_dir.glob("*.whl"))
    if not wheels:
        raise FileNotFoundError(f"No wheel found in {dist_dir}")
    return wheels[-1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build this package and ship it to a target uv project."
    )
    parser.add_argument(
        "target",
        help="Path to the target project directory (must contain pyproject.toml)",
    )
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="Skip build step and reinstall the most recent wheel already in dist/",
    )
    args = parser.parse_args()

    repo_root = _repo_root()
    pyproject = repo_root / "pyproject.toml"
    if not pyproject.exists():
        print("Error: pyproject.toml not found — run from the package root")
        return 1

    target = Path(args.target).resolve()
    if not target.exists():
        print(f"Error: target '{target}' does not exist")
        return 1
    if not (target / "pyproject.toml").exists():
        print(f"Error: '{target}' is not a Python project (no pyproject.toml)")
        return 1

    name = _package_name(pyproject)
    dist_dir = repo_root / "dist"

    if not args.no_build:
        print(f"Building {name}...")
        subprocess.run(["uv", "build", "--wheel"], check=True, cwd=repo_root)

    wheel = _latest_wheel(dist_dir, name)
    shipped_dir = target / ".shipped-packages"
    shipped_dir.mkdir(exist_ok=True)
    dest = shipped_dir / wheel.name
    shutil.copy2(wheel, dest)
    print(f"Copied  {wheel.name} → {dest.relative_to(target)}")

    print(f"Installing into {target.name}...")
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    subprocess.run(
        ["uv", "--directory", str(target), "pip", "install", "--force-reinstall", str(dest)],
        check=True,
        env=env,
    )
    print(f"Done — {name} is now available in {target.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
