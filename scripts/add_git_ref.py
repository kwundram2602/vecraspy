from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def _remote_url() -> str:
    result = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        capture_output=True,
        text=True,
        check=False,
        cwd=Path(__file__).resolve().parents[1],
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Could not detect git remote URL. Pass --url explicitly."
        )
    return result.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Add this package as a git dependency to a target uv project."
    )
    parser.add_argument(
        "target",
        help="Path to the target project directory (must contain pyproject.toml)",
    )
    parser.add_argument(
        "--url",
        default=None,
        help="Git URL to add (default: origin remote of this repo)",
    )
    parser.add_argument(
        "--rev",
        default=None,
        help="Commit hash, branch, or tag to pin (e.g. main, v1.2.3, abc1234)",
    )
    args = parser.parse_args()

    target = Path(args.target).resolve()
    if not (target / "pyproject.toml").exists():
        print(f"Error: '{target}' is not a Python project (no pyproject.toml)")
        return 1

    url = args.url or _remote_url()
    ref = f"@{args.rev}" if args.rev else ""
    spec = f"git+{url}{ref}"

    print(f"Adding {spec} to {target.name}...")
    subprocess.run(
        ["uv", "--project", str(target), "add", spec],
        check=True,
    )
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
