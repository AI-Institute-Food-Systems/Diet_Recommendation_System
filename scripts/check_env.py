#!/usr/bin/env python3
"""Compare installed package versions with the pins in requirements.txt.

Exits non-zero if a required package is missing. Version differences only warn:
the pipeline runs on other versions, but clustering labels and model weights
can change.
"""
from __future__ import annotations

import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

REQUIREMENTS = Path(__file__).resolve().parent.parent / "requirements.txt"


def pinned_requirements() -> dict[str, str]:
    pins = {}
    for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if "==" in line:
            name, ver = line.split("==", 1)
            pins[name.strip()] = ver.strip()
    return pins


def main() -> int:
    print(f"python {sys.version.split()[0]} ({sys.executable})")
    if sys.version_info[:2] != (3, 11):
        print("  warning: the pinned versions are for Python 3.11")
    missing, different = [], []
    for name, pinned in pinned_requirements().items():
        try:
            installed = version(name)
        except PackageNotFoundError:
            missing.append(name)
            print(f"  MISSING  {name} (pinned {pinned})")
            continue
        if installed != pinned:
            different.append(name)
            print(f"  differs  {name} {installed} (pinned {pinned})")
    if different:
        print(f"{len(different)} package(s) differ from requirements.txt; results may differ from those of the pinned environment.")
    if missing:
        print(f"{len(missing)} required package(s) missing: pip install -r requirements.txt")
        return 1
    print("environment OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
