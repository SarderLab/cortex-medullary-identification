"""Print the groups, datasets, and attributes in a TRIDENT feature H5 file."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def _format_attribute(value: object) -> str:
    if isinstance(value, bytes):
        return repr(value.decode("utf-8", errors="replace"))
    if isinstance(value, np.ndarray):
        if value.size > 10:
            return f"array(shape={value.shape}, dtype={value.dtype})"
        return np.array2string(value)
    if isinstance(value, np.generic):
        return repr(value.item())
    return repr(value)


def _print_attributes(obj: h5py.Group | h5py.Dataset, indent: str) -> None:
    for key, value in obj.attrs.items():
        print(f"{indent}@{key}: {_format_attribute(value)}")


def inspect_h5(path: Path) -> None:
    path = path.expanduser().resolve()
    with h5py.File(path, "r") as handle:
        print(f"File: {path}")
        print("/ (group)")
        _print_attributes(handle, "  ")

        def describe(name: str, obj: h5py.Group | h5py.Dataset) -> None:
            indent = "  " * (name.count("/") + 1)
            if isinstance(obj, h5py.Dataset):
                print(f"{indent}{name.rsplit('/', 1)[-1]} (dataset: shape={obj.shape}, "
                      f"dtype={obj.dtype}, compression={obj.compression or 'none'})")
            else:
                print(f"{indent}{name.rsplit('/', 1)[-1]}/ (group)")
            _print_attributes(obj, indent + "  ")

        handle.visititems(describe)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="TRIDENT feature H5 file to inspect")
    args = parser.parse_args()
    inspect_h5(args.path)


if __name__ == "__main__":
    main()
