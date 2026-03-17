"""Apply Bayer interpolation to a directory of .npy captures.

At pixels where (x + y) is even, replaces the value with the average of
cardinal neighbors (up/down/left/right). Odd pixels are kept as-is.

Usage:
    python -m ptych.tools.bayer_interpolate /tmp/usaf/captures_real -o /tmp/usaf/captures_real_bayer
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def bayer_interpolate(arr: np.ndarray) -> np.ndarray:
    """Apply Bayer interpolation to a 2D array.

    Pixels at even (x+y) positions are replaced by the mean of their
    cardinal neighbors. Edge pixels use available neighbors only.
    """
    h, w = arr.shape
    out = arr.copy()

    # Build mask of even (x+y) positions
    yy, xx = np.mgrid[:h, :w]
    mask = (xx + yy) % 2 == 0

    # Sum cardinal neighbors with boundary handling
    accum = np.zeros_like(arr, dtype=np.float64)
    count = np.zeros((h, w), dtype=np.float64)

    # Up
    accum[1:, :] += arr[:-1, :]
    count[1:, :] += 1
    # Down
    accum[:-1, :] += arr[1:, :]
    count[:-1, :] += 1
    # Left
    accum[:, 1:] += arr[:, :-1]
    count[:, 1:] += 1
    # Right
    accum[:, :-1] += arr[:, 1:]
    count[:, :-1] += 1

    avg = accum / np.maximum(count, 1)
    out[mask] = avg[mask]
    return out.astype(arr.dtype)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply Bayer interpolation to a directory of .npy captures."
    )
    parser.add_argument("directory", type=Path, help="Input directory of .npy files")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Output directory (default: <input>_bayer)")
    args = parser.parse_args()

    if not args.directory.is_dir():
        print(f"Not a directory: {args.directory}", file=sys.stderr)
        sys.exit(1)

    output = args.output or args.directory.parent / (args.directory.name + "_bayer")
    output.mkdir(parents=True, exist_ok=True)

    files = sorted(args.directory.glob("*.npy"))
    if not files:
        print(f"No .npy files found in {args.directory}", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(files)} files → {output}")
    for f in files:
        arr = np.load(f)
        if arr.ndim != 2:
            print(f"  Skipping {f.name} (ndim={arr.ndim}, expected 2)")
            continue
        out = bayer_interpolate(arr)
        np.save(output / f.name, out)

    print("Done.")


if __name__ == "__main__":
    main()
