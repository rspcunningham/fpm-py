"""Utility to convert .npy capture files to PNGs for visualization."""
import numpy as np
from PIL import Image
from pathlib import Path


def save_captures_as_png(captures_dir: str | Path, output_dir: str | Path | None = None) -> None:
    """
    Read all .npy images in a directory and save them as PNGs.

    Args:
        captures_dir: Directory containing .npy files
        output_dir: Directory to save PNGs (defaults to captures_dir)
    """
    captures_dir = Path(captures_dir)
    output_dir = Path(output_dir) if output_dir else captures_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    npy_files = sorted(captures_dir.glob("*.npy"))

    for npy_path in npy_files:
        img_array = np.load(npy_path)

        # Normalize to 0-255
        img_min = img_array.min()
        img_max = img_array.max()
        if img_max > img_min:
            img_normalized = (img_array - img_min) / (img_max - img_min) * 255
        else:
            img_normalized = np.zeros_like(img_array)

        img_u8 = img_normalized.astype(np.uint8)

        png_name = npy_path.stem + ".png"
        Image.fromarray(img_u8).save(output_dir / png_name)
        print(f"Saved {png_name}")


if __name__ == "__main__":
    save_captures_as_png("demo/captures")
