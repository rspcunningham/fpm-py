import torch
import torch.nn.functional as F
from jaxtyping import Float


def _make_masks(
    pattern: str, height: int, width: int, device: torch.device
) -> dict[str, torch.Tensor]:
    """
    Build boolean masks for R, G, B pixel positions from a Bayer pattern string.

    The pattern string (e.g. "RGGB") maps to a 2x2 tile:
        pattern[0] pattern[1]
        pattern[2] pattern[3]
    """
    tile: dict[str, list[tuple[int, int]]] = {c: [] for c in "RGB"}
    for idx, c in enumerate(pattern):
        tile[c].append((idx // 2, idx % 2))  # (row, col) within the 2x2 tile

    y = torch.arange(height, device=device).view(-1, 1)
    x = torch.arange(width, device=device).view(1, -1)

    masks: dict[str, torch.Tensor] = {}
    for c in "RGB":
        mask: torch.Tensor = torch.zeros(height, width, dtype=torch.bool, device=device)
        for row, col in tile[c]:
            mask |= (y % 2 == row) & (x % 2 == col)
        masks[c] = mask

    return masks


def _avg_conv(
    data: Float[torch.Tensor, "image height width"], kernel: torch.Tensor
) -> Float[torch.Tensor, "image height width"]:
    """Apply a 3x3 averaging kernel with replicate padding."""
    x = data.unsqueeze(1)
    x = F.pad(x, (1, 1, 1, 1), mode="replicate")
    x = F.conv2d(x, kernel.view(1, 1, 3, 3).to(x))
    return x.squeeze(1)


def demosaic(
    data: Float[torch.Tensor, "image height width"],
    pattern: str = "RGGB",
) -> Float[torch.Tensor, "image channel height width"]:
    """
    Bilinear Bayer demosaicing.

    Args:
        data:    [image, height, width] single-channel Bayer mosaic
        pattern: one of "RGGB", "GRBG", "GBRG", "BGGR"

    Returns:
        [image, channel, height, width] full-color image (channels in RGB order)
    """
    assert pattern in ("RGGB", "GRBG", "GBRG", "BGGR"), f"Unknown pattern: {pattern}"
    _, height, width = data.shape
    masks = _make_masks(pattern, height, width, data.device)

    # --- Interpolation kernels (unnormalized, we divide by count) ---

    # Cardinal neighbors (used for green, and for R/B at green sites)
    #   . 1 .
    #   1 . 1
    #   . 1 .
    cross = torch.tensor(
        [
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ],
        dtype=data.dtype,
    )

    # Diagonal neighbors (used for R at B sites and B at R sites)
    #   1 . 1
    #   . . .
    #   1 . 1
    diag = torch.tensor(
        [
            [1, 0, 1],
            [0, 0, 0],
            [1, 0, 1],
        ],
        dtype=data.dtype,
    )

    # Horizontal neighbors (used for R/B at green sites in the same row)
    #   . . .
    #   1 . 1
    #   . . .
    horiz = torch.tensor(
        [
            [0, 0, 0],
            [1, 0, 1],
            [0, 0, 0],
        ],
        dtype=data.dtype,
    )

    # Vertical neighbors (used for R/B at green sites in the same column)
    #   . 1 .
    #   . . .
    #   . 1 .
    vert = torch.tensor(
        [
            [0, 1, 0],
            [0, 0, 0],
            [0, 1, 0],
        ],
        dtype=data.dtype,
    )

    # Precompute filtered images
    cross_avg = _avg_conv(data, cross) / 4
    diag_avg = _avg_conv(data, diag) / 4
    horiz_avg = _avg_conv(data, horiz) / 2
    vert_avg = _avg_conv(data, vert) / 2

    # --- Assemble each channel ---
    out = torch.empty(
        data.shape[0], 3, height, width, dtype=data.dtype, device=data.device
    )

    for i, color in enumerate("RGB"):
        ch = data.clone()

        if color == "G":
            # Green is known at G sites; interpolate at R and B sites via cardinal avg
            rb_mask = masks["R"] | masks["B"]
            ch[rb_mask.expand_as(ch)] = cross_avg[rb_mask.expand_as(ch)]
        else:
            # For R (or B): known at own sites. Need to fill in two other site types.
            opposite = "B" if color == "R" else "R"

            # At opposite-color sites: use diagonal average
            opp_mask = masks[opposite]
            ch[opp_mask.expand_as(ch)] = diag_avg[opp_mask.expand_as(ch)]

            # At green sites: use horiz or vert depending on whether this green
            # shares a row or column with the target color.
            # The color appears in specific rows of the tile. Green sites in those
            # rows get horizontal interpolation; green sites in other rows get vertical.
            color_rows: set[int] = {r for r, _c in _tile_positions(pattern, color)}
            g_mask: torch.Tensor = masks["G"]

            y = torch.arange(height, device=data.device).view(-1, 1)
            same_row = torch.zeros(height, width, dtype=torch.bool, device=data.device)
            for r in color_rows:
                same_row |= y % 2 == r

            g_same_row = g_mask & same_row
            g_diff_row = g_mask & ~same_row

            ch[g_same_row.expand_as(ch)] = horiz_avg[g_same_row.expand_as(ch)]
            ch[g_diff_row.expand_as(ch)] = vert_avg[g_diff_row.expand_as(ch)]

        out[:, i] = ch

    return out


def _tile_positions(pattern: str, color: str) -> list[tuple[int, int]]:
    """Return (row, col) positions of a color within the 2x2 Bayer tile."""
    return [(i // 2, i % 2) for i, c in enumerate(pattern) if c == color]
