# `info.json` schema

`info.json` describes one single-wavelength Fourier ptychography acquisition made
from raw Bayer captures. Unknown JSON object members are rejected everywhere
except inside `metadata`.

All physical LED coordinates, wavelengths, and sensor pixel sizes use meters.
Exposures use seconds. Magnification and numerical aperture are dimensionless;
capture dimensions are pixel counts.

## Root object

| Field | Type | Required | Contract |
|---|---|---:|---|
| `study_id` | UUID string | Yes | Any valid UUID. |
| `created_at` | ISO 8601 string | Yes | Study creation date/time. |
| `magnification` | number | Yes | Positive and finite. |
| `numerical_aperture` | number | Yes | Positive and finite. |
| `sensor_pixel_size` | number | Yes | Positive and finite, in meters. |
| `bayer_format` | string | Yes | `RGGB`, `GRBG`, `GBRG`, or `BGGR`. |
| `capture_dimensions` | object | Yes | Positive integer `width` and `height`. |
| `captures` | array | Yes | At least one illuminated capture. |
| `metadata` | object | No | Opaque JSON metadata, retained but not interpreted. |

`bayer_format` is the row-major 2×2 Bayer tile at the top-left of the complete
stored image. When `PtychStudy.load` loads a crop, it adjusts the effective tile
for the crop offset.

The contents of `metadata` are outside this schema, including its unit
semantics.

## Illuminated capture

An illuminated capture has exactly one LED position.

| Field | Type | Required | Contract |
|---|---|---:|---|
| `filename` | string | Yes | Unique flat basename ending in lowercase `.npy`. |
| `wavelength` | number | Yes | Positive finite illumination wavelength, in meters. |
| `channel` | string | Yes | Exactly `R`, `G`, or `B`. |
| `exposure` | number | Yes | Positive finite exposure, in seconds. |
| `led_positions` | array | Yes | Exactly one LED position. |
| `captured_at` | ISO 8601 string | Yes | Capture date/time. |

`channel` is authoritative: it selects the demosaiced sensor channel and is not
inferred from `wavelength`. All illuminated captures in a dataset must use the
same wavelength.

## Dark capture

A dark capture has no LED positions and no `wavelength`.

| Field | Type | Required | Contract |
|---|---|---:|---|
| `filename` | string | Yes | Unique flat basename ending in lowercase `.npy`. |
| `channel` | string | Yes | Exactly `R`, `G`, or `B`. |
| `exposure` | number | Yes | Positive finite exposure, in seconds. |
| `led_positions` | array | Yes | Must be empty. |
| `captured_at` | ISO 8601 string | Yes | Capture date/time. |

Dark captures are optional. If any are present, their distinct
`(channel, exposure)` pairs must exactly match the pairs used by illuminated
captures. Each illuminated capture is corrected using only the dark captures
with its own `(channel, exposure)` pair. How multiple darks for one pair are
combined is selected at load time: `average_all` subtracts their mean, and
`nearest_only` subtracts the single dark whose `captured_at` is closest to the
illuminated capture's `captured_at`, choosing the earlier one on a tie.

## LED position

| Field | Type | Required | Contract |
|---|---|---:|---|
| `x` | number | Yes | Finite, in meters. |
| `y` | number | Yes | Finite, in meters. |
| `z` | number | Yes | Positive and finite, in meters. |

Coordinates describe the LED position relative to the sample center. Positive
`z` points from the sample toward the LED array.

## Capture files

The required dataset layout is:

```text
dataset/
├── info.json
└── captures/
    └── *.npy
```

The entries in `captures/` must exactly match the filenames listed in the
manifest's `captures` array. Every entry must be a file containing one real
numeric 2-D NumPy array with shape `(height, width)`. Raw intensities must be
finite and non-negative.

## Example

```json
{
  "study_id": "550e8400-e29b-41d4-a716-446655440000",
  "created_at": "2025-01-14T10:30:00",
  "magnification": 1.7,
  "numerical_aperture": 0.18,
  "sensor_pixel_size": 1.12e-6,
  "bayer_format": "BGGR",
  "capture_dimensions": {
    "width": 3280,
    "height": 2464
  },
  "captures": [
    {
      "filename": "im_000.npy",
      "wavelength": 5.25e-7,
      "channel": "G",
      "exposure": 0.005198,
      "captured_at": "2025-01-14T10:30:01",
      "led_positions": [
        {"x": 0.0, "y": 0.0, "z": 0.0605}
      ]
    },
    {
      "filename": "dark_000.npy",
      "channel": "G",
      "exposure": 0.005198,
      "captured_at": "2025-01-14T10:30:02",
      "led_positions": []
    }
  ],
  "metadata": {
    "sample": "malaria"
  }
}
```
