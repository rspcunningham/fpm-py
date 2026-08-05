# `info.json` schema

`info.json` describes one single-wavelength Fourier ptychography acquisition made
from raw Bayer captures. Unknown fields are rejected everywhere except inside
`metadata`.

All positions, wavelengths, and sensor pixel sizes use meters. Exposure uses
seconds.

## Root object

| Field | Type | Required | Contract |
|---|---|---:|---|
| `study_id` | UUID string | Yes | Any valid UUID. |
| `created_at` | ISO 8601 string | Yes | Study acquisition timestamp. |
| `magnification` | number | Yes | Positive and finite. |
| `numerical_aperture` | number | Yes | Positive and finite. |
| `sensor_pixel_size` | number | Yes | Positive and finite, in meters. |
| `bayer_format` | string | Yes | `RGGB`, `GRBG`, `GBRG`, or `BGGR`. |
| `capture_dimensions` | object | Yes | Positive integer `width` and `height`. |
| `captures` | array | Yes | At least one illuminated capture. |
| `metadata` | object | No | Opaque JSON metadata, ignored by `ptych`. |

`bayer_format` is the row-major 2×2 Bayer tile at the top-left of the complete
stored image. Cropping adjusts the effective tile automatically.

The contents of `metadata` are outside this schema, including its unit
semantics.

## Illuminated capture

An illuminated capture has exactly one LED position.

| Field | Type | Required | Contract |
|---|---|---:|---|
| `filename` | string | Yes | Unique flat `.npy` basename. |
| `wavelength` | number | Yes | Positive finite illumination wavelength, in meters. |
| `channel` | string | Yes | Exactly `R`, `G`, or `B`. |
| `exposure` | number | Yes | Positive finite exposure, in seconds. |
| `led_positions` | array | Yes | Exactly one LED position. |
| `captured_at` | ISO 8601 string | No | Capture timestamp. |

`channel` is authoritative: it selects the demosaiced sensor channel and is not
inferred from `wavelength`. All illuminated captures in a dataset must use the
same wavelength.

## Dark capture

A dark capture has no LED positions and no `wavelength`.

| Field | Type | Required | Contract |
|---|---|---:|---|
| `filename` | string | Yes | Unique flat `.npy` basename. |
| `channel` | string | Yes | Exactly `R`, `G`, or `B`. |
| `exposure` | number | Yes | Positive finite exposure, in seconds. |
| `led_positions` | array | Yes | Must be empty. |
| `captured_at` | ISO 8601 string | No | Capture timestamp. |

Dark captures are optional. If any are present, their distinct
`(channel, exposure)` pairs must exactly match the pairs used by illuminated
captures. Multiple dark captures for one pair are averaged.

## LED position

| Field | Type | Required | Contract |
|---|---|---:|---|
| `x` | number | Yes | Finite, in meters. |
| `y` | number | Yes | Finite, in meters. |
| `z` | number | Yes | Positive and finite, in meters. |

Coordinates describe the LED position relative to the sample center. Positive
`z` points from the sample toward the LED array.

## Capture files

The dataset layout is:

```text
dataset/
├── info.json
└── captures/
    └── *.npy
```

The contents of `captures/` must exactly match the filenames in `captures`.
Every file must contain one real numeric 2-D NumPy array with shape
`(height, width)`. Raw intensities must be finite and non-negative.

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
      "led_positions": [
        {"x": 0.0, "y": 0.0, "z": 0.0605}
      ]
    },
    {
      "filename": "dark_000.npy",
      "channel": "G",
      "exposure": 0.005198,
      "led_positions": []
    }
  ],
  "metadata": {
    "sample": "malaria"
  }
}
```
