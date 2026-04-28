# info.json Schema Documentation

This document describes the schema for `info.json` files, which store `StudyManifest` objects representing Fourier ptychography acquisition data.

## Overview

The `info.json` file is the manifest for a Fourier ptychography study. It contains metadata about the optical system, along with a list of captures—each capture representing a single image taken with specific illumination parameters.

**Note:** Darkfield images are represented as captures with no LED positions and are used for dark-frame subtraction during preprocessing. Multiplexed images (captures with multiple LEDs on simultaneously) and captures at different wavelengths are supported in the schema but not yet processed; loading them for reconstruction will raise an error.

## Units

All physical measurements use SI units:

| Measurement | Unit |
|-------------|------|
| Positions (x, y, z) | meters (m) |
| Wavelength | meters (m) |
| Sensor pixel size | meters (m) |
| Exposure | milliseconds (ms) |
| Numerical aperture | dimensionless |

Numeric values can be written as decimals (e.g., `0.000000625`) or in scientific notation (e.g., `6.25e-7`). Both formats are parsed correctly.

## Coordinate System

LED positions use a coordinate system where:

- **Origin (0, 0, 0)**: Center of the sample
- **X and Y axes**: The intuitive directions when looking down at the top of the LED board. If you're viewing the LED board from above, X and Y correspond to horizontal and vertical movement across the board surface.
- **Z-axis**: The distance from the LED board to the sample. Z should always be positive.

Formally, this follows the left-hand rule with the Z-axis pointing from sample toward LEDs.

## Schema

### Root Object: StudyManifest

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `study_id` | string (UUID) | Yes | Unique identifier for this study. Must be a valid UUID v4 format. |
| `created_at` | string (ISO 8601) | Yes | Timestamp when the study was created. Format: `YYYY-MM-DDTHH:MM:SS` or `YYYY-MM-DDTHH:MM:SS.mmm` |
| `magnification` | number | Yes | Objective magnification factor (e.g., `4` for 4x, `10` for 10x). |
| `numerical_aperture` | number | Yes | Estimated objective numerical aperture. This is a property of the capture hardware, not a reconstruction runtime setting. It is used to initialize the pupil radius and does not need to be absolutely precise. |
| `sensor_pixel_size` | number | Yes | Physical size of sensor pixels in **meters**. |
| `capture_dimensions` | object | Yes | Dimensions of all capture images in pixels. See `CaptureDimensions` below. |
| `captures` | array | Yes | List of illuminated `Capture` or darkfield capture objects (see below). |
| `version` | string | No | Schema version. Defaults to `"1.0"` if omitted. |
| `metadata` | object | No | Arbitrary user-defined metadata (key-value pairs). |

### Capture Object

Each capture represents a single image acquired with specific illumination.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `filename` | string | Yes | Basename of the image file (e.g., `"im_0.npy"`). Files are stored in the `captures/` subdirectory. Must be a `.npy` file (NumPy array). |
| `wavelength` | number | Illuminated captures only | Captured (ie. what the sensor measured) wavelength in **meters**. Omit this for darkfield captures. |
| `led_positions` | array | Yes | Array of `LedPosition` objects. Typically contains one element per illuminated capture. An empty array indicates a darkfield image (no illumination). |
| `captured_at` | string (ISO 8601) | No | Timestamp when the image was captured. Format: `YYYY-MM-DDTHH:MM:SS` or `YYYY-MM-DDTHH:MM:SS.mmm` |
| `exposure` | number | No | Exposure time in **milliseconds**. |

### LedPosition Object

Represents the 3D position of an LED in the illumination array.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `x` | number | Yes | X coordinate in **meters**. |
| `y` | number | Yes | Y coordinate in **meters**. |
| `z` | number | Yes | Z coordinate (distance to sample) in **meters**. Should always be positive. |

### CaptureDimensions Object

Specifies the pixel dimensions of all capture images in the study. All captures must have identical dimensions.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `width` | integer | Yes | Image width in pixels (number of columns). |
| `height` | integer | Yes | Image height in pixels (number of rows). |

## Image Files

All image files must be:

- **Format**: NumPy array files (`.npy`) containing a single-channel, 2D array
- **Location**: Stored in a `captures/` subdirectory relative to `info.json`
- **Filename convention**: The `filename` field in each capture contains only the basename (e.g., `"im_0.npy"`), not the full path. Files are always located at `captures/<filename>`
- **Shape**: All captures must have identical dimensions matching the `capture_dimensions` field. Dimensions are specified as `{width, height}` where width is columns and height is rows (note: NumPy arrays store shape as `(height, width)`). 

## Example

```json
{
    "study_id": "550e8400-e29b-41d4-a716-446655440000",
    "created_at": "2025-01-14T10:30:00.000",
    "magnification": 10,
    "numerical_aperture": 0.25,
    "sensor_pixel_size": 0.00000167,
    "capture_dimensions": {"width": 2048, "height": 2048},
    "version": "1.0",
    "metadata": {
        "camera_model": "FLIR BFS-U3-50S5C",
        "notes": "Test acquisition with USAF target"
    },
    "captures": [
        {
            "filename": "im_0_R.npy",
            "wavelength": 6.25e-7,
            "captured_at": "2025-01-14T10:30:01.042",
            "exposure": 10,
            "led_positions": [
                {
                    "x": 0,
                    "y": 0,
                    "z": 0.07
                }
            ]
        },
        {
            "filename": "im_0_G.npy",
            "wavelength": 5.3e-7,
            "captured_at": "2025-01-14T10:30:01.103",
            "exposure": 8,
            "led_positions": [
                {
                    "x": 0.0001,
                    "y": 0,
                    "z": 0.07
                }
            ]
        },
        {
            "filename": "im_0_B.npy",
            "wavelength": 4.7e-7,
            "captured_at": "2025-01-14T10:30:01.178",
            "exposure": 12,
            "led_positions": [
                {
                    "x": -0.0001,
                    "y": 0,
                    "z": 0.07
                }
            ]
        },
        {
            "filename": "im_1_R.npy",
            "wavelength": 6.25e-7,
            "captured_at": "2025-01-14T10:30:01.256",
            "exposure": 10,
            "led_positions": [
                {
                    "x": 0.004,
                    "y": 0,
                    "z": 0.07
                }
            ]
        },
        {
            "filename": "im_1_G.npy",
            "wavelength": 5.3e-7,
            "captured_at": "2025-01-14T10:30:01.317",
            "exposure": 8,
            "led_positions": [
                {
                    "x": 0.0041,
                    "y": 0,
                    "z": 0.07
                }
            ]
        },
        {
            "filename": "im_1_B.npy",
            "wavelength": 4.7e-7,
            "captured_at": "2025-01-14T10:30:01.392",
            "exposure": 12,
            "led_positions": [
                {
                    "x": 0.0039,
                    "y": 0,
                    "z": 0.07
                }
            ]
        },
        {
            "filename": "darkfield_0.npy",
            "captured_at": "2025-01-14T10:30:01.503",
            "exposure": 50,
            "led_positions": []
        }
    ]
}
```

## Validation Notes

When creating `info.json` files programmatically:

1. **study_id**: Must be a valid UUID string (e.g., generated via `uuid.uuid4()`)
2. **created_at**: Must be ISO 8601 format without timezone (e.g., `2025-01-14T10:30:00` or `2025-01-14T10:30:00.123` for millisecond precision)
3. **captures**: Must contain at least one capture
4. **led_positions**: Must be an array. Use an empty array `[]` for darkfield images (no illumination). For standard captures, typically contains one LED position
5. **wavelength**: Common values are approximately `4.7e-7` (blue), `5.3e-7` (green), `6.25e-7` (red)
6. **numerical_aperture**: Estimated objective NA for the capture hardware, for example `0.13` or `0.25`. This initializes the pupil radius; it does not need to be absolutely precise.
