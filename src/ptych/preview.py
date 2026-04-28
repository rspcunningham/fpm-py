from __future__ import annotations

import base64
import json
import tempfile
import webbrowser
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, assert_never, cast

import numpy as np
import numpy.typing as npt
import torch
from jaxtyping import Float

from ptych.data.region import CaptureRegion
from ptych.data.study import PtychStudy
from ptych.data.types import Capture, is_illuminated_capture

type ObjectPreviewMode = Literal["intensity", "amplitude", "phase"]

__all__ = [
    "prepare_study_capture_rgb",
    "render_object_preview_png",
    "render_study_capture_preview_png",
    "show_object_preview",
    "show_study_capture",
]


def prepare_study_capture_rgb(
    study: PtychStudy,
    capture_index: int,
    capture_region: CaptureRegion | None = None,
) -> Float[torch.Tensor, "3 H W"]:
    if capture_index < 0 or capture_index >= study.captures.shape[0]:
        raise IndexError(
            f"capture_index {capture_index} out of range for {study.captures.shape[0]} captures"
        )

    capture = study.captures[capture_index]
    rgb = capture.unsqueeze(0).repeat(3, 1, 1)

    if capture_region is not None:
        rgb = _crop_capture_channels(rgb, capture_region)

    return rgb


def _crop_capture_channels(
    capture: torch.Tensor,
    region: CaptureRegion,
) -> torch.Tensor:
    height = capture.shape[-2]
    width = capture.shape[-1]
    region.validate(width=width, height=height)

    return capture[..., region.y_top:region.y_bottom, region.x_left:region.x_right]


def _valid_study_captures(study: PtychStudy) -> list[Capture]:
    valid_captures = [
        capture
        for capture in study.manifest.captures
        if is_illuminated_capture(capture)
    ]
    if len(valid_captures) != study.captures.shape[0]:
        raise ValueError(
            "Study manifest captures do not align with loaded study tensors. "
            + "Expected one loaded tensor per valid manifest capture."
        )
    return valid_captures


def _capture_preview_title(
    study: PtychStudy,
    capture_index: int,
    capture_meta: Capture,
) -> str:
    wavelength_nm = capture_meta.wavelength * 1e9
    return (
        f"{capture_meta.filename} | "
        f"lambda={wavelength_nm:.0f} nm | "
        f"k=({float(study.kx_batch[capture_index]):.4f}, {float(study.ky_batch[capture_index]):.4f})"
    )

def _prepare_study_capture_channel(
    study: PtychStudy,
    capture_index: int,
    capture_region: CaptureRegion | None = None,
) -> Float[torch.Tensor, "H W"]:
    capture = study.captures[capture_index]
    if capture_region is not None:
        return _crop_capture_channels(capture, capture_region)
    return capture


def _prepare_study_capture_scalar(
    study: PtychStudy,
    capture_index: int,
    capture_region: CaptureRegion | None = None,
) -> Float[torch.Tensor, "H W"]:
    return _prepare_study_capture_channel(
        study,
        capture_index,
        capture_region=capture_region,
    )


def _prepare_object_preview_scalar(
    object_source: torch.Tensor | str | Path,
    *,
    mode: ObjectPreviewMode = "intensity",
) -> Float[torch.Tensor, "H W"]:
    if isinstance(object_source, (str, Path)):
        array = cast(npt.NDArray[np.generic], np.load(Path(object_source)))
        tensor = torch.from_numpy(np.asarray(array))
    else:
        tensor = object_source.detach().cpu()

    if tensor.ndim != 2:
        raise ValueError(f"Object preview expects a 2D tensor/array, got shape {tuple(tensor.shape)}")

    if mode == "intensity":
        return torch.abs(tensor).to(torch.float32).square()
    if mode == "amplitude":
        return torch.abs(tensor).to(torch.float32)
    if mode == "phase":
        return torch.angle(tensor).to(torch.float32)
    assert_never(mode)


def _suggest_level_window(
    image: Float[torch.Tensor, "H W"],
    *,
    mode: ObjectPreviewMode | Literal["scalar"] = "scalar",
) -> tuple[float, float, float, float]:
    data = image.detach().cpu().to(torch.float32)
    finite = data[torch.isfinite(data)]
    if finite.numel() == 0:
        return 0.5, 1.0, 0.0, 1.0

    data_min = float(torch.min(finite).item())
    data_max = float(torch.max(finite).item())

    if mode == "phase":
        return 0.0, float(2 * np.pi), data_min, data_max

    lo = float(torch.quantile(finite, 0.005).item())
    hi = float(torch.quantile(finite, 0.995).item())
    if hi <= lo:
        lo = data_min
        hi = data_max

    if hi <= lo:
        center = data_min
        width = 1.0
    else:
        center = (lo + hi) / 2.0
        width = hi - lo

    return center, width, data_min, data_max


def _resolve_level_window(
    image: Float[torch.Tensor, "H W"],
    *,
    level: float | None,
    window: float | None,
    mode: ObjectPreviewMode | Literal["scalar"] = "scalar",
) -> tuple[float, float, float, float]:
    default_level, default_window, data_min, data_max = _suggest_level_window(image, mode=mode)

    if level is None and window is None:
        return default_level, default_window, data_min, data_max
    if level is None or window is None:
        raise ValueError("level and window must be provided together")
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")

    return float(level), float(window), data_min, data_max


def render_scalar_preview_image(
    image: Float[torch.Tensor, "H W"],
    *,
    level: float | None = None,
    window: float | None = None,
    mode: ObjectPreviewMode | Literal["scalar"] = "scalar",
) -> np.ndarray:
    resolved_level, resolved_window, _, _ = _resolve_level_window(
        image,
        level=level,
        window=window,
        mode=mode,
    )
    lower = resolved_level - resolved_window / 2.0
    upper = resolved_level + resolved_window / 2.0

    array = image.detach().cpu().to(torch.float32).numpy()
    clipped = np.clip(array, lower, upper)
    denom = upper - lower
    if denom <= 0.0:
        normalized = np.zeros_like(clipped, dtype=np.float32)
    else:
        normalized = (clipped - lower) / denom

    image_u8 = np.asarray(np.rint(normalized * 255.0), dtype=np.uint8)
    return image_u8


def _default_temp_path(suffix: str) -> Path:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        return Path(handle.name)


def render_scalar_preview_png(
    image: Float[torch.Tensor, "H W"],
    *,
    level: float | None = None,
    window: float | None = None,
    mode: ObjectPreviewMode | Literal["scalar"] = "scalar",
    path: str | Path | None = None,
) -> Path:
    from PIL import Image

    output_path = Path(path) if path is not None else _default_temp_path(".png")
    rendered = render_scalar_preview_image(
        image,
        level=level,
        window=window,
        mode=mode,
    )
    Image.fromarray(rendered, mode="L").save(output_path)
    return output_path


def render_object_preview_png(
    object_source: torch.Tensor | str | Path,
    *,
    mode: ObjectPreviewMode = "intensity",
    level: float | None = None,
    window: float | None = None,
    path: str | Path | None = None,
) -> Path:
    image = _prepare_object_preview_scalar(object_source, mode=mode)
    return render_scalar_preview_png(
        image,
        level=level,
        window=window,
        mode=mode,
        path=path,
    )


def render_study_capture_preview_png(
    study: PtychStudy,
    capture_index: int,
    capture_region: CaptureRegion | None = None,
    *,
    level: float | None = None,
    window: float | None = None,
    path: str | Path | None = None,
) -> Path:
    image = _prepare_study_capture_scalar(
        study,
        capture_index,
        capture_region=capture_region,
    )
    return render_scalar_preview_png(
        image,
        level=level,
        window=window,
        mode="scalar",
        path=path,
    )


def _encode_float32_image(image: Float[torch.Tensor, "H W"]) -> str:
    array = np.asarray(image.detach().cpu().to(torch.float32).numpy(), dtype="<f4")
    return base64.b64encode(array.tobytes()).decode("ascii")


def _write_scalar_preview_html(
    image: Float[torch.Tensor, "H W"],
    *,
    title: str = "Scalar Preview",
    level: float | None = None,
    window: float | None = None,
    mode: ObjectPreviewMode | Literal["scalar"] = "scalar",
    path: str | Path | None = None,
) -> Path:
    resolved_level, resolved_window, data_min, data_max = _resolve_level_window(
        image,
        level=level,
        window=window,
        mode=mode,
    )

    window_min = max((data_max - data_min) / 1000.0, 1e-6)
    window_max = max(data_max - data_min, resolved_window, window_min * 10.0)
    if data_max <= data_min:
        level_min = resolved_level - 1.0
        level_max = resolved_level + 1.0
    else:
        margin = max((data_max - data_min) * 0.1, 1e-6)
        level_min = data_min - margin
        level_max = data_max + margin

    payload = {
        "title": title,
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "data_base64": _encode_float32_image(image),
        "initial_level": resolved_level,
        "initial_window": resolved_window,
        "default_level": resolved_level,
        "default_window": resolved_window,
        "level_min": level_min,
        "level_max": level_max,
        "window_min": window_min,
        "window_max": window_max,
        "data_min": data_min,
        "data_max": data_max,
    }

    output_path = Path(path) if path is not None else _default_temp_path(".html")
    output_path.write_text(_build_scalar_preview_html(payload), encoding="utf-8")
    return output_path


def _write_object_preview_html(
    object_source: torch.Tensor | str | Path,
    *,
    mode: ObjectPreviewMode = "intensity",
    title: str | None = None,
    level: float | None = None,
    window: float | None = None,
    path: str | Path | None = None,
) -> Path:
    image = _prepare_object_preview_scalar(object_source, mode=mode)
    object_label = str(object_source) if isinstance(object_source, str | Path) else "object tensor"
    resolved_title = title or f"{object_label} | mode={mode}"
    return _write_scalar_preview_html(
        image,
        title=resolved_title,
        level=level,
        window=window,
        mode=mode,
        path=path,
    )


def _write_study_capture_preview_html(
    study: PtychStudy,
    capture_index: int,
    capture_region: CaptureRegion | None = None,
    *,
    title: str | None = None,
    level: float | None = None,
    window: float | None = None,
    path: str | Path | None = None,
) -> Path:
    image = _prepare_study_capture_scalar(
        study,
        capture_index,
        capture_region=capture_region,
    )
    valid_captures = _valid_study_captures(study)
    capture_meta = valid_captures[capture_index]
    resolved_title = title or _capture_preview_title(study, capture_index, capture_meta)
    return _write_scalar_preview_html(
        image,
        title=resolved_title,
        level=level,
        window=window,
        mode="scalar",
        path=path,
    )


def _build_scalar_preview_html(payload: Mapping[str, object]) -> str:
    payload_json = json.dumps(payload)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Scalar Preview</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f3f0ea;
      --panel: #fffdf8;
      --ink: #1f1a16;
      --muted: #6c625b;
      --accent: #b44c2f;
      --line: #d9cfc5;
    }}
    body {{
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", serif;
      background: radial-gradient(circle at top, #faf6ef 0%, var(--bg) 60%);
      color: var(--ink);
    }}
    .shell {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 24px;
      display: grid;
      gap: 24px;
      grid-template-columns: minmax(320px, 1fr) 320px;
    }}
    .viewer, .controls {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: 0 12px 32px rgba(54, 37, 26, 0.08);
    }}
    .viewer {{
      padding: 18px;
    }}
    .controls {{
      padding: 20px;
      display: grid;
      gap: 18px;
      align-content: start;
    }}
    h1 {{
      margin: 0 0 12px;
      font-size: 1.3rem;
      line-height: 1.25;
    }}
    canvas {{
      width: 100%;
      height: auto;
      display: block;
      border-radius: 12px;
      background: #111;
      image-rendering: pixelated;
    }}
    .group {{
      display: grid;
      gap: 8px;
    }}
    .slider-row {{
      display: grid;
      grid-template-columns: 1fr 96px;
      gap: 12px;
      align-items: center;
    }}
    label {{
      font-size: 0.92rem;
      color: var(--muted);
    }}
    input[type="range"] {{
      width: 100%;
      accent-color: var(--accent);
    }}
    input[type="number"] {{
      width: 100%;
      box-sizing: border-box;
      padding: 8px 10px;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #fff;
      color: var(--ink);
    }}
    button {{
      border: 0;
      border-radius: 999px;
      padding: 10px 14px;
      background: var(--accent);
      color: white;
      font-weight: 600;
      cursor: pointer;
    }}
    .stats {{
      font-size: 0.9rem;
      color: var(--muted);
      line-height: 1.5;
    }}
    @media (max-width: 900px) {{
      .shell {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="viewer">
      <h1 id="title"></h1>
      <canvas id="canvas"></canvas>
    </section>
    <aside class="controls">
      <div class="group">
        <label for="level-slider">Level</label>
        <div class="slider-row">
          <input id="level-slider" type="range">
          <input id="level-number" type="number" step="any">
        </div>
      </div>
      <div class="group">
        <label for="window-slider">Window</label>
        <div class="slider-row">
          <input id="window-slider" type="range">
          <input id="window-number" type="number" step="any">
        </div>
      </div>
      <button id="reset-button" type="button">Reset Auto Window</button>
      <div class="stats" id="stats"></div>
    </aside>
  </div>
  <script>
    const payload = {payload_json};
    const titleEl = document.getElementById("title");
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");
    const levelSlider = document.getElementById("level-slider");
    const levelNumber = document.getElementById("level-number");
    const windowSlider = document.getElementById("window-slider");
    const windowNumber = document.getElementById("window-number");
    const resetButton = document.getElementById("reset-button");
    const stats = document.getElementById("stats");

    titleEl.textContent = payload.title;
    canvas.width = payload.width;
    canvas.height = payload.height;

    function decodeBase64ToFloat32(base64) {{
      const binary = atob(base64);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i += 1) {{
        bytes[i] = binary.charCodeAt(i);
      }}
      return new Float32Array(bytes.buffer);
    }}

    const data = decodeBase64ToFloat32(payload.data_base64);

    function bindSlider(slider, numberInput, min, max, value) {{
      slider.min = String(min);
      slider.max = String(max);
      slider.step = String((max - min) / 1000 || 1e-6);
      slider.value = String(value);
      numberInput.value = String(value);
      numberInput.min = String(min);
      numberInput.max = String(max);
    }}

    bindSlider(levelSlider, levelNumber, payload.level_min, payload.level_max, payload.initial_level);
    bindSlider(windowSlider, windowNumber, payload.window_min, payload.window_max, payload.initial_window);

    function clamp(value, min, max) {{
      return Math.min(Math.max(value, min), max);
    }}

    function syncPair(source, target) {{
      target.value = source.value;
      render();
    }}

    function readNumber(input, fallback) {{
      const parsed = Number.parseFloat(input.value);
      return Number.isFinite(parsed) ? parsed : fallback;
    }}

    function render() {{
      const level = clamp(readNumber(levelNumber, payload.default_level), payload.level_min, payload.level_max);
      const window = clamp(readNumber(windowNumber, payload.default_window), payload.window_min, payload.window_max);

      levelSlider.value = String(level);
      levelNumber.value = String(level);
      windowSlider.value = String(window);
      windowNumber.value = String(window);

      const lower = level - window / 2;
      const upper = level + window / 2;
      const denom = Math.max(upper - lower, 1e-12);

      const imageData = ctx.createImageData(payload.width, payload.height);
      for (let i = 0; i < data.length; i += 1) {{
        const normalized = clamp((data[i] - lower) / denom, 0, 1);
        const value = Math.round(normalized * 255);
        const offset = i * 4;
        imageData.data[offset] = value;
        imageData.data[offset + 1] = value;
        imageData.data[offset + 2] = value;
        imageData.data[offset + 3] = 255;
      }}
      ctx.putImageData(imageData, 0, 0);
      stats.innerHTML = `
        data range: [${{payload.data_min.toFixed(6)}}, ${{payload.data_max.toFixed(6)}}]<br>
        display range: [${{lower.toFixed(6)}}, ${{upper.toFixed(6)}}]
      `;
    }}

    levelSlider.addEventListener("input", () => syncPair(levelSlider, levelNumber));
    windowSlider.addEventListener("input", () => syncPair(windowSlider, windowNumber));
    levelNumber.addEventListener("change", render);
    windowNumber.addEventListener("change", render);
    resetButton.addEventListener("click", () => {{
      bindSlider(levelSlider, levelNumber, payload.level_min, payload.level_max, payload.default_level);
      bindSlider(windowSlider, windowNumber, payload.window_min, payload.window_max, payload.default_window);
      render();
    }});

    render();
  </script>
</body>
</html>
"""

def _open_preview_html(path: Path) -> None:
    webbrowser.open(path.resolve().as_uri(), new=2)


def show_object_preview(
    object_source: torch.Tensor | str | Path,
    *,
    mode: ObjectPreviewMode = "intensity",
    title: str | None = None,
    level: float | None = None,
    window: float | None = None,
    open_browser: bool = True,
) -> None:
    html_path = _write_object_preview_html(
        object_source,
        mode=mode,
        title=title,
        level=level,
        window=window,
    )
    if open_browser:
        _open_preview_html(html_path)


def show_study_capture(
    study: PtychStudy,
    capture_index: int,
    capture_region: CaptureRegion | None = None,
    *,
    title: str | None = None,
    level: float | None = None,
    window: float | None = None,
    open_browser: bool = True,
) -> None:
    valid_captures = _valid_study_captures(study)
    capture_meta = valid_captures[capture_index]
    resolved_title = title
    if resolved_title is None:
        resolved_title = _capture_preview_title(study, capture_index, capture_meta)
    html_path = _write_study_capture_preview_html(
        study,
        capture_index,
        capture_region=capture_region,
        title=resolved_title,
        level=level,
        window=window,
    )
    if open_browser:
        _open_preview_html(html_path)
