from __future__ import annotations

from importlib.resources import files
from typing import Any

import torch
from jaxtyping import Complex, Float
from torch import Tensor


_shader_library: Any | None = None


def _get_shader_library() -> Any:
    global _shader_library
    if _shader_library is None:
        shader_source = (
            files("ptych.core").joinpath("fourier_shift_filter.metal").read_text()
        )
        _shader_library = torch.mps.compile_shader(shader_source)
    return _shader_library


def _shift_filter(
    object_fourier: Complex[Tensor, "patch_batch object_height object_width"],
    pupil: Complex[Tensor, "patch_batch object_height object_width"],
    shift_y: Float[Tensor, "illumination"],
    shift_x: Float[Tensor, "illumination"],
    kernel_name: str,
) -> Complex[Tensor, "patch_batch illumination object_height object_width"]:
    object_fourier = object_fourier.contiguous()
    pupil = pupil.contiguous()
    shift_y = shift_y.to(device=object_fourier.device, dtype=torch.float32).contiguous()
    shift_x = shift_x.to(device=object_fourier.device, dtype=torch.float32).contiguous()
    patch_batch_size, height, width = object_fourier.shape
    num_illuminations = shift_x.shape[0]
    out = torch.empty(
        patch_batch_size,
        num_illuminations,
        height,
        width,
        dtype=torch.complex64,
        device=object_fourier.device,
    )
    getattr(_get_shader_library(), kernel_name)(
        out,
        object_fourier,
        pupil,
        shift_y,
        shift_x,
        patch_batch_size,
        num_illuminations,
        height,
        width,
    )
    return out


def shift_filter_bilinear(
    object_fourier: Complex[Tensor, "patch_batch object_height object_width"],
    pupil: Complex[Tensor, "patch_batch object_height object_width"],
    shift_y: Float[Tensor, "illumination"],
    shift_x: Float[Tensor, "illumination"],
) -> Complex[Tensor, "patch_batch illumination object_height object_width"]:
    return _shift_filter(
        object_fourier,
        pupil,
        shift_y,
        shift_x,
        "shift_filter_bilinear",
    )


def shift_filter_lanczos3(
    object_fourier: Complex[Tensor, "patch_batch object_height object_width"],
    pupil: Complex[Tensor, "patch_batch object_height object_width"],
    shift_y: Float[Tensor, "illumination"],
    shift_x: Float[Tensor, "illumination"],
) -> Complex[Tensor, "patch_batch illumination object_height object_width"]:
    return _shift_filter(
        object_fourier,
        pupil,
        shift_y,
        shift_x,
        "shift_filter_lanczos3",
    )


def _shift_filter_oversampled(
    object_fourier: Complex[Tensor, "patch_batch source_height source_width"],
    pupil: Complex[Tensor, "patch_batch object_height object_width"],
    shift_y: Float[Tensor, "illumination"],
    shift_x: Float[Tensor, "illumination"],
    kernel_name: str,
) -> Complex[Tensor, "patch_batch illumination object_height object_width"]:
    object_fourier = object_fourier.contiguous()
    pupil = pupil.contiguous()
    shift_y = shift_y.to(device=object_fourier.device, dtype=torch.float32).contiguous()
    shift_x = shift_x.to(device=object_fourier.device, dtype=torch.float32).contiguous()
    patch_batch_size, source_height, source_width = object_fourier.shape
    _, height, width = pupil.shape
    num_illuminations = shift_x.shape[0]
    out = torch.empty(
        patch_batch_size,
        num_illuminations,
        height,
        width,
        dtype=torch.complex64,
        device=object_fourier.device,
    )
    getattr(_get_shader_library(), kernel_name)(
        out,
        object_fourier,
        pupil,
        shift_y,
        shift_x,
        patch_batch_size,
        num_illuminations,
        height,
        width,
        source_height,
        source_width,
    )
    return out


def shift_filter_oversampled_lanczos3(
    object_fourier: Complex[Tensor, "patch_batch source_height source_width"],
    pupil: Complex[Tensor, "patch_batch object_height object_width"],
    shift_y: Float[Tensor, "illumination"],
    shift_x: Float[Tensor, "illumination"],
) -> Complex[Tensor, "patch_batch illumination object_height object_width"]:
    return _ShiftFilterOversampledLanczos3.apply(
        object_fourier, pupil, shift_y, shift_x
    )


def shift_filter_oversampled_bilinear(
    object_fourier: Complex[Tensor, "patch_batch source_height source_width"],
    pupil: Complex[Tensor, "patch_batch object_height object_width"],
    shift_y: Float[Tensor, "illumination"],
    shift_x: Float[Tensor, "illumination"],
) -> Complex[Tensor, "patch_batch illumination object_height object_width"]:
    return _shift_filter_oversampled(
        object_fourier,
        pupil,
        shift_y,
        shift_x,
        "shift_filter_oversampled_bilinear",
    )


class _ShiftFilterOversampledLanczos3(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        object_fourier: Complex[Tensor, "patch_batch source_height source_width"],
        pupil: Complex[Tensor, "patch_batch object_height object_width"],
        shift_y: Float[Tensor, "illumination"],
        shift_x: Float[Tensor, "illumination"],
    ) -> Complex[Tensor, "patch_batch illumination object_height object_width"]:
        object_fourier = object_fourier.contiguous()
        pupil = pupil.contiguous()
        shift_y = shift_y.to(
            device=object_fourier.device,
            dtype=torch.float32,
        ).contiguous()
        shift_x = shift_x.to(
            device=object_fourier.device,
            dtype=torch.float32,
        ).contiguous()
        patch_batch_size, source_height, source_width = object_fourier.shape
        _, height, width = pupil.shape
        num_illuminations = shift_x.shape[0]
        out = torch.empty(
            patch_batch_size,
            num_illuminations,
            height,
            width,
            dtype=torch.complex64,
            device=object_fourier.device,
        )
        _get_shader_library().shift_filter_oversampled_lanczos3(
            out,
            object_fourier,
            pupil,
            shift_y,
            shift_x,
            patch_batch_size,
            num_illuminations,
            height,
            width,
            source_height,
            source_width,
        )
        ctx.save_for_backward(object_fourier, pupil, shift_y, shift_x)
        ctx.shape = (
            patch_batch_size,
            num_illuminations,
            height,
            width,
            source_height,
            source_width,
        )
        return out

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        (grad_out,) = grad_outputs
        object_fourier, pupil, shift_y, shift_x = ctx.saved_tensors
        (
            patch_batch_size,
            num_illuminations,
            height,
            width,
            source_height,
            source_width,
        ) = ctx.shape
        grad_object_fourier = torch.zeros_like(object_fourier)
        grad_pupil = torch.zeros_like(pupil)
        _get_shader_library().shift_filter_oversampled_lanczos3_backward(
            grad_out.contiguous(),
            grad_object_fourier,
            grad_pupil,
            object_fourier,
            pupil,
            shift_y,
            shift_x,
            patch_batch_size,
            num_illuminations,
            height,
            width,
            source_height,
            source_width,
        )
        return grad_object_fourier, grad_pupil, None, None
