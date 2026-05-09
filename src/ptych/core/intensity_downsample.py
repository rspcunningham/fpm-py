from __future__ import annotations

from importlib.resources import files
from typing import Any

import torch
import torch.nn.functional as F
from jaxtyping import Complex, Float
from torch import Tensor


_shader_library: Any | None = None


def _get_shader_library() -> Any:
    global _shader_library
    if _shader_library is None:
        shader_source = (
            files("ptych.core").joinpath("intensity_downsample.metal").read_text()
        )
        _shader_library = torch.mps.compile_shader(shader_source)
    return _shader_library


class _IntensityDownsample2(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        fields: Complex[Tensor, "patch_batch illumination object_height object_width"],
    ) -> Float[Tensor, "patch_batch illumination height width"]:
        fields = fields.contiguous()
        patch_batch_size, num_illuminations, object_height, object_width = fields.shape
        height = object_height // 2
        width = object_width // 2
        out = torch.empty(
            patch_batch_size,
            num_illuminations,
            height,
            width,
            dtype=torch.float32,
            device=fields.device,
        )
        _get_shader_library().intensity_downsample2(
            out,
            fields,
            patch_batch_size,
            num_illuminations,
            height,
            width,
        )
        ctx.save_for_backward(fields)
        ctx.shape = (patch_batch_size, num_illuminations, height, width)
        return out

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        (grad_out,) = grad_outputs
        (fields,) = ctx.saved_tensors
        patch_batch_size, num_illuminations, height, width = ctx.shape
        grad_fields = torch.empty_like(fields)
        _get_shader_library().intensity_downsample2_backward(
            grad_fields,
            fields,
            grad_out.contiguous(),
            patch_batch_size,
            num_illuminations,
            height,
            width,
        )
        return (grad_fields,)


class _IntensityDownsample4(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        fields: Complex[Tensor, "patch_batch illumination object_height object_width"],
    ) -> Float[Tensor, "patch_batch illumination height width"]:
        fields = fields.contiguous()
        patch_batch_size, num_illuminations, object_height, object_width = fields.shape
        height = object_height // 4
        width = object_width // 4
        out = torch.empty(
            patch_batch_size,
            num_illuminations,
            height,
            width,
            dtype=torch.float32,
            device=fields.device,
        )
        _get_shader_library().intensity_downsample4(
            out,
            fields,
            patch_batch_size,
            num_illuminations,
            height,
            width,
        )
        ctx.save_for_backward(fields)
        ctx.shape = (patch_batch_size, num_illuminations, height, width)
        return out

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        (grad_out,) = grad_outputs
        (fields,) = ctx.saved_tensors
        patch_batch_size, num_illuminations, height, width = ctx.shape
        grad_fields = torch.empty_like(fields)
        _get_shader_library().intensity_downsample4_backward(
            grad_fields,
            fields,
            grad_out.contiguous(),
            patch_batch_size,
            num_illuminations,
            height,
            width,
        )
        return (grad_fields,)


def _can_use_mps_ratio(
    fields: Tensor,
    object_to_capture_ratio: int,
) -> bool:
    object_height, object_width = fields.shape[-2:]
    return (
        fields.device.type == "mps"
        and fields.dtype == torch.complex64
        and object_height % object_to_capture_ratio == 0
        and object_width % object_to_capture_ratio == 0
    )


def intensity_downsample(
    fields: Complex[Tensor, "patch_batch illumination object_height object_width"],
    object_to_capture_ratio: int,
) -> Float[Tensor, "patch_batch illumination height width"]:
    if object_to_capture_ratio == 2 and _can_use_mps_ratio(fields, 2):
        return _IntensityDownsample2.apply(fields)

    if object_to_capture_ratio == 4 and _can_use_mps_ratio(fields, 4):
        return _IntensityDownsample4.apply(fields)

    patch_batch_size, num_illuminations, object_size, _ = fields.shape
    predicted_intensities_full_res = fields.abs().square()
    return F.avg_pool2d(
        predicted_intensities_full_res.reshape(
            patch_batch_size * num_illuminations,
            1,
            object_size,
            object_size,
        ),
        kernel_size=object_to_capture_ratio,
        stride=object_to_capture_ratio,
    ).reshape(
        patch_batch_size,
        num_illuminations,
        object_size // object_to_capture_ratio,
        object_size // object_to_capture_ratio,
    )
