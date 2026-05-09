#include <metal_stdlib>
using namespace metal;

kernel void intensity_downsample2(device float *out,
                                  device const float2 *fields,
                                  constant uint &batch_size,
                                  constant uint &num_illuminations,
                                  constant uint &height, constant uint &width,
                                  uint idx [[thread_position_in_grid]]) {
  uint total = batch_size * num_illuminations * height * width;
  if (idx >= total) {
    return;
  }

  uint x_low = idx % width;
  uint tmp = idx / width;
  uint y_low = tmp % height;
  tmp = tmp / height;
  uint illumination = tmp % num_illuminations;
  uint batch = tmp / num_illuminations;

  uint object_height = height * 2;
  uint object_width = width * 2;
  uint base =
      ((batch * num_illuminations + illumination) * object_height + y_low * 2) *
          object_width +
      x_low * 2;

  float2 v00 = fields[base];
  float2 v01 = fields[base + 1];
  float2 v10 = fields[base + object_width];
  float2 v11 = fields[base + object_width + 1];

  out[idx] =
      0.25f * (dot(v00, v00) + dot(v01, v01) + dot(v10, v10) + dot(v11, v11));
}

kernel void intensity_downsample2_backward(
    device float2 *grad_fields, device const float2 *fields,
    device const float *grad_out, constant uint &batch_size,
    constant uint &num_illuminations, constant uint &height,
    constant uint &width, uint idx [[thread_position_in_grid]]) {
  uint object_height = height * 2;
  uint object_width = width * 2;
  uint total = batch_size * num_illuminations * object_height * object_width;
  if (idx >= total) {
    return;
  }

  uint x_high = idx % object_width;
  uint tmp = idx / object_width;
  uint y_high = tmp % object_height;
  tmp = tmp / object_height;
  uint illumination = tmp % num_illuminations;
  uint batch = tmp / num_illuminations;

  uint low_idx =
      ((batch * num_illuminations + illumination) * height + (y_high / 2)) *
          width +
      (x_high / 2);

  float scale = 0.5f * grad_out[low_idx];
  float2 value = fields[idx];
  grad_fields[idx] = float2(scale * value.x, scale * value.y);
}

kernel void intensity_downsample4(device float *out,
                                  device const float2 *fields,
                                  constant uint &batch_size,
                                  constant uint &num_illuminations,
                                  constant uint &height, constant uint &width,
                                  uint idx [[thread_position_in_grid]]) {
  uint total = batch_size * num_illuminations * height * width;
  if (idx >= total) {
    return;
  }

  uint x_low = idx % width;
  uint tmp = idx / width;
  uint y_low = tmp % height;
  tmp = tmp / height;
  uint illumination = tmp % num_illuminations;
  uint batch = tmp / num_illuminations;

  uint object_height = height * 4;
  uint object_width = width * 4;
  uint base =
      ((batch * num_illuminations + illumination) * object_height + y_low * 4) *
          object_width +
      x_low * 4;

  float sum = 0.0f;
  for (uint dy = 0; dy < 4; ++dy) {
    uint row = base + dy * object_width;
    float4 left = *reinterpret_cast<device const float4 *>(fields + row);
    float4 right = *reinterpret_cast<device const float4 *>(fields + row + 2);
    sum += dot(left, left) + dot(right, right);
  }

  out[idx] = 0.0625f * sum;
}

kernel void intensity_downsample4_backward(
    device float2 *grad_fields, device const float2 *fields,
    device const float *grad_out, constant uint &batch_size,
    constant uint &num_illuminations, constant uint &height,
    constant uint &width, uint idx [[thread_position_in_grid]]) {
  uint object_height = height * 4;
  uint object_width = width * 4;
  uint total = batch_size * num_illuminations * object_height * object_width;
  if (idx >= total) {
    return;
  }

  uint x_high = idx % object_width;
  uint tmp = idx / object_width;
  uint y_high = tmp % object_height;
  tmp = tmp / object_height;
  uint illumination = tmp % num_illuminations;
  uint batch = tmp / num_illuminations;

  uint low_idx =
      ((batch * num_illuminations + illumination) * height + (y_high / 4)) *
          width +
      (x_high / 4);

  float scale = 0.125f * grad_out[low_idx];
  float2 value = fields[idx];
  grad_fields[idx] = float2(scale * value.x, scale * value.y);
}
