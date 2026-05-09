#include <metal_stdlib>
using namespace metal;

static inline float2 complex_mul(float2 a, float2 b) {
  return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

static inline float2 complex_mul_conj(float2 a, float2 b) {
  return float2(a.x * b.x + a.y * b.y, a.y * b.x - a.x * b.y);
}

static inline void atomic_add_complex(device float2 *target, uint index,
                                      float2 value) {
  device atomic_float *atomic_target =
      reinterpret_cast<device atomic_float *>(target + index);
  atomic_fetch_add_explicit(atomic_target, value.x, memory_order_relaxed);
  atomic_fetch_add_explicit(atomic_target + 1, value.y, memory_order_relaxed);
}

static inline uint wrap_index(int value, uint size) {
  int wrapped = value % int(size);
  return uint(wrapped < 0 ? wrapped + int(size) : wrapped);
}

static inline float sinc(float x) {
  float ax = abs(x);
  if (ax < 1.0e-6f) {
    return 1.0f;
  }
  float pix = M_PI_F * x;
  return sin(pix) / pix;
}

static inline float lanczos3_weight(float x) {
  float ax = abs(x);
  if (ax >= 3.0f) {
    return 0.0f;
  }
  return sinc(x) * sinc(x / 3.0f);
}

kernel void shift_filter_bilinear(
    device float2 *out, device const float2 *object_fourier,
    device const float2 *pupil, device const float *shift_y,
    device const float *shift_x, constant uint &batch_size,
    constant uint &num_illuminations, constant uint &height,
    constant uint &width, uint idx [[thread_position_in_grid]]) {
  uint total = batch_size * num_illuminations * height * width;
  if (idx >= total) {
    return;
  }

  uint x = idx % width;
  uint tmp = idx / width;
  uint y = tmp % height;
  tmp = tmp / height;
  uint illumination = tmp % num_illuminations;
  uint batch = tmp / num_illuminations;

  float source_y = float(y) - shift_y[illumination];
  float source_x = float(x) - shift_x[illumination];
  int y0_signed = int(floor(source_y));
  int x0_signed = int(floor(source_x));
  float wy = source_y - float(y0_signed);
  float wx = source_x - float(x0_signed);

  uint y0 = wrap_index(y0_signed, height);
  uint x0 = wrap_index(x0_signed, width);
  uint y1 = wrap_index(y0_signed + 1, height);
  uint x1 = wrap_index(x0_signed + 1, width);

  uint plane_base = batch * height * width;
  float2 v00 = object_fourier[plane_base + y0 * width + x0];
  float2 v01 = object_fourier[plane_base + y0 * width + x1];
  float2 v10 = object_fourier[plane_base + y1 * width + x0];
  float2 v11 = object_fourier[plane_base + y1 * width + x1];

  float2 top = mix(v00, v01, wx);
  float2 bottom = mix(v10, v11, wx);
  float2 shifted = mix(top, bottom, wy);
  float2 pupil_value = pupil[plane_base + y * width + x];
  out[idx] = complex_mul(pupil_value, shifted);
}

kernel void shift_filter_lanczos3(
    device float2 *out, device const float2 *object_fourier,
    device const float2 *pupil, device const float *shift_y,
    device const float *shift_x, constant uint &batch_size,
    constant uint &num_illuminations, constant uint &height,
    constant uint &width, uint idx [[thread_position_in_grid]]) {
  uint total = batch_size * num_illuminations * height * width;
  if (idx >= total) {
    return;
  }

  uint x = idx % width;
  uint tmp = idx / width;
  uint y = tmp % height;
  tmp = tmp / height;
  uint illumination = tmp % num_illuminations;
  uint batch = tmp / num_illuminations;

  float source_y = float(y) - shift_y[illumination];
  float source_x = float(x) - shift_x[illumination];
  int y_floor = int(floor(source_y));
  int x_floor = int(floor(source_x));

  float wy[6];
  float wx[6];
  float y_sum = 0.0f;
  float x_sum = 0.0f;
  for (uint tap = 0; tap < 6; ++tap) {
    int offset = int(tap) - 2;
    float y_weight = lanczos3_weight(source_y - float(y_floor + offset));
    float x_weight = lanczos3_weight(source_x - float(x_floor + offset));
    wy[tap] = y_weight;
    wx[tap] = x_weight;
    y_sum += y_weight;
    x_sum += x_weight;
  }

  uint plane_base = batch * height * width;
  float2 shifted = float2(0.0f, 0.0f);
  for (uint dy = 0; dy < 6; ++dy) {
    uint source_row = wrap_index(y_floor + int(dy) - 2, height);
    for (uint dx = 0; dx < 6; ++dx) {
      uint source_col = wrap_index(x_floor + int(dx) - 2, width);
      float weight = wy[dy] * wx[dx];
      shifted += weight * object_fourier[plane_base + source_row * width + source_col];
    }
  }
  shifted /= y_sum * x_sum;

  float2 pupil_value = pupil[plane_base + y * width + x];
  out[idx] = complex_mul(pupil_value, shifted);
}

kernel void shift_filter_oversampled_lanczos3(
    device float2 *out, device const float2 *object_fourier,
    device const float2 *pupil, device const float *shift_y,
    device const float *shift_x, constant uint &batch_size,
    constant uint &num_illuminations, constant uint &height,
    constant uint &width, constant uint &source_height,
    constant uint &source_width, uint idx [[thread_position_in_grid]]) {
  uint total = batch_size * num_illuminations * height * width;
  if (idx >= total) {
    return;
  }

  uint x = idx % width;
  uint tmp = idx / width;
  uint y = tmp % height;
  tmp = tmp / height;
  uint illumination = tmp % num_illuminations;
  uint batch = tmp / num_illuminations;

  float y_scale = float(source_height) / float(height);
  float x_scale = float(source_width) / float(width);
  float source_y = (float(y) - shift_y[illumination]) * y_scale;
  float source_x = (float(x) - shift_x[illumination]) * x_scale;
  int y_floor = int(floor(source_y));
  int x_floor = int(floor(source_x));

  float wy[6];
  float wx[6];
  float y_sum = 0.0f;
  float x_sum = 0.0f;
  for (uint tap = 0; tap < 6; ++tap) {
    int offset = int(tap) - 2;
    float y_weight = lanczos3_weight(source_y - float(y_floor + offset));
    float x_weight = lanczos3_weight(source_x - float(x_floor + offset));
    wy[tap] = y_weight;
    wx[tap] = x_weight;
    y_sum += y_weight;
    x_sum += x_weight;
  }

  uint source_plane_base = batch * source_height * source_width;
  float2 shifted = float2(0.0f, 0.0f);
  for (uint dy = 0; dy < 6; ++dy) {
    uint source_row = wrap_index(y_floor + int(dy) - 2, source_height);
    for (uint dx = 0; dx < 6; ++dx) {
      uint source_col = wrap_index(x_floor + int(dx) - 2, source_width);
      float weight = wy[dy] * wx[dx];
      shifted += weight *
                 object_fourier[source_plane_base + source_row * source_width +
                                source_col];
    }
  }
  shifted *= sqrt(y_scale * x_scale) / (y_sum * x_sum);

  uint pupil_plane_base = batch * height * width;
  float2 pupil_value = pupil[pupil_plane_base + y * width + x];
  out[idx] = complex_mul(pupil_value, shifted);
}

kernel void shift_filter_oversampled_lanczos3_backward(
    device const float2 *grad_out, device float2 *grad_object_fourier,
    device float2 *grad_pupil, device const float2 *object_fourier,
    device const float2 *pupil, device const float *shift_y,
    device const float *shift_x, constant uint &batch_size,
    constant uint &num_illuminations, constant uint &height,
    constant uint &width, constant uint &source_height,
    constant uint &source_width, uint idx [[thread_position_in_grid]]) {
  uint total = batch_size * num_illuminations * height * width;
  if (idx >= total) {
    return;
  }

  uint x = idx % width;
  uint tmp = idx / width;
  uint y = tmp % height;
  tmp = tmp / height;
  uint illumination = tmp % num_illuminations;
  uint batch = tmp / num_illuminations;

  float y_scale = float(source_height) / float(height);
  float x_scale = float(source_width) / float(width);
  float source_y = (float(y) - shift_y[illumination]) * y_scale;
  float source_x = (float(x) - shift_x[illumination]) * x_scale;
  int y_floor = int(floor(source_y));
  int x_floor = int(floor(source_x));

  float wy[6];
  float wx[6];
  float y_sum = 0.0f;
  float x_sum = 0.0f;
  for (uint tap = 0; tap < 6; ++tap) {
    int offset = int(tap) - 2;
    float y_weight = lanczos3_weight(source_y - float(y_floor + offset));
    float x_weight = lanczos3_weight(source_x - float(x_floor + offset));
    wy[tap] = y_weight;
    wx[tap] = x_weight;
    y_sum += y_weight;
    x_sum += x_weight;
  }

  float norm_scale = sqrt(y_scale * x_scale) / (y_sum * x_sum);
  uint source_plane_base = batch * source_height * source_width;
  float2 shifted = float2(0.0f, 0.0f);
  for (uint dy = 0; dy < 6; ++dy) {
    uint source_row = wrap_index(y_floor + int(dy) - 2, source_height);
    for (uint dx = 0; dx < 6; ++dx) {
      uint source_col = wrap_index(x_floor + int(dx) - 2, source_width);
      float weight = norm_scale * wy[dy] * wx[dx];
      shifted += weight *
                 object_fourier[source_plane_base + source_row * source_width +
                                source_col];
    }
  }

  uint pupil_plane_base = batch * height * width;
  uint pupil_index = pupil_plane_base + y * width + x;
  float2 pupil_value = pupil[pupil_index];
  float2 grad_value = grad_out[idx];
  float2 grad_shifted = complex_mul_conj(grad_value, pupil_value);
  atomic_add_complex(grad_pupil, pupil_index, complex_mul_conj(grad_value, shifted));

  for (uint dy = 0; dy < 6; ++dy) {
    uint source_row = wrap_index(y_floor + int(dy) - 2, source_height);
    for (uint dx = 0; dx < 6; ++dx) {
      uint source_col = wrap_index(x_floor + int(dx) - 2, source_width);
      float weight = norm_scale * wy[dy] * wx[dx];
      atomic_add_complex(
          grad_object_fourier,
          source_plane_base + source_row * source_width + source_col,
          weight * grad_shifted);
    }
  }
}

kernel void shift_filter_oversampled_bilinear(
    device float2 *out, device const float2 *object_fourier,
    device const float2 *pupil, device const float *shift_y,
    device const float *shift_x, constant uint &batch_size,
    constant uint &num_illuminations, constant uint &height,
    constant uint &width, constant uint &source_height,
    constant uint &source_width, uint idx [[thread_position_in_grid]]) {
  uint total = batch_size * num_illuminations * height * width;
  if (idx >= total) {
    return;
  }

  uint x = idx % width;
  uint tmp = idx / width;
  uint y = tmp % height;
  tmp = tmp / height;
  uint illumination = tmp % num_illuminations;
  uint batch = tmp / num_illuminations;

  float y_scale = float(source_height) / float(height);
  float x_scale = float(source_width) / float(width);
  float source_y = (float(y) - shift_y[illumination]) * y_scale;
  float source_x = (float(x) - shift_x[illumination]) * x_scale;
  int y0_signed = int(floor(source_y));
  int x0_signed = int(floor(source_x));
  float wy = source_y - float(y0_signed);
  float wx = source_x - float(x0_signed);

  uint y0 = wrap_index(y0_signed, source_height);
  uint x0 = wrap_index(x0_signed, source_width);
  uint y1 = wrap_index(y0_signed + 1, source_height);
  uint x1 = wrap_index(x0_signed + 1, source_width);

  uint source_plane_base = batch * source_height * source_width;
  float2 v00 = object_fourier[source_plane_base + y0 * source_width + x0];
  float2 v01 = object_fourier[source_plane_base + y0 * source_width + x1];
  float2 v10 = object_fourier[source_plane_base + y1 * source_width + x0];
  float2 v11 = object_fourier[source_plane_base + y1 * source_width + x1];

  float2 top = mix(v00, v01, wx);
  float2 bottom = mix(v10, v11, wx);
  float2 shifted = sqrt(y_scale * x_scale) * mix(top, bottom, wy);

  uint pupil_plane_base = batch * height * width;
  float2 pupil_value = pupil[pupil_plane_base + y * width + x];
  out[idx] = complex_mul(pupil_value, shifted);
}
