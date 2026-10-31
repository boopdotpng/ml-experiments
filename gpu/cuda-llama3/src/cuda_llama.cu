#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(expr)                                                        \
  do {                                                                          \
    cudaError_t error__ = (expr);                                                \
    if (error__ != cudaSuccess) {                                                \
      std::ostringstream message__;                                              \
      message__ << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": "   \
                << cudaGetErrorString(error__);                                  \
      throw std::runtime_error(message__.str());                                 \
    }                                                                            \
  } while (0)

namespace {

using bf16 = __nv_bfloat16;
namespace wmma = nvcuda::wmma;

constexpr int kBlock = 256;
constexpr int kNameBytes = 96;
constexpr int kWarp = 32;
constexpr int kRowsPerWarp = 4;
constexpr int kRowsPerBlock = (kBlock / kWarp) * kRowsPerWarp;

#pragma pack(push, 1)
struct HllmHeader {
  char magic[8];
  uint32_t version;
  uint32_t tensor_count;
  int32_t dim;
  int32_t hidden_dim;
  int32_t n_layers;
  int32_t n_heads;
  int32_t n_kv_heads;
  int32_t head_dim;
  int32_t vocab_size;
  int32_t model_max_seq;
  float rope_theta;
  float rms_norm_eps;
  int32_t bos_id;
  int32_t eos_id;
  int32_t pad_id;
};

struct TensorEntry {
  char name[kNameBytes];
  uint64_t offset;
  uint64_t count;
};
#pragma pack(pop)

struct DeviceTensor {
  bf16* data = nullptr;
  uint64_t count = 0;
};

struct Model {
  HllmHeader h{};
  std::map<std::string, DeviceTensor> tensors;

  const bf16* tensor(const std::string& name) const {
    auto it = tensors.find(name);
    if (it == tensors.end()) throw std::runtime_error("missing tensor: " + name);
    return it->second.data;
  }

  bool has(const std::string& name) const {
    return tensors.find(name) != tensors.end();
  }
};

struct Args {
  std::string model_path = "../hip-llama3/models/llama3.2-1b-instruct.hllm";
  std::vector<int> tokens;
  int prompt_length = 0;
  int steps = 128;
  int max_seq = 2048;
  int warmup = 1;
  int runs = 5;
};

struct PerfStats {
  float prefill_ms = 0.0f;
  float decode_ms = 0.0f;
  int prefill_tokens = 0;
  int decode_tokens = 0;
};

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

// One block gathers one token embedding. Padding positions are explicitly
// zeroed so prefill can round its matrix dimensions up to tensor-core tiles.
__global__ void embedding_kernel(const bf16* __restrict__ table,
                                 const int* __restrict__ tokens,
                                 bf16* __restrict__ x, int positions,
                                 int padded_positions, int dim) {
  int position = blockIdx.x;
  if (position >= padded_positions) return;
  if (position >= positions) {
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
      x[position * dim + d] = __float2bfloat16(0.0f);
    }
    return;
  }
  int token = tokens[position];
  for (int d = threadIdx.x; d < dim; d += blockDim.x) {
    x[position * dim + d] = table[token * dim + d];
  }
}

// RMSNorm keeps the sum of squares in FP32, then rounds the normalized output
// back to BF16. A block owns one token; warp shuffles replace most barriers.
__global__ void rmsnorm_kernel(const bf16* __restrict__ x,
                               const bf16* __restrict__ weight,
                               bf16* __restrict__ out, int dim, float eps) {
  __shared__ float warp_sums[8];
  float sum = 0.0f;
  const bf16* row = x + blockIdx.x * dim;
  for (int d = threadIdx.x; d < dim; d += blockDim.x) {
    float value = __bfloat162float(row[d]);
    sum = fmaf(value, value, sum);
  }
  sum = warp_sum(sum);
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  if (lane == 0) warp_sums[warp] = sum;
  __syncthreads();
  if (warp == 0) {
    float total = lane < 8 ? warp_sums[lane] : 0.0f;
    total = warp_sum(total);
    if (lane == 0) warp_sums[0] = total;
  }
  __syncthreads();
  float scale = rsqrtf(warp_sums[0] / dim + eps);
  for (int d = threadIdx.x; d < dim; d += blockDim.x) {
    float value = __bfloat162float(row[d]);
    float gain = __bfloat162float(weight[d]);
    out[blockIdx.x * dim + d] = __float2bfloat16(value * scale * gain);
  }
}

// Decode does not materialize the normalized vector. This small kernel emits
// only its FP32 scalar; projection kernels apply scale*weight while loading x.
__global__ void rms_scale_kernel(const bf16* __restrict__ x,
                                 float* __restrict__ scale_out,
                                 int dim, float eps) {
  __shared__ float warp_sums[8];
  float sum = 0.0f;
  for (int d = threadIdx.x; d < dim; d += blockDim.x) {
    float value = __bfloat162float(x[d]);
    sum = fmaf(value, value, sum);
  }
  sum = warp_sum(sum);
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  if (lane == 0) warp_sums[warp] = sum;
  __syncthreads();
  if (warp == 0) {
    float total = lane < 8 ? warp_sums[lane] : 0.0f;
    total = warp_sum(total);
    if (lane == 0) *scale_out = rsqrtf(total / dim + eps);
  }
}

// --------------------------- Prefill GEMM ---------------------------------
//
// Eight warps compute a 64x128 output tile. Each warp owns 16 output columns
// and four 16-row accumulator fragments. The public WMMA API maps to Blackwell
// BF16 tensor-core MMA instructions. A and B are BF16; every fragment
// accumulator is FP32. W is stored [N,K] row-major, which is exactly B=[K,N]
// column-major without a transpose or repack.
//
// Up to three matrices can be concatenated logically. Q/K/V therefore share
// one launch while retaining their simple independent weight files/buffers.
template <int kRowTiles, int kWarps>
__global__ void tensorcore_gemm_kernel(
    const bf16* __restrict__ x, int m, int k,
    const bf16* __restrict__ w0, bf16* __restrict__ y0, int n0,
    const bf16* __restrict__ w1, bf16* __restrict__ y1, int n1,
    const bf16* __restrict__ w2, bf16* __restrict__ y2, int n2,
    const bf16* __restrict__ residual) {
  constexpr int kTile = 16;
  constexpr int kRows = kRowTiles * kTile;
  constexpr int kColumns = kWarps * kTile;
  __shared__ __align__(32) float tile_out[kWarps][kRowTiles][kTile * kTile];

  int warp = threadIdx.x / kWarp;
  if (warp >= kWarps) return;

  int virtual_n = blockIdx.x * kColumns;
  const bf16* weight;
  bf16* output;
  int segment_n;
  int local_n;
  if (virtual_n < n0) {
    weight = w0; output = y0; segment_n = n0; local_n = virtual_n;
  } else if (virtual_n < n0 + n1) {
    weight = w1; output = y1; segment_n = n1; local_n = virtual_n - n0;
  } else {
    weight = w2; output = y2; segment_n = n2; local_n = virtual_n - n0 - n1;
  }

  int row0 = blockIdx.y * kRows;
  int column0 = local_n + warp * kTile;
  wmma::fragment<wmma::matrix_b, 16, 16, 16,
                 __nv_bfloat16,
                 wmma::col_major> b;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float>
      accumulator[kRowTiles];
#pragma unroll
  for (int row_tile = 0; row_tile < kRowTiles; ++row_tile) {
    wmma::fill_fragment(accumulator[row_tile], 0.0f);
  }

  for (int k0 = 0; k0 < k; k0 += kTile) {
    // B (the weight tile) is loaded once, then reused for 64 prompt tokens.
    wmma::load_matrix_sync(b, weight + column0 * k + k0, k);
#pragma unroll
    for (int row_tile = 0; row_tile < kRowTiles; ++row_tile) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16,
                     wmma::row_major> a;
      wmma::load_matrix_sync(
          a, x + (row0 + row_tile * kTile) * k + k0, k);
      wmma::mma_sync(accumulator[row_tile], a, b, accumulator[row_tile]);
    }
  }
#pragma unroll
  for (int row_tile = 0; row_tile < kRowTiles; ++row_tile) {
    wmma::store_matrix_sync(tile_out[warp][row_tile], accumulator[row_tile],
                            kTile, wmma::mem_row_major);
  }
  __syncthreads();

  // All 256 threads cooperatively run the transparent BF16 epilogue. For O and
  // down projections, residual!=nullptr fuses the residual add into this store.
  for (int index = threadIdx.x; index < kRows * kColumns;
       index += blockDim.x) {
    int local_row = index / kColumns;
    int local_column = index % kColumns;
    int owner_warp = local_column / kTile;
    int owner_column = local_column % kTile;
    int owner_row_tile = local_row / kTile;
    int owner_row = local_row % kTile;
    int row = row0 + local_row;
    int column = local_n + local_column;
    if (row < m && column < segment_n) {
      float value = tile_out[owner_warp][owner_row_tile]
                            [owner_row * kTile + owner_column];
      if (residual) value += __bfloat162float(residual[row * segment_n + column]);
      output[row * segment_n + column] = __float2bfloat16(value);
    }
  }
}

__global__ void silu_mul_kernel(bf16* __restrict__ gate,
                                const bf16* __restrict__ up, int count) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    float g = __bfloat162float(gate[i]);
    float u = __bfloat162float(up[i]);
    gate[i] = __float2bfloat16((g / (1.0f + __expf(-g))) * u);
  }
}

// Llama 3.2 changes the base RoPE frequencies outside the original 8k window.
// Keeping this formula here avoids a lookup table and makes the exact positional
// math visible. Q and K are treated as one logical array to save a launch.
__device__ __forceinline__ float llama3_inv_frequency(int pair, int head_dim,
                                                       float theta) {
  float inv = __expf(-__logf(theta) * (2.0f * pair / head_dim));
  constexpr float factor = 32.0f;
  constexpr float low_factor = 1.0f;
  constexpr float high_factor = 4.0f;
  constexpr float old_context = 8192.0f;
  float wavelength = 6.283185307179586f / inv;
  float low_wavelength = old_context / low_factor;
  float high_wavelength = old_context / high_factor;
  if (wavelength > low_wavelength) return inv / factor;
  if (wavelength < high_wavelength) return inv;
  float smooth = (old_context / wavelength - low_factor) /
                 (high_factor - low_factor);
  return (1.0f - smooth) * inv / factor + smooth * inv;
}

__global__ void rope_prefill_kernel(bf16* __restrict__ q,
                                    bf16* __restrict__ k,
                                    int positions, int n_heads,
                                    int n_kv_heads, int head_dim, float theta) {
  int half = head_dim / 2;
  int q_pairs = positions * n_heads * half;
  int total = q_pairs + positions * n_kv_heads * half;
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= total) return;
  bool is_q = index < q_pairs;
  int logical = is_q ? index : index - q_pairs;
  int heads = is_q ? n_heads : n_kv_heads;
  int pair = logical % half;
  int head = (logical / half) % heads;
  int position = logical / (half * heads);
  bf16* data = is_q ? q : k;
  // Hugging Face Llama's rotate_half pairs the two halves of a head:
  // (0,32), (1,33), ... for head_dim=64. The HIP teaching kernel paired
  // adjacent channels; that is a different positional encoding.
  int base = (position * heads + head) * head_dim + pair;
  float angle = position * llama3_inv_frequency(pair, head_dim, theta);
  float sine, cosine;
  __sincosf(angle, &sine, &cosine);
  float a = __bfloat162float(data[base]);
  float b = __bfloat162float(data[base + half]);
  data[base] = __float2bfloat16(a * cosine - b * sine);
  data[base + half] = __float2bfloat16(a * sine + b * cosine);
}

__global__ void copy_prompt_kv_kernel(const bf16* __restrict__ k,
                                      const bf16* __restrict__ v,
                                      bf16* __restrict__ key_cache,
                                      bf16* __restrict__ value_cache,
                                      int positions, int kv_dim) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int count = positions * kv_dim;
  if (index < count) {
    key_cache[index] = k[index];
    value_cache[index] = v[index];
  }
}

// Simplified FlashAttention for causal prefill. One warp owns one (token,head)
// row, computes each Q.K score cooperatively, and updates (m,l,acc) online.
// Nothing proportional to T*T is ever written to memory.
__global__ void flash_prefill_kernel(const bf16* __restrict__ q,
                                     const bf16* __restrict__ k,
                                     const bf16* __restrict__ v,
                                     bf16* __restrict__ out, int positions,
                                     int padded_positions, int n_heads,
                                     int n_kv_heads, int head_dim) {
  int lane = threadIdx.x & 31;
  int item = blockIdx.x * (blockDim.x / kWarp) + (threadIdx.x / kWarp);
  int total = padded_positions * n_heads;
  if (item >= total) return;
  int token = item / n_heads;
  int head = item % n_heads;
  if (token >= positions) {
    for (int d = lane; d < head_dim; d += kWarp) {
      out[(token * n_heads + head) * head_dim + d] = __float2bfloat16(0.0f);
    }
    return;
  }
  int kv_head = head / (n_heads / n_kv_heads);
  const bf16* query = q + (token * n_heads + head) * head_dim;
  float q0 = __bfloat162float(query[lane]);
  float q1 = __bfloat162float(query[lane + kWarp]);
  float maximum = -INFINITY;
  float denominator = 0.0f;
  float accumulator0 = 0.0f;
  float accumulator1 = 0.0f;
  float score_scale = rsqrtf(static_cast<float>(head_dim));
  int kv_dim = n_kv_heads * head_dim;
  for (int source = 0; source <= token; ++source) {
    const bf16* key = k + source * kv_dim + kv_head * head_dim;
    const bf16* value = v + source * kv_dim + kv_head * head_dim;
    float dot = q0 * __bfloat162float(key[lane]);
    dot = fmaf(q1, __bfloat162float(key[lane + kWarp]), dot);
    dot = warp_sum(dot);
    dot = __shfl_sync(0xffffffff, dot, 0);
    float score = dot * score_scale;
    float next_maximum = fmaxf(maximum, score);
    float old_scale = __expf(maximum - next_maximum);
    float new_scale = __expf(score - next_maximum);
    accumulator0 = accumulator0 * old_scale +
                   new_scale * __bfloat162float(value[lane]);
    accumulator1 = accumulator1 * old_scale +
                   new_scale * __bfloat162float(value[lane + kWarp]);
    denominator = denominator * old_scale + new_scale;
    maximum = next_maximum;
  }
  bf16* output = out + (token * n_heads + head) * head_dim;
  output[lane] = __float2bfloat16(accumulator0 / denominator);
  output[lane + kWarp] = __float2bfloat16(accumulator1 / denominator);
}

// ---------------------------- Decode kernels -------------------------------

__global__ void embedding_one_kernel(const bf16* __restrict__ table,
                                     const int* __restrict__ token,
                                     bf16* __restrict__ x, int dim) {
  int id = *token;
  for (int d = threadIdx.x; d < dim; d += blockDim.x) x[d] = table[id * dim + d];
}

// Each warp produces four rows. It loads x once, then streams four coalesced
// weight rows in parallel. This is deliberately a CUDA-core kernel: batch-one
// GEMV is limited by reading 2.3 GiB of weights, not by BF16 arithmetic, and a
// tensor-core tile would manufacture 15 unused rows for every useful row.
__global__ void qkv_decode_kernel(
    const bf16* __restrict__ x, const bf16* __restrict__ norm,
    const float* __restrict__ rms_scale,
    const bf16* __restrict__ wq, const bf16* __restrict__ wk,
    const bf16* __restrict__ wv, bf16* __restrict__ q,
    bf16* __restrict__ key_cache, bf16* __restrict__ value_cache,
    const int* __restrict__ position, int dim, int kv_dim, int max_seq) {
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  int virtual_row = (blockIdx.x * (blockDim.x / kWarp) + warp) * kRowsPerWarp;
  int total_rows = dim + 2 * kv_dim;
  if (virtual_row >= total_rows) return;
  const bf16* weight;
  bf16* output;
  int local_row;
  if (virtual_row < dim) {
    weight = wq; output = q; local_row = virtual_row;
  } else if (virtual_row < dim + kv_dim) {
    weight = wk;
    output = key_cache + (*position) * kv_dim;
    local_row = virtual_row - dim;
  } else {
    weight = wv;
    output = value_cache + (*position) * kv_dim;
    local_row = virtual_row - dim - kv_dim;
  }
  float sums[kRowsPerWarp] = {};
  float scale = *rms_scale;
  for (int column = lane; column < dim; column += kWarp) {
    float input = __bfloat162float(x[column]) * scale *
                  __bfloat162float(norm[column]);
#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) {
      sums[r] = fmaf(__bfloat162float(weight[(local_row + r) * dim + column]),
                     input, sums[r]);
    }
  }
#pragma unroll
  for (int r = 0; r < kRowsPerWarp; ++r) {
    sums[r] = warp_sum(sums[r]);
    if (lane == 0) output[local_row + r] = __float2bfloat16(sums[r]);
  }
}

__global__ void rope_decode_kernel(bf16* __restrict__ q,
                                   bf16* __restrict__ key_cache,
                                   const int* __restrict__ position,
                                   int n_heads, int n_kv_heads,
                                   int head_dim, int kv_dim, float theta) {
  int half = head_dim / 2;
  int q_pairs = n_heads * half;
  int total = q_pairs + n_kv_heads * half;
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= total) return;
  bool is_q = index < q_pairs;
  int logical = is_q ? index : index - q_pairs;
  int pair = logical % half;
  int head = logical / half;
  bf16* data = is_q ? q : key_cache + (*position) * kv_dim;
  int base = head * head_dim + pair;
  float angle = (*position) * llama3_inv_frequency(pair, head_dim, theta);
  float sine, cosine;
  __sincosf(angle, &sine, &cosine);
  float a = __bfloat162float(data[base]);
  float b = __bfloat162float(data[base + half]);
  data[base] = __float2bfloat16(a * cosine - b * sine);
  data[base + half] = __float2bfloat16(a * sine + b * cosine);
}

// Decode FlashAttention splits one head's history across all eight warps in a
// block. Each warp builds a valid online-softmax state for positions
//   warp, warp+8, warp+16, ...
// and the first warp merges those eight states. If stripe r has (m_r,l_r,a_r),
// the common maximum M gives
//   L = sum_r l_r*exp(m_r-M), A = sum_r a_r*exp(m_r-M).
// This is the same associative rescaling identity used inside the streaming
// loop. It exposes context parallelism without materializing a score vector.
__global__ void flash_decode_kernel(const bf16* __restrict__ q,
                                    const bf16* __restrict__ key_cache,
                                    const bf16* __restrict__ value_cache,
                                    bf16* __restrict__ out,
                                    const int* __restrict__ position,
                                    int n_heads, int n_kv_heads,
                                    int head_dim, int kv_dim) {
  __shared__ float stripe_maximum[8];
  __shared__ float stripe_denominator[8];
  __shared__ float stripe_accumulator[8][64];
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  int head = blockIdx.x;
  int kv_head = head / (n_heads / n_kv_heads);
  float q0 = __bfloat162float(q[head * head_dim + lane]);
  float q1 = __bfloat162float(q[head * head_dim + lane + kWarp]);
  float maximum = -INFINITY;
  float denominator = 0.0f;
  float accumulator0 = 0.0f;
  float accumulator1 = 0.0f;
  float score_scale = rsqrtf(static_cast<float>(head_dim));
  int last = *position;
  for (int source = warp; source <= last; source += 8) {
    const bf16* key = key_cache + source * kv_dim + kv_head * head_dim;
    const bf16* value = value_cache + source * kv_dim + kv_head * head_dim;
    float dot = q0 * __bfloat162float(key[lane]);
    dot = fmaf(q1, __bfloat162float(key[lane + kWarp]), dot);
    dot = warp_sum(dot);
    dot = __shfl_sync(0xffffffff, dot, 0);
    float score = dot * score_scale;
    float next_maximum = fmaxf(maximum, score);
    float old_scale = __expf(maximum - next_maximum);
    float new_scale = __expf(score - next_maximum);
    accumulator0 = accumulator0 * old_scale +
                   new_scale * __bfloat162float(value[lane]);
    accumulator1 = accumulator1 * old_scale +
                   new_scale * __bfloat162float(value[lane + kWarp]);
    denominator = denominator * old_scale + new_scale;
    maximum = next_maximum;
  }
  if (lane == 0) {
    stripe_maximum[warp] = maximum;
    stripe_denominator[warp] = denominator;
  }
  stripe_accumulator[warp][lane] = accumulator0;
  stripe_accumulator[warp][lane + kWarp] = accumulator1;
  __syncthreads();

  if (warp == 0) {
    float global_maximum = stripe_maximum[0];
#pragma unroll
    for (int stripe = 1; stripe < 8; ++stripe) {
      global_maximum = fmaxf(global_maximum, stripe_maximum[stripe]);
    }
    float global_denominator = 0.0f;
    float global_accumulator0 = 0.0f;
    float global_accumulator1 = 0.0f;
#pragma unroll
    for (int stripe = 0; stripe < 8; ++stripe) {
      float rescale = __expf(stripe_maximum[stripe] - global_maximum);
      global_denominator += stripe_denominator[stripe] * rescale;
      global_accumulator0 += stripe_accumulator[stripe][lane] * rescale;
      global_accumulator1 += stripe_accumulator[stripe][lane + kWarp] * rescale;
    }
    out[head * head_dim + lane] =
        __float2bfloat16(global_accumulator0 / global_denominator);
    out[head * head_dim + lane + kWarp] =
        __float2bfloat16(global_accumulator1 / global_denominator);
  }
}

__global__ void linear_residual_decode_kernel(
    const bf16* __restrict__ weight, const bf16* __restrict__ input,
    bf16* __restrict__ residual, int rows, int columns) {
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  int row0 = (blockIdx.x * (blockDim.x / kWarp) + warp) * kRowsPerWarp;
  if (row0 >= rows) return;
  float sums[kRowsPerWarp] = {};
  for (int column = lane; column < columns; column += kWarp) {
    float value = __bfloat162float(input[column]);
#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) {
      sums[r] = fmaf(__bfloat162float(weight[(row0 + r) * columns + column]),
                     value, sums[r]);
    }
  }
#pragma unroll
  for (int r = 0; r < kRowsPerWarp; ++r) {
    sums[r] = warp_sum(sums[r]);
    if (lane == 0) {
      float value = sums[r] + __bfloat162float(residual[row0 + r]);
      residual[row0 + r] = __float2bfloat16(value);
    }
  }
}

// Gate and up projections share the normalized input and are immediately
// combined by SiLU in FP32, avoiding both an 8192-element temporary and a launch.
__global__ void gate_up_decode_kernel(
    const bf16* __restrict__ gate_weight, const bf16* __restrict__ up_weight,
    const bf16* __restrict__ x, const bf16* __restrict__ norm,
    const float* __restrict__ rms_scale, bf16* __restrict__ out,
    int rows, int columns) {
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  int row0 = (blockIdx.x * (blockDim.x / kWarp) + warp) * kRowsPerWarp;
  if (row0 >= rows) return;
  float gate[kRowsPerWarp] = {};
  float up[kRowsPerWarp] = {};
  float scale = *rms_scale;
  for (int column = lane; column < columns; column += kWarp) {
    float value = __bfloat162float(x[column]) * scale *
                  __bfloat162float(norm[column]);
#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) {
      gate[r] = fmaf(__bfloat162float(gate_weight[(row0 + r) * columns + column]),
                     value, gate[r]);
      up[r] = fmaf(__bfloat162float(up_weight[(row0 + r) * columns + column]),
                   value, up[r]);
    }
  }
#pragma unroll
  for (int r = 0; r < kRowsPerWarp; ++r) {
    gate[r] = warp_sum(gate[r]);
    up[r] = warp_sum(up[r]);
    if (lane == 0) {
      float activated = gate[r] / (1.0f + __expf(-gate[r]));
      out[row0 + r] = __float2bfloat16(activated * up[r]);
    }
  }
}

__global__ void logits_decode_kernel(
    const bf16* __restrict__ weight, const bf16* __restrict__ x,
    const bf16* __restrict__ norm, const float* __restrict__ rms_scale,
    bf16* __restrict__ logits, int rows, int columns) {
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  int row0 = (blockIdx.x * (blockDim.x / kWarp) + warp) * kRowsPerWarp;
  if (row0 >= rows) return;
  float sums[kRowsPerWarp] = {};
  float scale = *rms_scale;
  for (int column = lane; column < columns; column += kWarp) {
    float value = __bfloat162float(x[column]) * scale *
                  __bfloat162float(norm[column]);
#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) {
      sums[r] = fmaf(__bfloat162float(weight[(row0 + r) * columns + column]),
                     value, sums[r]);
    }
  }
#pragma unroll
  for (int r = 0; r < kRowsPerWarp; ++r) {
    sums[r] = warp_sum(sums[r]);
    if (lane == 0) logits[row0 + r] = __float2bfloat16(sums[r]);
  }
}

// A single block is sufficient for 128k BF16 logits. It records tokens entirely
// on device, which lets one captured CUDA Graph replay without a CPU round-trip.
__global__ void argmax_record_kernel(const bf16* __restrict__ logits, int count,
                                     int* __restrict__ current_token,
                                     int* __restrict__ generated,
                                     int* __restrict__ step,
                                     int* __restrict__ position,
                                     bool advance_position) {
  __shared__ float warp_values[8];
  __shared__ int warp_indices[8];
  float best = -INFINITY;
  int best_index = 0;
  for (int i = threadIdx.x; i < count; i += blockDim.x) {
    float value = __bfloat162float(logits[i]);
    if (value > best) { best = value; best_index = i; }
  }
  for (int offset = 16; offset; offset >>= 1) {
    float other_value = __shfl_down_sync(0xffffffff, best, offset);
    int other_index = __shfl_down_sync(0xffffffff, best_index, offset);
    if (other_value > best) { best = other_value; best_index = other_index; }
  }
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  if (lane == 0) { warp_values[warp] = best; warp_indices[warp] = best_index; }
  __syncthreads();
  if (warp == 0) {
    best = lane < 8 ? warp_values[lane] : -INFINITY;
    best_index = lane < 8 ? warp_indices[lane] : 0;
    for (int offset = 16; offset; offset >>= 1) {
      float other_value = __shfl_down_sync(0xffffffff, best, offset);
      int other_index = __shfl_down_sync(0xffffffff, best_index, offset);
      if (other_value > best) { best = other_value; best_index = other_index; }
    }
    if (lane == 0) {
      *current_token = best_index;
      generated[*step] = best_index;
      ++*step;
      if (advance_position) ++*position;
    }
  }
}

void read_exact(std::ifstream& file, void* destination, size_t bytes) {
  file.read(reinterpret_cast<char*>(destination), static_cast<std::streamsize>(bytes));
  if (!file) throw std::runtime_error("unexpected EOF while reading model");
}

Model load_model(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error("could not open model: " + path);
  Model model;
  read_exact(file, &model.h, sizeof(model.h));
  if (std::memcmp(model.h.magic, "HLLAMA3\0", 8) != 0 || model.h.version != 1) {
    throw std::runtime_error("bad .hllm header");
  }
  std::vector<TensorEntry> entries(model.h.tensor_count);
  read_exact(file, entries.data(), entries.size() * sizeof(TensorEntry));
  for (const TensorEntry& entry : entries) {
    std::string name(entry.name, strnlen(entry.name, kNameBytes));
    std::vector<uint16_t> host(entry.count);
    file.seekg(static_cast<std::streamoff>(entry.offset), std::ios::beg);
    read_exact(file, host.data(), host.size() * sizeof(uint16_t));
    DeviceTensor tensor;
    tensor.count = entry.count;
    CUDA_CHECK(cudaMalloc(&tensor.data, host.size() * sizeof(uint16_t)));
    CUDA_CHECK(cudaMemcpy(tensor.data, host.data(), host.size() * sizeof(uint16_t),
                          cudaMemcpyHostToDevice));
    model.tensors.emplace(name, tensor);
  }
  return model;
}

int div_up(int value, int divisor) { return (value + divisor - 1) / divisor; }
int round_up(int value, int multiple) { return div_up(value, multiple) * multiple; }
std::string layer_prefix(int layer) {
  return "model.layers." + std::to_string(layer);
}

class Runner {
 public:
  Runner(Model&& model, int max_seq, int max_steps)
      : model_(std::move(model)), max_seq_(max_seq), max_steps_(max_steps) {
    const auto& h = model_.h;
    if (h.head_dim != 64) throw std::runtime_error("fast path requires head_dim=64");
    if (h.dim % 64 || h.hidden_dim % 64 || h.vocab_size % 64) {
      throw std::runtime_error("fast path requires dimensions divisible by 64");
    }
    kv_dim_ = h.n_kv_heads * h.head_dim;
    alloc(&d_tokens_, max_seq_);
    alloc(&d_token_, 1); alloc(&d_position_, 1); alloc(&d_step_, 1);
    alloc(&d_generated_, max_steps_);
    activation_seq_ = round_up(max_seq_, 64);
    alloc(&x_, static_cast<size_t>(activation_seq_) * h.dim);
    alloc(&xb_, static_cast<size_t>(activation_seq_) * h.dim);
    alloc(&q_, static_cast<size_t>(activation_seq_) * h.dim);
    alloc(&k_, static_cast<size_t>(activation_seq_) * kv_dim_);
    alloc(&v_, static_cast<size_t>(activation_seq_) * kv_dim_);
    alloc(&attention_, static_cast<size_t>(activation_seq_) * h.dim);
    alloc(&gate_, static_cast<size_t>(activation_seq_) * h.hidden_dim);
    alloc(&up_, static_cast<size_t>(activation_seq_) * h.hidden_dim);
    alloc(&logits_, h.vocab_size);
    alloc(&key_cache_, static_cast<size_t>(h.n_layers) * max_seq_ * kv_dim_);
    alloc(&value_cache_, static_cast<size_t>(h.n_layers) * max_seq_ * kv_dim_);
    alloc(&rms_scale_, 1);
    CUDA_CHECK(cudaStreamCreate(&stream_));
    CUDA_CHECK(cudaEventCreate(&start_));
    CUDA_CHECK(cudaEventCreate(&stop_));
  }

  PerfStats generate(const std::vector<int>& tokens, int steps,
                     std::vector<int>* generated) {
    if (tokens.empty()) throw std::runtime_error("empty prompt");
    if (static_cast<int>(tokens.size()) + steps > max_seq_) {
      throw std::runtime_error("prompt + steps exceeds --max-seq");
    }
    if (steps > max_steps_) throw std::runtime_error("steps exceeds allocation");
    int position = static_cast<int>(tokens.size());
    int step = 0;
    CUDA_CHECK(cudaMemcpyAsync(d_tokens_, tokens.data(), tokens.size() * sizeof(int),
                               cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaMemcpyAsync(d_position_, &position, sizeof(int),
                               cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaMemcpyAsync(d_step_, &step, sizeof(int),
                               cudaMemcpyHostToDevice, stream_));

    CUDA_CHECK(cudaEventRecord(start_, stream_));
    prefill(static_cast<int>(tokens.size()));
    argmax_record_kernel<<<1, kBlock, 0, stream_>>>(
        logits_, model_.h.vocab_size, d_token_, d_generated_, d_step_,
        d_position_, false);
    CUDA_CHECK(cudaEventRecord(stop_, stream_));
    CUDA_CHECK(cudaEventSynchronize(stop_));
    float prefill_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&prefill_ms, start_, stop_));

    if (!graph_exec_) capture_decode_graph();
    CUDA_CHECK(cudaEventRecord(start_, stream_));
    for (int i = 1; i < steps; ++i) {
      CUDA_CHECK(cudaGraphLaunch(graph_exec_, stream_));
    }
    CUDA_CHECK(cudaEventRecord(stop_, stream_));
    CUDA_CHECK(cudaEventSynchronize(stop_));
    float decode_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&decode_ms, start_, stop_));

    if (generated) {
      generated->resize(steps);
      CUDA_CHECK(cudaMemcpy(generated->data(), d_generated_, steps * sizeof(int),
                            cudaMemcpyDeviceToHost));
    }
    return {prefill_ms, decode_ms, static_cast<int>(tokens.size()), steps - 1};
  }

 private:
  template <class T> void alloc(T** pointer, size_t count) {
    CUDA_CHECK(cudaMalloc(pointer, count * sizeof(T)));
  }

  void gemm(const bf16* x, int m, int k,
            const bf16* w0, bf16* y0, int n0,
            const bf16* w1 = nullptr, bf16* y1 = nullptr, int n1 = 0,
            const bf16* w2 = nullptr, bf16* y2 = nullptr, int n2 = 0,
    const bf16* residual = nullptr) {
    int total_n = n0 + n1 + n2;
    if (m <= 128) {
      // More, smaller blocks expose enough parallelism on short prompts.
      dim3 grid(div_up(total_n, 64), div_up(m, 16));
      tensorcore_gemm_kernel<1, 4><<<grid, 128, 0, stream_>>>(
          x, m, k, w0, y0, n0, w1, y1, n1, w2, y2, n2, residual);
    } else {
      // Longer prompts amortize a larger tile that reuses each weight fragment
      // for four 16-token rows before loading the next K slice.
      dim3 grid(div_up(total_n, 128), div_up(m, 64));
      tensorcore_gemm_kernel<4, 8><<<grid, kBlock, 0, stream_>>>(
          x, m, k, w0, y0, n0, w1, y1, n1, w2, y2, n2, residual);
    }
  }

  void prefill(int positions) {
    const auto& h = model_.h;
    int m = round_up(positions, positions <= 128 ? 16 : 64);
    embedding_kernel<<<m, kBlock, 0, stream_>>>(
        model_.tensor("model.embed_tokens.weight"), d_tokens_, x_, positions, m, h.dim);
    for (int layer = 0; layer < h.n_layers; ++layer) {
      std::string p = layer_prefix(layer);
      rmsnorm_kernel<<<m, kBlock, 0, stream_>>>(
          x_, model_.tensor(p + ".input_layernorm.weight"), xb_, h.dim,
          h.rms_norm_eps);
      gemm(xb_, m, h.dim,
           model_.tensor(p + ".self_attn.q_proj.weight"), q_, h.dim,
           model_.tensor(p + ".self_attn.k_proj.weight"), k_, kv_dim_,
           model_.tensor(p + ".self_attn.v_proj.weight"), v_, kv_dim_);
      int rope_items = positions * (h.n_heads + h.n_kv_heads) * (h.head_dim / 2);
      rope_prefill_kernel<<<div_up(rope_items, kBlock), kBlock, 0, stream_>>>(
          q_, k_, positions, h.n_heads, h.n_kv_heads, h.head_dim, h.rope_theta);
      bf16* layer_k = key_cache_ + static_cast<size_t>(layer) * max_seq_ * kv_dim_;
      bf16* layer_v = value_cache_ + static_cast<size_t>(layer) * max_seq_ * kv_dim_;
      copy_prompt_kv_kernel<<<div_up(positions * kv_dim_, kBlock), kBlock, 0, stream_>>>(
          k_, v_, layer_k, layer_v, positions, kv_dim_);
      flash_prefill_kernel<<<div_up(m * h.n_heads, 8), kBlock, 0, stream_>>>(
          q_, k_, v_, attention_, positions, m, h.n_heads, h.n_kv_heads, h.head_dim);
      gemm(attention_, m, h.dim,
           model_.tensor(p + ".self_attn.o_proj.weight"), x_, h.dim,
           nullptr, nullptr, 0, nullptr, nullptr, 0, x_);
      rmsnorm_kernel<<<m, kBlock, 0, stream_>>>(
          x_, model_.tensor(p + ".post_attention_layernorm.weight"), xb_,
          h.dim, h.rms_norm_eps);
      gemm(xb_, m, h.dim,
           model_.tensor(p + ".mlp.gate_proj.weight"), gate_, h.hidden_dim,
           model_.tensor(p + ".mlp.up_proj.weight"), up_, h.hidden_dim);
      silu_mul_kernel<<<div_up(m * h.hidden_dim, kBlock), kBlock, 0, stream_>>>(
          gate_, up_, m * h.hidden_dim);
      gemm(gate_, m, h.hidden_dim,
           model_.tensor(p + ".mlp.down_proj.weight"), x_, h.dim,
           nullptr, nullptr, 0, nullptr, nullptr, 0, x_);
    }
    const bf16* lm = model_.has("lm_head.weight")
                         ? model_.tensor("lm_head.weight")
                         : model_.tensor("model.embed_tokens.weight");
    const bf16* last = x_ + static_cast<size_t>(positions - 1) * h.dim;
    rms_scale_kernel<<<1, kBlock, 0, stream_>>>(
        last, rms_scale_, h.dim, h.rms_norm_eps);
    logits_decode_kernel<<<div_up(h.vocab_size, kRowsPerBlock), kBlock, 0, stream_>>>(
        lm, last, model_.tensor("model.norm.weight"), rms_scale_, logits_,
        h.vocab_size, h.dim);
  }

  void decode_body() {
    const auto& h = model_.h;
    embedding_one_kernel<<<1, kBlock, 0, stream_>>>(
        model_.tensor("model.embed_tokens.weight"), d_token_, x_, h.dim);
    for (int layer = 0; layer < h.n_layers; ++layer) {
      std::string p = layer_prefix(layer);
      bf16* layer_k = key_cache_ + static_cast<size_t>(layer) * max_seq_ * kv_dim_;
      bf16* layer_v = value_cache_ + static_cast<size_t>(layer) * max_seq_ * kv_dim_;
      rms_scale_kernel<<<1, kBlock, 0, stream_>>>(x_, rms_scale_, h.dim,
                                                  h.rms_norm_eps);
      qkv_decode_kernel<<<div_up(h.dim + 2 * kv_dim_, kRowsPerBlock),
                          kBlock, 0, stream_>>>(
          x_, model_.tensor(p + ".input_layernorm.weight"), rms_scale_,
          model_.tensor(p + ".self_attn.q_proj.weight"),
          model_.tensor(p + ".self_attn.k_proj.weight"),
          model_.tensor(p + ".self_attn.v_proj.weight"), q_, layer_k, layer_v,
          d_position_, h.dim, kv_dim_, max_seq_);
      int rope_items = (h.n_heads + h.n_kv_heads) * (h.head_dim / 2);
      rope_decode_kernel<<<div_up(rope_items, kBlock), kBlock, 0, stream_>>>(
          q_, layer_k, d_position_, h.n_heads, h.n_kv_heads, h.head_dim,
          kv_dim_, h.rope_theta);
      flash_decode_kernel<<<h.n_heads, kBlock, 0, stream_>>>(
          q_, layer_k, layer_v, attention_, d_position_, h.n_heads,
          h.n_kv_heads, h.head_dim, kv_dim_);
      linear_residual_decode_kernel<<<div_up(h.dim, kRowsPerBlock),
                                      kBlock, 0, stream_>>>(
          model_.tensor(p + ".self_attn.o_proj.weight"), attention_, x_,
          h.dim, h.dim);
      rms_scale_kernel<<<1, kBlock, 0, stream_>>>(x_, rms_scale_, h.dim,
                                                  h.rms_norm_eps);
      gate_up_decode_kernel<<<div_up(h.hidden_dim, kRowsPerBlock),
                              kBlock, 0, stream_>>>(
          model_.tensor(p + ".mlp.gate_proj.weight"),
          model_.tensor(p + ".mlp.up_proj.weight"), x_,
          model_.tensor(p + ".post_attention_layernorm.weight"), rms_scale_,
          gate_, h.hidden_dim, h.dim);
      linear_residual_decode_kernel<<<div_up(h.dim, kRowsPerBlock),
                                      kBlock, 0, stream_>>>(
          model_.tensor(p + ".mlp.down_proj.weight"), gate_, x_,
          h.dim, h.hidden_dim);
    }
    const bf16* lm = model_.has("lm_head.weight")
                         ? model_.tensor("lm_head.weight")
                         : model_.tensor("model.embed_tokens.weight");
    rms_scale_kernel<<<1, kBlock, 0, stream_>>>(x_, rms_scale_, h.dim,
                                                h.rms_norm_eps);
    logits_decode_kernel<<<div_up(h.vocab_size, kRowsPerBlock),
                           kBlock, 0, stream_>>>(
        lm, x_, model_.tensor("model.norm.weight"), rms_scale_, logits_,
        h.vocab_size, h.dim);
    argmax_record_kernel<<<1, kBlock, 0, stream_>>>(
        logits_, h.vocab_size, d_token_, d_generated_, d_step_, d_position_, true);
  }

  void capture_decode_graph() {
    CUDA_CHECK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal));
    decode_body();
    CUDA_CHECK(cudaStreamEndCapture(stream_, &graph_));
    CUDA_CHECK(cudaGraphInstantiate(&graph_exec_, graph_, 0));
  }

  Model model_;
  int max_seq_, max_steps_, kv_dim_, activation_seq_;
  int *d_tokens_ = nullptr, *d_token_ = nullptr, *d_position_ = nullptr;
  int *d_step_ = nullptr, *d_generated_ = nullptr;
  bf16 *x_ = nullptr, *xb_ = nullptr, *q_ = nullptr, *k_ = nullptr, *v_ = nullptr;
  bf16 *attention_ = nullptr, *gate_ = nullptr, *up_ = nullptr, *logits_ = nullptr;
  bf16 *key_cache_ = nullptr, *value_cache_ = nullptr;
  float* rms_scale_ = nullptr;
  cudaStream_t stream_{};
  cudaEvent_t start_{}, stop_{};
  cudaGraph_t graph_{};
  cudaGraphExec_t graph_exec_{};
};

Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    std::string flag = argv[i];
    auto value = [&]() {
      if (++i >= argc) throw std::runtime_error("missing value for " + flag);
      return std::string(argv[i]);
    };
    if (flag == "--model") args.model_path = value();
    else if (flag == "--tokens") {
      std::string text = value();
      std::replace(text.begin(), text.end(), ',', ' ');
      std::stringstream stream(text);
      int token;
      while (stream >> token) args.tokens.push_back(token);
    } else if (flag == "--prompt-length") args.prompt_length = std::stoi(value());
    else if (flag == "--steps") args.steps = std::stoi(value());
    else if (flag == "--max-seq") args.max_seq = std::stoi(value());
    else if (flag == "--warmup") args.warmup = std::stoi(value());
    else if (flag == "--runs") args.runs = std::stoi(value());
    else if (flag == "--help" || flag == "-h") {
      std::cout << "cuda_llama [--model model.hllm] (--tokens 1,2 | --prompt-length N)\n"
                   "           [--steps 128] [--max-seq 2048] [--warmup 1] [--runs 5]\n";
      std::exit(0);
    } else throw std::runtime_error("unknown argument: " + flag);
  }
  if (args.tokens.empty()) {
    if (args.prompt_length <= 0) throw std::runtime_error("provide --tokens or --prompt-length");
    args.tokens.resize(args.prompt_length);
    args.tokens[0] = 128000;
    for (int i = 1; i < args.prompt_length; ++i) args.tokens[i] = 100 + (i % 30000);
  }
  return args;
}

float median(std::vector<float> values) {
  std::sort(values.begin(), values.end());
  size_t middle = values.size() / 2;
  return values.size() % 2 ? values[middle] : (values[middle - 1] + values[middle]) * 0.5f;
}

}  // namespace

int cuda_llama_main(int argc, char** argv) {
  try {
    Args args = parse_args(argc, argv);
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
    std::cerr << "GPU: " << properties.name << " | SM " << properties.major
              << "." << properties.minor << " | "
              << properties.multiProcessorCount << " SMs\n";
    Model model = load_model(args.model_path);
    std::cerr << "loaded " << args.model_path << " | layers=" << model.h.n_layers
              << " dim=" << model.h.dim << " hidden=" << model.h.hidden_dim
              << " vocab=" << model.h.vocab_size << "\n";
    Runner runner(std::move(model), args.max_seq, args.steps);
    std::vector<float> prefill_rates, decode_rates;
    std::vector<int> generated;
    int total_runs = args.warmup + args.runs;
    for (int run = 0; run < total_runs; ++run) {
      std::vector<int>* output = run == total_runs - 1 ? &generated : nullptr;
      PerfStats stats = runner.generate(args.tokens, args.steps, output);
      float prefill_rate = 1000.0f * stats.prefill_tokens / stats.prefill_ms;
      float decode_rate = stats.decode_tokens > 0
                              ? 1000.0f * stats.decode_tokens / stats.decode_ms
                              : 0.0f;
      if (run >= args.warmup) {
        prefill_rates.push_back(prefill_rate);
        decode_rates.push_back(decode_rate);
        std::cerr << "run " << (run - args.warmup + 1) << ": prefill "
                  << std::fixed << std::setprecision(1) << prefill_rate
                  << " tok/s, decode " << decode_rate << " tok/s\n";
      }
    }
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "PREFILL_TOKENS: " << args.tokens.size() << "\n";
    std::cout << "PREFILL_TOKENS_PER_SECOND: " << median(prefill_rates) << "\n";
    std::cout << "DECODE_TOKENS: " << (args.steps - 1) << "\n";
    std::cout << "DECODE_TOKENS_PER_SECOND: " << median(decode_rates) << "\n";
    std::cout << "GENERATED_TOKENS: ";
    for (size_t i = 0; i < generated.size(); ++i) {
      if (i) std::cout << ',';
      std::cout << generated[i];
    }
    std::cout << "\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << "\n";
    return 1;
  }
}
