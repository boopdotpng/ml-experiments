#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>

#define HIP_CHECK(expr)                                                         \
  do {                                                                         \
    hipError_t err__ = (expr);                                                  \
    if (err__ != hipSuccess) {                                                  \
      std::ostringstream os__;                                                  \
      os__ << "HIP error at " << __FILE__ << ":" << __LINE__ << ": "          \
           << hipGetErrorString(err__);                                         \
      throw std::runtime_error(os__.str());                                     \
    }                                                                          \
  } while (0)

namespace {

constexpr int kBlock = 256;
constexpr int kNameBytes = 96;

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
  uint16_t* data = nullptr;
  uint64_t count = 0;
};

struct Model {
  HllmHeader h{};
  std::map<std::string, DeviceTensor> tensors;

  const uint16_t* tensor(const std::string& name) const {
    auto it = tensors.find(name);
    if (it == tensors.end()) {
      throw std::runtime_error("missing tensor: " + name);
    }
    return it->second.data;
  }

  bool has(const std::string& name) const {
    return tensors.find(name) != tensors.end();
  }
};

struct Args {
  std::string model_path = "models/llama3.2-1b-instruct.hllm";
  std::string mode = "kv";
  std::vector<int> tokens;
  int steps = 32;
  int max_seq = 512;
};

__device__ __forceinline__ float bf16_to_float(uint16_t bits) {
  return __uint_as_float(static_cast<uint32_t>(bits) << 16);
}

__global__ void embedding_kernel(const uint16_t* __restrict__ emb,
                                 const int* __restrict__ tokens,
                                 float* __restrict__ x, int dim) {
  int pos = blockIdx.x;
  int token = tokens[pos];
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    x[pos * dim + i] = bf16_to_float(emb[token * dim + i]);
  }
}

__global__ void rmsnorm_kernel(const float* __restrict__ x,
                               const uint16_t* __restrict__ weight,
                               float* __restrict__ out, int dim, float eps) {
  int b = blockIdx.x;
  __shared__ float sh[kBlock];
  float sum = 0.0f;
  const float* xb = x + b * dim;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    float v = xb[i];
    sum += v * v;
  }
  sh[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) sh[threadIdx.x] += sh[threadIdx.x + stride];
    __syncthreads();
  }
  float scale = rsqrtf(sh[0] / dim + eps);
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    out[b * dim + i] = xb[i] * scale * bf16_to_float(weight[i]);
  }
}

__global__ void matvec_bf16_kernel(const uint16_t* __restrict__ w,
                                   const float* __restrict__ x,
                                   float* __restrict__ y, int rows, int cols) {
  int row = blockIdx.x;
  int batch = blockIdx.y;
  __shared__ float sh[kBlock];
  float sum = 0.0f;
  const uint16_t* wr = w + row * cols;
  const float* xb = x + batch * cols;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) {
    sum += bf16_to_float(wr[i]) * xb[i];
  }
  sh[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) sh[threadIdx.x] += sh[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) y[batch * rows + row] = sh[0];
}

__global__ void add_kernel(float* __restrict__ x, const float* __restrict__ y,
                           int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] += y[i];
}

__global__ void silu_mul_kernel(float* __restrict__ gate,
                                const float* __restrict__ up, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float v = gate[i];
    gate[i] = (v / (1.0f + expf(-v))) * up[i];
  }
}

__global__ void rope_kernel(float* __restrict__ x, int positions, int heads,
                            int head_dim, int pos_offset, float theta) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int half = head_dim / 2;
  int total = positions * heads * half;
  if (idx >= total) return;

  int pair = idx % half;
  int h = (idx / half) % heads;
  int pos = idx / (half * heads);
  int abs_pos = pos + pos_offset;
  int base = (pos * heads + h) * head_dim + pair * 2;

  float inv_freq = expf(-logf(theta) * (2.0f * pair / head_dim));
  float angle = abs_pos * inv_freq;
  float c = cosf(angle);
  float s = sinf(angle);
  float a = x[base];
  float b = x[base + 1];
  x[base] = a * c - b * s;
  x[base + 1] = a * s + b * c;
}

__global__ void attention_scores_kernel(const float* __restrict__ q,
                                        const float* __restrict__ k,
                                        float* __restrict__ scores, int T,
                                        int n_heads, int n_kv_heads,
                                        int head_dim) {
  int s = blockIdx.x;
  int t = blockIdx.y;
  int h = blockIdx.z;
  __shared__ float sh[kBlock];

  float dot = 0.0f;
  if (s <= t) {
    int group = n_heads / n_kv_heads;
    int kvh = h / group;
    const float* qv = q + (t * n_heads + h) * head_dim;
    const float* kv = k + (s * n_kv_heads + kvh) * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
      dot += qv[d] * kv[d];
    }
  }
  sh[threadIdx.x] = dot;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) sh[threadIdx.x] += sh[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    scores[(t * n_heads + h) * T + s] =
        (s <= t) ? sh[0] * rsqrtf(static_cast<float>(head_dim)) : -1.0e30f;
  }
}

__global__ void softmax_rows_kernel(float* __restrict__ scores, int T,
                                    int n_heads) {
  int t = blockIdx.x;
  int h = blockIdx.y;
  float* row = scores + (t * n_heads + h) * T;
  __shared__ float sh[kBlock];

  float mx = -1.0e30f;
  for (int s = threadIdx.x; s < T; s += blockDim.x) mx = fmaxf(mx, row[s]);
  sh[threadIdx.x] = mx;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + stride]);
    __syncthreads();
  }
  mx = sh[0];

  float sum = 0.0f;
  for (int s = threadIdx.x; s < T; s += blockDim.x) {
    float e = expf(row[s] - mx);
    row[s] = e;
    sum += e;
  }
  sh[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) sh[threadIdx.x] += sh[threadIdx.x + stride];
    __syncthreads();
  }
  float inv = 1.0f / sh[0];
  for (int s = threadIdx.x; s < T; s += blockDim.x) row[s] *= inv;
}

__global__ void attention_apply_kernel(const float* __restrict__ probs,
                                       const float* __restrict__ v,
                                       float* __restrict__ out, int T,
                                       int n_heads, int n_kv_heads,
                                       int head_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = T * n_heads * head_dim;
  if (idx >= total) return;
  int d = idx % head_dim;
  int h = (idx / head_dim) % n_heads;
  int t = idx / (head_dim * n_heads);
  int group = n_heads / n_kv_heads;
  int kvh = h / group;

  float acc = 0.0f;
  const float* row = probs + (t * n_heads + h) * T;
  for (int s = 0; s <= t; ++s) {
    acc += row[s] * v[(s * n_kv_heads + kvh) * head_dim + d];
  }
  out[idx] = acc;
}

__global__ void store_kv_cache_kernel(const float* __restrict__ k,
                                      const float* __restrict__ v,
                                      float* __restrict__ kc,
                                      float* __restrict__ vc, int layer,
                                      int pos, int max_seq, int kv_dim) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= kv_dim) return;
  int off = (layer * max_seq + pos) * kv_dim + i;
  kc[off] = k[i];
  vc[off] = v[i];
}

__global__ void kv_attention_kernel(const float* __restrict__ q,
                                    const float* __restrict__ kc,
                                    const float* __restrict__ vc,
                                    float* __restrict__ out, int layer,
                                    int pos, int max_seq, int n_heads,
                                    int n_kv_heads, int head_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = n_heads * head_dim;
  if (idx >= total) return;
  int d = idx % head_dim;
  int h = idx / head_dim;
  int group = n_heads / n_kv_heads;
  int kvh = h / group;
  int kv_dim = n_kv_heads * head_dim;
  const float scale = rsqrtf(static_cast<float>(head_dim));

  float mx = -1.0e30f;
  for (int s = 0; s <= pos; ++s) {
    const float* key = kc + (layer * max_seq + s) * kv_dim + kvh * head_dim;
    float dot = 0.0f;
    for (int j = 0; j < head_dim; ++j) dot += q[h * head_dim + j] * key[j];
    mx = fmaxf(mx, dot * scale);
  }

  float denom = 0.0f;
  float acc = 0.0f;
  for (int s = 0; s <= pos; ++s) {
    const float* key = kc + (layer * max_seq + s) * kv_dim + kvh * head_dim;
    const float* val = vc + (layer * max_seq + s) * kv_dim + kvh * head_dim;
    float dot = 0.0f;
    for (int j = 0; j < head_dim; ++j) dot += q[h * head_dim + j] * key[j];
    float p = expf(dot * scale - mx);
    denom += p;
    acc += p * val[d];
  }
  out[idx] = acc / denom;
}

__global__ void flash_decode_attention_kernel(const float* __restrict__ q,
                                              const float* __restrict__ kc,
                                              const float* __restrict__ vc,
                                              float* __restrict__ out,
                                              int layer, int pos, int max_seq,
                                              int n_heads, int n_kv_heads,
                                              int head_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = n_heads * head_dim;
  if (idx >= total) return;
  int d = idx % head_dim;
  int h = idx / head_dim;
  int group = n_heads / n_kv_heads;
  int kvh = h / group;
  int kv_dim = n_kv_heads * head_dim;
  const float scale = rsqrtf(static_cast<float>(head_dim));

  float m = -1.0e30f;
  float l = 0.0f;
  float acc = 0.0f;
  for (int s = 0; s <= pos; ++s) {
    const float* key = kc + (layer * max_seq + s) * kv_dim + kvh * head_dim;
    const float* val = vc + (layer * max_seq + s) * kv_dim + kvh * head_dim;
    float dot = 0.0f;
    for (int j = 0; j < head_dim; ++j) dot += q[h * head_dim + j] * key[j];
    float score = dot * scale;
    float next_m = fmaxf(m, score);
    float old_scale = expf(m - next_m);
    float new_scale = expf(score - next_m);
    acc = acc * old_scale + new_scale * val[d];
    l = l * old_scale + new_scale;
    m = next_m;
  }
  out[idx] = acc / l;
}

void read_exact(std::ifstream& f, void* dst, size_t n) {
  f.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(n));
  if (!f) throw std::runtime_error("unexpected EOF while reading model");
}

Model load_model(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("could not open model: " + path);

  Model m;
  read_exact(f, &m.h, sizeof(m.h));
  if (std::memcmp(m.h.magic, "HLLAMA3\0", 8) != 0 || m.h.version != 1) {
    throw std::runtime_error("bad .hllm header");
  }

  std::vector<TensorEntry> entries(m.h.tensor_count);
  read_exact(f, entries.data(), entries.size() * sizeof(TensorEntry));

  for (const TensorEntry& e : entries) {
    std::string name(e.name, strnlen(e.name, kNameBytes));
    std::vector<uint16_t> host(e.count);
    f.seekg(static_cast<std::streamoff>(e.offset), std::ios::beg);
    read_exact(f, host.data(), host.size() * sizeof(uint16_t));

    DeviceTensor dt;
    dt.count = e.count;
    HIP_CHECK(hipMalloc(&dt.data, host.size() * sizeof(uint16_t)));
    HIP_CHECK(hipMemcpy(dt.data, host.data(), host.size() * sizeof(uint16_t),
                        hipMemcpyHostToDevice));
    m.tensors.emplace(name, dt);
  }

  return m;
}

std::string layer_prefix(int layer) {
  return "model.layers." + std::to_string(layer);
}

int div_up(int a, int b) { return (a + b - 1) / b; }

void launch_matvec(const uint16_t* w, const float* x, float* y, int rows,
                   int cols, int batch) {
  dim3 grid(rows, batch);
  matvec_bf16_kernel<<<grid, kBlock>>>(w, x, y, rows, cols);
  HIP_CHECK(hipGetLastError());
}

void launch_rms(const float* x, const uint16_t* w, float* y, int dim, int batch,
                float eps) {
  rmsnorm_kernel<<<batch, kBlock>>>(x, w, y, dim, eps);
  HIP_CHECK(hipGetLastError());
}

class Runner {
 public:
  Runner(Model&& model, int max_seq) : model_(std::move(model)), max_seq_(max_seq) {
    const auto& h = model_.h;
    if (max_seq_ > h.model_max_seq) max_seq_ = h.model_max_seq;
    kv_dim_ = h.n_kv_heads * h.head_dim;

    alloc(&d_tokens_, max_seq_);
    alloc(&x_, max_seq_ * h.dim);
    alloc(&xb_, max_seq_ * h.dim);
    alloc(&q_, max_seq_ * h.dim);
    alloc(&k_, max_seq_ * kv_dim_);
    alloc(&v_, max_seq_ * kv_dim_);
    alloc(&att_, max_seq_ * h.dim);
    alloc(&proj_, max_seq_ * h.dim);
    alloc(&h1_, max_seq_ * h.hidden_dim);
    alloc(&h2_, max_seq_ * h.hidden_dim);
    alloc(&scores_, max_seq_ * h.n_heads * max_seq_);
    alloc(&logits_, h.vocab_size);
    alloc(&kc_, h.n_layers * max_seq_ * kv_dim_);
    alloc(&vc_, h.n_layers * max_seq_ * kv_dim_);
  }

  std::vector<int> generate_naive(std::vector<int> tokens, int steps) {
    for (int i = 0; i < steps; ++i) {
      if (static_cast<int>(tokens.size()) >= max_seq_) break;
      forward_full(tokens);
      int next = argmax_logits();
      tokens.push_back(next);
      if (next == model_.h.eos_id) break;
    }
    return tokens;
  }

  std::vector<int> generate_cached(std::vector<int> tokens, int steps,
                                   bool flash) {
    if (tokens.empty()) throw std::runtime_error("no input tokens");
    if (static_cast<int>(tokens.size()) + steps > max_seq_) {
      throw std::runtime_error("tokens + steps exceeds --max-seq");
    }

    HIP_CHECK(hipMemset(kc_, 0, model_.h.n_layers * max_seq_ * kv_dim_ * sizeof(float)));
    HIP_CHECK(hipMemset(vc_, 0, model_.h.n_layers * max_seq_ * kv_dim_ * sizeof(float)));

    int pos = 0;
    for (; pos < static_cast<int>(tokens.size()); ++pos) {
      decode_one(tokens[pos], pos, flash);
    }

    for (int i = 0; i < steps; ++i) {
      int next = argmax_logits();
      tokens.push_back(next);
      if (next == model_.h.eos_id || static_cast<int>(tokens.size()) >= max_seq_) break;
      decode_one(next, pos, flash);
      ++pos;
    }
    return tokens;
  }

 private:
  template <typename T>
  void alloc(T** ptr, size_t count) {
    HIP_CHECK(hipMalloc(ptr, count * sizeof(T)));
  }

  void forward_full(const std::vector<int>& tokens) {
    const auto& h = model_.h;
    int T = static_cast<int>(tokens.size());
    if (T > max_seq_) throw std::runtime_error("input exceeds --max-seq");

    HIP_CHECK(hipMemcpy(d_tokens_, tokens.data(), T * sizeof(int), hipMemcpyHostToDevice));
    embedding_kernel<<<T, kBlock>>>(model_.tensor("model.embed_tokens.weight"), d_tokens_, x_, h.dim);
    HIP_CHECK(hipGetLastError());

    for (int layer = 0; layer < h.n_layers; ++layer) {
      std::string p = layer_prefix(layer);
      launch_rms(x_, model_.tensor(p + ".input_layernorm.weight"), xb_, h.dim, T, h.rms_norm_eps);
      launch_matvec(model_.tensor(p + ".self_attn.q_proj.weight"), xb_, q_, h.dim, h.dim, T);
      launch_matvec(model_.tensor(p + ".self_attn.k_proj.weight"), xb_, k_, kv_dim_, h.dim, T);
      launch_matvec(model_.tensor(p + ".self_attn.v_proj.weight"), xb_, v_, kv_dim_, h.dim, T);
      rope_kernel<<<div_up(T * h.n_heads * (h.head_dim / 2), kBlock), kBlock>>>(
          q_, T, h.n_heads, h.head_dim, 0, h.rope_theta);
      rope_kernel<<<div_up(T * h.n_kv_heads * (h.head_dim / 2), kBlock), kBlock>>>(
          k_, T, h.n_kv_heads, h.head_dim, 0, h.rope_theta);
      HIP_CHECK(hipGetLastError());

      dim3 score_grid(T, T, h.n_heads);
      attention_scores_kernel<<<score_grid, kBlock>>>(q_, k_, scores_, T, h.n_heads,
                                                      h.n_kv_heads, h.head_dim);
      dim3 softmax_grid(T, h.n_heads);
      softmax_rows_kernel<<<softmax_grid, kBlock>>>(scores_, T, h.n_heads);
      attention_apply_kernel<<<div_up(T * h.dim, kBlock), kBlock>>>(
          scores_, v_, att_, T, h.n_heads, h.n_kv_heads, h.head_dim);
      HIP_CHECK(hipGetLastError());

      launch_matvec(model_.tensor(p + ".self_attn.o_proj.weight"), att_, proj_, h.dim, h.dim, T);
      add_kernel<<<div_up(T * h.dim, kBlock), kBlock>>>(x_, proj_, T * h.dim);

      launch_rms(x_, model_.tensor(p + ".post_attention_layernorm.weight"), xb_, h.dim, T, h.rms_norm_eps);
      launch_matvec(model_.tensor(p + ".mlp.gate_proj.weight"), xb_, h1_, h.hidden_dim, h.dim, T);
      launch_matvec(model_.tensor(p + ".mlp.up_proj.weight"), xb_, h2_, h.hidden_dim, h.dim, T);
      silu_mul_kernel<<<div_up(T * h.hidden_dim, kBlock), kBlock>>>(h1_, h2_, T * h.hidden_dim);
      launch_matvec(model_.tensor(p + ".mlp.down_proj.weight"), h1_, proj_, h.dim, h.hidden_dim, T);
      add_kernel<<<div_up(T * h.dim, kBlock), kBlock>>>(x_, proj_, T * h.dim);
      HIP_CHECK(hipGetLastError());
    }

    launch_rms(x_, model_.tensor("model.norm.weight"), xb_, h.dim, T, h.rms_norm_eps);
    const uint16_t* lm = model_.has("lm_head.weight") ? model_.tensor("lm_head.weight")
                                                      : model_.tensor("model.embed_tokens.weight");
    launch_matvec(lm, xb_ + (T - 1) * h.dim, logits_, h.vocab_size, h.dim, 1);
    HIP_CHECK(hipDeviceSynchronize());
  }

  void decode_one(int token, int pos, bool flash) {
    const auto& h = model_.h;
    HIP_CHECK(hipMemcpy(d_tokens_, &token, sizeof(int), hipMemcpyHostToDevice));
    embedding_kernel<<<1, kBlock>>>(model_.tensor("model.embed_tokens.weight"), d_tokens_, x_, h.dim);
    HIP_CHECK(hipGetLastError());

    for (int layer = 0; layer < h.n_layers; ++layer) {
      std::string p = layer_prefix(layer);
      launch_rms(x_, model_.tensor(p + ".input_layernorm.weight"), xb_, h.dim, 1, h.rms_norm_eps);
      launch_matvec(model_.tensor(p + ".self_attn.q_proj.weight"), xb_, q_, h.dim, h.dim, 1);
      launch_matvec(model_.tensor(p + ".self_attn.k_proj.weight"), xb_, k_, kv_dim_, h.dim, 1);
      launch_matvec(model_.tensor(p + ".self_attn.v_proj.weight"), xb_, v_, kv_dim_, h.dim, 1);
      rope_kernel<<<div_up(h.n_heads * (h.head_dim / 2), kBlock), kBlock>>>(
          q_, 1, h.n_heads, h.head_dim, pos, h.rope_theta);
      rope_kernel<<<div_up(h.n_kv_heads * (h.head_dim / 2), kBlock), kBlock>>>(
          k_, 1, h.n_kv_heads, h.head_dim, pos, h.rope_theta);
      store_kv_cache_kernel<<<div_up(kv_dim_, kBlock), kBlock>>>(k_, v_, kc_, vc_, layer,
                                                                 pos, max_seq_, kv_dim_);
      if (flash) {
        flash_decode_attention_kernel<<<div_up(h.dim, kBlock), kBlock>>>(
            q_, kc_, vc_, att_, layer, pos, max_seq_, h.n_heads, h.n_kv_heads, h.head_dim);
      } else {
        kv_attention_kernel<<<div_up(h.dim, kBlock), kBlock>>>(
            q_, kc_, vc_, att_, layer, pos, max_seq_, h.n_heads, h.n_kv_heads, h.head_dim);
      }
      HIP_CHECK(hipGetLastError());

      launch_matvec(model_.tensor(p + ".self_attn.o_proj.weight"), att_, proj_, h.dim, h.dim, 1);
      add_kernel<<<div_up(h.dim, kBlock), kBlock>>>(x_, proj_, h.dim);

      launch_rms(x_, model_.tensor(p + ".post_attention_layernorm.weight"), xb_, h.dim, 1, h.rms_norm_eps);
      launch_matvec(model_.tensor(p + ".mlp.gate_proj.weight"), xb_, h1_, h.hidden_dim, h.dim, 1);
      launch_matvec(model_.tensor(p + ".mlp.up_proj.weight"), xb_, h2_, h.hidden_dim, h.dim, 1);
      silu_mul_kernel<<<div_up(h.hidden_dim, kBlock), kBlock>>>(h1_, h2_, h.hidden_dim);
      launch_matvec(model_.tensor(p + ".mlp.down_proj.weight"), h1_, proj_, h.dim, h.hidden_dim, 1);
      add_kernel<<<div_up(h.dim, kBlock), kBlock>>>(x_, proj_, h.dim);
      HIP_CHECK(hipGetLastError());
    }

    launch_rms(x_, model_.tensor("model.norm.weight"), xb_, h.dim, 1, h.rms_norm_eps);
    const uint16_t* lm = model_.has("lm_head.weight") ? model_.tensor("lm_head.weight")
                                                      : model_.tensor("model.embed_tokens.weight");
    launch_matvec(lm, xb_, logits_, h.vocab_size, h.dim, 1);
    HIP_CHECK(hipDeviceSynchronize());
  }

  int argmax_logits() {
    std::vector<float> host(model_.h.vocab_size);
    HIP_CHECK(hipMemcpy(host.data(), logits_, host.size() * sizeof(float), hipMemcpyDeviceToHost));
    return static_cast<int>(std::max_element(host.begin(), host.end()) - host.begin());
  }

  Model model_;
  int max_seq_ = 0;
  int kv_dim_ = 0;
  int* d_tokens_ = nullptr;
  float *x_ = nullptr, *xb_ = nullptr, *q_ = nullptr, *k_ = nullptr, *v_ = nullptr;
  float *att_ = nullptr, *proj_ = nullptr, *h1_ = nullptr, *h2_ = nullptr;
  float *scores_ = nullptr, *logits_ = nullptr, *kc_ = nullptr, *vc_ = nullptr;
};

Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto need = [&](const char* flag) -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + flag);
      return argv[++i];
    };
    if (a == "--model") args.model_path = need("--model");
    else if (a == "--tokens") {
      std::string s = need("--tokens");
      std::replace(s.begin(), s.end(), ',', ' ');
      std::stringstream ss(s);
      int id;
      while (ss >> id) args.tokens.push_back(id);
    } else if (a == "--steps") args.steps = std::stoi(need("--steps"));
    else if (a == "--mode") args.mode = need("--mode");
    else if (a == "--max-seq") args.max_seq = std::stoi(need("--max-seq"));
    else if (a == "--help" || a == "-h") {
      std::cout << "usage: hip_llama --model model.hllm --tokens 128000,9906 --steps 16 "
                   "--mode naive|kv|flash [--max-seq 512]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + a);
    }
  }
  if (args.tokens.empty()) throw std::runtime_error("--tokens is required");
  if (args.mode != "naive" && args.mode != "kv" && args.mode != "flash") {
    throw std::runtime_error("--mode must be naive, kv, or flash");
  }
  return args;
}

void print_tokens(const char* label, const std::vector<int>& tokens) {
  std::cout << label << ": ";
  for (size_t i = 0; i < tokens.size(); ++i) {
    if (i) std::cout << ",";
    std::cout << tokens[i];
  }
  std::cout << "\n";
}

}  // namespace

int hip_llama_main(int argc, char** argv) {
  try {
    Args args = parse_args(argc, argv);
    auto start = std::chrono::steady_clock::now();
    Model model = load_model(args.model_path);
    std::cerr << "loaded " << args.model_path << " | layers=" << model.h.n_layers
              << " dim=" << model.h.dim << " heads=" << model.h.n_heads
              << " kv_heads=" << model.h.n_kv_heads << " vocab=" << model.h.vocab_size
              << "\n";
    Runner runner(std::move(model), args.max_seq);

    std::vector<int> out;
    if (args.mode == "naive") out = runner.generate_naive(args.tokens, args.steps);
    else out = runner.generate_cached(args.tokens, args.steps, args.mode == "flash");

    auto end = std::chrono::steady_clock::now();
    double sec = std::chrono::duration<double>(end - start).count();
    print_tokens("INPUT_TOKENS", args.tokens);
    print_tokens("ALL_TOKENS", out);
    std::cout << "GENERATED_COUNT: " << (out.size() - args.tokens.size()) << "\n";
    std::cout << "SECONDS: " << sec << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
