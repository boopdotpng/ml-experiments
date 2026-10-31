// GENERATED ordinary HIP for 1b: all dimensions below are fixed literals.
// The 1B version is also saved as src/kernels.hip for reading in VS Code.
// No templates, transpose flags, or generic GEMM helper.
// Readable FP32 training kernels. Each forward is adjacent to its backward.
// Function bodies used by the Python emitter. Read src/kernels.hip for fixed 1B values.
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstddef>

constexpr int BLOCK = 256;

// Reduce across all 256 threads using shared memory and block-wide barriers.
__device__ float block_sum(float v, float* scratch) {
  int lane = threadIdx.x;
  scratch[lane] = v;
  __syncthreads();
  for (int stride = BLOCK / 2; stride; stride /= 2) {
    if (lane < stride) scratch[lane] += scratch[lane + stride];
    __syncthreads();
  }
  float result = scratch[0];
  __syncthreads(); // All readers finish before scratch is reused.
  return result;
}
__device__ float block_max(float v, float* scratch) {
  int lane = threadIdx.x;
  scratch[lane] = v;
  __syncthreads();
  for (int stride = BLOCK / 2; stride; stride /= 2) {
    if (lane < stride) scratch[lane] = fmaxf(scratch[lane], scratch[lane + stride]);
    __syncthreads();
  }
  float result = scratch[0];
  __syncthreads();
  return result;
}

constexpr int HEAD_DIM = 64;
constexpr int POSITION_OFFSET = 0;

__device__ double rope_frequency(int pair) {
  // Scalar angle math uses FP64: tiny frequency errors grow with position.
  // Activations and stored gradients remain FP32.
  double f = pow(500000.0, -2.0 * pair / HEAD_DIM);
  double wavelength = 6.2831853071795864769 / f;
  if (wavelength > 8192.0) return f / 32.0;
  if (wavelength < 2048.0) return f;
  double smooth = (8192.0 / wavelength - 1.0) / 3.0;
  return (1.0 - smooth) * f / 32.0 + smooth * f;
}


// embedding_forward: tokens=8, dim=2048
__global__ void embedding_forward(const float* table, const int* ids, float* x) {
  const int tokens = 8;
  const int dim = 2048;

  int i = blockIdx.x * BLOCK + threadIdx.x;
  if (i < tokens * dim) x[i] = table[ids[i / dim] * dim + i % dim];
}

// embedding_backward: tokens=8, dim=2048
__global__ void embedding_backward(const float* dx, const int* ids, float* dtable) {
  const int tokens = 8;
  const int dim = 2048;

  int i = blockIdx.x * BLOCK + threadIdx.x;
  // Repeated IDs collide. Also ACCUMULATE onto the output-head gradient.
  if (i < tokens * dim) atomicAdd(dtable + ids[i / dim] * dim + i % dim, dx[i]);
}

// rmsnorm_forward: rows=8, dim=2048
__global__ void rmsnorm_forward(const float* x, const float* w, float* y, float* r) {
  const int rows = 8;
  const int dim = 2048;

  __shared__ float scratch[BLOCK];
  int row = blockIdx.x;
  float sum = 0;
  for (int d = threadIdx.x; d < dim; d += BLOCK)
    sum += x[row * dim + d] * x[row * dim + d];
  float inv = rsqrtf(block_sum(sum, scratch) / dim + 1e-5f);
  if (threadIdx.x == 0) r[row] = inv;
  for (int d = threadIdx.x; d < dim; d += BLOCK)
    y[row * dim + d] = x[row * dim + d] * inv * w[d];
}

// rmsnorm_backward_input: rows=8, dim=2048
__global__ void rmsnorm_backward_input(const float* x, const float* w, const float* r,
                                       const float* dy, float* dx) {
  const int rows = 8;
  const int dim = 2048;

  __shared__ float scratch[BLOCK];
  int row = blockIdx.x;
  float dot = 0;
  for (int d = threadIdx.x; d < dim; d += BLOCK)
    dot += dy[row * dim + d] * w[d] * x[row * dim + d];
  dot = block_sum(dot, scratch) / dim;
  float inv = r[row];
  for (int d = threadIdx.x; d < dim; d += BLOCK) {
    int i = row * dim + d;
    dx[i] += inv * w[d] * dy[i] - x[i] * inv * inv * inv * dot;
  }
}

// rmsnorm_backward_weight: rows=8, dim=2048
__global__ void rmsnorm_backward_weight(const float* x, const float* r, const float* dy,
                                        float* dw) {
  const int rows = 8;
  const int dim = 2048;

  int d = blockIdx.x * BLOCK + threadIdx.x;
  if (d >= dim) return;
  float sum = 0;
  for (int row = 0; row < rows; ++row)
    sum += dy[row * dim + d] * x[row * dim + d] * r[row];
  dw[d] += sum;
}

// linear_forward_hidden: tokens=8, output_dim=2048, input_dim=2048
__global__ void linear_forward_hidden(const float* x, const float* w, float* y) {
  const int tokens = 8;
  const int output_dim = 2048;
  const int input_dim = 2048;
  // y[token, out] = sum_in x[token, in] * w[out, in].

  int token = blockIdx.y * blockDim.y + threadIdx.y;
  int out = blockIdx.x * blockDim.x + threadIdx.x;
  if (token >= tokens || out >= output_dim) return;

  float sum = 0.0f;
  for (int in = 0; in < input_dim; ++in) {
    sum += x[token * input_dim + in] * w[out * input_dim + in];
  }
  y[token * output_dim + out] = sum;
}

// linear_backward_input_hidden: tokens=8, output_dim=2048, input_dim=2048
__global__ void linear_backward_input_hidden(const float* dy, const float* w, float* dx) {
  const int tokens = 8;
  const int output_dim = 2048;
  const int input_dim = 2048;
  // dx[token, in] += sum_out dy[token, out] * w[out, in].

  int token = blockIdx.y * blockDim.y + threadIdx.y;
  int in = blockIdx.x * blockDim.x + threadIdx.x;
  if (token >= tokens || in >= input_dim) return;

  float sum = 0.0f;
  for (int out = 0; out < output_dim; ++out) {
    sum += dy[token * output_dim + out] * w[out * input_dim + in];
  }
  dx[token * input_dim + in] += sum;
}

// linear_backward_weight_hidden: tokens=8, output_dim=2048, input_dim=2048
__global__ void linear_backward_weight_hidden(const float* dy, const float* x, float* dw) {
  const int tokens = 8;
  const int output_dim = 2048;
  const int input_dim = 2048;
  // dw[out, in] += sum_token dy[token, out] * x[token, in].

  int out = blockIdx.y * blockDim.y + threadIdx.y;
  int in = blockIdx.x * blockDim.x + threadIdx.x;
  if (out >= output_dim || in >= input_dim) return;

  float sum = 0.0f;
  for (int token = 0; token < tokens; ++token) {
    sum += dy[token * output_dim + out] * x[token * input_dim + in];
  }
  dw[out * input_dim + in] += sum;
}

// linear_forward_kv: tokens=8, output_dim=512, input_dim=2048
__global__ void linear_forward_kv(const float* x, const float* w, float* y) {
  const int tokens = 8;
  const int output_dim = 512;
  const int input_dim = 2048;
  // y[token, out] = sum_in x[token, in] * w[out, in].

  int token = blockIdx.y * blockDim.y + threadIdx.y;
  int out = blockIdx.x * blockDim.x + threadIdx.x;
  if (token >= tokens || out >= output_dim) return;

  float sum = 0.0f;
  for (int in = 0; in < input_dim; ++in) {
    sum += x[token * input_dim + in] * w[out * input_dim + in];
  }
  y[token * output_dim + out] = sum;
}

// linear_backward_input_kv: tokens=8, output_dim=512, input_dim=2048
__global__ void linear_backward_input_kv(const float* dy, const float* w, float* dx) {
  const int tokens = 8;
  const int output_dim = 512;
  const int input_dim = 2048;
  // dx[token, in] += sum_out dy[token, out] * w[out, in].

  int token = blockIdx.y * blockDim.y + threadIdx.y;
  int in = blockIdx.x * blockDim.x + threadIdx.x;
  if (token >= tokens || in >= input_dim) return;

  float sum = 0.0f;
  for (int out = 0; out < output_dim; ++out) {
    sum += dy[token * output_dim + out] * w[out * input_dim + in];
  }
  dx[token * input_dim + in] += sum;
}

// linear_backward_weight_kv: tokens=8, output_dim=512, input_dim=2048
__global__ void linear_backward_weight_kv(const float* dy, const float* x, float* dw) {
  const int tokens = 8;
  const int output_dim = 512;
  const int input_dim = 2048;
  // dw[out, in] += sum_token dy[token, out] * x[token, in].

  int out = blockIdx.y * blockDim.y + threadIdx.y;
  int in = blockIdx.x * blockDim.x + threadIdx.x;
  if (out >= output_dim || in >= input_dim) return;

  float sum = 0.0f;
  for (int token = 0; token < tokens; ++token) {
    sum += dy[token * output_dim + out] * x[token * input_dim + in];
  }
  dw[out * input_dim + in] += sum;
}

// rope_forward_q: batch=1, seq_len=8, heads=32, head_dim=64
__global__ void rope_forward_q(const float* x, float* y) {
  const int batch = 1;
  const int seq_len = 8;
  const int heads = 32;
  const int head_dim = 64;

  int i = blockIdx.x * BLOCK + threadIdx.x;
  if (i >= batch * seq_len * heads * (head_dim / 2)) return;

  int pair = i % (head_dim / 2);
  int head = i / (head_dim / 2);
  int position = (head / heads) % seq_len;
  int left = head * head_dim + pair;
  int right = left + head_dim / 2;
  double angle = (position + POSITION_OFFSET) * rope_frequency(pair);
  float c = float(cos(angle));
  float s = float(sin(angle));

  y[left] = c * x[left] - s * x[right];
  y[right] = s * x[left] + c * x[right];
}

// rope_backward_q: batch=1, seq_len=8, heads=32, head_dim=64
__global__ void rope_backward_q(const float* dy, float* dx) {
  const int batch = 1;
  const int seq_len = 8;
  const int heads = 32;
  const int head_dim = 64;

  int i = blockIdx.x * BLOCK + threadIdx.x;
  if (i >= batch * seq_len * heads * (head_dim / 2)) return;

  int pair = i % (head_dim / 2);
  int head = i / (head_dim / 2);
  int position = (head / heads) % seq_len;
  int left = head * head_dim + pair;
  int right = left + head_dim / 2;
  double angle = (position + POSITION_OFFSET) * rope_frequency(pair);
  float c = float(cos(angle));
  float s = float(sin(angle));

  // Same rotation with the sine signs reversed: R(theta)^T = R(-theta).
  dx[left] += c * dy[left] + s * dy[right];
  dx[right] += -s * dy[left] + c * dy[right];
}

// rope_forward_k: batch=1, seq_len=8, heads=8, head_dim=64
__global__ void rope_forward_k(const float* x, float* y) {
  const int batch = 1;
  const int seq_len = 8;
  const int heads = 8;
  const int head_dim = 64;

  int i = blockIdx.x * BLOCK + threadIdx.x;
  if (i >= batch * seq_len * heads * (head_dim / 2)) return;

  int pair = i % (head_dim / 2);
  int head = i / (head_dim / 2);
  int position = (head / heads) % seq_len;
  int left = head * head_dim + pair;
  int right = left + head_dim / 2;
  double angle = (position + POSITION_OFFSET) * rope_frequency(pair);
  float c = float(cos(angle));
  float s = float(sin(angle));

  y[left] = c * x[left] - s * x[right];
  y[right] = s * x[left] + c * x[right];
}

// rope_backward_k: batch=1, seq_len=8, heads=8, head_dim=64
__global__ void rope_backward_k(const float* dy, float* dx) {
  const int batch = 1;
  const int seq_len = 8;
  const int heads = 8;
  const int head_dim = 64;

  int i = blockIdx.x * BLOCK + threadIdx.x;
  if (i >= batch * seq_len * heads * (head_dim / 2)) return;

  int pair = i % (head_dim / 2);
  int head = i / (head_dim / 2);
  int position = (head / heads) % seq_len;
  int left = head * head_dim + pair;
  int right = left + head_dim / 2;
  double angle = (position + POSITION_OFFSET) * rope_frequency(pair);
  float c = float(cos(angle));
  float s = float(sin(angle));

  // Same rotation with the sine signs reversed: R(theta)^T = R(-theta).
  dx[left] += c * dy[left] + s * dy[right];
  dx[right] += -s * dy[left] + c * dy[right];
}

// attention_scores_forward: batch=1, seq_len=8, q_heads=32, kv_heads=8, head_dim=64
__global__ void attention_scores_forward(const float* q, const float* k, float* s) {
  const int batch = 1;
  const int seq_len = 8;
  const int q_heads = 32;
  const int kv_heads = 8;
  const int head_dim = 64;

  int i = blockIdx.x * BLOCK + threadIdx.x;
  if (i >= batch * q_heads * seq_len * seq_len) return;
  int key = i % seq_len, t = (i / seq_len) % seq_len, h = (i / (seq_len * seq_len)) % q_heads,
      b = i / (q_heads * seq_len * seq_len);
  if (key > t) {
    s[i] = -INFINITY;
    return;
  }
  int qi = ((b * seq_len + t) * q_heads + h) * head_dim,
      ki = ((b * seq_len + key) * kv_heads + h / (q_heads / kv_heads)) * head_dim;
  float sum = 0;
  for (int d = 0; d < head_dim; ++d)
    sum += q[qi + d] * k[ki + d];
  s[i] = sum / sqrtf(float(head_dim));
}

// attention_scores_backward_q: batch=1, seq_len=8, q_heads=32, kv_heads=8, head_dim=64
__global__ void attention_scores_backward_q(const float* ds, const float* k, float* dq) {
  const int batch = 1;
  const int seq_len = 8;
  const int q_heads = 32;
  const int kv_heads = 8;
  const int head_dim = 64;

  int i = blockIdx.x * BLOCK + threadIdx.x;
  if (i >= batch * seq_len * q_heads * head_dim) return;
  int d = i % head_dim, h = (i / head_dim) % q_heads, t = (i / (head_dim * q_heads)) % seq_len,
      b = i / (head_dim * q_heads * seq_len);
  float sum = 0;
  for (int key = 0; key <= t; ++key)
    sum += ds[((b * q_heads + h) * seq_len + t) * seq_len + key] *
           k[((b * seq_len + key) * kv_heads + h / (q_heads / kv_heads)) * head_dim + d];
  dq[i] += sum / sqrtf(float(head_dim));
}

// attention_scores_backward_k: batch=1, seq_len=8, q_heads=32, kv_heads=8, head_dim=64
__global__ void attention_scores_backward_k(const float* ds, const float* q, float* dk) {
  const int batch = 1;
  const int seq_len = 8;
  const int q_heads = 32;
  const int kv_heads = 8;
  const int head_dim = 64;

  int i = blockIdx.x * BLOCK + threadIdx.x;
  if (i >= batch * seq_len * kv_heads * head_dim) return;
  int d = i % head_dim, g = (i / head_dim) % kv_heads, key = (i / (head_dim * kv_heads)) % seq_len,
      b = i / (head_dim * kv_heads * seq_len);
  float sum = 0;
  // Every Q head sharing this K head contributes; SUM, do not average.
  for (int h = g * (q_heads / kv_heads); h < (g + 1) * (q_heads / kv_heads); ++h)
    for (int t = key; t < seq_len; ++t)
      sum += ds[((b * q_heads + h) * seq_len + t) * seq_len + key] *
             q[((b * seq_len + t) * q_heads + h) * head_dim + d];
  dk[i] += sum / sqrtf(float(head_dim));
}

// softmax_forward: rows=256, cols=8
__global__ void softmax_forward(const float* x, float* p) {
  const int rows = 256;
  const int cols = 8;

  __shared__ float scratch[BLOCK];
  int row = blockIdx.x;
  float mx = -INFINITY;
  for (int j = threadIdx.x; j < cols; j += BLOCK)
    mx = fmaxf(mx, x[row * cols + j]);
  mx = block_max(mx, scratch);
  float sum = 0;
  for (int j = threadIdx.x; j < cols; j += BLOCK)
    sum += expf(x[row * cols + j] - mx);
  sum = block_sum(sum, scratch);
  for (int j = threadIdx.x; j < cols; j += BLOCK)
    p[row * cols + j] = expf(x[row * cols + j] - mx) / sum;
}

// softmax_backward: rows=256, cols=8
__global__ void softmax_backward(const float* p, const float* dp, float* dx) {
  const int rows = 256;
  const int cols = 8;

  __shared__ float scratch[BLOCK];
  int row = blockIdx.x;
  float dot = 0;
  for (int j = threadIdx.x; j < cols; j += BLOCK)
    dot += dp[row * cols + j] * p[row * cols + j];
  dot = block_sum(dot, scratch);
  for (int j = threadIdx.x; j < cols; j += BLOCK) {
    int i = row * cols + j;
    dx[i] += p[i] * (dp[i] - dot); // Causal positions have p=0, hence dx=0.
  }
}

// attention_values_forward: batch=1, seq_len=8, q_heads=32, kv_heads=8, head_dim=64
__global__ void attention_values_forward(const float* p, const float* v, float* o) {
  const int batch = 1;
  const int seq_len = 8;
  const int q_heads = 32;
  const int kv_heads = 8;
  const int head_dim = 64;

  int i = blockIdx.x * BLOCK + threadIdx.x;
  if (i >= batch * seq_len * q_heads * head_dim) return;
  int d = i % head_dim, h = (i / head_dim) % q_heads, t = (i / (head_dim * q_heads)) % seq_len,
      b = i / (head_dim * q_heads * seq_len);
  float sum = 0;
  for (int key = 0; key <= t; ++key)
    sum += p[((b * q_heads + h) * seq_len + t) * seq_len + key] *
           v[((b * seq_len + key) * kv_heads + h / (q_heads / kv_heads)) * head_dim + d];
  o[i] = sum;
}

// attention_values_backward_p: batch=1, seq_len=8, q_heads=32, kv_heads=8, head_dim=64
__global__ void attention_values_backward_p(const float* dout, const float* v, float* dp) {
  const int batch = 1;
  const int seq_len = 8;
  const int q_heads = 32;
  const int kv_heads = 8;
  const int head_dim = 64;

  int i = blockIdx.x * BLOCK + threadIdx.x;
  if (i >= batch * q_heads * seq_len * seq_len) return;
  int key = i % seq_len, t = (i / seq_len) % seq_len, h = (i / (seq_len * seq_len)) % q_heads,
      b = i / (q_heads * seq_len * seq_len);
  if (key > t) return;
  float sum = 0;
  for (int d = 0; d < head_dim; ++d)
    sum += dout[((b * seq_len + t) * q_heads + h) * head_dim + d] *
           v[((b * seq_len + key) * kv_heads + h / (q_heads / kv_heads)) * head_dim + d];
  dp[i] += sum;
}

// attention_values_backward_v: batch=1, seq_len=8, q_heads=32, kv_heads=8, head_dim=64
__global__ void attention_values_backward_v(const float* dout, const float* p, float* dv) {
  const int batch = 1;
  const int seq_len = 8;
  const int q_heads = 32;
  const int kv_heads = 8;
  const int head_dim = 64;

  int i = blockIdx.x * BLOCK + threadIdx.x;
  if (i >= batch * seq_len * kv_heads * head_dim) return;
  int d = i % head_dim, g = (i / head_dim) % kv_heads, key = (i / (head_dim * kv_heads)) % seq_len,
      b = i / (head_dim * kv_heads * seq_len);
  float sum = 0;
  for (int h = g * (q_heads / kv_heads); h < (g + 1) * (q_heads / kv_heads); ++h)
    for (int t = key; t < seq_len; ++t)
      sum += p[((b * q_heads + h) * seq_len + t) * seq_len + key] *
             dout[((b * seq_len + t) * q_heads + h) * head_dim + d];
  dv[i] += sum;
}

// add_forward: elements=16384
__global__ void add_forward(const float* a, const float* b, float* y) {
  const int elements = 16384;

  int i = blockIdx.x * BLOCK + threadIdx.x;
  if (i < elements) y[i] = a[i] + b[i];
}

// add_backward: elements=16384
__global__ void add_backward(const float* dy, float* da, float* db) {
  const int elements = 16384;

  int i = blockIdx.x * BLOCK + threadIdx.x;
  if (i < elements) {
    da[i] += dy[i];
    db[i] += dy[i];
  }
}

// linear_forward_mlp_up: tokens=8, output_dim=8192, input_dim=2048
__global__ void linear_forward_mlp_up(const float* x, const float* w, float* y) {
  const int tokens = 8;
  const int output_dim = 8192;
  const int input_dim = 2048;
  // y[token, out] = sum_in x[token, in] * w[out, in].

  int token = blockIdx.y * blockDim.y + threadIdx.y;
  int out = blockIdx.x * blockDim.x + threadIdx.x;
  if (token >= tokens || out >= output_dim) return;

  float sum = 0.0f;
  for (int in = 0; in < input_dim; ++in) {
    sum += x[token * input_dim + in] * w[out * input_dim + in];
  }
  y[token * output_dim + out] = sum;
}

// linear_backward_input_mlp_up: tokens=8, output_dim=8192, input_dim=2048
__global__ void linear_backward_input_mlp_up(const float* dy, const float* w, float* dx) {
  const int tokens = 8;
  const int output_dim = 8192;
  const int input_dim = 2048;
  // dx[token, in] += sum_out dy[token, out] * w[out, in].

  int token = blockIdx.y * blockDim.y + threadIdx.y;
  int in = blockIdx.x * blockDim.x + threadIdx.x;
  if (token >= tokens || in >= input_dim) return;

  float sum = 0.0f;
  for (int out = 0; out < output_dim; ++out) {
    sum += dy[token * output_dim + out] * w[out * input_dim + in];
  }
  dx[token * input_dim + in] += sum;
}

// linear_backward_weight_mlp_up: tokens=8, output_dim=8192, input_dim=2048
__global__ void linear_backward_weight_mlp_up(const float* dy, const float* x, float* dw) {
  const int tokens = 8;
  const int output_dim = 8192;
  const int input_dim = 2048;
  // dw[out, in] += sum_token dy[token, out] * x[token, in].

  int out = blockIdx.y * blockDim.y + threadIdx.y;
  int in = blockIdx.x * blockDim.x + threadIdx.x;
  if (out >= output_dim || in >= input_dim) return;

  float sum = 0.0f;
  for (int token = 0; token < tokens; ++token) {
    sum += dy[token * output_dim + out] * x[token * input_dim + in];
  }
  dw[out * input_dim + in] += sum;
}

// swiglu_forward: elements=65536
__global__ void swiglu_forward(const float* gate, const float* up, float* y) {
  const int elements = 65536;

  int i = blockIdx.x * BLOCK + threadIdx.x;
  if (i < elements) y[i] = gate[i] / (1.0f + expf(-gate[i])) * up[i];
}

// swiglu_backward: elements=65536
__global__ void swiglu_backward(const float* gate, const float* up, const float* dy, float* dg,
                                float* du) {
  const int elements = 65536;

  int i = blockIdx.x * BLOCK + threadIdx.x;
  if (i >= elements) return;
  float s = 1.0f / (1.0f + expf(-gate[i]));
  dg[i] += dy[i] * up[i] * (s + gate[i] * s * (1.0f - s));
  du[i] += dy[i] * gate[i] * s;
}

// linear_forward_mlp_down: tokens=8, output_dim=2048, input_dim=8192
__global__ void linear_forward_mlp_down(const float* x, const float* w, float* y) {
  const int tokens = 8;
  const int output_dim = 2048;
  const int input_dim = 8192;
  // y[token, out] = sum_in x[token, in] * w[out, in].

  int token = blockIdx.y * blockDim.y + threadIdx.y;
  int out = blockIdx.x * blockDim.x + threadIdx.x;
  if (token >= tokens || out >= output_dim) return;

  float sum = 0.0f;
  for (int in = 0; in < input_dim; ++in) {
    sum += x[token * input_dim + in] * w[out * input_dim + in];
  }
  y[token * output_dim + out] = sum;
}

// linear_backward_input_mlp_down: tokens=8, output_dim=2048, input_dim=8192
__global__ void linear_backward_input_mlp_down(const float* dy, const float* w, float* dx) {
  const int tokens = 8;
  const int output_dim = 2048;
  const int input_dim = 8192;
  // dx[token, in] += sum_out dy[token, out] * w[out, in].

  int token = blockIdx.y * blockDim.y + threadIdx.y;
  int in = blockIdx.x * blockDim.x + threadIdx.x;
  if (token >= tokens || in >= input_dim) return;

  float sum = 0.0f;
  for (int out = 0; out < output_dim; ++out) {
    sum += dy[token * output_dim + out] * w[out * input_dim + in];
  }
  dx[token * input_dim + in] += sum;
}

// linear_backward_weight_mlp_down: tokens=8, output_dim=2048, input_dim=8192
__global__ void linear_backward_weight_mlp_down(const float* dy, const float* x, float* dw) {
  const int tokens = 8;
  const int output_dim = 2048;
  const int input_dim = 8192;
  // dw[out, in] += sum_token dy[token, out] * x[token, in].

  int out = blockIdx.y * blockDim.y + threadIdx.y;
  int in = blockIdx.x * blockDim.x + threadIdx.x;
  if (out >= output_dim || in >= input_dim) return;

  float sum = 0.0f;
  for (int token = 0; token < tokens; ++token) {
    sum += dy[token * output_dim + out] * x[token * input_dim + in];
  }
  dw[out * input_dim + in] += sum;
}

// linear_forward_lm_head: tokens=8, output_dim=128256, input_dim=2048
__global__ void linear_forward_lm_head(const float* x, const float* w, float* y) {
  const int tokens = 8;
  const int output_dim = 128256;
  const int input_dim = 2048;
  // y[token, out] = sum_in x[token, in] * w[out, in].

  int token = blockIdx.y * blockDim.y + threadIdx.y;
  int out = blockIdx.x * blockDim.x + threadIdx.x;
  if (token >= tokens || out >= output_dim) return;

  float sum = 0.0f;
  for (int in = 0; in < input_dim; ++in) {
    sum += x[token * input_dim + in] * w[out * input_dim + in];
  }
  y[token * output_dim + out] = sum;
}

// linear_backward_input_lm_head: tokens=8, output_dim=128256, input_dim=2048
__global__ void linear_backward_input_lm_head(const float* dy, const float* w, float* dx) {
  const int tokens = 8;
  const int output_dim = 128256;
  const int input_dim = 2048;
  // dx[token, in] += sum_out dy[token, out] * w[out, in].

  int token = blockIdx.y * blockDim.y + threadIdx.y;
  int in = blockIdx.x * blockDim.x + threadIdx.x;
  if (token >= tokens || in >= input_dim) return;

  float sum = 0.0f;
  for (int out = 0; out < output_dim; ++out) {
    sum += dy[token * output_dim + out] * w[out * input_dim + in];
  }
  dx[token * input_dim + in] += sum;
}

// linear_backward_weight_lm_head: tokens=8, output_dim=128256, input_dim=2048
__global__ void linear_backward_weight_lm_head(const float* dy, const float* x, float* dw) {
  const int tokens = 8;
  const int output_dim = 128256;
  const int input_dim = 2048;
  // dw[out, in] += sum_token dy[token, out] * x[token, in].

  int out = blockIdx.y * blockDim.y + threadIdx.y;
  int in = blockIdx.x * blockDim.x + threadIdx.x;
  if (out >= output_dim || in >= input_dim) return;

  float sum = 0.0f;
  for (int token = 0; token < tokens; ++token) {
    sum += dy[token * output_dim + out] * x[token * input_dim + in];
  }
  dw[out * input_dim + in] += sum;
}

// cross_entropy_forward: rows=8, vocab=128256
__global__ void cross_entropy_forward(const float* logits, const int* targets, float* losses) {
  const int rows = 8;
  const int vocab = 128256;

  __shared__ float scratch[BLOCK];
  int row = blockIdx.x;
  float mx = -INFINITY;
  for (int v = threadIdx.x; v < vocab; v += BLOCK)
    mx = fmaxf(mx, logits[row * vocab + v]);
  mx = block_max(mx, scratch);
  float sum = 0;
  for (int v = threadIdx.x; v < vocab; v += BLOCK)
    sum += expf(logits[row * vocab + v] - mx);
  sum = block_sum(sum, scratch);
  if (threadIdx.x == 0) losses[row] = logf(sum) + (mx - logits[row * vocab + targets[row]]);
}

// cross_entropy_backward: rows=8, vocab=128256
__global__ void cross_entropy_backward(const float* logits, const int* targets,
                                       const float* dlosses, float* dlogits) {
  const int rows = 8;
  const int vocab = 128256;

  __shared__ float scratch[BLOCK];
  int row = blockIdx.x;
  float mx = -INFINITY;
  for (int v = threadIdx.x; v < vocab; v += BLOCK)
    mx = fmaxf(mx, logits[row * vocab + v]);
  mx = block_max(mx, scratch);
  float sum = 0;
  for (int v = threadIdx.x; v < vocab; v += BLOCK)
    sum += expf(logits[row * vocab + v] - mx);
  sum = block_sum(sum, scratch);
  for (int v = threadIdx.x; v < vocab; v += BLOCK)
    dlogits[row * vocab + v] +=
        dlosses[row] * (expf(logits[row * vocab + v] - mx) / sum - (v == targets[row]));
}

// mean_forward: tokens=8
__global__ void mean_forward(const float* losses, float* loss) {
  const int tokens = 8;

  __shared__ float scratch[BLOCK];
  float sum = 0;
  for (int i = threadIdx.x; i < tokens; i += BLOCK)
    sum += losses[i];
  sum = block_sum(sum, scratch);
  if (threadIdx.x == 0) loss[0] = sum / tokens;
}

// mean_backward: tokens=8
__global__ void mean_backward(const float* dloss, float* dlosses) {
  const int tokens = 8;

  int i = blockIdx.x * BLOCK + threadIdx.x;
  if (i < tokens) dlosses[i] += dloss[0] / tokens;
}

// sgd_update_262668288: elements=262668288
__global__ void sgd_update_262668288(float* w, const float* dw, float lr) {
  const int elements = 262668288;

  int i = blockIdx.x * BLOCK + threadIdx.x;
  if (i < elements) w[i] -= lr * dw[i];
}

// sgd_update_2048: elements=2048
__global__ void sgd_update_2048(float* w, const float* dw, float lr) {
  const int elements = 2048;

  int i = blockIdx.x * BLOCK + threadIdx.x;
  if (i < elements) w[i] -= lr * dw[i];
}

// sgd_update_4194304: elements=4194304
__global__ void sgd_update_4194304(float* w, const float* dw, float lr) {
  const int elements = 4194304;

  int i = blockIdx.x * BLOCK + threadIdx.x;
  if (i < elements) w[i] -= lr * dw[i];
}

// sgd_update_1048576: elements=1048576
__global__ void sgd_update_1048576(float* w, const float* dw, float lr) {
  const int elements = 1048576;

  int i = blockIdx.x * BLOCK + threadIdx.x;
  if (i < elements) w[i] -= lr * dw[i];
}

// sgd_update_16777216: elements=16777216
__global__ void sgd_update_16777216(float* w, const float* dw, float lr) {
  const int elements = 16777216;

  int i = blockIdx.x * BLOCK + threadIdx.x;
  if (i < elements) w[i] -= lr * dw[i];
}

// Tiny C ABI used by the dependency-light Python runner.
extern "C" {
int gpu_init(int device) { return hipSetDevice(device); }
const char* gpu_error(int code) { return hipGetErrorString((hipError_t)code); }
const char* gpu_name() {
  static hipDeviceProp_t p;
  int d = 0;
  if (hipGetDevice(&d) != hipSuccess || hipGetDeviceProperties(&p, d) != hipSuccess)
    return "unknown GPU";
  return p.name;
}
int gpu_memory(size_t* available, size_t* total) { return hipMemGetInfo(available, total); }
int gpu_alloc(void** p, size_t n) { return hipMalloc(p, n); }
int gpu_free(void* p) { return hipFree(p); }
int gpu_zero(void* p, size_t n) { return hipMemset(p, 0, n); }
int gpu_upload(void* p, const void* host, size_t n) {
  return hipMemcpy(p, host, n, hipMemcpyHostToDevice);
}
int gpu_download(void* host, const void* p, size_t n) {
  return hipMemcpy(host, p, n, hipMemcpyDeviceToHost);
}
}
