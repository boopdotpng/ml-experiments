// GENERATED FORWARD execution order. Do not edit; run train.py emit.
// v[] = saved forward values; g[] = accumulated gradients.
extern "C" int run_forward(float** v, float** g, const int* ids, const int* targets) {
  // embedding: embedding -> (16, 64)
  embedding_forward<<<4, 256>>>(v[0], ids, v[1]);
  CHECK(hipGetLastError());
  // layer0.attention_norm: rmsnorm -> (16, 64)
  rmsnorm_forward<<<16, 256>>>(v[1], v[2], v[3], v[4]);
  CHECK(hipGetLastError());
  // layer0.q: linear -> (16, 64)
  linear_forward_hidden<<<dim3(4, 1), dim3(16, 16)>>>(v[3], v[5], v[6]);
  CHECK(hipGetLastError());
  // layer0.k: linear -> (16, 16)
  linear_forward_kv<<<dim3(1, 1), dim3(16, 16)>>>(v[3], v[7], v[8]);
  CHECK(hipGetLastError());
  // layer0.v: linear -> (16, 16)
  linear_forward_kv<<<dim3(1, 1), dim3(16, 16)>>>(v[3], v[9], v[10]);
  CHECK(hipGetLastError());
  // layer0.q_rope: rope -> (16, 64)
  rope_forward_q<<<2, 256>>>(v[6], v[11]);
  CHECK(hipGetLastError());
  // layer0.k_rope: rope -> (16, 16)
  rope_forward_k<<<1, 256>>>(v[8], v[12]);
  CHECK(hipGetLastError());
  // layer0.scores: scores -> (2, 4, 8, 8)
  attention_scores_forward<<<2, 256>>>(v[11], v[12], v[13]);
  CHECK(hipGetLastError());
  // layer0.probabilities: softmax -> (2, 4, 8, 8)
  softmax_forward<<<64, 256>>>(v[13], v[14]);
  CHECK(hipGetLastError());
  // layer0.context: values -> (16, 64)
  attention_values_forward<<<4, 256>>>(v[14], v[10], v[15]);
  CHECK(hipGetLastError());
  // layer0.o: linear -> (16, 64)
  linear_forward_hidden<<<dim3(4, 1), dim3(16, 16)>>>(v[15], v[16], v[17]);
  CHECK(hipGetLastError());
  // layer0.attention_residual: add -> (16, 64)
  add_forward<<<4, 256>>>(v[1], v[17], v[18]);
  CHECK(hipGetLastError());
  // layer0.ffn_norm: rmsnorm -> (16, 64)
  rmsnorm_forward<<<16, 256>>>(v[18], v[19], v[20], v[21]);
  CHECK(hipGetLastError());
  // layer0.gate: linear -> (16, 256)
  linear_forward_mlp_up<<<dim3(16, 1), dim3(16, 16)>>>(v[20], v[22], v[23]);
  CHECK(hipGetLastError());
  // layer0.up: linear -> (16, 256)
  linear_forward_mlp_up<<<dim3(16, 1), dim3(16, 16)>>>(v[20], v[24], v[25]);
  CHECK(hipGetLastError());
  // layer0.swiglu: swiglu -> (16, 256)
  swiglu_forward<<<16, 256>>>(v[23], v[25], v[26]);
  CHECK(hipGetLastError());
  // layer0.down: linear -> (16, 64)
  linear_forward_mlp_down<<<dim3(4, 1), dim3(16, 16)>>>(v[26], v[27], v[28]);
  CHECK(hipGetLastError());
  // layer0.ffn_residual: add -> (16, 64)
  add_forward<<<4, 256>>>(v[18], v[28], v[29]);
  CHECK(hipGetLastError());
  // layer1.attention_norm: rmsnorm -> (16, 64)
  rmsnorm_forward<<<16, 256>>>(v[29], v[30], v[31], v[32]);
  CHECK(hipGetLastError());
  // layer1.q: linear -> (16, 64)
  linear_forward_hidden<<<dim3(4, 1), dim3(16, 16)>>>(v[31], v[33], v[34]);
  CHECK(hipGetLastError());
  // layer1.k: linear -> (16, 16)
  linear_forward_kv<<<dim3(1, 1), dim3(16, 16)>>>(v[31], v[35], v[36]);
  CHECK(hipGetLastError());
  // layer1.v: linear -> (16, 16)
  linear_forward_kv<<<dim3(1, 1), dim3(16, 16)>>>(v[31], v[37], v[38]);
  CHECK(hipGetLastError());
  // layer1.q_rope: rope -> (16, 64)
  rope_forward_q<<<2, 256>>>(v[34], v[39]);
  CHECK(hipGetLastError());
  // layer1.k_rope: rope -> (16, 16)
  rope_forward_k<<<1, 256>>>(v[36], v[40]);
  CHECK(hipGetLastError());
  // layer1.scores: scores -> (2, 4, 8, 8)
  attention_scores_forward<<<2, 256>>>(v[39], v[40], v[41]);
  CHECK(hipGetLastError());
  // layer1.probabilities: softmax -> (2, 4, 8, 8)
  softmax_forward<<<64, 256>>>(v[41], v[42]);
  CHECK(hipGetLastError());
  // layer1.context: values -> (16, 64)
  attention_values_forward<<<4, 256>>>(v[42], v[38], v[43]);
  CHECK(hipGetLastError());
  // layer1.o: linear -> (16, 64)
  linear_forward_hidden<<<dim3(4, 1), dim3(16, 16)>>>(v[43], v[44], v[45]);
  CHECK(hipGetLastError());
  // layer1.attention_residual: add -> (16, 64)
  add_forward<<<4, 256>>>(v[29], v[45], v[46]);
  CHECK(hipGetLastError());
  // layer1.ffn_norm: rmsnorm -> (16, 64)
  rmsnorm_forward<<<16, 256>>>(v[46], v[47], v[48], v[49]);
  CHECK(hipGetLastError());
  // layer1.gate: linear -> (16, 256)
  linear_forward_mlp_up<<<dim3(16, 1), dim3(16, 16)>>>(v[48], v[50], v[51]);
  CHECK(hipGetLastError());
  // layer1.up: linear -> (16, 256)
  linear_forward_mlp_up<<<dim3(16, 1), dim3(16, 16)>>>(v[48], v[52], v[53]);
  CHECK(hipGetLastError());
  // layer1.swiglu: swiglu -> (16, 256)
  swiglu_forward<<<16, 256>>>(v[51], v[53], v[54]);
  CHECK(hipGetLastError());
  // layer1.down: linear -> (16, 64)
  linear_forward_mlp_down<<<dim3(4, 1), dim3(16, 16)>>>(v[54], v[55], v[56]);
  CHECK(hipGetLastError());
  // layer1.ffn_residual: add -> (16, 64)
  add_forward<<<4, 256>>>(v[46], v[56], v[57]);
  CHECK(hipGetLastError());
  // final_norm: rmsnorm -> (16, 64)
  rmsnorm_forward<<<16, 256>>>(v[57], v[58], v[59], v[60]);
  CHECK(hipGetLastError());
  // lm_head: linear -> (16, 128)
  linear_forward_lm_head<<<dim3(8, 1), dim3(16, 16)>>>(v[59], v[0], v[61]);
  CHECK(hipGetLastError());
  // token_losses: cross_entropy -> (16,)
  cross_entropy_forward<<<16, 256>>>(v[61], targets, v[62]);
  CHECK(hipGetLastError());
  // loss: mean -> (1,)
  mean_forward<<<1, 256>>>(v[62], v[63]);
  CHECK(hipGetLastError());
  return hipDeviceSynchronize();
}
