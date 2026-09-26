// GENERATED BACKWARD execution order. Do not edit; run train.py emit.
// v[] = saved forward values; g[] = accumulated gradients.
extern "C" int run_backward(float** v, float** g, const int* ids, const int* targets) {
  // The runner zeros all g[] before backward. Seed dLoss/dLoss = 1.
  float seed = 1.0f;
  CHECK(hipMemcpy(g[63], &seed, sizeof(float), hipMemcpyHostToDevice));
  // loss: mean -> (1,)
  mean_backward<<<1, 256>>>(g[63], g[62]);
  CHECK(hipGetLastError());
  // token_losses: cross_entropy -> (16,)
  cross_entropy_backward<<<16, 256>>>(v[61], targets, g[62], g[61]);
  CHECK(hipGetLastError());
  // lm_head: linear -> (16, 128)
  linear_backward_input_lm_head<<<dim3(4, 1), dim3(16, 16)>>>(g[61], v[0], g[59]);
  CHECK(hipGetLastError());
  linear_backward_weight_lm_head<<<dim3(4, 8), dim3(16, 16)>>>(g[61], v[59], g[0]);
  CHECK(hipGetLastError());
  // final_norm: rmsnorm -> (16, 64)
  rmsnorm_backward_input<<<16, 256>>>(v[57], v[58], v[60], g[59], g[57]);
  CHECK(hipGetLastError());
  rmsnorm_backward_weight<<<1, 256>>>(v[57], v[60], g[59], g[58]);
  CHECK(hipGetLastError());
  // layer1.ffn_residual: add -> (16, 64)
  add_backward<<<4, 256>>>(g[57], g[46], g[56]);
  CHECK(hipGetLastError());
  // layer1.down: linear -> (16, 64)
  linear_backward_input_mlp_down<<<dim3(16, 1), dim3(16, 16)>>>(g[56], v[55], g[54]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_down<<<dim3(16, 4), dim3(16, 16)>>>(g[56], v[54], g[55]);
  CHECK(hipGetLastError());
  // layer1.swiglu: swiglu -> (16, 256)
  swiglu_backward<<<16, 256>>>(v[51], v[53], g[54], g[51], g[53]);
  CHECK(hipGetLastError());
  // layer1.up: linear -> (16, 256)
  linear_backward_input_mlp_up<<<dim3(4, 1), dim3(16, 16)>>>(g[53], v[52], g[48]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_up<<<dim3(4, 16), dim3(16, 16)>>>(g[53], v[48], g[52]);
  CHECK(hipGetLastError());
  // layer1.gate: linear -> (16, 256)
  linear_backward_input_mlp_up<<<dim3(4, 1), dim3(16, 16)>>>(g[51], v[50], g[48]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_up<<<dim3(4, 16), dim3(16, 16)>>>(g[51], v[48], g[50]);
  CHECK(hipGetLastError());
  // layer1.ffn_norm: rmsnorm -> (16, 64)
  rmsnorm_backward_input<<<16, 256>>>(v[46], v[47], v[49], g[48], g[46]);
  CHECK(hipGetLastError());
  rmsnorm_backward_weight<<<1, 256>>>(v[46], v[49], g[48], g[47]);
  CHECK(hipGetLastError());
  // layer1.attention_residual: add -> (16, 64)
  add_backward<<<4, 256>>>(g[46], g[29], g[45]);
  CHECK(hipGetLastError());
  // layer1.o: linear -> (16, 64)
  linear_backward_input_hidden<<<dim3(4, 1), dim3(16, 16)>>>(g[45], v[44], g[43]);
  CHECK(hipGetLastError());
  linear_backward_weight_hidden<<<dim3(4, 4), dim3(16, 16)>>>(g[45], v[43], g[44]);
  CHECK(hipGetLastError());
  // layer1.context: values -> (16, 64)
  attention_values_backward_p<<<2, 256>>>(g[43], v[38], g[42]);
  CHECK(hipGetLastError());
  attention_values_backward_v<<<1, 256>>>(g[43], v[42], g[38]);
  CHECK(hipGetLastError());
  // layer1.probabilities: softmax -> (2, 4, 8, 8)
  softmax_backward<<<64, 256>>>(v[42], g[42], g[41]);
  CHECK(hipGetLastError());
  // layer1.scores: scores -> (2, 4, 8, 8)
  attention_scores_backward_q<<<4, 256>>>(g[41], v[40], g[39]);
  CHECK(hipGetLastError());
  attention_scores_backward_k<<<1, 256>>>(g[41], v[39], g[40]);
  CHECK(hipGetLastError());
  // layer1.k_rope: rope -> (16, 16)
  rope_backward_k<<<1, 256>>>(g[40], g[36]);
  CHECK(hipGetLastError());
  // layer1.q_rope: rope -> (16, 64)
  rope_backward_q<<<2, 256>>>(g[39], g[34]);
  CHECK(hipGetLastError());
  // layer1.v: linear -> (16, 16)
  linear_backward_input_kv<<<dim3(4, 1), dim3(16, 16)>>>(g[38], v[37], g[31]);
  CHECK(hipGetLastError());
  linear_backward_weight_kv<<<dim3(4, 1), dim3(16, 16)>>>(g[38], v[31], g[37]);
  CHECK(hipGetLastError());
  // layer1.k: linear -> (16, 16)
  linear_backward_input_kv<<<dim3(4, 1), dim3(16, 16)>>>(g[36], v[35], g[31]);
  CHECK(hipGetLastError());
  linear_backward_weight_kv<<<dim3(4, 1), dim3(16, 16)>>>(g[36], v[31], g[35]);
  CHECK(hipGetLastError());
  // layer1.q: linear -> (16, 64)
  linear_backward_input_hidden<<<dim3(4, 1), dim3(16, 16)>>>(g[34], v[33], g[31]);
  CHECK(hipGetLastError());
  linear_backward_weight_hidden<<<dim3(4, 4), dim3(16, 16)>>>(g[34], v[31], g[33]);
  CHECK(hipGetLastError());
  // layer1.attention_norm: rmsnorm -> (16, 64)
  rmsnorm_backward_input<<<16, 256>>>(v[29], v[30], v[32], g[31], g[29]);
  CHECK(hipGetLastError());
  rmsnorm_backward_weight<<<1, 256>>>(v[29], v[32], g[31], g[30]);
  CHECK(hipGetLastError());
  // layer0.ffn_residual: add -> (16, 64)
  add_backward<<<4, 256>>>(g[29], g[18], g[28]);
  CHECK(hipGetLastError());
  // layer0.down: linear -> (16, 64)
  linear_backward_input_mlp_down<<<dim3(16, 1), dim3(16, 16)>>>(g[28], v[27], g[26]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_down<<<dim3(16, 4), dim3(16, 16)>>>(g[28], v[26], g[27]);
  CHECK(hipGetLastError());
  // layer0.swiglu: swiglu -> (16, 256)
  swiglu_backward<<<16, 256>>>(v[23], v[25], g[26], g[23], g[25]);
  CHECK(hipGetLastError());
  // layer0.up: linear -> (16, 256)
  linear_backward_input_mlp_up<<<dim3(4, 1), dim3(16, 16)>>>(g[25], v[24], g[20]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_up<<<dim3(4, 16), dim3(16, 16)>>>(g[25], v[20], g[24]);
  CHECK(hipGetLastError());
  // layer0.gate: linear -> (16, 256)
  linear_backward_input_mlp_up<<<dim3(4, 1), dim3(16, 16)>>>(g[23], v[22], g[20]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_up<<<dim3(4, 16), dim3(16, 16)>>>(g[23], v[20], g[22]);
  CHECK(hipGetLastError());
  // layer0.ffn_norm: rmsnorm -> (16, 64)
  rmsnorm_backward_input<<<16, 256>>>(v[18], v[19], v[21], g[20], g[18]);
  CHECK(hipGetLastError());
  rmsnorm_backward_weight<<<1, 256>>>(v[18], v[21], g[20], g[19]);
  CHECK(hipGetLastError());
  // layer0.attention_residual: add -> (16, 64)
  add_backward<<<4, 256>>>(g[18], g[1], g[17]);
  CHECK(hipGetLastError());
  // layer0.o: linear -> (16, 64)
  linear_backward_input_hidden<<<dim3(4, 1), dim3(16, 16)>>>(g[17], v[16], g[15]);
  CHECK(hipGetLastError());
  linear_backward_weight_hidden<<<dim3(4, 4), dim3(16, 16)>>>(g[17], v[15], g[16]);
  CHECK(hipGetLastError());
  // layer0.context: values -> (16, 64)
  attention_values_backward_p<<<2, 256>>>(g[15], v[10], g[14]);
  CHECK(hipGetLastError());
  attention_values_backward_v<<<1, 256>>>(g[15], v[14], g[10]);
  CHECK(hipGetLastError());
  // layer0.probabilities: softmax -> (2, 4, 8, 8)
  softmax_backward<<<64, 256>>>(v[14], g[14], g[13]);
  CHECK(hipGetLastError());
  // layer0.scores: scores -> (2, 4, 8, 8)
  attention_scores_backward_q<<<4, 256>>>(g[13], v[12], g[11]);
  CHECK(hipGetLastError());
  attention_scores_backward_k<<<1, 256>>>(g[13], v[11], g[12]);
  CHECK(hipGetLastError());
  // layer0.k_rope: rope -> (16, 16)
  rope_backward_k<<<1, 256>>>(g[12], g[8]);
  CHECK(hipGetLastError());
  // layer0.q_rope: rope -> (16, 64)
  rope_backward_q<<<2, 256>>>(g[11], g[6]);
  CHECK(hipGetLastError());
  // layer0.v: linear -> (16, 16)
  linear_backward_input_kv<<<dim3(4, 1), dim3(16, 16)>>>(g[10], v[9], g[3]);
  CHECK(hipGetLastError());
  linear_backward_weight_kv<<<dim3(4, 1), dim3(16, 16)>>>(g[10], v[3], g[9]);
  CHECK(hipGetLastError());
  // layer0.k: linear -> (16, 16)
  linear_backward_input_kv<<<dim3(4, 1), dim3(16, 16)>>>(g[8], v[7], g[3]);
  CHECK(hipGetLastError());
  linear_backward_weight_kv<<<dim3(4, 1), dim3(16, 16)>>>(g[8], v[3], g[7]);
  CHECK(hipGetLastError());
  // layer0.q: linear -> (16, 64)
  linear_backward_input_hidden<<<dim3(4, 1), dim3(16, 16)>>>(g[6], v[5], g[3]);
  CHECK(hipGetLastError());
  linear_backward_weight_hidden<<<dim3(4, 4), dim3(16, 16)>>>(g[6], v[3], g[5]);
  CHECK(hipGetLastError());
  // layer0.attention_norm: rmsnorm -> (16, 64)
  rmsnorm_backward_input<<<16, 256>>>(v[1], v[2], v[4], g[3], g[1]);
  CHECK(hipGetLastError());
  rmsnorm_backward_weight<<<1, 256>>>(v[1], v[4], g[3], g[2]);
  CHECK(hipGetLastError());
  // embedding: embedding -> (16, 64)
  embedding_backward<<<4, 256>>>(g[1], ids, g[0]);
  CHECK(hipGetLastError());
  return hipDeviceSynchronize();
}
