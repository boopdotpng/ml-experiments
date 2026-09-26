#include "kernels.hip"
#define CHECK(expr) do { hipError_t e=(hipError_t)(expr); if(e!=hipSuccess) return int(e); } while(0)
#include "forward.hip"
#include "backward.hip"
extern "C" int run_sgd(float** v, float** g, float lr) {
  // embedding.weight
  sgd_update_8192<<<32, 256>>>(v[0], g[0], lr);
  CHECK(hipGetLastError());
  // layer0.attention_norm.weight
  sgd_update_64<<<1, 256>>>(v[2], g[2], lr);
  CHECK(hipGetLastError());
  // layer0.q.weight
  sgd_update_4096<<<16, 256>>>(v[5], g[5], lr);
  CHECK(hipGetLastError());
  // layer0.k.weight
  sgd_update_1024<<<4, 256>>>(v[7], g[7], lr);
  CHECK(hipGetLastError());
  // layer0.v.weight
  sgd_update_1024<<<4, 256>>>(v[9], g[9], lr);
  CHECK(hipGetLastError());
  // layer0.o.weight
  sgd_update_4096<<<16, 256>>>(v[16], g[16], lr);
  CHECK(hipGetLastError());
  // layer0.ffn_norm.weight
  sgd_update_64<<<1, 256>>>(v[19], g[19], lr);
  CHECK(hipGetLastError());
  // layer0.gate.weight
  sgd_update_16384<<<64, 256>>>(v[22], g[22], lr);
  CHECK(hipGetLastError());
  // layer0.up.weight
  sgd_update_16384<<<64, 256>>>(v[24], g[24], lr);
  CHECK(hipGetLastError());
  // layer0.down.weight
  sgd_update_16384<<<64, 256>>>(v[27], g[27], lr);
  CHECK(hipGetLastError());
  // layer1.attention_norm.weight
  sgd_update_64<<<1, 256>>>(v[30], g[30], lr);
  CHECK(hipGetLastError());
  // layer1.q.weight
  sgd_update_4096<<<16, 256>>>(v[33], g[33], lr);
  CHECK(hipGetLastError());
  // layer1.k.weight
  sgd_update_1024<<<4, 256>>>(v[35], g[35], lr);
  CHECK(hipGetLastError());
  // layer1.v.weight
  sgd_update_1024<<<4, 256>>>(v[37], g[37], lr);
  CHECK(hipGetLastError());
  // layer1.o.weight
  sgd_update_4096<<<16, 256>>>(v[44], g[44], lr);
  CHECK(hipGetLastError());
  // layer1.ffn_norm.weight
  sgd_update_64<<<1, 256>>>(v[47], g[47], lr);
  CHECK(hipGetLastError());
  // layer1.gate.weight
  sgd_update_16384<<<64, 256>>>(v[50], g[50], lr);
  CHECK(hipGetLastError());
  // layer1.up.weight
  sgd_update_16384<<<64, 256>>>(v[52], g[52], lr);
  CHECK(hipGetLastError());
  // layer1.down.weight
  sgd_update_16384<<<64, 256>>>(v[55], g[55], lr);
  CHECK(hipGetLastError());
  // final_norm.weight
  sgd_update_64<<<1, 256>>>(v[58], g[58], lr);
  CHECK(hipGetLastError());
  return hipDeviceSynchronize();
}
extern "C" int single_forward(int op, float** v, float** g, const int* ids, const int* targets) {
  switch(op) {
  case 0: {
  embedding_forward<<<4, 256>>>(v[0], ids, v[1]);
  CHECK(hipGetLastError());
    break;
  }
  case 1: {
  rmsnorm_forward<<<16, 256>>>(v[1], v[2], v[3], v[4]);
  CHECK(hipGetLastError());
    break;
  }
  case 2: {
  linear_forward_hidden<<<dim3(4, 1), dim3(16, 16)>>>(v[3], v[5], v[6]);
  CHECK(hipGetLastError());
    break;
  }
  case 3: {
  linear_forward_kv<<<dim3(1, 1), dim3(16, 16)>>>(v[3], v[7], v[8]);
  CHECK(hipGetLastError());
    break;
  }
  case 4: {
  linear_forward_kv<<<dim3(1, 1), dim3(16, 16)>>>(v[3], v[9], v[10]);
  CHECK(hipGetLastError());
    break;
  }
  case 5: {
  rope_forward_q<<<2, 256>>>(v[6], v[11]);
  CHECK(hipGetLastError());
    break;
  }
  case 6: {
  rope_forward_k<<<1, 256>>>(v[8], v[12]);
  CHECK(hipGetLastError());
    break;
  }
  case 7: {
  attention_scores_forward<<<2, 256>>>(v[11], v[12], v[13]);
  CHECK(hipGetLastError());
    break;
  }
  case 8: {
  softmax_forward<<<64, 256>>>(v[13], v[14]);
  CHECK(hipGetLastError());
    break;
  }
  case 9: {
  attention_values_forward<<<4, 256>>>(v[14], v[10], v[15]);
  CHECK(hipGetLastError());
    break;
  }
  case 10: {
  linear_forward_hidden<<<dim3(4, 1), dim3(16, 16)>>>(v[15], v[16], v[17]);
  CHECK(hipGetLastError());
    break;
  }
  case 11: {
  add_forward<<<4, 256>>>(v[1], v[17], v[18]);
  CHECK(hipGetLastError());
    break;
  }
  case 12: {
  rmsnorm_forward<<<16, 256>>>(v[18], v[19], v[20], v[21]);
  CHECK(hipGetLastError());
    break;
  }
  case 13: {
  linear_forward_mlp_up<<<dim3(16, 1), dim3(16, 16)>>>(v[20], v[22], v[23]);
  CHECK(hipGetLastError());
    break;
  }
  case 14: {
  linear_forward_mlp_up<<<dim3(16, 1), dim3(16, 16)>>>(v[20], v[24], v[25]);
  CHECK(hipGetLastError());
    break;
  }
  case 15: {
  swiglu_forward<<<16, 256>>>(v[23], v[25], v[26]);
  CHECK(hipGetLastError());
    break;
  }
  case 16: {
  linear_forward_mlp_down<<<dim3(4, 1), dim3(16, 16)>>>(v[26], v[27], v[28]);
  CHECK(hipGetLastError());
    break;
  }
  case 17: {
  add_forward<<<4, 256>>>(v[18], v[28], v[29]);
  CHECK(hipGetLastError());
    break;
  }
  case 18: {
  rmsnorm_forward<<<16, 256>>>(v[29], v[30], v[31], v[32]);
  CHECK(hipGetLastError());
    break;
  }
  case 19: {
  linear_forward_hidden<<<dim3(4, 1), dim3(16, 16)>>>(v[31], v[33], v[34]);
  CHECK(hipGetLastError());
    break;
  }
  case 20: {
  linear_forward_kv<<<dim3(1, 1), dim3(16, 16)>>>(v[31], v[35], v[36]);
  CHECK(hipGetLastError());
    break;
  }
  case 21: {
  linear_forward_kv<<<dim3(1, 1), dim3(16, 16)>>>(v[31], v[37], v[38]);
  CHECK(hipGetLastError());
    break;
  }
  case 22: {
  rope_forward_q<<<2, 256>>>(v[34], v[39]);
  CHECK(hipGetLastError());
    break;
  }
  case 23: {
  rope_forward_k<<<1, 256>>>(v[36], v[40]);
  CHECK(hipGetLastError());
    break;
  }
  case 24: {
  attention_scores_forward<<<2, 256>>>(v[39], v[40], v[41]);
  CHECK(hipGetLastError());
    break;
  }
  case 25: {
  softmax_forward<<<64, 256>>>(v[41], v[42]);
  CHECK(hipGetLastError());
    break;
  }
  case 26: {
  attention_values_forward<<<4, 256>>>(v[42], v[38], v[43]);
  CHECK(hipGetLastError());
    break;
  }
  case 27: {
  linear_forward_hidden<<<dim3(4, 1), dim3(16, 16)>>>(v[43], v[44], v[45]);
  CHECK(hipGetLastError());
    break;
  }
  case 28: {
  add_forward<<<4, 256>>>(v[29], v[45], v[46]);
  CHECK(hipGetLastError());
    break;
  }
  case 29: {
  rmsnorm_forward<<<16, 256>>>(v[46], v[47], v[48], v[49]);
  CHECK(hipGetLastError());
    break;
  }
  case 30: {
  linear_forward_mlp_up<<<dim3(16, 1), dim3(16, 16)>>>(v[48], v[50], v[51]);
  CHECK(hipGetLastError());
    break;
  }
  case 31: {
  linear_forward_mlp_up<<<dim3(16, 1), dim3(16, 16)>>>(v[48], v[52], v[53]);
  CHECK(hipGetLastError());
    break;
  }
  case 32: {
  swiglu_forward<<<16, 256>>>(v[51], v[53], v[54]);
  CHECK(hipGetLastError());
    break;
  }
  case 33: {
  linear_forward_mlp_down<<<dim3(4, 1), dim3(16, 16)>>>(v[54], v[55], v[56]);
  CHECK(hipGetLastError());
    break;
  }
  case 34: {
  add_forward<<<4, 256>>>(v[46], v[56], v[57]);
  CHECK(hipGetLastError());
    break;
  }
  case 35: {
  rmsnorm_forward<<<16, 256>>>(v[57], v[58], v[59], v[60]);
  CHECK(hipGetLastError());
    break;
  }
  case 36: {
  linear_forward_lm_head<<<dim3(8, 1), dim3(16, 16)>>>(v[59], v[0], v[61]);
  CHECK(hipGetLastError());
    break;
  }
  case 37: {
  cross_entropy_forward<<<16, 256>>>(v[61], targets, v[62]);
  CHECK(hipGetLastError());
    break;
  }
  case 38: {
  mean_forward<<<1, 256>>>(v[62], v[63]);
  CHECK(hipGetLastError());
    break;
  }
  default: return int(hipErrorInvalidValue);
  }
  return hipDeviceSynchronize();
}
extern "C" int single_backward(int op, float** v, float** g, const int* ids, const int* targets) {
  switch(op) {
  case 0: {
  embedding_backward<<<4, 256>>>(g[1], ids, g[0]);
  CHECK(hipGetLastError());
    break;
  }
  case 1: {
  rmsnorm_backward_input<<<16, 256>>>(v[1], v[2], v[4], g[3], g[1]);
  CHECK(hipGetLastError());
  rmsnorm_backward_weight<<<1, 256>>>(v[1], v[4], g[3], g[2]);
  CHECK(hipGetLastError());
    break;
  }
  case 2: {
  linear_backward_input_hidden<<<dim3(4, 1), dim3(16, 16)>>>(g[6], v[5], g[3]);
  CHECK(hipGetLastError());
  linear_backward_weight_hidden<<<dim3(4, 4), dim3(16, 16)>>>(g[6], v[3], g[5]);
  CHECK(hipGetLastError());
    break;
  }
  case 3: {
  linear_backward_input_kv<<<dim3(4, 1), dim3(16, 16)>>>(g[8], v[7], g[3]);
  CHECK(hipGetLastError());
  linear_backward_weight_kv<<<dim3(4, 1), dim3(16, 16)>>>(g[8], v[3], g[7]);
  CHECK(hipGetLastError());
    break;
  }
  case 4: {
  linear_backward_input_kv<<<dim3(4, 1), dim3(16, 16)>>>(g[10], v[9], g[3]);
  CHECK(hipGetLastError());
  linear_backward_weight_kv<<<dim3(4, 1), dim3(16, 16)>>>(g[10], v[3], g[9]);
  CHECK(hipGetLastError());
    break;
  }
  case 5: {
  rope_backward_q<<<2, 256>>>(g[11], g[6]);
  CHECK(hipGetLastError());
    break;
  }
  case 6: {
  rope_backward_k<<<1, 256>>>(g[12], g[8]);
  CHECK(hipGetLastError());
    break;
  }
  case 7: {
  attention_scores_backward_q<<<4, 256>>>(g[13], v[12], g[11]);
  CHECK(hipGetLastError());
  attention_scores_backward_k<<<1, 256>>>(g[13], v[11], g[12]);
  CHECK(hipGetLastError());
    break;
  }
  case 8: {
  softmax_backward<<<64, 256>>>(v[14], g[14], g[13]);
  CHECK(hipGetLastError());
    break;
  }
  case 9: {
  attention_values_backward_p<<<2, 256>>>(g[15], v[10], g[14]);
  CHECK(hipGetLastError());
  attention_values_backward_v<<<1, 256>>>(g[15], v[14], g[10]);
  CHECK(hipGetLastError());
    break;
  }
  case 10: {
  linear_backward_input_hidden<<<dim3(4, 1), dim3(16, 16)>>>(g[17], v[16], g[15]);
  CHECK(hipGetLastError());
  linear_backward_weight_hidden<<<dim3(4, 4), dim3(16, 16)>>>(g[17], v[15], g[16]);
  CHECK(hipGetLastError());
    break;
  }
  case 11: {
  add_backward<<<4, 256>>>(g[18], g[1], g[17]);
  CHECK(hipGetLastError());
    break;
  }
  case 12: {
  rmsnorm_backward_input<<<16, 256>>>(v[18], v[19], v[21], g[20], g[18]);
  CHECK(hipGetLastError());
  rmsnorm_backward_weight<<<1, 256>>>(v[18], v[21], g[20], g[19]);
  CHECK(hipGetLastError());
    break;
  }
  case 13: {
  linear_backward_input_mlp_up<<<dim3(4, 1), dim3(16, 16)>>>(g[23], v[22], g[20]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_up<<<dim3(4, 16), dim3(16, 16)>>>(g[23], v[20], g[22]);
  CHECK(hipGetLastError());
    break;
  }
  case 14: {
  linear_backward_input_mlp_up<<<dim3(4, 1), dim3(16, 16)>>>(g[25], v[24], g[20]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_up<<<dim3(4, 16), dim3(16, 16)>>>(g[25], v[20], g[24]);
  CHECK(hipGetLastError());
    break;
  }
  case 15: {
  swiglu_backward<<<16, 256>>>(v[23], v[25], g[26], g[23], g[25]);
  CHECK(hipGetLastError());
    break;
  }
  case 16: {
  linear_backward_input_mlp_down<<<dim3(16, 1), dim3(16, 16)>>>(g[28], v[27], g[26]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_down<<<dim3(16, 4), dim3(16, 16)>>>(g[28], v[26], g[27]);
  CHECK(hipGetLastError());
    break;
  }
  case 17: {
  add_backward<<<4, 256>>>(g[29], g[18], g[28]);
  CHECK(hipGetLastError());
    break;
  }
  case 18: {
  rmsnorm_backward_input<<<16, 256>>>(v[29], v[30], v[32], g[31], g[29]);
  CHECK(hipGetLastError());
  rmsnorm_backward_weight<<<1, 256>>>(v[29], v[32], g[31], g[30]);
  CHECK(hipGetLastError());
    break;
  }
  case 19: {
  linear_backward_input_hidden<<<dim3(4, 1), dim3(16, 16)>>>(g[34], v[33], g[31]);
  CHECK(hipGetLastError());
  linear_backward_weight_hidden<<<dim3(4, 4), dim3(16, 16)>>>(g[34], v[31], g[33]);
  CHECK(hipGetLastError());
    break;
  }
  case 20: {
  linear_backward_input_kv<<<dim3(4, 1), dim3(16, 16)>>>(g[36], v[35], g[31]);
  CHECK(hipGetLastError());
  linear_backward_weight_kv<<<dim3(4, 1), dim3(16, 16)>>>(g[36], v[31], g[35]);
  CHECK(hipGetLastError());
    break;
  }
  case 21: {
  linear_backward_input_kv<<<dim3(4, 1), dim3(16, 16)>>>(g[38], v[37], g[31]);
  CHECK(hipGetLastError());
  linear_backward_weight_kv<<<dim3(4, 1), dim3(16, 16)>>>(g[38], v[31], g[37]);
  CHECK(hipGetLastError());
    break;
  }
  case 22: {
  rope_backward_q<<<2, 256>>>(g[39], g[34]);
  CHECK(hipGetLastError());
    break;
  }
  case 23: {
  rope_backward_k<<<1, 256>>>(g[40], g[36]);
  CHECK(hipGetLastError());
    break;
  }
  case 24: {
  attention_scores_backward_q<<<4, 256>>>(g[41], v[40], g[39]);
  CHECK(hipGetLastError());
  attention_scores_backward_k<<<1, 256>>>(g[41], v[39], g[40]);
  CHECK(hipGetLastError());
    break;
  }
  case 25: {
  softmax_backward<<<64, 256>>>(v[42], g[42], g[41]);
  CHECK(hipGetLastError());
    break;
  }
  case 26: {
  attention_values_backward_p<<<2, 256>>>(g[43], v[38], g[42]);
  CHECK(hipGetLastError());
  attention_values_backward_v<<<1, 256>>>(g[43], v[42], g[38]);
  CHECK(hipGetLastError());
    break;
  }
  case 27: {
  linear_backward_input_hidden<<<dim3(4, 1), dim3(16, 16)>>>(g[45], v[44], g[43]);
  CHECK(hipGetLastError());
  linear_backward_weight_hidden<<<dim3(4, 4), dim3(16, 16)>>>(g[45], v[43], g[44]);
  CHECK(hipGetLastError());
    break;
  }
  case 28: {
  add_backward<<<4, 256>>>(g[46], g[29], g[45]);
  CHECK(hipGetLastError());
    break;
  }
  case 29: {
  rmsnorm_backward_input<<<16, 256>>>(v[46], v[47], v[49], g[48], g[46]);
  CHECK(hipGetLastError());
  rmsnorm_backward_weight<<<1, 256>>>(v[46], v[49], g[48], g[47]);
  CHECK(hipGetLastError());
    break;
  }
  case 30: {
  linear_backward_input_mlp_up<<<dim3(4, 1), dim3(16, 16)>>>(g[51], v[50], g[48]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_up<<<dim3(4, 16), dim3(16, 16)>>>(g[51], v[48], g[50]);
  CHECK(hipGetLastError());
    break;
  }
  case 31: {
  linear_backward_input_mlp_up<<<dim3(4, 1), dim3(16, 16)>>>(g[53], v[52], g[48]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_up<<<dim3(4, 16), dim3(16, 16)>>>(g[53], v[48], g[52]);
  CHECK(hipGetLastError());
    break;
  }
  case 32: {
  swiglu_backward<<<16, 256>>>(v[51], v[53], g[54], g[51], g[53]);
  CHECK(hipGetLastError());
    break;
  }
  case 33: {
  linear_backward_input_mlp_down<<<dim3(16, 1), dim3(16, 16)>>>(g[56], v[55], g[54]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_down<<<dim3(16, 4), dim3(16, 16)>>>(g[56], v[54], g[55]);
  CHECK(hipGetLastError());
    break;
  }
  case 34: {
  add_backward<<<4, 256>>>(g[57], g[46], g[56]);
  CHECK(hipGetLastError());
    break;
  }
  case 35: {
  rmsnorm_backward_input<<<16, 256>>>(v[57], v[58], v[60], g[59], g[57]);
  CHECK(hipGetLastError());
  rmsnorm_backward_weight<<<1, 256>>>(v[57], v[60], g[59], g[58]);
  CHECK(hipGetLastError());
    break;
  }
  case 36: {
  linear_backward_input_lm_head<<<dim3(4, 1), dim3(16, 16)>>>(g[61], v[0], g[59]);
  CHECK(hipGetLastError());
  linear_backward_weight_lm_head<<<dim3(4, 8), dim3(16, 16)>>>(g[61], v[59], g[0]);
  CHECK(hipGetLastError());
    break;
  }
  case 37: {
  cross_entropy_backward<<<16, 256>>>(v[61], targets, g[62], g[61]);
  CHECK(hipGetLastError());
    break;
  }
  case 38: {
  mean_backward<<<1, 256>>>(g[63], g[62]);
  CHECK(hipGetLastError());
    break;
  }
  default: return int(hipErrorInvalidValue);
  }
  return hipDeviceSynchronize();
}
