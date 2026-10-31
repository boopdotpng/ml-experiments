#include "kernels.hip"
#define CHECK(expr) do { hipError_t e=(hipError_t)(expr); if(e!=hipSuccess) return int(e); } while(0)
#include "forward.hip"
#include "backward.hip"
extern "C" int run_sgd(float** v, float** g, float lr) {
  // embedding.weight
  sgd_update_262668288<<<1026048, 256>>>(v[0], g[0], lr);
  CHECK(hipGetLastError());
  // layer0.attention_norm.weight
  sgd_update_2048<<<8, 256>>>(v[2], g[2], lr);
  CHECK(hipGetLastError());
  // layer0.q.weight
  sgd_update_4194304<<<16384, 256>>>(v[5], g[5], lr);
  CHECK(hipGetLastError());
  // layer0.k.weight
  sgd_update_1048576<<<4096, 256>>>(v[7], g[7], lr);
  CHECK(hipGetLastError());
  // layer0.v.weight
  sgd_update_1048576<<<4096, 256>>>(v[9], g[9], lr);
  CHECK(hipGetLastError());
  // layer0.o.weight
  sgd_update_4194304<<<16384, 256>>>(v[16], g[16], lr);
  CHECK(hipGetLastError());
  // layer0.ffn_norm.weight
  sgd_update_2048<<<8, 256>>>(v[19], g[19], lr);
  CHECK(hipGetLastError());
  // layer0.gate.weight
  sgd_update_16777216<<<65536, 256>>>(v[22], g[22], lr);
  CHECK(hipGetLastError());
  // layer0.up.weight
  sgd_update_16777216<<<65536, 256>>>(v[24], g[24], lr);
  CHECK(hipGetLastError());
  // layer0.down.weight
  sgd_update_16777216<<<65536, 256>>>(v[27], g[27], lr);
  CHECK(hipGetLastError());
  // layer1.attention_norm.weight
  sgd_update_2048<<<8, 256>>>(v[30], g[30], lr);
  CHECK(hipGetLastError());
  // layer1.q.weight
  sgd_update_4194304<<<16384, 256>>>(v[33], g[33], lr);
  CHECK(hipGetLastError());
  // layer1.k.weight
  sgd_update_1048576<<<4096, 256>>>(v[35], g[35], lr);
  CHECK(hipGetLastError());
  // layer1.v.weight
  sgd_update_1048576<<<4096, 256>>>(v[37], g[37], lr);
  CHECK(hipGetLastError());
  // layer1.o.weight
  sgd_update_4194304<<<16384, 256>>>(v[44], g[44], lr);
  CHECK(hipGetLastError());
  // layer1.ffn_norm.weight
  sgd_update_2048<<<8, 256>>>(v[47], g[47], lr);
  CHECK(hipGetLastError());
  // layer1.gate.weight
  sgd_update_16777216<<<65536, 256>>>(v[50], g[50], lr);
  CHECK(hipGetLastError());
  // layer1.up.weight
  sgd_update_16777216<<<65536, 256>>>(v[52], g[52], lr);
  CHECK(hipGetLastError());
  // layer1.down.weight
  sgd_update_16777216<<<65536, 256>>>(v[55], g[55], lr);
  CHECK(hipGetLastError());
  // layer2.attention_norm.weight
  sgd_update_2048<<<8, 256>>>(v[58], g[58], lr);
  CHECK(hipGetLastError());
  // layer2.q.weight
  sgd_update_4194304<<<16384, 256>>>(v[61], g[61], lr);
  CHECK(hipGetLastError());
  // layer2.k.weight
  sgd_update_1048576<<<4096, 256>>>(v[63], g[63], lr);
  CHECK(hipGetLastError());
  // layer2.v.weight
  sgd_update_1048576<<<4096, 256>>>(v[65], g[65], lr);
  CHECK(hipGetLastError());
  // layer2.o.weight
  sgd_update_4194304<<<16384, 256>>>(v[72], g[72], lr);
  CHECK(hipGetLastError());
  // layer2.ffn_norm.weight
  sgd_update_2048<<<8, 256>>>(v[75], g[75], lr);
  CHECK(hipGetLastError());
  // layer2.gate.weight
  sgd_update_16777216<<<65536, 256>>>(v[78], g[78], lr);
  CHECK(hipGetLastError());
  // layer2.up.weight
  sgd_update_16777216<<<65536, 256>>>(v[80], g[80], lr);
  CHECK(hipGetLastError());
  // layer2.down.weight
  sgd_update_16777216<<<65536, 256>>>(v[83], g[83], lr);
  CHECK(hipGetLastError());
  // layer3.attention_norm.weight
  sgd_update_2048<<<8, 256>>>(v[86], g[86], lr);
  CHECK(hipGetLastError());
  // layer3.q.weight
  sgd_update_4194304<<<16384, 256>>>(v[89], g[89], lr);
  CHECK(hipGetLastError());
  // layer3.k.weight
  sgd_update_1048576<<<4096, 256>>>(v[91], g[91], lr);
  CHECK(hipGetLastError());
  // layer3.v.weight
  sgd_update_1048576<<<4096, 256>>>(v[93], g[93], lr);
  CHECK(hipGetLastError());
  // layer3.o.weight
  sgd_update_4194304<<<16384, 256>>>(v[100], g[100], lr);
  CHECK(hipGetLastError());
  // layer3.ffn_norm.weight
  sgd_update_2048<<<8, 256>>>(v[103], g[103], lr);
  CHECK(hipGetLastError());
  // layer3.gate.weight
  sgd_update_16777216<<<65536, 256>>>(v[106], g[106], lr);
  CHECK(hipGetLastError());
  // layer3.up.weight
  sgd_update_16777216<<<65536, 256>>>(v[108], g[108], lr);
  CHECK(hipGetLastError());
  // layer3.down.weight
  sgd_update_16777216<<<65536, 256>>>(v[111], g[111], lr);
  CHECK(hipGetLastError());
  // layer4.attention_norm.weight
  sgd_update_2048<<<8, 256>>>(v[114], g[114], lr);
  CHECK(hipGetLastError());
  // layer4.q.weight
  sgd_update_4194304<<<16384, 256>>>(v[117], g[117], lr);
  CHECK(hipGetLastError());
  // layer4.k.weight
  sgd_update_1048576<<<4096, 256>>>(v[119], g[119], lr);
  CHECK(hipGetLastError());
  // layer4.v.weight
  sgd_update_1048576<<<4096, 256>>>(v[121], g[121], lr);
  CHECK(hipGetLastError());
  // layer4.o.weight
  sgd_update_4194304<<<16384, 256>>>(v[128], g[128], lr);
  CHECK(hipGetLastError());
  // layer4.ffn_norm.weight
  sgd_update_2048<<<8, 256>>>(v[131], g[131], lr);
  CHECK(hipGetLastError());
  // layer4.gate.weight
  sgd_update_16777216<<<65536, 256>>>(v[134], g[134], lr);
  CHECK(hipGetLastError());
  // layer4.up.weight
  sgd_update_16777216<<<65536, 256>>>(v[136], g[136], lr);
  CHECK(hipGetLastError());
  // layer4.down.weight
  sgd_update_16777216<<<65536, 256>>>(v[139], g[139], lr);
  CHECK(hipGetLastError());
  // layer5.attention_norm.weight
  sgd_update_2048<<<8, 256>>>(v[142], g[142], lr);
  CHECK(hipGetLastError());
  // layer5.q.weight
  sgd_update_4194304<<<16384, 256>>>(v[145], g[145], lr);
  CHECK(hipGetLastError());
  // layer5.k.weight
  sgd_update_1048576<<<4096, 256>>>(v[147], g[147], lr);
  CHECK(hipGetLastError());
  // layer5.v.weight
  sgd_update_1048576<<<4096, 256>>>(v[149], g[149], lr);
  CHECK(hipGetLastError());
  // layer5.o.weight
  sgd_update_4194304<<<16384, 256>>>(v[156], g[156], lr);
  CHECK(hipGetLastError());
  // layer5.ffn_norm.weight
  sgd_update_2048<<<8, 256>>>(v[159], g[159], lr);
  CHECK(hipGetLastError());
  // layer5.gate.weight
  sgd_update_16777216<<<65536, 256>>>(v[162], g[162], lr);
  CHECK(hipGetLastError());
  // layer5.up.weight
  sgd_update_16777216<<<65536, 256>>>(v[164], g[164], lr);
  CHECK(hipGetLastError());
  // layer5.down.weight
  sgd_update_16777216<<<65536, 256>>>(v[167], g[167], lr);
  CHECK(hipGetLastError());
  // layer6.attention_norm.weight
  sgd_update_2048<<<8, 256>>>(v[170], g[170], lr);
  CHECK(hipGetLastError());
  // layer6.q.weight
  sgd_update_4194304<<<16384, 256>>>(v[173], g[173], lr);
  CHECK(hipGetLastError());
  // layer6.k.weight
  sgd_update_1048576<<<4096, 256>>>(v[175], g[175], lr);
  CHECK(hipGetLastError());
  // layer6.v.weight
  sgd_update_1048576<<<4096, 256>>>(v[177], g[177], lr);
  CHECK(hipGetLastError());
  // layer6.o.weight
  sgd_update_4194304<<<16384, 256>>>(v[184], g[184], lr);
  CHECK(hipGetLastError());
  // layer6.ffn_norm.weight
  sgd_update_2048<<<8, 256>>>(v[187], g[187], lr);
  CHECK(hipGetLastError());
  // layer6.gate.weight
  sgd_update_16777216<<<65536, 256>>>(v[190], g[190], lr);
  CHECK(hipGetLastError());
  // layer6.up.weight
  sgd_update_16777216<<<65536, 256>>>(v[192], g[192], lr);
  CHECK(hipGetLastError());
  // layer6.down.weight
  sgd_update_16777216<<<65536, 256>>>(v[195], g[195], lr);
  CHECK(hipGetLastError());
  // layer7.attention_norm.weight
  sgd_update_2048<<<8, 256>>>(v[198], g[198], lr);
  CHECK(hipGetLastError());
  // layer7.q.weight
  sgd_update_4194304<<<16384, 256>>>(v[201], g[201], lr);
  CHECK(hipGetLastError());
  // layer7.k.weight
  sgd_update_1048576<<<4096, 256>>>(v[203], g[203], lr);
  CHECK(hipGetLastError());
  // layer7.v.weight
  sgd_update_1048576<<<4096, 256>>>(v[205], g[205], lr);
  CHECK(hipGetLastError());
  // layer7.o.weight
  sgd_update_4194304<<<16384, 256>>>(v[212], g[212], lr);
  CHECK(hipGetLastError());
  // layer7.ffn_norm.weight
  sgd_update_2048<<<8, 256>>>(v[215], g[215], lr);
  CHECK(hipGetLastError());
  // layer7.gate.weight
  sgd_update_16777216<<<65536, 256>>>(v[218], g[218], lr);
  CHECK(hipGetLastError());
  // layer7.up.weight
  sgd_update_16777216<<<65536, 256>>>(v[220], g[220], lr);
  CHECK(hipGetLastError());
  // layer7.down.weight
  sgd_update_16777216<<<65536, 256>>>(v[223], g[223], lr);
  CHECK(hipGetLastError());
  // layer8.attention_norm.weight
  sgd_update_2048<<<8, 256>>>(v[226], g[226], lr);
  CHECK(hipGetLastError());
  // layer8.q.weight
  sgd_update_4194304<<<16384, 256>>>(v[229], g[229], lr);
  CHECK(hipGetLastError());
  // layer8.k.weight
  sgd_update_1048576<<<4096, 256>>>(v[231], g[231], lr);
  CHECK(hipGetLastError());
  // layer8.v.weight
  sgd_update_1048576<<<4096, 256>>>(v[233], g[233], lr);
  CHECK(hipGetLastError());
  // layer8.o.weight
  sgd_update_4194304<<<16384, 256>>>(v[240], g[240], lr);
  CHECK(hipGetLastError());
  // layer8.ffn_norm.weight
  sgd_update_2048<<<8, 256>>>(v[243], g[243], lr);
  CHECK(hipGetLastError());
  // layer8.gate.weight
  sgd_update_16777216<<<65536, 256>>>(v[246], g[246], lr);
  CHECK(hipGetLastError());
  // layer8.up.weight
  sgd_update_16777216<<<65536, 256>>>(v[248], g[248], lr);
  CHECK(hipGetLastError());
  // layer8.down.weight
  sgd_update_16777216<<<65536, 256>>>(v[251], g[251], lr);
  CHECK(hipGetLastError());
  // layer9.attention_norm.weight
  sgd_update_2048<<<8, 256>>>(v[254], g[254], lr);
  CHECK(hipGetLastError());
  // layer9.q.weight
  sgd_update_4194304<<<16384, 256>>>(v[257], g[257], lr);
  CHECK(hipGetLastError());
  // layer9.k.weight
  sgd_update_1048576<<<4096, 256>>>(v[259], g[259], lr);
  CHECK(hipGetLastError());
  // layer9.v.weight
  sgd_update_1048576<<<4096, 256>>>(v[261], g[261], lr);
  CHECK(hipGetLastError());
  // layer9.o.weight
  sgd_update_4194304<<<16384, 256>>>(v[268], g[268], lr);
  CHECK(hipGetLastError());
  // layer9.ffn_norm.weight
  sgd_update_2048<<<8, 256>>>(v[271], g[271], lr);
  CHECK(hipGetLastError());
  // layer9.gate.weight
  sgd_update_16777216<<<65536, 256>>>(v[274], g[274], lr);
  CHECK(hipGetLastError());
  // layer9.up.weight
  sgd_update_16777216<<<65536, 256>>>(v[276], g[276], lr);
  CHECK(hipGetLastError());
  // layer9.down.weight
  sgd_update_16777216<<<65536, 256>>>(v[279], g[279], lr);
  CHECK(hipGetLastError());
  // layer10.attention_norm.weight
  sgd_update_2048<<<8, 256>>>(v[282], g[282], lr);
  CHECK(hipGetLastError());
  // layer10.q.weight
  sgd_update_4194304<<<16384, 256>>>(v[285], g[285], lr);
  CHECK(hipGetLastError());
  // layer10.k.weight
  sgd_update_1048576<<<4096, 256>>>(v[287], g[287], lr);
  CHECK(hipGetLastError());
  // layer10.v.weight
  sgd_update_1048576<<<4096, 256>>>(v[289], g[289], lr);
  CHECK(hipGetLastError());
  // layer10.o.weight
  sgd_update_4194304<<<16384, 256>>>(v[296], g[296], lr);
  CHECK(hipGetLastError());
  // layer10.ffn_norm.weight
  sgd_update_2048<<<8, 256>>>(v[299], g[299], lr);
  CHECK(hipGetLastError());
  // layer10.gate.weight
  sgd_update_16777216<<<65536, 256>>>(v[302], g[302], lr);
  CHECK(hipGetLastError());
  // layer10.up.weight
  sgd_update_16777216<<<65536, 256>>>(v[304], g[304], lr);
  CHECK(hipGetLastError());
  // layer10.down.weight
  sgd_update_16777216<<<65536, 256>>>(v[307], g[307], lr);
  CHECK(hipGetLastError());
  // layer11.attention_norm.weight
  sgd_update_2048<<<8, 256>>>(v[310], g[310], lr);
  CHECK(hipGetLastError());
  // layer11.q.weight
  sgd_update_4194304<<<16384, 256>>>(v[313], g[313], lr);
  CHECK(hipGetLastError());
  // layer11.k.weight
  sgd_update_1048576<<<4096, 256>>>(v[315], g[315], lr);
  CHECK(hipGetLastError());
  // layer11.v.weight
  sgd_update_1048576<<<4096, 256>>>(v[317], g[317], lr);
  CHECK(hipGetLastError());
  // layer11.o.weight
  sgd_update_4194304<<<16384, 256>>>(v[324], g[324], lr);
  CHECK(hipGetLastError());
  // layer11.ffn_norm.weight
  sgd_update_2048<<<8, 256>>>(v[327], g[327], lr);
  CHECK(hipGetLastError());
  // layer11.gate.weight
  sgd_update_16777216<<<65536, 256>>>(v[330], g[330], lr);
  CHECK(hipGetLastError());
  // layer11.up.weight
  sgd_update_16777216<<<65536, 256>>>(v[332], g[332], lr);
  CHECK(hipGetLastError());
  // layer11.down.weight
  sgd_update_16777216<<<65536, 256>>>(v[335], g[335], lr);
  CHECK(hipGetLastError());
  // layer12.attention_norm.weight
  sgd_update_2048<<<8, 256>>>(v[338], g[338], lr);
  CHECK(hipGetLastError());
  // layer12.q.weight
  sgd_update_4194304<<<16384, 256>>>(v[341], g[341], lr);
  CHECK(hipGetLastError());
  // layer12.k.weight
  sgd_update_1048576<<<4096, 256>>>(v[343], g[343], lr);
  CHECK(hipGetLastError());
  // layer12.v.weight
  sgd_update_1048576<<<4096, 256>>>(v[345], g[345], lr);
  CHECK(hipGetLastError());
  // layer12.o.weight
  sgd_update_4194304<<<16384, 256>>>(v[352], g[352], lr);
  CHECK(hipGetLastError());
  // layer12.ffn_norm.weight
  sgd_update_2048<<<8, 256>>>(v[355], g[355], lr);
  CHECK(hipGetLastError());
  // layer12.gate.weight
  sgd_update_16777216<<<65536, 256>>>(v[358], g[358], lr);
  CHECK(hipGetLastError());
  // layer12.up.weight
  sgd_update_16777216<<<65536, 256>>>(v[360], g[360], lr);
  CHECK(hipGetLastError());
  // layer12.down.weight
  sgd_update_16777216<<<65536, 256>>>(v[363], g[363], lr);
  CHECK(hipGetLastError());
  // layer13.attention_norm.weight
  sgd_update_2048<<<8, 256>>>(v[366], g[366], lr);
  CHECK(hipGetLastError());
  // layer13.q.weight
  sgd_update_4194304<<<16384, 256>>>(v[369], g[369], lr);
  CHECK(hipGetLastError());
  // layer13.k.weight
  sgd_update_1048576<<<4096, 256>>>(v[371], g[371], lr);
  CHECK(hipGetLastError());
  // layer13.v.weight
  sgd_update_1048576<<<4096, 256>>>(v[373], g[373], lr);
  CHECK(hipGetLastError());
  // layer13.o.weight
  sgd_update_4194304<<<16384, 256>>>(v[380], g[380], lr);
  CHECK(hipGetLastError());
  // layer13.ffn_norm.weight
  sgd_update_2048<<<8, 256>>>(v[383], g[383], lr);
  CHECK(hipGetLastError());
  // layer13.gate.weight
  sgd_update_16777216<<<65536, 256>>>(v[386], g[386], lr);
  CHECK(hipGetLastError());
  // layer13.up.weight
  sgd_update_16777216<<<65536, 256>>>(v[388], g[388], lr);
  CHECK(hipGetLastError());
  // layer13.down.weight
  sgd_update_16777216<<<65536, 256>>>(v[391], g[391], lr);
  CHECK(hipGetLastError());
  // layer14.attention_norm.weight
  sgd_update_2048<<<8, 256>>>(v[394], g[394], lr);
  CHECK(hipGetLastError());
  // layer14.q.weight
  sgd_update_4194304<<<16384, 256>>>(v[397], g[397], lr);
  CHECK(hipGetLastError());
  // layer14.k.weight
  sgd_update_1048576<<<4096, 256>>>(v[399], g[399], lr);
  CHECK(hipGetLastError());
  // layer14.v.weight
  sgd_update_1048576<<<4096, 256>>>(v[401], g[401], lr);
  CHECK(hipGetLastError());
  // layer14.o.weight
  sgd_update_4194304<<<16384, 256>>>(v[408], g[408], lr);
  CHECK(hipGetLastError());
  // layer14.ffn_norm.weight
  sgd_update_2048<<<8, 256>>>(v[411], g[411], lr);
  CHECK(hipGetLastError());
  // layer14.gate.weight
  sgd_update_16777216<<<65536, 256>>>(v[414], g[414], lr);
  CHECK(hipGetLastError());
  // layer14.up.weight
  sgd_update_16777216<<<65536, 256>>>(v[416], g[416], lr);
  CHECK(hipGetLastError());
  // layer14.down.weight
  sgd_update_16777216<<<65536, 256>>>(v[419], g[419], lr);
  CHECK(hipGetLastError());
  // layer15.attention_norm.weight
  sgd_update_2048<<<8, 256>>>(v[422], g[422], lr);
  CHECK(hipGetLastError());
  // layer15.q.weight
  sgd_update_4194304<<<16384, 256>>>(v[425], g[425], lr);
  CHECK(hipGetLastError());
  // layer15.k.weight
  sgd_update_1048576<<<4096, 256>>>(v[427], g[427], lr);
  CHECK(hipGetLastError());
  // layer15.v.weight
  sgd_update_1048576<<<4096, 256>>>(v[429], g[429], lr);
  CHECK(hipGetLastError());
  // layer15.o.weight
  sgd_update_4194304<<<16384, 256>>>(v[436], g[436], lr);
  CHECK(hipGetLastError());
  // layer15.ffn_norm.weight
  sgd_update_2048<<<8, 256>>>(v[439], g[439], lr);
  CHECK(hipGetLastError());
  // layer15.gate.weight
  sgd_update_16777216<<<65536, 256>>>(v[442], g[442], lr);
  CHECK(hipGetLastError());
  // layer15.up.weight
  sgd_update_16777216<<<65536, 256>>>(v[444], g[444], lr);
  CHECK(hipGetLastError());
  // layer15.down.weight
  sgd_update_16777216<<<65536, 256>>>(v[447], g[447], lr);
  CHECK(hipGetLastError());
  // final_norm.weight
  sgd_update_2048<<<8, 256>>>(v[450], g[450], lr);
  CHECK(hipGetLastError());
  return hipDeviceSynchronize();
}
extern "C" int single_forward(int op, float** v, float** g, const int* ids, const int* targets) {
  switch(op) {
  case 0: {
  embedding_forward<<<64, 256>>>(v[0], ids, v[1]);
  CHECK(hipGetLastError());
    break;
  }
  case 1: {
  rmsnorm_forward<<<8, 256>>>(v[1], v[2], v[3], v[4]);
  CHECK(hipGetLastError());
    break;
  }
  case 2: {
  linear_forward_hidden<<<dim3(128, 1), dim3(16, 16)>>>(v[3], v[5], v[6]);
  CHECK(hipGetLastError());
    break;
  }
  case 3: {
  linear_forward_kv<<<dim3(32, 1), dim3(16, 16)>>>(v[3], v[7], v[8]);
  CHECK(hipGetLastError());
    break;
  }
  case 4: {
  linear_forward_kv<<<dim3(32, 1), dim3(16, 16)>>>(v[3], v[9], v[10]);
  CHECK(hipGetLastError());
    break;
  }
  case 5: {
  rope_forward_q<<<32, 256>>>(v[6], v[11]);
  CHECK(hipGetLastError());
    break;
  }
  case 6: {
  rope_forward_k<<<8, 256>>>(v[8], v[12]);
  CHECK(hipGetLastError());
    break;
  }
  case 7: {
  attention_scores_forward<<<8, 256>>>(v[11], v[12], v[13]);
  CHECK(hipGetLastError());
    break;
  }
  case 8: {
  softmax_forward<<<256, 256>>>(v[13], v[14]);
  CHECK(hipGetLastError());
    break;
  }
  case 9: {
  attention_values_forward<<<64, 256>>>(v[14], v[10], v[15]);
  CHECK(hipGetLastError());
    break;
  }
  case 10: {
  linear_forward_hidden<<<dim3(128, 1), dim3(16, 16)>>>(v[15], v[16], v[17]);
  CHECK(hipGetLastError());
    break;
  }
  case 11: {
  add_forward<<<64, 256>>>(v[1], v[17], v[18]);
  CHECK(hipGetLastError());
    break;
  }
  case 12: {
  rmsnorm_forward<<<8, 256>>>(v[18], v[19], v[20], v[21]);
  CHECK(hipGetLastError());
    break;
  }
  case 13: {
  linear_forward_mlp_up<<<dim3(512, 1), dim3(16, 16)>>>(v[20], v[22], v[23]);
  CHECK(hipGetLastError());
    break;
  }
  case 14: {
  linear_forward_mlp_up<<<dim3(512, 1), dim3(16, 16)>>>(v[20], v[24], v[25]);
  CHECK(hipGetLastError());
    break;
  }
  case 15: {
  swiglu_forward<<<256, 256>>>(v[23], v[25], v[26]);
  CHECK(hipGetLastError());
    break;
  }
  case 16: {
  linear_forward_mlp_down<<<dim3(128, 1), dim3(16, 16)>>>(v[26], v[27], v[28]);
  CHECK(hipGetLastError());
    break;
  }
  case 17: {
  add_forward<<<64, 256>>>(v[18], v[28], v[29]);
  CHECK(hipGetLastError());
    break;
  }
  case 18: {
  rmsnorm_forward<<<8, 256>>>(v[29], v[30], v[31], v[32]);
  CHECK(hipGetLastError());
    break;
  }
  case 19: {
  linear_forward_hidden<<<dim3(128, 1), dim3(16, 16)>>>(v[31], v[33], v[34]);
  CHECK(hipGetLastError());
    break;
  }
  case 20: {
  linear_forward_kv<<<dim3(32, 1), dim3(16, 16)>>>(v[31], v[35], v[36]);
  CHECK(hipGetLastError());
    break;
  }
  case 21: {
  linear_forward_kv<<<dim3(32, 1), dim3(16, 16)>>>(v[31], v[37], v[38]);
  CHECK(hipGetLastError());
    break;
  }
  case 22: {
  rope_forward_q<<<32, 256>>>(v[34], v[39]);
  CHECK(hipGetLastError());
    break;
  }
  case 23: {
  rope_forward_k<<<8, 256>>>(v[36], v[40]);
  CHECK(hipGetLastError());
    break;
  }
  case 24: {
  attention_scores_forward<<<8, 256>>>(v[39], v[40], v[41]);
  CHECK(hipGetLastError());
    break;
  }
  case 25: {
  softmax_forward<<<256, 256>>>(v[41], v[42]);
  CHECK(hipGetLastError());
    break;
  }
  case 26: {
  attention_values_forward<<<64, 256>>>(v[42], v[38], v[43]);
  CHECK(hipGetLastError());
    break;
  }
  case 27: {
  linear_forward_hidden<<<dim3(128, 1), dim3(16, 16)>>>(v[43], v[44], v[45]);
  CHECK(hipGetLastError());
    break;
  }
  case 28: {
  add_forward<<<64, 256>>>(v[29], v[45], v[46]);
  CHECK(hipGetLastError());
    break;
  }
  case 29: {
  rmsnorm_forward<<<8, 256>>>(v[46], v[47], v[48], v[49]);
  CHECK(hipGetLastError());
    break;
  }
  case 30: {
  linear_forward_mlp_up<<<dim3(512, 1), dim3(16, 16)>>>(v[48], v[50], v[51]);
  CHECK(hipGetLastError());
    break;
  }
  case 31: {
  linear_forward_mlp_up<<<dim3(512, 1), dim3(16, 16)>>>(v[48], v[52], v[53]);
  CHECK(hipGetLastError());
    break;
  }
  case 32: {
  swiglu_forward<<<256, 256>>>(v[51], v[53], v[54]);
  CHECK(hipGetLastError());
    break;
  }
  case 33: {
  linear_forward_mlp_down<<<dim3(128, 1), dim3(16, 16)>>>(v[54], v[55], v[56]);
  CHECK(hipGetLastError());
    break;
  }
  case 34: {
  add_forward<<<64, 256>>>(v[46], v[56], v[57]);
  CHECK(hipGetLastError());
    break;
  }
  case 35: {
  rmsnorm_forward<<<8, 256>>>(v[57], v[58], v[59], v[60]);
  CHECK(hipGetLastError());
    break;
  }
  case 36: {
  linear_forward_hidden<<<dim3(128, 1), dim3(16, 16)>>>(v[59], v[61], v[62]);
  CHECK(hipGetLastError());
    break;
  }
  case 37: {
  linear_forward_kv<<<dim3(32, 1), dim3(16, 16)>>>(v[59], v[63], v[64]);
  CHECK(hipGetLastError());
    break;
  }
  case 38: {
  linear_forward_kv<<<dim3(32, 1), dim3(16, 16)>>>(v[59], v[65], v[66]);
  CHECK(hipGetLastError());
    break;
  }
  case 39: {
  rope_forward_q<<<32, 256>>>(v[62], v[67]);
  CHECK(hipGetLastError());
    break;
  }
  case 40: {
  rope_forward_k<<<8, 256>>>(v[64], v[68]);
  CHECK(hipGetLastError());
    break;
  }
  case 41: {
  attention_scores_forward<<<8, 256>>>(v[67], v[68], v[69]);
  CHECK(hipGetLastError());
    break;
  }
  case 42: {
  softmax_forward<<<256, 256>>>(v[69], v[70]);
  CHECK(hipGetLastError());
    break;
  }
  case 43: {
  attention_values_forward<<<64, 256>>>(v[70], v[66], v[71]);
  CHECK(hipGetLastError());
    break;
  }
  case 44: {
  linear_forward_hidden<<<dim3(128, 1), dim3(16, 16)>>>(v[71], v[72], v[73]);
  CHECK(hipGetLastError());
    break;
  }
  case 45: {
  add_forward<<<64, 256>>>(v[57], v[73], v[74]);
  CHECK(hipGetLastError());
    break;
  }
  case 46: {
  rmsnorm_forward<<<8, 256>>>(v[74], v[75], v[76], v[77]);
  CHECK(hipGetLastError());
    break;
  }
  case 47: {
  linear_forward_mlp_up<<<dim3(512, 1), dim3(16, 16)>>>(v[76], v[78], v[79]);
  CHECK(hipGetLastError());
    break;
  }
  case 48: {
  linear_forward_mlp_up<<<dim3(512, 1), dim3(16, 16)>>>(v[76], v[80], v[81]);
  CHECK(hipGetLastError());
    break;
  }
  case 49: {
  swiglu_forward<<<256, 256>>>(v[79], v[81], v[82]);
  CHECK(hipGetLastError());
    break;
  }
  case 50: {
  linear_forward_mlp_down<<<dim3(128, 1), dim3(16, 16)>>>(v[82], v[83], v[84]);
  CHECK(hipGetLastError());
    break;
  }
  case 51: {
  add_forward<<<64, 256>>>(v[74], v[84], v[85]);
  CHECK(hipGetLastError());
    break;
  }
  case 52: {
  rmsnorm_forward<<<8, 256>>>(v[85], v[86], v[87], v[88]);
  CHECK(hipGetLastError());
    break;
  }
  case 53: {
  linear_forward_hidden<<<dim3(128, 1), dim3(16, 16)>>>(v[87], v[89], v[90]);
  CHECK(hipGetLastError());
    break;
  }
  case 54: {
  linear_forward_kv<<<dim3(32, 1), dim3(16, 16)>>>(v[87], v[91], v[92]);
  CHECK(hipGetLastError());
    break;
  }
  case 55: {
  linear_forward_kv<<<dim3(32, 1), dim3(16, 16)>>>(v[87], v[93], v[94]);
  CHECK(hipGetLastError());
    break;
  }
  case 56: {
  rope_forward_q<<<32, 256>>>(v[90], v[95]);
  CHECK(hipGetLastError());
    break;
  }
  case 57: {
  rope_forward_k<<<8, 256>>>(v[92], v[96]);
  CHECK(hipGetLastError());
    break;
  }
  case 58: {
  attention_scores_forward<<<8, 256>>>(v[95], v[96], v[97]);
  CHECK(hipGetLastError());
    break;
  }
  case 59: {
  softmax_forward<<<256, 256>>>(v[97], v[98]);
  CHECK(hipGetLastError());
    break;
  }
  case 60: {
  attention_values_forward<<<64, 256>>>(v[98], v[94], v[99]);
  CHECK(hipGetLastError());
    break;
  }
  case 61: {
  linear_forward_hidden<<<dim3(128, 1), dim3(16, 16)>>>(v[99], v[100], v[101]);
  CHECK(hipGetLastError());
    break;
  }
  case 62: {
  add_forward<<<64, 256>>>(v[85], v[101], v[102]);
  CHECK(hipGetLastError());
    break;
  }
  case 63: {
  rmsnorm_forward<<<8, 256>>>(v[102], v[103], v[104], v[105]);
  CHECK(hipGetLastError());
    break;
  }
  case 64: {
  linear_forward_mlp_up<<<dim3(512, 1), dim3(16, 16)>>>(v[104], v[106], v[107]);
  CHECK(hipGetLastError());
    break;
  }
  case 65: {
  linear_forward_mlp_up<<<dim3(512, 1), dim3(16, 16)>>>(v[104], v[108], v[109]);
  CHECK(hipGetLastError());
    break;
  }
  case 66: {
  swiglu_forward<<<256, 256>>>(v[107], v[109], v[110]);
  CHECK(hipGetLastError());
    break;
  }
  case 67: {
  linear_forward_mlp_down<<<dim3(128, 1), dim3(16, 16)>>>(v[110], v[111], v[112]);
  CHECK(hipGetLastError());
    break;
  }
  case 68: {
  add_forward<<<64, 256>>>(v[102], v[112], v[113]);
  CHECK(hipGetLastError());
    break;
  }
  case 69: {
  rmsnorm_forward<<<8, 256>>>(v[113], v[114], v[115], v[116]);
  CHECK(hipGetLastError());
    break;
  }
  case 70: {
  linear_forward_hidden<<<dim3(128, 1), dim3(16, 16)>>>(v[115], v[117], v[118]);
  CHECK(hipGetLastError());
    break;
  }
  case 71: {
  linear_forward_kv<<<dim3(32, 1), dim3(16, 16)>>>(v[115], v[119], v[120]);
  CHECK(hipGetLastError());
    break;
  }
  case 72: {
  linear_forward_kv<<<dim3(32, 1), dim3(16, 16)>>>(v[115], v[121], v[122]);
  CHECK(hipGetLastError());
    break;
  }
  case 73: {
  rope_forward_q<<<32, 256>>>(v[118], v[123]);
  CHECK(hipGetLastError());
    break;
  }
  case 74: {
  rope_forward_k<<<8, 256>>>(v[120], v[124]);
  CHECK(hipGetLastError());
    break;
  }
  case 75: {
  attention_scores_forward<<<8, 256>>>(v[123], v[124], v[125]);
  CHECK(hipGetLastError());
    break;
  }
  case 76: {
  softmax_forward<<<256, 256>>>(v[125], v[126]);
  CHECK(hipGetLastError());
    break;
  }
  case 77: {
  attention_values_forward<<<64, 256>>>(v[126], v[122], v[127]);
  CHECK(hipGetLastError());
    break;
  }
  case 78: {
  linear_forward_hidden<<<dim3(128, 1), dim3(16, 16)>>>(v[127], v[128], v[129]);
  CHECK(hipGetLastError());
    break;
  }
  case 79: {
  add_forward<<<64, 256>>>(v[113], v[129], v[130]);
  CHECK(hipGetLastError());
    break;
  }
  case 80: {
  rmsnorm_forward<<<8, 256>>>(v[130], v[131], v[132], v[133]);
  CHECK(hipGetLastError());
    break;
  }
  case 81: {
  linear_forward_mlp_up<<<dim3(512, 1), dim3(16, 16)>>>(v[132], v[134], v[135]);
  CHECK(hipGetLastError());
    break;
  }
  case 82: {
  linear_forward_mlp_up<<<dim3(512, 1), dim3(16, 16)>>>(v[132], v[136], v[137]);
  CHECK(hipGetLastError());
    break;
  }
  case 83: {
  swiglu_forward<<<256, 256>>>(v[135], v[137], v[138]);
  CHECK(hipGetLastError());
    break;
  }
  case 84: {
  linear_forward_mlp_down<<<dim3(128, 1), dim3(16, 16)>>>(v[138], v[139], v[140]);
  CHECK(hipGetLastError());
    break;
  }
  case 85: {
  add_forward<<<64, 256>>>(v[130], v[140], v[141]);
  CHECK(hipGetLastError());
    break;
  }
  case 86: {
  rmsnorm_forward<<<8, 256>>>(v[141], v[142], v[143], v[144]);
  CHECK(hipGetLastError());
    break;
  }
  case 87: {
  linear_forward_hidden<<<dim3(128, 1), dim3(16, 16)>>>(v[143], v[145], v[146]);
  CHECK(hipGetLastError());
    break;
  }
  case 88: {
  linear_forward_kv<<<dim3(32, 1), dim3(16, 16)>>>(v[143], v[147], v[148]);
  CHECK(hipGetLastError());
    break;
  }
  case 89: {
  linear_forward_kv<<<dim3(32, 1), dim3(16, 16)>>>(v[143], v[149], v[150]);
  CHECK(hipGetLastError());
    break;
  }
  case 90: {
  rope_forward_q<<<32, 256>>>(v[146], v[151]);
  CHECK(hipGetLastError());
    break;
  }
  case 91: {
  rope_forward_k<<<8, 256>>>(v[148], v[152]);
  CHECK(hipGetLastError());
    break;
  }
  case 92: {
  attention_scores_forward<<<8, 256>>>(v[151], v[152], v[153]);
  CHECK(hipGetLastError());
    break;
  }
  case 93: {
  softmax_forward<<<256, 256>>>(v[153], v[154]);
  CHECK(hipGetLastError());
    break;
  }
  case 94: {
  attention_values_forward<<<64, 256>>>(v[154], v[150], v[155]);
  CHECK(hipGetLastError());
    break;
  }
  case 95: {
  linear_forward_hidden<<<dim3(128, 1), dim3(16, 16)>>>(v[155], v[156], v[157]);
  CHECK(hipGetLastError());
    break;
  }
  case 96: {
  add_forward<<<64, 256>>>(v[141], v[157], v[158]);
  CHECK(hipGetLastError());
    break;
  }
  case 97: {
  rmsnorm_forward<<<8, 256>>>(v[158], v[159], v[160], v[161]);
  CHECK(hipGetLastError());
    break;
  }
  case 98: {
  linear_forward_mlp_up<<<dim3(512, 1), dim3(16, 16)>>>(v[160], v[162], v[163]);
  CHECK(hipGetLastError());
    break;
  }
  case 99: {
  linear_forward_mlp_up<<<dim3(512, 1), dim3(16, 16)>>>(v[160], v[164], v[165]);
  CHECK(hipGetLastError());
    break;
  }
  case 100: {
  swiglu_forward<<<256, 256>>>(v[163], v[165], v[166]);
  CHECK(hipGetLastError());
    break;
  }
  case 101: {
  linear_forward_mlp_down<<<dim3(128, 1), dim3(16, 16)>>>(v[166], v[167], v[168]);
  CHECK(hipGetLastError());
    break;
  }
  case 102: {
  add_forward<<<64, 256>>>(v[158], v[168], v[169]);
  CHECK(hipGetLastError());
    break;
  }
  case 103: {
  rmsnorm_forward<<<8, 256>>>(v[169], v[170], v[171], v[172]);
  CHECK(hipGetLastError());
    break;
  }
  case 104: {
  linear_forward_hidden<<<dim3(128, 1), dim3(16, 16)>>>(v[171], v[173], v[174]);
  CHECK(hipGetLastError());
    break;
  }
  case 105: {
  linear_forward_kv<<<dim3(32, 1), dim3(16, 16)>>>(v[171], v[175], v[176]);
  CHECK(hipGetLastError());
    break;
  }
  case 106: {
  linear_forward_kv<<<dim3(32, 1), dim3(16, 16)>>>(v[171], v[177], v[178]);
  CHECK(hipGetLastError());
    break;
  }
  case 107: {
  rope_forward_q<<<32, 256>>>(v[174], v[179]);
  CHECK(hipGetLastError());
    break;
  }
  case 108: {
  rope_forward_k<<<8, 256>>>(v[176], v[180]);
  CHECK(hipGetLastError());
    break;
  }
  case 109: {
  attention_scores_forward<<<8, 256>>>(v[179], v[180], v[181]);
  CHECK(hipGetLastError());
    break;
  }
  case 110: {
  softmax_forward<<<256, 256>>>(v[181], v[182]);
  CHECK(hipGetLastError());
    break;
  }
  case 111: {
  attention_values_forward<<<64, 256>>>(v[182], v[178], v[183]);
  CHECK(hipGetLastError());
    break;
  }
  case 112: {
  linear_forward_hidden<<<dim3(128, 1), dim3(16, 16)>>>(v[183], v[184], v[185]);
  CHECK(hipGetLastError());
    break;
  }
  case 113: {
  add_forward<<<64, 256>>>(v[169], v[185], v[186]);
  CHECK(hipGetLastError());
    break;
  }
  case 114: {
  rmsnorm_forward<<<8, 256>>>(v[186], v[187], v[188], v[189]);
  CHECK(hipGetLastError());
    break;
  }
  case 115: {
  linear_forward_mlp_up<<<dim3(512, 1), dim3(16, 16)>>>(v[188], v[190], v[191]);
  CHECK(hipGetLastError());
    break;
  }
  case 116: {
  linear_forward_mlp_up<<<dim3(512, 1), dim3(16, 16)>>>(v[188], v[192], v[193]);
  CHECK(hipGetLastError());
    break;
  }
  case 117: {
  swiglu_forward<<<256, 256>>>(v[191], v[193], v[194]);
  CHECK(hipGetLastError());
    break;
  }
  case 118: {
  linear_forward_mlp_down<<<dim3(128, 1), dim3(16, 16)>>>(v[194], v[195], v[196]);
  CHECK(hipGetLastError());
    break;
  }
  case 119: {
  add_forward<<<64, 256>>>(v[186], v[196], v[197]);
  CHECK(hipGetLastError());
    break;
  }
  case 120: {
  rmsnorm_forward<<<8, 256>>>(v[197], v[198], v[199], v[200]);
  CHECK(hipGetLastError());
    break;
  }
  case 121: {
  linear_forward_hidden<<<dim3(128, 1), dim3(16, 16)>>>(v[199], v[201], v[202]);
  CHECK(hipGetLastError());
    break;
  }
  case 122: {
  linear_forward_kv<<<dim3(32, 1), dim3(16, 16)>>>(v[199], v[203], v[204]);
  CHECK(hipGetLastError());
    break;
  }
  case 123: {
  linear_forward_kv<<<dim3(32, 1), dim3(16, 16)>>>(v[199], v[205], v[206]);
  CHECK(hipGetLastError());
    break;
  }
  case 124: {
  rope_forward_q<<<32, 256>>>(v[202], v[207]);
  CHECK(hipGetLastError());
    break;
  }
  case 125: {
  rope_forward_k<<<8, 256>>>(v[204], v[208]);
  CHECK(hipGetLastError());
    break;
  }
  case 126: {
  attention_scores_forward<<<8, 256>>>(v[207], v[208], v[209]);
  CHECK(hipGetLastError());
    break;
  }
  case 127: {
  softmax_forward<<<256, 256>>>(v[209], v[210]);
  CHECK(hipGetLastError());
    break;
  }
  case 128: {
  attention_values_forward<<<64, 256>>>(v[210], v[206], v[211]);
  CHECK(hipGetLastError());
    break;
  }
  case 129: {
  linear_forward_hidden<<<dim3(128, 1), dim3(16, 16)>>>(v[211], v[212], v[213]);
  CHECK(hipGetLastError());
    break;
  }
  case 130: {
  add_forward<<<64, 256>>>(v[197], v[213], v[214]);
  CHECK(hipGetLastError());
    break;
  }
  case 131: {
  rmsnorm_forward<<<8, 256>>>(v[214], v[215], v[216], v[217]);
  CHECK(hipGetLastError());
    break;
  }
  case 132: {
  linear_forward_mlp_up<<<dim3(512, 1), dim3(16, 16)>>>(v[216], v[218], v[219]);
  CHECK(hipGetLastError());
    break;
  }
  case 133: {
  linear_forward_mlp_up<<<dim3(512, 1), dim3(16, 16)>>>(v[216], v[220], v[221]);
  CHECK(hipGetLastError());
    break;
  }
  case 134: {
  swiglu_forward<<<256, 256>>>(v[219], v[221], v[222]);
  CHECK(hipGetLastError());
    break;
  }
  case 135: {
  linear_forward_mlp_down<<<dim3(128, 1), dim3(16, 16)>>>(v[222], v[223], v[224]);
  CHECK(hipGetLastError());
    break;
  }
  case 136: {
  add_forward<<<64, 256>>>(v[214], v[224], v[225]);
  CHECK(hipGetLastError());
    break;
  }
  case 137: {
  rmsnorm_forward<<<8, 256>>>(v[225], v[226], v[227], v[228]);
  CHECK(hipGetLastError());
    break;
  }
  case 138: {
  linear_forward_hidden<<<dim3(128, 1), dim3(16, 16)>>>(v[227], v[229], v[230]);
  CHECK(hipGetLastError());
    break;
  }
  case 139: {
  linear_forward_kv<<<dim3(32, 1), dim3(16, 16)>>>(v[227], v[231], v[232]);
  CHECK(hipGetLastError());
    break;
  }
  case 140: {
  linear_forward_kv<<<dim3(32, 1), dim3(16, 16)>>>(v[227], v[233], v[234]);
  CHECK(hipGetLastError());
    break;
  }
  case 141: {
  rope_forward_q<<<32, 256>>>(v[230], v[235]);
  CHECK(hipGetLastError());
    break;
  }
  case 142: {
  rope_forward_k<<<8, 256>>>(v[232], v[236]);
  CHECK(hipGetLastError());
    break;
  }
  case 143: {
  attention_scores_forward<<<8, 256>>>(v[235], v[236], v[237]);
  CHECK(hipGetLastError());
    break;
  }
  case 144: {
  softmax_forward<<<256, 256>>>(v[237], v[238]);
  CHECK(hipGetLastError());
    break;
  }
  case 145: {
  attention_values_forward<<<64, 256>>>(v[238], v[234], v[239]);
  CHECK(hipGetLastError());
    break;
  }
  case 146: {
  linear_forward_hidden<<<dim3(128, 1), dim3(16, 16)>>>(v[239], v[240], v[241]);
  CHECK(hipGetLastError());
    break;
  }
  case 147: {
  add_forward<<<64, 256>>>(v[225], v[241], v[242]);
  CHECK(hipGetLastError());
    break;
  }
  case 148: {
  rmsnorm_forward<<<8, 256>>>(v[242], v[243], v[244], v[245]);
  CHECK(hipGetLastError());
    break;
  }
  case 149: {
  linear_forward_mlp_up<<<dim3(512, 1), dim3(16, 16)>>>(v[244], v[246], v[247]);
  CHECK(hipGetLastError());
    break;
  }
  case 150: {
  linear_forward_mlp_up<<<dim3(512, 1), dim3(16, 16)>>>(v[244], v[248], v[249]);
  CHECK(hipGetLastError());
    break;
  }
  case 151: {
  swiglu_forward<<<256, 256>>>(v[247], v[249], v[250]);
  CHECK(hipGetLastError());
    break;
  }
  case 152: {
  linear_forward_mlp_down<<<dim3(128, 1), dim3(16, 16)>>>(v[250], v[251], v[252]);
  CHECK(hipGetLastError());
    break;
  }
  case 153: {
  add_forward<<<64, 256>>>(v[242], v[252], v[253]);
  CHECK(hipGetLastError());
    break;
  }
  case 154: {
  rmsnorm_forward<<<8, 256>>>(v[253], v[254], v[255], v[256]);
  CHECK(hipGetLastError());
    break;
  }
  case 155: {
  linear_forward_hidden<<<dim3(128, 1), dim3(16, 16)>>>(v[255], v[257], v[258]);
  CHECK(hipGetLastError());
    break;
  }
  case 156: {
  linear_forward_kv<<<dim3(32, 1), dim3(16, 16)>>>(v[255], v[259], v[260]);
  CHECK(hipGetLastError());
    break;
  }
  case 157: {
  linear_forward_kv<<<dim3(32, 1), dim3(16, 16)>>>(v[255], v[261], v[262]);
  CHECK(hipGetLastError());
    break;
  }
  case 158: {
  rope_forward_q<<<32, 256>>>(v[258], v[263]);
  CHECK(hipGetLastError());
    break;
  }
  case 159: {
  rope_forward_k<<<8, 256>>>(v[260], v[264]);
  CHECK(hipGetLastError());
    break;
  }
  case 160: {
  attention_scores_forward<<<8, 256>>>(v[263], v[264], v[265]);
  CHECK(hipGetLastError());
    break;
  }
  case 161: {
  softmax_forward<<<256, 256>>>(v[265], v[266]);
  CHECK(hipGetLastError());
    break;
  }
  case 162: {
  attention_values_forward<<<64, 256>>>(v[266], v[262], v[267]);
  CHECK(hipGetLastError());
    break;
  }
  case 163: {
  linear_forward_hidden<<<dim3(128, 1), dim3(16, 16)>>>(v[267], v[268], v[269]);
  CHECK(hipGetLastError());
    break;
  }
  case 164: {
  add_forward<<<64, 256>>>(v[253], v[269], v[270]);
  CHECK(hipGetLastError());
    break;
  }
  case 165: {
  rmsnorm_forward<<<8, 256>>>(v[270], v[271], v[272], v[273]);
  CHECK(hipGetLastError());
    break;
  }
  case 166: {
  linear_forward_mlp_up<<<dim3(512, 1), dim3(16, 16)>>>(v[272], v[274], v[275]);
  CHECK(hipGetLastError());
    break;
  }
  case 167: {
  linear_forward_mlp_up<<<dim3(512, 1), dim3(16, 16)>>>(v[272], v[276], v[277]);
  CHECK(hipGetLastError());
    break;
  }
  case 168: {
  swiglu_forward<<<256, 256>>>(v[275], v[277], v[278]);
  CHECK(hipGetLastError());
    break;
  }
  case 169: {
  linear_forward_mlp_down<<<dim3(128, 1), dim3(16, 16)>>>(v[278], v[279], v[280]);
  CHECK(hipGetLastError());
    break;
  }
  case 170: {
  add_forward<<<64, 256>>>(v[270], v[280], v[281]);
  CHECK(hipGetLastError());
    break;
  }
  case 171: {
  rmsnorm_forward<<<8, 256>>>(v[281], v[282], v[283], v[284]);
  CHECK(hipGetLastError());
    break;
  }
  case 172: {
  linear_forward_hidden<<<dim3(128, 1), dim3(16, 16)>>>(v[283], v[285], v[286]);
  CHECK(hipGetLastError());
    break;
  }
  case 173: {
  linear_forward_kv<<<dim3(32, 1), dim3(16, 16)>>>(v[283], v[287], v[288]);
  CHECK(hipGetLastError());
    break;
  }
  case 174: {
  linear_forward_kv<<<dim3(32, 1), dim3(16, 16)>>>(v[283], v[289], v[290]);
  CHECK(hipGetLastError());
    break;
  }
  case 175: {
  rope_forward_q<<<32, 256>>>(v[286], v[291]);
  CHECK(hipGetLastError());
    break;
  }
  case 176: {
  rope_forward_k<<<8, 256>>>(v[288], v[292]);
  CHECK(hipGetLastError());
    break;
  }
  case 177: {
  attention_scores_forward<<<8, 256>>>(v[291], v[292], v[293]);
  CHECK(hipGetLastError());
    break;
  }
  case 178: {
  softmax_forward<<<256, 256>>>(v[293], v[294]);
  CHECK(hipGetLastError());
    break;
  }
  case 179: {
  attention_values_forward<<<64, 256>>>(v[294], v[290], v[295]);
  CHECK(hipGetLastError());
    break;
  }
  case 180: {
  linear_forward_hidden<<<dim3(128, 1), dim3(16, 16)>>>(v[295], v[296], v[297]);
  CHECK(hipGetLastError());
    break;
  }
  case 181: {
  add_forward<<<64, 256>>>(v[281], v[297], v[298]);
  CHECK(hipGetLastError());
    break;
  }
  case 182: {
  rmsnorm_forward<<<8, 256>>>(v[298], v[299], v[300], v[301]);
  CHECK(hipGetLastError());
    break;
  }
  case 183: {
  linear_forward_mlp_up<<<dim3(512, 1), dim3(16, 16)>>>(v[300], v[302], v[303]);
  CHECK(hipGetLastError());
    break;
  }
  case 184: {
  linear_forward_mlp_up<<<dim3(512, 1), dim3(16, 16)>>>(v[300], v[304], v[305]);
  CHECK(hipGetLastError());
    break;
  }
  case 185: {
  swiglu_forward<<<256, 256>>>(v[303], v[305], v[306]);
  CHECK(hipGetLastError());
    break;
  }
  case 186: {
  linear_forward_mlp_down<<<dim3(128, 1), dim3(16, 16)>>>(v[306], v[307], v[308]);
  CHECK(hipGetLastError());
    break;
  }
  case 187: {
  add_forward<<<64, 256>>>(v[298], v[308], v[309]);
  CHECK(hipGetLastError());
    break;
  }
  case 188: {
  rmsnorm_forward<<<8, 256>>>(v[309], v[310], v[311], v[312]);
  CHECK(hipGetLastError());
    break;
  }
  case 189: {
  linear_forward_hidden<<<dim3(128, 1), dim3(16, 16)>>>(v[311], v[313], v[314]);
  CHECK(hipGetLastError());
    break;
  }
  case 190: {
  linear_forward_kv<<<dim3(32, 1), dim3(16, 16)>>>(v[311], v[315], v[316]);
  CHECK(hipGetLastError());
    break;
  }
  case 191: {
  linear_forward_kv<<<dim3(32, 1), dim3(16, 16)>>>(v[311], v[317], v[318]);
  CHECK(hipGetLastError());
    break;
  }
  case 192: {
  rope_forward_q<<<32, 256>>>(v[314], v[319]);
  CHECK(hipGetLastError());
    break;
  }
  case 193: {
  rope_forward_k<<<8, 256>>>(v[316], v[320]);
  CHECK(hipGetLastError());
    break;
  }
  case 194: {
  attention_scores_forward<<<8, 256>>>(v[319], v[320], v[321]);
  CHECK(hipGetLastError());
    break;
  }
  case 195: {
  softmax_forward<<<256, 256>>>(v[321], v[322]);
  CHECK(hipGetLastError());
    break;
  }
  case 196: {
  attention_values_forward<<<64, 256>>>(v[322], v[318], v[323]);
  CHECK(hipGetLastError());
    break;
  }
  case 197: {
  linear_forward_hidden<<<dim3(128, 1), dim3(16, 16)>>>(v[323], v[324], v[325]);
  CHECK(hipGetLastError());
    break;
  }
  case 198: {
  add_forward<<<64, 256>>>(v[309], v[325], v[326]);
  CHECK(hipGetLastError());
    break;
  }
  case 199: {
  rmsnorm_forward<<<8, 256>>>(v[326], v[327], v[328], v[329]);
  CHECK(hipGetLastError());
    break;
  }
  case 200: {
  linear_forward_mlp_up<<<dim3(512, 1), dim3(16, 16)>>>(v[328], v[330], v[331]);
  CHECK(hipGetLastError());
    break;
  }
  case 201: {
  linear_forward_mlp_up<<<dim3(512, 1), dim3(16, 16)>>>(v[328], v[332], v[333]);
  CHECK(hipGetLastError());
    break;
  }
  case 202: {
  swiglu_forward<<<256, 256>>>(v[331], v[333], v[334]);
  CHECK(hipGetLastError());
    break;
  }
  case 203: {
  linear_forward_mlp_down<<<dim3(128, 1), dim3(16, 16)>>>(v[334], v[335], v[336]);
  CHECK(hipGetLastError());
    break;
  }
  case 204: {
  add_forward<<<64, 256>>>(v[326], v[336], v[337]);
  CHECK(hipGetLastError());
    break;
  }
  case 205: {
  rmsnorm_forward<<<8, 256>>>(v[337], v[338], v[339], v[340]);
  CHECK(hipGetLastError());
    break;
  }
  case 206: {
  linear_forward_hidden<<<dim3(128, 1), dim3(16, 16)>>>(v[339], v[341], v[342]);
  CHECK(hipGetLastError());
    break;
  }
  case 207: {
  linear_forward_kv<<<dim3(32, 1), dim3(16, 16)>>>(v[339], v[343], v[344]);
  CHECK(hipGetLastError());
    break;
  }
  case 208: {
  linear_forward_kv<<<dim3(32, 1), dim3(16, 16)>>>(v[339], v[345], v[346]);
  CHECK(hipGetLastError());
    break;
  }
  case 209: {
  rope_forward_q<<<32, 256>>>(v[342], v[347]);
  CHECK(hipGetLastError());
    break;
  }
  case 210: {
  rope_forward_k<<<8, 256>>>(v[344], v[348]);
  CHECK(hipGetLastError());
    break;
  }
  case 211: {
  attention_scores_forward<<<8, 256>>>(v[347], v[348], v[349]);
  CHECK(hipGetLastError());
    break;
  }
  case 212: {
  softmax_forward<<<256, 256>>>(v[349], v[350]);
  CHECK(hipGetLastError());
    break;
  }
  case 213: {
  attention_values_forward<<<64, 256>>>(v[350], v[346], v[351]);
  CHECK(hipGetLastError());
    break;
  }
  case 214: {
  linear_forward_hidden<<<dim3(128, 1), dim3(16, 16)>>>(v[351], v[352], v[353]);
  CHECK(hipGetLastError());
    break;
  }
  case 215: {
  add_forward<<<64, 256>>>(v[337], v[353], v[354]);
  CHECK(hipGetLastError());
    break;
  }
  case 216: {
  rmsnorm_forward<<<8, 256>>>(v[354], v[355], v[356], v[357]);
  CHECK(hipGetLastError());
    break;
  }
  case 217: {
  linear_forward_mlp_up<<<dim3(512, 1), dim3(16, 16)>>>(v[356], v[358], v[359]);
  CHECK(hipGetLastError());
    break;
  }
  case 218: {
  linear_forward_mlp_up<<<dim3(512, 1), dim3(16, 16)>>>(v[356], v[360], v[361]);
  CHECK(hipGetLastError());
    break;
  }
  case 219: {
  swiglu_forward<<<256, 256>>>(v[359], v[361], v[362]);
  CHECK(hipGetLastError());
    break;
  }
  case 220: {
  linear_forward_mlp_down<<<dim3(128, 1), dim3(16, 16)>>>(v[362], v[363], v[364]);
  CHECK(hipGetLastError());
    break;
  }
  case 221: {
  add_forward<<<64, 256>>>(v[354], v[364], v[365]);
  CHECK(hipGetLastError());
    break;
  }
  case 222: {
  rmsnorm_forward<<<8, 256>>>(v[365], v[366], v[367], v[368]);
  CHECK(hipGetLastError());
    break;
  }
  case 223: {
  linear_forward_hidden<<<dim3(128, 1), dim3(16, 16)>>>(v[367], v[369], v[370]);
  CHECK(hipGetLastError());
    break;
  }
  case 224: {
  linear_forward_kv<<<dim3(32, 1), dim3(16, 16)>>>(v[367], v[371], v[372]);
  CHECK(hipGetLastError());
    break;
  }
  case 225: {
  linear_forward_kv<<<dim3(32, 1), dim3(16, 16)>>>(v[367], v[373], v[374]);
  CHECK(hipGetLastError());
    break;
  }
  case 226: {
  rope_forward_q<<<32, 256>>>(v[370], v[375]);
  CHECK(hipGetLastError());
    break;
  }
  case 227: {
  rope_forward_k<<<8, 256>>>(v[372], v[376]);
  CHECK(hipGetLastError());
    break;
  }
  case 228: {
  attention_scores_forward<<<8, 256>>>(v[375], v[376], v[377]);
  CHECK(hipGetLastError());
    break;
  }
  case 229: {
  softmax_forward<<<256, 256>>>(v[377], v[378]);
  CHECK(hipGetLastError());
    break;
  }
  case 230: {
  attention_values_forward<<<64, 256>>>(v[378], v[374], v[379]);
  CHECK(hipGetLastError());
    break;
  }
  case 231: {
  linear_forward_hidden<<<dim3(128, 1), dim3(16, 16)>>>(v[379], v[380], v[381]);
  CHECK(hipGetLastError());
    break;
  }
  case 232: {
  add_forward<<<64, 256>>>(v[365], v[381], v[382]);
  CHECK(hipGetLastError());
    break;
  }
  case 233: {
  rmsnorm_forward<<<8, 256>>>(v[382], v[383], v[384], v[385]);
  CHECK(hipGetLastError());
    break;
  }
  case 234: {
  linear_forward_mlp_up<<<dim3(512, 1), dim3(16, 16)>>>(v[384], v[386], v[387]);
  CHECK(hipGetLastError());
    break;
  }
  case 235: {
  linear_forward_mlp_up<<<dim3(512, 1), dim3(16, 16)>>>(v[384], v[388], v[389]);
  CHECK(hipGetLastError());
    break;
  }
  case 236: {
  swiglu_forward<<<256, 256>>>(v[387], v[389], v[390]);
  CHECK(hipGetLastError());
    break;
  }
  case 237: {
  linear_forward_mlp_down<<<dim3(128, 1), dim3(16, 16)>>>(v[390], v[391], v[392]);
  CHECK(hipGetLastError());
    break;
  }
  case 238: {
  add_forward<<<64, 256>>>(v[382], v[392], v[393]);
  CHECK(hipGetLastError());
    break;
  }
  case 239: {
  rmsnorm_forward<<<8, 256>>>(v[393], v[394], v[395], v[396]);
  CHECK(hipGetLastError());
    break;
  }
  case 240: {
  linear_forward_hidden<<<dim3(128, 1), dim3(16, 16)>>>(v[395], v[397], v[398]);
  CHECK(hipGetLastError());
    break;
  }
  case 241: {
  linear_forward_kv<<<dim3(32, 1), dim3(16, 16)>>>(v[395], v[399], v[400]);
  CHECK(hipGetLastError());
    break;
  }
  case 242: {
  linear_forward_kv<<<dim3(32, 1), dim3(16, 16)>>>(v[395], v[401], v[402]);
  CHECK(hipGetLastError());
    break;
  }
  case 243: {
  rope_forward_q<<<32, 256>>>(v[398], v[403]);
  CHECK(hipGetLastError());
    break;
  }
  case 244: {
  rope_forward_k<<<8, 256>>>(v[400], v[404]);
  CHECK(hipGetLastError());
    break;
  }
  case 245: {
  attention_scores_forward<<<8, 256>>>(v[403], v[404], v[405]);
  CHECK(hipGetLastError());
    break;
  }
  case 246: {
  softmax_forward<<<256, 256>>>(v[405], v[406]);
  CHECK(hipGetLastError());
    break;
  }
  case 247: {
  attention_values_forward<<<64, 256>>>(v[406], v[402], v[407]);
  CHECK(hipGetLastError());
    break;
  }
  case 248: {
  linear_forward_hidden<<<dim3(128, 1), dim3(16, 16)>>>(v[407], v[408], v[409]);
  CHECK(hipGetLastError());
    break;
  }
  case 249: {
  add_forward<<<64, 256>>>(v[393], v[409], v[410]);
  CHECK(hipGetLastError());
    break;
  }
  case 250: {
  rmsnorm_forward<<<8, 256>>>(v[410], v[411], v[412], v[413]);
  CHECK(hipGetLastError());
    break;
  }
  case 251: {
  linear_forward_mlp_up<<<dim3(512, 1), dim3(16, 16)>>>(v[412], v[414], v[415]);
  CHECK(hipGetLastError());
    break;
  }
  case 252: {
  linear_forward_mlp_up<<<dim3(512, 1), dim3(16, 16)>>>(v[412], v[416], v[417]);
  CHECK(hipGetLastError());
    break;
  }
  case 253: {
  swiglu_forward<<<256, 256>>>(v[415], v[417], v[418]);
  CHECK(hipGetLastError());
    break;
  }
  case 254: {
  linear_forward_mlp_down<<<dim3(128, 1), dim3(16, 16)>>>(v[418], v[419], v[420]);
  CHECK(hipGetLastError());
    break;
  }
  case 255: {
  add_forward<<<64, 256>>>(v[410], v[420], v[421]);
  CHECK(hipGetLastError());
    break;
  }
  case 256: {
  rmsnorm_forward<<<8, 256>>>(v[421], v[422], v[423], v[424]);
  CHECK(hipGetLastError());
    break;
  }
  case 257: {
  linear_forward_hidden<<<dim3(128, 1), dim3(16, 16)>>>(v[423], v[425], v[426]);
  CHECK(hipGetLastError());
    break;
  }
  case 258: {
  linear_forward_kv<<<dim3(32, 1), dim3(16, 16)>>>(v[423], v[427], v[428]);
  CHECK(hipGetLastError());
    break;
  }
  case 259: {
  linear_forward_kv<<<dim3(32, 1), dim3(16, 16)>>>(v[423], v[429], v[430]);
  CHECK(hipGetLastError());
    break;
  }
  case 260: {
  rope_forward_q<<<32, 256>>>(v[426], v[431]);
  CHECK(hipGetLastError());
    break;
  }
  case 261: {
  rope_forward_k<<<8, 256>>>(v[428], v[432]);
  CHECK(hipGetLastError());
    break;
  }
  case 262: {
  attention_scores_forward<<<8, 256>>>(v[431], v[432], v[433]);
  CHECK(hipGetLastError());
    break;
  }
  case 263: {
  softmax_forward<<<256, 256>>>(v[433], v[434]);
  CHECK(hipGetLastError());
    break;
  }
  case 264: {
  attention_values_forward<<<64, 256>>>(v[434], v[430], v[435]);
  CHECK(hipGetLastError());
    break;
  }
  case 265: {
  linear_forward_hidden<<<dim3(128, 1), dim3(16, 16)>>>(v[435], v[436], v[437]);
  CHECK(hipGetLastError());
    break;
  }
  case 266: {
  add_forward<<<64, 256>>>(v[421], v[437], v[438]);
  CHECK(hipGetLastError());
    break;
  }
  case 267: {
  rmsnorm_forward<<<8, 256>>>(v[438], v[439], v[440], v[441]);
  CHECK(hipGetLastError());
    break;
  }
  case 268: {
  linear_forward_mlp_up<<<dim3(512, 1), dim3(16, 16)>>>(v[440], v[442], v[443]);
  CHECK(hipGetLastError());
    break;
  }
  case 269: {
  linear_forward_mlp_up<<<dim3(512, 1), dim3(16, 16)>>>(v[440], v[444], v[445]);
  CHECK(hipGetLastError());
    break;
  }
  case 270: {
  swiglu_forward<<<256, 256>>>(v[443], v[445], v[446]);
  CHECK(hipGetLastError());
    break;
  }
  case 271: {
  linear_forward_mlp_down<<<dim3(128, 1), dim3(16, 16)>>>(v[446], v[447], v[448]);
  CHECK(hipGetLastError());
    break;
  }
  case 272: {
  add_forward<<<64, 256>>>(v[438], v[448], v[449]);
  CHECK(hipGetLastError());
    break;
  }
  case 273: {
  rmsnorm_forward<<<8, 256>>>(v[449], v[450], v[451], v[452]);
  CHECK(hipGetLastError());
    break;
  }
  case 274: {
  linear_forward_lm_head<<<dim3(8016, 1), dim3(16, 16)>>>(v[451], v[0], v[453]);
  CHECK(hipGetLastError());
    break;
  }
  case 275: {
  cross_entropy_forward<<<8, 256>>>(v[453], targets, v[454]);
  CHECK(hipGetLastError());
    break;
  }
  case 276: {
  mean_forward<<<1, 256>>>(v[454], v[455]);
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
  embedding_backward<<<64, 256>>>(g[1], ids, g[0]);
  CHECK(hipGetLastError());
    break;
  }
  case 1: {
  rmsnorm_backward_input<<<8, 256>>>(v[1], v[2], v[4], g[3], g[1]);
  CHECK(hipGetLastError());
  rmsnorm_backward_weight<<<8, 256>>>(v[1], v[4], g[3], g[2]);
  CHECK(hipGetLastError());
    break;
  }
  case 2: {
  linear_backward_input_hidden<<<dim3(128, 1), dim3(16, 16)>>>(g[6], v[5], g[3]);
  CHECK(hipGetLastError());
  linear_backward_weight_hidden<<<dim3(128, 128), dim3(16, 16)>>>(g[6], v[3], g[5]);
  CHECK(hipGetLastError());
    break;
  }
  case 3: {
  linear_backward_input_kv<<<dim3(128, 1), dim3(16, 16)>>>(g[8], v[7], g[3]);
  CHECK(hipGetLastError());
  linear_backward_weight_kv<<<dim3(128, 32), dim3(16, 16)>>>(g[8], v[3], g[7]);
  CHECK(hipGetLastError());
    break;
  }
  case 4: {
  linear_backward_input_kv<<<dim3(128, 1), dim3(16, 16)>>>(g[10], v[9], g[3]);
  CHECK(hipGetLastError());
  linear_backward_weight_kv<<<dim3(128, 32), dim3(16, 16)>>>(g[10], v[3], g[9]);
  CHECK(hipGetLastError());
    break;
  }
  case 5: {
  rope_backward_q<<<32, 256>>>(g[11], g[6]);
  CHECK(hipGetLastError());
    break;
  }
  case 6: {
  rope_backward_k<<<8, 256>>>(g[12], g[8]);
  CHECK(hipGetLastError());
    break;
  }
  case 7: {
  attention_scores_backward_q<<<64, 256>>>(g[13], v[12], g[11]);
  CHECK(hipGetLastError());
  attention_scores_backward_k<<<16, 256>>>(g[13], v[11], g[12]);
  CHECK(hipGetLastError());
    break;
  }
  case 8: {
  softmax_backward<<<256, 256>>>(v[14], g[14], g[13]);
  CHECK(hipGetLastError());
    break;
  }
  case 9: {
  attention_values_backward_p<<<8, 256>>>(g[15], v[10], g[14]);
  CHECK(hipGetLastError());
  attention_values_backward_v<<<16, 256>>>(g[15], v[14], g[10]);
  CHECK(hipGetLastError());
    break;
  }
  case 10: {
  linear_backward_input_hidden<<<dim3(128, 1), dim3(16, 16)>>>(g[17], v[16], g[15]);
  CHECK(hipGetLastError());
  linear_backward_weight_hidden<<<dim3(128, 128), dim3(16, 16)>>>(g[17], v[15], g[16]);
  CHECK(hipGetLastError());
    break;
  }
  case 11: {
  add_backward<<<64, 256>>>(g[18], g[1], g[17]);
  CHECK(hipGetLastError());
    break;
  }
  case 12: {
  rmsnorm_backward_input<<<8, 256>>>(v[18], v[19], v[21], g[20], g[18]);
  CHECK(hipGetLastError());
  rmsnorm_backward_weight<<<8, 256>>>(v[18], v[21], g[20], g[19]);
  CHECK(hipGetLastError());
    break;
  }
  case 13: {
  linear_backward_input_mlp_up<<<dim3(128, 1), dim3(16, 16)>>>(g[23], v[22], g[20]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_up<<<dim3(128, 512), dim3(16, 16)>>>(g[23], v[20], g[22]);
  CHECK(hipGetLastError());
    break;
  }
  case 14: {
  linear_backward_input_mlp_up<<<dim3(128, 1), dim3(16, 16)>>>(g[25], v[24], g[20]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_up<<<dim3(128, 512), dim3(16, 16)>>>(g[25], v[20], g[24]);
  CHECK(hipGetLastError());
    break;
  }
  case 15: {
  swiglu_backward<<<256, 256>>>(v[23], v[25], g[26], g[23], g[25]);
  CHECK(hipGetLastError());
    break;
  }
  case 16: {
  linear_backward_input_mlp_down<<<dim3(512, 1), dim3(16, 16)>>>(g[28], v[27], g[26]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_down<<<dim3(512, 128), dim3(16, 16)>>>(g[28], v[26], g[27]);
  CHECK(hipGetLastError());
    break;
  }
  case 17: {
  add_backward<<<64, 256>>>(g[29], g[18], g[28]);
  CHECK(hipGetLastError());
    break;
  }
  case 18: {
  rmsnorm_backward_input<<<8, 256>>>(v[29], v[30], v[32], g[31], g[29]);
  CHECK(hipGetLastError());
  rmsnorm_backward_weight<<<8, 256>>>(v[29], v[32], g[31], g[30]);
  CHECK(hipGetLastError());
    break;
  }
  case 19: {
  linear_backward_input_hidden<<<dim3(128, 1), dim3(16, 16)>>>(g[34], v[33], g[31]);
  CHECK(hipGetLastError());
  linear_backward_weight_hidden<<<dim3(128, 128), dim3(16, 16)>>>(g[34], v[31], g[33]);
  CHECK(hipGetLastError());
    break;
  }
  case 20: {
  linear_backward_input_kv<<<dim3(128, 1), dim3(16, 16)>>>(g[36], v[35], g[31]);
  CHECK(hipGetLastError());
  linear_backward_weight_kv<<<dim3(128, 32), dim3(16, 16)>>>(g[36], v[31], g[35]);
  CHECK(hipGetLastError());
    break;
  }
  case 21: {
  linear_backward_input_kv<<<dim3(128, 1), dim3(16, 16)>>>(g[38], v[37], g[31]);
  CHECK(hipGetLastError());
  linear_backward_weight_kv<<<dim3(128, 32), dim3(16, 16)>>>(g[38], v[31], g[37]);
  CHECK(hipGetLastError());
    break;
  }
  case 22: {
  rope_backward_q<<<32, 256>>>(g[39], g[34]);
  CHECK(hipGetLastError());
    break;
  }
  case 23: {
  rope_backward_k<<<8, 256>>>(g[40], g[36]);
  CHECK(hipGetLastError());
    break;
  }
  case 24: {
  attention_scores_backward_q<<<64, 256>>>(g[41], v[40], g[39]);
  CHECK(hipGetLastError());
  attention_scores_backward_k<<<16, 256>>>(g[41], v[39], g[40]);
  CHECK(hipGetLastError());
    break;
  }
  case 25: {
  softmax_backward<<<256, 256>>>(v[42], g[42], g[41]);
  CHECK(hipGetLastError());
    break;
  }
  case 26: {
  attention_values_backward_p<<<8, 256>>>(g[43], v[38], g[42]);
  CHECK(hipGetLastError());
  attention_values_backward_v<<<16, 256>>>(g[43], v[42], g[38]);
  CHECK(hipGetLastError());
    break;
  }
  case 27: {
  linear_backward_input_hidden<<<dim3(128, 1), dim3(16, 16)>>>(g[45], v[44], g[43]);
  CHECK(hipGetLastError());
  linear_backward_weight_hidden<<<dim3(128, 128), dim3(16, 16)>>>(g[45], v[43], g[44]);
  CHECK(hipGetLastError());
    break;
  }
  case 28: {
  add_backward<<<64, 256>>>(g[46], g[29], g[45]);
  CHECK(hipGetLastError());
    break;
  }
  case 29: {
  rmsnorm_backward_input<<<8, 256>>>(v[46], v[47], v[49], g[48], g[46]);
  CHECK(hipGetLastError());
  rmsnorm_backward_weight<<<8, 256>>>(v[46], v[49], g[48], g[47]);
  CHECK(hipGetLastError());
    break;
  }
  case 30: {
  linear_backward_input_mlp_up<<<dim3(128, 1), dim3(16, 16)>>>(g[51], v[50], g[48]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_up<<<dim3(128, 512), dim3(16, 16)>>>(g[51], v[48], g[50]);
  CHECK(hipGetLastError());
    break;
  }
  case 31: {
  linear_backward_input_mlp_up<<<dim3(128, 1), dim3(16, 16)>>>(g[53], v[52], g[48]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_up<<<dim3(128, 512), dim3(16, 16)>>>(g[53], v[48], g[52]);
  CHECK(hipGetLastError());
    break;
  }
  case 32: {
  swiglu_backward<<<256, 256>>>(v[51], v[53], g[54], g[51], g[53]);
  CHECK(hipGetLastError());
    break;
  }
  case 33: {
  linear_backward_input_mlp_down<<<dim3(512, 1), dim3(16, 16)>>>(g[56], v[55], g[54]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_down<<<dim3(512, 128), dim3(16, 16)>>>(g[56], v[54], g[55]);
  CHECK(hipGetLastError());
    break;
  }
  case 34: {
  add_backward<<<64, 256>>>(g[57], g[46], g[56]);
  CHECK(hipGetLastError());
    break;
  }
  case 35: {
  rmsnorm_backward_input<<<8, 256>>>(v[57], v[58], v[60], g[59], g[57]);
  CHECK(hipGetLastError());
  rmsnorm_backward_weight<<<8, 256>>>(v[57], v[60], g[59], g[58]);
  CHECK(hipGetLastError());
    break;
  }
  case 36: {
  linear_backward_input_hidden<<<dim3(128, 1), dim3(16, 16)>>>(g[62], v[61], g[59]);
  CHECK(hipGetLastError());
  linear_backward_weight_hidden<<<dim3(128, 128), dim3(16, 16)>>>(g[62], v[59], g[61]);
  CHECK(hipGetLastError());
    break;
  }
  case 37: {
  linear_backward_input_kv<<<dim3(128, 1), dim3(16, 16)>>>(g[64], v[63], g[59]);
  CHECK(hipGetLastError());
  linear_backward_weight_kv<<<dim3(128, 32), dim3(16, 16)>>>(g[64], v[59], g[63]);
  CHECK(hipGetLastError());
    break;
  }
  case 38: {
  linear_backward_input_kv<<<dim3(128, 1), dim3(16, 16)>>>(g[66], v[65], g[59]);
  CHECK(hipGetLastError());
  linear_backward_weight_kv<<<dim3(128, 32), dim3(16, 16)>>>(g[66], v[59], g[65]);
  CHECK(hipGetLastError());
    break;
  }
  case 39: {
  rope_backward_q<<<32, 256>>>(g[67], g[62]);
  CHECK(hipGetLastError());
    break;
  }
  case 40: {
  rope_backward_k<<<8, 256>>>(g[68], g[64]);
  CHECK(hipGetLastError());
    break;
  }
  case 41: {
  attention_scores_backward_q<<<64, 256>>>(g[69], v[68], g[67]);
  CHECK(hipGetLastError());
  attention_scores_backward_k<<<16, 256>>>(g[69], v[67], g[68]);
  CHECK(hipGetLastError());
    break;
  }
  case 42: {
  softmax_backward<<<256, 256>>>(v[70], g[70], g[69]);
  CHECK(hipGetLastError());
    break;
  }
  case 43: {
  attention_values_backward_p<<<8, 256>>>(g[71], v[66], g[70]);
  CHECK(hipGetLastError());
  attention_values_backward_v<<<16, 256>>>(g[71], v[70], g[66]);
  CHECK(hipGetLastError());
    break;
  }
  case 44: {
  linear_backward_input_hidden<<<dim3(128, 1), dim3(16, 16)>>>(g[73], v[72], g[71]);
  CHECK(hipGetLastError());
  linear_backward_weight_hidden<<<dim3(128, 128), dim3(16, 16)>>>(g[73], v[71], g[72]);
  CHECK(hipGetLastError());
    break;
  }
  case 45: {
  add_backward<<<64, 256>>>(g[74], g[57], g[73]);
  CHECK(hipGetLastError());
    break;
  }
  case 46: {
  rmsnorm_backward_input<<<8, 256>>>(v[74], v[75], v[77], g[76], g[74]);
  CHECK(hipGetLastError());
  rmsnorm_backward_weight<<<8, 256>>>(v[74], v[77], g[76], g[75]);
  CHECK(hipGetLastError());
    break;
  }
  case 47: {
  linear_backward_input_mlp_up<<<dim3(128, 1), dim3(16, 16)>>>(g[79], v[78], g[76]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_up<<<dim3(128, 512), dim3(16, 16)>>>(g[79], v[76], g[78]);
  CHECK(hipGetLastError());
    break;
  }
  case 48: {
  linear_backward_input_mlp_up<<<dim3(128, 1), dim3(16, 16)>>>(g[81], v[80], g[76]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_up<<<dim3(128, 512), dim3(16, 16)>>>(g[81], v[76], g[80]);
  CHECK(hipGetLastError());
    break;
  }
  case 49: {
  swiglu_backward<<<256, 256>>>(v[79], v[81], g[82], g[79], g[81]);
  CHECK(hipGetLastError());
    break;
  }
  case 50: {
  linear_backward_input_mlp_down<<<dim3(512, 1), dim3(16, 16)>>>(g[84], v[83], g[82]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_down<<<dim3(512, 128), dim3(16, 16)>>>(g[84], v[82], g[83]);
  CHECK(hipGetLastError());
    break;
  }
  case 51: {
  add_backward<<<64, 256>>>(g[85], g[74], g[84]);
  CHECK(hipGetLastError());
    break;
  }
  case 52: {
  rmsnorm_backward_input<<<8, 256>>>(v[85], v[86], v[88], g[87], g[85]);
  CHECK(hipGetLastError());
  rmsnorm_backward_weight<<<8, 256>>>(v[85], v[88], g[87], g[86]);
  CHECK(hipGetLastError());
    break;
  }
  case 53: {
  linear_backward_input_hidden<<<dim3(128, 1), dim3(16, 16)>>>(g[90], v[89], g[87]);
  CHECK(hipGetLastError());
  linear_backward_weight_hidden<<<dim3(128, 128), dim3(16, 16)>>>(g[90], v[87], g[89]);
  CHECK(hipGetLastError());
    break;
  }
  case 54: {
  linear_backward_input_kv<<<dim3(128, 1), dim3(16, 16)>>>(g[92], v[91], g[87]);
  CHECK(hipGetLastError());
  linear_backward_weight_kv<<<dim3(128, 32), dim3(16, 16)>>>(g[92], v[87], g[91]);
  CHECK(hipGetLastError());
    break;
  }
  case 55: {
  linear_backward_input_kv<<<dim3(128, 1), dim3(16, 16)>>>(g[94], v[93], g[87]);
  CHECK(hipGetLastError());
  linear_backward_weight_kv<<<dim3(128, 32), dim3(16, 16)>>>(g[94], v[87], g[93]);
  CHECK(hipGetLastError());
    break;
  }
  case 56: {
  rope_backward_q<<<32, 256>>>(g[95], g[90]);
  CHECK(hipGetLastError());
    break;
  }
  case 57: {
  rope_backward_k<<<8, 256>>>(g[96], g[92]);
  CHECK(hipGetLastError());
    break;
  }
  case 58: {
  attention_scores_backward_q<<<64, 256>>>(g[97], v[96], g[95]);
  CHECK(hipGetLastError());
  attention_scores_backward_k<<<16, 256>>>(g[97], v[95], g[96]);
  CHECK(hipGetLastError());
    break;
  }
  case 59: {
  softmax_backward<<<256, 256>>>(v[98], g[98], g[97]);
  CHECK(hipGetLastError());
    break;
  }
  case 60: {
  attention_values_backward_p<<<8, 256>>>(g[99], v[94], g[98]);
  CHECK(hipGetLastError());
  attention_values_backward_v<<<16, 256>>>(g[99], v[98], g[94]);
  CHECK(hipGetLastError());
    break;
  }
  case 61: {
  linear_backward_input_hidden<<<dim3(128, 1), dim3(16, 16)>>>(g[101], v[100], g[99]);
  CHECK(hipGetLastError());
  linear_backward_weight_hidden<<<dim3(128, 128), dim3(16, 16)>>>(g[101], v[99], g[100]);
  CHECK(hipGetLastError());
    break;
  }
  case 62: {
  add_backward<<<64, 256>>>(g[102], g[85], g[101]);
  CHECK(hipGetLastError());
    break;
  }
  case 63: {
  rmsnorm_backward_input<<<8, 256>>>(v[102], v[103], v[105], g[104], g[102]);
  CHECK(hipGetLastError());
  rmsnorm_backward_weight<<<8, 256>>>(v[102], v[105], g[104], g[103]);
  CHECK(hipGetLastError());
    break;
  }
  case 64: {
  linear_backward_input_mlp_up<<<dim3(128, 1), dim3(16, 16)>>>(g[107], v[106], g[104]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_up<<<dim3(128, 512), dim3(16, 16)>>>(g[107], v[104], g[106]);
  CHECK(hipGetLastError());
    break;
  }
  case 65: {
  linear_backward_input_mlp_up<<<dim3(128, 1), dim3(16, 16)>>>(g[109], v[108], g[104]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_up<<<dim3(128, 512), dim3(16, 16)>>>(g[109], v[104], g[108]);
  CHECK(hipGetLastError());
    break;
  }
  case 66: {
  swiglu_backward<<<256, 256>>>(v[107], v[109], g[110], g[107], g[109]);
  CHECK(hipGetLastError());
    break;
  }
  case 67: {
  linear_backward_input_mlp_down<<<dim3(512, 1), dim3(16, 16)>>>(g[112], v[111], g[110]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_down<<<dim3(512, 128), dim3(16, 16)>>>(g[112], v[110], g[111]);
  CHECK(hipGetLastError());
    break;
  }
  case 68: {
  add_backward<<<64, 256>>>(g[113], g[102], g[112]);
  CHECK(hipGetLastError());
    break;
  }
  case 69: {
  rmsnorm_backward_input<<<8, 256>>>(v[113], v[114], v[116], g[115], g[113]);
  CHECK(hipGetLastError());
  rmsnorm_backward_weight<<<8, 256>>>(v[113], v[116], g[115], g[114]);
  CHECK(hipGetLastError());
    break;
  }
  case 70: {
  linear_backward_input_hidden<<<dim3(128, 1), dim3(16, 16)>>>(g[118], v[117], g[115]);
  CHECK(hipGetLastError());
  linear_backward_weight_hidden<<<dim3(128, 128), dim3(16, 16)>>>(g[118], v[115], g[117]);
  CHECK(hipGetLastError());
    break;
  }
  case 71: {
  linear_backward_input_kv<<<dim3(128, 1), dim3(16, 16)>>>(g[120], v[119], g[115]);
  CHECK(hipGetLastError());
  linear_backward_weight_kv<<<dim3(128, 32), dim3(16, 16)>>>(g[120], v[115], g[119]);
  CHECK(hipGetLastError());
    break;
  }
  case 72: {
  linear_backward_input_kv<<<dim3(128, 1), dim3(16, 16)>>>(g[122], v[121], g[115]);
  CHECK(hipGetLastError());
  linear_backward_weight_kv<<<dim3(128, 32), dim3(16, 16)>>>(g[122], v[115], g[121]);
  CHECK(hipGetLastError());
    break;
  }
  case 73: {
  rope_backward_q<<<32, 256>>>(g[123], g[118]);
  CHECK(hipGetLastError());
    break;
  }
  case 74: {
  rope_backward_k<<<8, 256>>>(g[124], g[120]);
  CHECK(hipGetLastError());
    break;
  }
  case 75: {
  attention_scores_backward_q<<<64, 256>>>(g[125], v[124], g[123]);
  CHECK(hipGetLastError());
  attention_scores_backward_k<<<16, 256>>>(g[125], v[123], g[124]);
  CHECK(hipGetLastError());
    break;
  }
  case 76: {
  softmax_backward<<<256, 256>>>(v[126], g[126], g[125]);
  CHECK(hipGetLastError());
    break;
  }
  case 77: {
  attention_values_backward_p<<<8, 256>>>(g[127], v[122], g[126]);
  CHECK(hipGetLastError());
  attention_values_backward_v<<<16, 256>>>(g[127], v[126], g[122]);
  CHECK(hipGetLastError());
    break;
  }
  case 78: {
  linear_backward_input_hidden<<<dim3(128, 1), dim3(16, 16)>>>(g[129], v[128], g[127]);
  CHECK(hipGetLastError());
  linear_backward_weight_hidden<<<dim3(128, 128), dim3(16, 16)>>>(g[129], v[127], g[128]);
  CHECK(hipGetLastError());
    break;
  }
  case 79: {
  add_backward<<<64, 256>>>(g[130], g[113], g[129]);
  CHECK(hipGetLastError());
    break;
  }
  case 80: {
  rmsnorm_backward_input<<<8, 256>>>(v[130], v[131], v[133], g[132], g[130]);
  CHECK(hipGetLastError());
  rmsnorm_backward_weight<<<8, 256>>>(v[130], v[133], g[132], g[131]);
  CHECK(hipGetLastError());
    break;
  }
  case 81: {
  linear_backward_input_mlp_up<<<dim3(128, 1), dim3(16, 16)>>>(g[135], v[134], g[132]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_up<<<dim3(128, 512), dim3(16, 16)>>>(g[135], v[132], g[134]);
  CHECK(hipGetLastError());
    break;
  }
  case 82: {
  linear_backward_input_mlp_up<<<dim3(128, 1), dim3(16, 16)>>>(g[137], v[136], g[132]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_up<<<dim3(128, 512), dim3(16, 16)>>>(g[137], v[132], g[136]);
  CHECK(hipGetLastError());
    break;
  }
  case 83: {
  swiglu_backward<<<256, 256>>>(v[135], v[137], g[138], g[135], g[137]);
  CHECK(hipGetLastError());
    break;
  }
  case 84: {
  linear_backward_input_mlp_down<<<dim3(512, 1), dim3(16, 16)>>>(g[140], v[139], g[138]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_down<<<dim3(512, 128), dim3(16, 16)>>>(g[140], v[138], g[139]);
  CHECK(hipGetLastError());
    break;
  }
  case 85: {
  add_backward<<<64, 256>>>(g[141], g[130], g[140]);
  CHECK(hipGetLastError());
    break;
  }
  case 86: {
  rmsnorm_backward_input<<<8, 256>>>(v[141], v[142], v[144], g[143], g[141]);
  CHECK(hipGetLastError());
  rmsnorm_backward_weight<<<8, 256>>>(v[141], v[144], g[143], g[142]);
  CHECK(hipGetLastError());
    break;
  }
  case 87: {
  linear_backward_input_hidden<<<dim3(128, 1), dim3(16, 16)>>>(g[146], v[145], g[143]);
  CHECK(hipGetLastError());
  linear_backward_weight_hidden<<<dim3(128, 128), dim3(16, 16)>>>(g[146], v[143], g[145]);
  CHECK(hipGetLastError());
    break;
  }
  case 88: {
  linear_backward_input_kv<<<dim3(128, 1), dim3(16, 16)>>>(g[148], v[147], g[143]);
  CHECK(hipGetLastError());
  linear_backward_weight_kv<<<dim3(128, 32), dim3(16, 16)>>>(g[148], v[143], g[147]);
  CHECK(hipGetLastError());
    break;
  }
  case 89: {
  linear_backward_input_kv<<<dim3(128, 1), dim3(16, 16)>>>(g[150], v[149], g[143]);
  CHECK(hipGetLastError());
  linear_backward_weight_kv<<<dim3(128, 32), dim3(16, 16)>>>(g[150], v[143], g[149]);
  CHECK(hipGetLastError());
    break;
  }
  case 90: {
  rope_backward_q<<<32, 256>>>(g[151], g[146]);
  CHECK(hipGetLastError());
    break;
  }
  case 91: {
  rope_backward_k<<<8, 256>>>(g[152], g[148]);
  CHECK(hipGetLastError());
    break;
  }
  case 92: {
  attention_scores_backward_q<<<64, 256>>>(g[153], v[152], g[151]);
  CHECK(hipGetLastError());
  attention_scores_backward_k<<<16, 256>>>(g[153], v[151], g[152]);
  CHECK(hipGetLastError());
    break;
  }
  case 93: {
  softmax_backward<<<256, 256>>>(v[154], g[154], g[153]);
  CHECK(hipGetLastError());
    break;
  }
  case 94: {
  attention_values_backward_p<<<8, 256>>>(g[155], v[150], g[154]);
  CHECK(hipGetLastError());
  attention_values_backward_v<<<16, 256>>>(g[155], v[154], g[150]);
  CHECK(hipGetLastError());
    break;
  }
  case 95: {
  linear_backward_input_hidden<<<dim3(128, 1), dim3(16, 16)>>>(g[157], v[156], g[155]);
  CHECK(hipGetLastError());
  linear_backward_weight_hidden<<<dim3(128, 128), dim3(16, 16)>>>(g[157], v[155], g[156]);
  CHECK(hipGetLastError());
    break;
  }
  case 96: {
  add_backward<<<64, 256>>>(g[158], g[141], g[157]);
  CHECK(hipGetLastError());
    break;
  }
  case 97: {
  rmsnorm_backward_input<<<8, 256>>>(v[158], v[159], v[161], g[160], g[158]);
  CHECK(hipGetLastError());
  rmsnorm_backward_weight<<<8, 256>>>(v[158], v[161], g[160], g[159]);
  CHECK(hipGetLastError());
    break;
  }
  case 98: {
  linear_backward_input_mlp_up<<<dim3(128, 1), dim3(16, 16)>>>(g[163], v[162], g[160]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_up<<<dim3(128, 512), dim3(16, 16)>>>(g[163], v[160], g[162]);
  CHECK(hipGetLastError());
    break;
  }
  case 99: {
  linear_backward_input_mlp_up<<<dim3(128, 1), dim3(16, 16)>>>(g[165], v[164], g[160]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_up<<<dim3(128, 512), dim3(16, 16)>>>(g[165], v[160], g[164]);
  CHECK(hipGetLastError());
    break;
  }
  case 100: {
  swiglu_backward<<<256, 256>>>(v[163], v[165], g[166], g[163], g[165]);
  CHECK(hipGetLastError());
    break;
  }
  case 101: {
  linear_backward_input_mlp_down<<<dim3(512, 1), dim3(16, 16)>>>(g[168], v[167], g[166]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_down<<<dim3(512, 128), dim3(16, 16)>>>(g[168], v[166], g[167]);
  CHECK(hipGetLastError());
    break;
  }
  case 102: {
  add_backward<<<64, 256>>>(g[169], g[158], g[168]);
  CHECK(hipGetLastError());
    break;
  }
  case 103: {
  rmsnorm_backward_input<<<8, 256>>>(v[169], v[170], v[172], g[171], g[169]);
  CHECK(hipGetLastError());
  rmsnorm_backward_weight<<<8, 256>>>(v[169], v[172], g[171], g[170]);
  CHECK(hipGetLastError());
    break;
  }
  case 104: {
  linear_backward_input_hidden<<<dim3(128, 1), dim3(16, 16)>>>(g[174], v[173], g[171]);
  CHECK(hipGetLastError());
  linear_backward_weight_hidden<<<dim3(128, 128), dim3(16, 16)>>>(g[174], v[171], g[173]);
  CHECK(hipGetLastError());
    break;
  }
  case 105: {
  linear_backward_input_kv<<<dim3(128, 1), dim3(16, 16)>>>(g[176], v[175], g[171]);
  CHECK(hipGetLastError());
  linear_backward_weight_kv<<<dim3(128, 32), dim3(16, 16)>>>(g[176], v[171], g[175]);
  CHECK(hipGetLastError());
    break;
  }
  case 106: {
  linear_backward_input_kv<<<dim3(128, 1), dim3(16, 16)>>>(g[178], v[177], g[171]);
  CHECK(hipGetLastError());
  linear_backward_weight_kv<<<dim3(128, 32), dim3(16, 16)>>>(g[178], v[171], g[177]);
  CHECK(hipGetLastError());
    break;
  }
  case 107: {
  rope_backward_q<<<32, 256>>>(g[179], g[174]);
  CHECK(hipGetLastError());
    break;
  }
  case 108: {
  rope_backward_k<<<8, 256>>>(g[180], g[176]);
  CHECK(hipGetLastError());
    break;
  }
  case 109: {
  attention_scores_backward_q<<<64, 256>>>(g[181], v[180], g[179]);
  CHECK(hipGetLastError());
  attention_scores_backward_k<<<16, 256>>>(g[181], v[179], g[180]);
  CHECK(hipGetLastError());
    break;
  }
  case 110: {
  softmax_backward<<<256, 256>>>(v[182], g[182], g[181]);
  CHECK(hipGetLastError());
    break;
  }
  case 111: {
  attention_values_backward_p<<<8, 256>>>(g[183], v[178], g[182]);
  CHECK(hipGetLastError());
  attention_values_backward_v<<<16, 256>>>(g[183], v[182], g[178]);
  CHECK(hipGetLastError());
    break;
  }
  case 112: {
  linear_backward_input_hidden<<<dim3(128, 1), dim3(16, 16)>>>(g[185], v[184], g[183]);
  CHECK(hipGetLastError());
  linear_backward_weight_hidden<<<dim3(128, 128), dim3(16, 16)>>>(g[185], v[183], g[184]);
  CHECK(hipGetLastError());
    break;
  }
  case 113: {
  add_backward<<<64, 256>>>(g[186], g[169], g[185]);
  CHECK(hipGetLastError());
    break;
  }
  case 114: {
  rmsnorm_backward_input<<<8, 256>>>(v[186], v[187], v[189], g[188], g[186]);
  CHECK(hipGetLastError());
  rmsnorm_backward_weight<<<8, 256>>>(v[186], v[189], g[188], g[187]);
  CHECK(hipGetLastError());
    break;
  }
  case 115: {
  linear_backward_input_mlp_up<<<dim3(128, 1), dim3(16, 16)>>>(g[191], v[190], g[188]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_up<<<dim3(128, 512), dim3(16, 16)>>>(g[191], v[188], g[190]);
  CHECK(hipGetLastError());
    break;
  }
  case 116: {
  linear_backward_input_mlp_up<<<dim3(128, 1), dim3(16, 16)>>>(g[193], v[192], g[188]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_up<<<dim3(128, 512), dim3(16, 16)>>>(g[193], v[188], g[192]);
  CHECK(hipGetLastError());
    break;
  }
  case 117: {
  swiglu_backward<<<256, 256>>>(v[191], v[193], g[194], g[191], g[193]);
  CHECK(hipGetLastError());
    break;
  }
  case 118: {
  linear_backward_input_mlp_down<<<dim3(512, 1), dim3(16, 16)>>>(g[196], v[195], g[194]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_down<<<dim3(512, 128), dim3(16, 16)>>>(g[196], v[194], g[195]);
  CHECK(hipGetLastError());
    break;
  }
  case 119: {
  add_backward<<<64, 256>>>(g[197], g[186], g[196]);
  CHECK(hipGetLastError());
    break;
  }
  case 120: {
  rmsnorm_backward_input<<<8, 256>>>(v[197], v[198], v[200], g[199], g[197]);
  CHECK(hipGetLastError());
  rmsnorm_backward_weight<<<8, 256>>>(v[197], v[200], g[199], g[198]);
  CHECK(hipGetLastError());
    break;
  }
  case 121: {
  linear_backward_input_hidden<<<dim3(128, 1), dim3(16, 16)>>>(g[202], v[201], g[199]);
  CHECK(hipGetLastError());
  linear_backward_weight_hidden<<<dim3(128, 128), dim3(16, 16)>>>(g[202], v[199], g[201]);
  CHECK(hipGetLastError());
    break;
  }
  case 122: {
  linear_backward_input_kv<<<dim3(128, 1), dim3(16, 16)>>>(g[204], v[203], g[199]);
  CHECK(hipGetLastError());
  linear_backward_weight_kv<<<dim3(128, 32), dim3(16, 16)>>>(g[204], v[199], g[203]);
  CHECK(hipGetLastError());
    break;
  }
  case 123: {
  linear_backward_input_kv<<<dim3(128, 1), dim3(16, 16)>>>(g[206], v[205], g[199]);
  CHECK(hipGetLastError());
  linear_backward_weight_kv<<<dim3(128, 32), dim3(16, 16)>>>(g[206], v[199], g[205]);
  CHECK(hipGetLastError());
    break;
  }
  case 124: {
  rope_backward_q<<<32, 256>>>(g[207], g[202]);
  CHECK(hipGetLastError());
    break;
  }
  case 125: {
  rope_backward_k<<<8, 256>>>(g[208], g[204]);
  CHECK(hipGetLastError());
    break;
  }
  case 126: {
  attention_scores_backward_q<<<64, 256>>>(g[209], v[208], g[207]);
  CHECK(hipGetLastError());
  attention_scores_backward_k<<<16, 256>>>(g[209], v[207], g[208]);
  CHECK(hipGetLastError());
    break;
  }
  case 127: {
  softmax_backward<<<256, 256>>>(v[210], g[210], g[209]);
  CHECK(hipGetLastError());
    break;
  }
  case 128: {
  attention_values_backward_p<<<8, 256>>>(g[211], v[206], g[210]);
  CHECK(hipGetLastError());
  attention_values_backward_v<<<16, 256>>>(g[211], v[210], g[206]);
  CHECK(hipGetLastError());
    break;
  }
  case 129: {
  linear_backward_input_hidden<<<dim3(128, 1), dim3(16, 16)>>>(g[213], v[212], g[211]);
  CHECK(hipGetLastError());
  linear_backward_weight_hidden<<<dim3(128, 128), dim3(16, 16)>>>(g[213], v[211], g[212]);
  CHECK(hipGetLastError());
    break;
  }
  case 130: {
  add_backward<<<64, 256>>>(g[214], g[197], g[213]);
  CHECK(hipGetLastError());
    break;
  }
  case 131: {
  rmsnorm_backward_input<<<8, 256>>>(v[214], v[215], v[217], g[216], g[214]);
  CHECK(hipGetLastError());
  rmsnorm_backward_weight<<<8, 256>>>(v[214], v[217], g[216], g[215]);
  CHECK(hipGetLastError());
    break;
  }
  case 132: {
  linear_backward_input_mlp_up<<<dim3(128, 1), dim3(16, 16)>>>(g[219], v[218], g[216]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_up<<<dim3(128, 512), dim3(16, 16)>>>(g[219], v[216], g[218]);
  CHECK(hipGetLastError());
    break;
  }
  case 133: {
  linear_backward_input_mlp_up<<<dim3(128, 1), dim3(16, 16)>>>(g[221], v[220], g[216]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_up<<<dim3(128, 512), dim3(16, 16)>>>(g[221], v[216], g[220]);
  CHECK(hipGetLastError());
    break;
  }
  case 134: {
  swiglu_backward<<<256, 256>>>(v[219], v[221], g[222], g[219], g[221]);
  CHECK(hipGetLastError());
    break;
  }
  case 135: {
  linear_backward_input_mlp_down<<<dim3(512, 1), dim3(16, 16)>>>(g[224], v[223], g[222]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_down<<<dim3(512, 128), dim3(16, 16)>>>(g[224], v[222], g[223]);
  CHECK(hipGetLastError());
    break;
  }
  case 136: {
  add_backward<<<64, 256>>>(g[225], g[214], g[224]);
  CHECK(hipGetLastError());
    break;
  }
  case 137: {
  rmsnorm_backward_input<<<8, 256>>>(v[225], v[226], v[228], g[227], g[225]);
  CHECK(hipGetLastError());
  rmsnorm_backward_weight<<<8, 256>>>(v[225], v[228], g[227], g[226]);
  CHECK(hipGetLastError());
    break;
  }
  case 138: {
  linear_backward_input_hidden<<<dim3(128, 1), dim3(16, 16)>>>(g[230], v[229], g[227]);
  CHECK(hipGetLastError());
  linear_backward_weight_hidden<<<dim3(128, 128), dim3(16, 16)>>>(g[230], v[227], g[229]);
  CHECK(hipGetLastError());
    break;
  }
  case 139: {
  linear_backward_input_kv<<<dim3(128, 1), dim3(16, 16)>>>(g[232], v[231], g[227]);
  CHECK(hipGetLastError());
  linear_backward_weight_kv<<<dim3(128, 32), dim3(16, 16)>>>(g[232], v[227], g[231]);
  CHECK(hipGetLastError());
    break;
  }
  case 140: {
  linear_backward_input_kv<<<dim3(128, 1), dim3(16, 16)>>>(g[234], v[233], g[227]);
  CHECK(hipGetLastError());
  linear_backward_weight_kv<<<dim3(128, 32), dim3(16, 16)>>>(g[234], v[227], g[233]);
  CHECK(hipGetLastError());
    break;
  }
  case 141: {
  rope_backward_q<<<32, 256>>>(g[235], g[230]);
  CHECK(hipGetLastError());
    break;
  }
  case 142: {
  rope_backward_k<<<8, 256>>>(g[236], g[232]);
  CHECK(hipGetLastError());
    break;
  }
  case 143: {
  attention_scores_backward_q<<<64, 256>>>(g[237], v[236], g[235]);
  CHECK(hipGetLastError());
  attention_scores_backward_k<<<16, 256>>>(g[237], v[235], g[236]);
  CHECK(hipGetLastError());
    break;
  }
  case 144: {
  softmax_backward<<<256, 256>>>(v[238], g[238], g[237]);
  CHECK(hipGetLastError());
    break;
  }
  case 145: {
  attention_values_backward_p<<<8, 256>>>(g[239], v[234], g[238]);
  CHECK(hipGetLastError());
  attention_values_backward_v<<<16, 256>>>(g[239], v[238], g[234]);
  CHECK(hipGetLastError());
    break;
  }
  case 146: {
  linear_backward_input_hidden<<<dim3(128, 1), dim3(16, 16)>>>(g[241], v[240], g[239]);
  CHECK(hipGetLastError());
  linear_backward_weight_hidden<<<dim3(128, 128), dim3(16, 16)>>>(g[241], v[239], g[240]);
  CHECK(hipGetLastError());
    break;
  }
  case 147: {
  add_backward<<<64, 256>>>(g[242], g[225], g[241]);
  CHECK(hipGetLastError());
    break;
  }
  case 148: {
  rmsnorm_backward_input<<<8, 256>>>(v[242], v[243], v[245], g[244], g[242]);
  CHECK(hipGetLastError());
  rmsnorm_backward_weight<<<8, 256>>>(v[242], v[245], g[244], g[243]);
  CHECK(hipGetLastError());
    break;
  }
  case 149: {
  linear_backward_input_mlp_up<<<dim3(128, 1), dim3(16, 16)>>>(g[247], v[246], g[244]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_up<<<dim3(128, 512), dim3(16, 16)>>>(g[247], v[244], g[246]);
  CHECK(hipGetLastError());
    break;
  }
  case 150: {
  linear_backward_input_mlp_up<<<dim3(128, 1), dim3(16, 16)>>>(g[249], v[248], g[244]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_up<<<dim3(128, 512), dim3(16, 16)>>>(g[249], v[244], g[248]);
  CHECK(hipGetLastError());
    break;
  }
  case 151: {
  swiglu_backward<<<256, 256>>>(v[247], v[249], g[250], g[247], g[249]);
  CHECK(hipGetLastError());
    break;
  }
  case 152: {
  linear_backward_input_mlp_down<<<dim3(512, 1), dim3(16, 16)>>>(g[252], v[251], g[250]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_down<<<dim3(512, 128), dim3(16, 16)>>>(g[252], v[250], g[251]);
  CHECK(hipGetLastError());
    break;
  }
  case 153: {
  add_backward<<<64, 256>>>(g[253], g[242], g[252]);
  CHECK(hipGetLastError());
    break;
  }
  case 154: {
  rmsnorm_backward_input<<<8, 256>>>(v[253], v[254], v[256], g[255], g[253]);
  CHECK(hipGetLastError());
  rmsnorm_backward_weight<<<8, 256>>>(v[253], v[256], g[255], g[254]);
  CHECK(hipGetLastError());
    break;
  }
  case 155: {
  linear_backward_input_hidden<<<dim3(128, 1), dim3(16, 16)>>>(g[258], v[257], g[255]);
  CHECK(hipGetLastError());
  linear_backward_weight_hidden<<<dim3(128, 128), dim3(16, 16)>>>(g[258], v[255], g[257]);
  CHECK(hipGetLastError());
    break;
  }
  case 156: {
  linear_backward_input_kv<<<dim3(128, 1), dim3(16, 16)>>>(g[260], v[259], g[255]);
  CHECK(hipGetLastError());
  linear_backward_weight_kv<<<dim3(128, 32), dim3(16, 16)>>>(g[260], v[255], g[259]);
  CHECK(hipGetLastError());
    break;
  }
  case 157: {
  linear_backward_input_kv<<<dim3(128, 1), dim3(16, 16)>>>(g[262], v[261], g[255]);
  CHECK(hipGetLastError());
  linear_backward_weight_kv<<<dim3(128, 32), dim3(16, 16)>>>(g[262], v[255], g[261]);
  CHECK(hipGetLastError());
    break;
  }
  case 158: {
  rope_backward_q<<<32, 256>>>(g[263], g[258]);
  CHECK(hipGetLastError());
    break;
  }
  case 159: {
  rope_backward_k<<<8, 256>>>(g[264], g[260]);
  CHECK(hipGetLastError());
    break;
  }
  case 160: {
  attention_scores_backward_q<<<64, 256>>>(g[265], v[264], g[263]);
  CHECK(hipGetLastError());
  attention_scores_backward_k<<<16, 256>>>(g[265], v[263], g[264]);
  CHECK(hipGetLastError());
    break;
  }
  case 161: {
  softmax_backward<<<256, 256>>>(v[266], g[266], g[265]);
  CHECK(hipGetLastError());
    break;
  }
  case 162: {
  attention_values_backward_p<<<8, 256>>>(g[267], v[262], g[266]);
  CHECK(hipGetLastError());
  attention_values_backward_v<<<16, 256>>>(g[267], v[266], g[262]);
  CHECK(hipGetLastError());
    break;
  }
  case 163: {
  linear_backward_input_hidden<<<dim3(128, 1), dim3(16, 16)>>>(g[269], v[268], g[267]);
  CHECK(hipGetLastError());
  linear_backward_weight_hidden<<<dim3(128, 128), dim3(16, 16)>>>(g[269], v[267], g[268]);
  CHECK(hipGetLastError());
    break;
  }
  case 164: {
  add_backward<<<64, 256>>>(g[270], g[253], g[269]);
  CHECK(hipGetLastError());
    break;
  }
  case 165: {
  rmsnorm_backward_input<<<8, 256>>>(v[270], v[271], v[273], g[272], g[270]);
  CHECK(hipGetLastError());
  rmsnorm_backward_weight<<<8, 256>>>(v[270], v[273], g[272], g[271]);
  CHECK(hipGetLastError());
    break;
  }
  case 166: {
  linear_backward_input_mlp_up<<<dim3(128, 1), dim3(16, 16)>>>(g[275], v[274], g[272]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_up<<<dim3(128, 512), dim3(16, 16)>>>(g[275], v[272], g[274]);
  CHECK(hipGetLastError());
    break;
  }
  case 167: {
  linear_backward_input_mlp_up<<<dim3(128, 1), dim3(16, 16)>>>(g[277], v[276], g[272]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_up<<<dim3(128, 512), dim3(16, 16)>>>(g[277], v[272], g[276]);
  CHECK(hipGetLastError());
    break;
  }
  case 168: {
  swiglu_backward<<<256, 256>>>(v[275], v[277], g[278], g[275], g[277]);
  CHECK(hipGetLastError());
    break;
  }
  case 169: {
  linear_backward_input_mlp_down<<<dim3(512, 1), dim3(16, 16)>>>(g[280], v[279], g[278]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_down<<<dim3(512, 128), dim3(16, 16)>>>(g[280], v[278], g[279]);
  CHECK(hipGetLastError());
    break;
  }
  case 170: {
  add_backward<<<64, 256>>>(g[281], g[270], g[280]);
  CHECK(hipGetLastError());
    break;
  }
  case 171: {
  rmsnorm_backward_input<<<8, 256>>>(v[281], v[282], v[284], g[283], g[281]);
  CHECK(hipGetLastError());
  rmsnorm_backward_weight<<<8, 256>>>(v[281], v[284], g[283], g[282]);
  CHECK(hipGetLastError());
    break;
  }
  case 172: {
  linear_backward_input_hidden<<<dim3(128, 1), dim3(16, 16)>>>(g[286], v[285], g[283]);
  CHECK(hipGetLastError());
  linear_backward_weight_hidden<<<dim3(128, 128), dim3(16, 16)>>>(g[286], v[283], g[285]);
  CHECK(hipGetLastError());
    break;
  }
  case 173: {
  linear_backward_input_kv<<<dim3(128, 1), dim3(16, 16)>>>(g[288], v[287], g[283]);
  CHECK(hipGetLastError());
  linear_backward_weight_kv<<<dim3(128, 32), dim3(16, 16)>>>(g[288], v[283], g[287]);
  CHECK(hipGetLastError());
    break;
  }
  case 174: {
  linear_backward_input_kv<<<dim3(128, 1), dim3(16, 16)>>>(g[290], v[289], g[283]);
  CHECK(hipGetLastError());
  linear_backward_weight_kv<<<dim3(128, 32), dim3(16, 16)>>>(g[290], v[283], g[289]);
  CHECK(hipGetLastError());
    break;
  }
  case 175: {
  rope_backward_q<<<32, 256>>>(g[291], g[286]);
  CHECK(hipGetLastError());
    break;
  }
  case 176: {
  rope_backward_k<<<8, 256>>>(g[292], g[288]);
  CHECK(hipGetLastError());
    break;
  }
  case 177: {
  attention_scores_backward_q<<<64, 256>>>(g[293], v[292], g[291]);
  CHECK(hipGetLastError());
  attention_scores_backward_k<<<16, 256>>>(g[293], v[291], g[292]);
  CHECK(hipGetLastError());
    break;
  }
  case 178: {
  softmax_backward<<<256, 256>>>(v[294], g[294], g[293]);
  CHECK(hipGetLastError());
    break;
  }
  case 179: {
  attention_values_backward_p<<<8, 256>>>(g[295], v[290], g[294]);
  CHECK(hipGetLastError());
  attention_values_backward_v<<<16, 256>>>(g[295], v[294], g[290]);
  CHECK(hipGetLastError());
    break;
  }
  case 180: {
  linear_backward_input_hidden<<<dim3(128, 1), dim3(16, 16)>>>(g[297], v[296], g[295]);
  CHECK(hipGetLastError());
  linear_backward_weight_hidden<<<dim3(128, 128), dim3(16, 16)>>>(g[297], v[295], g[296]);
  CHECK(hipGetLastError());
    break;
  }
  case 181: {
  add_backward<<<64, 256>>>(g[298], g[281], g[297]);
  CHECK(hipGetLastError());
    break;
  }
  case 182: {
  rmsnorm_backward_input<<<8, 256>>>(v[298], v[299], v[301], g[300], g[298]);
  CHECK(hipGetLastError());
  rmsnorm_backward_weight<<<8, 256>>>(v[298], v[301], g[300], g[299]);
  CHECK(hipGetLastError());
    break;
  }
  case 183: {
  linear_backward_input_mlp_up<<<dim3(128, 1), dim3(16, 16)>>>(g[303], v[302], g[300]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_up<<<dim3(128, 512), dim3(16, 16)>>>(g[303], v[300], g[302]);
  CHECK(hipGetLastError());
    break;
  }
  case 184: {
  linear_backward_input_mlp_up<<<dim3(128, 1), dim3(16, 16)>>>(g[305], v[304], g[300]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_up<<<dim3(128, 512), dim3(16, 16)>>>(g[305], v[300], g[304]);
  CHECK(hipGetLastError());
    break;
  }
  case 185: {
  swiglu_backward<<<256, 256>>>(v[303], v[305], g[306], g[303], g[305]);
  CHECK(hipGetLastError());
    break;
  }
  case 186: {
  linear_backward_input_mlp_down<<<dim3(512, 1), dim3(16, 16)>>>(g[308], v[307], g[306]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_down<<<dim3(512, 128), dim3(16, 16)>>>(g[308], v[306], g[307]);
  CHECK(hipGetLastError());
    break;
  }
  case 187: {
  add_backward<<<64, 256>>>(g[309], g[298], g[308]);
  CHECK(hipGetLastError());
    break;
  }
  case 188: {
  rmsnorm_backward_input<<<8, 256>>>(v[309], v[310], v[312], g[311], g[309]);
  CHECK(hipGetLastError());
  rmsnorm_backward_weight<<<8, 256>>>(v[309], v[312], g[311], g[310]);
  CHECK(hipGetLastError());
    break;
  }
  case 189: {
  linear_backward_input_hidden<<<dim3(128, 1), dim3(16, 16)>>>(g[314], v[313], g[311]);
  CHECK(hipGetLastError());
  linear_backward_weight_hidden<<<dim3(128, 128), dim3(16, 16)>>>(g[314], v[311], g[313]);
  CHECK(hipGetLastError());
    break;
  }
  case 190: {
  linear_backward_input_kv<<<dim3(128, 1), dim3(16, 16)>>>(g[316], v[315], g[311]);
  CHECK(hipGetLastError());
  linear_backward_weight_kv<<<dim3(128, 32), dim3(16, 16)>>>(g[316], v[311], g[315]);
  CHECK(hipGetLastError());
    break;
  }
  case 191: {
  linear_backward_input_kv<<<dim3(128, 1), dim3(16, 16)>>>(g[318], v[317], g[311]);
  CHECK(hipGetLastError());
  linear_backward_weight_kv<<<dim3(128, 32), dim3(16, 16)>>>(g[318], v[311], g[317]);
  CHECK(hipGetLastError());
    break;
  }
  case 192: {
  rope_backward_q<<<32, 256>>>(g[319], g[314]);
  CHECK(hipGetLastError());
    break;
  }
  case 193: {
  rope_backward_k<<<8, 256>>>(g[320], g[316]);
  CHECK(hipGetLastError());
    break;
  }
  case 194: {
  attention_scores_backward_q<<<64, 256>>>(g[321], v[320], g[319]);
  CHECK(hipGetLastError());
  attention_scores_backward_k<<<16, 256>>>(g[321], v[319], g[320]);
  CHECK(hipGetLastError());
    break;
  }
  case 195: {
  softmax_backward<<<256, 256>>>(v[322], g[322], g[321]);
  CHECK(hipGetLastError());
    break;
  }
  case 196: {
  attention_values_backward_p<<<8, 256>>>(g[323], v[318], g[322]);
  CHECK(hipGetLastError());
  attention_values_backward_v<<<16, 256>>>(g[323], v[322], g[318]);
  CHECK(hipGetLastError());
    break;
  }
  case 197: {
  linear_backward_input_hidden<<<dim3(128, 1), dim3(16, 16)>>>(g[325], v[324], g[323]);
  CHECK(hipGetLastError());
  linear_backward_weight_hidden<<<dim3(128, 128), dim3(16, 16)>>>(g[325], v[323], g[324]);
  CHECK(hipGetLastError());
    break;
  }
  case 198: {
  add_backward<<<64, 256>>>(g[326], g[309], g[325]);
  CHECK(hipGetLastError());
    break;
  }
  case 199: {
  rmsnorm_backward_input<<<8, 256>>>(v[326], v[327], v[329], g[328], g[326]);
  CHECK(hipGetLastError());
  rmsnorm_backward_weight<<<8, 256>>>(v[326], v[329], g[328], g[327]);
  CHECK(hipGetLastError());
    break;
  }
  case 200: {
  linear_backward_input_mlp_up<<<dim3(128, 1), dim3(16, 16)>>>(g[331], v[330], g[328]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_up<<<dim3(128, 512), dim3(16, 16)>>>(g[331], v[328], g[330]);
  CHECK(hipGetLastError());
    break;
  }
  case 201: {
  linear_backward_input_mlp_up<<<dim3(128, 1), dim3(16, 16)>>>(g[333], v[332], g[328]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_up<<<dim3(128, 512), dim3(16, 16)>>>(g[333], v[328], g[332]);
  CHECK(hipGetLastError());
    break;
  }
  case 202: {
  swiglu_backward<<<256, 256>>>(v[331], v[333], g[334], g[331], g[333]);
  CHECK(hipGetLastError());
    break;
  }
  case 203: {
  linear_backward_input_mlp_down<<<dim3(512, 1), dim3(16, 16)>>>(g[336], v[335], g[334]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_down<<<dim3(512, 128), dim3(16, 16)>>>(g[336], v[334], g[335]);
  CHECK(hipGetLastError());
    break;
  }
  case 204: {
  add_backward<<<64, 256>>>(g[337], g[326], g[336]);
  CHECK(hipGetLastError());
    break;
  }
  case 205: {
  rmsnorm_backward_input<<<8, 256>>>(v[337], v[338], v[340], g[339], g[337]);
  CHECK(hipGetLastError());
  rmsnorm_backward_weight<<<8, 256>>>(v[337], v[340], g[339], g[338]);
  CHECK(hipGetLastError());
    break;
  }
  case 206: {
  linear_backward_input_hidden<<<dim3(128, 1), dim3(16, 16)>>>(g[342], v[341], g[339]);
  CHECK(hipGetLastError());
  linear_backward_weight_hidden<<<dim3(128, 128), dim3(16, 16)>>>(g[342], v[339], g[341]);
  CHECK(hipGetLastError());
    break;
  }
  case 207: {
  linear_backward_input_kv<<<dim3(128, 1), dim3(16, 16)>>>(g[344], v[343], g[339]);
  CHECK(hipGetLastError());
  linear_backward_weight_kv<<<dim3(128, 32), dim3(16, 16)>>>(g[344], v[339], g[343]);
  CHECK(hipGetLastError());
    break;
  }
  case 208: {
  linear_backward_input_kv<<<dim3(128, 1), dim3(16, 16)>>>(g[346], v[345], g[339]);
  CHECK(hipGetLastError());
  linear_backward_weight_kv<<<dim3(128, 32), dim3(16, 16)>>>(g[346], v[339], g[345]);
  CHECK(hipGetLastError());
    break;
  }
  case 209: {
  rope_backward_q<<<32, 256>>>(g[347], g[342]);
  CHECK(hipGetLastError());
    break;
  }
  case 210: {
  rope_backward_k<<<8, 256>>>(g[348], g[344]);
  CHECK(hipGetLastError());
    break;
  }
  case 211: {
  attention_scores_backward_q<<<64, 256>>>(g[349], v[348], g[347]);
  CHECK(hipGetLastError());
  attention_scores_backward_k<<<16, 256>>>(g[349], v[347], g[348]);
  CHECK(hipGetLastError());
    break;
  }
  case 212: {
  softmax_backward<<<256, 256>>>(v[350], g[350], g[349]);
  CHECK(hipGetLastError());
    break;
  }
  case 213: {
  attention_values_backward_p<<<8, 256>>>(g[351], v[346], g[350]);
  CHECK(hipGetLastError());
  attention_values_backward_v<<<16, 256>>>(g[351], v[350], g[346]);
  CHECK(hipGetLastError());
    break;
  }
  case 214: {
  linear_backward_input_hidden<<<dim3(128, 1), dim3(16, 16)>>>(g[353], v[352], g[351]);
  CHECK(hipGetLastError());
  linear_backward_weight_hidden<<<dim3(128, 128), dim3(16, 16)>>>(g[353], v[351], g[352]);
  CHECK(hipGetLastError());
    break;
  }
  case 215: {
  add_backward<<<64, 256>>>(g[354], g[337], g[353]);
  CHECK(hipGetLastError());
    break;
  }
  case 216: {
  rmsnorm_backward_input<<<8, 256>>>(v[354], v[355], v[357], g[356], g[354]);
  CHECK(hipGetLastError());
  rmsnorm_backward_weight<<<8, 256>>>(v[354], v[357], g[356], g[355]);
  CHECK(hipGetLastError());
    break;
  }
  case 217: {
  linear_backward_input_mlp_up<<<dim3(128, 1), dim3(16, 16)>>>(g[359], v[358], g[356]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_up<<<dim3(128, 512), dim3(16, 16)>>>(g[359], v[356], g[358]);
  CHECK(hipGetLastError());
    break;
  }
  case 218: {
  linear_backward_input_mlp_up<<<dim3(128, 1), dim3(16, 16)>>>(g[361], v[360], g[356]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_up<<<dim3(128, 512), dim3(16, 16)>>>(g[361], v[356], g[360]);
  CHECK(hipGetLastError());
    break;
  }
  case 219: {
  swiglu_backward<<<256, 256>>>(v[359], v[361], g[362], g[359], g[361]);
  CHECK(hipGetLastError());
    break;
  }
  case 220: {
  linear_backward_input_mlp_down<<<dim3(512, 1), dim3(16, 16)>>>(g[364], v[363], g[362]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_down<<<dim3(512, 128), dim3(16, 16)>>>(g[364], v[362], g[363]);
  CHECK(hipGetLastError());
    break;
  }
  case 221: {
  add_backward<<<64, 256>>>(g[365], g[354], g[364]);
  CHECK(hipGetLastError());
    break;
  }
  case 222: {
  rmsnorm_backward_input<<<8, 256>>>(v[365], v[366], v[368], g[367], g[365]);
  CHECK(hipGetLastError());
  rmsnorm_backward_weight<<<8, 256>>>(v[365], v[368], g[367], g[366]);
  CHECK(hipGetLastError());
    break;
  }
  case 223: {
  linear_backward_input_hidden<<<dim3(128, 1), dim3(16, 16)>>>(g[370], v[369], g[367]);
  CHECK(hipGetLastError());
  linear_backward_weight_hidden<<<dim3(128, 128), dim3(16, 16)>>>(g[370], v[367], g[369]);
  CHECK(hipGetLastError());
    break;
  }
  case 224: {
  linear_backward_input_kv<<<dim3(128, 1), dim3(16, 16)>>>(g[372], v[371], g[367]);
  CHECK(hipGetLastError());
  linear_backward_weight_kv<<<dim3(128, 32), dim3(16, 16)>>>(g[372], v[367], g[371]);
  CHECK(hipGetLastError());
    break;
  }
  case 225: {
  linear_backward_input_kv<<<dim3(128, 1), dim3(16, 16)>>>(g[374], v[373], g[367]);
  CHECK(hipGetLastError());
  linear_backward_weight_kv<<<dim3(128, 32), dim3(16, 16)>>>(g[374], v[367], g[373]);
  CHECK(hipGetLastError());
    break;
  }
  case 226: {
  rope_backward_q<<<32, 256>>>(g[375], g[370]);
  CHECK(hipGetLastError());
    break;
  }
  case 227: {
  rope_backward_k<<<8, 256>>>(g[376], g[372]);
  CHECK(hipGetLastError());
    break;
  }
  case 228: {
  attention_scores_backward_q<<<64, 256>>>(g[377], v[376], g[375]);
  CHECK(hipGetLastError());
  attention_scores_backward_k<<<16, 256>>>(g[377], v[375], g[376]);
  CHECK(hipGetLastError());
    break;
  }
  case 229: {
  softmax_backward<<<256, 256>>>(v[378], g[378], g[377]);
  CHECK(hipGetLastError());
    break;
  }
  case 230: {
  attention_values_backward_p<<<8, 256>>>(g[379], v[374], g[378]);
  CHECK(hipGetLastError());
  attention_values_backward_v<<<16, 256>>>(g[379], v[378], g[374]);
  CHECK(hipGetLastError());
    break;
  }
  case 231: {
  linear_backward_input_hidden<<<dim3(128, 1), dim3(16, 16)>>>(g[381], v[380], g[379]);
  CHECK(hipGetLastError());
  linear_backward_weight_hidden<<<dim3(128, 128), dim3(16, 16)>>>(g[381], v[379], g[380]);
  CHECK(hipGetLastError());
    break;
  }
  case 232: {
  add_backward<<<64, 256>>>(g[382], g[365], g[381]);
  CHECK(hipGetLastError());
    break;
  }
  case 233: {
  rmsnorm_backward_input<<<8, 256>>>(v[382], v[383], v[385], g[384], g[382]);
  CHECK(hipGetLastError());
  rmsnorm_backward_weight<<<8, 256>>>(v[382], v[385], g[384], g[383]);
  CHECK(hipGetLastError());
    break;
  }
  case 234: {
  linear_backward_input_mlp_up<<<dim3(128, 1), dim3(16, 16)>>>(g[387], v[386], g[384]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_up<<<dim3(128, 512), dim3(16, 16)>>>(g[387], v[384], g[386]);
  CHECK(hipGetLastError());
    break;
  }
  case 235: {
  linear_backward_input_mlp_up<<<dim3(128, 1), dim3(16, 16)>>>(g[389], v[388], g[384]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_up<<<dim3(128, 512), dim3(16, 16)>>>(g[389], v[384], g[388]);
  CHECK(hipGetLastError());
    break;
  }
  case 236: {
  swiglu_backward<<<256, 256>>>(v[387], v[389], g[390], g[387], g[389]);
  CHECK(hipGetLastError());
    break;
  }
  case 237: {
  linear_backward_input_mlp_down<<<dim3(512, 1), dim3(16, 16)>>>(g[392], v[391], g[390]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_down<<<dim3(512, 128), dim3(16, 16)>>>(g[392], v[390], g[391]);
  CHECK(hipGetLastError());
    break;
  }
  case 238: {
  add_backward<<<64, 256>>>(g[393], g[382], g[392]);
  CHECK(hipGetLastError());
    break;
  }
  case 239: {
  rmsnorm_backward_input<<<8, 256>>>(v[393], v[394], v[396], g[395], g[393]);
  CHECK(hipGetLastError());
  rmsnorm_backward_weight<<<8, 256>>>(v[393], v[396], g[395], g[394]);
  CHECK(hipGetLastError());
    break;
  }
  case 240: {
  linear_backward_input_hidden<<<dim3(128, 1), dim3(16, 16)>>>(g[398], v[397], g[395]);
  CHECK(hipGetLastError());
  linear_backward_weight_hidden<<<dim3(128, 128), dim3(16, 16)>>>(g[398], v[395], g[397]);
  CHECK(hipGetLastError());
    break;
  }
  case 241: {
  linear_backward_input_kv<<<dim3(128, 1), dim3(16, 16)>>>(g[400], v[399], g[395]);
  CHECK(hipGetLastError());
  linear_backward_weight_kv<<<dim3(128, 32), dim3(16, 16)>>>(g[400], v[395], g[399]);
  CHECK(hipGetLastError());
    break;
  }
  case 242: {
  linear_backward_input_kv<<<dim3(128, 1), dim3(16, 16)>>>(g[402], v[401], g[395]);
  CHECK(hipGetLastError());
  linear_backward_weight_kv<<<dim3(128, 32), dim3(16, 16)>>>(g[402], v[395], g[401]);
  CHECK(hipGetLastError());
    break;
  }
  case 243: {
  rope_backward_q<<<32, 256>>>(g[403], g[398]);
  CHECK(hipGetLastError());
    break;
  }
  case 244: {
  rope_backward_k<<<8, 256>>>(g[404], g[400]);
  CHECK(hipGetLastError());
    break;
  }
  case 245: {
  attention_scores_backward_q<<<64, 256>>>(g[405], v[404], g[403]);
  CHECK(hipGetLastError());
  attention_scores_backward_k<<<16, 256>>>(g[405], v[403], g[404]);
  CHECK(hipGetLastError());
    break;
  }
  case 246: {
  softmax_backward<<<256, 256>>>(v[406], g[406], g[405]);
  CHECK(hipGetLastError());
    break;
  }
  case 247: {
  attention_values_backward_p<<<8, 256>>>(g[407], v[402], g[406]);
  CHECK(hipGetLastError());
  attention_values_backward_v<<<16, 256>>>(g[407], v[406], g[402]);
  CHECK(hipGetLastError());
    break;
  }
  case 248: {
  linear_backward_input_hidden<<<dim3(128, 1), dim3(16, 16)>>>(g[409], v[408], g[407]);
  CHECK(hipGetLastError());
  linear_backward_weight_hidden<<<dim3(128, 128), dim3(16, 16)>>>(g[409], v[407], g[408]);
  CHECK(hipGetLastError());
    break;
  }
  case 249: {
  add_backward<<<64, 256>>>(g[410], g[393], g[409]);
  CHECK(hipGetLastError());
    break;
  }
  case 250: {
  rmsnorm_backward_input<<<8, 256>>>(v[410], v[411], v[413], g[412], g[410]);
  CHECK(hipGetLastError());
  rmsnorm_backward_weight<<<8, 256>>>(v[410], v[413], g[412], g[411]);
  CHECK(hipGetLastError());
    break;
  }
  case 251: {
  linear_backward_input_mlp_up<<<dim3(128, 1), dim3(16, 16)>>>(g[415], v[414], g[412]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_up<<<dim3(128, 512), dim3(16, 16)>>>(g[415], v[412], g[414]);
  CHECK(hipGetLastError());
    break;
  }
  case 252: {
  linear_backward_input_mlp_up<<<dim3(128, 1), dim3(16, 16)>>>(g[417], v[416], g[412]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_up<<<dim3(128, 512), dim3(16, 16)>>>(g[417], v[412], g[416]);
  CHECK(hipGetLastError());
    break;
  }
  case 253: {
  swiglu_backward<<<256, 256>>>(v[415], v[417], g[418], g[415], g[417]);
  CHECK(hipGetLastError());
    break;
  }
  case 254: {
  linear_backward_input_mlp_down<<<dim3(512, 1), dim3(16, 16)>>>(g[420], v[419], g[418]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_down<<<dim3(512, 128), dim3(16, 16)>>>(g[420], v[418], g[419]);
  CHECK(hipGetLastError());
    break;
  }
  case 255: {
  add_backward<<<64, 256>>>(g[421], g[410], g[420]);
  CHECK(hipGetLastError());
    break;
  }
  case 256: {
  rmsnorm_backward_input<<<8, 256>>>(v[421], v[422], v[424], g[423], g[421]);
  CHECK(hipGetLastError());
  rmsnorm_backward_weight<<<8, 256>>>(v[421], v[424], g[423], g[422]);
  CHECK(hipGetLastError());
    break;
  }
  case 257: {
  linear_backward_input_hidden<<<dim3(128, 1), dim3(16, 16)>>>(g[426], v[425], g[423]);
  CHECK(hipGetLastError());
  linear_backward_weight_hidden<<<dim3(128, 128), dim3(16, 16)>>>(g[426], v[423], g[425]);
  CHECK(hipGetLastError());
    break;
  }
  case 258: {
  linear_backward_input_kv<<<dim3(128, 1), dim3(16, 16)>>>(g[428], v[427], g[423]);
  CHECK(hipGetLastError());
  linear_backward_weight_kv<<<dim3(128, 32), dim3(16, 16)>>>(g[428], v[423], g[427]);
  CHECK(hipGetLastError());
    break;
  }
  case 259: {
  linear_backward_input_kv<<<dim3(128, 1), dim3(16, 16)>>>(g[430], v[429], g[423]);
  CHECK(hipGetLastError());
  linear_backward_weight_kv<<<dim3(128, 32), dim3(16, 16)>>>(g[430], v[423], g[429]);
  CHECK(hipGetLastError());
    break;
  }
  case 260: {
  rope_backward_q<<<32, 256>>>(g[431], g[426]);
  CHECK(hipGetLastError());
    break;
  }
  case 261: {
  rope_backward_k<<<8, 256>>>(g[432], g[428]);
  CHECK(hipGetLastError());
    break;
  }
  case 262: {
  attention_scores_backward_q<<<64, 256>>>(g[433], v[432], g[431]);
  CHECK(hipGetLastError());
  attention_scores_backward_k<<<16, 256>>>(g[433], v[431], g[432]);
  CHECK(hipGetLastError());
    break;
  }
  case 263: {
  softmax_backward<<<256, 256>>>(v[434], g[434], g[433]);
  CHECK(hipGetLastError());
    break;
  }
  case 264: {
  attention_values_backward_p<<<8, 256>>>(g[435], v[430], g[434]);
  CHECK(hipGetLastError());
  attention_values_backward_v<<<16, 256>>>(g[435], v[434], g[430]);
  CHECK(hipGetLastError());
    break;
  }
  case 265: {
  linear_backward_input_hidden<<<dim3(128, 1), dim3(16, 16)>>>(g[437], v[436], g[435]);
  CHECK(hipGetLastError());
  linear_backward_weight_hidden<<<dim3(128, 128), dim3(16, 16)>>>(g[437], v[435], g[436]);
  CHECK(hipGetLastError());
    break;
  }
  case 266: {
  add_backward<<<64, 256>>>(g[438], g[421], g[437]);
  CHECK(hipGetLastError());
    break;
  }
  case 267: {
  rmsnorm_backward_input<<<8, 256>>>(v[438], v[439], v[441], g[440], g[438]);
  CHECK(hipGetLastError());
  rmsnorm_backward_weight<<<8, 256>>>(v[438], v[441], g[440], g[439]);
  CHECK(hipGetLastError());
    break;
  }
  case 268: {
  linear_backward_input_mlp_up<<<dim3(128, 1), dim3(16, 16)>>>(g[443], v[442], g[440]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_up<<<dim3(128, 512), dim3(16, 16)>>>(g[443], v[440], g[442]);
  CHECK(hipGetLastError());
    break;
  }
  case 269: {
  linear_backward_input_mlp_up<<<dim3(128, 1), dim3(16, 16)>>>(g[445], v[444], g[440]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_up<<<dim3(128, 512), dim3(16, 16)>>>(g[445], v[440], g[444]);
  CHECK(hipGetLastError());
    break;
  }
  case 270: {
  swiglu_backward<<<256, 256>>>(v[443], v[445], g[446], g[443], g[445]);
  CHECK(hipGetLastError());
    break;
  }
  case 271: {
  linear_backward_input_mlp_down<<<dim3(512, 1), dim3(16, 16)>>>(g[448], v[447], g[446]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_down<<<dim3(512, 128), dim3(16, 16)>>>(g[448], v[446], g[447]);
  CHECK(hipGetLastError());
    break;
  }
  case 272: {
  add_backward<<<64, 256>>>(g[449], g[438], g[448]);
  CHECK(hipGetLastError());
    break;
  }
  case 273: {
  rmsnorm_backward_input<<<8, 256>>>(v[449], v[450], v[452], g[451], g[449]);
  CHECK(hipGetLastError());
  rmsnorm_backward_weight<<<8, 256>>>(v[449], v[452], g[451], g[450]);
  CHECK(hipGetLastError());
    break;
  }
  case 274: {
  linear_backward_input_lm_head<<<dim3(128, 1), dim3(16, 16)>>>(g[453], v[0], g[451]);
  CHECK(hipGetLastError());
  linear_backward_weight_lm_head<<<dim3(128, 8016), dim3(16, 16)>>>(g[453], v[451], g[0]);
  CHECK(hipGetLastError());
    break;
  }
  case 275: {
  cross_entropy_backward<<<8, 256>>>(v[453], targets, g[454], g[453]);
  CHECK(hipGetLastError());
    break;
  }
  case 276: {
  mean_backward<<<1, 256>>>(g[455], g[454]);
  CHECK(hipGetLastError());
    break;
  }
  default: return int(hipErrorInvalidValue);
  }
  return hipDeviceSynchronize();
}
