// GENERATED BACKWARD execution order. Do not edit; run train.py emit.
// v[] = saved forward values; g[] = accumulated gradients.
extern "C" int run_backward(float** v, float** g, const int* ids, const int* targets) {
  // The runner zeros all g[] before backward. Seed dLoss/dLoss = 1.
  float seed = 1.0f;
  CHECK(hipMemcpy(g[455], &seed, sizeof(float), hipMemcpyHostToDevice));
  // loss: mean -> (1,)
  mean_backward<<<1, 256>>>(g[455], g[454]);
  CHECK(hipGetLastError());
  // token_losses: cross_entropy -> (8,)
  cross_entropy_backward<<<8, 256>>>(v[453], targets, g[454], g[453]);
  CHECK(hipGetLastError());
  // lm_head: linear -> (8, 128256)
  linear_backward_input_lm_head<<<dim3(128, 1), dim3(16, 16)>>>(g[453], v[0], g[451]);
  CHECK(hipGetLastError());
  linear_backward_weight_lm_head<<<dim3(128, 8016), dim3(16, 16)>>>(g[453], v[451], g[0]);
  CHECK(hipGetLastError());
  // final_norm: rmsnorm -> (8, 2048)
  rmsnorm_backward_input<<<8, 256>>>(v[449], v[450], v[452], g[451], g[449]);
  CHECK(hipGetLastError());
  rmsnorm_backward_weight<<<8, 256>>>(v[449], v[452], g[451], g[450]);
  CHECK(hipGetLastError());
  // layer15.ffn_residual: add -> (8, 2048)
  add_backward<<<64, 256>>>(g[449], g[438], g[448]);
  CHECK(hipGetLastError());
  // layer15.down: linear -> (8, 2048)
  linear_backward_input_mlp_down<<<dim3(512, 1), dim3(16, 16)>>>(g[448], v[447], g[446]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_down<<<dim3(512, 128), dim3(16, 16)>>>(g[448], v[446], g[447]);
  CHECK(hipGetLastError());
  // layer15.swiglu: swiglu -> (8, 8192)
  swiglu_backward<<<256, 256>>>(v[443], v[445], g[446], g[443], g[445]);
  CHECK(hipGetLastError());
  // layer15.up: linear -> (8, 8192)
  linear_backward_input_mlp_up<<<dim3(128, 1), dim3(16, 16)>>>(g[445], v[444], g[440]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_up<<<dim3(128, 512), dim3(16, 16)>>>(g[445], v[440], g[444]);
  CHECK(hipGetLastError());
  // layer15.gate: linear -> (8, 8192)
  linear_backward_input_mlp_up<<<dim3(128, 1), dim3(16, 16)>>>(g[443], v[442], g[440]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_up<<<dim3(128, 512), dim3(16, 16)>>>(g[443], v[440], g[442]);
  CHECK(hipGetLastError());
  // layer15.ffn_norm: rmsnorm -> (8, 2048)
  rmsnorm_backward_input<<<8, 256>>>(v[438], v[439], v[441], g[440], g[438]);
  CHECK(hipGetLastError());
  rmsnorm_backward_weight<<<8, 256>>>(v[438], v[441], g[440], g[439]);
  CHECK(hipGetLastError());
  // layer15.attention_residual: add -> (8, 2048)
  add_backward<<<64, 256>>>(g[438], g[421], g[437]);
  CHECK(hipGetLastError());
  // layer15.o: linear -> (8, 2048)
  linear_backward_input_hidden<<<dim3(128, 1), dim3(16, 16)>>>(g[437], v[436], g[435]);
  CHECK(hipGetLastError());
  linear_backward_weight_hidden<<<dim3(128, 128), dim3(16, 16)>>>(g[437], v[435], g[436]);
  CHECK(hipGetLastError());
  // layer15.context: values -> (8, 2048)
  attention_values_backward_p<<<8, 256>>>(g[435], v[430], g[434]);
  CHECK(hipGetLastError());
  attention_values_backward_v<<<16, 256>>>(g[435], v[434], g[430]);
  CHECK(hipGetLastError());
  // layer15.probabilities: softmax -> (1, 32, 8, 8)
  softmax_backward<<<256, 256>>>(v[434], g[434], g[433]);
  CHECK(hipGetLastError());
  // layer15.scores: scores -> (1, 32, 8, 8)
  attention_scores_backward_q<<<64, 256>>>(g[433], v[432], g[431]);
  CHECK(hipGetLastError());
  attention_scores_backward_k<<<16, 256>>>(g[433], v[431], g[432]);
  CHECK(hipGetLastError());
  // layer15.k_rope: rope -> (8, 512)
  rope_backward_k<<<8, 256>>>(g[432], g[428]);
  CHECK(hipGetLastError());
  // layer15.q_rope: rope -> (8, 2048)
  rope_backward_q<<<32, 256>>>(g[431], g[426]);
  CHECK(hipGetLastError());
  // layer15.v: linear -> (8, 512)
  linear_backward_input_kv<<<dim3(128, 1), dim3(16, 16)>>>(g[430], v[429], g[423]);
  CHECK(hipGetLastError());
  linear_backward_weight_kv<<<dim3(128, 32), dim3(16, 16)>>>(g[430], v[423], g[429]);
  CHECK(hipGetLastError());
  // layer15.k: linear -> (8, 512)
  linear_backward_input_kv<<<dim3(128, 1), dim3(16, 16)>>>(g[428], v[427], g[423]);
  CHECK(hipGetLastError());
  linear_backward_weight_kv<<<dim3(128, 32), dim3(16, 16)>>>(g[428], v[423], g[427]);
  CHECK(hipGetLastError());
  // layer15.q: linear -> (8, 2048)
  linear_backward_input_hidden<<<dim3(128, 1), dim3(16, 16)>>>(g[426], v[425], g[423]);
  CHECK(hipGetLastError());
  linear_backward_weight_hidden<<<dim3(128, 128), dim3(16, 16)>>>(g[426], v[423], g[425]);
  CHECK(hipGetLastError());
  // layer15.attention_norm: rmsnorm -> (8, 2048)
  rmsnorm_backward_input<<<8, 256>>>(v[421], v[422], v[424], g[423], g[421]);
  CHECK(hipGetLastError());
  rmsnorm_backward_weight<<<8, 256>>>(v[421], v[424], g[423], g[422]);
  CHECK(hipGetLastError());
  // layer14.ffn_residual: add -> (8, 2048)
  add_backward<<<64, 256>>>(g[421], g[410], g[420]);
  CHECK(hipGetLastError());
  // layer14.down: linear -> (8, 2048)
  linear_backward_input_mlp_down<<<dim3(512, 1), dim3(16, 16)>>>(g[420], v[419], g[418]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_down<<<dim3(512, 128), dim3(16, 16)>>>(g[420], v[418], g[419]);
  CHECK(hipGetLastError());
  // layer14.swiglu: swiglu -> (8, 8192)
  swiglu_backward<<<256, 256>>>(v[415], v[417], g[418], g[415], g[417]);
  CHECK(hipGetLastError());
  // layer14.up: linear -> (8, 8192)
  linear_backward_input_mlp_up<<<dim3(128, 1), dim3(16, 16)>>>(g[417], v[416], g[412]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_up<<<dim3(128, 512), dim3(16, 16)>>>(g[417], v[412], g[416]);
  CHECK(hipGetLastError());
  // layer14.gate: linear -> (8, 8192)
  linear_backward_input_mlp_up<<<dim3(128, 1), dim3(16, 16)>>>(g[415], v[414], g[412]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_up<<<dim3(128, 512), dim3(16, 16)>>>(g[415], v[412], g[414]);
  CHECK(hipGetLastError());
  // layer14.ffn_norm: rmsnorm -> (8, 2048)
  rmsnorm_backward_input<<<8, 256>>>(v[410], v[411], v[413], g[412], g[410]);
  CHECK(hipGetLastError());
  rmsnorm_backward_weight<<<8, 256>>>(v[410], v[413], g[412], g[411]);
  CHECK(hipGetLastError());
  // layer14.attention_residual: add -> (8, 2048)
  add_backward<<<64, 256>>>(g[410], g[393], g[409]);
  CHECK(hipGetLastError());
  // layer14.o: linear -> (8, 2048)
  linear_backward_input_hidden<<<dim3(128, 1), dim3(16, 16)>>>(g[409], v[408], g[407]);
  CHECK(hipGetLastError());
  linear_backward_weight_hidden<<<dim3(128, 128), dim3(16, 16)>>>(g[409], v[407], g[408]);
  CHECK(hipGetLastError());
  // layer14.context: values -> (8, 2048)
  attention_values_backward_p<<<8, 256>>>(g[407], v[402], g[406]);
  CHECK(hipGetLastError());
  attention_values_backward_v<<<16, 256>>>(g[407], v[406], g[402]);
  CHECK(hipGetLastError());
  // layer14.probabilities: softmax -> (1, 32, 8, 8)
  softmax_backward<<<256, 256>>>(v[406], g[406], g[405]);
  CHECK(hipGetLastError());
  // layer14.scores: scores -> (1, 32, 8, 8)
  attention_scores_backward_q<<<64, 256>>>(g[405], v[404], g[403]);
  CHECK(hipGetLastError());
  attention_scores_backward_k<<<16, 256>>>(g[405], v[403], g[404]);
  CHECK(hipGetLastError());
  // layer14.k_rope: rope -> (8, 512)
  rope_backward_k<<<8, 256>>>(g[404], g[400]);
  CHECK(hipGetLastError());
  // layer14.q_rope: rope -> (8, 2048)
  rope_backward_q<<<32, 256>>>(g[403], g[398]);
  CHECK(hipGetLastError());
  // layer14.v: linear -> (8, 512)
  linear_backward_input_kv<<<dim3(128, 1), dim3(16, 16)>>>(g[402], v[401], g[395]);
  CHECK(hipGetLastError());
  linear_backward_weight_kv<<<dim3(128, 32), dim3(16, 16)>>>(g[402], v[395], g[401]);
  CHECK(hipGetLastError());
  // layer14.k: linear -> (8, 512)
  linear_backward_input_kv<<<dim3(128, 1), dim3(16, 16)>>>(g[400], v[399], g[395]);
  CHECK(hipGetLastError());
  linear_backward_weight_kv<<<dim3(128, 32), dim3(16, 16)>>>(g[400], v[395], g[399]);
  CHECK(hipGetLastError());
  // layer14.q: linear -> (8, 2048)
  linear_backward_input_hidden<<<dim3(128, 1), dim3(16, 16)>>>(g[398], v[397], g[395]);
  CHECK(hipGetLastError());
  linear_backward_weight_hidden<<<dim3(128, 128), dim3(16, 16)>>>(g[398], v[395], g[397]);
  CHECK(hipGetLastError());
  // layer14.attention_norm: rmsnorm -> (8, 2048)
  rmsnorm_backward_input<<<8, 256>>>(v[393], v[394], v[396], g[395], g[393]);
  CHECK(hipGetLastError());
  rmsnorm_backward_weight<<<8, 256>>>(v[393], v[396], g[395], g[394]);
  CHECK(hipGetLastError());
  // layer13.ffn_residual: add -> (8, 2048)
  add_backward<<<64, 256>>>(g[393], g[382], g[392]);
  CHECK(hipGetLastError());
  // layer13.down: linear -> (8, 2048)
  linear_backward_input_mlp_down<<<dim3(512, 1), dim3(16, 16)>>>(g[392], v[391], g[390]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_down<<<dim3(512, 128), dim3(16, 16)>>>(g[392], v[390], g[391]);
  CHECK(hipGetLastError());
  // layer13.swiglu: swiglu -> (8, 8192)
  swiglu_backward<<<256, 256>>>(v[387], v[389], g[390], g[387], g[389]);
  CHECK(hipGetLastError());
  // layer13.up: linear -> (8, 8192)
  linear_backward_input_mlp_up<<<dim3(128, 1), dim3(16, 16)>>>(g[389], v[388], g[384]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_up<<<dim3(128, 512), dim3(16, 16)>>>(g[389], v[384], g[388]);
  CHECK(hipGetLastError());
  // layer13.gate: linear -> (8, 8192)
  linear_backward_input_mlp_up<<<dim3(128, 1), dim3(16, 16)>>>(g[387], v[386], g[384]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_up<<<dim3(128, 512), dim3(16, 16)>>>(g[387], v[384], g[386]);
  CHECK(hipGetLastError());
  // layer13.ffn_norm: rmsnorm -> (8, 2048)
  rmsnorm_backward_input<<<8, 256>>>(v[382], v[383], v[385], g[384], g[382]);
  CHECK(hipGetLastError());
  rmsnorm_backward_weight<<<8, 256>>>(v[382], v[385], g[384], g[383]);
  CHECK(hipGetLastError());
  // layer13.attention_residual: add -> (8, 2048)
  add_backward<<<64, 256>>>(g[382], g[365], g[381]);
  CHECK(hipGetLastError());
  // layer13.o: linear -> (8, 2048)
  linear_backward_input_hidden<<<dim3(128, 1), dim3(16, 16)>>>(g[381], v[380], g[379]);
  CHECK(hipGetLastError());
  linear_backward_weight_hidden<<<dim3(128, 128), dim3(16, 16)>>>(g[381], v[379], g[380]);
  CHECK(hipGetLastError());
  // layer13.context: values -> (8, 2048)
  attention_values_backward_p<<<8, 256>>>(g[379], v[374], g[378]);
  CHECK(hipGetLastError());
  attention_values_backward_v<<<16, 256>>>(g[379], v[378], g[374]);
  CHECK(hipGetLastError());
  // layer13.probabilities: softmax -> (1, 32, 8, 8)
  softmax_backward<<<256, 256>>>(v[378], g[378], g[377]);
  CHECK(hipGetLastError());
  // layer13.scores: scores -> (1, 32, 8, 8)
  attention_scores_backward_q<<<64, 256>>>(g[377], v[376], g[375]);
  CHECK(hipGetLastError());
  attention_scores_backward_k<<<16, 256>>>(g[377], v[375], g[376]);
  CHECK(hipGetLastError());
  // layer13.k_rope: rope -> (8, 512)
  rope_backward_k<<<8, 256>>>(g[376], g[372]);
  CHECK(hipGetLastError());
  // layer13.q_rope: rope -> (8, 2048)
  rope_backward_q<<<32, 256>>>(g[375], g[370]);
  CHECK(hipGetLastError());
  // layer13.v: linear -> (8, 512)
  linear_backward_input_kv<<<dim3(128, 1), dim3(16, 16)>>>(g[374], v[373], g[367]);
  CHECK(hipGetLastError());
  linear_backward_weight_kv<<<dim3(128, 32), dim3(16, 16)>>>(g[374], v[367], g[373]);
  CHECK(hipGetLastError());
  // layer13.k: linear -> (8, 512)
  linear_backward_input_kv<<<dim3(128, 1), dim3(16, 16)>>>(g[372], v[371], g[367]);
  CHECK(hipGetLastError());
  linear_backward_weight_kv<<<dim3(128, 32), dim3(16, 16)>>>(g[372], v[367], g[371]);
  CHECK(hipGetLastError());
  // layer13.q: linear -> (8, 2048)
  linear_backward_input_hidden<<<dim3(128, 1), dim3(16, 16)>>>(g[370], v[369], g[367]);
  CHECK(hipGetLastError());
  linear_backward_weight_hidden<<<dim3(128, 128), dim3(16, 16)>>>(g[370], v[367], g[369]);
  CHECK(hipGetLastError());
  // layer13.attention_norm: rmsnorm -> (8, 2048)
  rmsnorm_backward_input<<<8, 256>>>(v[365], v[366], v[368], g[367], g[365]);
  CHECK(hipGetLastError());
  rmsnorm_backward_weight<<<8, 256>>>(v[365], v[368], g[367], g[366]);
  CHECK(hipGetLastError());
  // layer12.ffn_residual: add -> (8, 2048)
  add_backward<<<64, 256>>>(g[365], g[354], g[364]);
  CHECK(hipGetLastError());
  // layer12.down: linear -> (8, 2048)
  linear_backward_input_mlp_down<<<dim3(512, 1), dim3(16, 16)>>>(g[364], v[363], g[362]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_down<<<dim3(512, 128), dim3(16, 16)>>>(g[364], v[362], g[363]);
  CHECK(hipGetLastError());
  // layer12.swiglu: swiglu -> (8, 8192)
  swiglu_backward<<<256, 256>>>(v[359], v[361], g[362], g[359], g[361]);
  CHECK(hipGetLastError());
  // layer12.up: linear -> (8, 8192)
  linear_backward_input_mlp_up<<<dim3(128, 1), dim3(16, 16)>>>(g[361], v[360], g[356]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_up<<<dim3(128, 512), dim3(16, 16)>>>(g[361], v[356], g[360]);
  CHECK(hipGetLastError());
  // layer12.gate: linear -> (8, 8192)
  linear_backward_input_mlp_up<<<dim3(128, 1), dim3(16, 16)>>>(g[359], v[358], g[356]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_up<<<dim3(128, 512), dim3(16, 16)>>>(g[359], v[356], g[358]);
  CHECK(hipGetLastError());
  // layer12.ffn_norm: rmsnorm -> (8, 2048)
  rmsnorm_backward_input<<<8, 256>>>(v[354], v[355], v[357], g[356], g[354]);
  CHECK(hipGetLastError());
  rmsnorm_backward_weight<<<8, 256>>>(v[354], v[357], g[356], g[355]);
  CHECK(hipGetLastError());
  // layer12.attention_residual: add -> (8, 2048)
  add_backward<<<64, 256>>>(g[354], g[337], g[353]);
  CHECK(hipGetLastError());
  // layer12.o: linear -> (8, 2048)
  linear_backward_input_hidden<<<dim3(128, 1), dim3(16, 16)>>>(g[353], v[352], g[351]);
  CHECK(hipGetLastError());
  linear_backward_weight_hidden<<<dim3(128, 128), dim3(16, 16)>>>(g[353], v[351], g[352]);
  CHECK(hipGetLastError());
  // layer12.context: values -> (8, 2048)
  attention_values_backward_p<<<8, 256>>>(g[351], v[346], g[350]);
  CHECK(hipGetLastError());
  attention_values_backward_v<<<16, 256>>>(g[351], v[350], g[346]);
  CHECK(hipGetLastError());
  // layer12.probabilities: softmax -> (1, 32, 8, 8)
  softmax_backward<<<256, 256>>>(v[350], g[350], g[349]);
  CHECK(hipGetLastError());
  // layer12.scores: scores -> (1, 32, 8, 8)
  attention_scores_backward_q<<<64, 256>>>(g[349], v[348], g[347]);
  CHECK(hipGetLastError());
  attention_scores_backward_k<<<16, 256>>>(g[349], v[347], g[348]);
  CHECK(hipGetLastError());
  // layer12.k_rope: rope -> (8, 512)
  rope_backward_k<<<8, 256>>>(g[348], g[344]);
  CHECK(hipGetLastError());
  // layer12.q_rope: rope -> (8, 2048)
  rope_backward_q<<<32, 256>>>(g[347], g[342]);
  CHECK(hipGetLastError());
  // layer12.v: linear -> (8, 512)
  linear_backward_input_kv<<<dim3(128, 1), dim3(16, 16)>>>(g[346], v[345], g[339]);
  CHECK(hipGetLastError());
  linear_backward_weight_kv<<<dim3(128, 32), dim3(16, 16)>>>(g[346], v[339], g[345]);
  CHECK(hipGetLastError());
  // layer12.k: linear -> (8, 512)
  linear_backward_input_kv<<<dim3(128, 1), dim3(16, 16)>>>(g[344], v[343], g[339]);
  CHECK(hipGetLastError());
  linear_backward_weight_kv<<<dim3(128, 32), dim3(16, 16)>>>(g[344], v[339], g[343]);
  CHECK(hipGetLastError());
  // layer12.q: linear -> (8, 2048)
  linear_backward_input_hidden<<<dim3(128, 1), dim3(16, 16)>>>(g[342], v[341], g[339]);
  CHECK(hipGetLastError());
  linear_backward_weight_hidden<<<dim3(128, 128), dim3(16, 16)>>>(g[342], v[339], g[341]);
  CHECK(hipGetLastError());
  // layer12.attention_norm: rmsnorm -> (8, 2048)
  rmsnorm_backward_input<<<8, 256>>>(v[337], v[338], v[340], g[339], g[337]);
  CHECK(hipGetLastError());
  rmsnorm_backward_weight<<<8, 256>>>(v[337], v[340], g[339], g[338]);
  CHECK(hipGetLastError());
  // layer11.ffn_residual: add -> (8, 2048)
  add_backward<<<64, 256>>>(g[337], g[326], g[336]);
  CHECK(hipGetLastError());
  // layer11.down: linear -> (8, 2048)
  linear_backward_input_mlp_down<<<dim3(512, 1), dim3(16, 16)>>>(g[336], v[335], g[334]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_down<<<dim3(512, 128), dim3(16, 16)>>>(g[336], v[334], g[335]);
  CHECK(hipGetLastError());
  // layer11.swiglu: swiglu -> (8, 8192)
  swiglu_backward<<<256, 256>>>(v[331], v[333], g[334], g[331], g[333]);
  CHECK(hipGetLastError());
  // layer11.up: linear -> (8, 8192)
  linear_backward_input_mlp_up<<<dim3(128, 1), dim3(16, 16)>>>(g[333], v[332], g[328]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_up<<<dim3(128, 512), dim3(16, 16)>>>(g[333], v[328], g[332]);
  CHECK(hipGetLastError());
  // layer11.gate: linear -> (8, 8192)
  linear_backward_input_mlp_up<<<dim3(128, 1), dim3(16, 16)>>>(g[331], v[330], g[328]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_up<<<dim3(128, 512), dim3(16, 16)>>>(g[331], v[328], g[330]);
  CHECK(hipGetLastError());
  // layer11.ffn_norm: rmsnorm -> (8, 2048)
  rmsnorm_backward_input<<<8, 256>>>(v[326], v[327], v[329], g[328], g[326]);
  CHECK(hipGetLastError());
  rmsnorm_backward_weight<<<8, 256>>>(v[326], v[329], g[328], g[327]);
  CHECK(hipGetLastError());
  // layer11.attention_residual: add -> (8, 2048)
  add_backward<<<64, 256>>>(g[326], g[309], g[325]);
  CHECK(hipGetLastError());
  // layer11.o: linear -> (8, 2048)
  linear_backward_input_hidden<<<dim3(128, 1), dim3(16, 16)>>>(g[325], v[324], g[323]);
  CHECK(hipGetLastError());
  linear_backward_weight_hidden<<<dim3(128, 128), dim3(16, 16)>>>(g[325], v[323], g[324]);
  CHECK(hipGetLastError());
  // layer11.context: values -> (8, 2048)
  attention_values_backward_p<<<8, 256>>>(g[323], v[318], g[322]);
  CHECK(hipGetLastError());
  attention_values_backward_v<<<16, 256>>>(g[323], v[322], g[318]);
  CHECK(hipGetLastError());
  // layer11.probabilities: softmax -> (1, 32, 8, 8)
  softmax_backward<<<256, 256>>>(v[322], g[322], g[321]);
  CHECK(hipGetLastError());
  // layer11.scores: scores -> (1, 32, 8, 8)
  attention_scores_backward_q<<<64, 256>>>(g[321], v[320], g[319]);
  CHECK(hipGetLastError());
  attention_scores_backward_k<<<16, 256>>>(g[321], v[319], g[320]);
  CHECK(hipGetLastError());
  // layer11.k_rope: rope -> (8, 512)
  rope_backward_k<<<8, 256>>>(g[320], g[316]);
  CHECK(hipGetLastError());
  // layer11.q_rope: rope -> (8, 2048)
  rope_backward_q<<<32, 256>>>(g[319], g[314]);
  CHECK(hipGetLastError());
  // layer11.v: linear -> (8, 512)
  linear_backward_input_kv<<<dim3(128, 1), dim3(16, 16)>>>(g[318], v[317], g[311]);
  CHECK(hipGetLastError());
  linear_backward_weight_kv<<<dim3(128, 32), dim3(16, 16)>>>(g[318], v[311], g[317]);
  CHECK(hipGetLastError());
  // layer11.k: linear -> (8, 512)
  linear_backward_input_kv<<<dim3(128, 1), dim3(16, 16)>>>(g[316], v[315], g[311]);
  CHECK(hipGetLastError());
  linear_backward_weight_kv<<<dim3(128, 32), dim3(16, 16)>>>(g[316], v[311], g[315]);
  CHECK(hipGetLastError());
  // layer11.q: linear -> (8, 2048)
  linear_backward_input_hidden<<<dim3(128, 1), dim3(16, 16)>>>(g[314], v[313], g[311]);
  CHECK(hipGetLastError());
  linear_backward_weight_hidden<<<dim3(128, 128), dim3(16, 16)>>>(g[314], v[311], g[313]);
  CHECK(hipGetLastError());
  // layer11.attention_norm: rmsnorm -> (8, 2048)
  rmsnorm_backward_input<<<8, 256>>>(v[309], v[310], v[312], g[311], g[309]);
  CHECK(hipGetLastError());
  rmsnorm_backward_weight<<<8, 256>>>(v[309], v[312], g[311], g[310]);
  CHECK(hipGetLastError());
  // layer10.ffn_residual: add -> (8, 2048)
  add_backward<<<64, 256>>>(g[309], g[298], g[308]);
  CHECK(hipGetLastError());
  // layer10.down: linear -> (8, 2048)
  linear_backward_input_mlp_down<<<dim3(512, 1), dim3(16, 16)>>>(g[308], v[307], g[306]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_down<<<dim3(512, 128), dim3(16, 16)>>>(g[308], v[306], g[307]);
  CHECK(hipGetLastError());
  // layer10.swiglu: swiglu -> (8, 8192)
  swiglu_backward<<<256, 256>>>(v[303], v[305], g[306], g[303], g[305]);
  CHECK(hipGetLastError());
  // layer10.up: linear -> (8, 8192)
  linear_backward_input_mlp_up<<<dim3(128, 1), dim3(16, 16)>>>(g[305], v[304], g[300]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_up<<<dim3(128, 512), dim3(16, 16)>>>(g[305], v[300], g[304]);
  CHECK(hipGetLastError());
  // layer10.gate: linear -> (8, 8192)
  linear_backward_input_mlp_up<<<dim3(128, 1), dim3(16, 16)>>>(g[303], v[302], g[300]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_up<<<dim3(128, 512), dim3(16, 16)>>>(g[303], v[300], g[302]);
  CHECK(hipGetLastError());
  // layer10.ffn_norm: rmsnorm -> (8, 2048)
  rmsnorm_backward_input<<<8, 256>>>(v[298], v[299], v[301], g[300], g[298]);
  CHECK(hipGetLastError());
  rmsnorm_backward_weight<<<8, 256>>>(v[298], v[301], g[300], g[299]);
  CHECK(hipGetLastError());
  // layer10.attention_residual: add -> (8, 2048)
  add_backward<<<64, 256>>>(g[298], g[281], g[297]);
  CHECK(hipGetLastError());
  // layer10.o: linear -> (8, 2048)
  linear_backward_input_hidden<<<dim3(128, 1), dim3(16, 16)>>>(g[297], v[296], g[295]);
  CHECK(hipGetLastError());
  linear_backward_weight_hidden<<<dim3(128, 128), dim3(16, 16)>>>(g[297], v[295], g[296]);
  CHECK(hipGetLastError());
  // layer10.context: values -> (8, 2048)
  attention_values_backward_p<<<8, 256>>>(g[295], v[290], g[294]);
  CHECK(hipGetLastError());
  attention_values_backward_v<<<16, 256>>>(g[295], v[294], g[290]);
  CHECK(hipGetLastError());
  // layer10.probabilities: softmax -> (1, 32, 8, 8)
  softmax_backward<<<256, 256>>>(v[294], g[294], g[293]);
  CHECK(hipGetLastError());
  // layer10.scores: scores -> (1, 32, 8, 8)
  attention_scores_backward_q<<<64, 256>>>(g[293], v[292], g[291]);
  CHECK(hipGetLastError());
  attention_scores_backward_k<<<16, 256>>>(g[293], v[291], g[292]);
  CHECK(hipGetLastError());
  // layer10.k_rope: rope -> (8, 512)
  rope_backward_k<<<8, 256>>>(g[292], g[288]);
  CHECK(hipGetLastError());
  // layer10.q_rope: rope -> (8, 2048)
  rope_backward_q<<<32, 256>>>(g[291], g[286]);
  CHECK(hipGetLastError());
  // layer10.v: linear -> (8, 512)
  linear_backward_input_kv<<<dim3(128, 1), dim3(16, 16)>>>(g[290], v[289], g[283]);
  CHECK(hipGetLastError());
  linear_backward_weight_kv<<<dim3(128, 32), dim3(16, 16)>>>(g[290], v[283], g[289]);
  CHECK(hipGetLastError());
  // layer10.k: linear -> (8, 512)
  linear_backward_input_kv<<<dim3(128, 1), dim3(16, 16)>>>(g[288], v[287], g[283]);
  CHECK(hipGetLastError());
  linear_backward_weight_kv<<<dim3(128, 32), dim3(16, 16)>>>(g[288], v[283], g[287]);
  CHECK(hipGetLastError());
  // layer10.q: linear -> (8, 2048)
  linear_backward_input_hidden<<<dim3(128, 1), dim3(16, 16)>>>(g[286], v[285], g[283]);
  CHECK(hipGetLastError());
  linear_backward_weight_hidden<<<dim3(128, 128), dim3(16, 16)>>>(g[286], v[283], g[285]);
  CHECK(hipGetLastError());
  // layer10.attention_norm: rmsnorm -> (8, 2048)
  rmsnorm_backward_input<<<8, 256>>>(v[281], v[282], v[284], g[283], g[281]);
  CHECK(hipGetLastError());
  rmsnorm_backward_weight<<<8, 256>>>(v[281], v[284], g[283], g[282]);
  CHECK(hipGetLastError());
  // layer9.ffn_residual: add -> (8, 2048)
  add_backward<<<64, 256>>>(g[281], g[270], g[280]);
  CHECK(hipGetLastError());
  // layer9.down: linear -> (8, 2048)
  linear_backward_input_mlp_down<<<dim3(512, 1), dim3(16, 16)>>>(g[280], v[279], g[278]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_down<<<dim3(512, 128), dim3(16, 16)>>>(g[280], v[278], g[279]);
  CHECK(hipGetLastError());
  // layer9.swiglu: swiglu -> (8, 8192)
  swiglu_backward<<<256, 256>>>(v[275], v[277], g[278], g[275], g[277]);
  CHECK(hipGetLastError());
  // layer9.up: linear -> (8, 8192)
  linear_backward_input_mlp_up<<<dim3(128, 1), dim3(16, 16)>>>(g[277], v[276], g[272]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_up<<<dim3(128, 512), dim3(16, 16)>>>(g[277], v[272], g[276]);
  CHECK(hipGetLastError());
  // layer9.gate: linear -> (8, 8192)
  linear_backward_input_mlp_up<<<dim3(128, 1), dim3(16, 16)>>>(g[275], v[274], g[272]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_up<<<dim3(128, 512), dim3(16, 16)>>>(g[275], v[272], g[274]);
  CHECK(hipGetLastError());
  // layer9.ffn_norm: rmsnorm -> (8, 2048)
  rmsnorm_backward_input<<<8, 256>>>(v[270], v[271], v[273], g[272], g[270]);
  CHECK(hipGetLastError());
  rmsnorm_backward_weight<<<8, 256>>>(v[270], v[273], g[272], g[271]);
  CHECK(hipGetLastError());
  // layer9.attention_residual: add -> (8, 2048)
  add_backward<<<64, 256>>>(g[270], g[253], g[269]);
  CHECK(hipGetLastError());
  // layer9.o: linear -> (8, 2048)
  linear_backward_input_hidden<<<dim3(128, 1), dim3(16, 16)>>>(g[269], v[268], g[267]);
  CHECK(hipGetLastError());
  linear_backward_weight_hidden<<<dim3(128, 128), dim3(16, 16)>>>(g[269], v[267], g[268]);
  CHECK(hipGetLastError());
  // layer9.context: values -> (8, 2048)
  attention_values_backward_p<<<8, 256>>>(g[267], v[262], g[266]);
  CHECK(hipGetLastError());
  attention_values_backward_v<<<16, 256>>>(g[267], v[266], g[262]);
  CHECK(hipGetLastError());
  // layer9.probabilities: softmax -> (1, 32, 8, 8)
  softmax_backward<<<256, 256>>>(v[266], g[266], g[265]);
  CHECK(hipGetLastError());
  // layer9.scores: scores -> (1, 32, 8, 8)
  attention_scores_backward_q<<<64, 256>>>(g[265], v[264], g[263]);
  CHECK(hipGetLastError());
  attention_scores_backward_k<<<16, 256>>>(g[265], v[263], g[264]);
  CHECK(hipGetLastError());
  // layer9.k_rope: rope -> (8, 512)
  rope_backward_k<<<8, 256>>>(g[264], g[260]);
  CHECK(hipGetLastError());
  // layer9.q_rope: rope -> (8, 2048)
  rope_backward_q<<<32, 256>>>(g[263], g[258]);
  CHECK(hipGetLastError());
  // layer9.v: linear -> (8, 512)
  linear_backward_input_kv<<<dim3(128, 1), dim3(16, 16)>>>(g[262], v[261], g[255]);
  CHECK(hipGetLastError());
  linear_backward_weight_kv<<<dim3(128, 32), dim3(16, 16)>>>(g[262], v[255], g[261]);
  CHECK(hipGetLastError());
  // layer9.k: linear -> (8, 512)
  linear_backward_input_kv<<<dim3(128, 1), dim3(16, 16)>>>(g[260], v[259], g[255]);
  CHECK(hipGetLastError());
  linear_backward_weight_kv<<<dim3(128, 32), dim3(16, 16)>>>(g[260], v[255], g[259]);
  CHECK(hipGetLastError());
  // layer9.q: linear -> (8, 2048)
  linear_backward_input_hidden<<<dim3(128, 1), dim3(16, 16)>>>(g[258], v[257], g[255]);
  CHECK(hipGetLastError());
  linear_backward_weight_hidden<<<dim3(128, 128), dim3(16, 16)>>>(g[258], v[255], g[257]);
  CHECK(hipGetLastError());
  // layer9.attention_norm: rmsnorm -> (8, 2048)
  rmsnorm_backward_input<<<8, 256>>>(v[253], v[254], v[256], g[255], g[253]);
  CHECK(hipGetLastError());
  rmsnorm_backward_weight<<<8, 256>>>(v[253], v[256], g[255], g[254]);
  CHECK(hipGetLastError());
  // layer8.ffn_residual: add -> (8, 2048)
  add_backward<<<64, 256>>>(g[253], g[242], g[252]);
  CHECK(hipGetLastError());
  // layer8.down: linear -> (8, 2048)
  linear_backward_input_mlp_down<<<dim3(512, 1), dim3(16, 16)>>>(g[252], v[251], g[250]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_down<<<dim3(512, 128), dim3(16, 16)>>>(g[252], v[250], g[251]);
  CHECK(hipGetLastError());
  // layer8.swiglu: swiglu -> (8, 8192)
  swiglu_backward<<<256, 256>>>(v[247], v[249], g[250], g[247], g[249]);
  CHECK(hipGetLastError());
  // layer8.up: linear -> (8, 8192)
  linear_backward_input_mlp_up<<<dim3(128, 1), dim3(16, 16)>>>(g[249], v[248], g[244]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_up<<<dim3(128, 512), dim3(16, 16)>>>(g[249], v[244], g[248]);
  CHECK(hipGetLastError());
  // layer8.gate: linear -> (8, 8192)
  linear_backward_input_mlp_up<<<dim3(128, 1), dim3(16, 16)>>>(g[247], v[246], g[244]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_up<<<dim3(128, 512), dim3(16, 16)>>>(g[247], v[244], g[246]);
  CHECK(hipGetLastError());
  // layer8.ffn_norm: rmsnorm -> (8, 2048)
  rmsnorm_backward_input<<<8, 256>>>(v[242], v[243], v[245], g[244], g[242]);
  CHECK(hipGetLastError());
  rmsnorm_backward_weight<<<8, 256>>>(v[242], v[245], g[244], g[243]);
  CHECK(hipGetLastError());
  // layer8.attention_residual: add -> (8, 2048)
  add_backward<<<64, 256>>>(g[242], g[225], g[241]);
  CHECK(hipGetLastError());
  // layer8.o: linear -> (8, 2048)
  linear_backward_input_hidden<<<dim3(128, 1), dim3(16, 16)>>>(g[241], v[240], g[239]);
  CHECK(hipGetLastError());
  linear_backward_weight_hidden<<<dim3(128, 128), dim3(16, 16)>>>(g[241], v[239], g[240]);
  CHECK(hipGetLastError());
  // layer8.context: values -> (8, 2048)
  attention_values_backward_p<<<8, 256>>>(g[239], v[234], g[238]);
  CHECK(hipGetLastError());
  attention_values_backward_v<<<16, 256>>>(g[239], v[238], g[234]);
  CHECK(hipGetLastError());
  // layer8.probabilities: softmax -> (1, 32, 8, 8)
  softmax_backward<<<256, 256>>>(v[238], g[238], g[237]);
  CHECK(hipGetLastError());
  // layer8.scores: scores -> (1, 32, 8, 8)
  attention_scores_backward_q<<<64, 256>>>(g[237], v[236], g[235]);
  CHECK(hipGetLastError());
  attention_scores_backward_k<<<16, 256>>>(g[237], v[235], g[236]);
  CHECK(hipGetLastError());
  // layer8.k_rope: rope -> (8, 512)
  rope_backward_k<<<8, 256>>>(g[236], g[232]);
  CHECK(hipGetLastError());
  // layer8.q_rope: rope -> (8, 2048)
  rope_backward_q<<<32, 256>>>(g[235], g[230]);
  CHECK(hipGetLastError());
  // layer8.v: linear -> (8, 512)
  linear_backward_input_kv<<<dim3(128, 1), dim3(16, 16)>>>(g[234], v[233], g[227]);
  CHECK(hipGetLastError());
  linear_backward_weight_kv<<<dim3(128, 32), dim3(16, 16)>>>(g[234], v[227], g[233]);
  CHECK(hipGetLastError());
  // layer8.k: linear -> (8, 512)
  linear_backward_input_kv<<<dim3(128, 1), dim3(16, 16)>>>(g[232], v[231], g[227]);
  CHECK(hipGetLastError());
  linear_backward_weight_kv<<<dim3(128, 32), dim3(16, 16)>>>(g[232], v[227], g[231]);
  CHECK(hipGetLastError());
  // layer8.q: linear -> (8, 2048)
  linear_backward_input_hidden<<<dim3(128, 1), dim3(16, 16)>>>(g[230], v[229], g[227]);
  CHECK(hipGetLastError());
  linear_backward_weight_hidden<<<dim3(128, 128), dim3(16, 16)>>>(g[230], v[227], g[229]);
  CHECK(hipGetLastError());
  // layer8.attention_norm: rmsnorm -> (8, 2048)
  rmsnorm_backward_input<<<8, 256>>>(v[225], v[226], v[228], g[227], g[225]);
  CHECK(hipGetLastError());
  rmsnorm_backward_weight<<<8, 256>>>(v[225], v[228], g[227], g[226]);
  CHECK(hipGetLastError());
  // layer7.ffn_residual: add -> (8, 2048)
  add_backward<<<64, 256>>>(g[225], g[214], g[224]);
  CHECK(hipGetLastError());
  // layer7.down: linear -> (8, 2048)
  linear_backward_input_mlp_down<<<dim3(512, 1), dim3(16, 16)>>>(g[224], v[223], g[222]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_down<<<dim3(512, 128), dim3(16, 16)>>>(g[224], v[222], g[223]);
  CHECK(hipGetLastError());
  // layer7.swiglu: swiglu -> (8, 8192)
  swiglu_backward<<<256, 256>>>(v[219], v[221], g[222], g[219], g[221]);
  CHECK(hipGetLastError());
  // layer7.up: linear -> (8, 8192)
  linear_backward_input_mlp_up<<<dim3(128, 1), dim3(16, 16)>>>(g[221], v[220], g[216]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_up<<<dim3(128, 512), dim3(16, 16)>>>(g[221], v[216], g[220]);
  CHECK(hipGetLastError());
  // layer7.gate: linear -> (8, 8192)
  linear_backward_input_mlp_up<<<dim3(128, 1), dim3(16, 16)>>>(g[219], v[218], g[216]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_up<<<dim3(128, 512), dim3(16, 16)>>>(g[219], v[216], g[218]);
  CHECK(hipGetLastError());
  // layer7.ffn_norm: rmsnorm -> (8, 2048)
  rmsnorm_backward_input<<<8, 256>>>(v[214], v[215], v[217], g[216], g[214]);
  CHECK(hipGetLastError());
  rmsnorm_backward_weight<<<8, 256>>>(v[214], v[217], g[216], g[215]);
  CHECK(hipGetLastError());
  // layer7.attention_residual: add -> (8, 2048)
  add_backward<<<64, 256>>>(g[214], g[197], g[213]);
  CHECK(hipGetLastError());
  // layer7.o: linear -> (8, 2048)
  linear_backward_input_hidden<<<dim3(128, 1), dim3(16, 16)>>>(g[213], v[212], g[211]);
  CHECK(hipGetLastError());
  linear_backward_weight_hidden<<<dim3(128, 128), dim3(16, 16)>>>(g[213], v[211], g[212]);
  CHECK(hipGetLastError());
  // layer7.context: values -> (8, 2048)
  attention_values_backward_p<<<8, 256>>>(g[211], v[206], g[210]);
  CHECK(hipGetLastError());
  attention_values_backward_v<<<16, 256>>>(g[211], v[210], g[206]);
  CHECK(hipGetLastError());
  // layer7.probabilities: softmax -> (1, 32, 8, 8)
  softmax_backward<<<256, 256>>>(v[210], g[210], g[209]);
  CHECK(hipGetLastError());
  // layer7.scores: scores -> (1, 32, 8, 8)
  attention_scores_backward_q<<<64, 256>>>(g[209], v[208], g[207]);
  CHECK(hipGetLastError());
  attention_scores_backward_k<<<16, 256>>>(g[209], v[207], g[208]);
  CHECK(hipGetLastError());
  // layer7.k_rope: rope -> (8, 512)
  rope_backward_k<<<8, 256>>>(g[208], g[204]);
  CHECK(hipGetLastError());
  // layer7.q_rope: rope -> (8, 2048)
  rope_backward_q<<<32, 256>>>(g[207], g[202]);
  CHECK(hipGetLastError());
  // layer7.v: linear -> (8, 512)
  linear_backward_input_kv<<<dim3(128, 1), dim3(16, 16)>>>(g[206], v[205], g[199]);
  CHECK(hipGetLastError());
  linear_backward_weight_kv<<<dim3(128, 32), dim3(16, 16)>>>(g[206], v[199], g[205]);
  CHECK(hipGetLastError());
  // layer7.k: linear -> (8, 512)
  linear_backward_input_kv<<<dim3(128, 1), dim3(16, 16)>>>(g[204], v[203], g[199]);
  CHECK(hipGetLastError());
  linear_backward_weight_kv<<<dim3(128, 32), dim3(16, 16)>>>(g[204], v[199], g[203]);
  CHECK(hipGetLastError());
  // layer7.q: linear -> (8, 2048)
  linear_backward_input_hidden<<<dim3(128, 1), dim3(16, 16)>>>(g[202], v[201], g[199]);
  CHECK(hipGetLastError());
  linear_backward_weight_hidden<<<dim3(128, 128), dim3(16, 16)>>>(g[202], v[199], g[201]);
  CHECK(hipGetLastError());
  // layer7.attention_norm: rmsnorm -> (8, 2048)
  rmsnorm_backward_input<<<8, 256>>>(v[197], v[198], v[200], g[199], g[197]);
  CHECK(hipGetLastError());
  rmsnorm_backward_weight<<<8, 256>>>(v[197], v[200], g[199], g[198]);
  CHECK(hipGetLastError());
  // layer6.ffn_residual: add -> (8, 2048)
  add_backward<<<64, 256>>>(g[197], g[186], g[196]);
  CHECK(hipGetLastError());
  // layer6.down: linear -> (8, 2048)
  linear_backward_input_mlp_down<<<dim3(512, 1), dim3(16, 16)>>>(g[196], v[195], g[194]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_down<<<dim3(512, 128), dim3(16, 16)>>>(g[196], v[194], g[195]);
  CHECK(hipGetLastError());
  // layer6.swiglu: swiglu -> (8, 8192)
  swiglu_backward<<<256, 256>>>(v[191], v[193], g[194], g[191], g[193]);
  CHECK(hipGetLastError());
  // layer6.up: linear -> (8, 8192)
  linear_backward_input_mlp_up<<<dim3(128, 1), dim3(16, 16)>>>(g[193], v[192], g[188]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_up<<<dim3(128, 512), dim3(16, 16)>>>(g[193], v[188], g[192]);
  CHECK(hipGetLastError());
  // layer6.gate: linear -> (8, 8192)
  linear_backward_input_mlp_up<<<dim3(128, 1), dim3(16, 16)>>>(g[191], v[190], g[188]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_up<<<dim3(128, 512), dim3(16, 16)>>>(g[191], v[188], g[190]);
  CHECK(hipGetLastError());
  // layer6.ffn_norm: rmsnorm -> (8, 2048)
  rmsnorm_backward_input<<<8, 256>>>(v[186], v[187], v[189], g[188], g[186]);
  CHECK(hipGetLastError());
  rmsnorm_backward_weight<<<8, 256>>>(v[186], v[189], g[188], g[187]);
  CHECK(hipGetLastError());
  // layer6.attention_residual: add -> (8, 2048)
  add_backward<<<64, 256>>>(g[186], g[169], g[185]);
  CHECK(hipGetLastError());
  // layer6.o: linear -> (8, 2048)
  linear_backward_input_hidden<<<dim3(128, 1), dim3(16, 16)>>>(g[185], v[184], g[183]);
  CHECK(hipGetLastError());
  linear_backward_weight_hidden<<<dim3(128, 128), dim3(16, 16)>>>(g[185], v[183], g[184]);
  CHECK(hipGetLastError());
  // layer6.context: values -> (8, 2048)
  attention_values_backward_p<<<8, 256>>>(g[183], v[178], g[182]);
  CHECK(hipGetLastError());
  attention_values_backward_v<<<16, 256>>>(g[183], v[182], g[178]);
  CHECK(hipGetLastError());
  // layer6.probabilities: softmax -> (1, 32, 8, 8)
  softmax_backward<<<256, 256>>>(v[182], g[182], g[181]);
  CHECK(hipGetLastError());
  // layer6.scores: scores -> (1, 32, 8, 8)
  attention_scores_backward_q<<<64, 256>>>(g[181], v[180], g[179]);
  CHECK(hipGetLastError());
  attention_scores_backward_k<<<16, 256>>>(g[181], v[179], g[180]);
  CHECK(hipGetLastError());
  // layer6.k_rope: rope -> (8, 512)
  rope_backward_k<<<8, 256>>>(g[180], g[176]);
  CHECK(hipGetLastError());
  // layer6.q_rope: rope -> (8, 2048)
  rope_backward_q<<<32, 256>>>(g[179], g[174]);
  CHECK(hipGetLastError());
  // layer6.v: linear -> (8, 512)
  linear_backward_input_kv<<<dim3(128, 1), dim3(16, 16)>>>(g[178], v[177], g[171]);
  CHECK(hipGetLastError());
  linear_backward_weight_kv<<<dim3(128, 32), dim3(16, 16)>>>(g[178], v[171], g[177]);
  CHECK(hipGetLastError());
  // layer6.k: linear -> (8, 512)
  linear_backward_input_kv<<<dim3(128, 1), dim3(16, 16)>>>(g[176], v[175], g[171]);
  CHECK(hipGetLastError());
  linear_backward_weight_kv<<<dim3(128, 32), dim3(16, 16)>>>(g[176], v[171], g[175]);
  CHECK(hipGetLastError());
  // layer6.q: linear -> (8, 2048)
  linear_backward_input_hidden<<<dim3(128, 1), dim3(16, 16)>>>(g[174], v[173], g[171]);
  CHECK(hipGetLastError());
  linear_backward_weight_hidden<<<dim3(128, 128), dim3(16, 16)>>>(g[174], v[171], g[173]);
  CHECK(hipGetLastError());
  // layer6.attention_norm: rmsnorm -> (8, 2048)
  rmsnorm_backward_input<<<8, 256>>>(v[169], v[170], v[172], g[171], g[169]);
  CHECK(hipGetLastError());
  rmsnorm_backward_weight<<<8, 256>>>(v[169], v[172], g[171], g[170]);
  CHECK(hipGetLastError());
  // layer5.ffn_residual: add -> (8, 2048)
  add_backward<<<64, 256>>>(g[169], g[158], g[168]);
  CHECK(hipGetLastError());
  // layer5.down: linear -> (8, 2048)
  linear_backward_input_mlp_down<<<dim3(512, 1), dim3(16, 16)>>>(g[168], v[167], g[166]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_down<<<dim3(512, 128), dim3(16, 16)>>>(g[168], v[166], g[167]);
  CHECK(hipGetLastError());
  // layer5.swiglu: swiglu -> (8, 8192)
  swiglu_backward<<<256, 256>>>(v[163], v[165], g[166], g[163], g[165]);
  CHECK(hipGetLastError());
  // layer5.up: linear -> (8, 8192)
  linear_backward_input_mlp_up<<<dim3(128, 1), dim3(16, 16)>>>(g[165], v[164], g[160]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_up<<<dim3(128, 512), dim3(16, 16)>>>(g[165], v[160], g[164]);
  CHECK(hipGetLastError());
  // layer5.gate: linear -> (8, 8192)
  linear_backward_input_mlp_up<<<dim3(128, 1), dim3(16, 16)>>>(g[163], v[162], g[160]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_up<<<dim3(128, 512), dim3(16, 16)>>>(g[163], v[160], g[162]);
  CHECK(hipGetLastError());
  // layer5.ffn_norm: rmsnorm -> (8, 2048)
  rmsnorm_backward_input<<<8, 256>>>(v[158], v[159], v[161], g[160], g[158]);
  CHECK(hipGetLastError());
  rmsnorm_backward_weight<<<8, 256>>>(v[158], v[161], g[160], g[159]);
  CHECK(hipGetLastError());
  // layer5.attention_residual: add -> (8, 2048)
  add_backward<<<64, 256>>>(g[158], g[141], g[157]);
  CHECK(hipGetLastError());
  // layer5.o: linear -> (8, 2048)
  linear_backward_input_hidden<<<dim3(128, 1), dim3(16, 16)>>>(g[157], v[156], g[155]);
  CHECK(hipGetLastError());
  linear_backward_weight_hidden<<<dim3(128, 128), dim3(16, 16)>>>(g[157], v[155], g[156]);
  CHECK(hipGetLastError());
  // layer5.context: values -> (8, 2048)
  attention_values_backward_p<<<8, 256>>>(g[155], v[150], g[154]);
  CHECK(hipGetLastError());
  attention_values_backward_v<<<16, 256>>>(g[155], v[154], g[150]);
  CHECK(hipGetLastError());
  // layer5.probabilities: softmax -> (1, 32, 8, 8)
  softmax_backward<<<256, 256>>>(v[154], g[154], g[153]);
  CHECK(hipGetLastError());
  // layer5.scores: scores -> (1, 32, 8, 8)
  attention_scores_backward_q<<<64, 256>>>(g[153], v[152], g[151]);
  CHECK(hipGetLastError());
  attention_scores_backward_k<<<16, 256>>>(g[153], v[151], g[152]);
  CHECK(hipGetLastError());
  // layer5.k_rope: rope -> (8, 512)
  rope_backward_k<<<8, 256>>>(g[152], g[148]);
  CHECK(hipGetLastError());
  // layer5.q_rope: rope -> (8, 2048)
  rope_backward_q<<<32, 256>>>(g[151], g[146]);
  CHECK(hipGetLastError());
  // layer5.v: linear -> (8, 512)
  linear_backward_input_kv<<<dim3(128, 1), dim3(16, 16)>>>(g[150], v[149], g[143]);
  CHECK(hipGetLastError());
  linear_backward_weight_kv<<<dim3(128, 32), dim3(16, 16)>>>(g[150], v[143], g[149]);
  CHECK(hipGetLastError());
  // layer5.k: linear -> (8, 512)
  linear_backward_input_kv<<<dim3(128, 1), dim3(16, 16)>>>(g[148], v[147], g[143]);
  CHECK(hipGetLastError());
  linear_backward_weight_kv<<<dim3(128, 32), dim3(16, 16)>>>(g[148], v[143], g[147]);
  CHECK(hipGetLastError());
  // layer5.q: linear -> (8, 2048)
  linear_backward_input_hidden<<<dim3(128, 1), dim3(16, 16)>>>(g[146], v[145], g[143]);
  CHECK(hipGetLastError());
  linear_backward_weight_hidden<<<dim3(128, 128), dim3(16, 16)>>>(g[146], v[143], g[145]);
  CHECK(hipGetLastError());
  // layer5.attention_norm: rmsnorm -> (8, 2048)
  rmsnorm_backward_input<<<8, 256>>>(v[141], v[142], v[144], g[143], g[141]);
  CHECK(hipGetLastError());
  rmsnorm_backward_weight<<<8, 256>>>(v[141], v[144], g[143], g[142]);
  CHECK(hipGetLastError());
  // layer4.ffn_residual: add -> (8, 2048)
  add_backward<<<64, 256>>>(g[141], g[130], g[140]);
  CHECK(hipGetLastError());
  // layer4.down: linear -> (8, 2048)
  linear_backward_input_mlp_down<<<dim3(512, 1), dim3(16, 16)>>>(g[140], v[139], g[138]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_down<<<dim3(512, 128), dim3(16, 16)>>>(g[140], v[138], g[139]);
  CHECK(hipGetLastError());
  // layer4.swiglu: swiglu -> (8, 8192)
  swiglu_backward<<<256, 256>>>(v[135], v[137], g[138], g[135], g[137]);
  CHECK(hipGetLastError());
  // layer4.up: linear -> (8, 8192)
  linear_backward_input_mlp_up<<<dim3(128, 1), dim3(16, 16)>>>(g[137], v[136], g[132]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_up<<<dim3(128, 512), dim3(16, 16)>>>(g[137], v[132], g[136]);
  CHECK(hipGetLastError());
  // layer4.gate: linear -> (8, 8192)
  linear_backward_input_mlp_up<<<dim3(128, 1), dim3(16, 16)>>>(g[135], v[134], g[132]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_up<<<dim3(128, 512), dim3(16, 16)>>>(g[135], v[132], g[134]);
  CHECK(hipGetLastError());
  // layer4.ffn_norm: rmsnorm -> (8, 2048)
  rmsnorm_backward_input<<<8, 256>>>(v[130], v[131], v[133], g[132], g[130]);
  CHECK(hipGetLastError());
  rmsnorm_backward_weight<<<8, 256>>>(v[130], v[133], g[132], g[131]);
  CHECK(hipGetLastError());
  // layer4.attention_residual: add -> (8, 2048)
  add_backward<<<64, 256>>>(g[130], g[113], g[129]);
  CHECK(hipGetLastError());
  // layer4.o: linear -> (8, 2048)
  linear_backward_input_hidden<<<dim3(128, 1), dim3(16, 16)>>>(g[129], v[128], g[127]);
  CHECK(hipGetLastError());
  linear_backward_weight_hidden<<<dim3(128, 128), dim3(16, 16)>>>(g[129], v[127], g[128]);
  CHECK(hipGetLastError());
  // layer4.context: values -> (8, 2048)
  attention_values_backward_p<<<8, 256>>>(g[127], v[122], g[126]);
  CHECK(hipGetLastError());
  attention_values_backward_v<<<16, 256>>>(g[127], v[126], g[122]);
  CHECK(hipGetLastError());
  // layer4.probabilities: softmax -> (1, 32, 8, 8)
  softmax_backward<<<256, 256>>>(v[126], g[126], g[125]);
  CHECK(hipGetLastError());
  // layer4.scores: scores -> (1, 32, 8, 8)
  attention_scores_backward_q<<<64, 256>>>(g[125], v[124], g[123]);
  CHECK(hipGetLastError());
  attention_scores_backward_k<<<16, 256>>>(g[125], v[123], g[124]);
  CHECK(hipGetLastError());
  // layer4.k_rope: rope -> (8, 512)
  rope_backward_k<<<8, 256>>>(g[124], g[120]);
  CHECK(hipGetLastError());
  // layer4.q_rope: rope -> (8, 2048)
  rope_backward_q<<<32, 256>>>(g[123], g[118]);
  CHECK(hipGetLastError());
  // layer4.v: linear -> (8, 512)
  linear_backward_input_kv<<<dim3(128, 1), dim3(16, 16)>>>(g[122], v[121], g[115]);
  CHECK(hipGetLastError());
  linear_backward_weight_kv<<<dim3(128, 32), dim3(16, 16)>>>(g[122], v[115], g[121]);
  CHECK(hipGetLastError());
  // layer4.k: linear -> (8, 512)
  linear_backward_input_kv<<<dim3(128, 1), dim3(16, 16)>>>(g[120], v[119], g[115]);
  CHECK(hipGetLastError());
  linear_backward_weight_kv<<<dim3(128, 32), dim3(16, 16)>>>(g[120], v[115], g[119]);
  CHECK(hipGetLastError());
  // layer4.q: linear -> (8, 2048)
  linear_backward_input_hidden<<<dim3(128, 1), dim3(16, 16)>>>(g[118], v[117], g[115]);
  CHECK(hipGetLastError());
  linear_backward_weight_hidden<<<dim3(128, 128), dim3(16, 16)>>>(g[118], v[115], g[117]);
  CHECK(hipGetLastError());
  // layer4.attention_norm: rmsnorm -> (8, 2048)
  rmsnorm_backward_input<<<8, 256>>>(v[113], v[114], v[116], g[115], g[113]);
  CHECK(hipGetLastError());
  rmsnorm_backward_weight<<<8, 256>>>(v[113], v[116], g[115], g[114]);
  CHECK(hipGetLastError());
  // layer3.ffn_residual: add -> (8, 2048)
  add_backward<<<64, 256>>>(g[113], g[102], g[112]);
  CHECK(hipGetLastError());
  // layer3.down: linear -> (8, 2048)
  linear_backward_input_mlp_down<<<dim3(512, 1), dim3(16, 16)>>>(g[112], v[111], g[110]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_down<<<dim3(512, 128), dim3(16, 16)>>>(g[112], v[110], g[111]);
  CHECK(hipGetLastError());
  // layer3.swiglu: swiglu -> (8, 8192)
  swiglu_backward<<<256, 256>>>(v[107], v[109], g[110], g[107], g[109]);
  CHECK(hipGetLastError());
  // layer3.up: linear -> (8, 8192)
  linear_backward_input_mlp_up<<<dim3(128, 1), dim3(16, 16)>>>(g[109], v[108], g[104]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_up<<<dim3(128, 512), dim3(16, 16)>>>(g[109], v[104], g[108]);
  CHECK(hipGetLastError());
  // layer3.gate: linear -> (8, 8192)
  linear_backward_input_mlp_up<<<dim3(128, 1), dim3(16, 16)>>>(g[107], v[106], g[104]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_up<<<dim3(128, 512), dim3(16, 16)>>>(g[107], v[104], g[106]);
  CHECK(hipGetLastError());
  // layer3.ffn_norm: rmsnorm -> (8, 2048)
  rmsnorm_backward_input<<<8, 256>>>(v[102], v[103], v[105], g[104], g[102]);
  CHECK(hipGetLastError());
  rmsnorm_backward_weight<<<8, 256>>>(v[102], v[105], g[104], g[103]);
  CHECK(hipGetLastError());
  // layer3.attention_residual: add -> (8, 2048)
  add_backward<<<64, 256>>>(g[102], g[85], g[101]);
  CHECK(hipGetLastError());
  // layer3.o: linear -> (8, 2048)
  linear_backward_input_hidden<<<dim3(128, 1), dim3(16, 16)>>>(g[101], v[100], g[99]);
  CHECK(hipGetLastError());
  linear_backward_weight_hidden<<<dim3(128, 128), dim3(16, 16)>>>(g[101], v[99], g[100]);
  CHECK(hipGetLastError());
  // layer3.context: values -> (8, 2048)
  attention_values_backward_p<<<8, 256>>>(g[99], v[94], g[98]);
  CHECK(hipGetLastError());
  attention_values_backward_v<<<16, 256>>>(g[99], v[98], g[94]);
  CHECK(hipGetLastError());
  // layer3.probabilities: softmax -> (1, 32, 8, 8)
  softmax_backward<<<256, 256>>>(v[98], g[98], g[97]);
  CHECK(hipGetLastError());
  // layer3.scores: scores -> (1, 32, 8, 8)
  attention_scores_backward_q<<<64, 256>>>(g[97], v[96], g[95]);
  CHECK(hipGetLastError());
  attention_scores_backward_k<<<16, 256>>>(g[97], v[95], g[96]);
  CHECK(hipGetLastError());
  // layer3.k_rope: rope -> (8, 512)
  rope_backward_k<<<8, 256>>>(g[96], g[92]);
  CHECK(hipGetLastError());
  // layer3.q_rope: rope -> (8, 2048)
  rope_backward_q<<<32, 256>>>(g[95], g[90]);
  CHECK(hipGetLastError());
  // layer3.v: linear -> (8, 512)
  linear_backward_input_kv<<<dim3(128, 1), dim3(16, 16)>>>(g[94], v[93], g[87]);
  CHECK(hipGetLastError());
  linear_backward_weight_kv<<<dim3(128, 32), dim3(16, 16)>>>(g[94], v[87], g[93]);
  CHECK(hipGetLastError());
  // layer3.k: linear -> (8, 512)
  linear_backward_input_kv<<<dim3(128, 1), dim3(16, 16)>>>(g[92], v[91], g[87]);
  CHECK(hipGetLastError());
  linear_backward_weight_kv<<<dim3(128, 32), dim3(16, 16)>>>(g[92], v[87], g[91]);
  CHECK(hipGetLastError());
  // layer3.q: linear -> (8, 2048)
  linear_backward_input_hidden<<<dim3(128, 1), dim3(16, 16)>>>(g[90], v[89], g[87]);
  CHECK(hipGetLastError());
  linear_backward_weight_hidden<<<dim3(128, 128), dim3(16, 16)>>>(g[90], v[87], g[89]);
  CHECK(hipGetLastError());
  // layer3.attention_norm: rmsnorm -> (8, 2048)
  rmsnorm_backward_input<<<8, 256>>>(v[85], v[86], v[88], g[87], g[85]);
  CHECK(hipGetLastError());
  rmsnorm_backward_weight<<<8, 256>>>(v[85], v[88], g[87], g[86]);
  CHECK(hipGetLastError());
  // layer2.ffn_residual: add -> (8, 2048)
  add_backward<<<64, 256>>>(g[85], g[74], g[84]);
  CHECK(hipGetLastError());
  // layer2.down: linear -> (8, 2048)
  linear_backward_input_mlp_down<<<dim3(512, 1), dim3(16, 16)>>>(g[84], v[83], g[82]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_down<<<dim3(512, 128), dim3(16, 16)>>>(g[84], v[82], g[83]);
  CHECK(hipGetLastError());
  // layer2.swiglu: swiglu -> (8, 8192)
  swiglu_backward<<<256, 256>>>(v[79], v[81], g[82], g[79], g[81]);
  CHECK(hipGetLastError());
  // layer2.up: linear -> (8, 8192)
  linear_backward_input_mlp_up<<<dim3(128, 1), dim3(16, 16)>>>(g[81], v[80], g[76]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_up<<<dim3(128, 512), dim3(16, 16)>>>(g[81], v[76], g[80]);
  CHECK(hipGetLastError());
  // layer2.gate: linear -> (8, 8192)
  linear_backward_input_mlp_up<<<dim3(128, 1), dim3(16, 16)>>>(g[79], v[78], g[76]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_up<<<dim3(128, 512), dim3(16, 16)>>>(g[79], v[76], g[78]);
  CHECK(hipGetLastError());
  // layer2.ffn_norm: rmsnorm -> (8, 2048)
  rmsnorm_backward_input<<<8, 256>>>(v[74], v[75], v[77], g[76], g[74]);
  CHECK(hipGetLastError());
  rmsnorm_backward_weight<<<8, 256>>>(v[74], v[77], g[76], g[75]);
  CHECK(hipGetLastError());
  // layer2.attention_residual: add -> (8, 2048)
  add_backward<<<64, 256>>>(g[74], g[57], g[73]);
  CHECK(hipGetLastError());
  // layer2.o: linear -> (8, 2048)
  linear_backward_input_hidden<<<dim3(128, 1), dim3(16, 16)>>>(g[73], v[72], g[71]);
  CHECK(hipGetLastError());
  linear_backward_weight_hidden<<<dim3(128, 128), dim3(16, 16)>>>(g[73], v[71], g[72]);
  CHECK(hipGetLastError());
  // layer2.context: values -> (8, 2048)
  attention_values_backward_p<<<8, 256>>>(g[71], v[66], g[70]);
  CHECK(hipGetLastError());
  attention_values_backward_v<<<16, 256>>>(g[71], v[70], g[66]);
  CHECK(hipGetLastError());
  // layer2.probabilities: softmax -> (1, 32, 8, 8)
  softmax_backward<<<256, 256>>>(v[70], g[70], g[69]);
  CHECK(hipGetLastError());
  // layer2.scores: scores -> (1, 32, 8, 8)
  attention_scores_backward_q<<<64, 256>>>(g[69], v[68], g[67]);
  CHECK(hipGetLastError());
  attention_scores_backward_k<<<16, 256>>>(g[69], v[67], g[68]);
  CHECK(hipGetLastError());
  // layer2.k_rope: rope -> (8, 512)
  rope_backward_k<<<8, 256>>>(g[68], g[64]);
  CHECK(hipGetLastError());
  // layer2.q_rope: rope -> (8, 2048)
  rope_backward_q<<<32, 256>>>(g[67], g[62]);
  CHECK(hipGetLastError());
  // layer2.v: linear -> (8, 512)
  linear_backward_input_kv<<<dim3(128, 1), dim3(16, 16)>>>(g[66], v[65], g[59]);
  CHECK(hipGetLastError());
  linear_backward_weight_kv<<<dim3(128, 32), dim3(16, 16)>>>(g[66], v[59], g[65]);
  CHECK(hipGetLastError());
  // layer2.k: linear -> (8, 512)
  linear_backward_input_kv<<<dim3(128, 1), dim3(16, 16)>>>(g[64], v[63], g[59]);
  CHECK(hipGetLastError());
  linear_backward_weight_kv<<<dim3(128, 32), dim3(16, 16)>>>(g[64], v[59], g[63]);
  CHECK(hipGetLastError());
  // layer2.q: linear -> (8, 2048)
  linear_backward_input_hidden<<<dim3(128, 1), dim3(16, 16)>>>(g[62], v[61], g[59]);
  CHECK(hipGetLastError());
  linear_backward_weight_hidden<<<dim3(128, 128), dim3(16, 16)>>>(g[62], v[59], g[61]);
  CHECK(hipGetLastError());
  // layer2.attention_norm: rmsnorm -> (8, 2048)
  rmsnorm_backward_input<<<8, 256>>>(v[57], v[58], v[60], g[59], g[57]);
  CHECK(hipGetLastError());
  rmsnorm_backward_weight<<<8, 256>>>(v[57], v[60], g[59], g[58]);
  CHECK(hipGetLastError());
  // layer1.ffn_residual: add -> (8, 2048)
  add_backward<<<64, 256>>>(g[57], g[46], g[56]);
  CHECK(hipGetLastError());
  // layer1.down: linear -> (8, 2048)
  linear_backward_input_mlp_down<<<dim3(512, 1), dim3(16, 16)>>>(g[56], v[55], g[54]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_down<<<dim3(512, 128), dim3(16, 16)>>>(g[56], v[54], g[55]);
  CHECK(hipGetLastError());
  // layer1.swiglu: swiglu -> (8, 8192)
  swiglu_backward<<<256, 256>>>(v[51], v[53], g[54], g[51], g[53]);
  CHECK(hipGetLastError());
  // layer1.up: linear -> (8, 8192)
  linear_backward_input_mlp_up<<<dim3(128, 1), dim3(16, 16)>>>(g[53], v[52], g[48]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_up<<<dim3(128, 512), dim3(16, 16)>>>(g[53], v[48], g[52]);
  CHECK(hipGetLastError());
  // layer1.gate: linear -> (8, 8192)
  linear_backward_input_mlp_up<<<dim3(128, 1), dim3(16, 16)>>>(g[51], v[50], g[48]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_up<<<dim3(128, 512), dim3(16, 16)>>>(g[51], v[48], g[50]);
  CHECK(hipGetLastError());
  // layer1.ffn_norm: rmsnorm -> (8, 2048)
  rmsnorm_backward_input<<<8, 256>>>(v[46], v[47], v[49], g[48], g[46]);
  CHECK(hipGetLastError());
  rmsnorm_backward_weight<<<8, 256>>>(v[46], v[49], g[48], g[47]);
  CHECK(hipGetLastError());
  // layer1.attention_residual: add -> (8, 2048)
  add_backward<<<64, 256>>>(g[46], g[29], g[45]);
  CHECK(hipGetLastError());
  // layer1.o: linear -> (8, 2048)
  linear_backward_input_hidden<<<dim3(128, 1), dim3(16, 16)>>>(g[45], v[44], g[43]);
  CHECK(hipGetLastError());
  linear_backward_weight_hidden<<<dim3(128, 128), dim3(16, 16)>>>(g[45], v[43], g[44]);
  CHECK(hipGetLastError());
  // layer1.context: values -> (8, 2048)
  attention_values_backward_p<<<8, 256>>>(g[43], v[38], g[42]);
  CHECK(hipGetLastError());
  attention_values_backward_v<<<16, 256>>>(g[43], v[42], g[38]);
  CHECK(hipGetLastError());
  // layer1.probabilities: softmax -> (1, 32, 8, 8)
  softmax_backward<<<256, 256>>>(v[42], g[42], g[41]);
  CHECK(hipGetLastError());
  // layer1.scores: scores -> (1, 32, 8, 8)
  attention_scores_backward_q<<<64, 256>>>(g[41], v[40], g[39]);
  CHECK(hipGetLastError());
  attention_scores_backward_k<<<16, 256>>>(g[41], v[39], g[40]);
  CHECK(hipGetLastError());
  // layer1.k_rope: rope -> (8, 512)
  rope_backward_k<<<8, 256>>>(g[40], g[36]);
  CHECK(hipGetLastError());
  // layer1.q_rope: rope -> (8, 2048)
  rope_backward_q<<<32, 256>>>(g[39], g[34]);
  CHECK(hipGetLastError());
  // layer1.v: linear -> (8, 512)
  linear_backward_input_kv<<<dim3(128, 1), dim3(16, 16)>>>(g[38], v[37], g[31]);
  CHECK(hipGetLastError());
  linear_backward_weight_kv<<<dim3(128, 32), dim3(16, 16)>>>(g[38], v[31], g[37]);
  CHECK(hipGetLastError());
  // layer1.k: linear -> (8, 512)
  linear_backward_input_kv<<<dim3(128, 1), dim3(16, 16)>>>(g[36], v[35], g[31]);
  CHECK(hipGetLastError());
  linear_backward_weight_kv<<<dim3(128, 32), dim3(16, 16)>>>(g[36], v[31], g[35]);
  CHECK(hipGetLastError());
  // layer1.q: linear -> (8, 2048)
  linear_backward_input_hidden<<<dim3(128, 1), dim3(16, 16)>>>(g[34], v[33], g[31]);
  CHECK(hipGetLastError());
  linear_backward_weight_hidden<<<dim3(128, 128), dim3(16, 16)>>>(g[34], v[31], g[33]);
  CHECK(hipGetLastError());
  // layer1.attention_norm: rmsnorm -> (8, 2048)
  rmsnorm_backward_input<<<8, 256>>>(v[29], v[30], v[32], g[31], g[29]);
  CHECK(hipGetLastError());
  rmsnorm_backward_weight<<<8, 256>>>(v[29], v[32], g[31], g[30]);
  CHECK(hipGetLastError());
  // layer0.ffn_residual: add -> (8, 2048)
  add_backward<<<64, 256>>>(g[29], g[18], g[28]);
  CHECK(hipGetLastError());
  // layer0.down: linear -> (8, 2048)
  linear_backward_input_mlp_down<<<dim3(512, 1), dim3(16, 16)>>>(g[28], v[27], g[26]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_down<<<dim3(512, 128), dim3(16, 16)>>>(g[28], v[26], g[27]);
  CHECK(hipGetLastError());
  // layer0.swiglu: swiglu -> (8, 8192)
  swiglu_backward<<<256, 256>>>(v[23], v[25], g[26], g[23], g[25]);
  CHECK(hipGetLastError());
  // layer0.up: linear -> (8, 8192)
  linear_backward_input_mlp_up<<<dim3(128, 1), dim3(16, 16)>>>(g[25], v[24], g[20]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_up<<<dim3(128, 512), dim3(16, 16)>>>(g[25], v[20], g[24]);
  CHECK(hipGetLastError());
  // layer0.gate: linear -> (8, 8192)
  linear_backward_input_mlp_up<<<dim3(128, 1), dim3(16, 16)>>>(g[23], v[22], g[20]);
  CHECK(hipGetLastError());
  linear_backward_weight_mlp_up<<<dim3(128, 512), dim3(16, 16)>>>(g[23], v[20], g[22]);
  CHECK(hipGetLastError());
  // layer0.ffn_norm: rmsnorm -> (8, 2048)
  rmsnorm_backward_input<<<8, 256>>>(v[18], v[19], v[21], g[20], g[18]);
  CHECK(hipGetLastError());
  rmsnorm_backward_weight<<<8, 256>>>(v[18], v[21], g[20], g[19]);
  CHECK(hipGetLastError());
  // layer0.attention_residual: add -> (8, 2048)
  add_backward<<<64, 256>>>(g[18], g[1], g[17]);
  CHECK(hipGetLastError());
  // layer0.o: linear -> (8, 2048)
  linear_backward_input_hidden<<<dim3(128, 1), dim3(16, 16)>>>(g[17], v[16], g[15]);
  CHECK(hipGetLastError());
  linear_backward_weight_hidden<<<dim3(128, 128), dim3(16, 16)>>>(g[17], v[15], g[16]);
  CHECK(hipGetLastError());
  // layer0.context: values -> (8, 2048)
  attention_values_backward_p<<<8, 256>>>(g[15], v[10], g[14]);
  CHECK(hipGetLastError());
  attention_values_backward_v<<<16, 256>>>(g[15], v[14], g[10]);
  CHECK(hipGetLastError());
  // layer0.probabilities: softmax -> (1, 32, 8, 8)
  softmax_backward<<<256, 256>>>(v[14], g[14], g[13]);
  CHECK(hipGetLastError());
  // layer0.scores: scores -> (1, 32, 8, 8)
  attention_scores_backward_q<<<64, 256>>>(g[13], v[12], g[11]);
  CHECK(hipGetLastError());
  attention_scores_backward_k<<<16, 256>>>(g[13], v[11], g[12]);
  CHECK(hipGetLastError());
  // layer0.k_rope: rope -> (8, 512)
  rope_backward_k<<<8, 256>>>(g[12], g[8]);
  CHECK(hipGetLastError());
  // layer0.q_rope: rope -> (8, 2048)
  rope_backward_q<<<32, 256>>>(g[11], g[6]);
  CHECK(hipGetLastError());
  // layer0.v: linear -> (8, 512)
  linear_backward_input_kv<<<dim3(128, 1), dim3(16, 16)>>>(g[10], v[9], g[3]);
  CHECK(hipGetLastError());
  linear_backward_weight_kv<<<dim3(128, 32), dim3(16, 16)>>>(g[10], v[3], g[9]);
  CHECK(hipGetLastError());
  // layer0.k: linear -> (8, 512)
  linear_backward_input_kv<<<dim3(128, 1), dim3(16, 16)>>>(g[8], v[7], g[3]);
  CHECK(hipGetLastError());
  linear_backward_weight_kv<<<dim3(128, 32), dim3(16, 16)>>>(g[8], v[3], g[7]);
  CHECK(hipGetLastError());
  // layer0.q: linear -> (8, 2048)
  linear_backward_input_hidden<<<dim3(128, 1), dim3(16, 16)>>>(g[6], v[5], g[3]);
  CHECK(hipGetLastError());
  linear_backward_weight_hidden<<<dim3(128, 128), dim3(16, 16)>>>(g[6], v[3], g[5]);
  CHECK(hipGetLastError());
  // layer0.attention_norm: rmsnorm -> (8, 2048)
  rmsnorm_backward_input<<<8, 256>>>(v[1], v[2], v[4], g[3], g[1]);
  CHECK(hipGetLastError());
  rmsnorm_backward_weight<<<8, 256>>>(v[1], v[4], g[3], g[2]);
  CHECK(hipGetLastError());
  // embedding: embedding -> (8, 2048)
  embedding_backward<<<64, 256>>>(g[1], ids, g[0]);
  CHECK(hipGetLastError());
  return hipDeviceSynchronize();
}
