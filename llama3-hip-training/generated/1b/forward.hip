// GENERATED FORWARD execution order. Do not edit; run train.py emit.
// v[] = saved forward values; g[] = accumulated gradients.
extern "C" int run_forward(float** v, float** g, const int* ids, const int* targets) {
  // embedding: embedding -> (8, 2048)
  embedding_forward<<<64, 256>>>(v[0], ids, v[1]);
  CHECK(hipGetLastError());
  // layer0.attention_norm: rmsnorm -> (8, 2048)
  rmsnorm_forward<<<8, 256>>>(v[1], v[2], v[3], v[4]);
  CHECK(hipGetLastError());
  // layer0.q: linear -> (8, 2048)
  linear_forward_hidden<<<dim3(128, 1), dim3(16, 16)>>>(v[3], v[5], v[6]);
  CHECK(hipGetLastError());
  // layer0.k: linear -> (8, 512)
  linear_forward_kv<<<dim3(32, 1), dim3(16, 16)>>>(v[3], v[7], v[8]);
  CHECK(hipGetLastError());
  // layer0.v: linear -> (8, 512)
  linear_forward_kv<<<dim3(32, 1), dim3(16, 16)>>>(v[3], v[9], v[10]);
  CHECK(hipGetLastError());
  // layer0.q_rope: rope -> (8, 2048)
  rope_forward_q<<<32, 256>>>(v[6], v[11]);
  CHECK(hipGetLastError());
  // layer0.k_rope: rope -> (8, 512)
  rope_forward_k<<<8, 256>>>(v[8], v[12]);
  CHECK(hipGetLastError());
  // layer0.scores: scores -> (1, 32, 8, 8)
  attention_scores_forward<<<8, 256>>>(v[11], v[12], v[13]);
  CHECK(hipGetLastError());
  // layer0.probabilities: softmax -> (1, 32, 8, 8)
  softmax_forward<<<256, 256>>>(v[13], v[14]);
  CHECK(hipGetLastError());
  // layer0.context: values -> (8, 2048)
  attention_values_forward<<<64, 256>>>(v[14], v[10], v[15]);
  CHECK(hipGetLastError());
  // layer0.o: linear -> (8, 2048)
  linear_forward_hidden<<<dim3(128, 1), dim3(16, 16)>>>(v[15], v[16], v[17]);
  CHECK(hipGetLastError());
  // layer0.attention_residual: add -> (8, 2048)
  add_forward<<<64, 256>>>(v[1], v[17], v[18]);
  CHECK(hipGetLastError());
  // layer0.ffn_norm: rmsnorm -> (8, 2048)
  rmsnorm_forward<<<8, 256>>>(v[18], v[19], v[20], v[21]);
  CHECK(hipGetLastError());
  // layer0.gate: linear -> (8, 8192)
  linear_forward_mlp_up<<<dim3(512, 1), dim3(16, 16)>>>(v[20], v[22], v[23]);
  CHECK(hipGetLastError());
  // layer0.up: linear -> (8, 8192)
  linear_forward_mlp_up<<<dim3(512, 1), dim3(16, 16)>>>(v[20], v[24], v[25]);
  CHECK(hipGetLastError());
  // layer0.swiglu: swiglu -> (8, 8192)
  swiglu_forward<<<256, 256>>>(v[23], v[25], v[26]);
  CHECK(hipGetLastError());
  // layer0.down: linear -> (8, 2048)
  linear_forward_mlp_down<<<dim3(128, 1), dim3(16, 16)>>>(v[26], v[27], v[28]);
  CHECK(hipGetLastError());
  // layer0.ffn_residual: add -> (8, 2048)
  add_forward<<<64, 256>>>(v[18], v[28], v[29]);
  CHECK(hipGetLastError());
  // layer1.attention_norm: rmsnorm -> (8, 2048)
  rmsnorm_forward<<<8, 256>>>(v[29], v[30], v[31], v[32]);
  CHECK(hipGetLastError());
  // layer1.q: linear -> (8, 2048)
  linear_forward_hidden<<<dim3(128, 1), dim3(16, 16)>>>(v[31], v[33], v[34]);
  CHECK(hipGetLastError());
  // layer1.k: linear -> (8, 512)
  linear_forward_kv<<<dim3(32, 1), dim3(16, 16)>>>(v[31], v[35], v[36]);
  CHECK(hipGetLastError());
  // layer1.v: linear -> (8, 512)
  linear_forward_kv<<<dim3(32, 1), dim3(16, 16)>>>(v[31], v[37], v[38]);
  CHECK(hipGetLastError());
  // layer1.q_rope: rope -> (8, 2048)
  rope_forward_q<<<32, 256>>>(v[34], v[39]);
  CHECK(hipGetLastError());
  // layer1.k_rope: rope -> (8, 512)
  rope_forward_k<<<8, 256>>>(v[36], v[40]);
  CHECK(hipGetLastError());
  // layer1.scores: scores -> (1, 32, 8, 8)
  attention_scores_forward<<<8, 256>>>(v[39], v[40], v[41]);
  CHECK(hipGetLastError());
  // layer1.probabilities: softmax -> (1, 32, 8, 8)
  softmax_forward<<<256, 256>>>(v[41], v[42]);
  CHECK(hipGetLastError());
  // layer1.context: values -> (8, 2048)
  attention_values_forward<<<64, 256>>>(v[42], v[38], v[43]);
  CHECK(hipGetLastError());
  // layer1.o: linear -> (8, 2048)
  linear_forward_hidden<<<dim3(128, 1), dim3(16, 16)>>>(v[43], v[44], v[45]);
  CHECK(hipGetLastError());
  // layer1.attention_residual: add -> (8, 2048)
  add_forward<<<64, 256>>>(v[29], v[45], v[46]);
  CHECK(hipGetLastError());
  // layer1.ffn_norm: rmsnorm -> (8, 2048)
  rmsnorm_forward<<<8, 256>>>(v[46], v[47], v[48], v[49]);
  CHECK(hipGetLastError());
  // layer1.gate: linear -> (8, 8192)
  linear_forward_mlp_up<<<dim3(512, 1), dim3(16, 16)>>>(v[48], v[50], v[51]);
  CHECK(hipGetLastError());
  // layer1.up: linear -> (8, 8192)
  linear_forward_mlp_up<<<dim3(512, 1), dim3(16, 16)>>>(v[48], v[52], v[53]);
  CHECK(hipGetLastError());
  // layer1.swiglu: swiglu -> (8, 8192)
  swiglu_forward<<<256, 256>>>(v[51], v[53], v[54]);
  CHECK(hipGetLastError());
  // layer1.down: linear -> (8, 2048)
  linear_forward_mlp_down<<<dim3(128, 1), dim3(16, 16)>>>(v[54], v[55], v[56]);
  CHECK(hipGetLastError());
  // layer1.ffn_residual: add -> (8, 2048)
  add_forward<<<64, 256>>>(v[46], v[56], v[57]);
  CHECK(hipGetLastError());
  // layer2.attention_norm: rmsnorm -> (8, 2048)
  rmsnorm_forward<<<8, 256>>>(v[57], v[58], v[59], v[60]);
  CHECK(hipGetLastError());
  // layer2.q: linear -> (8, 2048)
  linear_forward_hidden<<<dim3(128, 1), dim3(16, 16)>>>(v[59], v[61], v[62]);
  CHECK(hipGetLastError());
  // layer2.k: linear -> (8, 512)
  linear_forward_kv<<<dim3(32, 1), dim3(16, 16)>>>(v[59], v[63], v[64]);
  CHECK(hipGetLastError());
  // layer2.v: linear -> (8, 512)
  linear_forward_kv<<<dim3(32, 1), dim3(16, 16)>>>(v[59], v[65], v[66]);
  CHECK(hipGetLastError());
  // layer2.q_rope: rope -> (8, 2048)
  rope_forward_q<<<32, 256>>>(v[62], v[67]);
  CHECK(hipGetLastError());
  // layer2.k_rope: rope -> (8, 512)
  rope_forward_k<<<8, 256>>>(v[64], v[68]);
  CHECK(hipGetLastError());
  // layer2.scores: scores -> (1, 32, 8, 8)
  attention_scores_forward<<<8, 256>>>(v[67], v[68], v[69]);
  CHECK(hipGetLastError());
  // layer2.probabilities: softmax -> (1, 32, 8, 8)
  softmax_forward<<<256, 256>>>(v[69], v[70]);
  CHECK(hipGetLastError());
  // layer2.context: values -> (8, 2048)
  attention_values_forward<<<64, 256>>>(v[70], v[66], v[71]);
  CHECK(hipGetLastError());
  // layer2.o: linear -> (8, 2048)
  linear_forward_hidden<<<dim3(128, 1), dim3(16, 16)>>>(v[71], v[72], v[73]);
  CHECK(hipGetLastError());
  // layer2.attention_residual: add -> (8, 2048)
  add_forward<<<64, 256>>>(v[57], v[73], v[74]);
  CHECK(hipGetLastError());
  // layer2.ffn_norm: rmsnorm -> (8, 2048)
  rmsnorm_forward<<<8, 256>>>(v[74], v[75], v[76], v[77]);
  CHECK(hipGetLastError());
  // layer2.gate: linear -> (8, 8192)
  linear_forward_mlp_up<<<dim3(512, 1), dim3(16, 16)>>>(v[76], v[78], v[79]);
  CHECK(hipGetLastError());
  // layer2.up: linear -> (8, 8192)
  linear_forward_mlp_up<<<dim3(512, 1), dim3(16, 16)>>>(v[76], v[80], v[81]);
  CHECK(hipGetLastError());
  // layer2.swiglu: swiglu -> (8, 8192)
  swiglu_forward<<<256, 256>>>(v[79], v[81], v[82]);
  CHECK(hipGetLastError());
  // layer2.down: linear -> (8, 2048)
  linear_forward_mlp_down<<<dim3(128, 1), dim3(16, 16)>>>(v[82], v[83], v[84]);
  CHECK(hipGetLastError());
  // layer2.ffn_residual: add -> (8, 2048)
  add_forward<<<64, 256>>>(v[74], v[84], v[85]);
  CHECK(hipGetLastError());
  // layer3.attention_norm: rmsnorm -> (8, 2048)
  rmsnorm_forward<<<8, 256>>>(v[85], v[86], v[87], v[88]);
  CHECK(hipGetLastError());
  // layer3.q: linear -> (8, 2048)
  linear_forward_hidden<<<dim3(128, 1), dim3(16, 16)>>>(v[87], v[89], v[90]);
  CHECK(hipGetLastError());
  // layer3.k: linear -> (8, 512)
  linear_forward_kv<<<dim3(32, 1), dim3(16, 16)>>>(v[87], v[91], v[92]);
  CHECK(hipGetLastError());
  // layer3.v: linear -> (8, 512)
  linear_forward_kv<<<dim3(32, 1), dim3(16, 16)>>>(v[87], v[93], v[94]);
  CHECK(hipGetLastError());
  // layer3.q_rope: rope -> (8, 2048)
  rope_forward_q<<<32, 256>>>(v[90], v[95]);
  CHECK(hipGetLastError());
  // layer3.k_rope: rope -> (8, 512)
  rope_forward_k<<<8, 256>>>(v[92], v[96]);
  CHECK(hipGetLastError());
  // layer3.scores: scores -> (1, 32, 8, 8)
  attention_scores_forward<<<8, 256>>>(v[95], v[96], v[97]);
  CHECK(hipGetLastError());
  // layer3.probabilities: softmax -> (1, 32, 8, 8)
  softmax_forward<<<256, 256>>>(v[97], v[98]);
  CHECK(hipGetLastError());
  // layer3.context: values -> (8, 2048)
  attention_values_forward<<<64, 256>>>(v[98], v[94], v[99]);
  CHECK(hipGetLastError());
  // layer3.o: linear -> (8, 2048)
  linear_forward_hidden<<<dim3(128, 1), dim3(16, 16)>>>(v[99], v[100], v[101]);
  CHECK(hipGetLastError());
  // layer3.attention_residual: add -> (8, 2048)
  add_forward<<<64, 256>>>(v[85], v[101], v[102]);
  CHECK(hipGetLastError());
  // layer3.ffn_norm: rmsnorm -> (8, 2048)
  rmsnorm_forward<<<8, 256>>>(v[102], v[103], v[104], v[105]);
  CHECK(hipGetLastError());
  // layer3.gate: linear -> (8, 8192)
  linear_forward_mlp_up<<<dim3(512, 1), dim3(16, 16)>>>(v[104], v[106], v[107]);
  CHECK(hipGetLastError());
  // layer3.up: linear -> (8, 8192)
  linear_forward_mlp_up<<<dim3(512, 1), dim3(16, 16)>>>(v[104], v[108], v[109]);
  CHECK(hipGetLastError());
  // layer3.swiglu: swiglu -> (8, 8192)
  swiglu_forward<<<256, 256>>>(v[107], v[109], v[110]);
  CHECK(hipGetLastError());
  // layer3.down: linear -> (8, 2048)
  linear_forward_mlp_down<<<dim3(128, 1), dim3(16, 16)>>>(v[110], v[111], v[112]);
  CHECK(hipGetLastError());
  // layer3.ffn_residual: add -> (8, 2048)
  add_forward<<<64, 256>>>(v[102], v[112], v[113]);
  CHECK(hipGetLastError());
  // layer4.attention_norm: rmsnorm -> (8, 2048)
  rmsnorm_forward<<<8, 256>>>(v[113], v[114], v[115], v[116]);
  CHECK(hipGetLastError());
  // layer4.q: linear -> (8, 2048)
  linear_forward_hidden<<<dim3(128, 1), dim3(16, 16)>>>(v[115], v[117], v[118]);
  CHECK(hipGetLastError());
  // layer4.k: linear -> (8, 512)
  linear_forward_kv<<<dim3(32, 1), dim3(16, 16)>>>(v[115], v[119], v[120]);
  CHECK(hipGetLastError());
  // layer4.v: linear -> (8, 512)
  linear_forward_kv<<<dim3(32, 1), dim3(16, 16)>>>(v[115], v[121], v[122]);
  CHECK(hipGetLastError());
  // layer4.q_rope: rope -> (8, 2048)
  rope_forward_q<<<32, 256>>>(v[118], v[123]);
  CHECK(hipGetLastError());
  // layer4.k_rope: rope -> (8, 512)
  rope_forward_k<<<8, 256>>>(v[120], v[124]);
  CHECK(hipGetLastError());
  // layer4.scores: scores -> (1, 32, 8, 8)
  attention_scores_forward<<<8, 256>>>(v[123], v[124], v[125]);
  CHECK(hipGetLastError());
  // layer4.probabilities: softmax -> (1, 32, 8, 8)
  softmax_forward<<<256, 256>>>(v[125], v[126]);
  CHECK(hipGetLastError());
  // layer4.context: values -> (8, 2048)
  attention_values_forward<<<64, 256>>>(v[126], v[122], v[127]);
  CHECK(hipGetLastError());
  // layer4.o: linear -> (8, 2048)
  linear_forward_hidden<<<dim3(128, 1), dim3(16, 16)>>>(v[127], v[128], v[129]);
  CHECK(hipGetLastError());
  // layer4.attention_residual: add -> (8, 2048)
  add_forward<<<64, 256>>>(v[113], v[129], v[130]);
  CHECK(hipGetLastError());
  // layer4.ffn_norm: rmsnorm -> (8, 2048)
  rmsnorm_forward<<<8, 256>>>(v[130], v[131], v[132], v[133]);
  CHECK(hipGetLastError());
  // layer4.gate: linear -> (8, 8192)
  linear_forward_mlp_up<<<dim3(512, 1), dim3(16, 16)>>>(v[132], v[134], v[135]);
  CHECK(hipGetLastError());
  // layer4.up: linear -> (8, 8192)
  linear_forward_mlp_up<<<dim3(512, 1), dim3(16, 16)>>>(v[132], v[136], v[137]);
  CHECK(hipGetLastError());
  // layer4.swiglu: swiglu -> (8, 8192)
  swiglu_forward<<<256, 256>>>(v[135], v[137], v[138]);
  CHECK(hipGetLastError());
  // layer4.down: linear -> (8, 2048)
  linear_forward_mlp_down<<<dim3(128, 1), dim3(16, 16)>>>(v[138], v[139], v[140]);
  CHECK(hipGetLastError());
  // layer4.ffn_residual: add -> (8, 2048)
  add_forward<<<64, 256>>>(v[130], v[140], v[141]);
  CHECK(hipGetLastError());
  // layer5.attention_norm: rmsnorm -> (8, 2048)
  rmsnorm_forward<<<8, 256>>>(v[141], v[142], v[143], v[144]);
  CHECK(hipGetLastError());
  // layer5.q: linear -> (8, 2048)
  linear_forward_hidden<<<dim3(128, 1), dim3(16, 16)>>>(v[143], v[145], v[146]);
  CHECK(hipGetLastError());
  // layer5.k: linear -> (8, 512)
  linear_forward_kv<<<dim3(32, 1), dim3(16, 16)>>>(v[143], v[147], v[148]);
  CHECK(hipGetLastError());
  // layer5.v: linear -> (8, 512)
  linear_forward_kv<<<dim3(32, 1), dim3(16, 16)>>>(v[143], v[149], v[150]);
  CHECK(hipGetLastError());
  // layer5.q_rope: rope -> (8, 2048)
  rope_forward_q<<<32, 256>>>(v[146], v[151]);
  CHECK(hipGetLastError());
  // layer5.k_rope: rope -> (8, 512)
  rope_forward_k<<<8, 256>>>(v[148], v[152]);
  CHECK(hipGetLastError());
  // layer5.scores: scores -> (1, 32, 8, 8)
  attention_scores_forward<<<8, 256>>>(v[151], v[152], v[153]);
  CHECK(hipGetLastError());
  // layer5.probabilities: softmax -> (1, 32, 8, 8)
  softmax_forward<<<256, 256>>>(v[153], v[154]);
  CHECK(hipGetLastError());
  // layer5.context: values -> (8, 2048)
  attention_values_forward<<<64, 256>>>(v[154], v[150], v[155]);
  CHECK(hipGetLastError());
  // layer5.o: linear -> (8, 2048)
  linear_forward_hidden<<<dim3(128, 1), dim3(16, 16)>>>(v[155], v[156], v[157]);
  CHECK(hipGetLastError());
  // layer5.attention_residual: add -> (8, 2048)
  add_forward<<<64, 256>>>(v[141], v[157], v[158]);
  CHECK(hipGetLastError());
  // layer5.ffn_norm: rmsnorm -> (8, 2048)
  rmsnorm_forward<<<8, 256>>>(v[158], v[159], v[160], v[161]);
  CHECK(hipGetLastError());
  // layer5.gate: linear -> (8, 8192)
  linear_forward_mlp_up<<<dim3(512, 1), dim3(16, 16)>>>(v[160], v[162], v[163]);
  CHECK(hipGetLastError());
  // layer5.up: linear -> (8, 8192)
  linear_forward_mlp_up<<<dim3(512, 1), dim3(16, 16)>>>(v[160], v[164], v[165]);
  CHECK(hipGetLastError());
  // layer5.swiglu: swiglu -> (8, 8192)
  swiglu_forward<<<256, 256>>>(v[163], v[165], v[166]);
  CHECK(hipGetLastError());
  // layer5.down: linear -> (8, 2048)
  linear_forward_mlp_down<<<dim3(128, 1), dim3(16, 16)>>>(v[166], v[167], v[168]);
  CHECK(hipGetLastError());
  // layer5.ffn_residual: add -> (8, 2048)
  add_forward<<<64, 256>>>(v[158], v[168], v[169]);
  CHECK(hipGetLastError());
  // layer6.attention_norm: rmsnorm -> (8, 2048)
  rmsnorm_forward<<<8, 256>>>(v[169], v[170], v[171], v[172]);
  CHECK(hipGetLastError());
  // layer6.q: linear -> (8, 2048)
  linear_forward_hidden<<<dim3(128, 1), dim3(16, 16)>>>(v[171], v[173], v[174]);
  CHECK(hipGetLastError());
  // layer6.k: linear -> (8, 512)
  linear_forward_kv<<<dim3(32, 1), dim3(16, 16)>>>(v[171], v[175], v[176]);
  CHECK(hipGetLastError());
  // layer6.v: linear -> (8, 512)
  linear_forward_kv<<<dim3(32, 1), dim3(16, 16)>>>(v[171], v[177], v[178]);
  CHECK(hipGetLastError());
  // layer6.q_rope: rope -> (8, 2048)
  rope_forward_q<<<32, 256>>>(v[174], v[179]);
  CHECK(hipGetLastError());
  // layer6.k_rope: rope -> (8, 512)
  rope_forward_k<<<8, 256>>>(v[176], v[180]);
  CHECK(hipGetLastError());
  // layer6.scores: scores -> (1, 32, 8, 8)
  attention_scores_forward<<<8, 256>>>(v[179], v[180], v[181]);
  CHECK(hipGetLastError());
  // layer6.probabilities: softmax -> (1, 32, 8, 8)
  softmax_forward<<<256, 256>>>(v[181], v[182]);
  CHECK(hipGetLastError());
  // layer6.context: values -> (8, 2048)
  attention_values_forward<<<64, 256>>>(v[182], v[178], v[183]);
  CHECK(hipGetLastError());
  // layer6.o: linear -> (8, 2048)
  linear_forward_hidden<<<dim3(128, 1), dim3(16, 16)>>>(v[183], v[184], v[185]);
  CHECK(hipGetLastError());
  // layer6.attention_residual: add -> (8, 2048)
  add_forward<<<64, 256>>>(v[169], v[185], v[186]);
  CHECK(hipGetLastError());
  // layer6.ffn_norm: rmsnorm -> (8, 2048)
  rmsnorm_forward<<<8, 256>>>(v[186], v[187], v[188], v[189]);
  CHECK(hipGetLastError());
  // layer6.gate: linear -> (8, 8192)
  linear_forward_mlp_up<<<dim3(512, 1), dim3(16, 16)>>>(v[188], v[190], v[191]);
  CHECK(hipGetLastError());
  // layer6.up: linear -> (8, 8192)
  linear_forward_mlp_up<<<dim3(512, 1), dim3(16, 16)>>>(v[188], v[192], v[193]);
  CHECK(hipGetLastError());
  // layer6.swiglu: swiglu -> (8, 8192)
  swiglu_forward<<<256, 256>>>(v[191], v[193], v[194]);
  CHECK(hipGetLastError());
  // layer6.down: linear -> (8, 2048)
  linear_forward_mlp_down<<<dim3(128, 1), dim3(16, 16)>>>(v[194], v[195], v[196]);
  CHECK(hipGetLastError());
  // layer6.ffn_residual: add -> (8, 2048)
  add_forward<<<64, 256>>>(v[186], v[196], v[197]);
  CHECK(hipGetLastError());
  // layer7.attention_norm: rmsnorm -> (8, 2048)
  rmsnorm_forward<<<8, 256>>>(v[197], v[198], v[199], v[200]);
  CHECK(hipGetLastError());
  // layer7.q: linear -> (8, 2048)
  linear_forward_hidden<<<dim3(128, 1), dim3(16, 16)>>>(v[199], v[201], v[202]);
  CHECK(hipGetLastError());
  // layer7.k: linear -> (8, 512)
  linear_forward_kv<<<dim3(32, 1), dim3(16, 16)>>>(v[199], v[203], v[204]);
  CHECK(hipGetLastError());
  // layer7.v: linear -> (8, 512)
  linear_forward_kv<<<dim3(32, 1), dim3(16, 16)>>>(v[199], v[205], v[206]);
  CHECK(hipGetLastError());
  // layer7.q_rope: rope -> (8, 2048)
  rope_forward_q<<<32, 256>>>(v[202], v[207]);
  CHECK(hipGetLastError());
  // layer7.k_rope: rope -> (8, 512)
  rope_forward_k<<<8, 256>>>(v[204], v[208]);
  CHECK(hipGetLastError());
  // layer7.scores: scores -> (1, 32, 8, 8)
  attention_scores_forward<<<8, 256>>>(v[207], v[208], v[209]);
  CHECK(hipGetLastError());
  // layer7.probabilities: softmax -> (1, 32, 8, 8)
  softmax_forward<<<256, 256>>>(v[209], v[210]);
  CHECK(hipGetLastError());
  // layer7.context: values -> (8, 2048)
  attention_values_forward<<<64, 256>>>(v[210], v[206], v[211]);
  CHECK(hipGetLastError());
  // layer7.o: linear -> (8, 2048)
  linear_forward_hidden<<<dim3(128, 1), dim3(16, 16)>>>(v[211], v[212], v[213]);
  CHECK(hipGetLastError());
  // layer7.attention_residual: add -> (8, 2048)
  add_forward<<<64, 256>>>(v[197], v[213], v[214]);
  CHECK(hipGetLastError());
  // layer7.ffn_norm: rmsnorm -> (8, 2048)
  rmsnorm_forward<<<8, 256>>>(v[214], v[215], v[216], v[217]);
  CHECK(hipGetLastError());
  // layer7.gate: linear -> (8, 8192)
  linear_forward_mlp_up<<<dim3(512, 1), dim3(16, 16)>>>(v[216], v[218], v[219]);
  CHECK(hipGetLastError());
  // layer7.up: linear -> (8, 8192)
  linear_forward_mlp_up<<<dim3(512, 1), dim3(16, 16)>>>(v[216], v[220], v[221]);
  CHECK(hipGetLastError());
  // layer7.swiglu: swiglu -> (8, 8192)
  swiglu_forward<<<256, 256>>>(v[219], v[221], v[222]);
  CHECK(hipGetLastError());
  // layer7.down: linear -> (8, 2048)
  linear_forward_mlp_down<<<dim3(128, 1), dim3(16, 16)>>>(v[222], v[223], v[224]);
  CHECK(hipGetLastError());
  // layer7.ffn_residual: add -> (8, 2048)
  add_forward<<<64, 256>>>(v[214], v[224], v[225]);
  CHECK(hipGetLastError());
  // layer8.attention_norm: rmsnorm -> (8, 2048)
  rmsnorm_forward<<<8, 256>>>(v[225], v[226], v[227], v[228]);
  CHECK(hipGetLastError());
  // layer8.q: linear -> (8, 2048)
  linear_forward_hidden<<<dim3(128, 1), dim3(16, 16)>>>(v[227], v[229], v[230]);
  CHECK(hipGetLastError());
  // layer8.k: linear -> (8, 512)
  linear_forward_kv<<<dim3(32, 1), dim3(16, 16)>>>(v[227], v[231], v[232]);
  CHECK(hipGetLastError());
  // layer8.v: linear -> (8, 512)
  linear_forward_kv<<<dim3(32, 1), dim3(16, 16)>>>(v[227], v[233], v[234]);
  CHECK(hipGetLastError());
  // layer8.q_rope: rope -> (8, 2048)
  rope_forward_q<<<32, 256>>>(v[230], v[235]);
  CHECK(hipGetLastError());
  // layer8.k_rope: rope -> (8, 512)
  rope_forward_k<<<8, 256>>>(v[232], v[236]);
  CHECK(hipGetLastError());
  // layer8.scores: scores -> (1, 32, 8, 8)
  attention_scores_forward<<<8, 256>>>(v[235], v[236], v[237]);
  CHECK(hipGetLastError());
  // layer8.probabilities: softmax -> (1, 32, 8, 8)
  softmax_forward<<<256, 256>>>(v[237], v[238]);
  CHECK(hipGetLastError());
  // layer8.context: values -> (8, 2048)
  attention_values_forward<<<64, 256>>>(v[238], v[234], v[239]);
  CHECK(hipGetLastError());
  // layer8.o: linear -> (8, 2048)
  linear_forward_hidden<<<dim3(128, 1), dim3(16, 16)>>>(v[239], v[240], v[241]);
  CHECK(hipGetLastError());
  // layer8.attention_residual: add -> (8, 2048)
  add_forward<<<64, 256>>>(v[225], v[241], v[242]);
  CHECK(hipGetLastError());
  // layer8.ffn_norm: rmsnorm -> (8, 2048)
  rmsnorm_forward<<<8, 256>>>(v[242], v[243], v[244], v[245]);
  CHECK(hipGetLastError());
  // layer8.gate: linear -> (8, 8192)
  linear_forward_mlp_up<<<dim3(512, 1), dim3(16, 16)>>>(v[244], v[246], v[247]);
  CHECK(hipGetLastError());
  // layer8.up: linear -> (8, 8192)
  linear_forward_mlp_up<<<dim3(512, 1), dim3(16, 16)>>>(v[244], v[248], v[249]);
  CHECK(hipGetLastError());
  // layer8.swiglu: swiglu -> (8, 8192)
  swiglu_forward<<<256, 256>>>(v[247], v[249], v[250]);
  CHECK(hipGetLastError());
  // layer8.down: linear -> (8, 2048)
  linear_forward_mlp_down<<<dim3(128, 1), dim3(16, 16)>>>(v[250], v[251], v[252]);
  CHECK(hipGetLastError());
  // layer8.ffn_residual: add -> (8, 2048)
  add_forward<<<64, 256>>>(v[242], v[252], v[253]);
  CHECK(hipGetLastError());
  // layer9.attention_norm: rmsnorm -> (8, 2048)
  rmsnorm_forward<<<8, 256>>>(v[253], v[254], v[255], v[256]);
  CHECK(hipGetLastError());
  // layer9.q: linear -> (8, 2048)
  linear_forward_hidden<<<dim3(128, 1), dim3(16, 16)>>>(v[255], v[257], v[258]);
  CHECK(hipGetLastError());
  // layer9.k: linear -> (8, 512)
  linear_forward_kv<<<dim3(32, 1), dim3(16, 16)>>>(v[255], v[259], v[260]);
  CHECK(hipGetLastError());
  // layer9.v: linear -> (8, 512)
  linear_forward_kv<<<dim3(32, 1), dim3(16, 16)>>>(v[255], v[261], v[262]);
  CHECK(hipGetLastError());
  // layer9.q_rope: rope -> (8, 2048)
  rope_forward_q<<<32, 256>>>(v[258], v[263]);
  CHECK(hipGetLastError());
  // layer9.k_rope: rope -> (8, 512)
  rope_forward_k<<<8, 256>>>(v[260], v[264]);
  CHECK(hipGetLastError());
  // layer9.scores: scores -> (1, 32, 8, 8)
  attention_scores_forward<<<8, 256>>>(v[263], v[264], v[265]);
  CHECK(hipGetLastError());
  // layer9.probabilities: softmax -> (1, 32, 8, 8)
  softmax_forward<<<256, 256>>>(v[265], v[266]);
  CHECK(hipGetLastError());
  // layer9.context: values -> (8, 2048)
  attention_values_forward<<<64, 256>>>(v[266], v[262], v[267]);
  CHECK(hipGetLastError());
  // layer9.o: linear -> (8, 2048)
  linear_forward_hidden<<<dim3(128, 1), dim3(16, 16)>>>(v[267], v[268], v[269]);
  CHECK(hipGetLastError());
  // layer9.attention_residual: add -> (8, 2048)
  add_forward<<<64, 256>>>(v[253], v[269], v[270]);
  CHECK(hipGetLastError());
  // layer9.ffn_norm: rmsnorm -> (8, 2048)
  rmsnorm_forward<<<8, 256>>>(v[270], v[271], v[272], v[273]);
  CHECK(hipGetLastError());
  // layer9.gate: linear -> (8, 8192)
  linear_forward_mlp_up<<<dim3(512, 1), dim3(16, 16)>>>(v[272], v[274], v[275]);
  CHECK(hipGetLastError());
  // layer9.up: linear -> (8, 8192)
  linear_forward_mlp_up<<<dim3(512, 1), dim3(16, 16)>>>(v[272], v[276], v[277]);
  CHECK(hipGetLastError());
  // layer9.swiglu: swiglu -> (8, 8192)
  swiglu_forward<<<256, 256>>>(v[275], v[277], v[278]);
  CHECK(hipGetLastError());
  // layer9.down: linear -> (8, 2048)
  linear_forward_mlp_down<<<dim3(128, 1), dim3(16, 16)>>>(v[278], v[279], v[280]);
  CHECK(hipGetLastError());
  // layer9.ffn_residual: add -> (8, 2048)
  add_forward<<<64, 256>>>(v[270], v[280], v[281]);
  CHECK(hipGetLastError());
  // layer10.attention_norm: rmsnorm -> (8, 2048)
  rmsnorm_forward<<<8, 256>>>(v[281], v[282], v[283], v[284]);
  CHECK(hipGetLastError());
  // layer10.q: linear -> (8, 2048)
  linear_forward_hidden<<<dim3(128, 1), dim3(16, 16)>>>(v[283], v[285], v[286]);
  CHECK(hipGetLastError());
  // layer10.k: linear -> (8, 512)
  linear_forward_kv<<<dim3(32, 1), dim3(16, 16)>>>(v[283], v[287], v[288]);
  CHECK(hipGetLastError());
  // layer10.v: linear -> (8, 512)
  linear_forward_kv<<<dim3(32, 1), dim3(16, 16)>>>(v[283], v[289], v[290]);
  CHECK(hipGetLastError());
  // layer10.q_rope: rope -> (8, 2048)
  rope_forward_q<<<32, 256>>>(v[286], v[291]);
  CHECK(hipGetLastError());
  // layer10.k_rope: rope -> (8, 512)
  rope_forward_k<<<8, 256>>>(v[288], v[292]);
  CHECK(hipGetLastError());
  // layer10.scores: scores -> (1, 32, 8, 8)
  attention_scores_forward<<<8, 256>>>(v[291], v[292], v[293]);
  CHECK(hipGetLastError());
  // layer10.probabilities: softmax -> (1, 32, 8, 8)
  softmax_forward<<<256, 256>>>(v[293], v[294]);
  CHECK(hipGetLastError());
  // layer10.context: values -> (8, 2048)
  attention_values_forward<<<64, 256>>>(v[294], v[290], v[295]);
  CHECK(hipGetLastError());
  // layer10.o: linear -> (8, 2048)
  linear_forward_hidden<<<dim3(128, 1), dim3(16, 16)>>>(v[295], v[296], v[297]);
  CHECK(hipGetLastError());
  // layer10.attention_residual: add -> (8, 2048)
  add_forward<<<64, 256>>>(v[281], v[297], v[298]);
  CHECK(hipGetLastError());
  // layer10.ffn_norm: rmsnorm -> (8, 2048)
  rmsnorm_forward<<<8, 256>>>(v[298], v[299], v[300], v[301]);
  CHECK(hipGetLastError());
  // layer10.gate: linear -> (8, 8192)
  linear_forward_mlp_up<<<dim3(512, 1), dim3(16, 16)>>>(v[300], v[302], v[303]);
  CHECK(hipGetLastError());
  // layer10.up: linear -> (8, 8192)
  linear_forward_mlp_up<<<dim3(512, 1), dim3(16, 16)>>>(v[300], v[304], v[305]);
  CHECK(hipGetLastError());
  // layer10.swiglu: swiglu -> (8, 8192)
  swiglu_forward<<<256, 256>>>(v[303], v[305], v[306]);
  CHECK(hipGetLastError());
  // layer10.down: linear -> (8, 2048)
  linear_forward_mlp_down<<<dim3(128, 1), dim3(16, 16)>>>(v[306], v[307], v[308]);
  CHECK(hipGetLastError());
  // layer10.ffn_residual: add -> (8, 2048)
  add_forward<<<64, 256>>>(v[298], v[308], v[309]);
  CHECK(hipGetLastError());
  // layer11.attention_norm: rmsnorm -> (8, 2048)
  rmsnorm_forward<<<8, 256>>>(v[309], v[310], v[311], v[312]);
  CHECK(hipGetLastError());
  // layer11.q: linear -> (8, 2048)
  linear_forward_hidden<<<dim3(128, 1), dim3(16, 16)>>>(v[311], v[313], v[314]);
  CHECK(hipGetLastError());
  // layer11.k: linear -> (8, 512)
  linear_forward_kv<<<dim3(32, 1), dim3(16, 16)>>>(v[311], v[315], v[316]);
  CHECK(hipGetLastError());
  // layer11.v: linear -> (8, 512)
  linear_forward_kv<<<dim3(32, 1), dim3(16, 16)>>>(v[311], v[317], v[318]);
  CHECK(hipGetLastError());
  // layer11.q_rope: rope -> (8, 2048)
  rope_forward_q<<<32, 256>>>(v[314], v[319]);
  CHECK(hipGetLastError());
  // layer11.k_rope: rope -> (8, 512)
  rope_forward_k<<<8, 256>>>(v[316], v[320]);
  CHECK(hipGetLastError());
  // layer11.scores: scores -> (1, 32, 8, 8)
  attention_scores_forward<<<8, 256>>>(v[319], v[320], v[321]);
  CHECK(hipGetLastError());
  // layer11.probabilities: softmax -> (1, 32, 8, 8)
  softmax_forward<<<256, 256>>>(v[321], v[322]);
  CHECK(hipGetLastError());
  // layer11.context: values -> (8, 2048)
  attention_values_forward<<<64, 256>>>(v[322], v[318], v[323]);
  CHECK(hipGetLastError());
  // layer11.o: linear -> (8, 2048)
  linear_forward_hidden<<<dim3(128, 1), dim3(16, 16)>>>(v[323], v[324], v[325]);
  CHECK(hipGetLastError());
  // layer11.attention_residual: add -> (8, 2048)
  add_forward<<<64, 256>>>(v[309], v[325], v[326]);
  CHECK(hipGetLastError());
  // layer11.ffn_norm: rmsnorm -> (8, 2048)
  rmsnorm_forward<<<8, 256>>>(v[326], v[327], v[328], v[329]);
  CHECK(hipGetLastError());
  // layer11.gate: linear -> (8, 8192)
  linear_forward_mlp_up<<<dim3(512, 1), dim3(16, 16)>>>(v[328], v[330], v[331]);
  CHECK(hipGetLastError());
  // layer11.up: linear -> (8, 8192)
  linear_forward_mlp_up<<<dim3(512, 1), dim3(16, 16)>>>(v[328], v[332], v[333]);
  CHECK(hipGetLastError());
  // layer11.swiglu: swiglu -> (8, 8192)
  swiglu_forward<<<256, 256>>>(v[331], v[333], v[334]);
  CHECK(hipGetLastError());
  // layer11.down: linear -> (8, 2048)
  linear_forward_mlp_down<<<dim3(128, 1), dim3(16, 16)>>>(v[334], v[335], v[336]);
  CHECK(hipGetLastError());
  // layer11.ffn_residual: add -> (8, 2048)
  add_forward<<<64, 256>>>(v[326], v[336], v[337]);
  CHECK(hipGetLastError());
  // layer12.attention_norm: rmsnorm -> (8, 2048)
  rmsnorm_forward<<<8, 256>>>(v[337], v[338], v[339], v[340]);
  CHECK(hipGetLastError());
  // layer12.q: linear -> (8, 2048)
  linear_forward_hidden<<<dim3(128, 1), dim3(16, 16)>>>(v[339], v[341], v[342]);
  CHECK(hipGetLastError());
  // layer12.k: linear -> (8, 512)
  linear_forward_kv<<<dim3(32, 1), dim3(16, 16)>>>(v[339], v[343], v[344]);
  CHECK(hipGetLastError());
  // layer12.v: linear -> (8, 512)
  linear_forward_kv<<<dim3(32, 1), dim3(16, 16)>>>(v[339], v[345], v[346]);
  CHECK(hipGetLastError());
  // layer12.q_rope: rope -> (8, 2048)
  rope_forward_q<<<32, 256>>>(v[342], v[347]);
  CHECK(hipGetLastError());
  // layer12.k_rope: rope -> (8, 512)
  rope_forward_k<<<8, 256>>>(v[344], v[348]);
  CHECK(hipGetLastError());
  // layer12.scores: scores -> (1, 32, 8, 8)
  attention_scores_forward<<<8, 256>>>(v[347], v[348], v[349]);
  CHECK(hipGetLastError());
  // layer12.probabilities: softmax -> (1, 32, 8, 8)
  softmax_forward<<<256, 256>>>(v[349], v[350]);
  CHECK(hipGetLastError());
  // layer12.context: values -> (8, 2048)
  attention_values_forward<<<64, 256>>>(v[350], v[346], v[351]);
  CHECK(hipGetLastError());
  // layer12.o: linear -> (8, 2048)
  linear_forward_hidden<<<dim3(128, 1), dim3(16, 16)>>>(v[351], v[352], v[353]);
  CHECK(hipGetLastError());
  // layer12.attention_residual: add -> (8, 2048)
  add_forward<<<64, 256>>>(v[337], v[353], v[354]);
  CHECK(hipGetLastError());
  // layer12.ffn_norm: rmsnorm -> (8, 2048)
  rmsnorm_forward<<<8, 256>>>(v[354], v[355], v[356], v[357]);
  CHECK(hipGetLastError());
  // layer12.gate: linear -> (8, 8192)
  linear_forward_mlp_up<<<dim3(512, 1), dim3(16, 16)>>>(v[356], v[358], v[359]);
  CHECK(hipGetLastError());
  // layer12.up: linear -> (8, 8192)
  linear_forward_mlp_up<<<dim3(512, 1), dim3(16, 16)>>>(v[356], v[360], v[361]);
  CHECK(hipGetLastError());
  // layer12.swiglu: swiglu -> (8, 8192)
  swiglu_forward<<<256, 256>>>(v[359], v[361], v[362]);
  CHECK(hipGetLastError());
  // layer12.down: linear -> (8, 2048)
  linear_forward_mlp_down<<<dim3(128, 1), dim3(16, 16)>>>(v[362], v[363], v[364]);
  CHECK(hipGetLastError());
  // layer12.ffn_residual: add -> (8, 2048)
  add_forward<<<64, 256>>>(v[354], v[364], v[365]);
  CHECK(hipGetLastError());
  // layer13.attention_norm: rmsnorm -> (8, 2048)
  rmsnorm_forward<<<8, 256>>>(v[365], v[366], v[367], v[368]);
  CHECK(hipGetLastError());
  // layer13.q: linear -> (8, 2048)
  linear_forward_hidden<<<dim3(128, 1), dim3(16, 16)>>>(v[367], v[369], v[370]);
  CHECK(hipGetLastError());
  // layer13.k: linear -> (8, 512)
  linear_forward_kv<<<dim3(32, 1), dim3(16, 16)>>>(v[367], v[371], v[372]);
  CHECK(hipGetLastError());
  // layer13.v: linear -> (8, 512)
  linear_forward_kv<<<dim3(32, 1), dim3(16, 16)>>>(v[367], v[373], v[374]);
  CHECK(hipGetLastError());
  // layer13.q_rope: rope -> (8, 2048)
  rope_forward_q<<<32, 256>>>(v[370], v[375]);
  CHECK(hipGetLastError());
  // layer13.k_rope: rope -> (8, 512)
  rope_forward_k<<<8, 256>>>(v[372], v[376]);
  CHECK(hipGetLastError());
  // layer13.scores: scores -> (1, 32, 8, 8)
  attention_scores_forward<<<8, 256>>>(v[375], v[376], v[377]);
  CHECK(hipGetLastError());
  // layer13.probabilities: softmax -> (1, 32, 8, 8)
  softmax_forward<<<256, 256>>>(v[377], v[378]);
  CHECK(hipGetLastError());
  // layer13.context: values -> (8, 2048)
  attention_values_forward<<<64, 256>>>(v[378], v[374], v[379]);
  CHECK(hipGetLastError());
  // layer13.o: linear -> (8, 2048)
  linear_forward_hidden<<<dim3(128, 1), dim3(16, 16)>>>(v[379], v[380], v[381]);
  CHECK(hipGetLastError());
  // layer13.attention_residual: add -> (8, 2048)
  add_forward<<<64, 256>>>(v[365], v[381], v[382]);
  CHECK(hipGetLastError());
  // layer13.ffn_norm: rmsnorm -> (8, 2048)
  rmsnorm_forward<<<8, 256>>>(v[382], v[383], v[384], v[385]);
  CHECK(hipGetLastError());
  // layer13.gate: linear -> (8, 8192)
  linear_forward_mlp_up<<<dim3(512, 1), dim3(16, 16)>>>(v[384], v[386], v[387]);
  CHECK(hipGetLastError());
  // layer13.up: linear -> (8, 8192)
  linear_forward_mlp_up<<<dim3(512, 1), dim3(16, 16)>>>(v[384], v[388], v[389]);
  CHECK(hipGetLastError());
  // layer13.swiglu: swiglu -> (8, 8192)
  swiglu_forward<<<256, 256>>>(v[387], v[389], v[390]);
  CHECK(hipGetLastError());
  // layer13.down: linear -> (8, 2048)
  linear_forward_mlp_down<<<dim3(128, 1), dim3(16, 16)>>>(v[390], v[391], v[392]);
  CHECK(hipGetLastError());
  // layer13.ffn_residual: add -> (8, 2048)
  add_forward<<<64, 256>>>(v[382], v[392], v[393]);
  CHECK(hipGetLastError());
  // layer14.attention_norm: rmsnorm -> (8, 2048)
  rmsnorm_forward<<<8, 256>>>(v[393], v[394], v[395], v[396]);
  CHECK(hipGetLastError());
  // layer14.q: linear -> (8, 2048)
  linear_forward_hidden<<<dim3(128, 1), dim3(16, 16)>>>(v[395], v[397], v[398]);
  CHECK(hipGetLastError());
  // layer14.k: linear -> (8, 512)
  linear_forward_kv<<<dim3(32, 1), dim3(16, 16)>>>(v[395], v[399], v[400]);
  CHECK(hipGetLastError());
  // layer14.v: linear -> (8, 512)
  linear_forward_kv<<<dim3(32, 1), dim3(16, 16)>>>(v[395], v[401], v[402]);
  CHECK(hipGetLastError());
  // layer14.q_rope: rope -> (8, 2048)
  rope_forward_q<<<32, 256>>>(v[398], v[403]);
  CHECK(hipGetLastError());
  // layer14.k_rope: rope -> (8, 512)
  rope_forward_k<<<8, 256>>>(v[400], v[404]);
  CHECK(hipGetLastError());
  // layer14.scores: scores -> (1, 32, 8, 8)
  attention_scores_forward<<<8, 256>>>(v[403], v[404], v[405]);
  CHECK(hipGetLastError());
  // layer14.probabilities: softmax -> (1, 32, 8, 8)
  softmax_forward<<<256, 256>>>(v[405], v[406]);
  CHECK(hipGetLastError());
  // layer14.context: values -> (8, 2048)
  attention_values_forward<<<64, 256>>>(v[406], v[402], v[407]);
  CHECK(hipGetLastError());
  // layer14.o: linear -> (8, 2048)
  linear_forward_hidden<<<dim3(128, 1), dim3(16, 16)>>>(v[407], v[408], v[409]);
  CHECK(hipGetLastError());
  // layer14.attention_residual: add -> (8, 2048)
  add_forward<<<64, 256>>>(v[393], v[409], v[410]);
  CHECK(hipGetLastError());
  // layer14.ffn_norm: rmsnorm -> (8, 2048)
  rmsnorm_forward<<<8, 256>>>(v[410], v[411], v[412], v[413]);
  CHECK(hipGetLastError());
  // layer14.gate: linear -> (8, 8192)
  linear_forward_mlp_up<<<dim3(512, 1), dim3(16, 16)>>>(v[412], v[414], v[415]);
  CHECK(hipGetLastError());
  // layer14.up: linear -> (8, 8192)
  linear_forward_mlp_up<<<dim3(512, 1), dim3(16, 16)>>>(v[412], v[416], v[417]);
  CHECK(hipGetLastError());
  // layer14.swiglu: swiglu -> (8, 8192)
  swiglu_forward<<<256, 256>>>(v[415], v[417], v[418]);
  CHECK(hipGetLastError());
  // layer14.down: linear -> (8, 2048)
  linear_forward_mlp_down<<<dim3(128, 1), dim3(16, 16)>>>(v[418], v[419], v[420]);
  CHECK(hipGetLastError());
  // layer14.ffn_residual: add -> (8, 2048)
  add_forward<<<64, 256>>>(v[410], v[420], v[421]);
  CHECK(hipGetLastError());
  // layer15.attention_norm: rmsnorm -> (8, 2048)
  rmsnorm_forward<<<8, 256>>>(v[421], v[422], v[423], v[424]);
  CHECK(hipGetLastError());
  // layer15.q: linear -> (8, 2048)
  linear_forward_hidden<<<dim3(128, 1), dim3(16, 16)>>>(v[423], v[425], v[426]);
  CHECK(hipGetLastError());
  // layer15.k: linear -> (8, 512)
  linear_forward_kv<<<dim3(32, 1), dim3(16, 16)>>>(v[423], v[427], v[428]);
  CHECK(hipGetLastError());
  // layer15.v: linear -> (8, 512)
  linear_forward_kv<<<dim3(32, 1), dim3(16, 16)>>>(v[423], v[429], v[430]);
  CHECK(hipGetLastError());
  // layer15.q_rope: rope -> (8, 2048)
  rope_forward_q<<<32, 256>>>(v[426], v[431]);
  CHECK(hipGetLastError());
  // layer15.k_rope: rope -> (8, 512)
  rope_forward_k<<<8, 256>>>(v[428], v[432]);
  CHECK(hipGetLastError());
  // layer15.scores: scores -> (1, 32, 8, 8)
  attention_scores_forward<<<8, 256>>>(v[431], v[432], v[433]);
  CHECK(hipGetLastError());
  // layer15.probabilities: softmax -> (1, 32, 8, 8)
  softmax_forward<<<256, 256>>>(v[433], v[434]);
  CHECK(hipGetLastError());
  // layer15.context: values -> (8, 2048)
  attention_values_forward<<<64, 256>>>(v[434], v[430], v[435]);
  CHECK(hipGetLastError());
  // layer15.o: linear -> (8, 2048)
  linear_forward_hidden<<<dim3(128, 1), dim3(16, 16)>>>(v[435], v[436], v[437]);
  CHECK(hipGetLastError());
  // layer15.attention_residual: add -> (8, 2048)
  add_forward<<<64, 256>>>(v[421], v[437], v[438]);
  CHECK(hipGetLastError());
  // layer15.ffn_norm: rmsnorm -> (8, 2048)
  rmsnorm_forward<<<8, 256>>>(v[438], v[439], v[440], v[441]);
  CHECK(hipGetLastError());
  // layer15.gate: linear -> (8, 8192)
  linear_forward_mlp_up<<<dim3(512, 1), dim3(16, 16)>>>(v[440], v[442], v[443]);
  CHECK(hipGetLastError());
  // layer15.up: linear -> (8, 8192)
  linear_forward_mlp_up<<<dim3(512, 1), dim3(16, 16)>>>(v[440], v[444], v[445]);
  CHECK(hipGetLastError());
  // layer15.swiglu: swiglu -> (8, 8192)
  swiglu_forward<<<256, 256>>>(v[443], v[445], v[446]);
  CHECK(hipGetLastError());
  // layer15.down: linear -> (8, 2048)
  linear_forward_mlp_down<<<dim3(128, 1), dim3(16, 16)>>>(v[446], v[447], v[448]);
  CHECK(hipGetLastError());
  // layer15.ffn_residual: add -> (8, 2048)
  add_forward<<<64, 256>>>(v[438], v[448], v[449]);
  CHECK(hipGetLastError());
  // final_norm: rmsnorm -> (8, 2048)
  rmsnorm_forward<<<8, 256>>>(v[449], v[450], v[451], v[452]);
  CHECK(hipGetLastError());
  // lm_head: linear -> (8, 128256)
  linear_forward_lm_head<<<dim3(8016, 1), dim3(16, 16)>>>(v[451], v[0], v[453]);
  CHECK(hipGetLastError());
  // token_losses: cross_entropy -> (8,)
  cross_entropy_forward<<<8, 256>>>(v[453], targets, v[454]);
  CHECK(hipGetLastError());
  // loss: mean -> (1,)
  mean_forward<<<1, 256>>>(v[454], v[455]);
  CHECK(hipGetLastError());
  return hipDeviceSynchronize();
}
