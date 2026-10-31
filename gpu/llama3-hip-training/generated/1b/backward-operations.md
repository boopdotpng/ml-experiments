# Generated backward execution order

Each gradient is accumulated with `+=`. The seed is `dLoss = 1`.

| Step | Forward operation | Backward kernel | Gradients written |
| --- | --- | --- | --- |
| 1 | loss | `mean_backward` | dtoken_losses |
| 2 | token_losses | `cross_entropy_backward` | dlm_head |
| 3 | lm_head | `linear_backward_input_lm_head` | dfinal_norm |
| 4 | lm_head | `linear_backward_weight_lm_head` | dembedding.weight |
| 5 | final_norm | `rmsnorm_backward_input` | dlayer15.ffn_residual |
| 6 | final_norm | `rmsnorm_backward_weight` | dfinal_norm.weight |
| 7 | layer15.ffn_residual | `add_backward` | dlayer15.attention_residual, dlayer15.down |
| 8 | layer15.down | `linear_backward_input_mlp_down` | dlayer15.swiglu |
| 9 | layer15.down | `linear_backward_weight_mlp_down` | dlayer15.down.weight |
| 10 | layer15.swiglu | `swiglu_backward` | dlayer15.gate, dlayer15.up |
| 11 | layer15.up | `linear_backward_input_mlp_up` | dlayer15.ffn_norm |
| 12 | layer15.up | `linear_backward_weight_mlp_up` | dlayer15.up.weight |
| 13 | layer15.gate | `linear_backward_input_mlp_up` | dlayer15.ffn_norm |
| 14 | layer15.gate | `linear_backward_weight_mlp_up` | dlayer15.gate.weight |
| 15 | layer15.ffn_norm | `rmsnorm_backward_input` | dlayer15.attention_residual |
| 16 | layer15.ffn_norm | `rmsnorm_backward_weight` | dlayer15.ffn_norm.weight |
| 17 | layer15.attention_residual | `add_backward` | dlayer14.ffn_residual, dlayer15.o |
| 18 | layer15.o | `linear_backward_input_hidden` | dlayer15.context |
| 19 | layer15.o | `linear_backward_weight_hidden` | dlayer15.o.weight |
| 20 | layer15.context | `attention_values_backward_p` | dlayer15.probabilities |
| 21 | layer15.context | `attention_values_backward_v` | dlayer15.v |
| 22 | layer15.probabilities | `softmax_backward` | dlayer15.scores |
| 23 | layer15.scores | `attention_scores_backward_q` | dlayer15.q_rope |
| 24 | layer15.scores | `attention_scores_backward_k` | dlayer15.k_rope |
| 25 | layer15.k_rope | `rope_backward_k` | dlayer15.k |
| 26 | layer15.q_rope | `rope_backward_q` | dlayer15.q |
| 27 | layer15.v | `linear_backward_input_kv` | dlayer15.attention_norm |
| 28 | layer15.v | `linear_backward_weight_kv` | dlayer15.v.weight |
| 29 | layer15.k | `linear_backward_input_kv` | dlayer15.attention_norm |
| 30 | layer15.k | `linear_backward_weight_kv` | dlayer15.k.weight |
| 31 | layer15.q | `linear_backward_input_hidden` | dlayer15.attention_norm |
| 32 | layer15.q | `linear_backward_weight_hidden` | dlayer15.q.weight |
| 33 | layer15.attention_norm | `rmsnorm_backward_input` | dlayer14.ffn_residual |
| 34 | layer15.attention_norm | `rmsnorm_backward_weight` | dlayer15.attention_norm.weight |
| 35 | layer14.ffn_residual | `add_backward` | dlayer14.attention_residual, dlayer14.down |
| 36 | layer14.down | `linear_backward_input_mlp_down` | dlayer14.swiglu |
| 37 | layer14.down | `linear_backward_weight_mlp_down` | dlayer14.down.weight |
| 38 | layer14.swiglu | `swiglu_backward` | dlayer14.gate, dlayer14.up |
| 39 | layer14.up | `linear_backward_input_mlp_up` | dlayer14.ffn_norm |
| 40 | layer14.up | `linear_backward_weight_mlp_up` | dlayer14.up.weight |
| 41 | layer14.gate | `linear_backward_input_mlp_up` | dlayer14.ffn_norm |
| 42 | layer14.gate | `linear_backward_weight_mlp_up` | dlayer14.gate.weight |
| 43 | layer14.ffn_norm | `rmsnorm_backward_input` | dlayer14.attention_residual |
| 44 | layer14.ffn_norm | `rmsnorm_backward_weight` | dlayer14.ffn_norm.weight |
| 45 | layer14.attention_residual | `add_backward` | dlayer13.ffn_residual, dlayer14.o |
| 46 | layer14.o | `linear_backward_input_hidden` | dlayer14.context |
| 47 | layer14.o | `linear_backward_weight_hidden` | dlayer14.o.weight |
| 48 | layer14.context | `attention_values_backward_p` | dlayer14.probabilities |
| 49 | layer14.context | `attention_values_backward_v` | dlayer14.v |
| 50 | layer14.probabilities | `softmax_backward` | dlayer14.scores |
| 51 | layer14.scores | `attention_scores_backward_q` | dlayer14.q_rope |
| 52 | layer14.scores | `attention_scores_backward_k` | dlayer14.k_rope |
| 53 | layer14.k_rope | `rope_backward_k` | dlayer14.k |
| 54 | layer14.q_rope | `rope_backward_q` | dlayer14.q |
| 55 | layer14.v | `linear_backward_input_kv` | dlayer14.attention_norm |
| 56 | layer14.v | `linear_backward_weight_kv` | dlayer14.v.weight |
| 57 | layer14.k | `linear_backward_input_kv` | dlayer14.attention_norm |
| 58 | layer14.k | `linear_backward_weight_kv` | dlayer14.k.weight |
| 59 | layer14.q | `linear_backward_input_hidden` | dlayer14.attention_norm |
| 60 | layer14.q | `linear_backward_weight_hidden` | dlayer14.q.weight |
| 61 | layer14.attention_norm | `rmsnorm_backward_input` | dlayer13.ffn_residual |
| 62 | layer14.attention_norm | `rmsnorm_backward_weight` | dlayer14.attention_norm.weight |
| 63 | layer13.ffn_residual | `add_backward` | dlayer13.attention_residual, dlayer13.down |
| 64 | layer13.down | `linear_backward_input_mlp_down` | dlayer13.swiglu |
| 65 | layer13.down | `linear_backward_weight_mlp_down` | dlayer13.down.weight |
| 66 | layer13.swiglu | `swiglu_backward` | dlayer13.gate, dlayer13.up |
| 67 | layer13.up | `linear_backward_input_mlp_up` | dlayer13.ffn_norm |
| 68 | layer13.up | `linear_backward_weight_mlp_up` | dlayer13.up.weight |
| 69 | layer13.gate | `linear_backward_input_mlp_up` | dlayer13.ffn_norm |
| 70 | layer13.gate | `linear_backward_weight_mlp_up` | dlayer13.gate.weight |
| 71 | layer13.ffn_norm | `rmsnorm_backward_input` | dlayer13.attention_residual |
| 72 | layer13.ffn_norm | `rmsnorm_backward_weight` | dlayer13.ffn_norm.weight |
| 73 | layer13.attention_residual | `add_backward` | dlayer12.ffn_residual, dlayer13.o |
| 74 | layer13.o | `linear_backward_input_hidden` | dlayer13.context |
| 75 | layer13.o | `linear_backward_weight_hidden` | dlayer13.o.weight |
| 76 | layer13.context | `attention_values_backward_p` | dlayer13.probabilities |
| 77 | layer13.context | `attention_values_backward_v` | dlayer13.v |
| 78 | layer13.probabilities | `softmax_backward` | dlayer13.scores |
| 79 | layer13.scores | `attention_scores_backward_q` | dlayer13.q_rope |
| 80 | layer13.scores | `attention_scores_backward_k` | dlayer13.k_rope |
| 81 | layer13.k_rope | `rope_backward_k` | dlayer13.k |
| 82 | layer13.q_rope | `rope_backward_q` | dlayer13.q |
| 83 | layer13.v | `linear_backward_input_kv` | dlayer13.attention_norm |
| 84 | layer13.v | `linear_backward_weight_kv` | dlayer13.v.weight |
| 85 | layer13.k | `linear_backward_input_kv` | dlayer13.attention_norm |
| 86 | layer13.k | `linear_backward_weight_kv` | dlayer13.k.weight |
| 87 | layer13.q | `linear_backward_input_hidden` | dlayer13.attention_norm |
| 88 | layer13.q | `linear_backward_weight_hidden` | dlayer13.q.weight |
| 89 | layer13.attention_norm | `rmsnorm_backward_input` | dlayer12.ffn_residual |
| 90 | layer13.attention_norm | `rmsnorm_backward_weight` | dlayer13.attention_norm.weight |
| 91 | layer12.ffn_residual | `add_backward` | dlayer12.attention_residual, dlayer12.down |
| 92 | layer12.down | `linear_backward_input_mlp_down` | dlayer12.swiglu |
| 93 | layer12.down | `linear_backward_weight_mlp_down` | dlayer12.down.weight |
| 94 | layer12.swiglu | `swiglu_backward` | dlayer12.gate, dlayer12.up |
| 95 | layer12.up | `linear_backward_input_mlp_up` | dlayer12.ffn_norm |
| 96 | layer12.up | `linear_backward_weight_mlp_up` | dlayer12.up.weight |
| 97 | layer12.gate | `linear_backward_input_mlp_up` | dlayer12.ffn_norm |
| 98 | layer12.gate | `linear_backward_weight_mlp_up` | dlayer12.gate.weight |
| 99 | layer12.ffn_norm | `rmsnorm_backward_input` | dlayer12.attention_residual |
| 100 | layer12.ffn_norm | `rmsnorm_backward_weight` | dlayer12.ffn_norm.weight |
| 101 | layer12.attention_residual | `add_backward` | dlayer11.ffn_residual, dlayer12.o |
| 102 | layer12.o | `linear_backward_input_hidden` | dlayer12.context |
| 103 | layer12.o | `linear_backward_weight_hidden` | dlayer12.o.weight |
| 104 | layer12.context | `attention_values_backward_p` | dlayer12.probabilities |
| 105 | layer12.context | `attention_values_backward_v` | dlayer12.v |
| 106 | layer12.probabilities | `softmax_backward` | dlayer12.scores |
| 107 | layer12.scores | `attention_scores_backward_q` | dlayer12.q_rope |
| 108 | layer12.scores | `attention_scores_backward_k` | dlayer12.k_rope |
| 109 | layer12.k_rope | `rope_backward_k` | dlayer12.k |
| 110 | layer12.q_rope | `rope_backward_q` | dlayer12.q |
| 111 | layer12.v | `linear_backward_input_kv` | dlayer12.attention_norm |
| 112 | layer12.v | `linear_backward_weight_kv` | dlayer12.v.weight |
| 113 | layer12.k | `linear_backward_input_kv` | dlayer12.attention_norm |
| 114 | layer12.k | `linear_backward_weight_kv` | dlayer12.k.weight |
| 115 | layer12.q | `linear_backward_input_hidden` | dlayer12.attention_norm |
| 116 | layer12.q | `linear_backward_weight_hidden` | dlayer12.q.weight |
| 117 | layer12.attention_norm | `rmsnorm_backward_input` | dlayer11.ffn_residual |
| 118 | layer12.attention_norm | `rmsnorm_backward_weight` | dlayer12.attention_norm.weight |
| 119 | layer11.ffn_residual | `add_backward` | dlayer11.attention_residual, dlayer11.down |
| 120 | layer11.down | `linear_backward_input_mlp_down` | dlayer11.swiglu |
| 121 | layer11.down | `linear_backward_weight_mlp_down` | dlayer11.down.weight |
| 122 | layer11.swiglu | `swiglu_backward` | dlayer11.gate, dlayer11.up |
| 123 | layer11.up | `linear_backward_input_mlp_up` | dlayer11.ffn_norm |
| 124 | layer11.up | `linear_backward_weight_mlp_up` | dlayer11.up.weight |
| 125 | layer11.gate | `linear_backward_input_mlp_up` | dlayer11.ffn_norm |
| 126 | layer11.gate | `linear_backward_weight_mlp_up` | dlayer11.gate.weight |
| 127 | layer11.ffn_norm | `rmsnorm_backward_input` | dlayer11.attention_residual |
| 128 | layer11.ffn_norm | `rmsnorm_backward_weight` | dlayer11.ffn_norm.weight |
| 129 | layer11.attention_residual | `add_backward` | dlayer10.ffn_residual, dlayer11.o |
| 130 | layer11.o | `linear_backward_input_hidden` | dlayer11.context |
| 131 | layer11.o | `linear_backward_weight_hidden` | dlayer11.o.weight |
| 132 | layer11.context | `attention_values_backward_p` | dlayer11.probabilities |
| 133 | layer11.context | `attention_values_backward_v` | dlayer11.v |
| 134 | layer11.probabilities | `softmax_backward` | dlayer11.scores |
| 135 | layer11.scores | `attention_scores_backward_q` | dlayer11.q_rope |
| 136 | layer11.scores | `attention_scores_backward_k` | dlayer11.k_rope |
| 137 | layer11.k_rope | `rope_backward_k` | dlayer11.k |
| 138 | layer11.q_rope | `rope_backward_q` | dlayer11.q |
| 139 | layer11.v | `linear_backward_input_kv` | dlayer11.attention_norm |
| 140 | layer11.v | `linear_backward_weight_kv` | dlayer11.v.weight |
| 141 | layer11.k | `linear_backward_input_kv` | dlayer11.attention_norm |
| 142 | layer11.k | `linear_backward_weight_kv` | dlayer11.k.weight |
| 143 | layer11.q | `linear_backward_input_hidden` | dlayer11.attention_norm |
| 144 | layer11.q | `linear_backward_weight_hidden` | dlayer11.q.weight |
| 145 | layer11.attention_norm | `rmsnorm_backward_input` | dlayer10.ffn_residual |
| 146 | layer11.attention_norm | `rmsnorm_backward_weight` | dlayer11.attention_norm.weight |
| 147 | layer10.ffn_residual | `add_backward` | dlayer10.attention_residual, dlayer10.down |
| 148 | layer10.down | `linear_backward_input_mlp_down` | dlayer10.swiglu |
| 149 | layer10.down | `linear_backward_weight_mlp_down` | dlayer10.down.weight |
| 150 | layer10.swiglu | `swiglu_backward` | dlayer10.gate, dlayer10.up |
| 151 | layer10.up | `linear_backward_input_mlp_up` | dlayer10.ffn_norm |
| 152 | layer10.up | `linear_backward_weight_mlp_up` | dlayer10.up.weight |
| 153 | layer10.gate | `linear_backward_input_mlp_up` | dlayer10.ffn_norm |
| 154 | layer10.gate | `linear_backward_weight_mlp_up` | dlayer10.gate.weight |
| 155 | layer10.ffn_norm | `rmsnorm_backward_input` | dlayer10.attention_residual |
| 156 | layer10.ffn_norm | `rmsnorm_backward_weight` | dlayer10.ffn_norm.weight |
| 157 | layer10.attention_residual | `add_backward` | dlayer9.ffn_residual, dlayer10.o |
| 158 | layer10.o | `linear_backward_input_hidden` | dlayer10.context |
| 159 | layer10.o | `linear_backward_weight_hidden` | dlayer10.o.weight |
| 160 | layer10.context | `attention_values_backward_p` | dlayer10.probabilities |
| 161 | layer10.context | `attention_values_backward_v` | dlayer10.v |
| 162 | layer10.probabilities | `softmax_backward` | dlayer10.scores |
| 163 | layer10.scores | `attention_scores_backward_q` | dlayer10.q_rope |
| 164 | layer10.scores | `attention_scores_backward_k` | dlayer10.k_rope |
| 165 | layer10.k_rope | `rope_backward_k` | dlayer10.k |
| 166 | layer10.q_rope | `rope_backward_q` | dlayer10.q |
| 167 | layer10.v | `linear_backward_input_kv` | dlayer10.attention_norm |
| 168 | layer10.v | `linear_backward_weight_kv` | dlayer10.v.weight |
| 169 | layer10.k | `linear_backward_input_kv` | dlayer10.attention_norm |
| 170 | layer10.k | `linear_backward_weight_kv` | dlayer10.k.weight |
| 171 | layer10.q | `linear_backward_input_hidden` | dlayer10.attention_norm |
| 172 | layer10.q | `linear_backward_weight_hidden` | dlayer10.q.weight |
| 173 | layer10.attention_norm | `rmsnorm_backward_input` | dlayer9.ffn_residual |
| 174 | layer10.attention_norm | `rmsnorm_backward_weight` | dlayer10.attention_norm.weight |
| 175 | layer9.ffn_residual | `add_backward` | dlayer9.attention_residual, dlayer9.down |
| 176 | layer9.down | `linear_backward_input_mlp_down` | dlayer9.swiglu |
| 177 | layer9.down | `linear_backward_weight_mlp_down` | dlayer9.down.weight |
| 178 | layer9.swiglu | `swiglu_backward` | dlayer9.gate, dlayer9.up |
| 179 | layer9.up | `linear_backward_input_mlp_up` | dlayer9.ffn_norm |
| 180 | layer9.up | `linear_backward_weight_mlp_up` | dlayer9.up.weight |
| 181 | layer9.gate | `linear_backward_input_mlp_up` | dlayer9.ffn_norm |
| 182 | layer9.gate | `linear_backward_weight_mlp_up` | dlayer9.gate.weight |
| 183 | layer9.ffn_norm | `rmsnorm_backward_input` | dlayer9.attention_residual |
| 184 | layer9.ffn_norm | `rmsnorm_backward_weight` | dlayer9.ffn_norm.weight |
| 185 | layer9.attention_residual | `add_backward` | dlayer8.ffn_residual, dlayer9.o |
| 186 | layer9.o | `linear_backward_input_hidden` | dlayer9.context |
| 187 | layer9.o | `linear_backward_weight_hidden` | dlayer9.o.weight |
| 188 | layer9.context | `attention_values_backward_p` | dlayer9.probabilities |
| 189 | layer9.context | `attention_values_backward_v` | dlayer9.v |
| 190 | layer9.probabilities | `softmax_backward` | dlayer9.scores |
| 191 | layer9.scores | `attention_scores_backward_q` | dlayer9.q_rope |
| 192 | layer9.scores | `attention_scores_backward_k` | dlayer9.k_rope |
| 193 | layer9.k_rope | `rope_backward_k` | dlayer9.k |
| 194 | layer9.q_rope | `rope_backward_q` | dlayer9.q |
| 195 | layer9.v | `linear_backward_input_kv` | dlayer9.attention_norm |
| 196 | layer9.v | `linear_backward_weight_kv` | dlayer9.v.weight |
| 197 | layer9.k | `linear_backward_input_kv` | dlayer9.attention_norm |
| 198 | layer9.k | `linear_backward_weight_kv` | dlayer9.k.weight |
| 199 | layer9.q | `linear_backward_input_hidden` | dlayer9.attention_norm |
| 200 | layer9.q | `linear_backward_weight_hidden` | dlayer9.q.weight |
| 201 | layer9.attention_norm | `rmsnorm_backward_input` | dlayer8.ffn_residual |
| 202 | layer9.attention_norm | `rmsnorm_backward_weight` | dlayer9.attention_norm.weight |
| 203 | layer8.ffn_residual | `add_backward` | dlayer8.attention_residual, dlayer8.down |
| 204 | layer8.down | `linear_backward_input_mlp_down` | dlayer8.swiglu |
| 205 | layer8.down | `linear_backward_weight_mlp_down` | dlayer8.down.weight |
| 206 | layer8.swiglu | `swiglu_backward` | dlayer8.gate, dlayer8.up |
| 207 | layer8.up | `linear_backward_input_mlp_up` | dlayer8.ffn_norm |
| 208 | layer8.up | `linear_backward_weight_mlp_up` | dlayer8.up.weight |
| 209 | layer8.gate | `linear_backward_input_mlp_up` | dlayer8.ffn_norm |
| 210 | layer8.gate | `linear_backward_weight_mlp_up` | dlayer8.gate.weight |
| 211 | layer8.ffn_norm | `rmsnorm_backward_input` | dlayer8.attention_residual |
| 212 | layer8.ffn_norm | `rmsnorm_backward_weight` | dlayer8.ffn_norm.weight |
| 213 | layer8.attention_residual | `add_backward` | dlayer7.ffn_residual, dlayer8.o |
| 214 | layer8.o | `linear_backward_input_hidden` | dlayer8.context |
| 215 | layer8.o | `linear_backward_weight_hidden` | dlayer8.o.weight |
| 216 | layer8.context | `attention_values_backward_p` | dlayer8.probabilities |
| 217 | layer8.context | `attention_values_backward_v` | dlayer8.v |
| 218 | layer8.probabilities | `softmax_backward` | dlayer8.scores |
| 219 | layer8.scores | `attention_scores_backward_q` | dlayer8.q_rope |
| 220 | layer8.scores | `attention_scores_backward_k` | dlayer8.k_rope |
| 221 | layer8.k_rope | `rope_backward_k` | dlayer8.k |
| 222 | layer8.q_rope | `rope_backward_q` | dlayer8.q |
| 223 | layer8.v | `linear_backward_input_kv` | dlayer8.attention_norm |
| 224 | layer8.v | `linear_backward_weight_kv` | dlayer8.v.weight |
| 225 | layer8.k | `linear_backward_input_kv` | dlayer8.attention_norm |
| 226 | layer8.k | `linear_backward_weight_kv` | dlayer8.k.weight |
| 227 | layer8.q | `linear_backward_input_hidden` | dlayer8.attention_norm |
| 228 | layer8.q | `linear_backward_weight_hidden` | dlayer8.q.weight |
| 229 | layer8.attention_norm | `rmsnorm_backward_input` | dlayer7.ffn_residual |
| 230 | layer8.attention_norm | `rmsnorm_backward_weight` | dlayer8.attention_norm.weight |
| 231 | layer7.ffn_residual | `add_backward` | dlayer7.attention_residual, dlayer7.down |
| 232 | layer7.down | `linear_backward_input_mlp_down` | dlayer7.swiglu |
| 233 | layer7.down | `linear_backward_weight_mlp_down` | dlayer7.down.weight |
| 234 | layer7.swiglu | `swiglu_backward` | dlayer7.gate, dlayer7.up |
| 235 | layer7.up | `linear_backward_input_mlp_up` | dlayer7.ffn_norm |
| 236 | layer7.up | `linear_backward_weight_mlp_up` | dlayer7.up.weight |
| 237 | layer7.gate | `linear_backward_input_mlp_up` | dlayer7.ffn_norm |
| 238 | layer7.gate | `linear_backward_weight_mlp_up` | dlayer7.gate.weight |
| 239 | layer7.ffn_norm | `rmsnorm_backward_input` | dlayer7.attention_residual |
| 240 | layer7.ffn_norm | `rmsnorm_backward_weight` | dlayer7.ffn_norm.weight |
| 241 | layer7.attention_residual | `add_backward` | dlayer6.ffn_residual, dlayer7.o |
| 242 | layer7.o | `linear_backward_input_hidden` | dlayer7.context |
| 243 | layer7.o | `linear_backward_weight_hidden` | dlayer7.o.weight |
| 244 | layer7.context | `attention_values_backward_p` | dlayer7.probabilities |
| 245 | layer7.context | `attention_values_backward_v` | dlayer7.v |
| 246 | layer7.probabilities | `softmax_backward` | dlayer7.scores |
| 247 | layer7.scores | `attention_scores_backward_q` | dlayer7.q_rope |
| 248 | layer7.scores | `attention_scores_backward_k` | dlayer7.k_rope |
| 249 | layer7.k_rope | `rope_backward_k` | dlayer7.k |
| 250 | layer7.q_rope | `rope_backward_q` | dlayer7.q |
| 251 | layer7.v | `linear_backward_input_kv` | dlayer7.attention_norm |
| 252 | layer7.v | `linear_backward_weight_kv` | dlayer7.v.weight |
| 253 | layer7.k | `linear_backward_input_kv` | dlayer7.attention_norm |
| 254 | layer7.k | `linear_backward_weight_kv` | dlayer7.k.weight |
| 255 | layer7.q | `linear_backward_input_hidden` | dlayer7.attention_norm |
| 256 | layer7.q | `linear_backward_weight_hidden` | dlayer7.q.weight |
| 257 | layer7.attention_norm | `rmsnorm_backward_input` | dlayer6.ffn_residual |
| 258 | layer7.attention_norm | `rmsnorm_backward_weight` | dlayer7.attention_norm.weight |
| 259 | layer6.ffn_residual | `add_backward` | dlayer6.attention_residual, dlayer6.down |
| 260 | layer6.down | `linear_backward_input_mlp_down` | dlayer6.swiglu |
| 261 | layer6.down | `linear_backward_weight_mlp_down` | dlayer6.down.weight |
| 262 | layer6.swiglu | `swiglu_backward` | dlayer6.gate, dlayer6.up |
| 263 | layer6.up | `linear_backward_input_mlp_up` | dlayer6.ffn_norm |
| 264 | layer6.up | `linear_backward_weight_mlp_up` | dlayer6.up.weight |
| 265 | layer6.gate | `linear_backward_input_mlp_up` | dlayer6.ffn_norm |
| 266 | layer6.gate | `linear_backward_weight_mlp_up` | dlayer6.gate.weight |
| 267 | layer6.ffn_norm | `rmsnorm_backward_input` | dlayer6.attention_residual |
| 268 | layer6.ffn_norm | `rmsnorm_backward_weight` | dlayer6.ffn_norm.weight |
| 269 | layer6.attention_residual | `add_backward` | dlayer5.ffn_residual, dlayer6.o |
| 270 | layer6.o | `linear_backward_input_hidden` | dlayer6.context |
| 271 | layer6.o | `linear_backward_weight_hidden` | dlayer6.o.weight |
| 272 | layer6.context | `attention_values_backward_p` | dlayer6.probabilities |
| 273 | layer6.context | `attention_values_backward_v` | dlayer6.v |
| 274 | layer6.probabilities | `softmax_backward` | dlayer6.scores |
| 275 | layer6.scores | `attention_scores_backward_q` | dlayer6.q_rope |
| 276 | layer6.scores | `attention_scores_backward_k` | dlayer6.k_rope |
| 277 | layer6.k_rope | `rope_backward_k` | dlayer6.k |
| 278 | layer6.q_rope | `rope_backward_q` | dlayer6.q |
| 279 | layer6.v | `linear_backward_input_kv` | dlayer6.attention_norm |
| 280 | layer6.v | `linear_backward_weight_kv` | dlayer6.v.weight |
| 281 | layer6.k | `linear_backward_input_kv` | dlayer6.attention_norm |
| 282 | layer6.k | `linear_backward_weight_kv` | dlayer6.k.weight |
| 283 | layer6.q | `linear_backward_input_hidden` | dlayer6.attention_norm |
| 284 | layer6.q | `linear_backward_weight_hidden` | dlayer6.q.weight |
| 285 | layer6.attention_norm | `rmsnorm_backward_input` | dlayer5.ffn_residual |
| 286 | layer6.attention_norm | `rmsnorm_backward_weight` | dlayer6.attention_norm.weight |
| 287 | layer5.ffn_residual | `add_backward` | dlayer5.attention_residual, dlayer5.down |
| 288 | layer5.down | `linear_backward_input_mlp_down` | dlayer5.swiglu |
| 289 | layer5.down | `linear_backward_weight_mlp_down` | dlayer5.down.weight |
| 290 | layer5.swiglu | `swiglu_backward` | dlayer5.gate, dlayer5.up |
| 291 | layer5.up | `linear_backward_input_mlp_up` | dlayer5.ffn_norm |
| 292 | layer5.up | `linear_backward_weight_mlp_up` | dlayer5.up.weight |
| 293 | layer5.gate | `linear_backward_input_mlp_up` | dlayer5.ffn_norm |
| 294 | layer5.gate | `linear_backward_weight_mlp_up` | dlayer5.gate.weight |
| 295 | layer5.ffn_norm | `rmsnorm_backward_input` | dlayer5.attention_residual |
| 296 | layer5.ffn_norm | `rmsnorm_backward_weight` | dlayer5.ffn_norm.weight |
| 297 | layer5.attention_residual | `add_backward` | dlayer4.ffn_residual, dlayer5.o |
| 298 | layer5.o | `linear_backward_input_hidden` | dlayer5.context |
| 299 | layer5.o | `linear_backward_weight_hidden` | dlayer5.o.weight |
| 300 | layer5.context | `attention_values_backward_p` | dlayer5.probabilities |
| 301 | layer5.context | `attention_values_backward_v` | dlayer5.v |
| 302 | layer5.probabilities | `softmax_backward` | dlayer5.scores |
| 303 | layer5.scores | `attention_scores_backward_q` | dlayer5.q_rope |
| 304 | layer5.scores | `attention_scores_backward_k` | dlayer5.k_rope |
| 305 | layer5.k_rope | `rope_backward_k` | dlayer5.k |
| 306 | layer5.q_rope | `rope_backward_q` | dlayer5.q |
| 307 | layer5.v | `linear_backward_input_kv` | dlayer5.attention_norm |
| 308 | layer5.v | `linear_backward_weight_kv` | dlayer5.v.weight |
| 309 | layer5.k | `linear_backward_input_kv` | dlayer5.attention_norm |
| 310 | layer5.k | `linear_backward_weight_kv` | dlayer5.k.weight |
| 311 | layer5.q | `linear_backward_input_hidden` | dlayer5.attention_norm |
| 312 | layer5.q | `linear_backward_weight_hidden` | dlayer5.q.weight |
| 313 | layer5.attention_norm | `rmsnorm_backward_input` | dlayer4.ffn_residual |
| 314 | layer5.attention_norm | `rmsnorm_backward_weight` | dlayer5.attention_norm.weight |
| 315 | layer4.ffn_residual | `add_backward` | dlayer4.attention_residual, dlayer4.down |
| 316 | layer4.down | `linear_backward_input_mlp_down` | dlayer4.swiglu |
| 317 | layer4.down | `linear_backward_weight_mlp_down` | dlayer4.down.weight |
| 318 | layer4.swiglu | `swiglu_backward` | dlayer4.gate, dlayer4.up |
| 319 | layer4.up | `linear_backward_input_mlp_up` | dlayer4.ffn_norm |
| 320 | layer4.up | `linear_backward_weight_mlp_up` | dlayer4.up.weight |
| 321 | layer4.gate | `linear_backward_input_mlp_up` | dlayer4.ffn_norm |
| 322 | layer4.gate | `linear_backward_weight_mlp_up` | dlayer4.gate.weight |
| 323 | layer4.ffn_norm | `rmsnorm_backward_input` | dlayer4.attention_residual |
| 324 | layer4.ffn_norm | `rmsnorm_backward_weight` | dlayer4.ffn_norm.weight |
| 325 | layer4.attention_residual | `add_backward` | dlayer3.ffn_residual, dlayer4.o |
| 326 | layer4.o | `linear_backward_input_hidden` | dlayer4.context |
| 327 | layer4.o | `linear_backward_weight_hidden` | dlayer4.o.weight |
| 328 | layer4.context | `attention_values_backward_p` | dlayer4.probabilities |
| 329 | layer4.context | `attention_values_backward_v` | dlayer4.v |
| 330 | layer4.probabilities | `softmax_backward` | dlayer4.scores |
| 331 | layer4.scores | `attention_scores_backward_q` | dlayer4.q_rope |
| 332 | layer4.scores | `attention_scores_backward_k` | dlayer4.k_rope |
| 333 | layer4.k_rope | `rope_backward_k` | dlayer4.k |
| 334 | layer4.q_rope | `rope_backward_q` | dlayer4.q |
| 335 | layer4.v | `linear_backward_input_kv` | dlayer4.attention_norm |
| 336 | layer4.v | `linear_backward_weight_kv` | dlayer4.v.weight |
| 337 | layer4.k | `linear_backward_input_kv` | dlayer4.attention_norm |
| 338 | layer4.k | `linear_backward_weight_kv` | dlayer4.k.weight |
| 339 | layer4.q | `linear_backward_input_hidden` | dlayer4.attention_norm |
| 340 | layer4.q | `linear_backward_weight_hidden` | dlayer4.q.weight |
| 341 | layer4.attention_norm | `rmsnorm_backward_input` | dlayer3.ffn_residual |
| 342 | layer4.attention_norm | `rmsnorm_backward_weight` | dlayer4.attention_norm.weight |
| 343 | layer3.ffn_residual | `add_backward` | dlayer3.attention_residual, dlayer3.down |
| 344 | layer3.down | `linear_backward_input_mlp_down` | dlayer3.swiglu |
| 345 | layer3.down | `linear_backward_weight_mlp_down` | dlayer3.down.weight |
| 346 | layer3.swiglu | `swiglu_backward` | dlayer3.gate, dlayer3.up |
| 347 | layer3.up | `linear_backward_input_mlp_up` | dlayer3.ffn_norm |
| 348 | layer3.up | `linear_backward_weight_mlp_up` | dlayer3.up.weight |
| 349 | layer3.gate | `linear_backward_input_mlp_up` | dlayer3.ffn_norm |
| 350 | layer3.gate | `linear_backward_weight_mlp_up` | dlayer3.gate.weight |
| 351 | layer3.ffn_norm | `rmsnorm_backward_input` | dlayer3.attention_residual |
| 352 | layer3.ffn_norm | `rmsnorm_backward_weight` | dlayer3.ffn_norm.weight |
| 353 | layer3.attention_residual | `add_backward` | dlayer2.ffn_residual, dlayer3.o |
| 354 | layer3.o | `linear_backward_input_hidden` | dlayer3.context |
| 355 | layer3.o | `linear_backward_weight_hidden` | dlayer3.o.weight |
| 356 | layer3.context | `attention_values_backward_p` | dlayer3.probabilities |
| 357 | layer3.context | `attention_values_backward_v` | dlayer3.v |
| 358 | layer3.probabilities | `softmax_backward` | dlayer3.scores |
| 359 | layer3.scores | `attention_scores_backward_q` | dlayer3.q_rope |
| 360 | layer3.scores | `attention_scores_backward_k` | dlayer3.k_rope |
| 361 | layer3.k_rope | `rope_backward_k` | dlayer3.k |
| 362 | layer3.q_rope | `rope_backward_q` | dlayer3.q |
| 363 | layer3.v | `linear_backward_input_kv` | dlayer3.attention_norm |
| 364 | layer3.v | `linear_backward_weight_kv` | dlayer3.v.weight |
| 365 | layer3.k | `linear_backward_input_kv` | dlayer3.attention_norm |
| 366 | layer3.k | `linear_backward_weight_kv` | dlayer3.k.weight |
| 367 | layer3.q | `linear_backward_input_hidden` | dlayer3.attention_norm |
| 368 | layer3.q | `linear_backward_weight_hidden` | dlayer3.q.weight |
| 369 | layer3.attention_norm | `rmsnorm_backward_input` | dlayer2.ffn_residual |
| 370 | layer3.attention_norm | `rmsnorm_backward_weight` | dlayer3.attention_norm.weight |
| 371 | layer2.ffn_residual | `add_backward` | dlayer2.attention_residual, dlayer2.down |
| 372 | layer2.down | `linear_backward_input_mlp_down` | dlayer2.swiglu |
| 373 | layer2.down | `linear_backward_weight_mlp_down` | dlayer2.down.weight |
| 374 | layer2.swiglu | `swiglu_backward` | dlayer2.gate, dlayer2.up |
| 375 | layer2.up | `linear_backward_input_mlp_up` | dlayer2.ffn_norm |
| 376 | layer2.up | `linear_backward_weight_mlp_up` | dlayer2.up.weight |
| 377 | layer2.gate | `linear_backward_input_mlp_up` | dlayer2.ffn_norm |
| 378 | layer2.gate | `linear_backward_weight_mlp_up` | dlayer2.gate.weight |
| 379 | layer2.ffn_norm | `rmsnorm_backward_input` | dlayer2.attention_residual |
| 380 | layer2.ffn_norm | `rmsnorm_backward_weight` | dlayer2.ffn_norm.weight |
| 381 | layer2.attention_residual | `add_backward` | dlayer1.ffn_residual, dlayer2.o |
| 382 | layer2.o | `linear_backward_input_hidden` | dlayer2.context |
| 383 | layer2.o | `linear_backward_weight_hidden` | dlayer2.o.weight |
| 384 | layer2.context | `attention_values_backward_p` | dlayer2.probabilities |
| 385 | layer2.context | `attention_values_backward_v` | dlayer2.v |
| 386 | layer2.probabilities | `softmax_backward` | dlayer2.scores |
| 387 | layer2.scores | `attention_scores_backward_q` | dlayer2.q_rope |
| 388 | layer2.scores | `attention_scores_backward_k` | dlayer2.k_rope |
| 389 | layer2.k_rope | `rope_backward_k` | dlayer2.k |
| 390 | layer2.q_rope | `rope_backward_q` | dlayer2.q |
| 391 | layer2.v | `linear_backward_input_kv` | dlayer2.attention_norm |
| 392 | layer2.v | `linear_backward_weight_kv` | dlayer2.v.weight |
| 393 | layer2.k | `linear_backward_input_kv` | dlayer2.attention_norm |
| 394 | layer2.k | `linear_backward_weight_kv` | dlayer2.k.weight |
| 395 | layer2.q | `linear_backward_input_hidden` | dlayer2.attention_norm |
| 396 | layer2.q | `linear_backward_weight_hidden` | dlayer2.q.weight |
| 397 | layer2.attention_norm | `rmsnorm_backward_input` | dlayer1.ffn_residual |
| 398 | layer2.attention_norm | `rmsnorm_backward_weight` | dlayer2.attention_norm.weight |
| 399 | layer1.ffn_residual | `add_backward` | dlayer1.attention_residual, dlayer1.down |
| 400 | layer1.down | `linear_backward_input_mlp_down` | dlayer1.swiglu |
| 401 | layer1.down | `linear_backward_weight_mlp_down` | dlayer1.down.weight |
| 402 | layer1.swiglu | `swiglu_backward` | dlayer1.gate, dlayer1.up |
| 403 | layer1.up | `linear_backward_input_mlp_up` | dlayer1.ffn_norm |
| 404 | layer1.up | `linear_backward_weight_mlp_up` | dlayer1.up.weight |
| 405 | layer1.gate | `linear_backward_input_mlp_up` | dlayer1.ffn_norm |
| 406 | layer1.gate | `linear_backward_weight_mlp_up` | dlayer1.gate.weight |
| 407 | layer1.ffn_norm | `rmsnorm_backward_input` | dlayer1.attention_residual |
| 408 | layer1.ffn_norm | `rmsnorm_backward_weight` | dlayer1.ffn_norm.weight |
| 409 | layer1.attention_residual | `add_backward` | dlayer0.ffn_residual, dlayer1.o |
| 410 | layer1.o | `linear_backward_input_hidden` | dlayer1.context |
| 411 | layer1.o | `linear_backward_weight_hidden` | dlayer1.o.weight |
| 412 | layer1.context | `attention_values_backward_p` | dlayer1.probabilities |
| 413 | layer1.context | `attention_values_backward_v` | dlayer1.v |
| 414 | layer1.probabilities | `softmax_backward` | dlayer1.scores |
| 415 | layer1.scores | `attention_scores_backward_q` | dlayer1.q_rope |
| 416 | layer1.scores | `attention_scores_backward_k` | dlayer1.k_rope |
| 417 | layer1.k_rope | `rope_backward_k` | dlayer1.k |
| 418 | layer1.q_rope | `rope_backward_q` | dlayer1.q |
| 419 | layer1.v | `linear_backward_input_kv` | dlayer1.attention_norm |
| 420 | layer1.v | `linear_backward_weight_kv` | dlayer1.v.weight |
| 421 | layer1.k | `linear_backward_input_kv` | dlayer1.attention_norm |
| 422 | layer1.k | `linear_backward_weight_kv` | dlayer1.k.weight |
| 423 | layer1.q | `linear_backward_input_hidden` | dlayer1.attention_norm |
| 424 | layer1.q | `linear_backward_weight_hidden` | dlayer1.q.weight |
| 425 | layer1.attention_norm | `rmsnorm_backward_input` | dlayer0.ffn_residual |
| 426 | layer1.attention_norm | `rmsnorm_backward_weight` | dlayer1.attention_norm.weight |
| 427 | layer0.ffn_residual | `add_backward` | dlayer0.attention_residual, dlayer0.down |
| 428 | layer0.down | `linear_backward_input_mlp_down` | dlayer0.swiglu |
| 429 | layer0.down | `linear_backward_weight_mlp_down` | dlayer0.down.weight |
| 430 | layer0.swiglu | `swiglu_backward` | dlayer0.gate, dlayer0.up |
| 431 | layer0.up | `linear_backward_input_mlp_up` | dlayer0.ffn_norm |
| 432 | layer0.up | `linear_backward_weight_mlp_up` | dlayer0.up.weight |
| 433 | layer0.gate | `linear_backward_input_mlp_up` | dlayer0.ffn_norm |
| 434 | layer0.gate | `linear_backward_weight_mlp_up` | dlayer0.gate.weight |
| 435 | layer0.ffn_norm | `rmsnorm_backward_input` | dlayer0.attention_residual |
| 436 | layer0.ffn_norm | `rmsnorm_backward_weight` | dlayer0.ffn_norm.weight |
| 437 | layer0.attention_residual | `add_backward` | dembedding, dlayer0.o |
| 438 | layer0.o | `linear_backward_input_hidden` | dlayer0.context |
| 439 | layer0.o | `linear_backward_weight_hidden` | dlayer0.o.weight |
| 440 | layer0.context | `attention_values_backward_p` | dlayer0.probabilities |
| 441 | layer0.context | `attention_values_backward_v` | dlayer0.v |
| 442 | layer0.probabilities | `softmax_backward` | dlayer0.scores |
| 443 | layer0.scores | `attention_scores_backward_q` | dlayer0.q_rope |
| 444 | layer0.scores | `attention_scores_backward_k` | dlayer0.k_rope |
| 445 | layer0.k_rope | `rope_backward_k` | dlayer0.k |
| 446 | layer0.q_rope | `rope_backward_q` | dlayer0.q |
| 447 | layer0.v | `linear_backward_input_kv` | dlayer0.attention_norm |
| 448 | layer0.v | `linear_backward_weight_kv` | dlayer0.v.weight |
| 449 | layer0.k | `linear_backward_input_kv` | dlayer0.attention_norm |
| 450 | layer0.k | `linear_backward_weight_kv` | dlayer0.k.weight |
| 451 | layer0.q | `linear_backward_input_hidden` | dlayer0.attention_norm |
| 452 | layer0.q | `linear_backward_weight_hidden` | dlayer0.q.weight |
| 453 | layer0.attention_norm | `rmsnorm_backward_input` | dembedding |
| 454 | layer0.attention_norm | `rmsnorm_backward_weight` | dlayer0.attention_norm.weight |
| 455 | embedding | `embedding_backward` | dembedding.weight |
