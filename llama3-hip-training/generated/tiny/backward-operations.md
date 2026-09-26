# Generated backward execution order

Each gradient is accumulated with `+=`. The seed is `dLoss = 1`.

| Step | Forward operation | Backward kernel | Gradients written |
| --- | --- | --- | --- |
| 1 | loss | `mean_backward` | dtoken_losses |
| 2 | token_losses | `cross_entropy_backward` | dlm_head |
| 3 | lm_head | `linear_backward_input_lm_head` | dfinal_norm |
| 4 | lm_head | `linear_backward_weight_lm_head` | dembedding.weight |
| 5 | final_norm | `rmsnorm_backward_input` | dlayer1.ffn_residual |
| 6 | final_norm | `rmsnorm_backward_weight` | dfinal_norm.weight |
| 7 | layer1.ffn_residual | `add_backward` | dlayer1.attention_residual, dlayer1.down |
| 8 | layer1.down | `linear_backward_input_mlp_down` | dlayer1.swiglu |
| 9 | layer1.down | `linear_backward_weight_mlp_down` | dlayer1.down.weight |
| 10 | layer1.swiglu | `swiglu_backward` | dlayer1.gate, dlayer1.up |
| 11 | layer1.up | `linear_backward_input_mlp_up` | dlayer1.ffn_norm |
| 12 | layer1.up | `linear_backward_weight_mlp_up` | dlayer1.up.weight |
| 13 | layer1.gate | `linear_backward_input_mlp_up` | dlayer1.ffn_norm |
| 14 | layer1.gate | `linear_backward_weight_mlp_up` | dlayer1.gate.weight |
| 15 | layer1.ffn_norm | `rmsnorm_backward_input` | dlayer1.attention_residual |
| 16 | layer1.ffn_norm | `rmsnorm_backward_weight` | dlayer1.ffn_norm.weight |
| 17 | layer1.attention_residual | `add_backward` | dlayer0.ffn_residual, dlayer1.o |
| 18 | layer1.o | `linear_backward_input_hidden` | dlayer1.context |
| 19 | layer1.o | `linear_backward_weight_hidden` | dlayer1.o.weight |
| 20 | layer1.context | `attention_values_backward_p` | dlayer1.probabilities |
| 21 | layer1.context | `attention_values_backward_v` | dlayer1.v |
| 22 | layer1.probabilities | `softmax_backward` | dlayer1.scores |
| 23 | layer1.scores | `attention_scores_backward_q` | dlayer1.q_rope |
| 24 | layer1.scores | `attention_scores_backward_k` | dlayer1.k_rope |
| 25 | layer1.k_rope | `rope_backward_k` | dlayer1.k |
| 26 | layer1.q_rope | `rope_backward_q` | dlayer1.q |
| 27 | layer1.v | `linear_backward_input_kv` | dlayer1.attention_norm |
| 28 | layer1.v | `linear_backward_weight_kv` | dlayer1.v.weight |
| 29 | layer1.k | `linear_backward_input_kv` | dlayer1.attention_norm |
| 30 | layer1.k | `linear_backward_weight_kv` | dlayer1.k.weight |
| 31 | layer1.q | `linear_backward_input_hidden` | dlayer1.attention_norm |
| 32 | layer1.q | `linear_backward_weight_hidden` | dlayer1.q.weight |
| 33 | layer1.attention_norm | `rmsnorm_backward_input` | dlayer0.ffn_residual |
| 34 | layer1.attention_norm | `rmsnorm_backward_weight` | dlayer1.attention_norm.weight |
| 35 | layer0.ffn_residual | `add_backward` | dlayer0.attention_residual, dlayer0.down |
| 36 | layer0.down | `linear_backward_input_mlp_down` | dlayer0.swiglu |
| 37 | layer0.down | `linear_backward_weight_mlp_down` | dlayer0.down.weight |
| 38 | layer0.swiglu | `swiglu_backward` | dlayer0.gate, dlayer0.up |
| 39 | layer0.up | `linear_backward_input_mlp_up` | dlayer0.ffn_norm |
| 40 | layer0.up | `linear_backward_weight_mlp_up` | dlayer0.up.weight |
| 41 | layer0.gate | `linear_backward_input_mlp_up` | dlayer0.ffn_norm |
| 42 | layer0.gate | `linear_backward_weight_mlp_up` | dlayer0.gate.weight |
| 43 | layer0.ffn_norm | `rmsnorm_backward_input` | dlayer0.attention_residual |
| 44 | layer0.ffn_norm | `rmsnorm_backward_weight` | dlayer0.ffn_norm.weight |
| 45 | layer0.attention_residual | `add_backward` | dembedding, dlayer0.o |
| 46 | layer0.o | `linear_backward_input_hidden` | dlayer0.context |
| 47 | layer0.o | `linear_backward_weight_hidden` | dlayer0.o.weight |
| 48 | layer0.context | `attention_values_backward_p` | dlayer0.probabilities |
| 49 | layer0.context | `attention_values_backward_v` | dlayer0.v |
| 50 | layer0.probabilities | `softmax_backward` | dlayer0.scores |
| 51 | layer0.scores | `attention_scores_backward_q` | dlayer0.q_rope |
| 52 | layer0.scores | `attention_scores_backward_k` | dlayer0.k_rope |
| 53 | layer0.k_rope | `rope_backward_k` | dlayer0.k |
| 54 | layer0.q_rope | `rope_backward_q` | dlayer0.q |
| 55 | layer0.v | `linear_backward_input_kv` | dlayer0.attention_norm |
| 56 | layer0.v | `linear_backward_weight_kv` | dlayer0.v.weight |
| 57 | layer0.k | `linear_backward_input_kv` | dlayer0.attention_norm |
| 58 | layer0.k | `linear_backward_weight_kv` | dlayer0.k.weight |
| 59 | layer0.q | `linear_backward_input_hidden` | dlayer0.attention_norm |
| 60 | layer0.q | `linear_backward_weight_hidden` | dlayer0.q.weight |
| 61 | layer0.attention_norm | `rmsnorm_backward_input` | dembedding |
| 62 | layer0.attention_norm | `rmsnorm_backward_weight` | dlayer0.attention_norm.weight |
| 63 | embedding | `embedding_backward` | dembedding.weight |
