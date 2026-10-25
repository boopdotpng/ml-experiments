# usage: python infer.py checkpoints/task2.safetensors a2f5c9d3f [...]
#        python infer.py checkpoints/task3.safetensors a2f5c9d3fc [...]   (also task4, task4_one_head)
import sys
from tinygrad import Tensor, nn
from dataset import encode, decode
from model import Model, emb_dim

path, queries = sys.argv[1], sys.argv[2:]
state = nn.state.safe_load(path)
# task 2 checkpoints have mlp weights and one more position (two query keys)
seq_len = state["pos_emb.weight"].shape[0]
# task 4 checkpoints have two heads per layer: v_proj writes emb_dim numbers per head
n_heads = state["layers.0.attn.v_proj.weight"].shape[0] // emb_dim
model = Model(mlp=any(".mlp." in k for k in state), seq_len=seq_len, n_heads=n_heads)
nn.state.load_state_dict(model, state, verbose=False)

for q in queries:
  assert len(q) == seq_len, f"{q!r}: need 4 key/value pairs plus {seq_len-8} query key(s), e.g. {'a2f5c9d3fc'[:seq_len]}"
  print(f"{q} -> {decode([model(Tensor([encode(q)])).argmax(-1).item()])}")
