# task 1 (plain lookup) with an mlp that is never trained: its weights stay at their random init.
# attention has to solve the task by itself, so removing the mlp afterwards shouldn't matter
from tinygrad import Tensor, nn, TinyJit, Context
from dataset import lookup_dataset
from model import Model
Tensor.manual_seed(0) # reproducible training runs

xtrain, xtest, ytrain, ytest = lookup_dataset()
model = Model(mlp=True)
mlp_params = nn.state.get_parameters(model.layers[-1].mlp)
for p in mlp_params: p.requires_grad = False
opt = nn.optim.Adam([p for p in nn.state.get_parameters(model) if not any(p is m for m in mlp_params)], lr=1e-3)

def test_acc() -> float: return ((model(xtest).argmax(axis=-1) == ytest).mean() * 100).item()

@TinyJit
@Context(TRAINING=1)
def train_step() -> Tensor:
  opt.zero_grad()
  samples = Tensor.randint(512, high=xtrain.shape[0])
  loss = model(xtrain[samples]).sparse_categorical_crossentropy(ytrain[samples]).backward()
  return loss.realize(*opt.schedule_step())

if __name__ == "__main__":
  for i in range(8000):
    loss = train_step()
    if i % 1000 == 999: print(f"loss and val acc at step {i+1}: {loss.item():.2f}, {test_acc():.2f}%")

  block = model.layers[-1]
  mlp = block.mlp
  print(f"frozen random mlp on: {test_acc():.2f}%")
  block.mlp = None
  print(f"mlp removed:          {test_acc():.2f}%")
  block.mlp = mlp
