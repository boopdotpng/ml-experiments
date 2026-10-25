import numpy as np

from data import load

(x_train, y_train), (x_val, y_val), (x_test, y_test) = load()
print("train", x_train.shape, y_train.shape)
print("val  ", x_val.shape, y_val.shape)
print("test ", x_test.shape, y_test.shape)

rng = np.random.default_rng(0)

# rough pytorch implementation
class Model:
  def __init__(self):
    self.lin1 = np.random()
    pass
  def forward(self, x):
    pass
  def backward(self):
    pass

def init_params():
    # TODO: weights/biases
    return {}


def forward(params, x):
    # TODO: return logits (and whatever cache backward needs)
    raise NotImplementedError


def backward(params, cache, y):
    # TODO: return loss and grads
    raise NotImplementedError


def accuracy(params, x, y):
    logits, _ = forward(params, x)
    return (logits.argmax(axis=1) == y).mean()


def main():
    params = init_params()
    # TODO: training loop over minibatches of (x_train, y_train), eval on (x_val, y_val)


if __name__ == "__main__":
    main()
