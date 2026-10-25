import gzip
import os
import urllib.request

import numpy as np

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
BASE_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def download():
    os.makedirs(DATASET_DIR, exist_ok=True)
    for name in FILES.values():
        path = os.path.join(DATASET_DIR, name)
        if not os.path.exists(path):
            print(f"downloading {name}")
            urllib.request.urlretrieve(BASE_URL + name, path)


def _read_idx(name):
    with gzip.open(os.path.join(DATASET_DIR, name), "rb") as f:
        buf = f.read()
    # header: 2 zero bytes, dtype code (0x08 = uint8), ndim, then big-endian uint32 dims
    ndim = buf[3]
    dims = np.frombuffer(buf, dtype=">u4", count=ndim, offset=4)
    return np.frombuffer(buf, dtype=np.uint8, offset=4 + 4 * ndim).reshape(dims)


def load(val_frac=0.1, seed=0):
    """Returns (x_train, y_train), (x_val, y_val), (x_test, y_test).

    x: float32 (N, 784) in [0, 1]; y: int64 (N,) class indices.
    """
    download()
    x = _read_idx(FILES["train_images"]).reshape(-1, 784).astype(np.float32) / 255.0
    y = _read_idx(FILES["train_labels"]).astype(np.int64)
    x_test = _read_idx(FILES["test_images"]).reshape(-1, 784).astype(np.float32) / 255.0
    y_test = _read_idx(FILES["test_labels"]).astype(np.int64)

    perm = np.random.default_rng(seed).permutation(len(x))
    n_val = int(len(x) * val_frac)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    return (x[train_idx], y[train_idx]), (x[val_idx], y[val_idx]), (x_test, y_test)
