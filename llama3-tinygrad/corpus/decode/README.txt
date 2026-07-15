CORPUS LAYOUT
=============

raw_graph.txt
  The only graph that should feed a TTIR matcher/lowerer. It is the complete
  lazy Tensor UOp DAG before scheduling and all kernel/codegen optimization.

manifest.txt
  Actual compute-launch order plus concise semantic names, dependencies, output
  storage, reduction witnesses, and links to pristine graph anchors.

kernels/*.txt
  One annotated file per scheduled compute launch. Each has a concise post-rangeify
  boundary witness and a pre-transform semantic slice. Do not lower the witness.

Regenerate from the repository root with:
  $(realpath ../.venv)/bin/python tools/generate_uop_corpus.py

The generator sets DEV=CPU as a storage tag, enters ALLOW_DEVICE_USAGE=0, uses only Tensor.empty
buffers, and never calls realize, compile, a renderer, TinyJit, or a runtime.
