"""Vision Transformer with Physics-Informed Graph Learning
for Climate-Adaptive Crop Disease Forecasting.

Importing this package pins HuggingFace ``transformers`` to its PyTorch backend
*before* transformers is ever imported. Without this, transformers probes for
TensorFlow and Flax at import time; this machine has a JAX build that expects
numpy >= 2 while torch 2.2 pins numpy < 2, so that probe raises
``AttributeError: module 'numpy.dtypes' has no attribute 'StringDType'`` and
takes the whole import down. We only ever use the torch backend, so the probe is
pure liability.
"""
from __future__ import annotations

import os

os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_JAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
# Keep tokenizer threads from fighting the dataloader workers on Windows.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

__version__ = "0.1.0"
