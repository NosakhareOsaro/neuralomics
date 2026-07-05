"""Central random-seed policy for NeuralOmics.

Policy: every module's training entry point calls `set_global_seed(seed)`
exactly once, before any data loading or model construction, using the seed
value from that run's config file rather than a value hardcoded in code.
This makes the seed an explicit, MLflow-logged experiment parameter instead
of an implicit default, and keeps the seeding logic itself in one place
rather than duplicated per module.
"""

from __future__ import annotations

import os
import random

import numpy as np

DEFAULT_SEED = 42


def set_global_seed(seed: int = DEFAULT_SEED) -> None:
    """Seed Python, NumPy, and (if installed) PyTorch RNGs.

    PyTorch is an optional, per-module dependency rather than a base one, so
    it is imported lazily and skipped if the calling module hasn't pulled it
    in.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
