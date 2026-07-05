import os
import random

import numpy as np

from common.seeding import DEFAULT_SEED, set_global_seed


def test_set_global_seed_reproduces_python_random():
    set_global_seed(123)
    first = [random.random() for _ in range(5)]
    set_global_seed(123)
    second = [random.random() for _ in range(5)]
    assert first == second


def test_set_global_seed_reproduces_numpy_random():
    set_global_seed(123)
    first = np.random.rand(5)
    set_global_seed(123)
    second = np.random.rand(5)
    assert np.array_equal(first, second)


def test_set_global_seed_sets_pythonhashseed_env_var():
    set_global_seed(7)
    assert os.environ.get("PYTHONHASHSEED") == "7"


def test_set_global_seed_default_matches_no_argument_call():
    set_global_seed()
    assert os.environ.get("PYTHONHASHSEED") == str(DEFAULT_SEED)
