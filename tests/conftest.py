import numpy as np
import pytest

rng = np.random.default_rng(42)


@pytest.fixture(scope="session")
def RS():
    return 42
