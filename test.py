import numpy as np

from roundmantissa import round_mantissa
from roundmantissa.numpy import round_mantissa as round_mantissa_numpy

assert round_mantissa(0.0, 2) == 0.0
assert round_mantissa(0.50123, 2) == 0.5

np.testing.assert_allclose(
    round_mantissa_numpy(np.array([0.0, 0.50123, 0.5 + 0.25 + 0.125]), 2),
    np.array([0.0, 0.5, 0.875]),
)
