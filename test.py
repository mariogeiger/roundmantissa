import numpy as np

from roundmantissa import round_mantissa
from roundmantissa.numpy import round_mantissa as round_mantissa_numpy
from roundmantissa.numpy import logspace as logspace_numpy

assert round_mantissa(5.0, 0) == 4.0
assert round_mantissa(5.0, 1) == 4.0
assert round_mantissa(5.0, 2) == 5.0
assert round_mantissa(0.0, 2) == 0.0
assert round_mantissa(0.50123, 2) == 0.5

np.testing.assert_allclose(
    round_mantissa_numpy(np.array([0.0, 0.50123, 0.5 + 0.25 + 0.125]), 2),
    np.array([0.0, 0.5, 0.875]),
)

start, stop, n = 2, 4, 3
np.testing.assert_allclose(
    np.unique(
        round_mantissa_numpy(
            np.logspace(np.log10(2**start), np.log10(2**stop), 1000),
            n,
        )
    ),
    logspace_numpy(start, stop, n),
)
