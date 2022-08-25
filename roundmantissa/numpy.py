import numpy as np


def round_mantissa(x: np.ndarray, n: int) -> np.ndarray:
    """Round number

    Args:
        x: number to round
        n: number of mantissa digits to keep

    Returns:
        rounded number

    Example:
        >>> round_mantissa(0.5 + 0.25 + 0.125, 0)
        1.0

        >>> round_mantissa(0.5 + 0.25 + 0.125, 2)
        0.875
    """

    def fn(x):
        s = np.sign(x)
        x = np.abs(x)
        a = np.floor(np.log2(x))
        x = x / 2**a
        assert np.all(1.0 <= x) and np.all(x < 2.0), x
        x = np.round(x * 2**n) / 2**n
        x = x * 2**a
        return s * x

    return np.where(x == 0.0, 0.0, fn(np.where(x == 0.0, 1.0, x)))


def logspace(start: int, stop: int, n: int) -> np.ndarray:
    """Logarithmically spaced array between ``2**start`` and ``2**stop``.

    Args:
        start: starting exponent in base 2
        stop: ending exponent in base 2
        n: number of mantissa digits to keep

    Returns:
        logarithmically spaced array
    """
    m = np.stack(
        np.meshgrid(*[np.array([0.0, 1.0])] * n, indexing="ij"), axis=-1
    ).reshape(-1, n)
    m = 1.0 + np.sum(m * 0.5 ** (np.arange(n) + 1.0), axis=-1)
    x = m * 2 ** np.arange(start, stop)[:, None]
    return np.concatenate([x.reshape(-1), [2**stop]])
