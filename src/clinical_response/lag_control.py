"""Delayed Gaussian coupling with independent prehistory."""

import numpy as np

from .score import _real_array


def finite_matrix(values):
    """Return a nonempty finite two-dimensional float64 array."""
    a = _real_array(values)
    if a.ndim != 2 or min(a.shape) < 1 or not np.isfinite(a).all():
        raise ValueError("A nonempty finite real matrix is required")
    return a


def lag_world(innovations, alpha):
    """Couple channel 1 to the previous innovation of channel 0.

    Input has shape (N+1, 7); its first row supplies independent prehistory.
    The N returned rows have no circular wraparound. For standard-normal
    independent innovations, the coupled channel retains unit variance.
    """
    bank = finite_matrix(innovations)
    if (
        bank.shape[1] != 7
        or len(bank) < 2
        or not np.isscalar(alpha)
        or not np.isfinite(alpha)
        or not 0 <= alpha <= 1
    ):
        raise ValueError("Need (N+1)-by-7 innovations and alpha in [0,1]")
    x = bank[1:].copy()
    x[:, 1] = float(alpha) * bank[:-1, 0] + np.sqrt(1 - float(alpha) ** 2) * bank[1:, 1]
    return x
