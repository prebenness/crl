"""Lan et al. (2024) MDL coding scheme for rational weights.

Implements the self-delimiting integer code E(n) and the per-weight
codelength l(w) exactly as described in the paper, following Li & Vitanyi.

References:
    - Lan et al. (2024), Section 3.2 / Appendix encoding scheme
    - Li & Vitanyi (2008), Chapter 1.4
"""

import math
from fractions import Fraction

import numpy as np
import jax.numpy as jnp


def integer_code_length(n: int) -> int:
    """Bit-length |E(n)| of the self-delimiting integer code for n >= 0.

    E(n) = 1^{k(n)} 0 bin_{k(n)}(n)
    |E(n)| = 2*k(n) + 1, where k(n) = ceil(log2(n+1)).

    Examples from Lan et al.:
        E(0): k=0, |E|=1   (just the '0' separator)
        E(1): k=1, |E|=3   (1 0 1)
        E(2): k=2, |E|=5   (11 0 10)
        E(5): k=3, |E|=7   (111 0 101)
    """
    assert n >= 0 and isinstance(n, int)
    k = math.ceil(math.log2(n + 1)) if n > 0 else 0
    return 2 * k + 1


def rational_codelength(w: Fraction) -> int:
    """Per-weight codelength l(w) in bits for a rational weight w = s * n/m.

    l(w) = 1 (sign) + |E(n)| + |E(m)|

    where w = s * n/m is in canonical reduced form with m >= 1, gcd(n,m)=1.
    """
    if w == 0:
        # 0 = +0/1: sign(1) + E(0) + E(1)
        return 1 + integer_code_length(0) + integer_code_length(1)
    n = abs(w.numerator)
    m = abs(w.denominator)
    return 1 + integer_code_length(n) + integer_code_length(m)


def build_rational_grid(n_max: int, m_max: int) -> list[Fraction]:
    """Build the finite set S of reduced rationals w = +/- n/m.

    S = {s * n/m : s in {-1,+1}, 0 <= n <= n_max, 1 <= m <= m_max, gcd(n,m)=1}
    with duplicates removed (e.g. 0/1 = 0/2 etc. are the same).

    Returns sorted list of unique Fractions.
    """
    rationals = set()
    for m in range(1, m_max + 1):
        for n in range(0, n_max + 1):
            f = Fraction(n, m)
            rationals.add(f)
            if n > 0:
                rationals.add(-f)
    return sorted(rationals)


def grid_values_and_codelengths(
    n_max: int, m_max: int
) -> tuple[np.ndarray, np.ndarray]:
    """Build grid S and precompute codelengths.

    Returns:
        values: float32 array of shape (M,) with the rational grid values
        codelengths: float32 array of shape (M,) with l(s_m) for each element
    """
    grid = build_rational_grid(n_max, m_max)
    values = np.array([float(w) for w in grid], dtype=np.float32)
    codelengths = np.array(
        [rational_codelength(w) for w in grid], dtype=np.float32
    )
    return values, codelengths
