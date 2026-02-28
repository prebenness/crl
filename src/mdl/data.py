"""a^n b^n language modeling data, matching Lan et al. (2024) exactly.

PCFG:  S -> aSb  (prob 1-p)  |  eps  (prob p),  with p = 0.3.

Strings are wrapped with start/end symbol '#'.
Network sees one symbol at a time and predicts the next symbol.

Alphabet: {#, a, b} encoded as integers 0, 1, 2.
"""

import numpy as np
import jax.numpy as jnp


SYMBOL_HASH = 0
SYMBOL_A = 1
SYMBOL_B = 2
NUM_SYMBOLS = 3


def generate_anbn_strings(
    num_strings: int,
    p: float = 0.3,
    seed: int = 0,
) -> list[list[int]]:
    """Sample strings from the a^n b^n PCFG.

    Each string is a list of symbol indices: [#, a, a, ..., b, b, ..., #].
    The PCFG generates length n with probability p * (1-p)^n.
    """
    rng = np.random.RandomState(seed)
    strings = []
    for _ in range(num_strings):
        # Sample n from geometric distribution
        # P(n) = p * (1-p)^n, so n ~ Geometric(p) shifted to start at 0
        n = 0
        while rng.rand() > p:
            n += 1
        # Build string: # a^n b^n #
        s = [SYMBOL_HASH] + [SYMBOL_A] * n + [SYMBOL_B] * n + [SYMBOL_HASH]
        strings.append(s)
    return strings


def make_anbn_dataset(
    num_strings: int = 1000,
    p: float = 0.3,
    seed: int = 0,
    max_n_train: int | None = None,
) -> tuple[list[list[int]], list[list[int]]]:
    """Generate training strings as (input_seq, target_seq) pairs.

    For language modeling, input is s[:-1] and target is s[1:].

    If max_n_train is specified, only strings with n <= max_n_train are kept
    (Lan et al. use all sampled strings, so leave None for faithful reproduction).

    Returns:
        inputs: list of int arrays, each of shape (seq_len,)
        targets: list of int arrays, each of shape (seq_len,)
    """
    strings = generate_anbn_strings(num_strings, p=p, seed=seed)
    inputs = []
    targets = []
    for s in strings:
        if max_n_train is not None:
            n = s.count(SYMBOL_A)
            if n > max_n_train:
                continue
        inputs.append(s[:-1])
        targets.append(s[1:])
    return inputs, targets


def make_anbn_fixed_n(n: int) -> tuple[list[int], list[int]]:
    """Make a single a^n b^n string for testing.

    Returns (input_seq, target_seq).
    """
    s = [SYMBOL_HASH] + [SYMBOL_A] * n + [SYMBOL_B] * n + [SYMBOL_HASH]
    return s[:-1], s[1:]


def make_test_set(max_n: int = 1500) -> tuple[list[list[int]], list[list[int]]]:
    """Test set: all a^n b^n strings for 1 <= n <= max_n.

    Matches Lan et al.: "test set consisted of all a^n b^n strings
    with 1 <= n <= 1500".
    """
    inputs = []
    targets = []
    for n in range(1, max_n + 1):
        inp, tgt = make_anbn_fixed_n(n)
        inputs.append(inp)
        targets.append(tgt)
    return inputs, targets


def make_validation_set(
    train_max_n: int, val_max_n: int = 71, val_min_n: int | None = None,
) -> tuple[list[list[int]], list[list[int]]]:
    """Validation set: all strings in a held-out n-range.

    By default, this uses all strings with `train_max_n < n <= val_max_n`.
    If `val_min_n` is provided, the lower bound is clamped to at least that
    value. This is useful when reproducing Lan et al., where the validation
    set was the fixed range `22 <= n <= 71`.
    """
    inputs = []
    targets = []
    start_n = train_max_n + 1
    if val_min_n is not None:
        start_n = max(start_n, int(val_min_n))
    for n in range(start_n, val_max_n + 1):
        inp, tgt = make_anbn_fixed_n(n)
        inputs.append(inp)
        targets.append(tgt)
    return inputs, targets


def compute_target_probs(p: float = 0.3) -> dict:
    """Compute the theoretically optimal output probabilities.

    At each time step the optimal network outputs:
    - After #:   P(a) = 1-p, P(#) = p, P(b) = 0
    - During a phase: P(a) = 1-p, P(b) = p, P(#) = 0
    - During b phase (not last b): P(b) = 1, P(a) = 0, P(#) = 0
    - Last b (count balanced): P(#) = 1, P(a) = 0, P(b) = 0
    """
    return {
        "start": np.array([p, 1 - p, 0.0], dtype=np.float32),
        "a_phase": np.array([0.0, 1 - p, p], dtype=np.float32),
        "b_phase": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        "last_b": np.array([1.0, 0.0, 0.0], dtype=np.float32),
    }


def sequences_to_padded_arrays(
    inputs: list[list[int]],
    targets: list[list[int]],
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Pad variable-length sequences to arrays for batched processing.

    Returns:
        x: int32 (N, max_len) input token indices
        y: int32 (N, max_len) target token indices
        mask: float32 (N, max_len) mask (1 where valid, 0 where padded)
    """
    max_len = max(len(s) for s in inputs)
    N = len(inputs)
    x = np.zeros((N, max_len), dtype=np.int32)
    y = np.zeros((N, max_len), dtype=np.int32)
    mask = np.zeros((N, max_len), dtype=np.float32)
    for i, (inp, tgt) in enumerate(zip(inputs, targets)):
        L = len(inp)
        x[i, :L] = inp
        y[i, :L] = tgt
        mask[i, :L] = 1.0
    return jnp.array(x), jnp.array(y), jnp.array(mask)
