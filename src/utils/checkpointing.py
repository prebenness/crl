"""Run management: directories, logging, checkpointing.

Shared utilities used by both differentiable_mdl.py and colored_mnist.py.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import jax.numpy as jnp


class TeeLogger:
    """Mirrors stdout to both the terminal and a log file simultaneously."""

    def __init__(self, log_path, mode="w"):
        self._log_path = log_path
        self._mode = mode
        self._file = None
        self._orig = None

    def __enter__(self):
        # Line-buffer the sidecar log so long batch jobs do not accumulate
        # buffered output.
        self._file = open(self._log_path, self._mode, buffering=1)
        self._orig = sys.stdout
        sys.stdout = self
        return self

    def __exit__(self, *_):
        sys.stdout = self._orig
        self._file.close()

    def write(self, data):
        self._orig.write(data)
        self._file.write(data)
        # Keep the on-disk log tail-able during long runs.
        self.flush()
        return len(data)

    def flush(self):
        self._orig.flush()
        self._file.flush()

    def fileno(self):
        return self._orig.fileno()


def save_checkpoint(params, path):
    """Save a params dict (JAX/numpy arrays) to a .npz file."""
    flat = {}
    for k, v in params.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                flat[f"{k}/{k2}"] = np.array(v2)
        else:
            flat[k] = np.array(v)
    np.savez(str(path), **flat)


def load_checkpoint(path):
    """Load a params dict from a .npz file. Returns dict of jnp arrays."""
    data = np.load(str(path))
    params = {}
    for k in data.files:
        if "/" in k:
            outer, inner = k.split("/", 1)
            if outer not in params:
                params[outer] = {}
            params[outer][inner] = jnp.array(data[k])
        else:
            params[k] = jnp.array(data[k])
    return params


def utc_timestamp() -> str:
    """Return a compact UTC timestamp safe for filenames."""
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())


def make_experiment_dir(experiment: str, run_name: str,
                        results_root: str = "results") -> Path:
    """Create and return a unique run directory under results/<experiment>/."""
    base_dir = Path(results_root) / experiment / run_name
    run_dir = base_dir
    suffix = 1
    while run_dir.exists():
        run_dir = Path(f"{base_dir}_r{suffix}")
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def checkpoint_path(run_dir, filename: str, create: bool = True) -> Path:
    """Return run_dir/checkpoints/<filename>, creating the directory if needed."""
    ckpt_dir = Path(run_dir) / "checkpoints"
    if create:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir / filename


def save_results(run_dir, results_dict):
    """Write final metrics to results.json."""
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    clean = {
        k: (v.item() if hasattr(v, "item") else v)
        for k, v in results_dict.items()
    }
    with open(Path(run_dir) / "results.json", "w") as f:
        json.dump(clean, f, indent=2)


def save_config(run_dir, config_dict):
    """Write config to config.json."""
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(run_dir) / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
