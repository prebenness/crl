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
    """Save a params dict (JAX/numpy arrays) to a .npz file.

    Handles arbitrary nesting depth by flattening keys with '/' separator.
    """
    flat = {}

    def _flatten(d, prefix=""):
        for k, v in d.items():
            key = f"{prefix}{k}" if not prefix else f"{prefix}/{k}"
            if isinstance(v, dict):
                _flatten(v, key)
            else:
                flat[key] = np.array(v)

    _flatten(params)
    np.savez(str(path), **flat)


def load_checkpoint(path):
    """Load a params dict from a .npz file. Returns nested dict of jnp arrays."""
    data = np.load(str(path), allow_pickle=True)
    params = {}
    for k in data.files:
        parts = k.split("/")
        d = params
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        val = data[k]
        # Handle legacy object arrays (dicts saved as 0-d object arrays)
        if val.dtype == object and val.shape == ():
            sub = val.item()
            if isinstance(sub, dict):
                d[parts[-1]] = {sk: jnp.array(sv) for sk, sv in sub.items()}
                continue
        d[parts[-1]] = jnp.array(val)
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


def save_checkpoint_meta(run_dir, last_epoch, best_test_acc,
                         best_checkpoint_epoch=None):
    """Write/update checkpoint metadata sidecar (checkpoints/meta.json)."""
    meta_path = checkpoint_path(run_dir, "meta.json")
    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
    meta["last_epoch"] = int(last_epoch)
    meta["best_test_acc"] = float(best_test_acc)
    if best_checkpoint_epoch is not None:
        meta["best_checkpoint_epoch"] = int(best_checkpoint_epoch)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


def load_checkpoint_meta(run_dir):
    """Read checkpoint metadata, return empty dict if missing."""
    meta_path = checkpoint_path(run_dir, "meta.json", create=False)
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return {}


def resolve_resume_checkpoint(run_dir, selection="auto"):
    """Find checkpoint file for resume. Returns (path, kind)."""
    candidates = {
        "best": checkpoint_path(run_dir, "best.npz", create=False),
        "final": checkpoint_path(run_dir, "final.npz", create=False),
    }
    order = ["best", "final"] if selection == "auto" else [selection]
    for kind in order:
        p = candidates[kind]
        if p.exists():
            return p, kind
    raise FileNotFoundError(
        f"No {selection} checkpoint found in {run_dir}"
    )


def resolve_resume_start_epoch(run_dir, checkpoint_kind,
                                default_final_epoch=0):
    """Choose start epoch based on checkpoint kind and metadata."""
    meta = load_checkpoint_meta(run_dir)
    if checkpoint_kind == "best":
        return int(meta.get("best_checkpoint_epoch", 0))
    return int(meta.get("last_epoch", default_final_epoch))


def save_config(run_dir, config_dict):
    """Write config to config.json."""
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(run_dir) / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
