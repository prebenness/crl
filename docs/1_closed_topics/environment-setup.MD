# Environment Setup

This repo uses one lightweight dependency spec (not a full `pip freeze` dump).
No environment matrix, just one path.

## Files

- `.python-version`: preferred project Python version
- `requirements.txt`: direct runtime deps + JAX CUDA13
- `scripts/bootstrap_venv.sh`: one-command environment bootstrap

## Typical command

```bash
bash scripts/bootstrap_venv.sh
```

Optional:

```bash
PY_BIN=python3.12 bash scripts/bootstrap_venv.sh
```
