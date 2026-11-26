"""Small result helpers for the first P2P experiment."""

# Standard library
import hashlib
import json
import statistics
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest_tensor(tensor):
    tensor = tensor.detach().cpu().contiguous()
    return hashlib.sha256(
        str((tuple(tensor.shape), str(tensor.dtype))).encode()
        + tensor.numpy().tobytes()
    ).hexdigest()


def distribution(values):
    values = sorted(values)
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": values[0],
        "max": values[-1],
        "p95": values[min(len(values) - 1, int(0.95 * len(values)))],
    }
