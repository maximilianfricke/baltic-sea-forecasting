"""Raw records and explicitly scoped paired-launch summaries."""

# Standard library
import argparse
import hashlib
import json
import random
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
        "std": statistics.pstdev(values),
    }


def summarize(directory, repeats):
    if repeats < 1:
        raise ValueError("At least one launch is required")
    runs = [
        json.loads((Path(directory) / f"repeat-{i}.json").read_text())
        for i in range(repeats)
    ]
    if not runs or any(r["status"] != "completed" for r in runs):
        raise ValueError("All requested launches must complete")
    controls = runs[0]["controls_sha256"]
    if any(r["controls_sha256"] != controls for r in runs):
        raise ValueError("Controls differ between launches")
    for run in runs:
        if any(
            not rank["passed"]
            for checks in run["validation"].values()
            for rank in checks
        ):
            raise ValueError("Original-model equivalence failed")
        if any(
            rank["cpu_final"]["main"] != rank["cpu_ready"]["main"]
            for rank in run["ranks"]
        ):
            raise ValueError("CPU main-thread placement changed")
    backends = list(runs[0]["cases"])
    medians = {
        b: [r["cases"][b]["summary"]["model_wall_ms"]["median"] for r in runs]
        for b in backends
    }
    result = {
        "status": "completed",
        "world_size": runs[0]["controls"]["world_size"],
        "independent_launches": repeats,
        "controls_sha256": controls,
        "per_launch_model_wall_medians_ms": medians,
        "all_original_equivalence_checks_passed": True,
        "performance_valid": all(r["performance_valid"] for r in runs),
        "timer": (
            "Cached/live model wall step after H2D/readiness; "
            "input and setup excluded"
        ),
    }
    if set(backends) == {"p2p", "physicsnemo"}:
        result["physicsnemo_speedup_over_p2p"] = statistics.median(
            medians["p2p"]
        ) / statistics.median(medians["physicsnemo"])
        result["paired_speedups"] = [
            a / b
            for a, b in zip(medians["p2p"], medians["physicsnemo"], strict=True)
        ]
        if repeats > 1:
            rng = random.Random(42)
            estimates = []
            for _ in range(4000):
                ids = [rng.randrange(repeats) for _ in range(repeats)]
                estimates.append(
                    statistics.median(medians["p2p"][i] for i in ids)
                    / statistics.median(medians["physicsnemo"][i] for i in ids)
                )
            estimates.sort()
            result["paired_launch_bootstrap_ci95"] = [
                estimates[100],
                estimates[3899],
            ]
            result["uncertainty_scope"] = (
                "Independent paired launches only; "
                "small repeat counts limit resolution"
            )
    write_json(Path(directory) / "comparison.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--repeats", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.directory, args.repeats), indent=2))


if __name__ == "__main__":
    main()
