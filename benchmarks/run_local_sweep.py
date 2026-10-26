"""Run the exploratory NJORD single-accelerator benchmark matrix."""

# Standard library
import argparse
import os
import subprocess
import sys
from pathlib import Path


COMMON = {
    "hidden_dim": 64,
    "processor_layers": 3,
    "batch_size": 1,
    "num_workers": 4,
    "ar_steps_train": 1,
    "ar_steps_eval": 1,
    "precision": "32",
    "checkpointing": False,
    "workload": "training",
    "ensemble_size": 1,
}


def _variants(max_workers=None):
    common = dict(COMMON)
    if max_workers is not None:
        common["num_workers"] = min(common["num_workers"], max_workers)
    variants = []
    for hidden_dim in (16, 32, 64, 128, 256):
        for checkpointing in (False, True):
            variants.append(
                (
                    "hidden",
                    f"hidden-h{hidden_dim}-ckpt-"
                    f"{'on' if checkpointing else 'off'}",
                    {
                        "hidden_dim": hidden_dim,
                        "checkpointing": checkpointing,
                    },
                )
            )
    worker_values = (0, 1, 2, 4, 8)
    if max_workers is not None:
        worker_values = tuple(
            workers for workers in worker_values if workers <= max_workers
        )
    for workers in worker_values:
        variants.append(
            ("workers", f"workers-w{workers}", {"num_workers": workers})
        )
    for layers in (1, 3, 6):
        variants.append(
            (
                "layers",
                f"layers-p{layers}",
                {"processor_layers": layers},
            )
        )
    for batch_size in (1, 2, 4):
        variants.append(
            (
                "batch",
                f"batch-b{batch_size}",
                {"batch_size": batch_size},
            )
        )
    for ar_steps in (1, 2, 4):
        variants.append(
            (
                "ar",
                f"ar-steps-{ar_steps}",
                {"ar_steps_train": ar_steps},
            )
        )
    for precision in ("32", "16-mixed"):
        variants.append(
            (
                "precision",
                f"precision-{precision}",
                {"precision": precision},
            )
        )
    for ensemble_size in (1, 5, 10):
        variants.append(
            (
                "inference",
                f"inference-ensemble-{ensemble_size}",
                {
                    "workload": "inference",
                    "ensemble_size": ensemble_size,
                },
            )
        )
    for horizon in (1, 2, 4):
        variants.append(
            (
                "inference",
                f"inference-horizon-{horizon}",
                {
                    "workload": "inference",
                    "ar_steps_train": horizon,
                    "ar_steps_eval": horizon,
                },
            )
        )
    # Preserve order but run identical configurations only once.
    unique = []
    seen = set()
    for group, label, changes in variants:
        config = {**common, **changes}
        key = tuple(sorted(config.items()))
        if key not in seen:
            unique.append((group, label, config))
            seen.add(key)
    return unique


def _command(config, output_path, label, warmup, steps):
    hidden = str(config["hidden_dim"])
    command = [
        sys.executable,
        "-m",
        "neural_lam.train_model",
        "--config_path",
        "data/global_ocean_1.yaml",
        "--model",
        "graph_efm",
        "--graph",
        "global_cluster_graph",
        "--hidden_dim",
        hidden,
        "--hidden_dim_grid",
        hidden,
        "--hidden_dim_mesh_nodes",
        hidden,
        "--hidden_dim_edge",
        hidden,
        "--processor_layers",
        str(config["processor_layers"]),
        "--encoder_processor_layers",
        "1",
        "--prior_processor_layers",
        "1",
        "--batch_size",
        str(config["batch_size"]),
        "--precision",
        str(config["precision"]),
        "--epochs",
        "20",
        "--ar_steps_train",
        str(config["ar_steps_train"]),
        "--ar_steps_eval",
        str(config["ar_steps_eval"]),
        "--num_workers",
        str(config["num_workers"]),
        "--devices",
        "auto",
        "--n_example_pred",
        "0",
        "--benchmark_warmup_steps",
        str(warmup),
        "--benchmark_steps",
        str(steps),
        "--benchmark_label",
        label,
        "--benchmark_output",
        str(output_path),
    ]
    if config["workload"] == "inference":
        command.extend(
            [
                "--benchmark_workload",
                "inference",
                "--benchmark_inference_ensemble_size",
                str(config["ensemble_size"]),
            ]
        )
    if config["checkpointing"]:
        command.append("--grad_checkpointing")
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group",
        choices=(
            "all",
            "hidden",
            "workers",
            "layers",
            "batch",
            "ar",
            "precision",
            "inference",
        ),
        default="all",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results/local-sweep"),
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Limit DataLoader workers to the allocated CPU cores.",
    )
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.setdefault("MPLCONFIGDIR", "/private/tmp/njord-mpl-cache")
    environment.setdefault("XDG_CACHE_HOME", "/private/tmp/njord-xdg")
    environment.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

    failures = []
    variants = [
        variant
        for variant in _variants(max_workers=args.max_workers)
        if args.group == "all" or variant[0] == args.group
    ]
    for index, (_, label, config) in enumerate(variants, start=1):
        output_path = args.output_dir / f"{label}.json"
        log_path = args.output_dir / f"{label}.log"
        if output_path.exists() and not args.rerun:
            print(f"[{index}/{len(variants)}] skip {label}")
            continue
        print(f"[{index}/{len(variants)}] run {label}", flush=True)
        with log_path.open("w", encoding="utf-8") as log_file:
            completed = subprocess.run(
                _command(
                    config,
                    output_path,
                    label,
                    args.warmup,
                    args.steps,
                ),
                env=environment,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0:
            failures.append((label, completed.returncode, log_path))
            print(f"  failed ({completed.returncode}); see {log_path}")

    if failures:
        print("Failed configurations:")
        for label, returncode, log_path in failures:
            print(f"- {label}: exit {returncode}, {log_path}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
