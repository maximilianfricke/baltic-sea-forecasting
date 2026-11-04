"""Measure the actual single-host transport before a graph scaling run.

Example: torchrun --standalone --nproc-per-node=2 -m
benchmarks.graph_parallel_probe --output benchmark_results/transport-2.json
"""

# Standard library
import argparse
import json
import os
import platform
import socket
import subprocess
import time
from pathlib import Path

# Third-party
import torch
import torch.distributed as dist


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 0:
        parser.error("iterations must be positive and warmup nonnegative")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank, size = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", local_rank)
    try:
        # Initialize the communicator on every rank before subset P2P calls.
        dist.all_reduce(torch.zeros(1, device=device))
        rank_metadata = {
            "rank": rank,
            "hostname": socket.gethostname(),
            "device": torch.cuda.get_device_name(device),
            "compute_capability": torch.cuda.get_device_capability(device),
        }
        ranks = [None] * size
        dist.all_gather_object(ranks, rank_metadata)
        if len({entry["hostname"] for entry in ranks}) != 1:
            raise ValueError("This probe is intended for one host")
        measurements = []
        for first in range(size):
            for second in range(first + 1, size):
                for byte_count in (4096, 1024**2, 16 * 1024**2):
                    dist.barrier()
                    elapsed = 0.0
                    if rank in (first, second):
                        peer = second if rank == first else first
                        outgoing = torch.zeros(
                            byte_count, dtype=torch.uint8, device=device
                        )
                        incoming = torch.empty_like(outgoing)
                        for iteration in range(args.warmup + args.iterations):
                            if iteration == args.warmup:
                                torch.cuda.synchronize()
                                started = time.perf_counter()
                            requests = dist.batch_isend_irecv(
                                [
                                    dist.P2POp(dist.isend, outgoing, peer),
                                    dist.P2POp(dist.irecv, incoming, peer),
                                ]
                            )
                            for request in requests:
                                request.wait()
                        torch.cuda.synchronize()
                        elapsed = time.perf_counter() - started
                    duration = torch.tensor(
                        elapsed, dtype=torch.float64, device=device
                    )
                    dist.all_reduce(duration, op=dist.ReduceOp.MAX)
                    seconds = duration.item() / args.iterations
                    measurements.append(
                        {
                            "ranks": [first, second],
                            "bytes_each_direction": byte_count,
                            "duplex_exchange_seconds": seconds,
                            "duplex_bytes_per_second": 2 * byte_count / seconds,
                        }
                    )
        if rank == 0:
            topology = subprocess.run(
                ["nvidia-smi", "topo", "-m"],
                capture_output=True,
                text=True,
                check=False,
            )
            result = {
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "nccl": torch.cuda.nccl.version(),
                "platform": platform.platform(),
                "ranks": ranks,
                "topology": topology.stdout,
                "topology_error": topology.stderr,
                "iterations": args.iterations,
                "warmup": args.warmup,
                "measurements": measurements,
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(result, indent=2), encoding="utf-8"
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
