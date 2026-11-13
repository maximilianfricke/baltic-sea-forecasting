"""Two-GPU NCCL smoke test for receiver-owned message passing.

Run this on a node with two visible CUDA devices, for example with::

    torchrun --standalone --nproc-per-node=2 tests/distributed_cuda_smoke.py

The test compares distributed forward outputs, input gradients and summed
parameter gradients with the unpartitioned reference computation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from neural_lam.distributed_message_passing import (
    DistributedFlexibleNet,
    build_bipartite_partition,
)
from neural_lam.interaction_net import FlexibleNet


WORLD_SIZE = 2
NODE_DIM = 32
EDGE_DIM = 16
BATCH_SIZE = 2


def _toy_graph():
    edge_index = torch.tensor(
        [
            [0, 3, 1, 4, 4, 2, 5, 6, 6, 7, 7, 0],
            [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        ],
        dtype=torch.long,
    )
    send_owners = torch.tensor(
        [0, 0, 0, 1, 1, 1, 0, 1], dtype=torch.long
    )
    rec_owners = torch.tensor([0, 0, 1, 1, 0, 1], dtype=torch.long)
    return edge_index, send_owners, rec_owners


def _inputs(device: torch.device, requires_grad: bool):
    generator = torch.Generator().manual_seed(1234)
    send_rep = torch.randn(BATCH_SIZE, 8, NODE_DIM, generator=generator)
    rec_rep = torch.randn(BATCH_SIZE, 6, NODE_DIM, generator=generator)
    edge_rep = torch.randn(BATCH_SIZE, 12, EDGE_DIM, generator=generator)
    tensors = tuple(tensor.to(device) for tensor in (send_rep, rec_rep, edge_rep))
    if requires_grad:
        tensors = tuple(tensor.requires_grad_() for tensor in tensors)
    return tensors


def _max_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float((actual - expected).abs().max().item())


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    started_at = _utc_now()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this smoke test")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != WORLD_SIZE:
        raise RuntimeError(f"expected {WORLD_SIZE} ranks, received {world_size}")

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl")

    try:
        edge_index, send_owners, rec_owners = _toy_graph()
        cpu_partition = build_bipartite_partition(
            edge_index,
            send_owners,
            rec_owners,
            rank=rank,
            world_size=world_size,
        )
        partition = cpu_partition.to(device)

        torch.manual_seed(5678)
        reference = FlexibleNet(
            edge_index=edge_index.to(device),
            send_node_dim=NODE_DIM,
            rec_node_dim=NODE_DIM,
            edge_dim=EDGE_DIM,
            hidden_layers=1,
            num_rec=6,
            propagation=True,
            aggr="mean",
        ).to(device)
        distributed_net = DistributedFlexibleNet(
            partition=partition,
            send_node_dim=NODE_DIM,
            rec_node_dim=NODE_DIM,
            edge_dim=EDGE_DIM,
            hidden_layers=1,
            propagation=True,
            aggr="mean",
        ).to(device)
        distributed_net.local_net.load_state_dict(reference.state_dict())

        full_send, full_rec, full_edge = _inputs(device, requires_grad=True)
        reference_output = reference(full_send, full_rec, full_edge)
        reference_output.square().sum().backward()

        local_send = (
            full_send.detach()
            .index_select(-2, partition.owned_send_nodes)
            .clone()
            .requires_grad_()
        )
        local_rec = (
            full_rec.detach()
            .index_select(-2, partition.owned_rec_nodes)
            .clone()
            .requires_grad_()
        )
        local_edge = (
            full_edge.detach()
            .index_select(-2, partition.edge_ids)
            .clone()
            .requires_grad_()
        )

        torch.cuda.reset_peak_memory_stats(device)
        memory_before = torch.cuda.memory_allocated(device)
        forward_start = torch.cuda.Event(enable_timing=True)
        forward_end = torch.cuda.Event(enable_timing=True)
        backward_start = torch.cuda.Event(enable_timing=True)
        backward_end = torch.cuda.Event(enable_timing=True)

        forward_start.record()
        local_output = distributed_net(local_send, local_rec, local_edge)
        forward_end.record()
        torch.cuda.synchronize(device)
        forward_ms = forward_start.elapsed_time(forward_end)
        expected_output = reference_output.detach().index_select(
            -2, partition.owned_rec_nodes
        )
        torch.testing.assert_close(local_output, expected_output)
        backward_start.record()
        local_output.square().sum().backward()
        backward_end.record()
        torch.cuda.synchronize(device)
        backward_ms = backward_start.elapsed_time(backward_end)

        expected_send_grad = full_send.grad.index_select(
            -2, partition.owned_send_nodes
        )
        expected_rec_grad = full_rec.grad.index_select(
            -2, partition.owned_rec_nodes
        )
        expected_edge_grad = full_edge.grad.index_select(-2, partition.edge_ids)
        torch.testing.assert_close(local_send.grad, expected_send_grad)
        torch.testing.assert_close(local_rec.grad, expected_rec_grad)
        torch.testing.assert_close(local_edge.grad, expected_edge_grad)

        parameter_error = 0.0
        reference_params = dict(reference.named_parameters())
        for name, parameter in distributed_net.local_net.named_parameters():
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
            torch.testing.assert_close(parameter.grad, reference_params[name].grad)
            parameter_error = max(
                parameter_error,
                _max_error(parameter.grad, reference_params[name].grad),
            )

        local_metrics = torch.tensor(
            [
                _max_error(local_output, expected_output),
                _max_error(local_send.grad, expected_send_grad),
                _max_error(local_rec.grad, expected_rec_grad),
                _max_error(local_edge.grad, expected_edge_grad),
                parameter_error,
                torch.cuda.max_memory_allocated(device) / (1024**2),
                (
                    torch.cuda.max_memory_allocated(device) - memory_before
                )
                / (1024**2),
                forward_ms,
                backward_ms,
            ],
            device=device,
            dtype=torch.float64,
        )
        gathered_metrics = [torch.zeros_like(local_metrics) for _ in range(world_size)]
        dist.all_gather(gathered_metrics, local_metrics)
        torch.cuda.synchronize(device)

        if rank == 0:
            print("Two-GPU NCCL distributed message-passing smoke: PASS")
            for peer, metrics in enumerate(gathered_metrics):
                values = metrics.cpu().tolist()
                print(
                    f"rank={peer} output_err={values[0]:.3e} "
                    f"send_grad_err={values[1]:.3e} "
                    f"rec_grad_err={values[2]:.3e} "
                    f"edge_grad_err={values[3]:.3e} "
                    f"parameter_grad_err={values[4]:.3e} "
                    f"peak_memory_mib={values[5]:.1f} "
                    f"incremental_peak_mib={values[6]:.1f} "
                    f"forward_ms={values[7]:.3f} "
                    f"backward_ms={values[8]:.3f}"
                )
            if args.output is not None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                result = {
                    "schema_version": 1,
                    "status": "completed",
                    "test": "distributed_message_passing_cuda_smoke",
                    "comparable_to_training_baseline": False,
                    "comparison_note": (
                        "Synthetic toy graph correctness/micro-timing only; "
                        "do not compare throughput with full GraphEFM runs."
                    ),
                    "started_at_utc": started_at,
                    "finished_at_utc": _utc_now(),
                    "git_commit": os.environ.get("NJORD_GIT_COMMIT"),
                    "git_dirty": os.environ.get("NJORD_GIT_DIRTY"),
                    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                    "backend": "nccl",
                    "world_size": world_size,
                    "torch_version": torch.__version__,
                    "cuda_runtime": torch.version.cuda,
                    "visible_device_names": [
                        torch.cuda.get_device_name(index)
                        for index in range(torch.cuda.device_count())
                    ],
                    "toy_workload": {
                        "batch_size": BATCH_SIZE,
                        "sender_nodes": 8,
                        "receiver_nodes": 6,
                        "edges": 12,
                        "node_dim": NODE_DIM,
                        "edge_dim": EDGE_DIM,
                    },
                    "rank_metrics": [
                        {
                            "rank": peer,
                            "output_max_abs_error": values[0],
                            "send_gradient_max_abs_error": values[1],
                            "receiver_gradient_max_abs_error": values[2],
                            "edge_gradient_max_abs_error": values[3],
                            "parameter_gradient_max_abs_error": values[4],
                            "peak_allocated_mib": values[5],
                            "incremental_peak_mib": values[6],
                            "forward_ms": values[7],
                            "backward_ms": values[8],
                        }
                        for peer, values in enumerate(
                            metrics.cpu().tolist()
                            for metrics in gathered_metrics
                        )
                    ],
                }
                args.output.write_text(
                    json.dumps(result, indent=2) + "\n",
                    encoding="utf-8",
                )
                print(f"Wrote {args.output}")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
