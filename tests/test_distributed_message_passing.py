"""Tests for receiver-owned distributed message passing."""

# Standard library
import os
import tempfile

# Third-party
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# First-party
from neural_lam.distributed_message_passing import (
    DistributedFlexibleNet,
    build_bipartite_partition,
    select_partition_inputs,
)
from neural_lam.interaction_net import FlexibleNet


WORLD_SIZE = 2
FOUR_RANK_WORLD_SIZE = 4
NODE_DIM = 4
EDGE_DIM = 3


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


def _inputs(requires_grad=False):
    generator = torch.Generator().manual_seed(1234)
    send_rep = torch.randn(2, 8, NODE_DIM, generator=generator)
    rec_rep = torch.randn(2, 6, NODE_DIM, generator=generator)
    edge_rep = torch.randn(2, 12, EDGE_DIM, generator=generator)
    if requires_grad:
        send_rep.requires_grad_()
        rec_rep.requires_grad_()
        edge_rep.requires_grad_()
    return send_rep, rec_rep, edge_rep


def _reference_net(edge_index, propagation, aggr):
    torch.manual_seed(5678)
    return FlexibleNet(
        edge_index=edge_index,
        send_node_dim=NODE_DIM,
        rec_node_dim=NODE_DIM,
        edge_dim=EDGE_DIM,
        hidden_layers=1,
        num_rec=6,
        propagation=propagation,
        aggr=aggr,
    )


def _partition_net(partition, reference, propagation, aggr):
    local_net = DistributedFlexibleNet(
        partition=partition,
        send_node_dim=NODE_DIM,
        rec_node_dim=NODE_DIM,
        edge_dim=EDGE_DIM,
        hidden_layers=1,
        propagation=propagation,
        aggr=aggr,
    )
    local_net.local_net.load_state_dict(reference.state_dict())
    return local_net


def test_receiver_owned_partition_deduplicates_halo_nodes():
    edge_index, send_owners, rec_owners = _toy_graph()
    partition = build_bipartite_partition(
        edge_index,
        send_owners,
        rec_owners,
        rank=0,
        world_size=WORLD_SIZE,
    )

    torch.testing.assert_close(
        partition.owned_send_nodes, torch.tensor([0, 1, 2, 6])
    )
    torch.testing.assert_close(
        partition.owned_rec_nodes, torch.tensor([0, 1, 4])
    )
    torch.testing.assert_close(partition.halo_nodes, torch.tensor([3, 4, 7]))
    torch.testing.assert_close(
        partition.send_local_indices_by_peer[1], torch.tensor([0, 2, 3])
    )
    torch.testing.assert_close(
        partition.local_edge_positions, torch.tensor([0, 2, 4])
    )
    torch.testing.assert_close(
        partition.boundary_edge_positions, torch.tensor([1, 3, 5])
    )


@pytest.mark.parametrize(
    ("propagation", "aggr"),
    [(True, "mean"), (False, "sum")],
)
def test_single_process_partition_matches_reference(propagation, aggr):
    edge_index, send_owners, rec_owners = _toy_graph()
    reference = _reference_net(edge_index, propagation, aggr)

    ref_send, ref_rec, ref_edge = _inputs(requires_grad=True)
    part_send, part_rec, part_edge = [
        tensor.detach().clone().requires_grad_()
        for tensor in _inputs(requires_grad=False)
    ]

    ref_output = reference(ref_send, ref_rec, ref_edge)
    ref_output.square().sum().backward()

    assembled_output = torch.zeros_like(ref_output)
    partition_nets = []
    partition_outputs = []
    for rank in range(WORLD_SIZE):
        partition = build_bipartite_partition(
            edge_index,
            send_owners,
            rec_owners,
            rank=rank,
            world_size=WORLD_SIZE,
        )
        local_net = _partition_net(
            partition, reference, propagation, aggr
        )
        local_send, local_rec, local_edge, halo = select_partition_inputs(
            part_send, part_rec, part_edge, partition
        )
        local_output = local_net(
            local_send,
            local_rec,
            local_edge,
            halo_rep=halo,
        )
        assembled_output = assembled_output.index_copy(
            -2, partition.owned_rec_nodes, local_output
        )
        partition_nets.append(local_net)
        partition_outputs.append(local_output)

    torch.testing.assert_close(assembled_output, ref_output)
    sum(output.square().sum() for output in partition_outputs).backward()

    torch.testing.assert_close(part_send.grad, ref_send.grad)
    torch.testing.assert_close(part_rec.grad, ref_rec.grad)
    torch.testing.assert_close(part_edge.grad, ref_edge.grad)

    reference_params = dict(reference.named_parameters())
    local_params = [
        dict(module.local_net.named_parameters())
        for module in partition_nets
    ]
    for name, reference_param in reference_params.items():
        partition_grad = sum(
            params[name].grad for params in local_params
        )
        torch.testing.assert_close(partition_grad, reference_param.grad)


def _distributed_case(rank, propagation, aggr):
    edge_index, send_owners, rec_owners = _toy_graph()
    partition = build_bipartite_partition(
        edge_index,
        send_owners,
        rec_owners,
        rank=rank,
        world_size=WORLD_SIZE,
    )
    reference = _reference_net(edge_index, propagation, aggr)
    distributed_net = _partition_net(
        partition, reference, propagation, aggr
    )

    full_send, full_rec, full_edge = _inputs(requires_grad=True)
    ref_output = reference(full_send, full_rec, full_edge)
    ref_output.square().sum().backward()

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

    local_output = distributed_net(local_send, local_rec, local_edge)
    expected_output = ref_output.detach().index_select(
        -2, partition.owned_rec_nodes
    )
    torch.testing.assert_close(local_output, expected_output)
    local_output.square().sum().backward()

    torch.testing.assert_close(
        local_send.grad,
        full_send.grad.index_select(-2, partition.owned_send_nodes),
    )
    torch.testing.assert_close(
        local_rec.grad,
        full_rec.grad.index_select(-2, partition.owned_rec_nodes),
    )
    torch.testing.assert_close(
        local_edge.grad,
        full_edge.grad.index_select(-2, partition.edge_ids),
    )

    reference_params = dict(reference.named_parameters())
    for name, parameter in distributed_net.local_net.named_parameters():
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        torch.testing.assert_close(
            parameter.grad, reference_params[name].grad
        )


def _distributed_worker(rank, init_method):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=WORLD_SIZE,
    )
    try:
        _distributed_case(rank, propagation=True, aggr="mean")
        dist.barrier()
        _distributed_case(rank, propagation=False, aggr="sum")
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _four_rank_distributed_case(rank):
    num_nodes = 8
    receivers = torch.arange(num_nodes).repeat_interleave(2)
    senders = torch.stack(
        (
            torch.arange(num_nodes),
            torch.remainder(torch.arange(num_nodes) - 1, num_nodes),
        ),
        dim=1,
    ).flatten()
    edge_index = torch.stack((senders, receivers), dim=0)
    owners = torch.arange(num_nodes, dtype=torch.long) // 2
    partition = build_bipartite_partition(
        edge_index,
        owners,
        owners,
        rank=rank,
        world_size=FOUR_RANK_WORLD_SIZE,
    )

    torch.manual_seed(9102)
    reference = FlexibleNet(
        edge_index=edge_index,
        send_node_dim=NODE_DIM,
        rec_node_dim=NODE_DIM,
        edge_dim=EDGE_DIM,
        hidden_layers=1,
        num_rec=num_nodes,
        propagation=True,
        aggr="mean",
    )
    distributed_net = _partition_net(
        partition,
        reference,
        propagation=True,
        aggr="mean",
    )

    generator = torch.Generator().manual_seed(1204)
    full_send = torch.randn(
        2,
        num_nodes,
        NODE_DIM,
        generator=generator,
        requires_grad=True,
    )
    full_rec = torch.randn(
        2,
        num_nodes,
        NODE_DIM,
        generator=generator,
        requires_grad=True,
    )
    full_edge = torch.randn(
        2,
        edge_index.shape[1],
        EDGE_DIM,
        generator=generator,
        requires_grad=True,
    )
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
    local_output = distributed_net(local_send, local_rec, local_edge)
    torch.testing.assert_close(
        local_output,
        reference_output.detach().index_select(
            -2,
            partition.owned_rec_nodes,
        ),
    )
    local_output.square().sum().backward()

    torch.testing.assert_close(
        local_send.grad,
        full_send.grad.index_select(-2, partition.owned_send_nodes),
    )
    torch.testing.assert_close(
        local_rec.grad,
        full_rec.grad.index_select(-2, partition.owned_rec_nodes),
    )
    torch.testing.assert_close(
        local_edge.grad,
        full_edge.grad.index_select(-2, partition.edge_ids),
    )
    reference_parameters = dict(reference.named_parameters())
    for name, parameter in distributed_net.local_net.named_parameters():
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        torch.testing.assert_close(
            parameter.grad,
            reference_parameters[name].grad,
        )


def _four_rank_distributed_worker(rank, init_method):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=FOUR_RANK_WORLD_SIZE,
    )
    try:
        _four_rank_distributed_case(rank)
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="Gloo is required for the local multi-process test",
)
def test_two_process_gloo_forward_and_backward(tmp_path):
    init_method = f"file://{tmp_path / 'gloo_init'}"
    mp.spawn(
        _distributed_worker,
        args=(init_method,),
        nprocs=WORLD_SIZE,
        join=True,
    )


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="Gloo is required for the local multi-process test",
)
def test_four_process_gloo_forward_and_backward(tmp_path):
    init_method = f"file://{tmp_path / 'four_rank_gloo_init'}"
    mp.spawn(
        _four_rank_distributed_worker,
        args=(init_method,),
        nprocs=FOUR_RANK_WORLD_SIZE,
        join=True,
    )


if __name__ == "__main__":
    if "RANK" in os.environ:
        _distributed_worker(int(os.environ["RANK"]), "env://")
    else:
        with tempfile.TemporaryDirectory() as temp_dir:
            init_method = f"file://{temp_dir}/gloo_init"
            mp.spawn(
                _distributed_worker,
                args=(init_method,),
                nprocs=WORLD_SIZE,
                join=True,
            )
