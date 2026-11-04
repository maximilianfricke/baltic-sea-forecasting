"""Differentiable halo exchange and replicated/sharded boundary adapters.

All ranks execute the same graph program. Only nodes and edges are sharded;
MLP weights remain replicated. This module intentionally does not use DDP.
"""

# Standard library
import time
from contextlib import contextmanager

# Third-party
import torch
import torch.distributed as dist

# Local
from .graphs.partition import (
    EdgePartition,
    MeshPartition,
    graph_fingerprint,
    spatial_owners,
)


def _start_timing(owner, tensor):
    if not owner.profile:
        return None
    if tensor.is_cuda:
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        return start
    return time.perf_counter()


def _finish_timing(owner, started, name):
    if started is None:
        return
    if isinstance(started, float):
        owner.timings.append((name, (time.perf_counter() - started) * 1000))
    else:
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        owner.timings.append((name, (started, end)))


def _timing_totals(owner):
    totals = {}
    for name, measurement in owner.timings:
        if isinstance(measurement, tuple):
            start, end = measurement
            end.synchronize()
            measurement = start.elapsed_time(end)
        totals[name] = totals.get(name, 0.0) + measurement
    return totals


def _exchange(tensor, plan, backward=False):
    """Exchange packed per-peer features; retain buffers until work finishes."""
    _, _, send_ids = plan.indices(tensor.device)
    ops, buffers, received = [], [], {}
    for peer in range(plan.size):
        if peer == plan.rank:
            continue
        if backward:
            receive_count = len(plan.send_ids.get(peer, []))
            outgoing = (
                tensor[..., plan._halo_slices[peer], :].contiguous()
                if peer in plan.recv_ids
                else None
            )
        else:
            receive_count = len(plan.recv_ids.get(peer, []))
            outgoing = (
                tensor.index_select(-2, send_ids[peer]).contiguous()
                if peer in send_ids
                else None
            )
        if receive_count:
            shape = (*tensor.shape[:-2], receive_count, tensor.shape[-1])
            key = ("recv", backward, peer, shape, tensor.dtype, tensor.device)
            if key not in plan._communication_buffers:
                with torch.inference_mode(False):
                    plan._communication_buffers[key] = tensor.new_empty(shape)
            received[peer] = plan._communication_buffers[key]
            ops.append(dist.P2POp(dist.irecv, received[peer], peer))
        if outgoing is not None:
            key = (
                "send",
                backward,
                peer,
                tuple(outgoing.shape),
                tensor.dtype,
                tensor.device,
            )
            if key not in plan._communication_buffers:
                with torch.inference_mode(False):
                    plan._communication_buffers[key] = torch.empty_like(
                        outgoing
                    )
            plan._communication_buffers[key].copy_(outgoing)
            outgoing = plan._communication_buffers[key]
            buffers.append(outgoing)
            ops.append(dist.P2POp(dist.isend, outgoing, peer))
    works = dist.batch_isend_irecv(ops) if ops else []
    byte_count = sum(t.numel() * t.element_size() for t in buffers)
    if backward:
        plan.backward_bytes += byte_count
    else:
        plan.forward_bytes += byte_count
    return works, received, buffers


def _halo_slices(plan):
    plan._halo_slices = {}
    offset = 0
    for peer, ids in plan.recv_ids.items():
        plan._halo_slices[peer] = slice(offset, offset + len(ids))
        offset += len(ids)


class _Halo(torch.autograd.Function):
    @staticmethod
    def forward(ctx, owned, plan):
        ctx.plan = plan
        started = _start_timing(plan, owned)
        _halo_slices(plan)
        works, received, buffers = _exchange(owned, plan)
        for work in works:
            work.wait()
        _finish_timing(plan, started, "forward_halo_window_ms")
        return torch.cat([owned] + [received[p] for p in plan.recv_ids], dim=-2)

    @staticmethod
    def backward(ctx, grad):
        plan = ctx.plan
        started = _start_timing(plan, grad)
        count = len(plan.owned_senders)
        result = grad[..., :count, :].contiguous().clone()
        works, received, buffers = _exchange(
            grad[..., count:, :], plan, backward=True
        )
        for work in works:
            work.wait()
        _, _, send_ids = plan.indices(grad.device)
        for peer, tensor in received.items():
            result.index_add_(-2, send_ids[peer], tensor)
        _finish_timing(plan, started, "backward_halo_window_ms")
        return result, None


class _OverlapMessages(torch.autograd.Function):
    """Overlap halo transfers with independent local edge MLPs in both passes.

    The inner autograd graphs use the original MLP weights. The outer function
    returns their gradients explicitly, preserving names and optimizer semantics.
    Communication buffers never back saved MLP inputs: gathered features are
    copied before the reusable buffers can be overwritten by the next layer.
    """

    @staticmethod
    def forward(ctx, owned, receiver, edges, emb, net, *parameters):
        plan = net.graph_edge_partition
        started = _start_timing(plan, owned)
        _halo_slices(plan)
        ctx.plan = plan
        send_index, rec_index, _ = plan.indices(owned.device)
        local_ids = plan.local_edge_ids.to(owned.device)
        remote_ids = plan.remote_edge_ids.to(owned.device)
        ctx.edge_ids = (local_ids, remote_ids)
        ctx.parameter_flags = [p.requires_grad for p in parameters]
        ctx.parameters = tuple(p for p in parameters if p.requires_grad)
        works, received, buffers = _exchange(owned, plan)
        with torch.enable_grad():
            sender = owned.detach().requires_grad_(True)
            rec = receiver.detach().requires_grad_(True)
            edge = edges.detach().requires_grad_(True)
            condition = (
                emb.detach().requires_grad_(True) if emb is not None else None
            )
            local_message = net.message(
                sender.index_select(-2, send_index[local_ids]),
                rec.index_select(-2, rec_index[local_ids]),
                edge.index_select(-2, local_ids),
                condition,
            )
            for work in works:
                work.wait()
            _finish_timing(plan, started, "forward_halo_window_ms")
            halo = (
                torch.cat([received[p] for p in plan.recv_ids], dim=-2)
                if plan.recv_ids
                else owned.new_empty((*owned.shape[:-2], 0, owned.shape[-1]))
            ).requires_grad_(True)
            remote_message = net.message(
                halo.index_select(
                    -2, send_index[remote_ids] - len(plan.owned_senders)
                ),
                rec.index_select(-2, rec_index[remote_ids]),
                edge.index_select(-2, remote_ids),
                condition,
            )
        ctx.inputs = (sender, rec, edge, condition, halo)
        ctx.messages = (local_message, remote_message)
        result = local_message.new_empty(
            (*edges.shape[:-1], local_message.shape[-1])
        )
        result.index_copy_(-2, local_ids, local_message)
        result.index_copy_(-2, remote_ids, remote_message)
        return result

    @staticmethod
    def backward(ctx, grad):
        plan = ctx.plan
        started = _start_timing(plan, grad)
        sender, rec, edge, condition, halo = ctx.inputs
        condition_inputs = (condition,) if condition is not None else ()
        common = (rec, edge) + condition_inputs + ctx.parameters
        local_ids, remote_ids = ctx.edge_ids
        remote_grads = torch.autograd.grad(
            ctx.messages[1],
            (halo,) + common,
            grad.index_select(-2, remote_ids),
            allow_unused=True,
        )
        halo_grad = remote_grads[0]
        if halo_grad is None:
            halo_grad = torch.zeros_like(halo)
        works, received, buffers = _exchange(halo_grad, plan, backward=True)
        # This MLP backward is independent of incoming sender gradients.
        local_grads = torch.autograd.grad(
            ctx.messages[0],
            (sender,) + common,
            grad.index_select(-2, local_ids),
            allow_unused=True,
        )
        sender_grad = local_grads[0]
        if sender_grad is None:
            sender_grad = torch.zeros_like(sender)
        for work in works:
            work.wait()
        _, _, send_ids = plan.indices(grad.device)
        for peer, contribution in received.items():
            sender_grad.index_add_(-2, send_ids[peer], contribution)
        _finish_timing(plan, started, "backward_halo_window_ms")
        combined = [
            (
                a + b
                if a is not None and b is not None
                else (a if b is None else b)
            )
            for a, b in zip(local_grads[1:], remote_grads[1:])
        ]
        rec_grad, edge_grad = combined[:2]
        emb_grad = combined[2] if condition is not None else None
        weight_grads = iter(combined[2 + len(condition_inputs) :])
        return (sender_grad, rec_grad, edge_grad, emb_grad, None) + tuple(
            next(weight_grads) if flag else None for flag in ctx.parameter_flags
        )


class _SelectOwned(torch.autograd.Function):
    @staticmethod
    def forward(ctx, replicated, ids):
        ctx.save_for_backward(ids)
        ctx.shape = replicated.shape
        return replicated.index_select(-2, ids)

    @staticmethod
    def backward(ctx, grad):
        (ids,) = ctx.saved_tensors
        result = grad.new_zeros(ctx.shape)
        result.index_copy_(-2, ids, grad)
        dist.all_reduce(result)
        return result, None


class _Replicate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, owned, ids, count):
        ctx.save_for_backward(ids)
        result = owned.new_zeros((*owned.shape[:-2], count, owned.shape[-1]))
        result.index_copy_(-2, ids, owned)
        dist.all_reduce(result)
        return result

    @staticmethod
    def backward(ctx, grad):
        (ids,) = ctx.saved_tensors
        grad = grad.contiguous().clone()
        dist.all_reduce(grad)
        grad /= dist.get_world_size()
        return grad.index_select(-2, ids), None, None


class GraphParallelContext:
    """Runtime metadata; deliberately not a Module or checkpoint buffer."""

    def __init__(self, partition, rank, edge_indices):
        self.partition, self.rank = partition, rank
        self.node_ids = [torch.where(o == rank)[0] for o in partition.owners]
        self.plans = {}
        self.global_edge_counts = {
            kind: [edges.shape[1] for edges in levels]
            for kind, levels in edge_indices.items()
        }
        self.profile = False
        self.timings = []
        for kind, levels in edge_indices.items():
            plans = []
            for level, edges in enumerate(levels):
                send_level = level + (kind == "mesh_down")
                rec_level = level + (kind == "mesh_up")
                plans.append(
                    EdgePartition(
                        edges,
                        partition.owners[send_level],
                        partition.owners[rec_level],
                        rank,
                        partition.size,
                    )
                )
            self.plans[kind] = plans

    def select(self, tensor, level):
        return _SelectOwned.apply(
            tensor, self.node_ids[level].to(tensor.device)
        )

    def replicate(self, tensor, level):
        return _Replicate.apply(
            tensor,
            self.node_ids[level].to(tensor.device),
            len(self.partition.owners[level]),
        )

    def metadata(self):
        return {
            "partition_identity": self.partition.identity,
            "global_nodes_per_level": [len(o) for o in self.partition.owners],
            "owned_nodes_per_level": [len(ids) for ids in self.node_ids],
            "global_edges_per_level": self.global_edge_counts,
            "diagnostic_timing_totals": _timing_totals(self),
            "relations": {
                kind: [
                    {
                        "owned_edges": len(p.edge_ids),
                        "halo_nodes": len(p.halo_ids),
                        "send_nodes": sum(
                            len(ids) for ids in p.send_ids.values()
                        ),
                        "forward_sent_bytes": p.forward_bytes,
                        "backward_sent_bytes": p.backward_bytes,
                        "diagnostic_timing_totals": _timing_totals(p),
                    }
                    for p in plans
                ]
                for kind, plans in self.plans.items()
            },
        }


def partitioned_forward(net, send_rep, rec_rep, edge_rep, emb, propagation):
    """Evaluate the original per-edge and per-node MLPs on owned receivers."""
    plan = net.graph_edge_partition
    send_index, rec_index, _ = plan.indices(send_rep.device)
    if plan.overlap and torch.is_grad_enabled():
        messages = _OverlapMessages.apply(
            send_rep, rec_rep, edge_rep, emb, net, *net.edge_mlp.parameters()
        )
        anchor = messages.sum() * 0
    else:
        send = _Halo.apply(send_rep, plan)
        messages = net.message(
            send.index_select(-2, send_index),
            rec_rep.index_select(-2, rec_index),
            edge_rep,
            emb,
        )
        anchor = send.sum() * 0
    aggregated = messages.new_zeros(
        (*messages.shape[:-2], net.num_rec, messages.shape[-1])
    )
    aggregated.index_add_(-2, rec_index, messages)
    if propagation or net.aggr == "mean":
        counts = torch.bincount(rec_index, minlength=net.num_rec).clamp_min(1)
        aggregated = aggregated / counts.unsqueeze(-1)
    # Keep the exchange in every rank's backward graph, including empty ranks.
    aggregated = aggregated + anchor
    difference = net.aggr_mlp(torch.cat((rec_rep, aggregated), dim=-1), emb)
    updated = (aggregated if propagation else rec_rep) + difference
    if net.update_edges:
        return updated, edge_rep + messages
    return updated


def make_mesh_partition(model, size, owners=None):
    """Build the public partition artifact from a CPU GraphEFM and owner arrays.

    External partitioners supply ``owners[level][global_node_id]``. This helper
    attaches and validates the graph identity used by the training loader.
    """
    positions = list(model.mesh_static_features)
    tensors = []
    for kind in ("m2m", "mesh_up", "mesh_down"):
        tensors.extend(getattr(model, f"{kind}_edge_index"))
        tensors.extend(getattr(model, f"{kind}_features"))
    fingerprint = graph_fingerprint(positions, tensors)
    if owners is None:
        owners = spatial_owners(positions, size)
    partition = MeshPartition(
        [o.detach().cpu().clone() for o in owners], size, fingerprint
    )
    partition.validate([len(p) for p in positions], fingerprint)
    return partition


def configure_graph_parallel(model, size, rank, partition_path=None):
    """Shard a freshly constructed CPU GraphEFM before moving it to a GPU."""
    # Local
    from .interaction_net import InteractionNet
    from .utils import BufferList

    if getattr(model, "graph_parallel", None) is not None:
        if (
            model.graph_parallel.partition.size != size
            or model.graph_parallel.rank != rank
        ):
            raise ValueError(
                "Cannot change graph ownership on an already sharded model"
            )
        return model.graph_parallel
    if not model.hierarchical_graph:
        raise ValueError(
            "Graph parallelism currently requires a hierarchical GraphEFM graph"
        )
    positions = list(model.mesh_static_features)
    edges = {
        kind: list(getattr(model, f"{kind}_edge_index"))
        for kind in ("m2m", "mesh_up", "mesh_down")
    }
    default_partition = make_mesh_partition(model, size)
    partition = (
        MeshPartition.load(partition_path)
        if partition_path
        else default_partition
    )
    partition.validate(
        [len(p) for p in positions], default_partition.graph_fingerprint
    )
    if partition.size != size:
        raise ValueError("Partition size must equal graph_parallel_size")
    identities = [None] * size
    dist.all_gather_object(identities, partition.identity)
    if len(set(identities)) != 1:
        raise ValueError("Ranks loaded different graphs or ownership schedules")
    context = GraphParallelContext(partition, rank, edges)
    context.profile = (
        getattr(model.args, "benchmark_output", None) is not None
        and getattr(model.args, "benchmark_timing_mode", None) == "phase"
    )
    for plans in context.plans.values():
        for plan in plans:
            plan.overlap = getattr(model.args, "graph_parallel_overlap", False)
            plan.profile = context.profile
    model.graph_parallel = context
    model._global_mesh_sizes = [len(p) for p in positions]

    for kind, plans in context.plans.items():
        features = getattr(model, f"{kind}_features")
        setattr(
            model,
            f"{kind}_features",
            BufferList(
                [f[p.edge_ids] for f, p in zip(features, plans)],
                persistent=False,
            ),
        )
        setattr(
            model,
            f"{kind}_edge_index",
            BufferList(
                [torch.stack((p.send_index, p.rec_index)) for p in plans],
                persistent=False,
            ),
        )
    model.mesh_static_features = BufferList(
        [p[ids] for p, ids in zip(positions, context.node_ids)],
        persistent=False,
    )

    for component in (model.encoder, model.prior_model, model.decoder):
        if not hasattr(component, "mesh_up_gnns"):
            continue  # ConstantLatentEncoder has no mesh computation.
        component.graph_parallel = context
        sequences = [("mesh_up", component.mesh_up_gnns)]
        if hasattr(component, "mesh_down_gnns"):
            sequences += [
                ("mesh_down", component.mesh_down_gnns),
                ("m2m", component.intra_up_gnns),
                ("m2m", component.intra_down_gnns),
            ]
        else:
            sequences += [("m2m", component.intra_level_gnns)]
        for kind, modules in sequences:
            for module, plan in zip(modules, context.plans[kind]):
                for net in module.modules():
                    if isinstance(net, InteractionNet):
                        net.graph_edge_partition = plan
                        net.num_rec = len(plan.owned_receivers)
                        net.edge_index = torch.stack(
                            (plan.send_index + net.num_rec, plan.rec_index)
                        )
    return context


def parameter_reductions(model):
    """Classify each parameter once, including currently frozen parameters."""
    sharded_prefixes = (
        "mesh_embedders.",
        "m2m_embedders.",
        "mesh_up_embedders.",
        "mesh_down_embedders.",
    )
    mesh_parts = (
        "mesh_up_gnns",
        "mesh_down_gnns",
        "intra_level_gnns",
        "intra_up_gnns",
        "intra_down_gnns",
        "pre_mesh_proj",
        "post_mesh_proj",
        "latent_param_map",
    )
    return {
        name: (
            "sum"
            if name.startswith(sharded_prefixes)
            or any(
                name.startswith(f"{component}.{part}.")
                for component in ("encoder", "prior_model", "decoder")
                for part in mesh_parts
            )
            else "mean"
        )
        for name, _ in model.named_parameters()
    }


def synchronize_gradients(model, reductions):
    """Reduce before clipping/optimizer transforms; preserve globally unused grads."""
    context = getattr(model, "graph_parallel", None)
    parameter = next(model.parameters(), None)
    started = (
        _start_timing(context, parameter)
        if context and parameter is not None
        else None
    )
    _synchronize_gradients(model, reductions)
    if context:
        _finish_timing(context, started, "gradient_sync_ms")


def _synchronize_gradients(model, reductions):
    parameters = [
        (n, p) for n, p in model.named_parameters() if p.requires_grad
    ]
    if not parameters:
        return
    device = parameters[0][1].device
    present = torch.tensor(
        [p.grad is not None for _, p in parameters],
        device=device,
        dtype=torch.int32,
    )
    dist.all_reduce(present)
    # Bucket by reduction and dtype to avoid one collective per MLP parameter.
    buckets = {}
    for used, (name, param) in zip(present.tolist(), parameters):
        if used:
            buckets.setdefault((reductions[name], param.dtype), []).append(
                param
            )
    for (reduction, _), params in buckets.items():
        packed = torch.cat(
            [
                (p.grad if p.grad is not None else torch.zeros_like(p)).reshape(
                    -1
                )
                for p in params
            ]
        )
        dist.all_reduce(packed)
        if reduction == "mean":
            packed /= dist.get_world_size()
        offset = 0
        for param in params:
            param.grad = (
                packed[offset : offset + param.numel()].view_as(param).clone()
            )
            offset += param.numel()


@contextmanager
def communicating_checkpoint():
    """All ranks must replay the complete checkpoint region in the same order."""
    with torch.utils.checkpoint.set_checkpoint_early_stop(False):
        yield
