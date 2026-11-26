"""Neighbour P2P halo exchange and reverse-gradient accumulation."""

# Third-party
import torch
import torch.distributed as dist


def _wait(operations):
    if operations:
        for request in dist.batch_isend_irecv(operations):
            request.wait()


class _P2PHalo(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, plan):
        ctx.plan, ctx.shape = plan, features.shape
        sends, receives, operations = [], [], []
        for peer in range(plan.world_size):
            if plan.send_indices[peer].numel():
                sends.append(
                    features.index_select(
                        -2, plan.send_indices[peer]
                    ).contiguous()
                )
                operations.append(dist.P2POp(dist.isend, sends[-1], peer))
            if plan.receive_counts[peer]:
                value = features.new_empty(
                    (
                        *features.shape[:-2],
                        plan.receive_counts[peer],
                        features.shape[-1],
                    )
                )
                receives.append(value)
                operations.append(dist.P2POp(dist.irecv, value, peer))
        _wait(operations)
        return (
            torch.cat(receives, -2)
            if receives
            else features.new_empty(
                (*features.shape[:-2], 0, features.shape[-1])
            )
        )

    @staticmethod
    def backward(ctx, gradient):
        plan = ctx.plan
        result = gradient.new_zeros(ctx.shape)
        sends, receives, operations = [], [], []
        offset = 0
        for peer in range(plan.world_size):
            count = plan.receive_counts[peer]
            if count:
                sends.append(
                    gradient[..., offset : offset + count, :].contiguous()
                )
                offset += count
                operations.append(dist.P2POp(dist.isend, sends[-1], peer))
            if plan.send_indices[peer].numel():
                value = gradient.new_empty(
                    (
                        *ctx.shape[:-2],
                        len(plan.send_indices[peer]),
                        ctx.shape[-1],
                    )
                )
                receives.append((peer, value))
                operations.append(dist.P2POp(dist.irecv, value, peer))
        _wait(operations)
        for peer, value in receives:
            result.index_add_(-2, plan.send_indices[peer], value)
        return result, None


def exchange_halo(features, plan):
    """Return owned sender features followed by the required remote halos."""
    if not dist.is_initialized() or (
        dist.get_rank(),
        dist.get_world_size(),
    ) != (plan.rank, plan.world_size):
        raise RuntimeError("Halo plan differs from the process group")
    if features.shape[-2] != len(plan.source_ids):
        raise ValueError("Sender features differ from owned source count")
    if plan.world_size == 1:
        return features
    return torch.cat((features, _P2PHalo.apply(features, plan)), dim=-2)
