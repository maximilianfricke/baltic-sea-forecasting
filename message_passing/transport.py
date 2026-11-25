"""Explicit interchangeable halo transports, including reverse gradients.

All ranks use the default process group. Both backends expose identical
local-senders-then-remote-halos layout; no global monkey-patching is used.
"""

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


class HaloTransport:
    def __init__(self, backend="p2p"):
        if backend not in ("p2p", "physicsnemo"):
            raise ValueError(f"Unknown halo backend: {backend}")
        self.backend = backend

    def __call__(self, features, plan):
        if not dist.is_initialized() or (
            dist.get_rank(),
            dist.get_world_size(),
        ) != (plan.rank, plan.world_size):
            raise RuntimeError(
                "Halo plan differs from the active process group"
            )
        if features.shape[-2] != len(plan.source_ids):
            raise ValueError("Sender features differ from owned source count")
        if self.backend == "p2p" or not any(map(sum, plan.sizes)):
            halo = _P2PHalo.apply(features, plan)
        else:
            # Local
            from .vendor.physicsnemo.distributed.autograd import (
                indexed_all_to_all_v,
            )

            halo = indexed_all_to_all_v(
                features.movedim(-2, 0),
                indices=list(plan.send_indices),
                sizes=[list(row) for row in plan.sizes],
                use_fp32=True,
                dim=0,
                group=dist.group.WORLD,
            ).movedim(0, -2)
        return torch.cat((features, halo), dim=-2)
