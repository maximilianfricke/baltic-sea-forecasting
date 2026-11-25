"""Benchmark-only transport injection; keep the model and graph operators intact.

Both backends return exactly the common partition's remote-only halo layout.
The PN case uses NVIDIA's indexed all-to-all/autograd primitive. It does not
replace the graph with Nils's graph or reproduce his separate model adapter.
"""

from contextlib import contextmanager
import hashlib
import json

import torch
import torch.distributed as dist

from neural_lam import distributed_message_passing as message_passing
from benchmarks.vendor.physicsnemo.distributed.autograd import indexed_all_to_all_v


P2P_EXCHANGE = message_passing.exchange_halo
BACKENDS = ("p2p", "physicsnemo")


def relations(partition):
    yield "g2m", partition.g2m
    yield "m2g", partition.m2g
    for name in ("m2m", "mesh_up", "mesh_down"):
        for level, relation in enumerate(getattr(partition, name)):
            yield f"{name}_{level}", relation


class HaloTransports:
    """Prepare both exchanges once, outside validation/warmup/timing."""

    def __init__(self, partition):
        self.plans = {}
        self.metadata = {}
        for name, relation in relations(partition):
            local = torch.tensor([v.numel() for v in relation.send_local_indices_by_peer],
                                 dtype=torch.int64, device=relation.owned_send_nodes.device)
            rows = [torch.empty_like(local) for _ in range(relation.world_size)]
            dist.all_gather(rows, local)
            sizes = [row.cpu().tolist() for row in rows]
            received = tuple(sizes[p][relation.rank] for p in range(relation.world_size))
            if received != relation.recv_counts_by_peer:
                raise ValueError(f"{name}: halo receive sizes differ from common partition")
            if any(sizes[r][r] for r in range(relation.world_size)):
                raise ValueError("Canonical halo exchange must not contain self traffic")
            self.plans[id(relation)] = (relation, sizes)
            self.metadata[name] = {"sizes_by_sender_receiver": sizes,
                                   "global_remote_rows_per_exchange": sum(map(sum, sizes))}

    def physicsnemo(self, features, partition):
        saved, sizes = self.plans[id(partition)]
        if saved is not partition:
            raise ValueError("Unknown relation object")
        # Both candidates use the existing local path when the ENTIRE relation
        # has no cross-rank traffic (including one GPU). Never skip only one rank.
        if not any(map(sum, sizes)):
            return P2P_EXCHANGE(features, partition)
        return indexed_all_to_all_v(
            features.movedim(-2, 0),
            indices=list(partition.send_local_indices_by_peer),
            sizes=sizes, use_fp32=True, dim=0, group=dist.group.WORLD,
        ).movedim(0, -2)

    @contextmanager
    def use(self, backend):
        if backend not in BACKENDS:
            raise ValueError(backend)
        if message_passing.exchange_halo is not P2P_EXCHANGE:
            raise RuntimeError("Nested or concurrent transport injection is unsupported")
        message_passing.exchange_halo = P2P_EXCHANGE if backend == "p2p" else self.physicsnemo
        try:
            # Context includes backward and checkpoint replay, not just forward.
            yield
        finally:
            message_passing.exchange_halo = P2P_EXCHANGE

    @property
    def signature(self):
        return hashlib.sha256(json.dumps(self.metadata, sort_keys=True).encode()).hexdigest()
