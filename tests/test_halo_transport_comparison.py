"""CPU correctness checks. PN's native NCCL execution is a CLUSTER gate.

Gloo cannot implement PN's unequal-list all_to_all. In these tests only, replace
that lowest transport call with CPU P2P. NVIDIA's actual packing, concatenation,
autograd and backward index-add code still execute, without modifications.
No performance conclusions follow from this emulation.
"""

import copy
from datetime import timedelta
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from benchmarks.compare_halo_transports import cached_batches, compare_snapshots, graph_digest, state_digest, summarize, validation_snapshot
from benchmarks import model_parallel_real as real
from benchmarks.halo_transport_backends import HaloTransports, P2P_EXCHANGE
from benchmarks.vendor.physicsnemo.distributed import utils as pn_utils
from neural_lam import distributed_message_passing as message_passing
from neural_lam.distributed_message_passing import build_bipartite_partition, build_graph_partition, select_owned
from neural_lam.models.distributed_graph_efm import distributed_core_from_graph_efm
from neural_lam.models.distributed_graph_efm_training import DistributedGraphEFMTraining
from tests.test_distributed_graph_efm_rollout import ToyGraphEFM, _batch


def emulated_all_to_all(outputs, inputs, group=None, async_op=False):
    assert not async_op
    rank = dist.get_rank(group)
    ops = []
    for peer in range(dist.get_world_size(group)):
        if peer == rank:
            outputs[peer].copy_(inputs[peer])
        else:
            if inputs[peer].numel(): ops.append(dist.P2POp(dist.isend, inputs[peer], peer, group))
            if outputs[peer].numel(): ops.append(dist.P2POp(dist.irecv, outputs[peer], peer, group))
    if ops:
        for work in dist.batch_isend_irecv(ops): work.wait()


def transport_case(rank, size):
    send = torch.arange(8) % size
    receive = torch.tensor([0, 0, 1, 1, 2, 3]) % size
    edges = torch.tensor([[0,1,1,2,4,6,7,0,3,3,5], [0,0,1,1,2,2,3,3,4,5,5]])
    relation = build_bipartite_partition(edges, send, receive, rank, size)
    partition = SimpleNamespace(g2m=relation, m2g=relation, m2m=(), mesh_up=(), mesh_down=())
    transports = HaloTransports(partition)
    features = torch.arange(2*8*3, dtype=torch.float64).reshape(2,8,3)[:, relation.owned_send_nodes].clone()
    observations = []
    for backend in ("p2p", "physicsnemo"):
        x = features.clone().requires_grad_()
        with transports.use(backend):
            result = message_passing.exchange_halo(x, relation)
            torch.testing.assert_close(result, torch.arange(2*8*3, dtype=torch.float64).reshape(2,8,3)[:, relation.halo_nodes])
            (result.square().sum() * (rank+1) + x.square().sum()).backward()
        observations.append((result.detach(), x.grad))
    torch.testing.assert_close(observations[0][0], observations[1][0], rtol=0, atol=0)
    torch.testing.assert_close(observations[0][1], observations[1][1], rtol=0, atol=0)
    assert message_passing.exchange_halo is P2P_EXCHANGE


def rollout_case(rank, size, checkpointing):
    torch.manual_seed(42)
    frontend = ToyGraphEFM(checkpointing)
    grid = torch.arange(8) % size
    meshes = (torch.arange(4) % size, (torch.arange(4)+1) % size)
    partition = build_graph_partition(frontend.g2m_edge_index, frontend.m2g_edge_index,
        frontend.m2m_edge_index, grid, grid, meshes, rank, size,
        frontend.mesh_up_edge_index, frontend.mesh_down_edge_index)
    core = distributed_core_from_graph_efm(frontend, partition)
    frontend.prior_model = frontend.encoder = frontend.decoder = None
    training = DistributedGraphEFMTraining(frontend, core, partition)
    batches = _batch(torch.device("cpu"))
    batch = tuple(select_owned(t, partition.interior_grid_nodes) if i in (0,1,2,4) else t for i,t in enumerate(batches))
    initial = copy.deepcopy(training.state_dict())
    transports = HaloTransports(partition)
    p2p = validation_snapshot(training, transports, "p2p", batch, initial)
    pnemo = validation_snapshot(training, transports, "physicsnemo", batch, initial)
    report = compare_snapshots(p2p, pnemo)
    assert report["passed"], report
    assert report["optimizer_updates"] == 2 and report["adamw_masked_elements"] == 0


def worker(rank, size, init_method, checkpointing):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    dist.init_process_group("gloo", rank=rank, world_size=size, init_method=init_method, timeout=timedelta(seconds=180))
    try:
        with patch.object(pn_utils.dist, "all_to_all", emulated_all_to_all):
            transport_case(rank, size)
            rollout_case(rank, size, checkpointing)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("size,checkpointing", [(1,True),(2,True),(4,True),(2,False)])
def test_cpu_emulated_pnemo_matches_p2p_ar4(tmp_path, size, checkpointing):
    mp.spawn(worker, args=(size, f"file://{tmp_path/'init'}", checkpointing), nprocs=size, join=True)


def repeat_result(index):
    return {"status":"completed", "controls_sha256":"same", "controls":{"world_size":2,"source_hashes":{"a":"same"}},
            "validation":[{"passed":True}], "interpretation":{"timer":"cached model"},
            "cases":{b:{"summary":{"model_wall_ms":{"median":v}}} for b,v in (("p2p",100+index),("physicsnemo",80+index))}}


def test_summary_requires_matching_controls_and_all_repeats(tmp_path):
    for i in range(3): (tmp_path/f"repeat-{i}.json").write_text(json.dumps(repeat_result(i)))
    report = summarize(tmp_path,3)
    assert report["physicsnemo_speedup_over_p2p"] == pytest.approx(101/81)
    bad = repeat_result(2); bad["controls_sha256"] = "different"
    (tmp_path/'repeat-2.json').write_text(json.dumps(bad))
    with pytest.raises(ValueError,match="Controls differ"): summarize(tmp_path,3)


def test_backend_context_restores_after_failure():
    transports = object.__new__(HaloTransports)
    with pytest.raises(RuntimeError,match="test"):
        with transports.use("physicsnemo"):
            raise RuntimeError("test")
    assert message_passing.exchange_halo is P2P_EXCHANGE


@pytest.mark.skipif(not Path("data/physical_repeat_16/global_ocean_1.yaml").exists(), reason="Local physical data optional")
def test_real_graph_and_rank_cache_preserve_sample_identity():
    torch.set_num_threads(2)
    args = SimpleNamespace(config_path=Path("data/physical_repeat_16/global_ocean_1.yaml"),
        partitioned_data_root=Path("data/partitioned/physical_repeat_16"), graph="global_cluster_graph",
        hidden_dim=64, processor_layers=2, batch_size=2, ar_steps=4, lr=.001, seed=42,
        cached_batches=1, num_workers=0, prefetch_factor=2, partition_strategy="geographic",
        latitude_cut=-5., longitude_seam=120.)
    _, model, config = real._load_workload(args, include_global_dataset=False)
    assert len(graph_digest(model)) == 64
    template = copy.deepcopy(model)
    _, _, partition, _ = real._build_sharded_training_modules(model,args,0,1,torch.device("cpu"))
    assert [v.numel() for v in partition.mesh_nodes] == [8456,2097]
    dataset, _ = real._load_rank_local_dataset(args,config,0,1)
    ids = torch.as_tensor(dataset.datastore._ds.global_surface_index.values,dtype=torch.long)
    assert torch.equal(ids,partition.interior_grid_nodes.cpu())
    cache = cached_batches(dataset,args)
    expected = torch.utils.data.default_collate([dataset[0],dataset[1]])
    for actual, value in zip(cache[0],expected,strict=True):
        torch.testing.assert_close(actual,value,rtol=0,atol=0,equal_nan=True)
    for size in (2,4):
        signatures = []
        for rank in range(size):
            frontend,core,part,_ = real._build_sharded_training_modules(copy.deepcopy(template),args,rank,size,torch.device('cpu'))
            local = DistributedGraphEFMTraining(frontend,core,part)
            signatures.append(state_digest(local.state_dict()))
            shard,_ = real._load_rank_local_dataset(args,config,rank,size)
            assert torch.equal(torch.as_tensor(shard.datastore._ds.global_surface_index.values),part.interior_grid_nodes.cpu())
            del frontend,core,part,local,shard
        assert len(set(signatures)) == 1, 'Initial state differs between ranks'
