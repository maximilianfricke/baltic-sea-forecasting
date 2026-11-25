"""Paired cached-model P2P/PhysicsNeMo comparison on OUR graph and AR4 path.

Every repeat is a fresh torchrun launch; its two candidates share a CPU cache,
ownership plan and initial state. Reset weights, AdamW and latent RNG between
candidates. Transfer and rank-readiness barriers are outside the named model
timer. This is not application end-to-end throughput or Nils's whole adapter.
"""

import argparse
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time

# Apply rank placement before torch or CUDA can create helper threads.
from benchmarks.cpu_affinity import AffinityGuard, POLICY

CPU_AFFINITY = AffinityGuard.from_environment()
CPU_AFFINITY.checkpoint("before_torch_import", repair=True)

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset

from benchmarks import model_parallel_real as real
from benchmarks.halo_transport_backends import BACKENDS, HaloTransports, relations
from neural_lam.models.distributed_graph_efm_training import DistributedGraphEFMTraining

CPU_AFFINITY.checkpoint("after_imports", repair=True)

REPO = Path(__file__).resolve().parents[1]


def digest_tensor(tensor):
    tensor = tensor.detach().cpu().contiguous()
    h = hashlib.sha256(str((str(tensor.dtype), tuple(tensor.shape))).encode())
    h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def state_digest(state):
    return hashlib.sha256(json.dumps({k: digest_tensor(v) for k,v in state.items()}, sort_keys=True).encode()).hexdigest()


def graph_digest(model):
    """Hash actual tensor contents, including original node/edge order."""
    values = {}
    for name in ("g2m_edge_index", "m2g_edge_index", "grid_static_features",
                 "g2m_features", "m2g_features", "loss_mask", "mesh_static_features",
                 "m2m_edge_index", "mesh_up_edge_index", "mesh_down_edge_index",
                 "m2m_features", "mesh_up_features", "mesh_down_features"):
        value = getattr(model, name)
        values[name] = digest_tensor(value) if isinstance(value, torch.Tensor) else [digest_tensor(x) for x in value]
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def optimizer_for(training):
    return torch.optim.AdamW(training.parameters(), lr=.001, betas=(.9, .95), weight_decay=.1)


def validation_snapshot(training, transports, backend, batch, initial, updates=2):
    """Full, unmasked AdamW updates, predictions and gradients, outside timing."""
    training.load_state_dict(initial)
    training.train()
    optimizer = optimizer_for(training)
    device = batch[0].device
    generator = torch.Generator(device=device).manual_seed(1042)
    result = []
    with transports.use(backend):
        for _ in range(updates):
            optimizer.zero_grad(set_to_none=True)
            output = training(batch, generator, return_predictions=True)
            output["loss"].backward()
            real._synchronize_gradients(list(training.parameters()), dist.get_world_size())
            gradients = {n: p.grad.detach().cpu().clone() if p.grad is not None else None
                         for n,p in training.named_parameters()}
            loss = output["loss"].detach().clone()
            dist.all_reduce(loss)
            predictions = output["predictions"].detach().cpu()
            optimizer.step()
            result.append({"loss": loss.cpu(), "predictions": predictions, "gradients": gradients,
                           "parameters": {n:p.detach().cpu().clone() for n,p in training.named_parameters()}})
            del output
    optimizer.zero_grad(set_to_none=True)
    del optimizer
    return result


def compare_snapshots(reference, candidate):
    errors = {"max_prediction_absolute_error": 0., "max_scaled_gradient_error": 0.,
              "max_parameter_absolute_error": 0.}
    try:
        for left, right in zip(reference, candidate, strict=True):
            torch.testing.assert_close(right["loss"], left["loss"], rtol=1e-5, atol=1.)
            torch.testing.assert_close(right["predictions"], left["predictions"], rtol=3e-4, atol=3e-5)
            errors["max_prediction_absolute_error"] = max(errors["max_prediction_absolute_error"], float((right["predictions"]-left["predictions"]).abs().max()))
            for name, expected in left["gradients"].items():
                actual = right["gradients"][name]
                if (actual is None) != (expected is None):
                    raise AssertionError(f"{name}: gradient presence differs")
                if expected is not None:
                    scale = max(1., float(expected.abs().max()))
                    torch.testing.assert_close(actual/scale, expected/scale, rtol=5e-4, atol=5e-5, msg=name)
                    errors["max_scaled_gradient_error"] = max(errors["max_scaled_gradient_error"], float((actual-expected).abs().max())/scale)
                a, b = right["parameters"][name], left["parameters"][name]
                torch.testing.assert_close(a, b, rtol=2e-4, atol=2e-4, msg=name)
                errors["max_parameter_absolute_error"] = max(errors["max_parameter_absolute_error"], float((a-b).abs().max()))
        return {"passed": True, "optimizer_updates": len(reference), "adamw_masked_elements": 0, **errors}
    except AssertionError as error:
        return {"passed": False, "error": str(error), **errors}


def cached_batches(dataset, args):
    count = args.cached_batches * args.batch_size
    if len(dataset) < count:
        raise ValueError(f"Need {count} distinct samples, got {len(dataset)}")
    options = real._data_loader_options(args.num_workers, args.prefetch_factor, torch.device("cpu"))
    # With zero workers the rank's main process prepares the cache. If workers
    # are explicitly enabled, exhausting this finite loader stops them first.
    options["persistent_workers"] = False
    loader = DataLoader(Subset(dataset, list(range(count))), batch_size=args.batch_size,
                        shuffle=False, generator=torch.Generator().manual_seed(args.seed), **options)
    batches = [tuple(batch) for batch in loader]
    del loader
    assert len(batches) == args.cached_batches
    return batches


def measured_case(training, transports, backend, batches, initial, args, device):
    CPU_AFFINITY.checkpoint(f"{backend}_before_warmup", repair=True)
    training.load_state_dict(initial)
    training.train()
    optimizer = optimizer_for(training)
    generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    torch.manual_seed(args.seed + 1)
    torch.cuda.empty_cache()
    records = []
    with transports.use(backend):
        for step in range(args.warmup_steps + args.steps):
            if step == args.warmup_steps:
                CPU_AFFINITY.checkpoint(f"{backend}_after_warmup", repair=True)
            copy_start = time.perf_counter()
            batch = tuple(t.to(device, non_blocking=True) for t in batches[step % len(batches)])
            torch.cuda.synchronize(device)
            transfer_ms = (time.perf_counter()-copy_start)*1000
            dist.barrier(); torch.cuda.synchronize(device)
            if step == args.warmup_steps:
                torch.cuda.reset_peak_memory_stats(device)
            begin = time.perf_counter()
            record = real._step_record(training, optimizer, list(training.parameters()), batch,
                                       generator, dist.get_world_size(), batch_is_local=True)
            record["model_wall_ms"] = (time.perf_counter()-begin)*1000
            record["h2d_outside_timer_ms"] = transfer_ms
            record["batch_slot"] = step % len(batches)
            if not math.isfinite(record["loss"]):
                raise RuntimeError("Non-finite loss; reject benchmark")
            if step >= args.warmup_steps:
                record["measurement_step"] = step-args.warmup_steps
                records.append(record)
            del batch
            if dist.get_rank() == 0 and (step == args.warmup_steps-1 or
                    (step >= args.warmup_steps and ((step-args.warmup_steps+1) % 10 == 0 or step+1 == args.warmup_steps+args.steps))):
                real._progress(0, f"{backend}: completed {'warmup' if step<args.warmup_steps else 'measurement'} step "
                               f"{step+1 if step<args.warmup_steps else step-args.warmup_steps+1}")
    CPU_AFFINITY.checkpoint(f"{backend}_after_measurement")
    optimizer.zero_grad(set_to_none=True)
    del optimizer
    torch.cuda.synchronize(device)
    return records


def aggregate(rank_records):
    result = []
    for step, values in enumerate(zip(*rank_records, strict=True)):
        critical = max(range(len(values)), key=lambda r:values[r]["model_wall_ms"])
        row = {key: (values[0][key] if key in ("loss", "negative_likelihood", "kl", "batch_samples", "batch_slot", "measurement_step")
                     else max(v[key] for v in values)) for key in values[0]}
        row["critical_rank"] = critical
        row["critical_rank_phases"] = {k: values[critical][k] for k in
                                       ("forward_ms", "backward_ms", "gradient_sync_ms", "optimizer_ms", "metric_reduce_ms")}
        assert row["measurement_step"] == step
        result.append(row)
    return result


def run(args):
    if os.environ.get("SLURM_JOB_ID") and CPU_AFFINITY.target is None:
        raise RuntimeError("Use the updated Slurm job script to verify CPU placement")
    if not torch.cuda.is_available():
        raise RuntimeError("Official transport timings require CUDA/NCCL; run the CPU tests locally")
    if args.deterministic == "on": os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    else: os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    torch.use_deterministic_algorithms(args.deterministic == "on")
    torch.set_num_threads(1)
    CPU_AFFINITY.checkpoint("after_torch_thread_setup", repair=True)
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    if "1080 Ti" not in torch.cuda.get_device_name(device):
        raise RuntimeError("This controlled suite requires the requested GTX 1080 Ti architecture")
    CPU_AFFINITY.checkpoint("after_cuda_init", repair=True)
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(minutes=5))
    rank, size = dist.get_rank(), dist.get_world_size()
    started = time.perf_counter()
    initialization = {}
    try:
        CPU_AFFINITY.checkpoint("after_nccl_init", repair=True)
        torch.manual_seed(args.seed)
        real._progress(rank, f"Loading common H64/AR4 workload on {size} GPU(s)")
        phase = time.perf_counter()
        _, model, config = real._load_workload(args, include_global_dataset=False)
        graph_hash = graph_digest(model)
        initialization['model_load_ms'] = (time.perf_counter()-phase)*1000
        phase = time.perf_counter()
        frontend, core, partition, _ = real._build_sharded_training_modules(model, args, rank, size, device)
        training = DistributedGraphEFMTraining(frontend, core, partition)
        initial = {k:v.detach().cpu().clone() for k,v in training.state_dict().items()}
        initialization['partition_and_model_build_ms'] = (time.perf_counter()-phase)*1000
        phase = time.perf_counter()
        real._progress(rank, f"Preparing {args.cached_batches} CPU batches with {args.num_workers} workers/rank")
        dataset, rank_root = real._load_rank_local_dataset(args, config, rank, size)
        ids = torch.as_tensor(dataset.datastore._ds.global_surface_index.values, dtype=torch.long)
        if not torch.equal(ids, partition.interior_grid_nodes.cpu()):
            raise ValueError("Shard IDs/order do not match common partition")
        batches = cached_batches(dataset, args)
        initialization['rank_dataset_and_cache_ms'] = (time.perf_counter()-phase)*1000
        real._progress(rank, "CPU cache ready; no loader workers active; preparing common halo plans")
        transports = HaloTransports(partition)
        CPU_AFFINITY.checkpoint("after_model_and_cache", repair=True)
        local_control = {"rank": rank, "batch_hashes": [[digest_tensor(t) for t in batch] for batch in batches],
                         "owned_grid_ids_sha256": digest_tensor(ids), "transport_plan_sha256": transports.signature,
                         "graph_sha256":graph_hash, "initial_state_sha256":state_digest(initial),
                         "partition_hashes":{name:{"edges":digest_tensor(p.edge_ids),
                             "senders":digest_tensor(p.owned_send_nodes), "receivers":digest_tensor(p.owned_rec_nodes),
                             "halos":digest_tensor(p.halo_nodes),
                             "pack_indices":[digest_tensor(v) for v in p.send_local_indices_by_peer]}
                             for name,p in relations(partition)},
                         "partition": real._partition_metadata(partition), "hardware": real._hardware_metadata(device, rank),
                         "cpu_affinity": sorted(os.sched_getaffinity(0)),
                         "cpu_placement": CPU_AFFINITY.report(), "rank_data_root": str(rank_root)}
        rank_controls = [None] * size
        dist.all_gather_object(rank_controls, local_control)
        if len({r['graph_sha256'] for r in rank_controls}) != 1 or len({r['initial_state_sha256'] for r in rank_controls}) != 1:
            raise ValueError("Ranks disagree about original graph or initial state")
        controls = {"graph_sha256": graph_hash, "initial_state_sha256": state_digest(initial),
                    "sample_indices": list(range(args.cached_batches*args.batch_size)), "world_size": size,
                    "model": vars(real._model_args(args)), "workers_per_rank": args.num_workers,
                    "prefetch_factor": args.prefetch_factor if args.num_workers > 0 else None, "pin_memory": False,
                    "deterministic_algorithms": args.deterministic == "on", "warmup_steps": args.warmup_steps,
                    "distribution_validate_args": bool(torch.distributions.Distribution._validate_args),
                    "measurement_steps": args.steps, "checkpoint_early_stop": False,
                    "cached_workers_active_during_timing": False,
                    "rank_input_hashes": [r["batch_hashes"] for r in rank_controls],
                    "rank_partition_hashes": [r["partition_hashes"] for r in rank_controls],
                    "transport_plans": transports.metadata}
        if CPU_AFFINITY.target is not None:
            rank_cpus = [r["cpu_placement"]["rank_cpu_ids"] for r in rank_controls]
            flat_cpus = [cpu for ids in rank_cpus for cpu in ids]
            if sorted(flat_cpus) != CPU_AFFINITY.job_cpus or len(set(flat_cpus)) != len(flat_cpus):
                raise RuntimeError("Rank CPU masks must cover the job allocation without overlap")
            controls["cpu_placement"] = {"policy": POLICY,
                "job_cpu_ids": CPU_AFFINITY.job_cpus, "rank_cpu_ids": rank_cpus}
        source_paths = [Path(__file__), REPO/"benchmarks/halo_transport_backends.py", REPO/"neural_lam/distributed_message_passing.py",
                        REPO/"neural_lam/models/distributed_graph_efm_training.py", REPO/"neural_lam/models/distributed_graph_efm.py",
                        REPO/"neural_lam/models/graph_efm.py", REPO/"benchmarks/model_parallel_real.py",
                        REPO/"benchmarks/vendor/physicsnemo/distributed/autograd.py", REPO/"benchmarks/vendor/physicsnemo/distributed/utils.py",
                        REPO/"benchmarks/cpu_affinity.py", REPO/"jobs/halo_transport_comparison.sbatch"]
        controls["source_hashes"] = {str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
        control_hash = hashlib.sha256(json.dumps(controls, sort_keys=True, default=str).encode()).hexdigest()
        if rank == 0: print(f"Common cache/graph ready; repeat={args.repeat}, controls={control_hash}", flush=True)
        phase = time.perf_counter()
        real._progress(rank, "Checking P2P and PhysicsNeMo predictions, gradients and two optimizer updates")
        validation_batch = tuple(t.to(device) for t in batches[0])
        reference = validation_snapshot(training, transports, "p2p", validation_batch, initial)
        candidate = validation_snapshot(training, transports, "physicsnemo", validation_batch, initial)
        validation = compare_snapshots(reference, candidate)
        validation_reports = [None] * size
        dist.all_gather_object(validation_reports, validation)
        del reference, candidate, validation_batch
        initialization['equivalence_gate_ms'] = (time.perf_counter()-phase)*1000
        if not all(v["passed"] for v in validation_reports):
            if rank == 0: atomic_json(args.output, {"status":"validation_failed", "validation": validation_reports, "controls": controls})
            raise RuntimeError("Backend equivalence gate failed; no timing accepted")
        if rank == 0: print("PASS: AR4 predictions/loss/gradients and two unmasked AdamW updates", flush=True)
        order = list(BACKENDS if args.repeat % 2 == 0 else reversed(BACKENDS))
        cases = {}
        for backend in order:
            if rank == 0: print(f"Timing {backend} after resetting weights, optimizer and latent RNG", flush=True)
            local = measured_case(training, transports, backend, batches, initial, args, device)
            payload = [None] * size
            dist.all_gather_object(payload, local)
            if rank == 0:
                records = aggregate(payload)
                summary = {k:real._summary([r[k] for r in records]) for k in
                           ("model_wall_ms", "step_ms", "forward_ms", "backward_ms", "gradient_sync_ms",
                            "optimizer_ms", "metric_reduce_ms", "h2d_outside_timer_ms", "loss")}
                cases[backend] = {"records":records, "rank_records":payload, "summary":summary,
                                  "peak_allocated_bytes_per_gpu":max(r["peak_memory_allocated_bytes"] for r in records)}
                print(f"{backend}: model wall median={summary['model_wall_ms']['median']:.3f} ms", flush=True)
        affinity_reports = [None] * size
        dist.all_gather_object(affinity_reports, CPU_AFFINITY.report())
        if rank == 0:
            atomic_json(args.output, {"status":"completed", "schema":"halo-transport-paired-v1", "repeat":args.repeat,
                         "order":order, "controls":controls, "controls_sha256":control_hash,
                         "initialization_ms":initialization,
                         "validation":validation_reports, "rank_metadata":rank_controls,
                         "cpu_affinity_reports":affinity_reports,
                         "git":real._git_metadata(), "cases":cases, "elapsed_seconds":time.perf_counter()-started,
                         "interpretation":{"timer":"Cached model wall step after H2D and rank barrier; excludes input/setup",
                             "backend_change":"Remote-only halo primitive; common original P2P graph layout and AR4 implementation",
                             "one_gpu":"Both use same zero-traffic local path; checks baseline stability, not network performance",
                             "cpu_emulation":False, "native_backend":dist.get_backend()}})
            print(f"Result: {args.output}", flush=True)
    finally:
        dist.destroy_process_group()


def summarize(directory, repeats):
    paths = [directory/f"repeat-{i}.json" for i in range(repeats)]
    results = [json.loads(p.read_text()) for p in paths]
    if any(r["status"] != "completed" for r in results): raise ValueError("Incomplete or failed repeat")
    if len({r["controls_sha256"] for r in results}) != 1: raise ValueError("Controls differ across independent launches")
    if len({json.dumps(r['controls']['source_hashes'], sort_keys=True) for r in results}) != 1: raise ValueError("Source changed between launches")
    if any(not all(v['passed'] for v in r['validation']) for r in results): raise ValueError("Validation failed")
    placement = results[0]['controls'].get('cpu_placement')
    cpu_checks = None
    if placement is not None:
        required = {f'{backend}_after_measurement' for backend in BACKENDS}
        for result in results:
            reports = result.get('cpu_affinity_reports', [])
            if len(reports) != result['controls']['world_size']:
                raise ValueError('CPU placement reports missing for one or more ranks')
            for rank, audit in enumerate(reports):
                stages = {c['stage'] for c in audit['checkpoints'] if c['passed']}
                if (audit['rank'] != rank or audit['policy'] != placement['policy']
                        or audit['job_cpu_ids'] != placement['job_cpu_ids']
                        or audit['rank_cpu_ids'] != placement['rank_cpu_ids'][rank]
                        or not required.issubset(stages)
                        or not all(c['passed'] for c in audit['checkpoints'])):
                    raise ValueError('CPU placement audit failed; reject these timings')
        cpu_checks = True
    medians = {b:[r['cases'][b]['summary']['model_wall_ms']['median'] for r in results] for b in BACKENDS}
    ratio = statistics.median(medians['p2p']) / statistics.median(medians['physicsnemo'])
    rng = __import__('random').Random(42)
    ratios = []
    for _ in range(4000):
        ids=[rng.randrange(repeats) for _ in range(repeats)]
        ratios.append(statistics.median(medians['p2p'][i] for i in ids)/statistics.median(medians['physicsnemo'][i] for i in ids))
    ratios.sort()
    report={"status":"completed", "world_size":results[0]['controls']['world_size'], "independent_paired_launches":repeats,
            "workers_per_rank":results[0]['controls'].get('workers_per_rank'),
            "prefetch_factor":results[0]['controls'].get('prefetch_factor'),
            "cpu_placement":placement, "all_cpu_affinity_checks_passed":cpu_checks,
            "controls_sha256":results[0]['controls_sha256'], "per_launch_model_wall_medians_ms":medians,
            "physicsnemo_speedup_over_p2p":ratio, "paired_launch_bootstrap_ci95":[ratios[99],ratios[3899]] if repeats>1 else None,
            "ci_scope":"Paired independent launches only; three repeats provide limited uncertainty resolution",
            "all_equivalence_checks_passed":True,
            "source_files":[str(p) for p in paths], "timer":results[0]['interpretation']['timer']}
    atomic_json(directory/'comparison.json',report)
    print(json.dumps(report,indent=2))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest="command",required=True)
    p=commands.add_parser("run")
    p.add_argument("--config-path",type=Path,required=True)
    p.add_argument("--partitioned-data-root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--repeat",type=int,default=0)
    p.add_argument("--warmup-steps",type=int,default=10)
    p.add_argument("--steps",type=int,default=50)
    p.add_argument("--cached-batches",type=int,default=4)
    p.add_argument("--num-workers",type=int,default=0,
                   help="CPU cache preparation workers per rank; 0 uses the rank's main process")
    p.add_argument("--prefetch-factor",type=int,default=2)
    p.add_argument("--deterministic",choices=("on","off"),default="on")
    p.set_defaults(graph="global_cluster_graph", hidden_dim=64, processor_layers=2, batch_size=2,
                   ar_steps=4, lr=.001, seed=42, partition_strategy="geographic", latitude_cut=-5., longitude_seam=120.)
    p=commands.add_parser("summarize")
    p.add_argument("--directory",type=Path,required=True)
    p.add_argument("--repeats",type=int,required=True)
    args=parser.parse_args()
    if args.command=="summarize":
        if args.repeats<1:parser.error("repeats must be positive")
        summarize(args.directory,args.repeats)
    else:
        if min(args.steps,args.cached_batches,args.prefetch_factor)<1 or min(args.warmup_steps,args.num_workers,args.repeat)<0:
            parser.error("Invalid count")
        run(args)


if __name__=="__main__":main()
