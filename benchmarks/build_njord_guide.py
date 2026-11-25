"""Build an offline English guide from source and immutable run archives.

Reads results and traces one synthetic CPU forecast for structural counts;
never trains, modifies the model, or submits a cluster job.
"""

import contextlib
import csv
import hashlib
import io
import json
import math
import re
import statistics
import subprocess
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from benchmarks.compare_model_parallel_real import WORKLOAD_KEYS, _fingerprints_match

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "benchmark_results/model-parallel-real"
OUT = BASE / "analysis/njord-guide-2026-11-23"
SITE = REPO.parent / "njord-message-passing-website"
PHASES = ("data_distribution_ms", "data_read_ms", "host_to_device_ms",
          "batch_broadcast_ms", "data_rank_wait_ms", "forward_ms", "backward_ms",
          "gradient_sync_ms", "optimizer_ms", "compute_ms", "metric_reduce_ms",
          "step_ms", "end_to_end_ms")
STACK = ("data_distribution_ms", "forward_ms", "backward_ms",
         "gradient_sync_ms", "optimizer_ms", "metric_reduce_ms")
CURRENT = {
    "4981": ("Original · full store", "Original", "Full grid", BASE / "baseline/original-ar4-workers8-prefetch2-4981/original-graph-efm-ar4-1gpu-4981.json"),
    "5031": ("Original · ocean shard", "Original", "Ocean shard", BASE / "baseline/original-ar4-ocean-shard-workers8-prefetch2-5031/original-graph-efm-ar4-ocean-shard-1gpu-5031.json"),
    **{str(job): (f"MP · {n} GPU" + ("s" if n > 1 else ""), "MP", "Ocean shards",
       BASE / f"full/ar4-workers8-prefetch2-4944-4949-4952/real-mp-max-ar4-ckpt-full-rank-local-{n}gpu-{job}.json")
       for n, job in ((1, 4944), (2, 4949), (4, 4952))},
}


def git(*args):
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def mean(records, key):
    return statistics.fmean(record[key] for record in records)


def describe(path):
    r = json.loads(path.read_text())
    w, s = r["workload"], r["summary"]
    records, ranks = r["records"], r["rank_metadata"]
    critical = [max((rank["records"][i] for rank in ranks), key=lambda x:x["end_to_end_ms"])
                for i in range(len(records))]
    timings = {k:mean(critical,k) for k in STACK}
    assert math.isclose(sum(timings.values()),s["end_to_end_ms"]["mean"],abs_tol=.03)
    job = str(r.get("slurm_job_id", ""))
    info = CURRENT.get(job)
    throughput = s.get("observed_samples_per_second",s["end_to_end_samples_per_second"])
    sample_count = s.get("measured_samples", sum(x.get("batch_samples", w["batch_size_global"]) for x in records))
    return {
        "job":job, "label":info[0] if info else f"Job {job}", "model":info[1] if info else "MP",
        "input":info[2] if info else w.get("data_mode","full / legacy"),
        "gpus":r["world_size"], "ar":w.get("ar_steps",1), "workers":w.get("num_workers"),
        "prefetch":w.get("prefetch_factor"), "steps":w["measurement_steps"],
        "warmup":w["warmup_steps"], "samples":sample_count, "throughput":throughput,
        "peakGiB":s.get("peak_memory_allocated_bytes_per_gpu",s.get("peak_memory_bytes_per_gpu",0))/2**30,
        "reservedGiB":s.get("peak_memory_reserved_bytes_per_gpu",0)/2**30,
        "published":{k:s[k]["mean"] for k in PHASES if k in s}, "critical":timings,
        "inputComponents":{k:mean(critical,k) for k in ("data_read_ms","host_to_device_ms","data_rank_wait_ms") if k in critical[0]},
        "e2eP95":s["end_to_end_ms"]["p95"], "cv":s["end_to_end_ms"]["std"]/s["end_to_end_ms"]["mean"],
        "gpuSeconds":r["world_size"]/throughput,
        "visibleInputShare":timings["data_distribution_ms"]/s["end_to_end_ms"]["mean"],
        "meanLoss":s["loss"]["mean"], "preflight":r["preflight"]["loss"],
        "loss":[x["loss"] for x in records], "times":[x["end_to_end_ms"] for x in records],
        "blocks":[mean(records[i:i+10],"end_to_end_ms") for i in range(0,len(records),10)],
        "hostname":ranks[0]["hardware"]["hostname"], "hardware":ranks[0]["hardware"],
        "checkpointEarlyStop":w.get("checkpoint_early_stop"), "git":r["git"]["commit"],
        "source":str(path.relative_to(REPO)), "sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
        "stagingSeconds":ranks[0].get("initialization_timings",{}).get("data_staging_ms",0)/1000,
        "initSeconds":max(x.get("initialization_timings",{}).get("total_before_training_ms",0) for x in ranks)/1000,
        "windowSeconds":s.get("observed_window_ms",0)/1000,
        "rankPartitions":[x.get("partition") for x in ranks],
    }


def load_runs():
    raw = [json.loads(info[3].read_text()) for info in CURRENT.values()]
    a = raw[0]
    for r in raw:
        assert r["status"] == "completed" and not r["git"]["tracked_dirty"]
        assert len(r["records"]) == 50 and r["summary"]["measured_samples"] == 100
        diffs = {k for k in WORKLOAD_KEYS if a["workload"].get(k)!=r["workload"].get(k)}
        assert diffs <= {"data_mode","checkpoint_early_stop","data_phase_rank_barrier"}, diffs
        assert _fingerprints_match(a["preflight"]["batch_fingerprint"],r["preflight"]["batch_fingerprint"])
        for left,right in zip(a["records"],r["records"]):
            assert math.isclose(left["loss"],right["loss"],rel_tol=1e-4,abs_tol=.01)
        for filename,blob in a["original_source"]["matching_git_blobs"].items():
            assert git("rev-parse",f"{r['git']['commit']}:{filename}")==blob
    current = [describe(info[3]) for info in CURRENT.values()]
    for row in current:
        row["maxLossError"] = max(abs(x-y)/max(abs(x),abs(y),1e-12)
                                 for x,y in zip(current[0]["loss"],row["loss"]))
        row["exactLoss"] = current[0]["loss"] == row["loss"]
    assert current[0]["exactLoss"] and current[1]["exactLoss"] and current[2]["exactLoss"]
    historic = []
    for path in sorted(BASE.rglob("*.json")):
        if "/analysis/" in str(path) or "/plots/" in str(path):continue
        r = json.loads(path.read_text())
        if r.get("benchmark")!="real_data_model_parallel_graph_efm":continue
        if str(r.get("slurm_job_id")) in CURRENT:continue
        row=describe(path)
        row["family"] = "Smoke" if row["steps"]<=1 else "Earlier AR1"
        historic.append(row)
    functional=[]
    for path in sorted((BASE/"smoke/ar4-workers8-prefetch2-4941-4948-4950").glob("ar4-equivalence*.json")):
        r=json.loads(path.read_text())
        functional.append({"file":str(path.relative_to(REPO)),"status":r["status"],"ranks":r["world_size"],"backend":r["backend"]})
    return current,historic,functional


def trace_and_map(current):
    from benchmarks.model_parallel_real import _load_workload
    from neural_lam.models.distributed_graph_efm import _full_grid_lon_lat, _mesh_level_lon_lat
    from neural_lam.geographic_partitioning import geographic_rcb_owners

    torch.set_num_threads(2);torch.manual_seed(42)
    args=SimpleNamespace(config_path=REPO/"data/physical_repeat_16/global_ocean_1.yaml",
        graph="global_cluster_graph",ar_steps=4,hidden_dim=64,processor_layers=2,batch_size=2,lr=.001,seed=42)
    with contextlib.redirect_stdout(io.StringIO()):
        _,model,_=_load_workload(args,include_global_dataset=False)
    model.eval();model.log_dict=lambda *a,**k:None
    trace=[];handles=[]
    def hook(name):
        def record(module,inputs,output):
            propagation=type(module).__name__=="PropagationNet" or getattr(module,"propagation",False)
            family=("g2m" if "g2m_gnn" in name else "m2g" if "m2g_gnn" in name else
                "up" if "mesh_up_gnns" in name else "down" if "mesh_down_gnns" in name else
                "m1" if re.search(r'intra_(?:level|up|down)_gnns\.1\.',name) else "m0")
            trace.append({"module":name,"branch":"posterior" if name.startswith("encoder.") else
                "prior" if name.startswith("prior_model.") else "decoder", "family":family,
                "class":type(module).__name__,"aggregation":str(module.aggr),"propagation":bool(propagation),
                "updateEdges":bool(getattr(module,"update_edges",False)),
                "senders":inputs[0].shape[-2],"receivers":inputs[1].shape[-2],"edges":inputs[2].shape[-2],
                "hidden":inputs[0].shape[-1]})
        return record
    for name,module in model.named_modules():
        if type(module).__name__ in ("FlexibleNet","InteractionNet","PropagationNet"):
            handles.append(module.register_forward_hook(hook(name)))
    n=model.num_interior_nodes
    prev=torch.zeros(1,2,n,28);target=torch.zeros(1,n,28)
    forcing=torch.zeros(1,n,6);atmosphere=torch.zeros(1,n,24);boundary=torch.empty(1,61200,0)
    with torch.no_grad():model.compute_step_loss(prev,target,forcing,boundary,atmosphere)
    training=list(trace);trace.clear()
    with torch.no_grad():
        grid,interior,graph=model.embedd_all(prev[:,-1],prev[:,0],forcing,boundary,atmosphere)
        prior=model.prior_model(grid,graph_emb=graph)
        model.decoder(grid,interior,prior.rsample(),prev[:,-1],graph)
    inference=list(trace)
    for h in handles:h.remove()
    assert len(training)==18,(len(training),training)
    assert len(inference)==14,(len(inference),inference)
    assert Counter(x["family"] for x in training)==dict(g2m=3,m0=6,up=3,m1=4,down=1,m2g=1)
    assert [x["branch"] for x in training]==["posterior"]*4+["decoder"]*10+["prior"]*4
    counts={"grid":int(n),"mesh0":model.mesh_static_features[0].shape[0],"mesh1":model.mesh_static_features[1].shape[0],
            "g2m":model.g2m_edge_index.shape[1],"m2g":model.m2g_edge_index.shape[1],
            "m0":model.m2m_edge_index[0].shape[1],"m1":model.m2m_edge_index[1].shape[1],
            "up":model.mesh_up_edge_index[0].shape[1],"down":model.mesh_down_edge_index[0].shape[1]}
    levels=[_full_grid_lon_lat(model),*_mesh_level_lon_lat(model)]
    snapshot={"description":"Sampled real longitude/latitude node positions; owners from the measured geographic RCB implementation.","layers":{}}
    for name,coords in zip(("grid","mesh0","mesh1"),levels):
        owners={str(p):geographic_rcb_owners(coords,p) for p in (1,2,4)}
        indices=torch.arange(0,len(coords),max(1,math.ceil(len(coords)/2200)))
        snapshot["layers"][name]={"total":len(coords),"points":[
            [round(float(coords[i,0]),3),round(float(coords[i,1]),3),int(owners['2'][i]),int(owners['4'][i])]
            for i in indices],"counts":{str(p):torch.bincount(owners[str(p)],minlength=p).tolist() for p in (1,2,4)}}
    for row in current:
        if row["model"]!="MP":continue
        p=row["gpus"]
        for rank,partition in enumerate(row["rankPartitions"]):
            assert snapshot["layers"]["grid"]["counts"][str(p)][rank]==partition["owned_interior_grid_nodes"]
            for level,key in enumerate(("mesh0","mesh1")):
                assert snapshot["layers"][key]["counts"][str(p)][rank]==partition["owned_mesh_nodes_by_level"][level]
    return {"training":training,"inference":inference,"counts":counts,
            "method":"One synthetic zero-input CPU forecast on original H64 model, forward hooks on actual graph operators; structural counts only, not performance or forecast skill."},snapshot


def main():
    OUT.mkdir(parents=True,exist_ok=True);SITE.mkdir(exist_ok=True)
    current,historic,functional=load_runs()
    trace,snapshot=trace_and_map(current)
    refs={name:git("rev-parse",f"origin/{name}") for name in
          ("alex","lassi-partition-benchmark","lassi-graph-viewer","lassi-profiling","nils","max")}
    data={"asOf":"23 November 2026","git":git("rev-parse","HEAD"),"current":current,
          "historical":historic,"functional":functional,"trace":trace,"geography":snapshot,
          "teamRefs":refs,"validated":True}
    for name,obj in (("guide-data.json",data),("model-trace.json",trace),("graph-snapshot.json",snapshot)):
        (OUT/name).write_text(json.dumps(obj,indent=2)+"\n")
    with (OUT/"run-summary.csv").open("w",newline="") as stream:
        fields=["job","label","gpus","ar","workers","prefetch","steps","throughput","peakGiB",*PHASES]
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for r in current+historic:
            writer.writerow({**{k:r[k] for k in fields if k not in PHASES},**r["published"]})
    from benchmarks.render_njord_guide import write_guide
    write_guide(data)
    print(json.dumps({"html":str(SITE/"njord-data-model-results.html"),"portable":str(OUT/"njord-data-model-results.html"),
        "mainRuns":len(current),"historicalAndSmokeRuns":len(historic),"trainingRelations":len(trace["training"]),
        "inferenceRelations":len(trace["inference"]),"counts":trace["counts"]},indent=2))


if __name__=="__main__":main()
