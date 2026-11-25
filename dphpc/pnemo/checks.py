"""PhysicsNeMo transport, edge ordering and original-GraphEFM equivalence."""

import argparse
import copy
from datetime import timedelta
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from .distributed_graph_efm import DistributedGraphEFM
from .partitions import DEFAULT, CONFIG, GRAPH, METHODS, graph_data, load_owners
from .workload import load_workload


def reference_model(template, source, config, device):
    model = copy.deepcopy(template)
    model.prepare_clamping_params(config, source.datastore)
    model.log_dict = lambda *a, **kw: None
    return model.to(device)


def original_training(model, batch, seed):
    predictions = []
    original = model.compute_step_loss

    def capture(*args):
        result = original(*args)
        if len(predictions) < batch[1].shape[1]:
            predictions.append(result[2].detach())
        return result

    model.compute_step_loss = capture
    torch.manual_seed(seed)
    loss = model.training_step(batch)
    model.compute_step_loss = original
    return loss, torch.stack(predictions[: batch[1].shape[1]], dim=1)


@torch.no_grad()
def check_sgd_step(pairs, lr):
    copies = []
    for _, distributed, original in pairs:
        local = torch.nn.Parameter(distributed.detach().clone())
        reference = torch.nn.Parameter(original.detach().clone())
        local.grad = distributed.grad
        reference.grad = original.grad
        copies.append((local, reference))
    torch.optim.SGD([p[0] for p in copies], lr=lr).step()
    torch.optim.SGD([p[1] for p in copies], lr=lr).step()
    for (name, distributed, original), (local, reference) in zip(pairs, copies):
        scale = (
            max(1.0, float(original.grad.abs().max()))
            if original.grad is not None
            else 1.0
        )
        # Compare actual SGD updates in gradient units at the unchanged tolerances.
        torch.testing.assert_close(
            (distributed - local) / lr / scale,
            (original - reference) / lr / scale,
            rtol=2e-3,
            atol=2e-5,
            msg=lambda s: f"Native SGD {name}: {s}",
        )


def check_case(
    args, template, source, config, batch, owners, relations, device, method
):
    model = DistributedGraphEFM(
        template, owners, relations, config, source.datastore, device
    )
    # API feature ordering: CSC permutation then library edge selection.
    for key, relation in model.relations.items():
        ids = torch.arange(
            relations[key][2].shape[1], dtype=torch.float32
        ).reshape(-1, 1)
        selected = relation.edge_features(ids)
        torch.testing.assert_close(
            selected.cpu().flatten(), relation.edge_ids.cpu().float()
        )
    result = dict(
        partitioner=method,
        K=dist.get_world_size(),
        rank=dist.get_rank(),
        relations=len(relations),
        edge_mapping="passed",
    )
    local = model.local_batch(batch)
    reference = reference_model(template, source, config, device).eval()
    seed = args.seed + 1000
    with torch.no_grad():
        torch.manual_seed(seed)
        expected = reference.unroll_prediction(
            batch[0], batch[2], batch[3], batch[4]
        )[0]
        actual = model.forecast(local, seed)
        torch.testing.assert_close(
            actual,
            expected.index_select(-2, model.nodes["grid"]),
            rtol=5e-4,
            atol=1e-4,
        )
        assert all(p.grad is None for p in model.parameters())
    result["forecast"] = "passed"
    if dist.get_rank() == 0:
        print(f"PASS {method}: forecast", flush=True)
    del reference, expected, actual, model
    for checkpointing in (False, True):
        reference = reference_model(template, source, config, device).train()
        reference.grad_checkpointing = (
            True  # Memory-bounded original; checkpointing preserves semantics.
        )
        model = DistributedGraphEFM(
            template, owners, relations, config, source.datastore, device
        )
        local = model.local_batch(batch)
        loss, predictions = original_training(reference, batch, seed)
        actual = model.training_result(local, seed, checkpointing)
        torch.testing.assert_close(actual["loss"], loss, rtol=2e-5, atol=1.0)
        torch.testing.assert_close(
            actual["predictions"],
            predictions.index_select(-2, model.nodes["grid"]),
            rtol=5e-4,
            atol=1e-4,
        )
        del predictions, actual["predictions"]
        loss.backward()
        actual["loss"].backward()
        model.synchronize_gradients()
        pairs = list(model.parameter_pairs(reference))
        gradient_error = 0.0
        for name, distributed, original in pairs:
            assert (distributed.grad is None) == (original.grad is None), name
            if original.grad is not None:
                scale = max(1.0, float(original.grad.abs().max()))
                torch.testing.assert_close(
                    distributed.grad / scale,
                    original.grad / scale,
                    rtol=2e-3,
                    atol=2e-5,
                    msg=lambda s: f"{name}: {s}",
                )
                gradient_error = max(
                    gradient_error,
                    float((distributed.grad - original.grad).abs().max())
                    / scale,
                )
        if dist.get_rank() == 0:
            print(
                f"PASS {method}: checkpointing={checkpointing} outputs/loss/gradients; "
                f"max scaled gradient error={gradient_error:.8g}",
                flush=True,
            )
        check_sgd_step(pairs, args.lr)
        original_opt = torch.optim.AdamW(
            reference.parameters(),
            lr=args.lr,
            betas=(0.9, 0.95),
            weight_decay=0.1,
        )
        distributed_opt = torch.optim.AdamW(
            model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1
        )
        before_step = {
            name: (distributed.detach().clone(), original.detach().clone())
            for name, distributed, original in pairs
        }
        original_opt.step()
        distributed_opt.step()
        adam_skipped = adam_compared = 0
        for name, distributed, original in pairs:
            if original.grad is None:
                mask = torch.ones_like(original, dtype=torch.bool)
            else:
                floor = 2e-5 * max(1.0, float(original.grad.abs().max()))
                mask = original.grad.abs() > floor
                adam_skipped += int((~mask).sum())
            adam_compared += int(mask.sum())
            try:
                torch.testing.assert_close(
                    distributed[mask],
                    original[mask],
                    rtol=5e-4,
                    atol=5e-5,
                    msg=lambda s: f"Native AdamW {name}: {s}",
                )
            except AssertionError:
                index = int(
                    ((distributed - original).abs() * mask).flatten().argmax()
                )
                distributed_before, original_before = before_step[name]
                grad_ref = float(original.grad.flatten()[index])
                grad_dist = float(distributed.grad.flatten()[index])
                grad_tolerance = 2e-5 * max(
                    1.0, float(original.grad.abs().max())
                )
                failure = dict(
                    result,
                    status="failed",
                    phase="native_optimizer_step",
                    checkpointing=checkpointing,
                    outputs="passed",
                    loss="passed",
                    gradients="passed",
                    max_scaled_gradient_error=gradient_error,
                    parameter=name,
                    flat_index=index,
                    reference_gradient=grad_ref,
                    distributed_gradient=grad_dist,
                    gradient_absolute_tolerance=grad_tolerance,
                    near_zero_sign_flip=(
                        abs(grad_ref) <= grad_tolerance
                        and abs(grad_dist) <= grad_tolerance
                        and grad_ref * grad_dist < 0
                    ),
                    reference_gradient_max=float(original.grad.abs().max()),
                    reference_before=float(original_before.flatten()[index]),
                    distributed_before=float(
                        distributed_before.flatten()[index]
                    ),
                    reference_weight=float(original.flatten()[index].detach()),
                    distributed_weight=float(
                        distributed.flatten()[index].detach()
                    ),
                    reference_update=float(
                        (original.detach() - original_before).flatten()[index]
                    ),
                    distributed_update=float(
                        (distributed.detach() - distributed_before).flatten()[
                            index
                        ]
                    ),
                    reference_exp_avg=float(
                        original_opt.state[original]["exp_avg"].flatten()[index]
                    ),
                    reference_exp_avg_sq=float(
                        original_opt.state[original]["exp_avg_sq"].flatten()[
                            index
                        ]
                    ),
                    distributed_exp_avg=float(
                        distributed_opt.state[distributed]["exp_avg"].flatten()[
                            index
                        ]
                    ),
                    distributed_exp_avg_sq=float(
                        distributed_opt.state[distributed][
                            "exp_avg_sq"
                        ].flatten()[index]
                    ),
                )
                if dist.get_rank() == 0:
                    print(json.dumps(failure, indent=2), flush=True)
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(json.dumps(failure, indent=2) + "\n")
                raise
        if dist.get_rank() == 0:
            print(
                f"PASS {method}: native SGD and masked AdamW; "
                f"AdamW compared={adam_compared}, skipped={adam_skipped}",
                flush=True,
            )
        result["checkpoint_on" if checkpointing else "checkpoint_off"] = dict(
            reference_checkpointing="on",
            outputs="passed",
            loss="passed",
            gradients="passed",
            native_sgd_step="passed",
            native_adamw_masked_step="passed",
            adamw_compared_elements=adam_compared,
            adamw_skipped_elements=adam_skipped,
            adamw_mask="abs(reference gradient) > 2e-5 * max(1, tensor max abs gradient)",
            max_scaled_gradient_error=gradient_error,
        )
        del (
            model,
            reference,
            actual,
            loss,
            pairs,
            distributed_opt,
            original_opt,
            before_step,
        )
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=["gloo", "nccl"], default="gloo")
    p.add_argument("--partitions", type=Path, default=DEFAULT)
    p.add_argument(
        "--output",
        type=Path,
        default=Path("dphpc/pnemo/results/equivalence.json"),
    )
    p.add_argument("--config-path", type=Path, default=CONFIG)
    p.add_argument("--graph", default=GRAPH)
    p.add_argument(
        "--strategies", nargs="+", choices=METHODS, default=list(METHODS)
    )
    for name, default in [
        ("hidden-dim", 4),
        ("processor-layers", 1),
        ("batch-size", 1),
        ("ar-steps", 4),
        ("seed", 42),
    ]:
        p.add_argument("--" + name, type=int, default=default)
    p.add_argument("--lr", type=float, default=0.001)
    args = p.parse_args()
    rank = int(os.environ["RANK"])
    K = int(os.environ["WORLD_SIZE"])
    torch.set_num_threads(1)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    device = (
        torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        if args.backend == "nccl"
        else torch.device("cpu")
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dist.init_process_group(args.backend, timeout=timedelta(minutes=10))
    reports = []
    try:
        info = graph_data(args.config_path, args.graph)
        torch.manual_seed(args.seed)
        source, template, config = load_workload(args)
        batch = tuple(
            t.to(device)
            for t in next(
                iter(
                    DataLoader(
                        source, batch_size=args.batch_size, num_workers=0
                    )
                )
            )
        )
        for method in args.strategies:
            if K == 1:
                owners = {
                    key: torch.zeros(size, dtype=torch.long)
                    for key, size in info[1].items()
                }
            else:
                owners, _ = load_owners(args.partitions, method, K, info)
            for node_set, mapping in owners.items():
                ratio = float(
                    torch.bincount(mapping, minlength=K).max()
                ) / (len(mapping) / K)
                assert ratio <= (
                    1.05 if node_set == "grid" else 1.25
                ), f"{method} K={K}: {node_set} imbalance {ratio:.3f} exceeds acceptance"
            reports.append(
                check_case(
                    args,
                    template,
                    source,
                    config,
                    batch,
                    owners,
                    info[2],
                    device,
                    method,
                )
            )
            print(f"PASS {method} K={K} rank={rank}", flush=True)
        gathered = [None] * K
        dist.all_gather_object(gathered, reports)
        if rank == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(
                    dict(
                        status="passed",
                        backend=args.backend,
                        ranks=gathered,
                    ),
                    indent=2,
                )
                + "\n"
            )
    except RuntimeError as error:
        message = str(error)
        unsupported = args.backend == "gloo" and (
            "does not support alltoall" in message.lower()
            or (
                "ProcessGroupGloo::alltoall" in message
                and "invalid tensor size" in message
            )
        )
        if not unsupported:
            raise
        print(
            "UNSUPPORTED: Gloo cannot perform PhysicsNeMo's unequal-sized list all_to_all; NCCL equivalence required.",
            flush=True,
        )
        if rank == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(
                    dict(
                        status="unsupported",
                        backend="gloo",
                        reason=message,
                        nccl_equivalence_required=True,
                    ),
                    indent=2,
                )
                + "\n"
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
