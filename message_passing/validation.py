"""Compare against original GraphEFM, including two unmasked AdamW updates."""

# Third-party
import torch
import torch.distributed as dist
from torch.utils.checkpoint import set_checkpoint_early_stop


def optimizer(model, lr=0.001):
    return torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1
    )


def original_result(model, batch, seed):
    predictions = []
    original = model.compute_step_loss

    def capture(*args):
        output = original(*args)
        if len(predictions) < batch[1].shape[1]:
            predictions.append(output[2].detach())
        return output

    model.compute_step_loss = capture
    model.log_dict = lambda *a, **kw: None
    devices = [batch[0].device.index] if batch[0].is_cuda else []
    try:
        with (
            torch.random.fork_rng(devices=devices),
            set_checkpoint_early_stop(False),
        ):
            torch.manual_seed(seed)
            loss = model.training_step(batch)
    finally:
        model.compute_step_loss = original
    return {"loss": loss, "predictions": torch.stack(predictions, 1)}


def validate_original(
    reference, candidate, full_batch, local_batch, seed=1042, lr=0.001
):
    """Start at initial states; the caller restores both models afterward."""
    for i, (full, local) in enumerate(
        zip(full_batch, local_batch, strict=True)
    ):
        if i in (0, 1, 2, 4):
            full = full.index_select(-2, candidate.nodes["grid"])
        if full.numel() or local.numel():
            torch.testing.assert_close(
                local, full, rtol=0, atol=0, equal_nan=True
            )
    reference.train()
    candidate.train()
    ref_opt, local_opt = optimizer(reference, lr), optimizer(candidate, lr)
    errors = {
        "prediction_max_absolute": 0.0,
        "gradient_max_scaled": 0.0,
        "parameter_max_absolute": 0.0,
    }
    losses = []
    for update in range(2):
        ref_opt.zero_grad(set_to_none=True)
        local_opt.zero_grad(set_to_none=True)
        expected = original_result(reference, full_batch, seed + update)
        expected["loss"].backward()
        actual = candidate(local_batch, seed + update, capture_predictions=True)
        selected = expected["predictions"].index_select(
            -2, candidate.nodes["grid"]
        )
        torch.testing.assert_close(
            actual["predictions"], selected, rtol=3e-4, atol=3e-5
        )
        total = actual["loss"].detach().clone()
        dist.all_reduce(total)
        torch.testing.assert_close(
            total, expected["loss"], rtol=1e-5, atol=1e-4
        )
        losses.append(float(total))
        errors["prediction_max_absolute"] = max(
            errors["prediction_max_absolute"],
            float((actual["predictions"] - selected).abs().max()),
        )
        actual["loss"].backward()
        candidate.synchronize_gradients()
        for name, local, full in candidate.parameter_pairs(reference):
            if (local.grad is None) != (full.grad is None):
                raise AssertionError(f"Gradient presence differs: {name}")
            if full.grad is not None:
                scale = max(1.0, float(full.grad.abs().max()))
                torch.testing.assert_close(
                    local.grad / scale,
                    full.grad / scale,
                    rtol=5e-4,
                    atol=5e-5,
                    msg=name,
                )
                errors["gradient_max_scaled"] = max(
                    errors["gradient_max_scaled"],
                    float((local.grad - full.grad).abs().max()) / scale,
                )
        ref_opt.step()
        local_opt.step()
        for name, local, full in candidate.parameter_pairs(reference):
            torch.testing.assert_close(
                local, full, rtol=2e-4, atol=2e-4, msg=name
            )
            errors["parameter_max_absolute"] = max(
                errors["parameter_max_absolute"],
                float((local - full).detach().abs().max()),
            )
    ref_opt.zero_grad(set_to_none=True)
    local_opt.zero_grad(set_to_none=True)
    reference.eval()
    devices = [full_batch[0].device.index] if full_batch[0].is_cuda else []
    with torch.no_grad(), torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed + 10)
        forecast = reference.unroll_prediction(
            full_batch[0], full_batch[2], full_batch[3], full_batch[4]
        )[0]
    predicted = candidate.forecast(local_batch, seed + 10)
    torch.testing.assert_close(
        predicted,
        forecast.index_select(-2, candidate.nodes["grid"]),
        rtol=3e-4,
        atol=3e-5,
    )
    return {
        "passed": True,
        "optimizer_updates": 2,
        "masked_adamw_elements": 0,
        "input_tensors_match": True,
        "prior_inference_matches": True,
        "losses": losses,
        **errors,
    }
