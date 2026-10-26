# Standard library
import json
import random
import time
from argparse import ArgumentParser

# Third-party
# for logging the model:
import pytorch_lightning as pl
import torch
from lightning_fabric.utilities import seed
from loguru import logger

# Local
from . import utils
from .benchmarking import BaselineBenchmark, InferenceBenchmarkModule
from .config import load_config_and_datastores
from .models import EDM, FM, SI, GraphCast, GraphCRPS, GraphEFM, GraphFM, GraphDET
from .weather_dataset import WeatherDataModule

MODELS = {
    "graphcast": GraphCast,
    "graph_fm": GraphFM,
    "graph_efm": GraphEFM,
    "graph_crps": GraphCRPS,
    "graph_det": GraphDET,
    "EDM": EDM,
    "FM": FM,
    "SI": SI,
}


@logger.catch(reraise=True)
def main(input_args=None):
    """Main function for training and evaluating models."""
    parser = ArgumentParser(
        description="Train or evaluate NeurWP models for LAM"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        help="Path to the configuration for neural-lam",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="graphcast",
        help="Model architecture to train/evaluate (default: graphcast)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed (default: 42)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers in data loader (default: 4)",
    )
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=1,
        help="Number of nodes to use in DDP (default: 1)",
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        type=str,
        default=["auto"],
        help="Devices to use for training. Can be the string 'auto' or a list "
        "of integer id's corresponding to the desired devices, e.g. "
        "'--devices 0 1'. Note that this cannot be used with SLURM, instead "
        "set 'ntasks-per-node' in the slurm setup (default: auto)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="upper epoch limit (default: 200)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="batch size (default: 4)"
    )
    parser.add_argument(
        "--load",
        type=str,
        help="Path to load model parameters from (default: None)",
    )
    parser.add_argument(
        "--restore_opt",
        action="store_true",
        help="If optimizer state should be restored with model "
        "(default: false)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default=32,
        help="Numerical precision to use for model (32/16/bf16) (default: 32)",
    )

    # Model architecture
    parser.add_argument(
        "--graph",
        type=str,
        default="multiscale",
        help="Graph to load and use in graph-based model "
        "(default: multiscale)",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=64,
        help="Dimensionality of all hidden representations (default: 64)",
    )
    parser.add_argument(
        "--hidden_dim_grid",
        type=int,
        help=(
            "Dimensionality of hidden representations related "
            "to grid nodes (default: None, use same as hidden_dim)."
        ),
    )
    parser.add_argument(
        "--hidden_dim_edge",
        type=int,
        help=(
            "Dimensionality of hidden representations related to edge nodes "
            "(edge encodings and in edge-level MLPs)"
            "(default: None, use same as hidden_dim_grid)"
        ),
    )
    parser.add_argument(
        "--hidden_dim_mesh_nodes",
        type=int,
        help=(
            "Dimensionality of hidden representations related to mesh nodes "
            "(mesh encodings and in mesh-level MLPs)"
            "(default: None, use same as hidden_dim)"
        ),
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=None,
        help="Dimensionality of latent R.V. at each node (if different than"
        " hidden_dim) (default: None (same as hidden_dim))",
    )
    parser.add_argument(
        "--hidden_layers",
        type=int,
        default=1,
        help="Number of hidden layers in all MLPs (default: 1)",
    )
    parser.add_argument(
        "--processor_layers",
        type=int,
        default=3,
        help="Number of GNN layers in processor GNN (for prob. model: in "
        "decoder) (default: 3)",
    )
    parser.add_argument(
        "--encoder_processor_layers",
        type=int,
        default=1,
        help="Number of on-mesh GNN layers in encoder GNN (default: 1)",
    )
    parser.add_argument(
        "--prior_processor_layers",
        type=int,
        default=1,
        help="Number of on-mesh GNN layers in prior GNN (default: 1)",
    )
    parser.add_argument(
        "--mesh_aggr",
        type=str,
        default="sum",
        help="Aggregation to use for m2m processor GNN layers (sum/mean) "
        "(default: sum)",
    )
    parser.add_argument(
        "--output_std",
        action="store_true",
        help="If models should additionally output std.-dev. per "
        "output dimensions "
        "(default: False (no))",
    )
    parser.add_argument(
        "--prior_dist",
        type=str,
        default="isotropic",
        help="Structure of Gaussian distribution in prior network output "
        "(isotropic/diagonal) (default: isotropic)",
    )
    parser.add_argument(
        "--learn_prior",
        type=int,
        default=1,
        help="If the prior should be learned as a mapping from previous state "
        "and forcing, otherwise static with mean 0 (default: 1 (yes))",
    )

    # EDM options
    parser.add_argument(
        "--sigma_min",
        type=float,
        default=0.002,
        help="Minimum value for sigma in EDM (default: 0.002)",
    )
    parser.add_argument(
        "--sigma_max",
        type=float,
        default=88,
        help="Maximum value for sigma in EDM (default: 88)",
    )
    parser.add_argument(
        "--sigma_data",
        type=float,
        default=1,
        help="Expected standard deviation of the data in EDM (default: 1.0)",
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=7,
        help="Rho parameter in EDM (default: 7)",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default="heun",
        help="Sampler to use in EDM (heun/edm/ddpm) (default: heun)",
    )
    parser.add_argument(
        "--sampler_steps",
        type=int,
        default=20,
        help="Number of steps to use in sampler (default: 20)",
    )
    parser.add_argument(
        "--backbone_model",
        type=str,
        default="graph_diff",
        help="Backbone model to use in EDM (graph_diff) (default: graph_diff)",
    )
    parser.add_argument(
        "--pred_residual",
        action="store_true",
        help="If the EDM model should predict residuals instead of the "
        "next state. NOTE: has no effect for graph_det or graph_crps, which "
        "always use the residual parametrization to match graph_fm",
    )
    parser.add_argument(
        "--channel_mult_noise",
        type=int,
        default=2,
        help="Channel multiplier for noise MLP (default: 2)",
    )

    # Graph-CRPS options
    parser.add_argument(
        "--noise_embedding",
        type=str,
        default="linear",
        help="Type of encoder to use (default: 'linear')",
    )
    parser.add_argument(
        "--noise_dim",
        type=int,
        default=32,
        help="Dimension of the noise vector z, 32 in FGN (default: 32)",
    )
    parser.add_argument(
        "--crps_alpha",
        type=float,
        default=0.95,
        help="Alpha parameter for the Almost Fair CRPS (default: 0.95)",
    )
    parser.add_argument(
        "--train_distr_crps",
        action="store_true",
        help="Run training using CRPS-loss distributed across multiple GPUs"
        "(default: False)",
    )
    parser.add_argument(
        "--channel_mult_emb",
        type=int,
        default=2,
        help="Channel multiplier for noise embedding MLP",
    )
    parser.add_argument(
        "--deterministic_noise_value",
        type=float,
        default=0.,
        help="Value for the noise parameter in the deterministisc version."
    )

    # Training options
    parser.add_argument(
        "--ar_steps_train",
        type=int,
        default=1,
        help="Number of steps to unroll prediction for in loss function "
        "(default: 1)",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="wmse",
        help="Loss function to use, see metric.py (default: wmse)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-3,
        help="Minimum learning rate for cosine annealing (default: 1e-4)",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adamw", "muon", "muon_flat", "soap", "soap_muon"],
        help="Optimizer to use. 'muon' applies the Muon optimizer to 2D "
        "hidden weights (AdamW for embeddings/output/1D params); 'muon_flat' "
        "additionally flattens 4D conv weights so Muon optimizes them too; "
        "'soap' runs Adam in the eigenbasis of Shampoo's preconditioner for "
        "all parameters; 'soap_muon' adds Muon-style orthogonalization on "
        "top of SOAP for the same 2D hidden weights 'muon' would take "
        "(default: adamw)",
    )
    parser.add_argument(
        "--muon_momentum",
        type=float,
        default=0.95,
        help="Momentum for the Muon optimizer (default: 0.95)",
    )
    parser.add_argument(
        "--muon_weight_decay",
        type=float,
        default=0.0,
        help="Weight decay for the muon optimizer"
        "(default: 0.0)",
    )
    parser.add_argument(
        "--muon_exclude_patterns",
        nargs="+",
        default=[
            "interior_embedder",
            "boundary_embedder",
            "g2m_embedder",
            "m2g_embedder",
            "output_map",
            "mesh_embedders",
            "mesh_same_embedders",
            "mesh_up_embedders",
            "mesh_down_embedders",
        ],
        help="Parameter-name substrings excluded from the Muon (and "
        "SOAP-Muon) parameter group, i.e. input embeddings and the final "
        "output layer",
    )
    parser.add_argument(
        "--soap_betas",
        type=float,
        nargs=2,
        default=[0.95, 0.95],
        help="Adam betas for the soap optimizer (default: 0.95 0.95)",
    )
    parser.add_argument(
        "--soap_muon_betas",
        type=float,
        nargs=2,
        default=[0.95, 0.98],
        help="Adam betas for the soap_muon optimizer (default: 0.95 0.98)",
    )
    parser.add_argument(
        "--soap_weight_decay",
        type=float,
        default=0.0,
        help="Weight decay for the soap and soap_muon optimizers. Matches "
        "the muon default here; upstream SOAP uses 0.01 (default: 0.0)",
    )
    parser.add_argument(
        "--soap_precondition_frequency",
        type=int,
        default=10,
        help="How often (in steps) soap/soap_muon refresh the preconditioner "
        "eigenbasis. soap_muon tolerates larger values (~40) since the "
        "orthogonalization runs every step (default: 10)",
    )
    parser.add_argument(
        "--soap_muon_disable_sqrt",
        action="store_true",
        help="Disable soap_muon's square-root correction after "
        "orthogonalization (singular-value power 0 instead of 0.5). Changes "
        "the update rule, not the cost: unlike the reference implementation "
        "this one always orthogonalizes with Newton-Schulz, never a full SVD. "
        "Reported to be unstable on some datasets (default: False)",
    )
    parser.add_argument(
        "--val_interval",
        type=int,
        default=1,
        help="Number of epochs training between each validation run "
        "(default: 1)",
    )
    parser.add_argument(
        "--kl_beta",
        type=float,
        default=1.0,
        help="Beta weighting in front of kl-term in ELBO (default: 1)",
    )
    parser.add_argument(
        "--crps_weight",
        type=float,
        default=0,
        help="Weighting for CRPS term of loss, not computed if = 0. CRPS is "
        "computed based on trajectories sampled using prior distribution. "
        "(default: 0)",
    )
    parser.add_argument(
        "--sample_obs_noise",
        type=int,
        default=0,
        help="If observation noise should be sampled during rollouts (both "
        "training and eval), or just mean prediction used "
        "(default: 0 (no))",
    )
    parser.add_argument(
        "--num_sanity_val_steps",
        type=int,
        default=4,
        help="Number of sanity check steps to run before training (default: 4)",
    )
    parser.add_argument(
        "--grad_checkpointing",
        action="store_true",
        help="Force gradient checkpointing at each unroll step, also for "
        "single-step phases. It is already enabled automatically whenever "
        "more than one step is unrolled (default: False)",
    )

    # Evaluation options
    parser.add_argument(
        "--eval",
        type=str,
        help="Eval model on given data split (val/test) "
        "(default: None (train model))",
    )
    parser.add_argument(
        "--ar_steps_eval",
        type=int,
        default=10,
        help="Number of steps to unroll prediction for during evaluation "
        "(default: 10)",
    )
    parser.add_argument(
        "--input_steps",
        type=int,
        default=2,
        choices=[1, 2],
        help="Number of state steps as input (default: 1)",
    )
    parser.add_argument(
        "--n_example_pred",
        type=int,
        default=1,
        help="Number of example predictions to plot during val/test "
        "(default: 1)",
    )
    parser.add_argument(
        "--num_latents_plot",
        type=int,
        default=4,
        help="Number of samples of latent variable to plot (default: 4)",
    )

    # Logger Settings
    parser.add_argument(
        "--logger",
        type=str,
        default="wandb",
        choices=["wandb", "mlflow"],
        help="Logger to use for training (wandb/mlflow) (default: wandb)",
    )
    parser.add_argument(
        "--logger-project",
        type=str,
        default="baltic-sea-forecasting",
        help="Logger project name, for eg. Wandb "
        "(default: baltic-sea-forecasting)",
    )
    parser.add_argument(
        "--val_steps_to_log",
        nargs="+",
        type=int,
        default=[1, 2, 3],
        help="Steps to log val loss for (default: 1 2 3 5 10 15 19)",
    )
    parser.add_argument(
        "--metrics_watch",
        nargs="+",
        default=[],
        help="List of metrics to watch, including any prefix (e.g. val_rmse)",
    )
    parser.add_argument(
        "--var_leads_metrics_watch",
        type=str,
        default="{}",
        help="""JSON string with variable-IDs and lead times to log watched
             metrics (e.g. '{"1": [1, 2], "3": [3, 4]}')""",
    )
    parser.add_argument(
        "--var_leads_val_plot",
        type=str,
        default="{}",
        help="""JSON string with variable-IDs and lead times to plot during
            validation step (e.g. '{"1": [1, 2], "3": [3, 4]}')""",
    )
    parser.add_argument(
        "--current_forcing_step",
        type=int,
        choices=[0, 1],
        default=1,
        help="Include current time t in forcing window (default: 1).",
    )
    parser.add_argument(
        "--num_past_forcing_steps",
        type=int,
        default=2,
        help="Number of past time steps to use as input for forcing data "
        "(default: 2).",
    )
    parser.add_argument(
        "--num_future_forcing_steps",
        type=int,
        default=0,
        help="Number of future time steps to use as input for forcing data",
    )
    parser.add_argument(
        "--current_boundary_step",
        type=int,
        choices=[0, 1],
        default=1,
        help="Include current time t in boundary window (default: 1).",
    )
    parser.add_argument(
        "--num_past_boundary_steps",
        type=int,
        default=2,
        help="Number of past time steps to use as boundary input (default: 2).",
    )
    parser.add_argument(
        "--num_future_boundary_steps",
        type=int,
        default=0,
        help="Number of future time steps to use as atmosphere input "
        "(default: 0)",
    )
    parser.add_argument(
        "--current_atmosphere_step",
        type=int,
        choices=[0, 1],
        default=1,
        help="Include current time t in atmosphere window (default: 1).",
    )
    parser.add_argument(
        "--num_past_atmosphere_steps",
        type=int,
        default=2,
        help="Number of past time steps to use as atmosphere input "
        "(default: 2).",
    )
    parser.add_argument(
        "--num_future_atmosphere_steps",
        type=int,
        default=0,
        help="Number of future time steps to use as boundary input "
        "(default: 0)",
    )
    parser.add_argument(
        "--use_atmosphere_g2m",
        action="store_true",
        help="Use atmosphere as separate grid nodes in g2m encoding "
        "(experimental, default: False).",
    )
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=5,
        help="Number of ensemble members during evaluation (default: 5)",
    )
    parser.add_argument(
        "--save_ensemble_zarr",
        action="store_true",
        help="Save full ensemble forecasts to zarr during test "
        "(default: False)",
    )
    parser.add_argument(
        "--save_forecasts",
        action="store_true",
        help="Save the full test set (all init_times) ensemble forecasts "
        "and targets to a single zarr store, appended along init_time "
        "one batch at a time (default: False)",
    )
    parser.add_argument(
        "--forecasts_zarr_path",
        type=str,
        default=None,
        help="Path to the zarr store used by --save_forecasts. "
        "If not provided, defaults to "
        "{logger.save_dir}/ensemble_forecasts.zarr",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default=None,
        choices=[
            "pretrain",
            "finetune",
            "probabilistic",
            "deterministic",
            "finetune_deterministic",
        ],
        help="Multi-phase training scheduler.",
    )
    parser.add_argument(
        "--benchmark_output",
        type=str,
        default=None,
        help="Write a controlled training benchmark to this JSON path. This "
        "disables validation, checkpoint saving, and external logging.",
    )
    parser.add_argument(
        "--benchmark_warmup_steps",
        type=int,
        default=5,
        help=(
            "Training steps excluded before benchmark measurement (default: 5)"
        ),
    )
    parser.add_argument(
        "--benchmark_steps",
        type=int,
        default=30,
        help="Number of measured training steps (default: 30)",
    )
    parser.add_argument(
        "--benchmark_label",
        type=str,
        default="baseline",
        help="Human-readable label stored with a benchmark result.",
    )
    parser.add_argument(
        "--benchmark_strategy",
        type=str,
        choices=["auto", "ddp"],
        default="auto",
        help="Lightning strategy for benchmarking (default: auto).",
    )
    parser.add_argument(
        "--benchmark_scaling_mode",
        choices=["single", "strong", "weak", "capacity"],
        default="single",
        help="Experiment family stored in the result metadata.",
    )
    parser.add_argument(
        "--benchmark_global_batch_size",
        type=int,
        default=None,
        help="Global batch size used for throughput. By default this is the "
        "local batch size multiplied by world size.",
    )
    parser.add_argument(
        "--benchmark_profiler_dir",
        type=str,
        default=None,
        help="Optional directory for a PyTorch operator trace. Profiling is "
        "separate from reportable timing runs because it adds overhead.",
    )
    parser.add_argument(
        "--benchmark_workload",
        choices=["training", "inference"],
        default="training",
        help="Benchmark a full training step or forecast-only inference.",
    )
    parser.add_argument(
        "--benchmark_inference_ensemble_size",
        type=int,
        default=1,
        help="Number of parallel forecast members in an inference benchmark.",
    )

    args = parser.parse_args(input_args)
    args.var_leads_metrics_watch = {
        int(k): v for k, v in json.loads(args.var_leads_metrics_watch).items()
    }
    args.var_leads_val_plot = {
        int(k): v for k, v in json.loads(args.var_leads_val_plot).items()
    }

    # Asserts for arguments
    assert (
        args.config_path is not None
    ), "Specify your config with --config_path"
    assert args.model in MODELS, f"Unknown model: {args.model}"
    assert args.eval in (
        None,
        "val",
        "test",
    ), f"Unknown eval setting: {args.eval}"
    if args.benchmark_output is not None:
        assert args.eval is None, "Do not combine --eval and benchmarking"
        assert args.scheduler is None, "Benchmarking does not support schedules"

    # Get an (actual) random run id as a unique identifier
    random_run_id = random.randint(0, 9999)

    # Set seed
    seed.seed_everything(args.seed)

    # Load neural-lam configuration and datastore to use
    (
        config,
        datastore,
        datastore_boundary,
        datastore_atmosphere,
        statistics_datastore,
        statistics_datastore_boundary,
        statistics_datastore_atmosphere,
    ) = load_config_and_datastores(config_path=args.config_path)

    # Create datamodule
    data_module = WeatherDataModule(
        datastore=datastore,
        datastore_boundary=datastore_boundary,
        datastore_atmosphere=datastore_atmosphere,
        statistics_datastore=statistics_datastore,
        statistics_datastore_boundary=statistics_datastore_boundary,
        statistics_datastore_atmosphere=statistics_datastore_atmosphere,
        ar_steps_train=args.ar_steps_train,
        ar_steps_eval=args.ar_steps_eval,
        input_steps=args.input_steps,
        standardize=True,
        num_past_forcing_steps=args.num_past_forcing_steps,
        num_future_forcing_steps=args.num_future_forcing_steps,
        current_forcing_step=bool(args.current_forcing_step),
        current_boundary_step=bool(args.current_boundary_step),
        current_atmosphere_step=bool(args.current_atmosphere_step),
        num_past_boundary_steps=args.num_past_boundary_steps,
        num_future_boundary_steps=args.num_future_boundary_steps,
        num_past_atmosphere_steps=args.num_past_atmosphere_steps,
        num_future_atmosphere_steps=args.num_future_atmosphere_steps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_atmosphere_g2m=args.use_atmosphere_g2m,
        density_channel=getattr(config.training, "density_channel", None),
        train_distr_crps=args.train_distr_crps,
    )

    # Instantiate model + trainer
    if torch.cuda.is_available():
        device_name = "cuda"
        torch.set_float32_matmul_precision(
            "high"
        )  # Allows using Tensor Cores on A100s
    elif torch.backends.mps.is_available():
        device_name = "mps"
    else:
        device_name = "cpu"

    # Set devices to use
    if args.devices == ["auto"]:
        devices = "auto"
    else:
        try:
            devices = [int(i) for i in args.devices]
        except ValueError:
            raise ValueError("devices should be 'auto' or a list of integers")

    # Load model parameters Use new args for model
    ModelClass = MODELS[args.model]
    model = ModelClass(
        args,
        config=config,
        datastore=datastore,
        datastore_boundary=datastore_boundary,
        datastore_atmosphere=datastore_atmosphere,
        statistics_datastore=statistics_datastore,
        statistics_datastore_boundary=statistics_datastore_boundary,
        statistics_datastore_atmosphere=statistics_datastore_atmosphere,
    )

    if args.eval:
        prefix = f"eval-{args.eval}-"
    else:
        prefix = "train-"
    run_name = (
        f"{prefix}{args.model}-{args.processor_layers}x{args.hidden_dim}-"
        f"{time.strftime('%m_%d_%H')}-{random_run_id:04d}"
    )

    benchmark_enabled = args.benchmark_output is not None
    training_logger = (
        False
        if benchmark_enabled
        else utils.setup_training_logger(
            datastore=datastore, args=args, run_name=run_name
        )
    )

    # Callbacks for saving model checkpoint
    callbacks = []
    if benchmark_enabled:
        callbacks.append(
            BaselineBenchmark(
                output_path=args.benchmark_output,
                warmup_steps=args.benchmark_warmup_steps,
                measurement_steps=args.benchmark_steps,
                run_label=args.benchmark_label,
                args=args,
            )
        )
    else:
        callbacks.append(
            pl.callbacks.ModelCheckpoint(
                dirpath=f"saved_models/{run_name}",
                filename="min_val_loss",
                monitor="val_mean_loss",
                mode="min",
            )
        )
    # Separate callback that keeps "last.ckpt" up to date after *every* epoch.
    # A monitored ModelCheckpoint only saves when validation runs, so with
    # --val_interval > 1 (check_val_every_n_epoch != 1, which also makes
    # Lightning save at validation end rather than at train epoch end)
    # last.ckpt would lag up to val_interval epochs behind, and restarting a
    # stage would silently redo that training.
    last_ckpt_callback = None
    if not benchmark_enabled:
        last_ckpt_callback = pl.callbacks.ModelCheckpoint(
            dirpath=f"saved_models/{run_name}",
            save_top_k=0,
            save_last=True,
            save_on_train_epoch_end=True,
        )
        callbacks.append(last_ckpt_callback)
    # Log the LR to the training logger, so that a mis-restored schedule after
    # a restart is visible instead of only showing up as a worse loss.
    if not benchmark_enabled:
        callbacks.append(
            pl.callbacks.LearningRateMonitor(logging_interval="epoch")
        )

    # To enable no validation during training, set val_interval to None
    if args.val_interval == 0:
        args.val_interval = None

    def _make_trainer(max_epochs, strategy, extra_callbacks=None):
        """Build a pl.Trainer with the given max_epochs and strategy."""
        cbs = list(callbacks)
        if extra_callbacks:
            cbs.extend(extra_callbacks)
        profiler = None
        if args.benchmark_profiler_dir:
            active_steps = min(5, args.benchmark_steps)
            profiler = pl.profilers.PyTorchProfiler(
                dirpath=args.benchmark_profiler_dir,
                filename=args.benchmark_label,
                schedule=torch.profiler.schedule(
                    wait=max(args.benchmark_warmup_steps - 1, 0),
                    warmup=1,
                    active=active_steps,
                    repeat=1,
                ),
                export_to_chrome=True,
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                with_flops=True,
            )
        return pl.Trainer(
            max_epochs=max_epochs,
            max_steps=(
                args.benchmark_warmup_steps + args.benchmark_steps
                if benchmark_enabled
                else -1
            ),
            # PyG scatter reductions have no deterministic MPS kernel. Allow
            # them for local Apple-GPU performance experiments while keeping
            # strict determinism for the official CUDA baseline.
            deterministic=False if device_name == "mps" else True,
            strategy=strategy,
            accelerator=device_name,
            num_nodes=args.num_nodes,
            devices=devices,
            logger=training_logger,
            log_every_n_steps=1,
            callbacks=cbs,
            check_val_every_n_epoch=args.val_interval,
            precision=args.precision,
            num_sanity_val_steps=(
                0 if benchmark_enabled else args.num_sanity_val_steps
            ),
            use_distributed_sampler=not args.train_distr_crps,
            limit_val_batches=0 if benchmark_enabled else 1.0,
            limit_test_batches=(
                args.benchmark_warmup_steps + args.benchmark_steps
                if benchmark_enabled
                and args.benchmark_workload == "inference"
                else 1.0
            ),
            enable_checkpointing=not benchmark_enabled,
            enable_progress_bar=not benchmark_enabled,
            profiler=profiler,
        )

    def _run_phase(
        model,
        data_module,
        max_epochs,
        strategy,
        ckpt_path=None,
        extra_callbacks=None,
    ):
        """Run one training phase, return path to last checkpoint."""
        t = _make_trainer(max_epochs, strategy,
                          extra_callbacks=extra_callbacks)
        if t.global_rank == 0 and not benchmark_enabled:
            utils.init_training_logger_metrics(
                training_logger, val_steps=args.val_steps_to_log
            )
        # Checkpoints store args (argparse.Namespace) and config objects,
        # which torch>=2.6 refuses to unpickle by default
        t.fit(
            model=model,
            datamodule=data_module,
            ckpt_path=ckpt_path,
            weights_only=False,
        )
        return last_ckpt_callback.last_model_path

    def _completed_epochs(ckpt_path):
        """Number of epochs already completed in the given checkpoint."""
        if not ckpt_path:
            return 0
        try:
            ckpt = torch.load(
                ckpt_path, map_location="cpu", mmap=True, weights_only=False
            )
        except (RuntimeError, TypeError, ValueError):
            # mmap requires a zipfile-serialized checkpoint; fall back to a
            # full read (slow, but only done once at startup).
            ckpt = torch.load(
                ckpt_path, map_location="cpu", weights_only=False
            )
        epoch = ckpt.get("epoch")
        if epoch is None:
            print(
                f"[warning] {ckpt_path} has no 'epoch' entry; assuming no "
                "epochs completed, so the schedule starts at phase 1"
            )
            return 0
        return int(epoch) + 1

    def _run_phases(phases):
        """Run a multi-phase schedule, resuming into the middle of it.

        Parameters
        ----------
        phases : list of (label, max_epochs, setup, strategy)
            ``max_epochs`` is cumulative over the whole schedule (as Lightning
            counts epochs). ``setup`` configures model/data for the phase and
            is called for *every* phase, including phases that the checkpoint
            passed with --load has already completed, so that resuming
            mid-schedule reproduces the configuration of the phase that is
            resumed. Phases already completed are not re-run.
        """
        ckpt_path = args.load
        done = _completed_epochs(ckpt_path)
        if done:
            print(
                f"[{args.scheduler}] Resuming from {ckpt_path} with {done} "
                "epoch(s) already completed"
            )
        for label, max_epochs, setup, strategy in phases:
            setup()
            if done >= max_epochs:
                print(f"[{args.scheduler}] {label} -- already done, skipping")
                continue
            print(f"[{args.scheduler}] {label}")
            new_ckpt = _run_phase(
                model, data_module, max_epochs, strategy, ckpt_path=ckpt_path
            )
            # Empty if the phase saved no checkpoint; keep resuming from the
            # checkpoint we came in with rather than restarting from scratch.
            ckpt_path = new_ckpt or ckpt_path
            done = max(done, max_epochs)

    def _set_ar_steps(n_steps):
        """Set the training unroll length for a phase.

        Gradient checkpointing at each unroll step is enabled whenever more
        than one step is unrolled, since the activations kept for backward
        otherwise scale with the unroll length (a 1-step phase that fits in
        memory can OOM at 2 steps). ``--grad_checkpointing`` forces it on for
        every phase, including 1-step ones.
        """
        data_module.ar_steps_train = n_steps
        model.grad_checkpointing = args.grad_checkpointing or n_steps > 1

    def _add_cosine_lr(model, max_epochs):
        """Attach a CosineAnnealingLR scheduler to the model."""
        _orig_configure = model.configure_optimizers

        def configure_optimizers_with_cosine(self_ref=model):
            opt = _orig_configure()
            if isinstance(opt, dict):
                optimizer = opt["optimizer"]
            else:
                optimizer = opt
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max_epochs, eta_min=1e-5
            )
            return {"optimizer": optimizer, "lr_scheduler": sched}

        model.configure_optimizers = configure_optimizers_with_cosine

    def _add_finetune_sequential_lr(
        model,
        warmup_epochs,
        finetune_epochs,
        initial_lr,
        lr,
        eta_min,
    ):
        """Linear LR warmup then cosine decay over finetune_epochs."""
        _orig_configure = model.configure_optimizers

        def configure_optimizers_finetune(self_ref=model):
            opt = _orig_configure()
            if isinstance(opt, dict):
                optimizer = opt["optimizer"]
            else:
                optimizer = opt
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            linear_warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=initial_lr / lr,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            cosine_annealing = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=finetune_epochs - warmup_epochs,
                eta_min=eta_min,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[linear_warmup, cosine_annealing],
                milestones=[warmup_epochs],
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}

        model.configure_optimizers = configure_optimizers_finetune

    if benchmark_enabled and args.benchmark_workload == "inference":
        strategy = args.benchmark_strategy
        trainer = _make_trainer(args.epochs, strategy)
        inference_model = InferenceBenchmarkModule(
            model,
            ensemble_size=args.benchmark_inference_ensemble_size,
        )
        # Performance runs use the larger training split to guarantee the
        # same requested sample count even for long autoregressive horizons.
        # No optimization or gradients are performed by trainer.test().
        data_module.setup("fit")
        trainer.test(
            model=inference_model,
            dataloaders=data_module.train_dataloader(),
        )

    elif args.eval:
        strategy = (
            "ddp" if args.kl_beta > 0 else "ddp_find_unused_parameters_true"
        )
        trainer = _make_trainer(args.epochs, strategy)
        if trainer.global_rank == 0:
            utils.init_training_logger_metrics(
                training_logger, val_steps=args.val_steps_to_log
            )
        trainer.test(
            model=model,
            datamodule=data_module,
            ckpt_path=args.load,
            weights_only=False,
        )

    elif args.scheduler == "pretrain":
        total_epochs = 100 + 200 + 25
        _add_cosine_lr(model, total_epochs)

        def _pretrain_p1():
            model.kl_beta = 0.0
            _set_ar_steps(1)

        def _pretrain_p2():
            model.kl_beta = 0.1

        def _pretrain_p3():
            _set_ar_steps(2)

        _run_phases(
            [
                (
                    "Phase 1/3: 100 epochs, ar=1, kl_beta=0",
                    100,
                    _pretrain_p1,
                    "ddp_find_unused_parameters_true",
                ),
                (
                    "Phase 2/3: 200 epochs, ar=1, kl_beta=0.1",
                    300,
                    _pretrain_p2,
                    "ddp",
                ),
                (
                    "Phase 3/3: 25 epochs, ar=2, kl_beta=0.1",
                    325,
                    _pretrain_p3,
                    "ddp",
                ),
            ]
        )

    elif args.scheduler == "finetune":
        prior_done = 325
        warmup_epochs = 5
        finetune_epochs = 165
        _add_finetune_sequential_lr(
            model,
            warmup_epochs=warmup_epochs,
            finetune_epochs=finetune_epochs,
            initial_lr=1e-5,
            lr=1e-4,
            eta_min=1e-5,
        )

        def _finetune_p1():
            model.kl_beta = 0.0
            model.crps_weight = 0.0
            _set_ar_steps(1)

        def _finetune_p2():
            model.kl_beta = 0.1
            model.crps_weight = 0.0
            _set_ar_steps(1)

        def _finetune_p3():
            model.kl_beta = 0.1
            model.crps_weight = 0.0
            _set_ar_steps(2)

        def _finetune_p4():
            model.kl_beta = 0.1
            model.crps_weight = 1e6
            _set_ar_steps(2)

        _run_phases(
            [
                (
                    "Phase 1/4: 5 ep warmup, ar=1, kl_beta=0, "
                    "strategy=ddp_find_unused_parameters_true",
                    prior_done + warmup_epochs,
                    _finetune_p1,
                    "ddp_find_unused_parameters_true",
                ),
                (
                    "Phase 2/4: 150 ep, ar=1, kl_beta=0.1, strategy=ddp",
                    prior_done + 155,
                    _finetune_p2,
                    "ddp",
                ),
                (
                    "Phase 3/4: 5 ep, ar=2, kl_beta=0.1, strategy=ddp",
                    prior_done + 160,
                    _finetune_p3,
                    "ddp",
                ),
                (
                    "Phase 4/4: 5 ep, ar=2, kl_beta=0.1, crps_weight=1e6, "
                    "strategy=ddp",
                    prior_done + finetune_epochs,
                    _finetune_p4,
                    "ddp",
                ),
            ]
        )

    elif args.scheduler == "probabilistic":
        total_epochs = 100 + 200 + 25 + 25
        _add_cosine_lr(model, total_epochs)

        def _prob_p1():
            model.crps_weight = 0.0
            model.kl_beta = 0.0
            _set_ar_steps(1)

        def _prob_p2():
            model.kl_beta = 0.1

        def _prob_p3():
            _set_ar_steps(2)

        def _prob_p4():
            model.crps_weight = 1e4

        _run_phases(
            [
                (
                    "Phase 1/4: 100 epochs, kl_beta=0",
                    100,
                    _prob_p1,
                    "ddp_find_unused_parameters_true",
                ),
                (
                    "Phase 2/4: 200 epochs, kl_beta=0.1",
                    300,
                    _prob_p2,
                    "ddp",
                ),
                (
                    "Phase 3/4: 25 epochs, ar=2, kl_beta=0.1",
                    325,
                    _prob_p3,
                    "ddp",
                ),
                (
                    "Phase 4/4: 25 epochs, ar=2, kl_beta=0.1, "
                    "crps_weight=1e4",
                    350,
                    _prob_p4,
                    "ddp",
                ),
            ]
        )

    elif args.scheduler == "deterministic":
        total_epochs = 150 + 25
        _add_cosine_lr(model, total_epochs)

        def _det_p1():
            _set_ar_steps(1)

        def _det_p2():
            _set_ar_steps(2)

        _run_phases(
            [
                ("Phase 1/2: 150 epochs, ar=1", 150, _det_p1, "ddp"),
                ("Phase 2/2: 25 epochs, ar=2", 175, _det_p2, "ddp"),
            ]
        )

    elif args.scheduler == "finetune_deterministic":
        prior_done = 175
        _add_finetune_sequential_lr(
            model,
            warmup_epochs=5,
            finetune_epochs=60,
            initial_lr=1e-5,
            lr=1e-4,
            eta_min=1e-5,
        )

        def _ft_det_p1():
            _set_ar_steps(1)

        def _ft_det_p2():
            _set_ar_steps(2)

        _run_phases(
            [
                (
                    "Phase 1/2: 50 ep, ar=1",
                    prior_done + 50,
                    _ft_det_p1,
                    "ddp",
                ),
                (
                    "Phase 2/2: 10 ep, ar=2",
                    prior_done + 60,
                    _ft_det_p2,
                    "ddp",
                ),
            ]
        )

    else:
        # Default: single-phase training
        if benchmark_enabled:
            strategy = args.benchmark_strategy
        else:
            strategy = (
                "ddp"
                if args.kl_beta > 0
                else "ddp_find_unused_parameters_true"
            )
        _add_cosine_lr(model, args.epochs)
        trainer = _make_trainer(args.epochs, strategy)
        if trainer.global_rank == 0 and not benchmark_enabled:
            utils.init_training_logger_metrics(
                training_logger, val_steps=args.val_steps_to_log
            )
        trainer.fit(
            model=model,
            datamodule=data_module,
            ckpt_path=args.load,
            weights_only=False,
        )


if __name__ == "__main__":
    main()
