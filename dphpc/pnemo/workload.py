"""Original GraphEFM model and WeatherDataset setup for the benchmark."""

from types import SimpleNamespace
from neural_lam.config import load_config_and_datastores
from neural_lam.models.graph_efm import GraphEFM
from neural_lam.weather_dataset import WeatherDataset


def make_model_args(args):
    """Build the normal GraphEFM arguments for the controlled workload."""

    return SimpleNamespace(
        ar_steps_train=args.ar_steps,
        batch_size=args.batch_size,
        crps_alpha=0.95,
        crps_weight=0.0,
        encoder_processor_layers=1,
        ensemble_size=5,
        epochs=100,
        grad_checkpointing=True,
        graph=args.graph,
        hidden_dim=args.hidden_dim,
        hidden_dim_edge=args.hidden_dim,
        hidden_dim_grid=args.hidden_dim,
        hidden_dim_mesh_nodes=args.hidden_dim,
        hidden_layers=1,
        input_steps=2,
        kl_beta=1.0,
        latent_dim=None,
        learn_prior=1,
        loss="wmse",
        lr=args.lr,
        metrics_watch=[],
        min_lr=args.lr,
        muon_exclude_patterns=[],
        muon_momentum=0.95,
        muon_weight_decay=0.0,
        n_example_pred=0,
        num_future_atmosphere_steps=0,
        num_future_boundary_steps=0,
        num_future_forcing_steps=0,
        num_latents_plot=4,
        num_past_atmosphere_steps=2,
        num_past_boundary_steps=2,
        num_past_forcing_steps=2,
        output_std=False,
        optimizer="adamw",
        prior_dist="isotropic",
        prior_processor_layers=1,
        processor_layers=args.processor_layers,
        restore_opt=False,
        sample_obs_noise=0,
        seed=args.seed,
        soap_betas=[0.95, 0.95],
        soap_muon_betas=[0.95, 0.98],
        soap_muon_disable_sqrt=False,
        soap_precondition_frequency=10,
        soap_weight_decay=0.0,
        train_distr_crps=False,
        val_steps_to_log=[1, 2, 3],
        var_leads_metrics_watch={},
    )


def make_dataset(
    args,
    model_args,
    config,
    datastore,
    datastore_boundary,
    datastore_atmosphere,
    statistics_datastore=None,
    statistics_datastore_boundary=None,
    statistics_datastore_atmosphere=None,
):
    return WeatherDataset(
        datastore=datastore,
        datastore_boundary=datastore_boundary,
        datastore_atmosphere=datastore_atmosphere,
        split="train",
        ar_steps=args.ar_steps,
        input_steps=model_args.input_steps,
        standardize=True,
        num_past_forcing_steps=model_args.num_past_forcing_steps,
        num_future_forcing_steps=model_args.num_future_forcing_steps,
        current_forcing_step=True,
        current_boundary_step=True,
        current_atmosphere_step=True,
        num_past_boundary_steps=model_args.num_past_boundary_steps,
        num_future_boundary_steps=model_args.num_future_boundary_steps,
        num_past_atmosphere_steps=model_args.num_past_atmosphere_steps,
        num_future_atmosphere_steps=model_args.num_future_atmosphere_steps,
        use_atmosphere_g2m=False,
        density_channel=getattr(config.training, "density_channel", None),
        statistics_datastore=statistics_datastore,
        statistics_datastore_boundary=statistics_datastore_boundary,
        statistics_datastore_atmosphere=statistics_datastore_atmosphere,
    )


def load_workload(args, include_global_dataset=True):
    loaded = load_config_and_datastores(config_path=str(args.config_path))
    (
        config,
        datastore,
        datastore_boundary,
        datastore_atmosphere,
        statistics_datastore,
        statistics_datastore_boundary,
        statistics_datastore_atmosphere,
    ) = loaded
    model_args = make_model_args(args)
    dataset = None
    if include_global_dataset:
        dataset = make_dataset(
            args,
            model_args,
            config,
            datastore,
            datastore_boundary,
            datastore_atmosphere,
            statistics_datastore,
            statistics_datastore_boundary,
            statistics_datastore_atmosphere,
        )
    model = GraphEFM(
        model_args,
        config=config,
        datastore=datastore,
        datastore_boundary=datastore_boundary,
        datastore_atmosphere=datastore_atmosphere,
        statistics_datastore=statistics_datastore,
        statistics_datastore_boundary=statistics_datastore_boundary,
        statistics_datastore_atmosphere=statistics_datastore_atmosphere,
    )
    return dataset, model, config
