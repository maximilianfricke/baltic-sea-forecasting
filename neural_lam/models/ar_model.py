# Standard library
import os
import warnings
from typing import List, Union

# Third-party
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import xarray as xr

# Local
from .. import metrics, utils, vis
from ..config import NeuralLAMConfig
from ..datastore import BaseDatastore
from ..loss_weighting import get_state_feature_weighting
from ..weather_dataset import WeatherDataset


class ARModel(pl.LightningModule):
    """
    Generic auto-regressive weather model.
    Abstract class that can be extended.
    """

    # pylint: disable=arguments-differ
    # Disable to override args/kwargs from superclass

    def __init__(
        self,
        args,
        config: NeuralLAMConfig,
        datastore: BaseDatastore,
        datastore_boundary: Union[BaseDatastore, None],
        datastore_atmosphere: Union[BaseDatastore, None],
        statistics_datastore: Union[BaseDatastore, None] = None,
        statistics_datastore_boundary: Union[BaseDatastore, None] = None,
        statistics_datastore_atmosphere: Union[BaseDatastore, None] = None,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=[
                "datastore",
                "datastore_boundary",
                "datastore_atmosphere",
                "statistics_datastore",
                "statistics_datastore_boundary",
                "statistics_datastore_atmosphere",
            ]
        )
        self.args = args
        self.input_steps = args.input_steps
        self._datastore = datastore
        self._datastore_boundary = datastore_boundary
        self._datastore_atmosphere = datastore_atmosphere
        self._statistics_datastore = (
            statistics_datastore
            if statistics_datastore is not None
            else datastore
        )
        self._statistics_datastore_boundary = (
            statistics_datastore_boundary
            if statistics_datastore_boundary is not None
            else datastore_boundary
        )
        self._statistics_datastore_atmosphere = (
            statistics_datastore_atmosphere
            if statistics_datastore_atmosphere is not None
            else datastore_atmosphere
        )
        self.num_state_vars = datastore.get_num_data_vars(category="state")
        num_forcing_vars = datastore.get_num_data_vars(category="forcing")
        # Load masks
        self.surface_mask = datastore.get_mask(
            surface=True, stacked=True, invert=False
        )
        self.interior_mask = datastore.get_mask(
            surface=False, stacked=True, invert=False
        )[
            self.surface_mask
        ]  # (num_grid_nodes, d_features), 1 for non-land

        # Optionally build latitude-weighted grid loss mask (approx. equal-area)
        # using cos(latitude), normalized to unit mean over interior grid.
        xy = datastore.get_xy("state", stacked=True)  # (N_full, 2) [lon, lat]
        xy_surface = xy[self.surface_mask]  # (num_grid_nodes, 2)
        lat_rad = np.deg2rad(xy_surface[:, 1].astype(np.float64))
        cos_lat = np.cos(lat_rad).astype(np.float32)
        interior_mask_arr = self.interior_mask.astype(np.float32)
        if getattr(config.training, "lat_weighted_loss", False):
            # Broadcast cos(lat) over state features and zero out land points
            loss_mask_np = interior_mask_arr * cos_lat[:, np.newaxis]
            total_weight = loss_mask_np.sum()
            num_active = interior_mask_arr.sum()
            # Normalize so mean weight over active grid points is 1
            loss_mask_np *= num_active / total_weight
        else:
            # Uniform weighting over interior grid points
            loss_mask_np = interior_mask_arr

        # Load static features and state stats from the statistics datastore
        # (falls back to the main datastore when no separate one is provided)
        da_static_features = self._statistics_datastore.get_dataarray(
            category="static", split=None, standardize=True
        )[self.surface_mask]
        # TODO: load static features from main ds and stats from stats ds
        # also for boundary, atmosphere (baltic sea had no grid mismatch).
        # Here we replace 2 nans * 7 features with zero in global finetune.
        _grid_static = torch.tensor(
            da_static_features.values, dtype=torch.float32
        )
        if not torch.isfinite(_grid_static).all():
            n_bad = int((~torch.isfinite(_grid_static)).sum().item())
            warnings.warn(f"Replacing {n_bad} static grid values with 0")
            _grid_static = torch.nan_to_num(_grid_static, nan=0.0)
        da_state_stats = (
            self._statistics_datastore.get_standardization_dataarray(
                category="state"
            )
        )
        num_past_forcing_steps = args.num_past_forcing_steps
        num_future_forcing_steps = args.num_future_forcing_steps
        include_current_forcing_step = bool(
            int(getattr(args, "current_forcing_step", 1))
        )
        num_forcing_steps = (
            num_past_forcing_steps
            + num_future_forcing_steps
            + (1 if include_current_forcing_step else 0)
        )

        # Load static features for grid/data,
        self.register_buffer(
            "grid_static_features",
            _grid_static,
            persistent=False,
        )

        state_stats = {
            "state_mean": torch.tensor(
                da_state_stats.state_mean.values, dtype=torch.float32
            ),
            "state_std": torch.tensor(
                da_state_stats.state_std.values, dtype=torch.float32
            ),
            # Note that the one-step-diff stats (diff_mean and diff_std) are
            # for differences computed on standardized data
            "diff_mean": torch.tensor(
                da_state_stats.state_diff_mean_standardized.values,
                dtype=torch.float32,
            ),
            "diff_std": torch.tensor(
                da_state_stats.state_diff_std_standardized.values,
                dtype=torch.float32,
            ),
        }

        # Density channel: extend stats and store indices
        density_cfg = getattr(config.training, "density_channel", None)
        self.use_density = density_cfg is not None
        if self.use_density:
            state_feature_names = datastore.get_vars_names(category="state")
            self._density_idx = len(state_feature_names)
            self._associated_idxs = [
                state_feature_names.index(v)
                for v in density_cfg.associated_vars
            ]
            self.num_state_vars += 1
            for key in state_stats:
                state_stats[key] = torch.cat(
                    [state_stats[key], torch.zeros(1)], dim=0
                )
            state_stats["state_std"][-1] = 1.0
            state_stats["diff_std"][-1] = 1.0

        for key, val in state_stats.items():
            self.register_buffer(key, val, persistent=False)

        state_feature_weights = get_state_feature_weighting(
            config=config, datastore=datastore
        )
        if self.use_density:
            state_feature_weights = np.append(state_feature_weights, 1.0)
        self.feature_weights = torch.tensor(
            state_feature_weights, dtype=torch.float32
        )

        # Double grid output dim. to also output std.-dev.
        self.output_std = bool(args.output_std)
        if self.output_std:
            # Pred. dim. in grid cell
            self.grid_output_dim = 2 * self.num_state_vars
        else:
            # Pred. dim. in grid cell
            self.grid_output_dim = self.num_state_vars
            # Store constant per-variable std.-dev. weighting
            # NOTE that this is the inverse of the multiplicative weighting
            # in wMSE/wMAE
            self.register_buffer(
                "per_var_std",
                self.diff_std / torch.sqrt(self.feature_weights),
                persistent=False,
            )

        # grid_dim from data + static
        (
            self.num_grid_nodes,
            grid_static_dim,
        ) = self.grid_static_features.shape
        self.num_total_grid_nodes = self.num_grid_nodes

        self.interior_input_dim = (
            self.input_steps * self.num_state_vars
            + grid_static_dim
            + num_forcing_vars * num_forcing_steps
        )

        # If datastore_boundary is given, the model is forced from boundary
        self.boundary_forced = datastore_boundary is not None

        if self.boundary_forced:
            # Load static features for boundary
            surface_mask_boundary = datastore_boundary.get_mask(
                surface=True, stacked=True, invert=False
            )
            da_boundary_static_features = (
                self._statistics_datastore_boundary.get_dataarray(
                    category="static", split=None, standardize=True
                )
            )[
                surface_mask_boundary
            ]  # mask static features
            self.register_buffer(
                "boundary_static_features",
                torch.tensor(
                    da_boundary_static_features.values, dtype=torch.float32
                ),
                persistent=False,
            )

            # Compute dimensionalities (e.g. to instantiate MLPs)
            (
                self.num_boundary_nodes,
                boundary_static_dim,
            ) = self.boundary_static_features.shape

            # Compute boundary input dim separately
            num_boundary_forcing_vars = datastore_boundary.get_num_data_vars(
                category="forcing"
            )

            num_past_boundary_steps = args.num_past_boundary_steps
            num_future_boundary_steps = args.num_future_boundary_steps
            include_current_boundary_step = bool(
                int(getattr(args, "current_boundary_step", 1))
            )
            num_boundary_steps = (
                num_past_boundary_steps
                + num_future_boundary_steps
                + (1 if include_current_boundary_step else 0)
            )
            self.boundary_dim = (
                boundary_static_dim
                + num_boundary_forcing_vars * num_boundary_steps
            )
            self.num_total_grid_nodes += self.num_boundary_nodes

        # If datastore_atmosphere is given, the model is forced from atmosphere
        self.atmosphere_forced = datastore_atmosphere is not None
        self.use_atmosphere_g2m = getattr(args, "use_atmosphere_g2m", False)
        # When atmosphere is given but not used in g2m: concat atmosphere
        # forcing (past/future timesteps) to interior state (same grid size).
        self.concat_atmosphere = (
            self.atmosphere_forced and not self.use_atmosphere_g2m
        )

        if self.atmosphere_forced:
            num_atmosphere_forcing_vars = (
                datastore_atmosphere.get_num_data_vars(category="forcing")
            )
            num_past_atmosphere_steps = args.num_past_atmosphere_steps
            num_future_atmosphere_steps = args.num_future_atmosphere_steps
            include_current_atmosphere_step = bool(
                int(getattr(args, "current_atmosphere_step", 1))
            )
            num_atmosphere_steps = (
                num_past_atmosphere_steps
                + num_future_atmosphere_steps
                + (1 if include_current_atmosphere_step else 0)
            )
            atmosphere_windowed_dim = (
                num_atmosphere_forcing_vars * num_atmosphere_steps
            )

            if self.use_atmosphere_g2m:
                # Atmosphere as separate grid nodes in g2m
                atmosphere_mask = datastore_atmosphere.get_atmosphere_mask(
                    stacked=True, invert=False
                )
                da_atmosphere_static_features = (
                    self._statistics_datastore_atmosphere.get_dataarray(
                        category="static", split=None, standardize=True
                    )[atmosphere_mask]
                )
                self.register_buffer(
                    "atmosphere_static_features",
                    torch.tensor(
                        da_atmosphere_static_features.values,
                        dtype=torch.float32,
                    ),
                    persistent=False,
                )
                (
                    self.num_atmosphere_nodes,
                    atmosphere_static_dim,
                ) = self.atmosphere_static_features.shape
                self.atmosphere_dim = (
                    atmosphere_static_dim + atmosphere_windowed_dim
                )
                self.num_total_grid_nodes += self.num_atmosphere_nodes
            else:
                # Atmosphere concat to interior (same grid size)
                self.num_atmosphere_nodes = 0
                self.atmosphere_dim = atmosphere_windowed_dim
                self.interior_input_dim += atmosphere_windowed_dim

        # Instantiate loss function
        self.loss = metrics.get_metric(args.loss)

        # Extend masks for density channel (surface variable, same mask as any
        # single-level ocean variable)
        if self.use_density:
            density_col = loss_mask_np[:, 0:1]
            loss_mask_np = np.concatenate([loss_mask_np, density_col], axis=1)
            density_interior = self.interior_mask[:, 0:1]
            self.interior_mask = np.concatenate(
                [self.interior_mask, density_interior], axis=1
            )

        # Grid loss/metric mask (float, can encode both mask and latitude
        # weighting). Shape (num_grid_nodes, d_features).
        self.register_buffer(
            "loss_mask",
            torch.tensor(loss_mask_np, dtype=torch.float32),
            persistent=False,
        )

        self.register_buffer(
            "interior_mask_bool",
            torch.as_tensor(self.interior_mask, dtype=torch.bool),
            persistent=False,
        )

        self.val_metrics = {
            "mse": [],
        }
        self.test_metrics = {
            "mse": [],
            "mae": [],
        }
        if self.output_std:
            self.test_metrics["output_std"] = []  # Treat as metric

        # For making restoring of optimizer state optional
        self.restore_opt = args.restore_opt

        # For example plotting
        self.n_example_pred = args.n_example_pred
        self.plotted_examples = 0

        # For storing spatial loss maps during evaluation
        self.spatial_loss_maps = []

        # Whether to perform gradient checkpointing at each unroll step.
        # On by default whenever more than one step is unrolled, since the
        # stored activations then scale with the number of steps. Kept as a
        # mutable flag so that the multi-phase schedules in train_model.py can
        # toggle it when they change ar_steps_train between phases.
        self.grad_checkpointing = (
            args.grad_checkpointing or args.ar_steps_train > 1
        ) and not getattr(
            args, "benchmark_disable_grad_checkpointing", False
        )

    def unroll_ckpt_func(self, func, *args):
        """Run one unroll step, optionally with gradient checkpointing.

        Checkpointing only saves memory while a graph is being built for
        backward, so it is skipped during validation/test and under
        torch.no_grad(), where it would only add bookkeeping overhead.
        """
        if (
            self.grad_checkpointing
            and self.training
            and torch.is_grad_enabled()
        ):
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False
            )
        return func(*args)

    def prepare_clamping_params(
        self, config: NeuralLAMConfig, datastore: BaseDatastore
    ):
        """
        Prepare parameters for clamping predicted values to valid range.
        Call this from subclass __init__ after super().__init__.
        """
        state_feature_names = datastore.get_vars_names(category="state")
        lower_lims = config.training.output_clamping.lower
        upper_lims = config.training.output_clamping.upper

        unknown_features_lower = set(lower_lims.keys()) - set(
            state_feature_names
        )
        unknown_features_upper = set(upper_lims.keys()) - set(
            state_feature_names
        )
        if unknown_features_lower or unknown_features_upper:
            raise ValueError(
                "State feature limits were provided for unknown features: "
                f"{unknown_features_lower.union(unknown_features_upper)}"
            )

        sigmoid_sharpness = 1
        softplus_sharpness = 1
        sigmoid_center = 0
        softplus_center = 0

        normalize_clamping_lim = (
            lambda x, feature_idx: (x - self.state_mean[feature_idx])
            / self.state_std[feature_idx]
        )

        sigmoid_lower_upper_idx = []
        sigmoid_lower_lims = []
        sigmoid_upper_lims = []

        softplus_lower_idx = []
        softplus_lower_lims = []

        softplus_upper_idx = []
        softplus_upper_lims = []

        for feature_idx, feature in enumerate(state_feature_names):
            if feature in lower_lims and feature in upper_lims:
                assert lower_lims[feature] < upper_lims[feature], (
                    f'Invalid clamping limits for feature "{feature}", '
                    f"lower: {lower_lims[feature]}, larger than "
                    f"upper: {upper_lims[feature]}"
                )
                sigmoid_lower_upper_idx.append(feature_idx)
                sigmoid_lower_lims.append(
                    normalize_clamping_lim(lower_lims[feature], feature_idx)
                )
                sigmoid_upper_lims.append(
                    normalize_clamping_lim(upper_lims[feature], feature_idx)
                )
            elif feature in lower_lims and feature not in upper_lims:
                softplus_lower_idx.append(feature_idx)
                softplus_lower_lims.append(
                    normalize_clamping_lim(lower_lims[feature], feature_idx)
                )
            elif feature not in lower_lims and feature in upper_lims:
                softplus_upper_idx.append(feature_idx)
                softplus_upper_lims.append(
                    normalize_clamping_lim(upper_lims[feature], feature_idx)
                )

        self.register_buffer(
            "sigmoid_lower_lims", torch.tensor(sigmoid_lower_lims)
        )
        self.register_buffer(
            "sigmoid_upper_lims", torch.tensor(sigmoid_upper_lims)
        )
        self.register_buffer(
            "softplus_lower_lims", torch.tensor(softplus_lower_lims)
        )
        self.register_buffer(
            "softplus_upper_lims", torch.tensor(softplus_upper_lims)
        )

        self.register_buffer(
            "clamp_lower_upper_idx", torch.tensor(sigmoid_lower_upper_idx)
        )
        self.register_buffer(
            "clamp_lower_idx", torch.tensor(softplus_lower_idx)
        )
        self.register_buffer(
            "clamp_upper_idx", torch.tensor(softplus_upper_idx)
        )

        self.clamp_lower_upper = lambda x: (
            self.sigmoid_lower_lims
            + (self.sigmoid_upper_lims - self.sigmoid_lower_lims)
            * torch.sigmoid(sigmoid_sharpness * (x - sigmoid_center))
        )
        self.clamp_lower = lambda x: (
            self.softplus_lower_lims
            + torch.nn.functional.softplus(
                x - softplus_center, beta=softplus_sharpness
            )
        )
        self.clamp_upper = lambda x: (
            self.softplus_upper_lims
            - torch.nn.functional.softplus(
                softplus_center - x, beta=softplus_sharpness
            )
        )

        self.inverse_clamp_lower_upper = lambda x: (
            sigmoid_center
            + utils.inverse_sigmoid(
                (x - self.sigmoid_lower_lims)
                / (self.sigmoid_upper_lims - self.sigmoid_lower_lims)
            )
            / sigmoid_sharpness
        )
        self.inverse_clamp_lower = lambda x: (
            utils.inverse_softplus(
                x - self.softplus_lower_lims, beta=softplus_sharpness
            )
            + softplus_center
        )
        self.inverse_clamp_upper = lambda x: (
            -utils.inverse_softplus(
                self.softplus_upper_lims - x, beta=softplus_sharpness
            )
            + softplus_center
        )

    def apply_density_threshold(self, state):
        """
        Density channel thresholding for autoregressive rollout.

        The predicted density channel is passed through sigmoid and thresholded
        at 0.5.  Where density < 0.5, the density channel and all associated
        ice variables are set to their normalized-zero values. Where density
        >= 0.5, density is set to 1 (in normalized space).

        This is applied only to the state that is fed back as input to the next
        step, not to the state used for loss computation.

        state: (B, num_grid_nodes, d_f) in standardized space
        returns: state with density thresholding applied (same shape)
        """
        if not self.use_density:
            return state

        idx_d = self._density_idx
        density_raw = state[:, :, idx_d]
        density_sigmoid = torch.sigmoid(density_raw)
        ice_present = density_sigmoid > 0.5

        state = state.clone()
        norm_zero = -self.state_mean / self.state_std
        state[:, :, idx_d] = torch.where(
            ice_present,
            (1.0 - self.state_mean[idx_d]) / self.state_std[idx_d],
            norm_zero[idx_d],
        )
        for idx_v in self._associated_idxs:
            state[:, :, idx_v] = torch.where(
                ice_present, state[:, :, idx_v], norm_zero[idx_v]
            )
        return state

    def get_clamped_new_state(self, state_delta, prev_state):
        """
        Clamp prediction to valid range supplied in config.
        Returns the clamped new state after adding delta to original state.

        Instead of the new state being computed as
        $X_{t+1} = X_t + delta$
        The clamped values will be
        $f(f^{-1}(X_t) + delta)$
        which ensures the output stays in the valid range while remaining
        differentiable.

        state_delta: (B, num_grid_nodes, feature_dim)
        prev_state: (B, num_grid_nodes, feature_dim)
        """
        new_state = prev_state + state_delta

        if self.clamp_lower_upper_idx.numel() > 0:
            idx = self.clamp_lower_upper_idx
            new_state[:, :, idx] = self.clamp_lower_upper(
                self.inverse_clamp_lower_upper(prev_state[:, :, idx])
                + state_delta[:, :, idx]
            )

        if self.clamp_lower_idx.numel() > 0:
            idx = self.clamp_lower_idx
            new_state[:, :, idx] = self.clamp_lower(
                self.inverse_clamp_lower(prev_state[:, :, idx])
                + state_delta[:, :, idx]
            )

        if self.clamp_upper_idx.numel() > 0:
            idx = self.clamp_upper_idx
            new_state[:, :, idx] = self.clamp_upper(
                self.inverse_clamp_upper(prev_state[:, :, idx])
                + state_delta[:, :, idx]
            )

        return new_state

    def _create_dataarray_from_tensor(
        self,
        tensor: torch.Tensor,
        time: Union[int, List[int]],
        split: str,
        category: str,
    ) -> xr.DataArray:
        """
        Create an `xr.DataArray` from a tensor, with the correct dimensions and
        coordinates to match the datastore used by the model. This function in
        in effect is the inverse of what is returned by
        `WeatherDataset.__getitem__`.

        Parameters
        ----------
        tensor : torch.Tensor
            The tensor to convert to a `xr.DataArray` with dimensions [time,
            grid_index, feature]. The tensor will be copied to the CPU if it is
            not already there.
        time : Union[int,List[int]]
            The time index or indices for the data, given as integers or a list
            of integers representing epoch time in nanoseconds. The ints will be
            copied to the CPU memory if they are not already there.
        split : str
            The split of the data, either 'train', 'val', or 'test'
        category : str
            The category of the data, either 'state' or 'forcing'
        """
        # TODO: creating an instance of WeatherDataset here on every call is
        # not how this should be done but whether WeatherDataset should be
        # provided to ARModel or where to put plotting still needs discussion
        sdb = self._statistics_datastore_boundary
        sda = self._statistics_datastore_atmosphere
        weather_dataset = WeatherDataset(
            datastore=self._datastore,
            datastore_boundary=self._datastore_boundary,
            datastore_atmosphere=self._datastore_atmosphere,
            split=split,
            use_atmosphere_g2m=getattr(self, "use_atmosphere_g2m", False),
            statistics_datastore=self._statistics_datastore,
            statistics_datastore_boundary=sdb,
            statistics_datastore_atmosphere=sda,
        )
        time = time.detach().cpu()
        time = np.array(time, dtype="datetime64[ns]")

        tensor = tensor.detach().cpu()
        if self.use_density and category == "state":
            tensor = tensor[..., : self._density_idx]
        da = weather_dataset.create_dataarray_from_tensor(
            tensor=tensor, time=time, category=category
        )
        return da

    def _split_params_for_muon(self, flatten, opt_name="muon"):
        """Partition parameters into groups for the Muon-style optimizers.

        Both ``torch.optim.Muon`` and SOAP-Muon only orthogonalize 2D hidden
        weight matrices. Embeddings, the final output layer (matched by name
        via ``args.muon_exclude_patterns``) and all 1D parameters (biases,
        norm gains) go to the auxiliary group instead, which is optimized by
        AdamW for ``muon`` and by plain SOAP for ``soap_muon``. Buffers
        (static graph/grid features, stats) are not in ``named_parameters()``
        and are excluded automatically.

        Parameters
        ----------
        flatten : bool
            If True (the ``muon_flat`` option), contiguous >2D weights (e.g. 4D
            conv filters) are flattened to 2D and optimized by Muon. Otherwise
            they go to the auxiliary group.
        opt_name : str
            Name of the selected optimizer, for logging only.

        Returns
        -------
        (muon_params, conv_muon_params, adamw_params) : tuple of lists
        """
        exclude_patterns = getattr(self.args, "muon_exclude_patterns", [])

        muon_params = []
        conv_muon_params = []
        adamw_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            excluded = any(pat in name for pat in exclude_patterns)
            if excluded or param.ndim < 2:
                adamw_params.append(param)
            elif param.ndim == 2:
                muon_params.append(param)
            else:
                # >2D (e.g. conv). Only Muon-optimize if flattening is enabled
                # and the tensor is contiguous (required for a storage-sharing
                # 2D view).
                if flatten and param.is_contiguous():
                    conv_muon_params.append(param)
                else:
                    if flatten:
                        print(
                            f"[optimizer] {name} is non-contiguous "
                            f"{tuple(param.shape)}; routing to the aux group."
                        )
                    adamw_params.append(param)

        def _summary(params):
            return len(params), sum(p.numel() for p in params)

        m_t, m_n = _summary(muon_params)
        c_t, c_n = _summary(conv_muon_params)
        a_t, a_n = _summary(adamw_params)
        print(
            f"[optimizer={opt_name}] "
            f"orthogonalized(2D): {m_t} tensors, {m_n:,} | "
            f"orthogonalized(conv-flat): {c_t} tensors, {c_n:,} | "
            f"aux: {a_t} tensors, {a_n:,}"
        )

        return muon_params, conv_muon_params, adamw_params

    def configure_optimizers(self):
        opt_name = getattr(self.args, "optimizer", "adamw")
        if opt_name == "adamw":
            opt = torch.optim.AdamW(
                self.parameters(), lr=self.args.lr, betas=(0.9, 0.95), weight_decay=0.1
            )
        elif opt_name in ("muon", "muon_flat"):
            from ..optim import MuonAuxAdam

            muon_params, conv_muon_params, adamw_params = (
                self._split_params_for_muon(
                    flatten=(opt_name == "muon_flat"), opt_name=opt_name
                )
            )
            opt = MuonAuxAdam(
                muon_params,
                conv_muon_params,
                adamw_params,
                lr=self.args.lr,
                weight_decay=self.args.muon_weight_decay,
                momentum=self.args.muon_momentum,
                betas=(0.9, 0.95),
            )
        elif opt_name == "soap":
            from ..optim import SOAP

            # SOAP preconditions parameters of any rank, so unlike Muon it
            # needs no auxiliary optimizer and no parameter split.
            opt = SOAP(
                self.parameters(),
                lr=self.args.lr,
                betas=tuple(self.args.soap_betas),
                weight_decay=self.args.soap_weight_decay,
                precondition_frequency=self.args.soap_precondition_frequency,
            )
        elif opt_name == "soap_muon":
            from ..optim import SoapMuon

            muon_params, _, other_params = self._split_params_for_muon(
                flatten=False, opt_name=opt_name
            )
            param_groups = [
                {"params": muon_params, "orthogonalize": True},
                {"params": other_params, "orthogonalize": False},
            ]
            opt = SoapMuon(
                [g for g in param_groups if g["params"]],
                lr=self.args.lr,
                betas=tuple(self.args.soap_muon_betas),
                weight_decay=self.args.soap_weight_decay,
                precondition_frequency=self.args.soap_precondition_frequency,
                sqrt_correction=not self.args.soap_muon_disable_sqrt,
            )
        else:
            raise ValueError(f"Unknown --optimizer {opt_name}")

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=self.args.epochs,
            eta_min=self.args.min_lr if hasattr(self.args, "min_lr") else 0.0,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    @staticmethod
    def expand_to_batch(x, batch_size):
        """
        Expand tensor with initial batch dimension
        """
        if x.ndim == 3:
            return x
        else:
            return x.unsqueeze(0).expand(batch_size, -1, -1)

    def predict_step(
        self,
        prev_state,
        prev_prev_state,
        forcing,
        boundary_forcing,
        atmosphere_forcing,
    ):
        """
        Step state one step ahead using prediction model, X_{t-1}, X_t -> X_t+1
        prev_state: (B, num_grid_nodes, feature_dim), X_t
        prev_prev_state: (B, num_grid_nodes, feature_dim), X_{t-1}
        forcing: (B, num_grid_nodes, forcing_dim)
        boundary_forcing: (B, num_boundary_nodes, boundary_forcing_dim)
        atmosphere_forcing: (B, num_atmosphere_nodes, atmosphere_forcing_dim)
        """
        raise NotImplementedError("No prediction step implemented")

    def unroll_prediction(
        self, init_states, forcing, boundary_forcing, atmosphere_forcing
    ):
        """
        Roll out prediction taking multiple autoregressive steps with model
        init_states: (B, input_steps, num_grid_nodes, d_f)
        forcing: (B, pred_steps, num_grid_nodes, d_static_f)
        boundary_forcing: (B, pred_steps, num_boundary_nodes, d_boundary_f)
        atmosphere_forcing:(B, pred_steps, num_atmosphere_nodes, d_atmosphere_f)
        """
        if self.input_steps == 1:
            prev_state = init_states[:, 0]
            prev_prev_state = None
        else:
            prev_prev_state = init_states[:, 0]
            prev_state = init_states[:, 1]
        prediction_list = []
        pred_std_list = []
        pred_steps = forcing.shape[1]

        for i in range(pred_steps):
            forcing_step = forcing[:, i]

            if self.boundary_forced:
                boundary_forcing_step = boundary_forcing[:, i]
            else:
                boundary_forcing_step = None

            if self.atmosphere_forced:
                atmosphere_forcing_step = atmosphere_forcing[:, i]
            else:
                atmosphere_forcing_step = None

            pred_state, pred_std = self.unroll_ckpt_func(
                self.predict_step,
                prev_state,
                prev_prev_state,
                forcing_step,
                boundary_forcing_step,
                atmosphere_forcing_step,
            )
            # state: (B, num_grid_nodes, d_f)
            # pred_std: (B, num_grid_nodes, d_f) or None

            prediction_list.append(pred_state)
            if self.output_std:
                pred_std_list.append(pred_std)

            # Apply density thresholding to the state fed back as input
            # (raw predictions kept for loss computation)
            feedback_state = self.apply_density_threshold(pred_state)

            # Update conditioning states
            if self.input_steps >= 2:
                prev_prev_state = prev_state
            prev_state = feedback_state

        prediction = torch.stack(
            prediction_list, dim=1
        )  # (B, pred_steps, num_grid_nodes, d_f)
        if self.output_std:
            pred_std = torch.stack(
                pred_std_list, dim=1
            )  # (B, pred_steps, num_grid_nodes, d_f)
        else:
            pred_std = self.per_var_std  # (d_f,)

        return prediction, pred_std

    def common_step(self, batch):
        """
        Predict on single batch batch consists of:
        init_states: (B, input_steps, num_grid_nodes, d_features)
        target_states: (B, pred_steps, num_grid_nodes, d_features)
        forcing_features: (B, pred_steps, num_grid_nodes, d_forcing),
        boundary_forcing:
            (B, pred_steps, num_boundary_nodes, d_boundary_forcing),
        atmosphere_forcing:
            (B, pred_steps, num_atmosphere_nodes, d_atmosphere_forcing),
        where index 0 corresponds to index 1 of init_states
        """
        (
            init_states,
            target_states,
            forcing,
            boundary_forcing,
            atmosphere_forcing,
            batch_times,
        ) = batch

        prediction, pred_std = self.unroll_prediction(
            init_states, forcing, boundary_forcing, atmosphere_forcing
        )  # (B, pred_steps, num_grid_nodes, d_f)
        # prediction: (B, pred_steps, num_grid_nodes, d_f)
        # pred_std: (B, pred_steps, num_grid_nodes, d_f) or (d_f,)

        return prediction, target_states, pred_std, batch_times

    def training_step(self, batch):
        """
        Train on single batch
        """
        prediction, target, pred_std, _ = self.common_step(batch)

        # Compute loss
        batch_loss = torch.mean(
            self.loss(prediction, target, pred_std, mask=self.loss_mask)
        )  # mean over unrolled times and batch

        log_dict = {"train_loss": batch_loss}
        self.log_dict(
            log_dict,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch[0].shape[0],
        )
        return batch_loss

    def all_gather_cat(self, tensor_to_gather):
        """
        Gather tensors across all ranks, and concatenate across dim. 0 (instead
        of stacking in new dim. 0)

        tensor_to_gather: (d1, d2, ...), distributed over K ranks

        returns: (K*d1, d2, ...)
        """
        return self.all_gather(tensor_to_gather).flatten(0, 1)

    # newer lightning versions requires batch_idx argument, even if unused
    # pylint: disable-next=unused-argument
    def validation_step(self, batch, batch_idx):
        """
        Run validation on single batch
        """
        prediction, target, pred_std, _ = self.common_step(batch)

        time_step_loss = torch.mean(
            self.loss(prediction, target, pred_std, mask=self.loss_mask),
            dim=0,
        )  # (time_steps-1)
        mean_loss = torch.mean(time_step_loss)

        # Log loss per time step forward and mean
        val_log_dict = {
            f"val_loss_unroll{step}": time_step_loss[step - 1]
            for step in self.args.val_steps_to_log
            if step <= len(time_step_loss)
        }
        val_log_dict["val_mean_loss"] = mean_loss
        self.log_dict(
            val_log_dict,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch[0].shape[0],
        )

        # Store MSEs
        entry_mses = metrics.mse(
            prediction,
            target,
            pred_std,
            mask=self.loss_mask,
            sum_vars=False,
        )  # (B, pred_steps, d_f)
        self.val_metrics["mse"].append(entry_mses)

    def on_validation_epoch_end(self):
        """
        Compute val metrics at the end of val epoch
        """
        # Create error maps for all test metrics
        self.aggregate_and_plot_metrics(self.val_metrics, prefix="val")

        # Clear lists with validation metrics values
        for metric_list in self.val_metrics.values():
            metric_list.clear()

    # pylint: disable-next=unused-argument
    def test_step(self, batch, batch_idx):
        """
        Run test on single batch
        """
        # TODO Here batch_times can be used for plotting routines
        prediction, target, pred_std, batch_times = self.common_step(batch)
        # prediction: (B, pred_steps, num_grid_nodes, d_f) pred_std: (B,
        # pred_steps, num_grid_nodes, d_f) or (d_f,)

        time_step_loss = torch.mean(
            self.loss(prediction, target, pred_std, mask=self.loss_mask),
            dim=0,
        )  # (time_steps-1,)
        mean_loss = torch.mean(time_step_loss)

        # Log loss per time step forward and mean
        test_log_dict = {
            f"test_loss_unroll{step}": time_step_loss[step - 1]
            for step in self.args.val_steps_to_log
        }
        test_log_dict["test_mean_loss"] = mean_loss

        self.log_dict(
            test_log_dict,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch[0].shape[0],
        )

        # Compute all evaluation metrics for error maps Note: explicitly list
        # metrics here, as test_metrics can contain additional ones, computed
        # differently, but that should be aggregated on_test_epoch_end
        for metric_name in ("mse", "mae"):
            metric_func = metrics.get_metric(metric_name)
            batch_metric_vals = metric_func(
                prediction,
                target,
                pred_std,
                mask=self.loss_mask,
                sum_vars=False,
            )  # (B, pred_steps, d_f)
            self.test_metrics[metric_name].append(batch_metric_vals)

        if self.output_std:
            # Store output std. per variable, spatially averaged
            mean_pred_std = torch.mean(
                pred_std[..., self.interior_mask_bool, :], dim=-2
            )  # (B, pred_steps, d_f)
            self.test_metrics["output_std"].append(mean_pred_std)

        # Save per-sample spatial loss for specific times
        spatial_loss = self.loss(
            prediction,
            target,
            pred_std,
            mask=self.loss_mask,
            average_grid=False,
        )  # (B, pred_steps, num_grid_nodes)
        log_spatial_losses = spatial_loss[
            :, [step - 1 for step in self.args.val_steps_to_log]
        ]
        self.spatial_loss_maps.append(log_spatial_losses)
        # (B, N_log, num_grid_nodes)

        # Plot example predictions (on rank 0 only)
        if (
            self.trainer.is_global_zero
            and self.plotted_examples < self.n_example_pred
        ):
            # Need to plot more example predictions
            n_additional_examples = min(
                prediction.shape[0],
                self.n_example_pred - self.plotted_examples,
            )

            self.plot_examples(
                batch,
                n_additional_examples,
                prediction=prediction,
                split="test",
            )

    def plot_examples(self, batch, n_examples, split, prediction=None):
        """
        Plot the first n_examples forecasts from batch

        batch: batch with data to plot corresponding forecasts for n_examples:
        number of forecasts to plot prediction: (B, pred_steps, num_grid_nodes,
        d_f), existing prediction.
            Generate if None.
        """
        if prediction is None:
            prediction, target, _, _ = self.common_step(batch)

        target = batch[1]
        time = batch[-1]

        # Rescale to original data scale
        prediction_rescaled = prediction * self.state_std + self.state_mean
        target_rescaled = target * self.state_std + self.state_mean

        # Iterate over the examples
        for pred_slice, target_slice, time_slice in zip(
            prediction_rescaled[:n_examples],
            target_rescaled[:n_examples],
            time[:n_examples],
        ):
            # Each slice is (pred_steps, num_grid_nodes, d_f)
            self.plotted_examples += 1  # Increment already here

            da_prediction = self._create_dataarray_from_tensor(
                tensor=pred_slice,
                time=time_slice,
                split=split,
                category="state",
            ).unstack("grid_index")
            da_target = self._create_dataarray_from_tensor(
                tensor=target_slice,
                time=time_slice,
                split=split,
                category="state",
            ).unstack("grid_index")

            # Save as Zarr
            save_dir = os.path.join(self.logger.save_dir, "example_forecasts")
            os.makedirs(save_dir, exist_ok=True)
            example_name = f"example_{self.plotted_examples}.zarr"

            ds_examples = xr.Dataset(
                {
                    "target": da_target,
                    "prediction": da_prediction,
                }
            )

            save_path = os.path.join(save_dir, example_name)
            # Coordinates inherit zarr v2 codecs from the datastore, which
            # zarr>=3 rejects when writing
            ds_examples.drop_encoding().to_zarr(save_path, mode="w")

            plot_pred = pred_slice
            plot_target = target_slice
            if self.use_density:
                plot_pred = pred_slice[..., : self._density_idx]
                plot_target = target_slice[..., : self._density_idx]

            var_vmin = (
                torch.minimum(
                    plot_pred.flatten(0, 1).min(dim=0)[0],
                    plot_target.flatten(0, 1).min(dim=0)[0],
                )
                .cpu()
                .numpy()
            )  # (d_f,)
            var_vmax = (
                torch.maximum(
                    plot_pred.flatten(0, 1).max(dim=0)[0],
                    plot_target.flatten(0, 1).max(dim=0)[0],
                )
                .cpu()
                .numpy()
            )  # (d_f,)
            var_vranges = list(zip(var_vmin, var_vmax))

            # Iterate over prediction horizon time steps
            for t_i, _ in enumerate(zip(pred_slice, target_slice), start=1):
                # Create one figure per variable at this time step
                var_figs = [
                    vis.plot_prediction(
                        datastore=self._datastore,
                        title=f"{var_name} ({var_unit}), "
                        f"t={t_i} ({self._datastore.step_length * t_i} h)",
                        vrange=var_vrange,
                        da_prediction=da_prediction.isel(
                            state_feature=var_i, time=t_i - 1
                        ).squeeze(),
                        da_target=da_target.isel(
                            state_feature=var_i, time=t_i - 1
                        ).squeeze(),
                    )
                    for var_i, (var_name, var_unit, var_vrange) in enumerate(
                        zip(
                            self._datastore.get_vars_names("state"),
                            self._datastore.get_vars_units("state"),
                            var_vranges,
                        )
                    )
                ]

                example_i = self.plotted_examples

                for var_name, fig in zip(
                    self._datastore.get_vars_names("state"), var_figs
                ):

                    # We need treat logging images differently for different
                    # loggers. WANDB can log multiple images to the same key,
                    # while other loggers, as MLFlow, need unique keys for
                    # each image.
                    if isinstance(self.logger, pl.loggers.WandbLogger):
                        key = f"{var_name}_example_{example_i}"
                    else:
                        key = f"{var_name}_example"

                    if hasattr(self.logger, "log_image"):
                        self.logger.log_image(key=key, images=[fig], step=t_i)
                    else:
                        warnings.warn(
                            f"{self.logger} does not support image logging."
                        )

                plt.close(
                    "all"
                )  # Close all figs for this time step, saves memory

    def create_metric_log_dict(self, metric_tensor, prefix, metric_name):
        """
        Put together a dict with everything to log for one metric. Also saves
        plots as pdf and csv if using test prefix.

        metric_tensor: (pred_steps, d_f), metric values per time and variable
        prefix: string, prefix to use for logging metric_name: string, name of
        the metric

        Return: log_dict: dict with everything to log for given metric
        """
        log_dict = {}
        metric_fig = vis.plot_error_map(
            errors=metric_tensor,
            datastore=self._datastore,
        )
        full_log_name = f"{prefix}_{metric_name}"
        log_dict[full_log_name] = metric_fig

        if prefix == "test":
            # Save pdf
            metric_fig.savefig(
                os.path.join(self.logger.save_dir, f"{full_log_name}.pdf")
            )
            # Save errors also as csv
            np.savetxt(
                os.path.join(self.logger.save_dir, f"{full_log_name}.csv"),
                metric_tensor.cpu().numpy(),
                delimiter=",",
            )

        # Check if metrics are watched, log exact values for specific vars
        var_names = self._datastore.get_vars_names(category="state")
        if full_log_name in self.args.metrics_watch:
            for var_i, timesteps in self.args.var_leads_metrics_watch.items():
                var_name = var_names[var_i]
                for step in timesteps:
                    key = f"{full_log_name}_{var_name}_step_{step}"
                    log_dict[key] = metric_tensor[step - 1, var_i]

        return log_dict

    def aggregate_and_plot_metrics(self, metrics_dict, prefix):
        """
        Aggregate and create error map plots for all metrics in metrics_dict

        metrics_dict: dictionary with metric_names and list of tensors
            with step-evals.
        prefix: string, prefix to use for logging
        """
        log_dict = {}
        for metric_name, metric_val_list in metrics_dict.items():
            metric_tensor = self.all_gather_cat(
                torch.cat(metric_val_list, dim=0)
            )  # (N_eval, pred_steps, d_f)

            if self.trainer.is_global_zero:
                metric_tensor_averaged = torch.mean(metric_tensor, dim=0)
                # (pred_steps, d_f)

                # Strip density channel before plotting/logging
                if self.use_density:
                    metric_tensor_averaged = metric_tensor_averaged[
                        :, : self._density_idx
                    ]

                # Take square root after averaging to change squared metrics
                if "mse" in metric_name:
                    metric_tensor_averaged = torch.sqrt(metric_tensor_averaged)
                    metric_name = metric_name.replace("mse", "rmse")
                elif metric_name.endswith("_squared"):
                    metric_tensor_averaged = torch.sqrt(metric_tensor_averaged)
                    metric_name = metric_name[: -len("_squared")]

                # NOTE: we here assume rescaling for all metrics is linear
                state_std = self.state_std[: metric_tensor_averaged.shape[1]]
                metric_rescaled = metric_tensor_averaged * state_std
                # (pred_steps, d_f)
                log_dict.update(
                    self.create_metric_log_dict(
                        metric_rescaled, prefix, metric_name
                    )
                )

        # Ensure that log_dict has structure for
        # logging as dict(str, plt.Figure)
        assert all(
            isinstance(key, str) and isinstance(value, plt.Figure)
            for key, value in log_dict.items()
        )

        if self.trainer.is_global_zero and not self.trainer.sanity_checking:

            current_epoch = self.trainer.current_epoch

            for key, figure in log_dict.items():
                # For other loggers than wandb, add epoch to key.
                # Wandb can log multiple images to the same key, while other
                # loggers, such as MLFlow need unique keys for each image.
                if not isinstance(self.logger, pl.loggers.WandbLogger):
                    key = f"{key}-{current_epoch}"

                if hasattr(self.logger, "log_image"):
                    self.logger.log_image(key=key, images=[figure])

            plt.close("all")  # Close all figs

    def on_test_epoch_end(self):
        """
        Compute test metrics and make plots at the end of test epoch. Will
        gather stored tensors and perform plotting and logging on rank 0.
        """
        # Create error maps for all test metrics
        self.aggregate_and_plot_metrics(self.test_metrics, prefix="test")

        # Plot spatial loss maps
        spatial_loss_tensor = self.all_gather_cat(
            torch.cat(self.spatial_loss_maps, dim=0)
        )  # (N_test, N_log, num_grid_nodes)
        if self.trainer.is_global_zero:
            mean_spatial_loss = torch.mean(
                spatial_loss_tensor, dim=0
            )  # (N_log, num_grid_nodes)

            loss_map_figs = [
                vis.plot_spatial_error(
                    error=loss_map,
                    datastore=self._datastore,
                    title=f"Test loss, t={t_i} "
                    f"({self._datastore.step_length * t_i} h)",
                )
                for t_i, loss_map in zip(
                    self.args.val_steps_to_log, mean_spatial_loss
                )
            ]

            # log all to same key, sequentially
            for i, fig in enumerate(loss_map_figs):
                key = "test_loss"
                if not isinstance(self.logger, pl.loggers.WandbLogger):
                    key = f"{key}_{i}"
                if hasattr(self.logger, "log_image"):
                    self.logger.log_image(key=key, images=[fig])

            # also make without title and save as pdf
            pdf_loss_map_figs = [
                vis.plot_spatial_error(
                    error=loss_map, datastore=self._datastore
                )
                for loss_map in mean_spatial_loss
            ]
            pdf_loss_maps_dir = os.path.join(
                self.logger.save_dir, "spatial_loss_maps"
            )
            os.makedirs(pdf_loss_maps_dir, exist_ok=True)
            for t_i, fig in zip(self.args.val_steps_to_log, pdf_loss_map_figs):
                fig.savefig(os.path.join(
                    pdf_loss_maps_dir, f"loss_t{t_i}.pdf"))
            # save mean spatial loss as .pt file also
            torch.save(
                mean_spatial_loss.cpu(),
                os.path.join(self.logger.save_dir, "mean_spatial_loss.pt"),
            )

        self.spatial_loss_maps.clear()

    @staticmethod
    def _fix_legacy_muon_lr(checkpoint):
        """Recover the scheduled LR from old MuonAuxAdam checkpoints.

        Checkpoints written before ``MuonAuxAdam.state_dict`` included its own
        param_groups ("base") carry no scheduled LR for the wrapper, only the
        one-step-stale LRs of the sub-optimizers. Copy the exact LR from the
        saved scheduler into them, so that the fallback in
        ``MuonAuxAdam.load_state_dict`` restores the right value instead of
        continuing from the initial LR.
        """
        opt_states = checkpoint.get("optimizer_states") or []
        sched_states = checkpoint.get("lr_schedulers") or []
        for opt_state, sched_state in zip(opt_states, sched_states):
            if not isinstance(opt_state, dict) or "base" in opt_state:
                continue
            last_lr = (sched_state or {}).get("_last_lr")
            if not last_lr:
                continue
            for key in ("muon", "adamw"):
                sub = opt_state.get(key)
                if isinstance(sub, dict):
                    for group in sub.get("param_groups", []):
                        group["lr"] = last_lr[0]

    @staticmethod
    def _check_optimizer_matches(checkpoint, current_opt):
        """Reject ``--restore_opt`` across a change of optimizer.

        Optimizer states are not interchangeable between the optimizers
        offered by ``--optimizer``: loading one into another either raises a
        bare ``KeyError`` deep inside ``load_state_dict`` or, when the state
        dicts happen to have the same shape (AdamW and SOAP both keep
        ``exp_avg``/``exp_avg_sq`` over a single group), silently restores
        moments that mean something else.
        """
        saved_opt = checkpoint.get("optimizer_name")
        if saved_opt is None:
            # Written before optimizer_name was recorded. The wrapped
            # Muon state is still recognizable by its sub-optimizer keys.
            states = checkpoint.get("optimizer_states") or []
            if not states or not isinstance(states[0], dict):
                return
            was_muon = "base" in states[0] or "muon" in states[0]
            if was_muon == current_opt.startswith("muon"):
                return
            saved_opt = "muon/muon_flat" if was_muon else "adamw/soap"
        elif saved_opt == current_opt:
            return

        raise ValueError(
            f"Checkpoint was written with --optimizer {saved_opt} but this "
            f"run uses --optimizer {current_opt}. Their states are not "
            f"interchangeable; drop --restore_opt to start {current_opt} "
            "from a clean state."
        )

    def on_save_checkpoint(self, checkpoint):
        """Record which optimizer wrote the state, so that resuming into a
        different one is caught by ``on_load_checkpoint``."""
        checkpoint["optimizer_name"] = getattr(self.args, "optimizer", "adamw")

    def on_load_checkpoint(self, checkpoint):
        """
        Perform any changes to state dict before loading checkpoint
        """
        loaded_state_dict = checkpoint["state_dict"]

        # Fix for loading older models after IneractionNet refactoring, where
        # the grid MLP was moved outside the encoder InteractionNet class
        if "g2m_gnn.grid_mlp.0.weight" in loaded_state_dict:
            replace_keys = list(
                filter(
                    lambda key: key.startswith("g2m_gnn.grid_mlp"),
                    loaded_state_dict.keys(),
                )
            )
            for old_key in replace_keys:
                new_key = old_key.replace(
                    "g2m_gnn.grid_mlp", "encoding_grid_mlp"
                )
                loaded_state_dict[new_key] = loaded_state_dict[old_key]
                del loaded_state_dict[old_key]
        if self.restore_opt:
            self._check_optimizer_matches(
                checkpoint, getattr(self.args, "optimizer", "adamw")
            )
            self._fix_legacy_muon_lr(checkpoint)
        else:
            opt = self.configure_optimizers()
            if isinstance(opt, dict):
                checkpoint["optimizer_states"] = [
                    opt["optimizer"].state_dict()]
                lr_cfg = opt.get("lr_scheduler")
                if lr_cfg is not None:
                    sched = (
                        lr_cfg["scheduler"]
                        if isinstance(lr_cfg, dict) and "scheduler" in lr_cfg
                        else lr_cfg
                    )
                    # Match current schedule (e.g. SequentialLR on finetune)
                    checkpoint["lr_schedulers"] = [sched.state_dict()]
                else:
                    checkpoint.pop("lr_schedulers", None)
            else:
                checkpoint["optimizer_states"] = [opt.state_dict()]
                checkpoint.pop("lr_schedulers", None)
