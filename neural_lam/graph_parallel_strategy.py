"""Lightning process launching without DDP wrapping or buffer broadcasts."""

# Third-party
import torch.distributed as dist
from lightning_fabric.utilities.optimizer import _optimizers_to_device
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.trainer.states import TrainerFn

# Local
from .graph_parallel import configure_graph_parallel, parameter_reductions


class GraphParallelStrategy(DDPStrategy):
    """One logical batch shared by all ranks; gradients reduce in the model hook.

    Inherit process launching, collectives and rank-zero checkpoint I/O from DDP,
    but never construct DistributedDataParallel or synchronize graph buffers.
    """

    def __init__(self, size, partition_path=None, **kwargs):
        super().__init__(**kwargs)
        self.graph_parallel_size = size
        self.partition_path = partition_path

    def setup(self, trainer):
        if self.world_size != self.graph_parallel_size:
            raise ValueError(
                "graph_parallel_size must equal the participating GPU/process count"
            )
        if self.num_nodes != 1:
            raise ValueError("Graph parallelism currently supports one host")
        if getattr(self.precision_plugin, "scaler", None) is not None:
            raise ValueError(
                "Graph parallelism does not yet support FP16 gradient scaling; use FP32"
            )
        self.accelerator.setup(trainer)
        self.precision_plugin.convert_module(self.model)
        self.model_to_device()
        # Only weights are replicated. Graph buffers intentionally differ.
        for parameter in self.model.parameters():
            dist.broadcast(parameter.data, src=0)
        if trainer.state.fn == TrainerFn.FITTING:
            self.setup_optimizers(trainer)
        self.setup_precision_plugin()
        if trainer.state.fn == TrainerFn.FITTING:
            _optimizers_to_device(self.optimizers, self.root_device)

    def model_to_device(self):
        configure_graph_parallel(
            self.model,
            self.graph_parallel_size,
            self.global_rank,
            self.partition_path,
        )
        self.model.graph_parameter_reductions = parameter_reductions(self.model)
        super().model_to_device()

    def configure_ddp(self):
        """The model stays a LightningModule; reductions are explicit."""

    @property
    def restore_checkpoint_after_setup(self):
        # Ownership must exist before validating a checkpoint's partition and
        # restoring the RNG on the correct device.
        return True
