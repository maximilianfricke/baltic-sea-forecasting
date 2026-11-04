# Graph viewer and partition cost model

An interactive 3D viewer for Njord's global graphs, built to understand how
information flows through the graph and to compare strategies for splitting
the graph across GPUs.

It shows every node layer and edge set as a toggleable layer, partitions the
graph with a chosen strategy, and reports what each split costs: how much
data every GPU has to receive from the others (communication), and whether
memory is spread evenly (a check).

## Quick start

```bash
# Environment (CPU-only torch is enough)
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/Scripts/python.exe -e . polyscope

# 1-degree sample data with prebuilt graphs (~190 MB), see the main README
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='deinal/njord-sample-data', repo_type='dataset', local_dir='sample_data')"

# Open the viewer
python -m neural_lam.graph_viewer --config_path sample_data/global_ocean_1.yaml --strategy rcb --k 4
```

Useful options: `--graph_name` (default `global_cluster_graph`), `--strategy`
(`none`, `lon_bands`, `rcb`), `--k`, `--cost_model` (a JSON file or
`uniform`), `--show`/`--hide` (structures to toggle at start, e.g.
`"g2m edges"`), `--screenshot out.png` (render one frame and exit).

The viewer needs a display, so run it on your own machine, not on a cluster
login node.

## Where the graph comes from

The viewer does not build or change any connections. It loads a graph saved
by the graph builder (`neural_lam/create_global_graph.py`) through the same
`utils.load_graph` the model uses, so it shows exactly what the model trains
on. The 1-degree sample graphs come prebuilt with the Hugging Face sample
data.

A global graph has three node layers and six edge sets:

| Layer / edge set | What it is |
|---|---|
| Grid | Ocean cells of the dataset (42,348 at 1 degree). Input and output live here. |
| Mesh 0, Mesh 1 | Coarser processing nodes (8,456 and 2,097 in `global_cluster_graph`). |
| g2m | Grid to mesh 0: each grid cell to the mesh nodes within a radius (encoder). |
| m2g | Mesh 0 to grid: each grid cell from its 3 nearest mesh nodes (decoder). |
| m2m 0, m2m 1 | Neighbours within one mesh level (processor). |
| up 0->1, down 1->0 | Each mesh-0 node to/from its nearest mesh-1 node. |

In 3D the grid lies on the unit sphere and each mesh level on a larger shell.
The radii are only for display: all layers really lie on the same sphere.

## Panel overview

- **Visibility**: one checkbox per layer and edge set, node size, edge width.
- **Color by**: layer, partition (cut edges in white) or information flow.
- **Partitioning**: strategy and number of partitions, then
  - **Communication** (the comparison): the GPU that receives the most data
    and how much, a breakdown of where its traffic comes from, and a table
    for all GPUs.
  - **Memory** (a check): whether memory is balanced, the memory model the
    balancing is based on, and memory per GPU.
- **Information flow**: click a node to see which nodes it reaches (or is
  reached from) within a number of hops, over the edge sets currently shown.

## The cost model

Every edge is assumed to live on the GPU that owns its receiver node. Then,
for GPU p and edge set s:

- **H(p, s), the halo**: distinct nodes on other GPUs that send an edge of s
  into GPU p. Each such node's state must be copied to GPU p once, however
  many of p's nodes it feeds.
- **u(s), uses**: how many times edge set s is applied per forecast step
  (in the baseline: m2m 0 6x, m2m 1 4x, g2m 3x, up 3x, m2g 1x, down 1x,
  because the posterior encoder, prior and decoder each run part of the
  graph).

**Communication** is the number of node states GPU p receives per forecast
step:

    C(p) = sum_s u(s) * H(p, s)

Multiply by hidden_dim x bytes per value for bytes. A full training step
also multiplies by 3 (forward, recompute under gradient checkpointing,
backward) and by the number of unrolled steps, but those factors are the
same for every strategy and are left out.

**Memory** is what GPU p stores during one gradient-checkpointed forecast
step, the part of peak memory that partitioning controls:

    M(p) = sum_l n(l) * N(p, l) + sum_s m(s) * E(p, s) + sum_s u(s) * H(p, s)

with N and E the nodes and edges GPU p holds, and n(l), m(s) the values
stored per node and per edge (in units of hidden_dim). The last term is the
copies of halo nodes; it is tiny (about 0.03% per GPU).

Only terms that differ between partitionings are kept. Fixed memory (weights,
gradients, optimizer state, checkpoint inputs) and constant factors are
dropped, so these numbers compare strategies; they are not absolute GB.

### Cost model file

`weights/realistic_baseline.json` holds n, m and u for the realistic
Graph-EFM baseline of the benchmarking work (`jobs/submit_realistic_baseline.sh`
on the `max` branch: hidden_dim 64, 2 processor layers, kl_beta 1,
`global_cluster_graph`). It was measured on the benchmarking branch (commit
in `source`); measuring again on `main` gives identical values.

- `memory`: values stored per node and per edge, measured by running one
  forecast step and recording every tensor saved for backward.
- `usage`: uses per step, counted with hooks on the message-passing layers.
- `source`: the exact arguments, graph, git commit and date of the
  measurement; `notes`: approximations made.

Sanity check: the measured total (427 M values per step, 1.6 GiB at fp32 for
batch size 1) matches the 3.41 GiB peak measured on a 1080 Ti at batch size
2.

To measure another configuration (about a minute on CPU):

```bash
python -m neural_lam.graph_viewer.measure_weights \
    --output my_model.json --name "My configuration" -- \
    <the train_model arguments of your run>
```

## Balancing

Partition borders are placed so that every GPU gets the same memory:

1. Each node gets a weight: its own stored values plus those of the edges it
   receives, `w(v) = n(l) + sum_s m(s) * (edges of s into v)`. With the
   baseline model a mesh-0 node weighs about 493, a mesh-1 node 206 and a
   grid node 49.
2. All nodes of all layers are pooled by position on the sphere.
3. The strategy cuts the sphere so every piece holds equal total weight.
4. The same borders are applied to every layer, so nodes stacked on
   different shells land on the same GPU.

Because the borders target this weight directly, every strategy here reaches
a memory imbalance of about 1.00. That is why communication is the headline
and memory is only a check.

## Adding your own partitioning strategy

Strategies live in `partition.py`. A geometric strategy needs two methods:

```python
class MyStrategy(Partitioner):
    name = "my_strategy"

    def fit(self, xyz, weights=None):
        # xyz: (N, 3) unit vectors of all nodes; weights: (N,) node loads.
        # Choose borders so that each of self.k parts gets equal weight.
        ...
        return self

    def assign(self, xyz):
        # Return (N,) partition ids in [0, self.k) for any points.
        ...

STRATEGIES[MyStrategy.name] = MyStrategy
```

It then appears in the Strategy dropdown and in `--strategy`. The existing
`LongitudeBands` and `RecursiveCoordinateBisection` are short examples.

Strategies that are not purely geometric (e.g. METIS on the graph
structure, or assignments computed elsewhere) do not fit this interface yet:
the viewer would need an option to load a fixed node-to-GPU assignment per
layer.

## Questions you are likely to have

**Is this the paper's model?** No. It is the 1-degree graph with the
realistic benchmark baseline configuration. The 0.25-degree graphs are not in
the sample data.

**Why count halo nodes, not cut edges?** Cut edges overcount. In m2g, one
mesh node feeds many grid cells across a border but its state is sent once:
under RCB with 4 GPUs, m2g has 2,920 cut edges but only 412 halo nodes.

**Where does most communication come from?** m2m 0 (it is used 6 times per
step) and g2m (its senders are the dense grid). Both scale with border
length, so strategies mainly differ in total border length. At higher grid
resolution g2m is expected to dominate.

**What if our implementation stores edges with the sender instead?** Then
memory and halo move: g2m edges would sit with grid nodes. The cost model
would need to change; the assumption is stated in `costs.py`.

**Can I use it on another graph?** Yes, but the cost model must name the
same layers and edge sets. Use `--cost_model uniform` to look around, and
measure a cost model for real comparisons. The viewer warns when the model
is uniform.

**How accurate is the memory model?** Good for comparing strategies, not for
absolute GB. Up/down edges and mesh-0 nodes have equal counts in this graph,
so their stored values cannot be told apart and are split evenly (noted in
the JSON). It was measured on CPU; CUDA may save slightly different tensors.

**What does information flow show?** Hop distance from a clicked node over
the edge sets currently visible. It follows what you show, not the model's
exact order of operations: show g2m, m2m 0, up and m2m 1 to trace the
encoder.

## Files

| File | Purpose |
|---|---|
| `graph.py` | Loads a saved graph into node layers and edge sets |
| `partition.py` | Partitioning strategies and cut statistics |
| `costs.py` | Cost model, node weights, M(p) and C(p) |
| `measure_weights.py` | Measures a cost model for a train_model configuration |
| `viewer.py`, `__main__.py` | The polyscope viewer and its command line |
| `weights/realistic_baseline.json` | Cost model of the realistic baseline |

Tests: `pytest --noconftest tests/test_graph_viewer_partition.py
tests/test_graph_viewer_costs.py` (`--noconftest` because `tests/conftest.py`
currently imports a datastore module that does not exist on this branch).
