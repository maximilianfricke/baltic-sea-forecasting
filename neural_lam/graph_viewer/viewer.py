"""Interactive 3D viewer of a global graph, built on polyscope.

Layout: the grid lies on the unit sphere and mesh level l on a shell of
radius 1 + (l + 1) * level_gap, so edges inside a layer lie on one shell and
g2m, m2g, up and down edges run between shells.

The panel has:
- a toggle per layer and per edge set,
- colouring by layer, by partition or by information flow,
- partition strategy and number of partitions, balanced by estimated memory
  load, with the relative memory M(p) and communication C(p) of every
  partition (see ``costs.py``),
- showing only cut edges,
- information flow from a clicked node over the edge sets currently shown.
"""

# Standard library
import os
from typing import Dict, List, Optional

# Third-party
import matplotlib
import numpy as np
import polyscope as ps
import polyscope.imgui as psim

# Local
from .costs import CostModel, node_weights, partition_costs
from .graph import EdgeSet, ViewerGraph
from .partition import STRATEGIES, cut_mask, cut_stats, partition_layers

GRID_COLOR = (0.20, 0.45, 0.85)
MESH_COLORS = [(0.95, 0.55, 0.10), (0.85, 0.25, 0.15), (0.60, 0.15, 0.45)]
EDGE_KIND_COLORS = {
    "g2m": (0.15, 0.65, 0.30),
    "m2g": (0.50, 0.30, 0.75),
    "m2m": (0.95, 0.60, 0.15),
    "up": (0.10, 0.70, 0.80),
    "down": (0.85, 0.30, 0.60),
}
DIM_COLOR = np.array([0.35, 0.35, 0.38])
CUT_COLOR = np.array([1.0, 1.0, 1.0])
COLOR_MODES = ["Layer", "Partition", "Flow"]
# Memory imbalance (max / mean) above which the memory check warns
MAX_BALANCED_IMBALANCE = 1.05
TEXT_COLOR = (1.0, 1.0, 1.0, 1.0)
# Sizes (relative to the UI font) of the bold fonts that are loaded
HEADLINE = 1.6
HEADLINE_SCALES = (1.0, HEADLINE)
# Bold system fonts look smaller than polyscope's UI font at equal size
BOLD_FONT_SIZE_MATCH = 1.1
# Bold fonts for headlines, first existing one is used (faux bold otherwise)
BOLD_FONT_PATHS = [
    "C:/Windows/Fonts/segoeuib.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
]
NO_PARTITION = "none"


def _short(edge_set_name):
    """Compact column label: 'up 0->1' -> 'up0', 'down 1->0' -> 'dn1'."""
    return edge_set_name.split("->")[0].replace(" ", "").replace("down", "dn")


def _wrapped(text, color):
    """Coloured text wrapped at the panel width."""
    psim.PushStyleColor(psim.ImGuiCol_Text, color)
    psim.TextWrapped(text)
    psim.PopStyleColor()


def _partition_palette(k):
    cmap = matplotlib.colormaps["tab20" if k > 10 else "tab10"]
    return np.array([cmap(i % cmap.N)[:3] for i in range(k)])


def _unit_sphere_mesh(radius, n_lat=48, n_lon=96):
    # Latitude rings without the poles, closed by one vertex fan per pole
    # (rings at the poles give zero-area triangles that render black)
    lat = np.linspace(-np.pi / 2, np.pi / 2, n_lat + 2)[1:-1]
    lon = np.linspace(-np.pi, np.pi, n_lon, endpoint=False)
    lat, lon = np.meshgrid(lat, lon, indexing="ij")
    rings = np.stack(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)],
        axis=-1,
    ).reshape(-1, 3)
    south, north = len(rings), len(rings) + 1
    verts = radius * np.concatenate([rings, [[0, 0, -1], [0, 0, 1]]])
    faces = []
    for j in range(n_lon):
        j_next = (j + 1) % n_lon
        faces.append((south, j_next, j))
        top = (n_lat - 1) * n_lon
        faces.append((north, top + j, top + j_next))
        for i in range(n_lat - 1):
            a, b = i * n_lon + j, i * n_lon + j_next
            c, d = a + n_lon, b + n_lon
            faces += [(a, b, d), (a, d, c)]
    return verts, np.array(faces)


class _DisplayEdges:
    """Edges of one edge set as drawn, with a map back to the directed set.

    Symmetric edge sets (m2m) are drawn once per undirected pair. ``owner``
    maps each directed edge to the display edge that draws it.
    """

    def __init__(self, es: EdgeSet):
        ei = es.edge_index
        if es.src == es.dst:
            lo, hi = np.minimum(ei[0], ei[1]), np.maximum(ei[0], ei[1])
            pairs, self.owner = np.unique(
                np.stack([lo, hi], axis=1), axis=0, return_inverse=True
            )
            self.owner = self.owner.ravel()
            self.edges = pairs.T
        else:
            self.owner = np.arange(ei.shape[1])
            self.edges = ei
        self.n = self.edges.shape[1]

    def reduce(self, directed_values):
        """OR a per-directed-edge boolean onto display edges."""
        out = np.zeros(self.n, dtype=bool)
        np.logical_or.at(out, self.owner, directed_values)
        return out


class GraphViewer:
    def __init__(
        self,
        graph: ViewerGraph,
        level_gap: float = 0.1,
        strategy: str = NO_PARTITION,
        k: int = 4,
        cost_model: Optional[CostModel] = None,
    ):
        self.graph = graph
        self.cost_model = cost_model or CostModel.uniform(graph)
        self.cost_model.check(graph)
        self.node_weights = node_weights(graph, self.cost_model)
        self.radii = [1.0] + [
            1.0 + (layer.level + 1) * level_gap for layer in graph.layers[1:]
        ]
        self.positions = [
            r * layer.xyz for r, layer in zip(self.radii, graph.layers)
        ]
        self.display_edges = [_DisplayEdges(es) for es in graph.edge_sets]

        self.node_scale = 1.0
        self.edge_scale = 1.0
        self.color_mode = "Layer"
        self.strategy = strategy
        self.k = k
        self.only_cut = False
        self.parts: Optional[List[np.ndarray]] = None
        self.stats = None
        self.costs = None
        # Set by load_fonts() once polyscope is initialised
        self.bold_fonts = {}  # scale -> ImFont

        self.flow_downstream = True
        self.flow_hops = 3
        self.flow_seed = None  # (layer index, node index)
        self.flow_dist: Optional[List[np.ndarray]] = None

    # ------------------------------------------------------------------ setup
    def load_fonts(self):
        """Load bold fonts for emphasis and headlines; call after ps.init().

        The polyscope renderer draws a font only at the size it was loaded
        with, so one font is loaded per size in HEADLINE_SCALES.
        """
        base = getattr(psim.GetStyle(), "FontSizeBase", 0.0) or 16.0
        path = next((p for p in BOLD_FONT_PATHS if os.path.isfile(p)), None)
        if path is None:
            return
        try:
            for scale in HEADLINE_SCALES:
                self.bold_fonts[scale] = psim.GetIO().Fonts.AddFontFromFileTTF(
                    path, base * scale * BOLD_FONT_SIZE_MATCH
                )
        except Exception:  # fall back to faux bold
            self.bold_fonts = {}

    def _node_radius(self, layer):
        return self.node_scale * (0.0012 if layer.level < 0 else 0.0025)

    def _edge_radius(self):
        return self.edge_scale * 0.0004

    def _layer_name(self, i):
        return f"{self.graph.layers[i].name} nodes"

    def _edge_name(self, j):
        return f"{self.graph.edge_sets[j].name} edges"

    def register(self):
        verts, faces = _unit_sphere_mesh(0.995)
        ps.register_surface_mesh(
            "Earth", verts, faces, color=(0.08, 0.10, 0.16), smooth_shade=True
        )
        for i, layer in enumerate(self.graph.layers):
            color = (
                GRID_COLOR
                if layer.level < 0
                else MESH_COLORS[layer.level % len(MESH_COLORS)]
            )
            ps.register_point_cloud(
                self._layer_name(i),
                self.positions[i],
                radius=self._node_radius(layer),
                color=color,
            )
        for j in range(len(self.graph.edge_sets)):
            self._register_edges(j)
        self.apply_partition()

    def _register_edges(self, j):
        """(Re)register edge set j, keeping its enabled state."""
        es, de = self.graph.edge_sets[j], self.display_edges[j]
        name = self._edge_name(j)
        enabled = (
            ps.get_curve_network(name).is_enabled()
            if ps.has_curve_network(name)
            else es.kind == "m2m"
        )
        keep = np.ones(de.n, dtype=bool)
        if self.only_cut and self.parts is not None:
            keep = de.reduce(
                cut_mask(es.edge_index, self.parts[es.src], self.parts[es.dst])
            )
        self._kept = getattr(self, "_kept", {})
        self._kept[j] = keep

        src_pos, dst_pos = self.positions[es.src], self.positions[es.dst]
        edges = de.edges[:, keep]
        nodes = np.concatenate([src_pos[edges[0]], dst_pos[edges[1]]])
        n = edges.shape[1]
        segments = np.stack([np.arange(n), np.arange(n) + n], axis=1)
        if n == 0:  # polyscope needs at least one node
            nodes, segments = np.zeros((1, 3)), np.zeros((0, 2), dtype=int)
        ps.register_curve_network(
            name,
            nodes,
            segments,
            enabled=enabled,
            radius=self._edge_radius(),
            color=EDGE_KIND_COLORS[es.kind],
        )

    # ------------------------------------------------------------ partitions
    def apply_partition(self):
        if self.strategy == NO_PARTITION:
            self.parts, self.stats, self.costs = None, None, None
            if self.color_mode == "Partition":
                self.color_mode = "Layer"
        else:
            self.parts = partition_layers(
                [layer.xyz for layer in self.graph.layers],
                self.strategy,
                self.k,
                layer_weights=self.node_weights,
            )
            self.stats = [
                cut_stats(es.edge_index, self.parts[es.src], self.parts[es.dst])
                for es in self.graph.edge_sets
            ]
            self.costs = partition_costs(
                self.graph, self.parts, self.k, self.cost_model
            )
            self.color_mode = "Partition"
        for j in range(len(self.graph.edge_sets)):
            self._register_edges(j)
        self.refresh_colors()

    # ------------------------------------------------------------------ flow
    def compute_flow(self):
        """Hop distance from the seed over the edge sets currently shown."""
        if self.flow_seed is None:
            self.flow_dist = None
            return
        dist = [np.full(len(layer), -1) for layer in self.graph.layers]
        seed_layer, seed_node = self.flow_seed
        dist[seed_layer][seed_node] = 0
        active = [
            es
            for j, es in enumerate(self.graph.edge_sets)
            if ps.get_curve_network(self._edge_name(j)).is_enabled()
        ]
        for hop in range(1, self.flow_hops + 1):
            new = [d.copy() for d in dist]
            for es in active:
                a, b = (
                    (es.src, es.dst)
                    if self.flow_downstream
                    else (
                        es.dst,
                        es.src,
                    )
                )
                ea, eb = (
                    (es.edge_index[0], es.edge_index[1])
                    if self.flow_downstream
                    else (es.edge_index[1], es.edge_index[0])
                )
                frontier = dist[a][ea] == hop - 1
                targets = eb[frontier]
                targets = targets[new[b][targets] < 0]
                new[b][targets] = hop
            dist = new
        self.flow_dist = dist

    def _flow_edge_used(self, es):
        d = self.flow_dist
        if self.flow_downstream:
            ds, dd = d[es.src][es.edge_index[0]], d[es.dst][es.edge_index[1]]
        else:
            ds, dd = d[es.dst][es.edge_index[1]], d[es.src][es.edge_index[0]]
        return (ds >= 0) & (dd == ds + 1)

    # ---------------------------------------------------------------- colors
    def _hop_colors(self, dist):
        cmap = matplotlib.colormaps["plasma"]
        colors = np.tile(DIM_COLOR, (len(dist), 1))
        reached = dist >= 0
        colors[reached] = cmap(dist[reached] / max(self.flow_hops, 1))[:, :3]
        return colors

    def refresh_colors(self):
        palette = None if self.parts is None else _partition_palette(self.k)
        for i in range(len(self.graph.layers)):
            pc = ps.get_point_cloud(self._layer_name(i))
            if self.color_mode == "Partition" and self.parts is not None:
                pc.add_color_quantity(
                    "color", palette[self.parts[i]], enabled=True
                )
            elif self.color_mode == "Flow" and self.flow_dist is not None:
                pc.add_color_quantity(
                    "color", self._hop_colors(self.flow_dist[i]), enabled=True
                )
            else:
                pc.remove_all_quantities()

        for j, es in enumerate(self.graph.edge_sets):
            cn = ps.get_curve_network(self._edge_name(j))
            de, keep = self.display_edges[j], self._kept[j]
            if self.color_mode == "Partition" and self.parts is not None:
                cut = de.reduce(
                    cut_mask(
                        es.edge_index, self.parts[es.src], self.parts[es.dst]
                    )
                )[keep]
                dst_part = self.parts[es.dst][de.edges[1, keep]]
                colors = 0.55 * palette[dst_part] + 0.45 * DIM_COLOR
                colors[cut] = CUT_COLOR
            elif self.color_mode == "Flow" and self.flow_dist is not None:
                used = de.reduce(self._flow_edge_used(es))[keep]
                colors = np.tile(DIM_COLOR * 0.6, (keep.sum(), 1))
                colors[used] = np.array([1.0, 0.9, 0.2])
            else:
                cn.remove_all_quantities()
                continue
            if len(colors):
                cn.add_color_quantity(
                    "color", colors, defined_on="edges", enabled=True
                )

    # -------------------------------------------------------------------- UI
    def _handle_pick(self):
        if not psim.GetIO().MouseClicked[0] or psim.GetIO().WantCaptureMouse:
            return
        if not ps.have_selection():
            return
        sel = ps.get_selection()
        for i in range(len(self.graph.layers)):
            if sel.structure_name == self._layer_name(i):
                self.flow_seed = (i, int(sel.local_index))
                self.compute_flow()
                self.color_mode = "Flow"
                self.refresh_colors()
                return

    def _bold(self, text, color=TEXT_COLOR, scale=1.0):
        """Bold text; ``scale`` is one of HEADLINE_SCALES.

        Without a bold font file, falls back to the regular font drawn twice,
        one pixel apart (no enlargement).
        """
        font = self.bold_fonts.get(scale)
        if font is not None:
            psim.PushFont(font, 0.0)
            psim.TextColored(color, text)
            psim.PopFont()
            return
        x, y = psim.GetCursorScreenPos()
        psim.TextColored(color, text)
        end = psim.GetCursorScreenPos()
        psim.SetCursorScreenPos((x + 1, y))
        psim.TextColored(color, text)
        psim.SetCursorScreenPos(end)

    def _table(self, table_id, headers, rows, row_colors=None, highlight=None):
        flags = psim.ImGuiTableFlags_Borders | psim.ImGuiTableFlags_RowBg
        if not psim.BeginTable(table_id, len(headers), flags):
            return
        for h in headers:
            psim.TableSetupColumn(h)
        psim.TableHeadersRow()
        for r, cells in enumerate(rows):
            psim.TableNextRow()
            if r == highlight:
                psim.TableSetBgColor(
                    psim.ImGuiTableBgTarget_RowBg1,
                    psim.GetColorU32((*row_colors[r], 0.30)),
                )
            for c, text in enumerate(cells):
                psim.TableSetColumnIndex(c)
                color = (
                    (*row_colors[r], 1.0)
                    if c == 0 and row_colors is not None
                    else TEXT_COLOR
                )
                if r == highlight:
                    self._bold(text, color)
                else:
                    psim.TextColored(color, text)
        psim.EndTable()

    def _cost_model_line(self):
        model = self.cost_model
        text = f"Cost model: {model.name}"
        if model.measured:
            psim.TextUnformatted(text)
        else:
            _wrapped(
                text + " - rankings may differ from the real model",
                (1.0, 0.75, 0.25, 1.0),
            )
        tooltip = "\n".join(
            [f"{key}: {value}" for key, value in model.source.items()]
            + [f"note: {note}" for note in model.notes]
        )
        if tooltip:
            psim.SetItemTooltip(tooltip)

    def _cost_panel(self):
        psim.SeparatorText("Communication")
        self._communication_section()
        psim.SeparatorText("Memory")
        self._memory_section()
        if psim.TreeNode("Details per edge set"):
            self._stats_table()
            psim.TreePop()

    def _communication_section(self):
        costs, k = self.costs, self.k
        comm = costs.communication
        busiest = int(np.argmax(comm))
        palette = _partition_palette(k)
        color = (*palette[busiest], 1.0)

        # Headline: the GPU that receives most, and how much
        self._bold(f"Busiest GPU {busiest}", color, scale=HEADLINE)
        self._bold(
            f"{comm[busiest]:,.0f} node states received per step",
            scale=HEADLINE,
        )
        psim.TextUnformatted(
            f"{comm[busiest] / comm.mean():.2f}x the average GPU   |   "
            f"total {comm.sum():,.0f}   |   up to {costs.neighbours.max()} "
            "neighbours"
        )

        # Where the busiest GPU's traffic comes from
        psim.Spacing()
        psim.TextUnformatted(f"Where GPU {busiest}'s traffic comes from:")
        parts = sorted(
            (
                (costs.halo_terms[es.name][busiest], es.name)
                for es in self.graph.edge_sets
            ),
            reverse=True,
        )
        psim.PushStyleColor(psim.ImGuiCol_PlotHistogram, color)
        for value, name in parts:
            fraction = value / comm[busiest] if comm[busiest] else 0.0
            uses = self.cost_model.uses[name]
            psim.ProgressBar(
                fraction,
                (-1, 0),
                f"{name}: {value:,.0f}  ({100 * fraction:.0f}%)  = "
                f"{value / uses:,.0f} halo x {uses:g} uses",
            )
        psim.PopStyleColor()

        # All GPUs
        psim.Spacing()
        psim.TextUnformatted("All GPUs (node states received per step):")
        edge_sets = self.graph.edge_sets
        rows = [
            [f"{p}", f"{comm[p]:,.0f}"]
            + [f"{costs.halo_terms[es.name][p]:,.0f}" for es in edge_sets]
            + [f"{costs.neighbours[p]}"]
            for p in range(k)
        ]
        self._table(
            "per_gpu_comm",
            ["GPU", "Total"] + [_short(es.name) for es in edge_sets] + ["Nbrs"],
            rows,
            row_colors=palette,
            highlight=busiest,
        )
        if psim.TreeNode("What do these numbers mean?"):
            psim.TextWrapped(
                "C(p) = sum over edge sets s of u(s) x H(p,s): the node "
                "states GPU p receives from other GPUs per forecast step. "
                "H(p,s) = distinct nodes on other GPUs with an edge of s into "
                "GPU p; u(s) = how often s is used per step. Multiply by "
                "hidden_dim x bytes for bytes. Nbrs = GPUs that send to this "
                "one (one exchange each)."
            )
            psim.TreePop()

    def _memory_model_table(self):
        """Values stored per node and per edge: what balancing is based on."""
        if not psim.TreeNode("Memory model (what the borders balance)"):
            return
        model = self.cost_model
        psim.TextWrapped(
            "Values stored per item, in units of hidden_dim, during one "
            "checkpointed forecast step. Each node's load is its own value "
            "plus that of the edges it receives."
        )
        items = [
            (layer.name, "nodes", len(layer), model.node[layer.name])
            for layer in self.graph.layers
        ] + [
            (es.name, "edges", es.n_edges, model.edge[es.name])
            for es in self.graph.edge_sets
        ]
        total = sum(count * value for _, _, count, value in items)
        rows = [
            [
                name,
                kind,
                f"{count:,}",
                f"{value:.1f}",
                f"{100 * count * value / total:.1f}",
            ]
            for name, kind, count, value in sorted(
                items, key=lambda item: -item[2] * item[3]
            )
        ]
        self._table(
            "memory_model",
            ["Item", "Kind", "Count", "Per item", "Share %"],
            rows,
        )
        for note in model.notes:
            _wrapped(
                f"Note: {note}",
                psim.GetStyle().Colors[psim.ImGuiCol_TextDisabled],
            )
        psim.TreePop()

    def _memory_section(self):
        costs, k = self.costs, self.k
        balanced = costs.imbalance <= MAX_BALANCED_IMBALANCE
        if balanced:
            psim.TextColored(
                (0.45, 0.85, 0.45, 1.0),
                f"Balanced: busiest GPU holds {costs.imbalance:.2f}x the "
                "mean memory",
            )
        else:
            self._bold(
                f"NOT balanced: busiest GPU holds {costs.imbalance:.2f}x the "
                "mean memory",
                (1.0, 0.4, 0.3, 1.0),
            )
        self._memory_model_table()
        if not psim.TreeNode("Memory per GPU"):
            return
        memory = costs.memory
        busiest = costs.busiest
        psim.TextWrapped(
            "Memory M(p): stored values of one checkpointed forecast step, "
            f"relative to one GPU. Busiest GPU {busiest}: "
            f"{100 * costs.max_share:.1f}% (ideal {100 / k:.1f}%)."
        )

        def share(values, decimals=1):
            return f"{100 * values / costs.single:.{decimals}f}"

        nodes = sum(costs.node_terms.values())
        edges = sum(costs.edge_terms.values())
        halo = sum(costs.halo_terms.values())
        grid_side = sum(
            costs.edge_terms[es.name]
            for es in self.graph.edge_sets
            if es.kind in ("g2m", "m2g")
        )
        rows = [
            [
                f"{p}",
                share(memory[p]),
                share(nodes[p]),
                share(edges[p]),
                share(grid_side[p]),
                share(halo[p], decimals=2),
            ]
            for p in range(k)
        ]
        psim.TextWrapped(
            "Per GPU (busiest highlighted), % of one-GPU memory: M = Nodes + "
            "Edges + "
            "Halo. g2m+m2g = grid-side edges (part of Edges). Halo = stored "
            "copies of remote senders."
        )
        self._table(
            "per_gpu_memory",
            ["GPU", "M", "Nodes", "Edges", "g2m+m2g", "Halo"],
            rows,
            row_colors=_partition_palette(k),
            highlight=busiest,
        )
        psim.TreePop()

    def _stats_table(self):
        psim.TextUnformatted(
            "Halo = distinct (sender, receiving partition) pairs over cut "
            "edges,\ni.e. node states sent per use of the edge set."
        )
        flags = psim.ImGuiTableFlags_Borders | psim.ImGuiTableFlags_RowBg
        if psim.BeginTable("cut_stats", 5, flags):
            for h in ["Edge set", "Edges", "Cut", "Cut %", "Halo"]:
                psim.TableSetupColumn(h)
            psim.TableHeadersRow()
            for es, st in zip(self.graph.edge_sets, self.stats):
                psim.TableNextRow()
                cells = [
                    es.name,
                    f"{st.n_edges}",
                    f"{st.n_cut}",
                    f"{100 * st.cut_fraction:.1f}",
                    f"{st.n_halo}",
                ]
                for c, text in enumerate(cells):
                    psim.TableSetColumnIndex(c)
                    psim.TextUnformatted(text)
            psim.EndTable()

        psim.TextUnformatted("Nodes per partition (min / max / max-to-mean):")
        for layer, parts in zip(self.graph.layers, self.parts):
            counts = np.bincount(parts, minlength=self.k)
            psim.TextUnformatted(
                f"  {layer.name:7s} {counts.min():6d} / {counts.max():6d} / "
                f"{counts.max() / counts.mean():.2f}"
            )

    def callback(self):
        self._handle_pick()
        g = self.graph
        psim.TextUnformatted(
            f"{g.name}: {len(g.layers[0])} grid nodes, {len(g.layers) - 1} "
            "mesh levels"
        )
        psim.Separator()

        psim.SetNextItemOpen(True, psim.ImGuiCond_FirstUseEver)
        if psim.TreeNode("Visibility"):
            flow_inputs_changed = False
            for i, layer in enumerate(g.layers):
                pc = ps.get_point_cloud(self._layer_name(i))
                changed, on = psim.Checkbox(
                    f"{self._layer_name(i)} ({len(layer)})", pc.is_enabled()
                )
                if changed:
                    pc.set_enabled(on)
            for j, es in enumerate(g.edge_sets):
                cn = ps.get_curve_network(self._edge_name(j))
                changed, on = psim.Checkbox(
                    f"{self._edge_name(j)} ({es.n_edges})", cn.is_enabled()
                )
                if changed:
                    cn.set_enabled(on)
                    flow_inputs_changed = True
            earth = ps.get_surface_mesh("Earth")
            changed, on = psim.Checkbox("Earth sphere", earth.is_enabled())
            if changed:
                earth.set_enabled(on)
            changed, self.node_scale = psim.SliderFloat(
                "Node size", self.node_scale, 0.1, 5.0
            )
            if changed:
                for i, layer in enumerate(g.layers):
                    ps.get_point_cloud(self._layer_name(i)).set_radius(
                        self._node_radius(layer)
                    )
            changed, self.edge_scale = psim.SliderFloat(
                "Edge width", self.edge_scale, 0.1, 5.0
            )
            if changed:
                for j in range(len(g.edge_sets)):
                    ps.get_curve_network(self._edge_name(j)).set_radius(
                        self._edge_radius()
                    )
            if flow_inputs_changed and self.flow_seed is not None:
                # The flow follows the edge sets shown
                self.compute_flow()
                if self.color_mode == "Flow":
                    self.refresh_colors()
            psim.TreePop()

        psim.TextUnformatted("Color by")
        for mode in COLOR_MODES:
            psim.SameLine()
            if psim.RadioButton(mode, self.color_mode == mode):
                self.color_mode = mode
                self.refresh_colors()

        psim.SetNextItemOpen(True, psim.ImGuiCond_FirstUseEver)
        if psim.TreeNode("Partitioning"):
            self._cost_model_line()
            options = [NO_PARTITION] + list(STRATEGIES)
            changed, idx = psim.Combo(
                "Strategy", options.index(self.strategy), options
            )
            changed_k, k = psim.SliderInt("Partitions", self.k, 2, 32)
            if changed or changed_k:
                self.strategy, self.k = options[idx], k
                self.apply_partition()
            psim.TextDisabled(
                "Borders balance memory per node (see Memory below)"
            )
            changed, self.only_cut = psim.Checkbox(
                "Only cut edges", self.only_cut
            )
            if changed:
                for j in range(len(g.edge_sets)):
                    self._register_edges(j)
                self.refresh_colors()
            if self.costs is not None:
                self._cost_panel()
            psim.TreePop()

        psim.SetNextItemOpen(True, psim.ImGuiCond_FirstUseEver)
        if psim.TreeNode("Information flow"):
            psim.TextUnformatted(
                "Click a node. Follows the edge sets currently shown;\n"
                "toggle edge sets under Visibility to choose the path."
            )
            changed_d, self.flow_downstream = psim.Checkbox(
                "Downstream (off: upstream)", self.flow_downstream
            )
            changed_h, self.flow_hops = psim.SliderInt(
                "Hops", self.flow_hops, 1, 12
            )
            if psim.Button("Recompute") or changed_d or changed_h:
                self.compute_flow()
                if self.color_mode == "Flow":
                    self.refresh_colors()
            if self.flow_seed is not None:
                layer, node = self.flow_seed
                lon, lat = g.layers[layer].lon_lat[node]
                psim.TextUnformatted(
                    f"Seed: {g.layers[layer].name} node {node} "
                    f"({lon:.1f}, {lat:.1f})"
                )
                if self.flow_dist is not None:
                    for lyr, d in zip(g.layers, self.flow_dist):
                        psim.TextUnformatted(
                            f"  {lyr.name:7s} reached {(d >= 0).sum()}"
                        )
            psim.TreePop()


def run(
    graph: ViewerGraph,
    level_gap: float = 0.1,
    strategy: str = NO_PARTITION,
    k: int = 4,
    screenshot: Optional[str] = None,
    enable: Optional[Dict[str, bool]] = None,
    cost_model: Optional[CostModel] = None,
):
    """Open the viewer, or render one frame to ``screenshot`` and exit."""
    ps.set_program_name(f"Graph viewer: {graph.name}")
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("none")
    ps.set_background_color((0.02, 0.02, 0.03))
    if screenshot:
        ps.set_window_size(2400, 1350)
    ps.init()
    # The panel above replaces the verbose default structure list
    ps.set_build_default_gui_panels(False)
    viewer = GraphViewer(graph, level_gap, strategy, k, cost_model)
    viewer.load_fonts()
    viewer.register()
    for name, on in (enable or {}).items():
        if ps.has_curve_network(name):
            ps.get_curve_network(name).set_enabled(on)
        elif ps.has_point_cloud(name):
            ps.get_point_cloud(name).set_enabled(on)
        else:
            raise ValueError(f"No layer or edge set called {name!r}")
    ps.set_user_callback(viewer.callback)
    if screenshot:
        ps.look_at((2.2, -2.2, 1.6), (0.0, 0.0, 0.0))
        for _ in range(3):
            ps.frame_tick()
        ps.screenshot(screenshot, transparent_bg=False, include_UI=True)
    else:
        ps.show()
