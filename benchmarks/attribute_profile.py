"""Attribute GPU time in a GraphEFM profiler trace to model components.

The ``graph_efm/...`` profiler regions only enclose forward computation. The
backward pass runs inside PyTorch's autograd engine, outside those regions.
This script reads the Chrome trace written by ``--benchmark_profiler_dir`` and
assigns every GPU kernel to a (component, phase) pair:

- forward:   launched inside a ``graph_efm/...`` region in the forward pass.
- recompute: launched inside a ``graph_efm/...`` region during the backward
             pass. With gradient checkpointing, each unrolled step's forward
             pass is re-run there to regenerate saved activations.
- backward:  launched by an autograd backward function. The function is
             linked to the forward operator that created it, and inherits
             that operator's region.

Forward-to-backward links come from the trace's ``fwdbwd`` flow events where
present, and otherwise from the autograd sequence number and forward thread
id that both forward operators and backward functions carry.

Usage:
    python benchmarks/attribute_profile.py TRACE.json [--output SUMMARY.json]
"""

# Standard library
import argparse
import bisect
import collections
import gzip
import json
import sys

REGION_PREFIX = "graph_efm/"
BACKWARD_PREFIX = "autograd::engine::evaluate_function"
GPU_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset"}
HOST_CATEGORIES = {"cpu_op", "user_annotation", "cuda_runtime", "cuda_driver"}
UNLABELLED = "(unlabelled)"


def load_trace(path):
    """Load a Chrome trace, optionally gzip-compressed."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as trace_file:
        trace = json.load(trace_file)
    return trace["traceEvents"] if isinstance(trace, dict) else trace


class HostEvent:
    """A host-side duration event with its position in the call nesting."""

    __slots__ = ("event", "start", "end", "parent")

    def __init__(self, event):
        self.event = event
        self.start = float(event["ts"])
        self.end = self.start + float(event.get("dur", 0))
        self.parent = None

    @property
    def name(self):
        return self.event.get("name", "")

    @property
    def args(self):
        return self.event.get("args", {})

    def ancestors(self):
        node = self.parent
        while node is not None:
            yield node
            node = node.parent


def build_host_nesting(events):
    """Link every host event to its innermost enclosing event per thread."""
    by_thread = collections.defaultdict(list)
    for event in events:
        if event.get("ph") == "X" and event.get("cat") in HOST_CATEGORIES:
            by_thread[(event.get("pid"), event.get("tid"))].append(
                HostEvent(event)
            )

    for thread_events in by_thread.values():
        # Outer events first when two start at the same time.
        thread_events.sort(key=lambda node: (node.start, -node.end))
        stack = []
        for node in thread_events:
            while stack and stack[-1].end < node.end:
                stack.pop()
            node.parent = stack[-1] if stack else None
            stack.append(node)
    return by_thread


def region_path(node):
    """Return the nested graph_efm regions enclosing a node, outermost first.

    For example ``crps_prior_rollouts > prior_encoder``.
    """
    names = [
        ancestor.name[len(REGION_PREFIX) :]
        for ancestor in ([node] + list(node.ancestors()))
        if ancestor.event.get("cat") == "user_annotation"
        and ancestor.name.startswith(REGION_PREFIX)
    ]
    return " > ".join(reversed(names)) if names else None


def backward_function(node):
    """Return the enclosing autograd backward function of a node, if any."""
    for candidate in [node] + list(node.ancestors()):
        if candidate.name.startswith(BACKWARD_PREFIX):
            return candidate
    return None


def find_host_event(by_thread, thread_starts, pid, tid, ts):
    """Return the innermost host event on a thread that contains ``ts``."""
    thread_events = by_thread.get((pid, tid))
    if not thread_events:
        return None
    index = bisect.bisect_right(thread_starts[(pid, tid)], float(ts)) - 1
    while index >= 0:
        node = thread_events[index]
        if node.start <= float(ts) <= node.end:
            return node
        index -= 1
    return None


def with_sequence_number(node):
    """Walk outwards to the operator that carries the autograd link.

    A flow starts at the operator that created the autograd node, but child
    operators can start at the same timestamp and be found first.
    """
    for candidate in [node] + list(node.ancestors()):
        if "Sequence number" in candidate.args:
            return candidate
    return node


def forward_links_from_flows(events, by_thread):
    """Map each backward function to its forward operator via fwdbwd flows."""
    thread_starts = {
        thread: [node.start for node in thread_events]
        for thread, thread_events in by_thread.items()
    }
    starts, finishes = {}, {}
    for event in events:
        if event.get("cat") != "fwdbwd":
            continue
        if event.get("ph") == "s":
            starts[event["id"]] = event
        elif event.get("ph") == "f":
            finishes[event["id"]] = event

    links = {}
    for flow_id, finish in finishes.items():
        start = starts.get(flow_id)
        if start is None:
            continue
        forward_op = find_host_event(
            by_thread,
            thread_starts,
            start.get("pid"),
            start.get("tid"),
            start["ts"],
        )
        backward_op = find_host_event(
            by_thread,
            thread_starts,
            finish.get("pid"),
            finish.get("tid"),
            finish["ts"],
        )
        if forward_op is None or backward_op is None:
            continue
        backward_fn = backward_function(backward_op)
        if backward_fn is not None:
            links[id(backward_fn)] = with_sequence_number(forward_op)
    return links


def forward_links_from_sequence_numbers(by_thread):
    """Map backward functions to forward ops by (sequence number, thread).

    Both forward operators and backward functions carry the autograd
    sequence number and the forward thread id. Several forward operators can
    share a number (e.g. ``aten::t`` inside ``aten::linear``); the counter
    advances when an operator creates an autograd node, so the creating
    operator is the last one recorded with that number.
    """
    creator_by_key = {}
    backward_functions = []
    for thread_events in by_thread.values():
        for node in thread_events:
            sequence = node.args.get("Sequence number")
            thread = node.args.get("Fwd thread id")
            if sequence is None or thread is None:
                continue
            key = (sequence, thread)
            if node.name.startswith(BACKWARD_PREFIX):
                backward_functions.append((key, node))
            elif backward_function(node) is None:
                # Operators inside the backward pass (e.g. ``AddmmBackward0``
                # or checkpoint recomputation) repeat the sequence number but
                # did not create the node, so only forward ones count.
                current = creator_by_key.get(key)
                if current is None or node.start >= current.start:
                    creator_by_key[key] = node

    links = {}
    for key, node in backward_functions:
        if key in creator_by_key:
            links[id(node)] = creator_by_key[key]
    return links


CRPS_REGION = "crps_prior_rollouts"
POSTERIOR_REGIONS = ("posterior_input_embedding", "posterior_encoder")


def recompute_contexts(by_thread):
    """Decide which recomputations belong to the CRPS prior rollouts.

    Checkpointing re-runs one unrolled step's forward function inside the
    backward function that first needs its activations. The outer
    crps_prior_rollouts region only exists in the forward pass, so it does
    not enclose that recomputation. GraphEFM-specific rule: a main training
    step always runs the posterior encoder and a CRPS rollout step never
    does, so a recomputation without posterior regions is a CRPS rollout.
    """
    has_crps = False
    regions_by_backward_fn = collections.defaultdict(set)
    for thread_events in by_thread.values():
        for node in thread_events:
            if node.event.get("cat") != "user_annotation":
                continue
            if not node.name.startswith(REGION_PREFIX):
                continue
            region = node.name[len(REGION_PREFIX) :]
            backward_fn = backward_function(node)
            if backward_fn is None:
                has_crps = has_crps or region == CRPS_REGION
            else:
                regions_by_backward_fn[id(backward_fn)].add(region)

    return {
        backward_fn_id: (
            CRPS_REGION
            if has_crps and not regions & set(POSTERIOR_REGIONS)
            else None
        )
        for backward_fn_id, regions in regions_by_backward_fn.items()
    }


def classify_launch(node, forward_links, contexts):
    """Return (component, phase) for the host event that launched a kernel."""
    backward_fn = backward_function(node)
    own_region = region_path(node)

    if backward_fn is None:
        return own_region or UNLABELLED, "forward"

    # A graph_efm region inside the backward pass means checkpointing is
    # re-running the forward computation.
    if own_region is not None:
        context = contexts.get(id(backward_fn))
        if context is not None:
            own_region = f"{context} > {own_region}"
        return own_region, "recompute"

    if "AccumulateGrad" in backward_fn.name:
        return "(gradient accumulation)", "backward"

    forward_op = forward_links.get(id(backward_fn))
    if forward_op is None:
        return UNLABELLED, "backward"
    return region_path(forward_op) or UNLABELLED, "backward"


def complete_step_windows(by_thread, forward_links, min_linked_fraction=0.9):
    """Return (start, end) host-time windows of complete training steps.

    Lightning advances the profiler schedule at the end of training_step,
    i.e. between a step's forward and backward pass. A trace therefore starts
    with a backward pass whose forward pass is (mostly) missing and ends with
    a forward pass whose backward pass is missing. Backward functions are
    grouped into backward passes, split wherever forward operators run in
    between. A pass is complete when nearly all of its backward functions
    link to forward operators in the trace; its window runs from the first
    of those forward operators to the end of the pass.
    """
    forward_starts = sorted(
        node.start
        for thread_events in by_thread.values()
        for node in thread_events
        if "Sequence number" in node.args
        and not node.name.startswith(BACKWARD_PREFIX)
        and backward_function(node) is None
    )
    backward_functions = sorted(
        (
            node
            for thread_events in by_thread.values()
            for node in thread_events
            if node.name.startswith(BACKWARD_PREFIX)
        ),
        key=lambda node: node.start,
    )

    passes = []
    for node in backward_functions:
        if passes:
            previous_end = passes[-1][-1].end
            first_forward_after = bisect.bisect_right(
                forward_starts, previous_end
            )
            if (
                first_forward_after < len(forward_starts)
                and forward_starts[first_forward_after] < node.start
            ):
                passes.append([])
        else:
            passes.append([])
        passes[-1].append(node)

    windows = []
    for backward_pass in passes:
        linkable = [
            node for node in backward_pass if "AccumulateGrad" not in node.name
        ]
        linked = [node for node in linkable if id(node) in forward_links]
        if not linkable or len(linked) < min_linked_fraction * len(linkable):
            continue
        windows.append(
            (
                min(forward_links[id(node)].start for node in linked),
                max(node.end for node in backward_pass),
            )
        )
    return windows


def attribute(events):
    """Sum GPU time (in microseconds) per (component, phase)."""
    by_thread = build_host_nesting(events)

    # Explicit flow links are preferred; sequence numbers fill the rest.
    flow_links = forward_links_from_flows(events, by_thread)
    forward_links = forward_links_from_sequence_numbers(by_thread)
    forward_links.update(flow_links)
    link_source = (
        f"{len(flow_links)} fwdbwd flows + "
        f"{len(forward_links) - len(flow_links)} sequence numbers"
    )

    launches_by_correlation = {}
    for thread_events in by_thread.values():
        for node in thread_events:
            if node.event.get("cat") in ("cuda_runtime", "cuda_driver"):
                correlation = node.args.get("correlation")
                if correlation is not None:
                    launches_by_correlation[correlation] = node

    windows = complete_step_windows(by_thread, forward_links)
    contexts = recompute_contexts(by_thread)

    totals = collections.defaultdict(float)
    counts = collections.Counter()
    unmatched_gpu_us = 0.0
    outside_window_gpu_us = 0.0
    for event in events:
        if event.get("ph") != "X" or event.get("cat") not in GPU_CATEGORIES:
            continue
        duration = float(event.get("dur", 0))
        launch = launches_by_correlation.get(
            event.get("args", {}).get("correlation")
        )
        if launch is None:
            unmatched_gpu_us += duration
            continue
        if not any(start <= launch.start <= end for start, end in windows):
            outside_window_gpu_us += duration
            continue
        key = classify_launch(launch, forward_links, contexts)
        totals[key] += duration
        counts[key] += 1

    # Step markers appear on both the CPU and GPU timelines; count CPU only.
    profiler_steps = sum(
        1
        for event in events
        if event.get("ph") == "X"
        and event.get("cat") == "user_annotation"
        and str(event.get("name", "")).startswith("ProfilerStep#")
    )
    return {
        "link_source": link_source,
        "linked_backward_functions": len(forward_links),
        "profiler_steps": profiler_steps,
        "complete_steps": len(windows),
        "outside_window_gpu_us": outside_window_gpu_us,
        "unmatched_gpu_us": unmatched_gpu_us,
        "rows": [
            {
                "component": component,
                "phase": phase,
                "gpu_us": totals[(component, phase)],
                "kernels": counts[(component, phase)],
            }
            for component, phase in sorted(totals)
        ],
    }


def print_table(summary):
    """Print per-step GPU milliseconds per component and phase."""
    steps = summary["complete_steps"]
    by_component = collections.defaultdict(dict)
    for row in summary["rows"]:
        by_component[row["component"]][row["phase"]] = row["gpu_us"]

    phases = ["forward", "recompute", "backward"]
    total_us = sum(row["gpu_us"] for row in summary["rows"])
    print(
        f"GPU time per complete training step (ms), averaged over {steps} "
        f"step(s) from {summary['profiler_steps']} profiler steps; "
        f"links via {summary['link_source']}"
    )
    header = f"{'component':45s}" + "".join(f"{p:>11s}" for p in phases)
    print(header + f"{'total':>11s}{'share':>8s}")
    for component in sorted(
        by_component, key=lambda c: -sum(by_component[c].values())
    ):
        values = [by_component[component].get(p, 0.0) for p in phases]
        row_total = sum(values)
        print(
            f"{component:45s}"
            + "".join(f"{v / 1000 / steps:11.2f}" for v in values)
            + f"{row_total / 1000 / steps:11.2f}"
            + f"{100 * row_total / max(total_us, 1):7.1f}%"
        )
    print(
        f"Total: {total_us / 1000 / steps:.2f} ms/step. Excluded: "
        f"{summary['outside_window_gpu_us'] / 1000:.2f} ms of partial steps at "
        f"the trace edges, {summary['unmatched_gpu_us'] / 1000:.2f} ms of GPU "
        f"work not linked to a launch."
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("trace", help="Chrome trace JSON (.json or .json.gz)")
    parser.add_argument(
        "--output", help="Optional path for a machine-readable JSON summary"
    )
    args = parser.parse_args(argv)

    summary = attribute(load_trace(args.trace))
    print_table(summary)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as output_file:
            json.dump(summary, output_file, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
