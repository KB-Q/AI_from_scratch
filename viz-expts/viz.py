"""
Unified architecture-diagram generator for the from-scratch models in this repo.

Uses torchview, which traces a forward pass and draws the graph at the
nn.Module level (not autograd tensor-ops like torchviz, and with real depth
control unlike torchvista). Knobs that keep diagrams legible:

    depth      how far to expand nested modules (1 = top blocks; higher = drill
               into Attention / SwiGLU inside a Block)
    collapse   fold a run of *identical consecutive* blocks (e.g. 6x Block) into
               a single node labelled "Block x6" (see below)

On collapsing repeated blocks
-----------------------------
torchview's own `roll` only merges a module instance that is literally reused in
a loop (an RNN cell); a `ModuleList` of N distinct block objects - what every
model here uses - is never folded. So we do it ourselves: on a *copy* of the
model, each `ModuleList`/`Sequential` is scanned for maximal runs of consecutive
children that match on all three of

    (1) module class,
    (2) parameter shapes (internals), and
    (3) traced input and output shapes,

and each run is replaced by one representative tagged "<Class> xN". The triple
match is the "exact same block" rule: any difference in structure OR in the
tensor shapes flowing through splits the run and the blocks are drawn separately.

Generic core:  render(module, inputs, name, ...) works for ANY PyTorch module.
Convenience:   a small REGISTRY wires up the models already in this repo so you
               can do `python3 viz.py gpt` etc.

Add a future model in one of two ways:
    1. call render(your_module, example_inputs, "my_model") directly, or
    2. add a builder to REGISTRY returning (module, example_inputs, depth).

Requirements: torchview, graphviz (pip) and the graphviz `dot` binary.
    pip install torchview graphviz   # `brew install graphviz` for the binary

Usage:
    python3 viz.py gpt bert t5 qwen-backbone   # specific diagrams
    python3 viz.py all                         # everything in the registry
    python3 viz.py gpt --depth 3 --format png  # override depth / format
    python3 viz.py gpt --no-collapse           # show every repeated block
    python3 viz.py gpt --hide-all-ops          # drop all op nodes (add, mul, ...)
    python3 viz.py gpt --show-ops              # keep dunder ops too (__rpow__ ...)
    python3 viz.py all --out /tmp/diagrams     # output dir (default: the owning module's images/)

By default only dunder-operator op nodes (__rpow__, __rdiv__, ...) are hidden;
readable ops (add, mul, triu, cos, ...) are kept.
"""
import argparse
import copy
import importlib.util
import inspect
import re
import sys
import warnings
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent
TF_IMAGES = REPO / "08-transformers" / "images"
AUDIO_IMAGES = REPO / "23-llm-audio" / "images"
TF_TORCH = REPO / "08-transformers" / "scripts" / "torch"
AUDIO_TORCH = REPO / "23-llm-audio" / "scripts"


def _import_from_path(path, module_name):
    """Import a module from an arbitrary file path (handles hyphenated names)."""
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# Collapsing runs of identical consecutive blocks into one "<Class> xN" node.
# --------------------------------------------------------------------------
def _shape_of(x):
    """Nested shape signature of a tensor / tuple-of-tensors (or None)."""
    if isinstance(x, torch.Tensor):
        return tuple(x.shape)
    if isinstance(x, (list, tuple)):
        return tuple(_shape_of(y) for y in x)
    return None


def _capture_io_shapes(module, inputs):
    """Run one forward pass, recording (input_shape, output_shape) per child of
    every ModuleList/Sequential, keyed by id(child)."""
    shapes = {}
    handles = []

    def make_hook(child):
        def hook(_m, inp, out):
            shapes[id(child)] = (_shape_of(inp[0]) if inp else None, _shape_of(out))
        return hook

    for container in module.modules():
        if isinstance(container, (nn.ModuleList, nn.Sequential)):
            for child in container:
                handles.append(child.register_forward_hook(make_hook(child)))

    module.eval()
    with torch.no_grad():
        module(*inputs) if isinstance(inputs, tuple) else module(inputs)
    for h in handles:
        h.remove()
    return shapes


def _block_signature(child, io_shapes):
    """(class, parameter-shape signature, traced I/O shapes) for a child."""
    params = tuple(sorted((n, tuple(p.shape)) for n, p in child.named_parameters()))
    return (type(child).__name__, params, io_shapes.get(id(child)))


def collapse_repeated_blocks(module, inputs):
    """Return (collapsed_copy, collapses) where runs of identical consecutive
    blocks are replaced by one representative tagged '<Class> xN'.
    `collapses` is a list of (class_name, count) for reporting."""
    collapsed = copy.deepcopy(module)
    io_shapes = _capture_io_shapes(collapsed, inputs)
    collapses = []

    containers = [m for m in collapsed.modules() if isinstance(m, (nn.ModuleList, nn.Sequential))]
    for container in containers:
        children = list(container._modules.values())
        if len(children) < 2:
            continue
        sigs = [_block_signature(c, io_shapes) for c in children]

        # Group into maximal runs of consecutive equal signatures (with known shapes).
        runs, i = [], 0
        while i < len(children):
            j = i
            while j + 1 < len(children) and sigs[j + 1] == sigs[i] and sigs[i][2] is not None:
                j += 1
            runs.append((i, j - i + 1))
            i = j + 1
        if all(count == 1 for _, count in runs):
            continue

        rebuilt = OrderedDict()
        for new_idx, (start, count) in enumerate(runs):
            rep = children[start]
            if count > 1:
                base = type(rep)
                tagged = type(base.__name__, (base,), {})  # structural subclass
                tagged.__name__ = tagged.__qualname__ = f"{base.__name__} x{count}"
                rep.__class__ = tagged  # only the label changes; forward is inherited
                collapses.append((base.__name__, count))
            rebuilt[str(new_idx)] = rep
        container._modules = rebuilt

    # Collapsing shortens ModuleLists, which is only valid when they're consumed
    # by iteration (not indexed like `self.heads[k]`). Verify the collapsed copy
    # still runs a forward pass; if not, fall back to the uncollapsed module.
    if collapses:
        try:
            collapsed.eval()
            with torch.no_grad():
                collapsed(*inputs) if isinstance(inputs, tuple) else collapsed(inputs)
        except Exception:
            return module, []
    return collapsed, collapses


# --------------------------------------------------------------------------
# Hiding loose tensor-op nodes (residual adds, mask/RoPE math, etc.)
# torchview draws every op called in forward() that isn't wrapped in a module
# (add, triu, mul, __rpow__, cos, ...) as a pale-blue FunctionNode. They clutter
# the module-level view, so we drop them and reconnect edges *through* them:
# a kept node keeps whatever kept nodes it could reach via a chain of dropped
# ops. Node colors (torchview): module=darkseagreen1, tensor=lightyellow,
# function=aliceblue.
# --------------------------------------------------------------------------
_NODE_RE = re.compile(r"^\s*(\d+)\s+\[label=")
_EDGE_RE = re.compile(r"^\s*(\d+)\s*->\s*(\d+)")
_COLOR_RE = re.compile(r"fillcolor=(\w+)")
_OPNAME_RE = re.compile(r"<TD[^>]*>(\w+)<BR")


def prune_function_nodes(digraph, only_dunder=True):
    """Remove FunctionNode (aliceblue) nodes from a torchview Digraph in place,
    splicing edges transitively so module/tensor connectivity is preserved.

    only_dunder=True drops just the dunder-operator artifacts (__rpow__,
    __rdiv__, ... - from Python operators on plain scalars) and keeps readable
    ops (add, mul, triu, cos, ...). False drops every function node.
    """
    body = digraph.body
    removed, succ = set(), {}
    for entry in body:
        nm = _NODE_RE.match(entry)
        if nm:
            color = _COLOR_RE.search(entry)
            if color and color.group(1) == "aliceblue":
                op = _OPNAME_RE.search(entry)
                is_dunder = bool(op) and op.group(1).startswith("__") and op.group(1).endswith("__")
                if not only_dunder or is_dunder:
                    removed.add(nm.group(1))
            continue
        em = _EDGE_RE.match(entry)
        if em:
            succ.setdefault(em.group(1), []).append(em.group(2))

    def reachable_kept(start):
        """Kept nodes reachable from `start` through only removed nodes."""
        out, seen, stack = set(), set(), list(succ.get(start, []))
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            (stack.extend(succ.get(n, [])) if n in removed else out.add(n))
        return out

    new_edges = set()
    for tail in succ:
        if tail in removed:
            continue
        for head in reachable_kept(tail):
            if head != tail:
                new_edges.add((tail, head))

    # Rebuild body: keep every structural line verbatim (handles arbitrarily
    # nested clusters), drop removed node defs and all original edges, then
    # re-add the spliced edges at the end.
    kept = []
    for entry in body:
        nm, em = _NODE_RE.match(entry), _EDGE_RE.match(entry)
        if nm:
            if nm.group(1) not in removed:
                kept.append(entry)
        elif not em:
            kept.append(entry)  # cluster open/close, labels, graph attrs
    for tail, head in sorted(new_edges, key=lambda e: (int(e[0]), int(e[1]))):
        kept.append(f"\t{tail} -> {head}\n")
    digraph.body = kept


def disambiguate_siblings(module):
    """Relabel same-class sibling submodules by their attribute name so, e.g.,
    the four Linear projections in attention show as W_q/W_k/W_v/W_o instead of
    four identical 'Linear' boxes. Returns a relabeled copy; the original is
    untouched. Numeric (ModuleList) indices become 'Class[i]'."""
    labeled = copy.deepcopy(module)
    for parent in labeled.modules():
        groups = {}
        for attr, child in parent.named_children():
            groups.setdefault(type(child).__name__, []).append((attr, child))
        for _cls, members in groups.items():
            if len(members) < 2:
                continue  # only relabel when siblings share a class (ambiguous)
            for attr, child in members:
                base = type(child)
                # Keep the class name so type stays visible: "ln1: LayerNorm".
                label = f"{base.__name__}[{attr}]" if attr.isdigit() else f"{attr}: {base.__name__}"
                tagged = type(base.__name__, (base,), {})
                tagged.__name__ = tagged.__qualname__ = label
                child.__class__ = tagged  # label only; forward is inherited
    return labeled


def label_input_nodes(graph, module):
    """Rename the graph's 'input-tensor' nodes to the forward() argument names
    (token_ids, text_ids/speech_ids/speaker_vec, ...). Root nodes are in
    argument order; edits the label text in place."""
    params = list(inspect.signature(module.forward).parameters)
    body = graph.visual_graph.body
    for node, pname in zip(list(graph.root_container), params):
        gid = graph.id_dict.get(node.node_id)
        if gid is None:
            continue
        for i, entry in enumerate(body):
            if re.match(rf"^\s*{gid} \[label=", entry):
                body[i] = entry.replace("input-tensor", pname, 1)
                break


def add_compass_ports(digraph):
    """Force every edge to leave the tail's bottom-center (:s) and enter the
    head's top-center (:n), so arrows attach to the top/bottom faces instead of
    wandering into the sides."""
    edge = re.compile(r"^(\s*)(\d+)(?::\w+)? -> (\d+)(?::\w+)?(.*)$")
    rebuilt = []
    for e in digraph.body:
        m = edge.match(e.rstrip("\n"))
        rebuilt.append(f"{m.group(1)}{m.group(2)}:s -> {m.group(3)}:n{m.group(4)}\n" if m else e)
    digraph.body = rebuilt


def label_positional_adds(digraph):
    """Relabel the additive positional-encoding step. Sinusoidal-PE models
    (GPT/BERT/T5) do `x = token_emb + pos_encoding`; the pos buffer is a
    constant leaf torchview hides, so the op shows as a bare `add` fed only by
    an Embedding. Rename those to make the positional addition explicit."""
    body = digraph.body
    name, pred = {}, {}
    for e in body:
        nm = _NODE_RE.match(e)
        if nm:
            n = re.search(r"<TD[^>]*>(\w+)<BR/>", e)
            name[nm.group(1)] = n.group(1) if n else ""
        em = _EDGE_RE.match(e)
        if em:
            pred.setdefault(em.group(2), []).append(em.group(1))
    targets = {
        n for n, ps in pred.items()
        if name.get(n) == "add" and ps and all(name.get(p) == "Embedding" for p in ps)
    }
    if not targets:
        return
    rebuilt = []
    for e in body:
        nm = _NODE_RE.match(e)
        if nm and nm.group(1) in targets:
            e = e.replace(">add<BR/>", ">add (+ positional encoding)<BR/>", 1)
        rebuilt.append(e)
    digraph.body = rebuilt


def align_sibling_ranks(digraph):
    """Put parallel module branches on the same row. Modules sharing an
    identical set of predecessors (e.g. W_q/W_k/W_v all fed by the same tensor)
    get a `rank=same` constraint, injected *inside* their cluster so the dashed
    boxes stay intact. The long edge then falls on the output side."""
    body = digraph.body
    color, pred, node_cluster, cluster_close, stack = {}, {}, {}, {}, []
    for i, e in enumerate(body):
        s = e.strip()
        if s.startswith("subgraph cluster"):
            stack.append(s.split()[1].rstrip("{").strip())
        elif s == "}" and stack:
            cluster_close[stack.pop()] = i
        nm = _NODE_RE.match(e)
        if nm:
            node_cluster[nm.group(1)] = stack[-1] if stack else None
            c = _COLOR_RE.search(e)
            color[nm.group(1)] = c.group(1) if c else ""
        em = _EDGE_RE.match(e)
        if em:
            pred.setdefault(em.group(2), []).append(em.group(1))

    groups = {}
    for node, preds in pred.items():
        if color.get(node) == "darkseagreen1":  # module nodes only
            groups.setdefault(tuple(sorted(preds)), []).append(node)

    inserts = {}  # body index of a cluster's '}' (or None for top level) -> lines
    for members in groups.values():
        clusters = {node_cluster.get(n) for n in members}
        if len(members) < 2 or len(clusters) != 1:
            continue  # spans clusters -> skip, don't break the boxes
        cid = next(iter(clusters))
        pos = cluster_close.get(cid) if cid is not None else None
        inserts.setdefault(pos, []).append("\t{rank=same; " + " ".join(members) + "}\n")
    if not inserts:
        return

    rebuilt = []
    for i, e in enumerate(body):
        rebuilt.extend(inserts.get(i, []))
        rebuilt.append(e)
    rebuilt.extend(inserts.get(None, []))
    digraph.body = rebuilt


def render(
    module, inputs, name, out_dir=TF_IMAGES, depth=2, fmt="svg", roll=False, collapse=True,
    op_nodes="dunder", label_siblings=True, splines="spline", device="cpu",
):
    """Draw `module` to `out_dir/name.fmt`. The generic entry point.

    inputs: a single tensor, or a tuple of positional args matching forward().
    collapse: fold identical consecutive blocks into '<Class> xN' nodes.
    op_nodes: which loose tensor-op nodes to draw - "dunder" hides only
        __rpow__/__rdiv__-style ops, "all" hides every op, "keep" hides none.
    label_siblings: relabel same-class sibling modules by attribute name
        (W_q/W_k/W_v/W_o rather than four identical 'Linear' nodes).
    """
    from torchview import draw_graph

    if collapse:
        try:
            module, collapses = collapse_repeated_blocks(module, inputs)
            if collapses:
                print("  collapsed " + ", ".join(f"{c} x{n}" for c, n in collapses))
        except Exception as exc:  # never let collapsing break the diagram
            print(f"  (collapse skipped: {exc})")

    if label_siblings:
        module = disambiguate_siblings(module)

    module = module.to(device).eval()
    graph = draw_graph(
        module,
        input_data=inputs,
        depth=depth,
        roll=roll,
        expand_nested=True,
        graph_name=name,
        device=device,
    )
    if op_nodes != "keep":
        prune_function_nodes(graph.visual_graph, only_dunder=(op_nodes == "dunder"))
    if label_siblings:
        try:
            label_input_nodes(graph, module)
            label_positional_adds(graph.visual_graph)
            align_sibling_ranks(graph.visual_graph)
        except Exception as exc:
            print(f"  (labeling/alignment skipped: {exc})")

    # Attach edges to top/bottom faces, then route. Compass ports keep arrows
    # off the sides; splines="spline" then curves gently between those ports.
    add_compass_ports(graph.visual_graph)
    graph.visual_graph.graph_attr.update(nodesep="0.5", ranksep="0.6", splines=splines)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / name
    graph.visual_graph.render(str(out_path), format=fmt, cleanup=True)
    print(f"  saved {out_path}.{fmt}  (depth={depth}, collapse={collapse}, ops={op_nodes})")
    return graph


# --------------------------------------------------------------------------
# Builders for the models already in the repo. Each returns
# (module_to_visualize, example_inputs, default_depth).
# --------------------------------------------------------------------------
def _gpt():
    sys.path.insert(0, str(TF_TORCH))
    from gpt_torch import GPTModel

    m = GPTModel(vocab_size=100, d_model=128, num_layers=6, num_heads=4, d_ff=512, max_seq_len=64)
    return m, torch.randint(0, 100, (1, 32)), 2


def _bert():
    sys.path.insert(0, str(TF_TORCH))
    from bert_torch import BERTModel

    m = BERTModel(vocab_size=100, d_model=128, num_layers=6, num_heads=4, d_ff=512, max_seq_len=64)
    return m, torch.randint(0, 100, (1, 32)), 2


def _t5():
    sys.path.insert(0, str(TF_TORCH))
    from t5_torch import T5Model

    m = T5Model(vocab_size=100, d_model=128, num_layers=4, num_heads=4, d_ff=512, max_seq_len=64)
    enc = torch.randint(0, 100, (1, 20))
    dec = torch.randint(0, 100, (1, 16))
    return m, (enc, dec), 2


def _qwen_model():
    return _import_from_path(AUDIO_TORCH / "qwen3_tts.py", "qwen3_tts")


def _qwen_backbone():
    q = _qwen_model()
    m = q.Qwen3TTS(text_vocab=64)
    text = torch.randint(0, 64, (1, 10))
    code0 = torch.randint(0, 256, (1, 20))
    spk = torch.randn(1, m.d_model)
    return m.backbone, (text, code0, spk), 3


def _qwen_codec():
    q = _qwen_model()
    m = q.Qwen3TTS(text_vocab=64)
    return m.codec, torch.randn(1, 16000) * 0.1, 2


def _qwen_mtp():
    q = _qwen_model()
    m = q.Qwen3TTS(text_vocab=64)
    hidden = torch.randn(4, m.d_model)
    codes = torch.randint(0, 256, (4, m.num_codebooks))
    return m.mtp, (hidden, codes), 2

def _qwen_tts():
    # Whole model via forward()=compute_loss (the LM training path): text +
    # codec tokens + reference wav -> loss. Note this traces the LM path only
    # (codec.encode/decode and the generate loop aren't part of forward).
    q = _qwen_model()
    m = q.Qwen3TTS(text_vocab=64)
    wav = torch.randn(1, 16000) * 0.1
    text = torch.randint(0, 64, (1, 10))
    codes = m.codec.encode(wav)          # (1, frames, num_codebooks)
    return m, (text, codes, wav), 2

REGISTRY = {
    "gpt": _gpt,
    "bert": _bert,
    "t5": _t5,
    "qwen-backbone": _qwen_backbone,
    "qwen-codec": _qwen_codec,
    "qwen-mtp": _qwen_mtp,
    "qwen-tts": _qwen_tts,
}


def main():
    parser = argparse.ArgumentParser(description="Generate model architecture diagrams.")
    parser.add_argument("models", nargs="+", help=f"one or more of {list(REGISTRY)} or 'all'")
    parser.add_argument("--depth", type=int, default=None, help="override nesting depth")
    parser.add_argument("--out", default=None, help="output directory (default: the owning module's images/)")
    parser.add_argument("--format", default="svg", choices=["png", "svg", "pdf"])
    parser.add_argument("--no-collapse", action="store_true", help="show every repeated block")
    parser.add_argument("--show-ops", action="store_true", help="keep every op node (incl. __rpow__ etc.)")
    parser.add_argument("--hide-all-ops", action="store_true", help="drop all op nodes, not just dunders")
    parser.add_argument("--no-labels", action="store_true", help="don't relabel same-class siblings (W_q/W_k/...)")
    parser.add_argument("--roll", action="store_true", help="merge identical ops (can give confusing 2-input nodes)")
    parser.add_argument("--splines", default="spline", choices=["spline", "polyline", "ortho", "line"],
                        help="edge routing style (default ortho = right angles)")
    args = parser.parse_args()

    names = list(REGISTRY) if "all" in args.models else args.models
    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        parser.error(f"unknown model(s) {unknown}; choose from {list(REGISTRY)} or 'all'")

    op_nodes = "keep" if args.show_ops else ("all" if args.hide_all_ops else "dunder")

    warnings.filterwarnings("ignore")
    for name in names:
        print(f"[{name}]")
        module, inputs, default_depth = REGISTRY[name]()
        render(
            module,
            inputs,
            name,
            out_dir=args.out or (AUDIO_IMAGES if name.startswith("qwen") else TF_IMAGES),
            depth=args.depth if args.depth is not None else default_depth,
            fmt=args.format,
            roll=args.roll,
            collapse=not args.no_collapse,
            op_nodes=op_nodes,
            label_siblings=not args.no_labels,
            splines=args.splines,
        )


if __name__ == "__main__":
    main()
