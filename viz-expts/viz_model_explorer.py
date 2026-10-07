"""
Interactive model-graph viewer via Google's Model Explorer.

An alternative to viz.py (which renders static graphviz PNG/SVG). Model Explorer
opens a browser UI with nested, collapsible layers, GPU-accelerated rendering
and clean edge routing - good for exploring a model rather than exporting a
figure.

How it works: each model is exported with `torch.export.export(model, args)`
into an ExportedProgram, then handed to `model_explorer.visualize_pytorch`,
which starts a local web server and opens the graph in your browser.

Model definitions/inputs are reused from viz.py's REGISTRY, so the same names
work here (gpt, bert, t5, qwen-backbone, qwen-codec, qwen-mtp, qwen-tts).

Requirements:
    pip install ai-edge-model-explorer torch

Usage:
    python3 viz_model_explorer.py gpt              # export + open in browser
    python3 viz_model_explorer.py qwen-backbone --port 8081
    python3 viz_model_explorer.py gpt bert t5      # all on one reused server

Notes:
    * The server is interactive and blocks until you stop it (Ctrl-C).
    * Model Explorer shows the ExportedProgram (ATen op) graph. It has its own
      collapse/expand and layer grouping, so viz.py's block-collapse / op-hiding
      passes are not applied here - that is the point of trying a second tool.
"""
import argparse
import warnings

import torch

import viz  # reuse REGISTRY (module + example inputs) and the model builders


def export(name):
    """Build a registered model and return its ExportedProgram."""
    module, inputs, _ = viz.REGISTRY[name]()
    module.eval()
    args = inputs if isinstance(inputs, tuple) else (inputs,)
    return torch.export.export(module, args)


def _patch_adapter():
    """Make Model Explorer's PyTorch adapter nicer for these models.

    By default it dumps every lifted weight/buffer placeholder into one flat
    `inputs` namespace - so a transformer shows ~70 tensors in a single row.
    We instead nest each parameter under a collapsible `params/<module path>`
    tree (derived from its fully-qualified name) and label it by its leaf name,
    leaving only the real user inputs under `inputs`.
    """
    from model_explorer import pytorch_exported_program_adater_impl as pe

    impl = pe.PytorchExportedProgramAdapterImpl
    orig_hierarchy, orig_label = impl.get_hierachy, impl.get_label

    def get_hierachy(self, fx_node):
        if fx_node.op == "placeholder":
            target = self.inputs_map.get(fx_node.name, [None])[0]
            if not target:
                return "inputs"                       # a real user input
            module_path = target.rsplit(".", 1)[0]    # drop .weight / .bias
            return "params/" + module_path.replace(".", "/") if module_path else "params"
        return orig_hierarchy(self, fx_node)

    def get_label(self, fx_node):
        if fx_node.op == "placeholder":
            target = self.inputs_map.get(fx_node.name, [None])[0]
            if target:
                return ".".join(target.split(".")[-2:])  # e.g. W_q.weight
        return orig_label(self, fx_node)

    impl.get_hierachy, impl.get_label = get_hierachy, get_label


def main():
    parser = argparse.ArgumentParser(description="Open model graphs in Model Explorer.")
    parser.add_argument("models", nargs="+", help=f"one or more of {list(viz.REGISTRY)} or 'all'")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    names = list(viz.REGISTRY) if "all" in args.models else args.models
    unknown = [n for n in names if n not in viz.REGISTRY]
    if unknown:
        parser.error(f"unknown model(s) {unknown}; choose from {list(viz.REGISTRY)} or 'all'")

    import model_explorer

    warnings.filterwarnings("ignore")
    _patch_adapter()
    # First model starts the server; the rest reuse it (each appears as a tab).
    for i, name in enumerate(names):
        print(f"[{name}] exporting...")
        ep = export(name)
        model_explorer.visualize_pytorch(
            name,
            exported_program=ep,
            host=args.host,
            port=args.port,
            reuse_server=(i > 0),
            reuse_server_host=args.host,
            reuse_server_port=args.port,
        )


if __name__ == "__main__":
    main()
