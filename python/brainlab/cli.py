from __future__ import annotations

import importlib
import sys
from pathlib import Path

from brainlab.spec.config import repo_root
from brainlab.util import configure_stdout

COMMANDS = {
    "golden": ("brainlab.export.make_golden", "main"),
    "prepare": ("brainlab.data.prepare", "main"),
    "train": ("brainlab.train.train", "main"),
    "export": ("brainlab.export.to_safetensors", "main"),
    "ppl": ("brainlab.eval.perplexity", "main"),
    "leak": ("brainlab.data.leakage", "main"),
    "generate": ("brainlab.eval.generate", "main"),
    "bench": ("bench.harness.run_all", "main"),
    "report": ("bench.harness.report", "main"),
}


def usage() -> int:
    print("brainlab <command> [args]\n")
    for name, (module, _) in sorted(COMMANDS.items()):
        print(f"  {name:9s} -> {module}")
    print("\nrun `brainlab <command> --help` for options")
    return 2


def main(argv: list[str] | None = None) -> int:
    configure_stdout()
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        return usage()
    command, *rest = argv
    if command not in COMMANDS:
        print(f"unknown command {command!r}", file=sys.stderr)
        return usage()
    root = repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    module_name, function = COMMANDS[command]
    module = importlib.import_module(module_name)
    return int(getattr(module, function)(rest) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
