"""Declare tool dependencies (soft)."""
from __future__ import annotations

# tool -> required tools that should exist
DEPS = {
    "refactor": ["read_file", "write_file"],
    "test_generator": ["read_file", "python_unit_test"],
    "patch_generator": ["read_file", "git_diff"],
}

def missing_deps(available: set[str], tool: str) -> list[str]:
    return [d for d in DEPS.get(tool, []) if d not in available]
