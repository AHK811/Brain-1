"""Discover and register tools from category packages."""
from __future__ import annotations
from Tools.Core.tool_registry import ToolRegistry

def load_all_builtin_tools(registry: ToolRegistry) -> ToolRegistry:
    loaders = [
        ("Tools.Filesystem", "register"),
        ("Tools.Code", "register"),
        ("Tools.Terminal", "register"),
        ("Tools.Python", "register"),
        ("Tools.Search", "register"),
        ("Tools.Git", "register"),
        ("Tools.Documents", "register"),
        ("Tools.Media", "register"),
        ("Tools.Security", "register"),
        ("Tools.APIs", "register"),
        ("Tools.Databases", "register"),
        ("Tools.Browser", "register"),
        ("Tools.GitHub", "register"),
        ("Tools.Cloud", "register"),
        ("Tools.Communication", "register"),
        ("Tools.Plugins", "register"),
    ]
    for mod_name, fn_name in loaders:
        try:
            mod = __import__(mod_name, fromlist=[fn_name])
            getattr(mod, fn_name)(registry)
        except Exception:
            continue
    return registry
