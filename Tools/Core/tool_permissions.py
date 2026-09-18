"""Permission helpers."""
from __future__ import annotations
from Tools.Core.tool_context import ToolContext

PERMISSIONS = {
    "fs:read", "fs:write", "fs:delete",
    "terminal:run", "python:eval",
    "net:read", "net:write",
    "git:write", "db:write", "secrets:read",
}

def grant(ctx: ToolContext, *perms: str) -> ToolContext:
    ctx.permissions |= set(perms)
    return ctx

def revoke(ctx: ToolContext, *perms: str) -> ToolContext:
    ctx.permissions -= set(perms)
    return ctx
