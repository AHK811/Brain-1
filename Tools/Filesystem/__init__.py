from __future__ import annotations
from pathlib import Path
from typing import Any
from Tools.Core.tool_factory import make_tool
from Tools.Core.tool_result import ToolResult

def _resolve(ctx, path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        base = ctx.workspace_root or ctx.cwd
        p = base / p
    return p.resolve()

def read_file(ctx=None, path: str = "", offset: int = 1, limit: int = 200, **kw) -> ToolResult:
    p = _resolve(ctx, path)
    if not p.exists():
        return ToolResult("read_file", False, f"not found: {p}", error_type="runtime")
    lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    start = max(int(offset) - 1, 0)
    chunk = lines[start:start + int(limit)]
    numbered = "\n".join(f"{start+i+1}| {ln}" for i, ln in enumerate(chunk))
    return ToolResult("read_file", True, numbered or "(empty)", meta={"path": str(p), "total_lines": len(lines)})

def write_file(ctx=None, path: str = "", content: str = "", **kw) -> ToolResult:
    p = _resolve(ctx, path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return ToolResult("write_file", True, f"wrote {len(content)} chars to {p}")

def append_file(ctx=None, path: str = "", content: str = "", **kw) -> ToolResult:
    p = _resolve(ctx, path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(content)
    return ToolResult("append_file", True, f"appended {len(content)} chars to {p}")

def edit_file(ctx=None, path: str = "", old_string: str = "", new_string: str = "", replace_all: bool = False, **kw) -> ToolResult:
    p = _resolve(ctx, path)
    text = p.read_text(encoding="utf-8", errors="ignore")
    if old_string not in text:
        return ToolResult("edit_file", False, "old_string not found", error_type="runtime")
    if replace_all:
        text2 = text.replace(old_string, new_string)
    else:
        text2 = text.replace(old_string, new_string, 1)
    p.write_text(text2, encoding="utf-8")
    return ToolResult("edit_file", True, f"edited {p}")

def replace_text(ctx=None, path: str = "", old_string: str = "", new_string: str = "", **kw):
    return edit_file(ctx=ctx, path=path, old_string=old_string, new_string=new_string, replace_all=True)

def move_file(ctx=None, src: str = "", dst: str = "", **kw) -> ToolResult:
    import shutil
    a, b = _resolve(ctx, src), _resolve(ctx, dst)
    b.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(a), str(b))
    return ToolResult("move_file", True, f"{a} -> {b}")

def copy_file(ctx=None, src: str = "", dst: str = "", **kw) -> ToolResult:
    import shutil
    a, b = _resolve(ctx, src), _resolve(ctx, dst)
    b.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(a), str(b))
    return ToolResult("copy_file", True, f"copied {a} -> {b}")

def delete_file(ctx=None, path: str = "", **kw) -> ToolResult:
    p = _resolve(ctx, path)
    if p.is_file():
        p.unlink()
        return ToolResult("delete_file", True, f"deleted {p}")
    return ToolResult("delete_file", False, f"not a file: {p}", error_type="runtime")

def create_directory(ctx=None, path: str = "", **kw) -> ToolResult:
    p = _resolve(ctx, path)
    p.mkdir(parents=True, exist_ok=True)
    return ToolResult("create_directory", True, f"created {p}")

def remove_directory(ctx=None, path: str = "", recursive: bool = False, **kw) -> ToolResult:
    import shutil
    p = _resolve(ctx, path)
    if recursive:
        shutil.rmtree(p)
    else:
        p.rmdir()
    return ToolResult("remove_directory", True, f"removed {p}")

def list_directory(ctx=None, path: str = ".", **kw) -> ToolResult:
    p = _resolve(ctx, path)
    if not p.is_dir():
        return ToolResult("list_directory", False, f"not a directory: {p}", error_type="runtime")
    entries = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
    lines = [("d " if e.is_dir() else "f ") + e.name for e in entries]
    return ToolResult("list_directory", True, "\n".join(lines) or "(empty)", meta={"path": str(p)})

def search_files(ctx=None, query: str = "", path: str = ".", max_hits: int = 50, **kw) -> ToolResult:
    root = _resolve(ctx, path)
    hits = []
    for fp in root.rglob("*"):
        if not fp.is_file():
            continue
        try:
            text = fp.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if query in text or query in fp.name:
            hits.append(str(fp))
            if len(hits) >= max_hits:
                break
    return ToolResult("search_files", True, "\n".join(hits) or "(no hits)")

def tree(ctx=None, path: str = ".", max_depth: int = 3, **kw) -> ToolResult:
    root = _resolve(ctx, path)
    lines = []
    def walk(d: Path, depth: int, prefix: str):
        if depth > max_depth:
            return
        try:
            kids = sorted(d.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
        except Exception:
            return
        for i, k in enumerate(kids):
            last = i == len(kids) - 1
            branch = "└── " if last else "├── "
            lines.append(prefix + branch + k.name + ("/" if k.is_dir() else ""))
            if k.is_dir():
                walk(k, depth + 1, prefix + ("    " if last else "│   "))
    lines.append(str(root))
    walk(root, 1, "")
    return ToolResult("tree", True, "\n".join(lines))

def metadata(ctx=None, path: str = "", **kw) -> ToolResult:
    p = _resolve(ctx, path)
    if not p.exists():
        return ToolResult("metadata", False, "not found", error_type="runtime")
    st = p.stat()
    return ToolResult("metadata", True, f"size={st.st_size} mtime={st.st_mtime} path={p}")

def permissions(ctx=None, path: str = "", **kw) -> ToolResult:
    import os, stat
    p = _resolve(ctx, path)
    mode = stat.filemode(p.stat().st_mode)
    return ToolResult("permissions", True, f"{mode} {p}")

def archive(ctx=None, path: str = "", out: str = "", **kw) -> ToolResult:
    import shutil
    src = _resolve(ctx, path)
    dst = _resolve(ctx, out or (str(src) + ".zip"))
    base = str(dst).removesuffix(".zip")
    outp = shutil.make_archive(base, "zip", root_dir=str(src if src.is_dir() else src.parent),
                               base_dir=src.name if src.is_dir() else src.name)
    return ToolResult("archive", True, f"created {outp}")

def extract(ctx=None, path: str = "", out: str = "", **kw) -> ToolResult:
    import zipfile, tarfile
    src = _resolve(ctx, path)
    dest = _resolve(ctx, out or str(src.with_suffix("")))
    dest.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(src):
        with zipfile.ZipFile(src) as z:
            z.extractall(dest)
    elif tarfile.is_tarfile(src):
        with tarfile.open(src) as t:
            t.extractall(dest)
    else:
        return ToolResult("extract", False, "unsupported archive", error_type="runtime")
    return ToolResult("extract", True, f"extracted to {dest}")

def checksum(ctx=None, path: str = "", algo: str = "sha256", **kw) -> ToolResult:
    import hashlib
    p = _resolve(ctx, path)
    h = hashlib.new(algo)
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return ToolResult("checksum", True, f"{algo}={h.hexdigest()} path={p}")

def register(registry) -> None:
    specs = [
        make_tool("read_file", "Read file lines", {"path": {"type":"string","required":True}, "offset":{"type":"integer"}, "limit":{"type":"integer"}}, read_file, "filesystem", "fs:read"),
        make_tool("write_file", "Write/overwrite file", {"path":{"type":"string","required":True}, "content":{"type":"string","required":True}}, write_file, "filesystem", "fs:write", parallel_safe=False),
        make_tool("append_file", "Append to file", {"path":{"type":"string","required":True}, "content":{"type":"string","required":True}}, append_file, "filesystem", "fs:write", parallel_safe=False),
        make_tool("edit_file", "Replace first/all occurrences", {"path":{"type":"string","required":True}, "old_string":{"type":"string","required":True}, "new_string":{"type":"string","required":True}, "replace_all":{"type":"boolean"}}, edit_file, "filesystem", "fs:write", parallel_safe=False),
        make_tool("replace_text", "Replace all in file", {"path":{"type":"string","required":True}, "old_string":{"type":"string","required":True}, "new_string":{"type":"string","required":True}}, replace_text, "filesystem", "fs:write", parallel_safe=False),
        make_tool("move_file", "Move/rename", {"src":{"type":"string","required":True}, "dst":{"type":"string","required":True}}, move_file, "filesystem", "fs:write", parallel_safe=False),
        make_tool("copy_file", "Copy file", {"src":{"type":"string","required":True}, "dst":{"type":"string","required":True}}, copy_file, "filesystem", "fs:write"),
        make_tool("delete_file", "Delete file", {"path":{"type":"string","required":True}}, delete_file, "filesystem", "fs:delete", parallel_safe=False),
        make_tool("create_directory", "mkdir -p", {"path":{"type":"string","required":True}}, create_directory, "filesystem", "fs:write"),
        make_tool("remove_directory", "Remove directory", {"path":{"type":"string","required":True}, "recursive":{"type":"boolean"}}, remove_directory, "filesystem", "fs:delete", parallel_safe=False),
        make_tool("list_directory", "List directory", {"path":{"type":"string"}}, list_directory, "filesystem", "fs:read"),
        make_tool("search_files", "Search file contents/names", {"query":{"type":"string","required":True}, "path":{"type":"string"}, "max_hits":{"type":"integer"}}, search_files, "filesystem", "fs:read"),
        make_tool("tree", "Directory tree", {"path":{"type":"string"}, "max_depth":{"type":"integer"}}, tree, "filesystem", "fs:read"),
        make_tool("file_metadata", "File metadata", {"path":{"type":"string","required":True}}, metadata, "filesystem", "fs:read"),
        make_tool("file_permissions", "Show permissions", {"path":{"type":"string","required":True}}, permissions, "filesystem", "fs:read"),
        make_tool("archive", "Zip a path", {"path":{"type":"string","required":True}, "out":{"type":"string"}}, archive, "filesystem", "fs:write"),
        make_tool("extract", "Extract archive", {"path":{"type":"string","required":True}, "out":{"type":"string"}}, extract, "filesystem", "fs:write"),
        make_tool("checksum", "File checksum", {"path":{"type":"string","required":True}, "algo":{"type":"string"}}, checksum, "filesystem", "fs:read"),
    ]
    for s in specs:
        registry.register(s)
