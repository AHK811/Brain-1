from Tools.Core.tool_factory import make_tool
from Tools.Core.tool_result import ToolResult
def sqlite_query(ctx=None, path: str = "", sql: str = "", **kw):
    try:
        import sqlite3
        con = sqlite3.connect(path)
        cur = con.execute(sql)
        if sql.strip().lower().startswith("select"):
            rows = cur.fetchmany(100)
            return ToolResult("sqlite_query", True, str(rows))
        con.commit()
        return ToolResult("sqlite_query", True, f"ok rowcount={cur.rowcount}")
    except Exception as e:
        return ToolResult("sqlite_query", False, str(e), error_type="runtime")
def register(registry):
    registry.register(make_tool("sqlite_query", "Run SQLite SQL", {"path":{"type":"string","required":True},"sql":{"type":"string","required":True}}, sqlite_query, "databases", "db:write", parallel_safe=False))
