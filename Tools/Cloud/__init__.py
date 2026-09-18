from Tools.Core.tool_factory import make_tool
from Tools.Core.tool_result import ToolResult
def docker_ps(ctx=None, **kw):
    import subprocess
    try:
        p = subprocess.run(["docker","ps"], capture_output=True, text=True, timeout=15)
        return ToolResult("docker_ps", p.returncode==0, p.stdout or p.stderr)
    except Exception as e:
        return ToolResult("docker_ps", False, str(e), error_type="runtime")
def register(registry):
    registry.register(make_tool("docker_ps", "List docker containers", {}, docker_ps, "cloud", "terminal:run"))
