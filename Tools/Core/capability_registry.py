"""Map high-level capabilities to tool names."""
from __future__ import annotations

CAPABILITIES = {
    "read_code": ["read_file", "list_directory", "search_files", "code_search"],
    "edit_code": ["write_file", "edit_file", "replace_text"],
    "run_tests": ["run_terminal", "python_unit_test"],
    "search_web": ["web_search", "web_open"],
    "git": ["git_status", "git_diff", "git_commit"],
    "docs": ["pdf_read", "docx_read", "xlsx_read"],
}

def tools_for(capability: str) -> list[str]:
    return list(CAPABILITIES.get(capability, []))
