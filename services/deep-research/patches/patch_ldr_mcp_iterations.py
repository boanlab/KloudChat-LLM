"""Widen local-deep-research's MCP `iterations` / `questions_per_iteration` params to accept strings.

MCP clients validate tool arguments against the inputSchema before dispatch, and
models often send "2" rather than 2; with the upstream ``Optional[int]`` the call
is rejected and research degrades to plain search. The types become
``Optional[Union[int, str]]`` and numeric strings are coerced in the validators.
Applied at image build time; idempotent.
"""
import pathlib

import local_deep_research.mcp.server as _m

p = pathlib.Path(_m.__file__)
s = p.read_text()

if "Optional[Union[int, str]]" in s:
    print("[patch_ldr_mcp_iterations] already applied, skipping")
    raise SystemExit(0)

orig = s
s = s.replace(
    "from typing import Any, Dict, Optional\n",
    "from typing import Any, Dict, Optional, Union\n",
)
s = s.replace(
    "iterations: Optional[int]",
    "iterations: Optional[Union[int, str]]",
)
s = s.replace(
    "questions_per_iteration: Optional[int]",
    "questions_per_iteration: Optional[Union[int, str]]",
)
s = s.replace(
    "    if not isinstance(iterations, int) or iterations < 1:",
    "    if isinstance(iterations, str) and iterations.strip().lstrip('+-').isdigit():\n"
    "        iterations = int(iterations)\n"
    "    if not isinstance(iterations, int) or iterations < 1:",
)
s = s.replace(
    "    if not isinstance(qpi, int) or qpi < 1:",
    "    if isinstance(qpi, str) and qpi.strip().lstrip('+-').isdigit():\n"
    "        qpi = int(qpi)\n"
    "    if not isinstance(qpi, int) or qpi < 1:",
)

assert "from typing import Any, Dict, Optional, Union" in s, "Union import failed"
assert "Optional[Union[int, str]]" in s, "type-hint widen failed"
assert "iterations = int(iterations)" in s, "iterations coercion failed"
assert "qpi = int(qpi)" in s, "qpi coercion failed"
assert s != orig, "patch made no changes — LDR upstream layout changed?"

p.write_text(s)
print("[patch_ldr_mcp_iterations] applied to", p)
