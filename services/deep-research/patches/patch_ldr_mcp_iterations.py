"""Patch local-deep-research's MCP tool schema so numeric params accept strings.

Why: MCP clients validate a tool_call's arguments against the tool's
inputSchema *before* dispatching to the MCP server. LDR types the exposed
`quick_research`/`detailed_research` params `iterations` and
`questions_per_iteration` as ``Optional[int]``, so the generated JSON schema is
``integer``. The Deep Research model frequently emits them as
JSON strings ("2" instead of 2); the client then rejects the call with
"Received tool input did not match expected schema", the deep_research engine
never runs, and the agent silently degrades to plain web_search/fetch_url.

Fix: widen the two exposed numeric tool params to ``Optional[Union[int, str]]``
(so the schema accepts a string) and coerce numeric strings to int inside the
validators (the schema check is at the MCP layer, so the function
runtime must accept the string form too).

Applied at image build time (Dockerfile.deep-research-mcp). Idempotent.
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
