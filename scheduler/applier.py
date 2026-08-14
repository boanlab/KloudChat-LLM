"""Placement decisions applied to nodes.

One placement implies three changes:

    (a) vLLM options in the node's .env — ``{env_prefix}_{MAX_LEN,GPU_UTIL}``
    (b) compose services started, stopped or recreated on the node
    (c) ``{env_prefix}_URL`` in the orchestrator's .env, read by
        gen-litellm-config.sh

A ChangePlan is built first and executed after confirmation. Applying twice is a
no-op, so a rebooted node reconverges.
"""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Sequence

from scheduler.inventory import WHISPER_PORT
from scheduler.planner import Placement, Plan
from scheduler.registry import ModelSpec
from scheduler.types import NodeSpec

#: Service-name prefix the scheduler manages. ``current`` holds the node's whole
#: ``docker ps``, including the resident transcription backend and, on a
#: single-host deployment, the stack itself — none of which may be stopped for
#: being absent from a placement plan.
MANAGED_SERVICE_PREFIX = "vllm-"


@dataclass(frozen=True)
class RemoteLayout:
    """Where compose runs on a node."""

    workdir: str = "kloudchat-llm"
    compose_file: str = "docker-compose.vllm.yml"
    env_file: str = ".env"


@dataclass(frozen=True)
class NodeAction:
    node_id: str
    host: str
    kind: str          # "env" | "start" | "stop" | "recreate"
    description: str
    command: str


@dataclass
class ChangePlan:
    actions: list[NodeAction] = field(default_factory=list)
    #: Values to write into the orchestrator's .env, key to value
    local_env: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.actions and not self.local_env


def _env_set(path: str, key: str, value: str) -> str:
    """Shell expression updating one .env line, appending it when absent."""
    q_path = shlex.quote(path)
    line = shlex.quote(f"{key}={value}")
    return (
        f"touch {q_path} && "
        f"if grep -q \"^{key}=\" {q_path}; then "
        f"sed -i \"s|^{key}=.*|{key}={value}|\" {q_path}; "
        f"else printf '%s\\n' {line} >> {q_path}; fi"
    )


def _read_env_keys(path: str, keys: Iterable[str]) -> dict[str, str]:
    """Current values of ``keys`` in the local .env. Missing keys read empty."""
    out = {k: "" for k in keys}
    try:
        with open(path) as f:
            for raw in f:
                line = raw.rstrip("\n")
                if "=" not in line or line.lstrip().startswith("#"):
                    continue
                k, _, v = line.partition("=")
                if k in out:
                    out[k] = v
    except FileNotFoundError:
        pass
    return out


def _url_csvs(target: Plan, specs: Sequence[ModelSpec],
              nodes: Sequence[NodeSpec],
              stt_hosts: Sequence[str] = ()) -> dict[str, str]:
    """URL CSVs per model. Unplaced models get an empty value, never a stale URL.

    ``stt_hosts`` are nodes whose transcription backend answered a probe, so
    nodes without one (arm64) never appear. An empty ``WHISPER_URLS`` is what
    makes gen-litellm-config register STT against OpenRouter.
    """
    host_of = {n.node_id: n.hostname.split("@")[-1] for n in nodes}
    by_id = {s.id: s for s in specs}
    urls: dict[str, set[str]] = {f"{s.env_prefix}_URL": set() for s in specs}
    for p in target.placements:
        spec = by_id.get(p.model_id)
        if spec is None:
            continue
        host = host_of.get(p.node_id, p.node_id)
        urls[f"{spec.env_prefix}_URL"].add(f"http://{host}:{spec.port}")
    out = {k: ",".join(sorted(v)) for k, v in urls.items()}
    out["WHISPER_URLS"] = ",".join(
        f"http://{h.split('@')[-1]}:{WHISPER_PORT}" for h in stt_hosts
    )
    return out


def compute_diff(
    *,
    target: Plan,
    current: dict[str, set[str]],
    specs: Sequence[ModelSpec],
    nodes: Sequence[NodeSpec],
    layout: RemoteLayout = RemoteLayout(),
    local_env_path: Optional[str] = None,
    stt_hosts: Sequence[str] = (),
) -> ChangePlan:
    """Changes that take ``current`` to ``target``.

    Args:
        current: node id to the compose services running there.
        local_env_path: the orchestrator's .env. None skips the URL update.
        stt_hosts: hosts whose transcription backend answered. Becomes
            ``WHISPER_URLS``.
    """
    change = ChangePlan(notes=list(target.notes))
    by_id = {s.id: s for s in specs}
    host_of = {n.node_id: n.hostname for n in nodes}

    cd = f"cd {shlex.quote(layout.workdir)}"
    compose = f"docker compose -f {shlex.quote(layout.compose_file)}"
    env_path = f"{layout.workdir}/{layout.env_file}"

    target_by_node: dict[str, list[Placement]] = {}
    for p in target.placements:
        target_by_node.setdefault(p.node_id, []).append(p)

    for node_id in sorted(set(target_by_node) | set(current)):
        host = host_of.get(node_id, node_id)
        placements = target_by_node.get(node_id, [])
        want = {by_id[p.model_id].service for p in placements if p.model_id in by_id}
        # Managed services only: the prefix, plus any name models.yaml declared
        have = {
            s for s in current.get(node_id, set())
            if s.startswith(MANAGED_SERVICE_PREFIX) or s in want
        }

        # (a) Options first — .env must be current before a service starts
        for p in sorted(placements, key=lambda x: x.model_id):
            spec = by_id.get(p.model_id)
            if spec is None:
                continue
            for key, value in (
                (f"{spec.env_prefix}_MAX_LEN", str(p.ctx)),
                (f"{spec.env_prefix}_GPU_UTIL", f"{p.gpu_util:.2f}"),
            ):
                change.actions.append(NodeAction(
                    node_id, host, "env", f"{key}={value}",
                    _env_set(env_path, key, value),
                ))

        for service in sorted(have - want):
            change.actions.append(NodeAction(
                node_id, host, "stop", f"stop {service}",
                f"{cd} && {compose} stop {shlex.quote(service)}",
            ))
        for service in sorted(want - have):
            change.actions.append(NodeAction(
                node_id, host, "start", f"start {service}",
                f"{cd} && {compose} up -d {shlex.quote(service)}",
            ))
        # Running already, but options may have changed
        for service in sorted(want & have):
            change.actions.append(NodeAction(
                node_id, host, "recreate", f"recreate {service}",
                f"{cd} && {compose} up -d --force-recreate {shlex.quote(service)}",
            ))

    if local_env_path:
        desired = _url_csvs(target, specs, nodes, stt_hosts)
        actual = _read_env_keys(local_env_path, desired)
        change.local_env = {k: v for k, v in desired.items() if actual.get(k, "") != v}

    return change


def _run(host: str, command: str, *, timeout: int = 300) -> tuple[int, str]:
    try:
        r = subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "LogLevel=ERROR",
             host, command],
            capture_output=True, text=True, timeout=timeout,
        )
        return r.returncode, (r.stdout + r.stderr).strip()
    except subprocess.TimeoutExpired:
        return 124, f"timeout after {timeout}s"
    except OSError as exc:
        return 1, str(exc)


def apply(
    change: ChangePlan,
    *,
    local_env_path: Optional[str] = None,
    runner: Callable[[str, str], tuple[int, str]] = _run,
) -> list[str]:
    """Execute the changes, returning failures.

    A failing node does not stop the others.
    """
    failures: list[str] = []
    for action in change.actions:
        rc, out = runner(action.host, action.command)
        if rc != 0:
            failures.append(f"{action.node_id} {action.description}: {out}")

    if local_env_path and change.local_env:
        _write_local_env(local_env_path, change.local_env)
    return failures


def _write_local_env(path: str, values: dict[str, str]) -> None:
    try:
        with open(path) as f:
            lines = f.read().splitlines()
    except FileNotFoundError:
        lines = []
    remaining = dict(values)
    out: list[str] = []
    for line in lines:
        key = line.partition("=")[0]
        if key in remaining and not line.lstrip().startswith("#"):
            out.append(f"{key}={remaining.pop(key)}")
        else:
            out.append(line)
    out.extend(f"{k}={v}" for k, v in remaining.items())
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")
