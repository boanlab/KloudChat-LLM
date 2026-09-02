"""Placement applied to nodes.

Per placement:
    (a) ``{env_prefix}_{MAX_LEN,GPU_UTIL}`` (and TP/DEVICES where set) in the node's .env
    (b) compose services started, stopped or recreated on the node
    (c) ``{env_prefix}_URL`` in the orchestrator's .env, read by gen-litellm-config.sh

A ChangePlan is built first and executed after confirmation. Re-applying an
unchanged plan is a no-op.
"""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Sequence

from scheduler.planner import Placement, Plan
from scheduler.registry import ModelSpec
from scheduler.types import NodeSpec

#: Service-name prefix the scheduler manages; other containers are never stopped
MANAGED_SERVICE_PREFIX = "vllm-"

#: Transcription model. Its URL CSV is written as ``WHISPER_URLS`` (read by
#: whisper-shim) instead of ``{env_prefix}_URL``; empty routes STT to OpenRouter.
STT_MODEL_ID = "whisper-large-v3"


@dataclass(frozen=True)
class RemoteLayout:
    """Where compose runs on a node."""

    workdir: str = "KloudChat-LLM"
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
    #: Orchestrator .env values to write
    local_env: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.actions and not self.local_env


def _env_set(path: str, key: str, value: str) -> str:
    """Shell expression setting one .env line, appending when absent."""
    q_path = shlex.quote(path)
    line = shlex.quote(f"{key}={value}")
    return (
        f"touch {q_path} && "
        f"if grep -q \"^{key}=\" {q_path}; then "
        f"sed -i \"s|^{key}=.*|{key}={value}|\" {q_path}; "
        f"else printf '%s\\n' {line} >> {q_path}; fi"
    )


def _read_env_keys(path: str, keys: Iterable[str]) -> dict[str, str]:
    """Values of ``keys`` in the local .env; missing keys read empty."""
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
              known: Sequence[ModelSpec] = ()) -> dict[str, str]:
    """URL CSV per model; unplaced models (including ``known`` ones outside the
    deployment) get an empty value so no stale route survives."""
    host_of = {n.node_id: n.hostname.split("@")[-1] for n in nodes}
    by_id = {s.id: s for s in specs}
    urls: dict[str, set[str]] = {
        f"{s.env_prefix}_URL": set() for s in (list(known) + list(specs))
    }
    for p in target.placements:
        spec = by_id.get(p.model_id)
        if spec is None:
            continue
        host = host_of.get(p.node_id, p.node_id)
        urls[f"{spec.env_prefix}_URL"].add(f"http://{host}:{spec.port}")
    out = {k: ",".join(sorted(v)) for k, v in urls.items()}

    stt = next(
        (s for s in list(known) + list(specs) if s.id == STT_MODEL_ID), None
    )
    out["WHISPER_URLS"] = out.pop(f"{stt.env_prefix}_URL", "") if stt else ""
    return out


def compute_diff(
    *,
    target: Plan,
    current: dict[str, set[str]],
    specs: Sequence[ModelSpec],
    nodes: Sequence[NodeSpec],
    layout: RemoteLayout = RemoteLayout(),
    local_env_path: Optional[str] = None,
    known: Sequence[ModelSpec] = (),
    node_env: Optional[dict[str, dict[str, str]]] = None,
) -> ChangePlan:
    """Changes that take ``current`` to ``target``.

    Args:
        current: node id to the compose services running there.
        local_env_path: the orchestrator's .env; None skips the URL update.
        node_env: node id to its current .env values. Given, unchanged services
            are left alone; omitted, every placed service is recreated.
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
        # Managed services only
        have = {
            s for s in current.get(node_id, set())
            if s.startswith(MANAGED_SERVICE_PREFIX) or s in want
        }

        # (a) .env options, before any service starts
        here = node_env.get(node_id) if node_env is not None else None
        restated: set[str] = set()
        for p in sorted(placements, key=lambda x: x.model_id):
            spec = by_id.get(p.model_id)
            if spec is None:
                continue
            options = [
                (f"{spec.env_prefix}_MAX_LEN", str(p.ctx)),
                (f"{spec.env_prefix}_GPU_UTIL", f"{p.gpu_util:.2f}"),
            ]
            # TP 1 is the compose default: written only to undo a sharded node,
            # since any new key costs a recreate
            tp_key = f"{spec.env_prefix}_TP"
            if p.tp > 1 or (here or {}).get(tp_key) not in (None, "", "1"):
                options.append((tp_key, str(p.tp)))

            # NVIDIA_VISIBLE_DEVICES, multi-card nodes only (CUDA_VISIBLE_DEVICES
            # fails engine init on GB10)
            dev_key = f"{spec.env_prefix}_DEVICES"
            devices = ",".join(str(d) for d in p.devices)
            node = next((n for n in nodes if n.node_id == node_id), None)
            if devices and node is not None and node.gpu_count > 1:
                options.append((dev_key, devices))
            elif (here or {}).get(dev_key):
                options.append((dev_key, devices))

            for key, value in options:
                if here is not None and here.get(key) == value:
                    continue
                restated.add(spec.service)
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
        # (b) Recreate only where an option moved — a recreate reloads the weights
        for service in sorted(want & have):
            if node_env is not None and service not in restated:
                continue
            change.actions.append(NodeAction(
                node_id, host, "recreate", f"recreate {service}",
                f"{cd} && {compose} up -d --force-recreate {shlex.quote(service)}",
            ))

    if local_env_path:
        desired = _url_csvs(target, specs, nodes, known)
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
    """Execute the changes; returns failures. A failing node does not stop the others."""
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
