"""Scheduler command line interface.

    python3 -m scheduler inventory          probe results per node
    python3 -m scheduler plan               compute placement, change nothing
    python3 -m scheduler apply [-y]         apply the computed placement
    python3 -m scheduler catalog            models.yaml as TSV, for the shell scripts

Nodes and models come from ``VLLM_NODES`` and ``VLLM_MODELS`` in .env, and can be
overridden in place with ``--hosts`` and ``--models``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

from scheduler import applier, endpoints, inventory, model_metadata, planner, registry
from scheduler.types import GB, ModelMetadata, NodeSpec

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODELS_YAML = Path(__file__).resolve().parent / "models.yaml"
#: This deployment's overrides of models.yaml (untracked; see README "Site overrides")
SITE_YAML = Path(__file__).resolve().parent / "site.yaml"
ENV_FILE = PROJECT_DIR / ".env"


def _env(key: str, default: str = "") -> str:
    try:
        for raw in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            if k.strip() == key:
                return v.strip()
    except FileNotFoundError:
        pass
    return default


def _csv(value: str) -> list[str]:
    return [p.strip() for p in value.split(",") if p.strip()]


def _remote_workdir() -> str:
    """Repository path on a node (``KLOUDCHAT_REMOTE_DIR``), the directory lib.sh rsyncs to."""
    return (
        os.environ.get("KLOUDCHAT_REMOTE_DIR")
        or _env("KLOUDCHAT_REMOTE_DIR")
        or "KloudChat-LLM"
    )


def _resolve_hosts(arg: Optional[str]) -> dict[str, str]:
    """Node id to SSH target from ``--hosts`` or VLLM_NODES; exits on a repeated target."""
    hosts = _csv(arg) if arg else _csv(_env("VLLM_NODES"))
    try:
        return inventory.node_ids_for_hosts(hosts)
    except ValueError as exc:
        sys.exit(str(exc))


def _held_nodes(hosts: dict[str, str]) -> frozenset[str]:
    """Node ids in VLLM_NODES_HOLD: probed and listed, never placed on or acted on.

    An entry is an SSH target as written in VLLM_NODES, its bare host, or a node
    id. Exits on an entry VLLM_NODES does not list, or on a held head node.
    """
    held: set[str] = set()
    for entry in _csv(_env("VLLM_NODES_HOLD")):
        bare = entry.split("@")[-1]
        match = [nid for nid, host in hosts.items()
                 if entry in (nid, host) or bare == host.split("@")[-1]]
        if not match:
            sys.exit(f"VLLM_NODES_HOLD: {entry} is not in VLLM_NODES")
        held.update(match)
    head = _head_node(hosts)
    if head is not None and head in held:
        sys.exit(f"VLLM_NODES_HOLD: {head} is the head node (the first in VLLM_NODES)")
    return frozenset(held)


def _kept(probes, specs) -> dict[str, set[str]]:
    """Node id to the services a VLLM_ENDPOINTS entry points at: hand-run, never stopped."""
    by_port = {s.port: s.service for s in specs}
    out: dict[str, set[str]] = {}
    for url in endpoints.urls(_env("VLLM_ENDPOINTS")):
        host, _, port = url.split("//")[-1].partition(":")
        port = port.split("/")[0]
        for probe in probes:
            if probe.spec.hostname.split("@")[-1] == host and port.isdigit() \
                    and int(port) in by_port:
                out.setdefault(probe.spec.node_id, set()).add(by_port[int(port)])
    return out


def _head_node(hosts: dict[str, str]) -> Optional[str]:
    """Head node id: the first target in VLLM_NODES. None for a single-node cluster."""
    return next(iter(hosts), None) if len(hosts) > 1 else None


def _load_specs(arg: Optional[str]) -> list[registry.ModelSpec]:
    wanted = _csv(arg) if arg else _csv(_env("VLLM_MODELS"))
    specs = registry.load(MODELS_YAML, only=wanted or None, site=SITE_YAML)
    return [s for s in specs if not s.manual]


def _bind(specs, probes, models_root: str):
    """Bind config.json and checkpoint size onto each spec; unprobeable models are dropped with a reason."""
    alive = [p for p in probes if p.alive]
    bound, failed = [], []
    for spec in specs:
        metadata: Optional[ModelMetadata] = None
        weight: Optional[int] = spec.weight_override
        if spec.is_omni:
            # No config.json, no KV: the weight is declared, or measured on a node
            for probe in alive:
                if weight is None and probe.spec.checkpoints is not None \
                        and spec.dir in probe.spec.checkpoints:
                    weight = model_metadata.measure_weight_bytes(
                        probe.spec.hostname, f"{models_root}/{spec.dir}")
            if not weight:
                failed.append((spec.id, "its weight size is neither declared nor measurable"))
                continue
            bound.append(spec.bind(model_metadata.image_metadata(spec.id, weight), 0))
            continue
        for probe in alive:
            host = probe.spec.hostname
            path = f"{models_root}/{spec.dir}"
            try:
                if weight is None:
                    weight = model_metadata.measure_weight_bytes(host, path)
                metadata = model_metadata.fetch(
                    spec.hf_repo,
                    probe_host=host,
                    probe_path=f"{path}/config.json",
                    on_disk_weight_bytes=weight,
                )
                break
            except FileNotFoundError:
                continue
        if metadata is None:
            failed.append((spec.id, "could not read the checkpoint's config.json"))
            continue
        native = spec.ctx_target_override or metadata.native_ctx
        if not native:
            failed.append((spec.id, "config.json declares no context length"))
            continue
        bound.append(spec.bind(metadata, native))
    return bound, failed


def _probe(hosts: dict[str, str], specs) -> list:
    if not hosts:
        print("VLLM_NODES is empty — there is nothing to probe", file=sys.stderr)
        return []
    return inventory.probe_cluster(
        hosts,
        services=[s.service for s in specs],
        models_root=_env("VLLM_MODELS_ROOT", "/var/lib/vllm/models"),
    )


def cmd_inventory(args) -> int:
    specs = _load_specs(args.models)
    hosts = _resolve_hosts(args.hosts)
    held = _held_nodes(hosts)
    probes = _probe(hosts, specs)
    if not probes:
        return 1
    print(f"{'NODE':<10} {'STATE':<9} {'GPU':<10} {'ARCH':<7} {'TOTAL':>9} {'USABLE':>9}  RUNNING")
    for p in probes:
        s: NodeSpec = p.spec
        running = ", ".join(sorted(p.running_services)) or "-"
        state = "held" if s.node_id in held else "alive" if p.alive else "no answer"
        print(f"{s.node_id:<10} {state:<9} "
              f"{s.gpu_class:<10} {s.arch or '?':<7} "
              f"{s.total_vram_bytes / GB:>7.1f}G {s.planner_vram_bytes / GB:>7.1f}G  {running}")
        for err in p.raw_errors:
            print(f"{'':<10} └ {err}")
    return 0


def _deployed(probes, specs) -> dict[str, frozenset[str]]:
    """Model id to the node ids running its container (from ``docker ps``)."""
    by_service = {s.service: s.id for s in specs}
    out: dict[str, set[str]] = {}
    for probe in probes:
        for container in probe.running_containers:
            model_id = by_service.get(container)
            if model_id:
                out.setdefault(model_id, set()).add(probe.spec.node_id)
    return {k: frozenset(v) for k, v in out.items()}


def _build_plan(args):
    specs = _load_specs(args.models)
    hosts = _resolve_hosts(args.hosts)
    held = _held_nodes(hosts)
    probes = _probe(hosts, specs)
    if not probes:
        return None, None, None, None, held
    bound, failed = _bind(specs, probes, _env("VLLM_MODELS_ROOT", "/var/lib/vllm/models"))
    nodes = [p.spec for p in probes if p.alive and p.spec.node_id not in held]
    head = _head_node(hosts)
    # Served elsewhere already (VLLM_ENDPOINTS); transcription is hand-run through WHISPER_URLS
    external = {
        mid: n for mid, n in endpoints.count_instances(_env("VLLM_ENDPOINTS")).items()
        if any(s.id == mid and s.runner != "transcription" for s in specs)
    }
    result = planner.plan(
        bound, nodes,
        deployed=_deployed(probes, bound),
        head=head,
        external=external,
    )
    image_ids = {s.id for s in specs if s.is_omni}
    for model_id, reason in failed:
        if not external.get(model_id):
            result.delegations.append(
                planner.Delegation(model_id, reason, model_id not in image_ids))
    for node_id in sorted(held):
        result.notes.append(f"{node_id}: held (VLLM_NODES_HOLD) — nothing placed, nothing stopped")
    for probe in probes:
        if not probe.alive and probe.spec.node_id not in held:
            why = "; ".join(probe.raw_errors) or "no answer"
            result.notes.append(f"{probe.spec.node_id}: not answering ({why}) — left out")
    return result, probes, bound, head, held


def _print_plan(result, head: Optional[str] = None) -> None:
    if result.placements:
        print("Placements" + (f"   head node: {head}" if head else ""))
        for p in sorted(result.placements, key=lambda x: (x.node_id, x.model_id)):
            # Cards and TP only when not the single-card default
            extra = ""
            if p.devices and p.devices != (0,):
                extra += f"  gpu {','.join(str(d) for d in p.devices)}"
            if p.tp > 1:
                extra += f"  TP{p.tp}"
            if p.ctx == 0:
                print(f"  {p.node_id:<8} {p.model_id:<20} image"
                      f"{'':<17}{p.charge / GB:>5.1f} GiB{extra}")
                continue
            ctx = f"{p.ctx // 1024}K" if p.ctx >= 1024 else str(p.ctx)
            print(f"  {p.node_id:<8} {p.model_id:<20} ctx {ctx:>5}  "
                  f"util {p.gpu_util:<5.2f} {p.charge / GB:>5.1f} GiB{extra}")
    else:
        print("no model could be placed")
    delegated = [d for d in result.delegations if d.fallback]
    if delegated:
        print("\nDelegated to OpenRouter")
        for d in delegated:
            print(f"  {d.model_id:<20} {d.reason}")
    unserved = [d for d in result.delegations if not d.fallback]
    if unserved:
        print("\nNot placed (no external fallback)")
        for d in unserved:
            print(f"  {d.model_id:<20} {d.reason}")
    for note in result.notes:
        print(f"\nNote: {note}")


def cmd_plan(args) -> int:
    result, _, _, head, _ = _build_plan(args)
    if result is None:
        return 1
    _print_plan(result, head)
    return 0


def cmd_apply(args) -> int:
    result, probes, bound, head, held = _build_plan(args)
    if result is None:
        return 1
    _print_plan(result, head)

    # Held nodes and nodes that did not answer in full are left exactly as they are
    acting = [p for p in probes if p.alive and p.spec.node_id not in held]
    current = {p.spec.node_id: set(p.running_containers) for p in acting}
    # Current node options, so an unchanged plan recreates nothing
    node_env = {
        p.spec.node_id: inventory.read_env(p.spec.hostname, f"{_remote_workdir()}/.env")
        for p in acting
    }
    unread = sorted(nid for nid, env in node_env.items() if env is None)
    if unread:
        print(f"\ncould not read {_remote_workdir()}/.env on {', '.join(unread)} — "
              "nothing applied", file=sys.stderr)
        return 1
    change = applier.compute_diff(
        target=result, current=current, specs=bound,
        nodes=[p.spec for p in acting],
        layout=applier.RemoteLayout(workdir=_remote_workdir()),
        local_env_path=str(ENV_FILE),
        node_env=node_env,
        kept=_kept(acting, bound),
    )
    if change.is_empty:
        print("\nNo changes — the cluster already matches the plan")
        return 0

    print("\nChanges to apply")
    for action in change.actions:
        print(f"  [{action.kind:<9}] {action.node_id:<8} {action.description}")
    for key, value in change.local_env.items():
        print(f"  [env      ] local    {key}={value or '(cleared)'}")

    if not args.yes:
        try:
            answer = input("\nApply? [y/N] ").strip().lower()
        except EOFError:
            answer = ""
        if answer not in ("y", "yes"):
            print("cancelled")
            return 1

    failures = applier.apply(change, local_env_path=str(ENV_FILE))
    for failure in failures:
        print(f"failed: {failure}", file=sys.stderr)
    if failures:
        print(f"\n{len(failures)} action(s) failed — the rest were applied", file=sys.stderr)
        return 1
    print("\nApplied")
    return 0


def cmd_catalog(args) -> int:
    """One line per model: id, api, env_prefix, openrouter, timeout,
    max_inflight, presence_penalty, temperature; "-" for an empty field."""
    del args
    for s in registry.load(MODELS_YAML):
        fields = (
            s.id, s.api, s.env_prefix, s.openrouter, s.timeout or "",
            s.max_inflight, "" if s.presence_penalty is None else s.presence_penalty,
            "" if s.temperature is None else s.temperature,
        )
        print("\t".join(str(f) if f != "" else "-" for f in fields))
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="scheduler", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("catalog", help="models.yaml as TSV").set_defaults(func=cmd_catalog)

    for name, handler, help_text in (
        ("inventory", cmd_inventory, "probe results per node"),
        ("plan", cmd_plan, "compute placement, change nothing"),
        ("apply", cmd_apply, "apply the computed placement"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--hosts", help="SSH target CSV (default: VLLM_NODES in .env)")
        p.add_argument("--models", help="model id CSV (default: VLLM_MODELS in .env)")
        if name == "apply":
            p.add_argument("-y", "--yes", action="store_true", help="apply without confirmation")
        p.set_defaults(func=handler)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
