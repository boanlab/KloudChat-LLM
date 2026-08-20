"""Scheduler command line interface.

    python3 -m scheduler inventory          probe results per node
    python3 -m scheduler plan               compute placement, change nothing
    python3 -m scheduler apply [-y]         apply the computed placement

Nodes and models come from ``NODES_VLLM`` and ``VLLM_MODELS`` in .env, and can be
overridden in place with ``--hosts`` and ``--models``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

from scheduler import applier, inventory, model_metadata, planner, registry
from scheduler.types import GB, ModelMetadata, NodeSpec

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODELS_YAML = Path(__file__).resolve().parent / "models.yaml"
ENV_FILE = PROJECT_DIR / ".env"

#: Node capacity held by the resident transcription backend, subtracted before
#: packing. Charged only to nodes whose probe answered.
WHISPER_RESERVE_BYTES = 6 * GB


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
    """Repository path on a node — the directory scripts/lib.sh rsyncs to.

    Must match ``KLOUDCHAT_REMOTE_DIR``: running compose somewhere other than the
    synced checkout applies stale options. ``NODE_WORKDIR`` is an alias.
    """
    return (
        os.environ.get("KLOUDCHAT_REMOTE_DIR")
        or _env("KLOUDCHAT_REMOTE_DIR")
        or _env("NODE_WORKDIR")
        or "KloudChat-LLM"
    )


def _resolve_hosts(arg: Optional[str]) -> dict[str, str]:
    hosts = _csv(arg) if arg else _csv(_env("NODES_VLLM"))
    return {inventory.node_id_from_host(h): h for h in hosts}


def _load_specs(arg: Optional[str]) -> list[registry.ModelSpec]:
    wanted = _csv(arg) if arg else _csv(_env("VLLM_MODELS"))
    return registry.load(MODELS_YAML, only=wanted or None)


def _bind(specs, probes, models_root: str):
    """config.json and checkpoint size bound onto each ModelSpec.

    Models whose probe failed are dropped with a reason: without metadata the KV
    arithmetic is meaningless.
    """
    alive = [p for p in probes if p.alive]
    bound, failed = [], []
    for spec in specs:
        metadata: Optional[ModelMetadata] = None
        weight: Optional[int] = spec.weight_override
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


def _reservations(probes) -> dict[str, int]:
    """Bytes reserved per node: subtract the transcription backend where it runs."""
    return {
        p.spec.node_id: WHISPER_RESERVE_BYTES
        for p in probes if p.whisper_running
    }


def _services(specs) -> dict[str, int]:
    return {s.service: s.port for s in specs}


def _probe(hosts: dict[str, str], specs) -> list:
    if not hosts:
        print("NODES_VLLM is empty — there is nothing to probe", file=sys.stderr)
        return []
    return inventory.probe_cluster(
        hosts,
        services=_services(specs),
        # So placement can tell "this node has no room" from "this node does not
        # have the weights" — the second is not a capacity problem and no
        # amount of VRAM fixes it.
        models_root=_env("VLLM_MODELS_ROOT", "/var/lib/vllm/models"),
    )


# ── commands ────────────────────────────────────────────────────────────


def cmd_inventory(args) -> int:
    specs = _load_specs(args.models)
    probes = _probe(_resolve_hosts(args.hosts), specs)
    if not probes:
        return 1
    print(f"{'NODE':<10} {'STATE':<9} {'GPU':<10} {'ARCH':<7} {'TOTAL':>9} {'USABLE':>9}  RUNNING")
    for p in probes:
        s: NodeSpec = p.spec
        running = ", ".join(sorted(w.container_name for w in p.running_workloads)) or "-"
        print(f"{s.node_id:<10} {'alive' if p.alive else 'no answer':<9} "
              f"{s.gpu_class:<10} {s.arch or '?':<7} "
              f"{s.total_vram_bytes / GB:>7.1f}G {s.planner_vram_bytes / GB:>7.1f}G  {running}")
        for err in p.raw_errors:
            print(f"{'':<10} └ {err}")
    return 0


def _deployed(probes, specs) -> dict[str, frozenset[str]]:
    """Model id to the node ids already running its container.

    Read from `docker ps`, not from the .env: a node whose container died is not
    a home to stay at.
    """
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
    probes = _probe(hosts, specs)
    if not probes:
        return None, None, None
    bound, failed = _bind(specs, probes, _env("VLLM_MODELS_ROOT", "/var/lib/vllm/models"))
    nodes = [p.spec for p in probes if p.alive]
    result = planner.plan(
        bound, nodes,
        reserved=_reservations(probes),
        replicas=args.replicas,
        deployed=_deployed(probes, bound),
    )
    for model_id, reason in failed:
        result.delegations.append(planner.Delegation(model_id, reason))
    return result, probes, bound


def _print_plan(result) -> None:
    if result.placements:
        print("Placements")
        for p in sorted(result.placements, key=lambda x: (x.node_id, x.model_id)):
            # Cards and TP width only when they say something: on the common
            # single-card node "gpu 0, TP 1" is noise in every row.
            extra = ""
            if p.devices and p.devices != (0,):
                extra += f"  gpu {','.join(str(d) for d in p.devices)}"
            if p.tp > 1:
                extra += f"  TP{p.tp}"
            # Whisper's 448 is a real context, and 448 // 1024 printed "0K"
            ctx = f"{p.ctx // 1024}K" if p.ctx >= 1024 else str(p.ctx)
            print(f"  {p.node_id:<8} {p.model_id:<18} ctx {ctx:>5}  "
                  f"util {p.gpu_util:<5.2f} {p.charge / GB:>5.1f} GiB{extra}")
    else:
        print("no model could be placed")
    if result.delegations:
        print("\nDelegated to OpenRouter")
        for d in result.delegations:
            print(f"  {d.model_id:<18} {d.reason}")
    for note in result.notes:
        print(f"\nNote: {note}")


def cmd_plan(args) -> int:
    result, _, _ = _build_plan(args)
    if result is None:
        return 1
    _print_plan(result)
    return 0


def cmd_apply(args) -> int:
    result, probes, bound = _build_plan(args)
    if result is None:
        return 1
    _print_plan(result)

    current = {p.spec.node_id: set(p.running_containers) for p in probes}
    change = applier.compute_diff(
        target=result, current=current, specs=bound,
        nodes=[p.spec for p in probes if p.alive],
        layout=applier.RemoteLayout(workdir=_remote_workdir()),
        local_env_path=str(ENV_FILE),
        # Probed backends only — an empty result routes STT to OpenRouter
        stt_hosts=[p.spec.hostname for p in probes if p.whisper_running],
        # The whole catalogue, so a model dropped from VLLM_MODELS has its URL
        # cleared rather than left pointing at a container that is now stopped
        known=registry.load(MODELS_YAML),
        # Each node's current options, so an apply that changes nothing does
        # nothing instead of reloading every model's weights
        node_env={
            p.spec.node_id: inventory.read_env(
                p.spec.hostname, f"{_remote_workdir()}/.env"
            )
            for p in probes if p.alive
        },
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


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="scheduler", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    for name, handler, help_text in (
        ("inventory", cmd_inventory, "probe results per node"),
        ("plan", cmd_plan, "compute placement, change nothing"),
        ("apply", cmd_apply, "apply the computed placement"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--hosts", help="SSH target CSV (default: NODES_VLLM in .env)")
        p.add_argument("--models", help="model id CSV (default: VLLM_MODELS in .env)")
        p.add_argument("--replicas", type=int, default=None,
                       help="cap instances per model (default: fill spare "
                            "capacity by priority; 1 disables replication)")
        if name == "apply":
            p.add_argument("-y", "--yes", action="store_true", help="apply without confirmation")
        p.set_defaults(func=handler)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
