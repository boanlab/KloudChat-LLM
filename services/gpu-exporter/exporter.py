"""GPU and node metrics of one GPU node, in the Prometheus text format.

GET /metrics runs `nvidia-smi` once per request and reads /proc. Standard
library only. KloudChat reads it through the gateway's /metrics/gpu-* paths.

On GB10 (unified memory) nvidia-smi reports memory as [N/A]; the node's
MemTotal/MemAvailable are the model memory there, so both are exported.
"""

from __future__ import annotations

import os
import socket
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("PORT", "9835"))
PROC = os.environ.get("PROC_ROOT", "/proc")
SMI_TIMEOUT = float(os.environ.get("SMI_TIMEOUT_SEC", "5"))
# The node's own /etc/hostname, mounted read-only; a container's hostname is its id
HOSTNAME_FILE = os.environ.get("HOSTNAME_FILE", "/host/etc/hostname")

#: nvidia-smi field → (metric, scale to base unit)
FIELDS: list[tuple[str, str | None, float]] = [
    ("index", None, 1),
    ("name", None, 1),
    ("utilization.gpu", "kloudchat_gpu_utilization_percent", 1),
    ("utilization.memory", "kloudchat_gpu_memory_utilization_percent", 1),
    ("memory.used", "kloudchat_gpu_memory_used_bytes", 1024 * 1024),
    ("memory.total", "kloudchat_gpu_memory_total_bytes", 1024 * 1024),
    ("temperature.gpu", "kloudchat_gpu_temperature_celsius", 1),
    ("power.draw", "kloudchat_gpu_power_watts", 1),
    ("power.limit", "kloudchat_gpu_power_limit_watts", 1),
    ("clocks.sm", "kloudchat_gpu_sm_clock_mhz", 1),
]


def _label(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


def _number(raw: str) -> float | None:
    """A reading, or None for [N/A], [Not Supported] and the like."""
    try:
        return float(raw.strip())
    except ValueError:
        return None


def parse_smi(csv: str) -> list[str]:
    """Metric lines for `nvidia-smi --query-gpu=... --format=csv,noheader,nounits`."""
    lines: list[str] = []
    for row in csv.splitlines():
        cells = [c.strip() for c in row.split(",")]
        if len(cells) != len(FIELDS):
            continue
        labels = f'gpu="{_label(cells[0])}",name="{_label(cells[1])}"'
        for (_, metric, scale), cell in zip(FIELDS, cells, strict=True):
            value = _number(cell) if metric else None
            if metric and value is not None:
                lines.append(f"{metric}{{{labels}}} {value * scale:g}")
    return lines


def parse_meminfo(text: str) -> list[str]:
    values: dict[str, int] = {}
    for row in text.splitlines():
        key, _, rest = row.partition(":")
        parts = rest.split()
        if parts and parts[0].isdigit():
            values[key] = int(parts[0]) * 1024
    out = []
    if "MemTotal" in values:
        out.append(f"kloudchat_node_memory_total_bytes {values['MemTotal']}")
    if "MemAvailable" in values:
        out.append(f"kloudchat_node_memory_available_bytes {values['MemAvailable']}")
    return out


def parse_stat(text: str) -> list[str]:
    """Cumulative CPU seconds (USER_HZ 100), total and idle+iowait."""
    for row in text.splitlines():
        if row.startswith("cpu "):
            ticks = [int(x) for x in row.split()[1:] if x.isdigit()]
            if len(ticks) < 5:
                return []
            idle = ticks[3] + ticks[4]
            return [
                f'kloudchat_node_cpu_seconds_total{{mode="all"}} {sum(ticks[:8]) / 100:g}',
                f'kloudchat_node_cpu_seconds_total{{mode="idle"}} {idle / 100:g}',
            ]
    return []


def _read(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    except OSError:
        return ""


def node_name() -> str:
    """NODE_NAME, else the mounted host /etc/hostname, else the container's own."""
    name = os.environ.get("NODE_NAME", "").strip() or _read(HOSTNAME_FILE).strip()
    return name or socket.gethostname()


def render() -> str:
    lines = [f'kloudchat_node_info{{hostname="{_label(node_name())}"}} 1']
    query = ",".join(field for field, _, _ in FIELDS)
    try:
        smi = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=SMI_TIMEOUT, check=False,
        )
        gpu = parse_smi(smi.stdout) if smi.returncode == 0 else []
    except (OSError, subprocess.TimeoutExpired):
        gpu = []
    lines.append(f"kloudchat_gpu_up {1 if gpu else 0}")
    lines += gpu
    lines += parse_meminfo(_read(f"{PROC}/meminfo"))
    lines += parse_stat(_read(f"{PROC}/stat"))
    load = _read(f"{PROC}/loadavg").split()
    if load:
        lines.append(f"kloudchat_node_load1 {load[0]}")
    lines.append(f"kloudchat_node_cpus {os.cpu_count() or 0}")
    return "\n".join(lines) + "\n"


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        if self.path.split("?", 1)[0] not in ("/metrics", "/health"):
            self.send_error(404)
            return
        body = (render() if self.path.startswith("/metrics") else "ok\n").encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args) -> None:
        pass


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
