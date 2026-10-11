"""nvidia-smi rows become metrics; unreadable fields are left out, not zeroed."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import exporter  # noqa: E402


def test_a_gb10_row_drops_its_na_memory_and_keeps_the_rest():
    row = "0, NVIDIA GB10, 37, [N/A], [N/A], [N/A], 51, 23.45, [Not Supported], 2405"
    lines = exporter.parse_smi(row)
    assert 'kloudchat_gpu_utilization_percent{gpu="0",name="NVIDIA GB10"} 37' in lines
    assert 'kloudchat_gpu_temperature_celsius{gpu="0",name="NVIDIA GB10"} 51' in lines
    assert 'kloudchat_gpu_power_watts{gpu="0",name="NVIDIA GB10"} 23.45' in lines
    assert not any("memory_used" in line or "power_limit" in line for line in lines)


def test_a_discrete_card_reports_memory_in_bytes():
    row = "1, NVIDIA RTX PRO 6000, 90, 40, 81920, 97887, 70, 400.1, 600.00, 2100"
    lines = exporter.parse_smi(row)
    assert f'kloudchat_gpu_memory_used_bytes{{gpu="1",name="NVIDIA RTX PRO 6000"}} {81920 * 1048576:g}' in lines


def test_a_short_row_is_ignored():
    assert exporter.parse_smi("0, GB10, 37") == []


def test_meminfo_and_stat():
    mem = exporter.parse_meminfo("MemTotal:  1000 kB\nMemFree: 10 kB\nMemAvailable:  400 kB\n")
    assert mem == ["kloudchat_node_memory_total_bytes 1024000",
                   "kloudchat_node_memory_available_bytes 409600"]
    cpu = exporter.parse_stat("cpu  100 0 100 700 100 0 0 0 0 0\ncpu0 1 1 1 1 1\n")
    assert cpu == ['kloudchat_node_cpu_seconds_total{mode="all"} 10',
                   'kloudchat_node_cpu_seconds_total{mode="idle"} 8']


def test_the_node_name_comes_from_the_host_not_the_container(tmp_path, monkeypatch):
    host = tmp_path / "hostname"
    host.write_text("gx10-ffe1\n")
    monkeypatch.delenv("NODE_NAME", raising=False)
    monkeypatch.setattr(exporter, "HOSTNAME_FILE", str(host))
    assert exporter.node_name() == "gx10-ffe1"
    monkeypatch.setenv("NODE_NAME", "gpu-a")
    assert exporter.node_name() == "gpu-a"
    monkeypatch.delenv("NODE_NAME")
    monkeypatch.setattr(exporter, "HOSTNAME_FILE", str(tmp_path / "missing"))
    assert exporter.node_name() == exporter.socket.gethostname()
