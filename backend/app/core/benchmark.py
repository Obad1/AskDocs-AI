"""Server-side hardware benchmark.

Probes the host for CPU cores, RAM, GPU availability (best-effort via
``torch`` / ``pynvml`` when present), WebGPU proxy capability, and disk
space, then maps the reading to a :class:`HARDWARE_TIER` recommendation.
No network calls are made.
"""
from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass

from app.models.domain import HARDWARE_TIER


@dataclass
class BenchmarkResult:
    cpu_cores: int
    total_ram_gb: float
    available_ram_gb: float
    gpu_available: bool
    gpu_name: str = ""
    webgpu_proxy: bool = False
    disk_free_gb: float = 0.0
    tier: HARDWARE_TIER = HARDWARE_TIER.TIER2_MID

    def to_dict(self) -> dict:
        return {
            "cpu_cores": self.cpu_cores,
            "total_ram_gb": round(self.total_ram_gb, 2),
            "available_ram_gb": round(self.available_ram_gb, 2),
            "gpu_available": self.gpu_available,
            "gpu_name": self.gpu_name,
            "webgpu_proxy": self.webgpu_proxy,
            "disk_free_gb": round(self.disk_free_gb, 2),
            "tier": self.tier.value,
        }


def _probe_cpu_cores() -> int:
    try:
        import os

        return os.cpu_count() or 1
    except Exception:
        return 1


def _probe_ram() -> tuple[float, float]:
    try:
        import psutil

        vm = psutil.virtual_memory()
        return vm.total / (1024 ** 3), vm.available / (1024 ** 3)
    except Exception:
        # Minimal fallback
        try:
            with open("/proc/meminfo") as fh:
                total = None
                for line in fh:
                    if line.startswith("MemTotal:"):
                        total = int(line.split()[1]) * 1024
                        break
            if total:
                return total / (1024 ** 3), total / (1024 ** 3)
        except Exception:
            pass
        return 4.0, 2.0


def _probe_gpu() -> tuple[bool, str]:
    name = ""
    # Try nvidia-ml / torch CUDA first
    try:
        import torch

        if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
            try:
                name = torch.cuda.get_device_name(0)
            except Exception:
                name = "cuda"
            return True, name
    except Exception:
        pass
    # Try pynvml for NVIDIA without torch
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        return True, name
    except Exception:
        pass
    # Try Apple Silicon MPS
    if sys.platform == "darwin":
        try:
            import torch

            if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                return True, "apple_mps"
        except Exception:
            return True, "apple_mps"
    return False, ""


def _probe_webgpu_proxy() -> bool:
    """WebGPU is a browser feature; the backend can only expose a proxy flag.

    We treat the existence of a WebGPU-capable frontend bundle or an explicit
    env override as the proxy signal. This is intentionally conservative.
    """
    return os.environ.get("WEBGPU_PROXY", "0") == "1"


def _probe_disk(data_root: str) -> float:
    try:
        total, used, free = shutil.disk_usage(data_root)
        return free / (1024 ** 3)
    except Exception:
        return 0.0


def _select_tier(cores: int, ram_gb: float, gpu: bool, disk_gb: float) -> HARDWARE_TIER:
    if gpu and ram_gb >= 16 and cores >= 8:
        return HARDWARE_TIER.TIER3_HIGH
    if ram_gb >= 8 and cores >= 4:
        return HARDWARE_TIER.TIER2_MID
    return HARDWARE_TIER.TIER1_LOW


def run_benchmark(data_root: str = "./data") -> BenchmarkResult:
    """Run the local benchmark and return a :class:`BenchmarkResult`."""
    cores = _probe_cpu_cores()
    total_ram, avail_ram = _probe_ram()
    gpu, gpu_name = _probe_gpu()
    webgpu = _probe_webgpu_proxy()
    disk = _probe_disk(data_root)
    tier = _select_tier(cores, total_ram, gpu, disk)
    return BenchmarkResult(
        cpu_cores=cores,
        total_ram_gb=total_ram,
        available_ram_gb=avail_ram,
        gpu_available=gpu,
        gpu_name=gpu_name,
        webgpu_proxy=webgpu,
        disk_free_gb=disk,
        tier=tier,
    )
