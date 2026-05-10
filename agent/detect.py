import logging
import re
import subprocess

from common.schemas import MacSpecs

logger = logging.getLogger(__name__)

BANDWIDTH_MAP = {
    "Apple M1": 68,
    "Apple M1 Pro": 200,
    "Apple M1 Max": 400,
    "Apple M1 Ultra": 800,
    "Apple M2": 100,
    "Apple M2 Pro": 200,
    "Apple M2 Max": 400,
    "Apple M2 Ultra": 800,
    "Apple M3": 100,
    "Apple M3 Pro": 150,
    "Apple M3 Max": 400,
    "Apple M4": 120,
    "Apple M4 Pro": 273,
    "Apple M4 Max": 546,
}

DEFAULT_BANDWIDTH_GBS = 100


def detect_chip_name() -> str:
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Fallback: parse system_profiler
    try:
        result = subprocess.run(
            ["system_profiler", "SPHardwareDataType"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if "Chip" in line and ":" in line:
                    return line.split(":", 1)[1].strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    raise RuntimeError("Could not detect chip name")


def detect_memory_total_gb() -> float:
    result = subprocess.run(
        ["sysctl", "-n", "hw.memsize"],
        capture_output=True, text=True, timeout=5,
    )
    if result.returncode != 0:
        raise RuntimeError("Could not detect total memory")
    return int(result.stdout.strip()) / (1024 ** 3)


def detect_memory_free_gb() -> float:
    result = subprocess.run(
        ["vm_stat"],
        capture_output=True, text=True, timeout=5,
    )
    if result.returncode != 0:
        raise RuntimeError("Could not run vm_stat")

    output = result.stdout

    # Parse page size
    page_size = 16384  # default for Apple Silicon
    page_size_match = re.search(r"page size of (\d+) bytes", output)
    if page_size_match:
        page_size = int(page_size_match.group(1))

    def parse_pages(label: str) -> int:
        match = re.search(rf"^{label}:\s+(\d+)\.", output, re.MULTILINE)
        return int(match.group(1)) if match else 0

    free_pages = parse_pages("Pages free")
    inactive_pages = parse_pages("Pages inactive")

    free_bytes = (free_pages + inactive_pages) * page_size
    return free_bytes / (1024 ** 3)


def detect_gpu_cores() -> int:
    try:
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            match = re.search(r"Total Number of Cores:\s+(\d+)", result.stdout)
            if match:
                return int(match.group(1))
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    logger.warning("Could not detect GPU core count, defaulting to 0")
    return 0


def detect_bandwidth(chip: str) -> float:
    bandwidth = BANDWIDTH_MAP.get(chip)
    if bandwidth is None:
        logger.warning(
            "Unknown chip %r, defaulting to %d GB/s", chip, DEFAULT_BANDWIDTH_GBS
        )
        return DEFAULT_BANDWIDTH_GBS
    return bandwidth


def detect_hardware() -> MacSpecs:
    chip = detect_chip_name()
    return MacSpecs(
        chip=chip,
        memory_total_gb=round(detect_memory_total_gb(), 1),
        memory_free_gb=round(detect_memory_free_gb(), 1),
        memory_bandwidth_gbs=detect_bandwidth(chip),
        gpu_cores=detect_gpu_cores(),
    )
