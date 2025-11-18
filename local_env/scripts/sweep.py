#!/usr/bin/env python3
"""
Extended sweep script for collecting startup times under a wide range of
resource limits and JVM heap settings.

This script explores a larger set of CPU core limits, container memory
limits and Java heap sizes to generate a more varied dataset for
training machine‑learning or reinforcement‑learning models.  It
performs the following steps for each valid combination:

1. Shuts down any running containers and clears volumes.
2. Starts the Docker Compose stack (`app` and `sidecar` services)
   while passing JVM arguments to configure the heap size.
3. Applies CPU and memory limits to the `app` container via
   `docker update`.
4. Spawns a standalone `sidecar` container, labelling the
   Prometheus metric with the current CPU, memory and heap values.
5. Waits for the application to become ready on its health endpoint.
6. Queries Prometheus for the `app_startup_seconds` metric.
7. Records the configuration and measured start‑up time to a CSV file.

Only configurations where the heap size does not exceed the container
memory limit are considered valid; invalid combinations are skipped.
Adjust the CPU, memory and heap lists below to generate as large and
varied a dataset as your hardware permits.
"""

import csv
import os
import subprocess
import sys
import time
from typing import Dict, List, Tuple

try:
    import requests  # type: ignore
except ImportError:
    print("The 'requests' library is required. Install it via 'pip install requests'.", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configuration values

# A broader set of CPU limits (cores) to test.  Feel free to add finer
# increments or extend the range, but ensure your host can allocate these
# CPU shares.  Values are strings because they are passed directly to
# docker update.
CPU_LIST: List[str] = [
    "0.5",
    "0.75",
    "1.0",
    "1.25",
    "1.5",
    "2.0",
    "2.5",
    "3.0",
]

# A wider selection of memory limits.  You can combine megabyte (M) and
# gigabyte (G) units.  When adding values, ensure they do not exceed the
# physical memory of your Docker host.
MEM_LIST: List[str] = [
    "512M",
    "768M",
    "1G",
    "1.5G",
    "2G",
    "3G",
    "4G",
]

# A varied list of heap sizes applied to both -Xms and -Xmx.  The script
# will automatically skip any configuration where the heap size is larger
# than the container memory limit.  You may include fractional gigabytes
# for finer granularity.
HEAP_LIST: List[str] = [
    "256M",
    "384M",
    "512M",
    "768M",
    "1G",
    "1.25G",
    "1.5G",
    "2G",
    "2.5G",
]

# Names of the application and sidecar containers as defined in your
# docker-compose.yml.  If you use different names, adjust these.
APP_CONTAINER = "app"
SIDECAR_CONTAINER = "sidecar"

# Health check URLs (host and container versions)
HEALTH_URL = "http://localhost:9080/health/ready"
HEALTH_URL_CONTAINER = "http://app:9080/health/ready"

# Prometheus endpoint for querying metrics
PROMETHEUS_QUERY_URL = "http://localhost:9090/api/v1/query"

# Timeouts
READINESS_TIMEOUT = 300  # seconds to wait for app readiness
PROMETHEUS_TIMEOUT = 180  # seconds to wait for Prometheus scrape


def parse_mem_to_mb(value: str) -> float:
    """Convert a memory spec (e.g. '512M', '1G', '1.25G') to megabytes."""
    cleaned = value.strip().upper()
    if cleaned.endswith("M"):
        return float(cleaned[:-1])
    if cleaned.endswith("G"):
        return float(cleaned[:-1]) * 1024.0
    raise ValueError(f"Memory value must end with 'M' or 'G': {value}")


def run_cmd(cmd: List[str], env: Dict[str, str] | None = None, capture: bool = False) -> Tuple[int, str, str]:
    """Run a subprocess command and return (returncode, stdout, stderr)."""
    env_vars = os.environ.copy()
    if env:
        env_vars.update(env)
    result = subprocess.run(cmd, check=False, text=True, capture_output=capture, env=env_vars)
    return result.returncode, result.stdout if capture else "", result.stderr if capture else ""


def get_default_network() -> str:
    """Return the first docker network ending with '_default' created by docker-compose."""
    rc, out, _ = run_cmd(["docker", "network", "ls", "--format", "{{.Name}}"], capture=True)
    if rc != 0:
        print("Error listing docker networks", file=sys.stderr)
        sys.exit(1)
    for name in out.strip().splitlines():
        if name.endswith("_default"):
            return name
    return "project_default"


def wait_for_readiness(url: str, timeout: int = READINESS_TIMEOUT) -> bool:
    """Wait until the given URL returns HTTP 200 or the timeout expires."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(url, timeout=2)
            if resp.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def query_prometheus(cpus: str, memory: str, heap: str, timeout: int = PROMETHEUS_TIMEOUT) -> float:
    """
    Retrieve the app_startup_seconds metric for the given labels.  Raises
    RuntimeError if Prometheus does not return a value within the timeout.
    """
    query = f'app_startup_seconds{{app="acmeair",cpus="{cpus}",memory="{memory}",heap="{heap}"}}'
    params = {"query": query}
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(PROMETHEUS_QUERY_URL, params=params, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("status") == "success":
                    result = data.get("data", {}).get("result", [])
                    if result:
                        return float(result[0]["value"][1])
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError(f"Metric not found for cpus={cpus}, memory={memory}, heap={heap}")


def stop_stack() -> None:
    """Stop the compose stack and remove volumes; delete any sidecar container."""
    run_cmd(["docker", "compose", "down", "-v"])
    run_cmd(["docker", "rm", "-f", SIDECAR_CONTAINER])


def start_stack(jvm_args: str) -> None:
    """Start the compose stack, passing JVM arguments via JVM_ARGS env var."""
    run_cmd(["docker", "compose", "up", "-d"], env={"JVM_ARGS": jvm_args})


def configure_app_resources(cpus: str, memory: str) -> None:
    """Apply CPU and memory limits to the application container."""
    mem_lower = memory.lower()
    rc, _, err = run_cmd(["docker", "update", "--cpus", cpus, "--memory", mem_lower, APP_CONTAINER])
    if rc != 0:
        print(f"Warning: failed to update app resources: {err.strip()}")


def restart_sidecar(cpus: str, memory: str, heap: str, network: str) -> None:
    """Start a new sidecar container with labelled environment variables."""
    run_cmd(["docker", "rm", "-f", SIDECAR_CONTAINER])
    image_name = "project-sidecar"
    run_cmd([
        "docker", "run", "-d",
        "--name", SIDECAR_CONTAINER,
        "-p", "9100:9100",
        "-e", f"APP_NAME=acmeair",
        "-e", f"HEALTH_URL={HEALTH_URL_CONTAINER}",
        "-e", f"CPUS={cpus}",
        "-e", f"CPU_LIMIT={cpus}",
        "-e", f"MEMORY={memory}",
        "-e", f"MEM_LIMIT={memory}",
        "-e", f"HEAP={heap}",
        "--network", network,
        f"{image_name}:latest",
    ])


def main() -> None:
    network = get_default_network()
    print(f"Detected docker network: {network}")
    results: List[Tuple[str, str, str, float]] = []

    for cpus in CPU_LIST:
        for memory in MEM_LIST:
            for heap in HEAP_LIST:
                # Skip configurations where heap > memory
                try:
                    if parse_mem_to_mb(heap) > parse_mem_to_mb(memory):
                        print(f"Skipping invalid combination: heap {heap} > memory {memory}")
                        continue
                except ValueError as e:
                    print(f"Invalid memory format: {e}")
                    continue

                print(f"\n---\nTesting: cpus={cpus}, memory={memory}, heap={heap}")
                # Reset environment
                stop_stack()
                # Build JVM args
                jvm_args = f"-Xms{heap} -Xmx{heap}"
                # Bring up stack
                start_stack(jvm_args)
                # Configure CPU/memory
                configure_app_resources(cpus, memory)
                # Start sidecar with labels
                restart_sidecar(cpus, memory, heap, network)
                # Wait for readiness
                if not wait_for_readiness(HEALTH_URL, READINESS_TIMEOUT):
                    print("Application did not become ready in time; skipping")
                    continue
                # Allow Prometheus time to scrape
                time.sleep(10)
                # Query metric
                try:
                    value = query_prometheus(cpus, memory, heap)
                    print(f"Startup time (s): {value}")
                    results.append((cpus, memory, heap, value))
                except RuntimeError as e:
                    print(str(e))
                finally:
                    stop_stack()

    # Write results
    csv_file = "startup_data.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cpus", "memory", "heap", "startup_seconds"])
        for row in results:
            writer.writerow(row)
    print(f"Completed. Wrote {len(results)} measurements to {csv_file}.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user.")