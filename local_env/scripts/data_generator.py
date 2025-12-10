import csv
import os
import subprocess
import sys
import time
from typing import Dict, List, Tuple
import requests

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

MEM_LIST: List[str] = [
    "512M",
    "768M",
    "1G",
    "1.5G",
    "2G",
    "3G",
    "4G",
]

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

APP_CONTAINER = "app"
SIDECAR_CONTAINER = "sidecar"

HEALTH_URL = "http://localhost:9080/local_app/"
HEALTH_URL_CONTAINER = "http://app:9080/local_app/"
PROMETHEUS_QUERY_URL = "http://localhost:9090/api/v1/query"

READINESS_TIMEOUT = 300
PROMETHEUS_TIMEOUT = 180


def format_mem(mem):
    mem = str(mem).strip().upper()
    if mem.endswith("G"):
        return int(float(mem[:-1]) * 1024)
    if mem.endswith("M"):
        return int(float(mem[:-1]))
    return int(float(mem))

def run_cmd(cmd, env = None, capture = False):
    env_vars = os.environ.copy()
    if env:
        env_vars.update(env)
    result = subprocess.run(cmd, check=False, text=True, capture_output=capture, env=env_vars)
    return_code = result.returncode
    if capture:
        stdout_value = result.stdout
    else:
        stdout_value = ""
    if capture:
        stderr_value = result.stderr
    else:
        stderr_value = ""
    return return_code, stdout_value, stderr_value

def get_default_network():
    rc, out, _ = run_cmd(["docker", "network", "ls", "--format", "{{.Name}}"], capture=True)
    if rc != 0:
        sys.exit(1)
    for name in out.strip().splitlines():
        if name.endswith("_default"):
            return name
    return "project_default"

def wait_for_readiness(url, timeout = READINESS_TIMEOUT):
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


def query_prometheus(cpus, memory, heap, timeout = PROMETHEUS_TIMEOUT):
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
    raise RuntimeError

def stop_stack():
    run_cmd(["docker", "compose", "down", "-v"])
    run_cmd(["docker", "rm", "-f", SIDECAR_CONTAINER])

def start_stack(jvm_args):
    run_cmd(["docker", "compose", "up", "-d"], env={"JVM_ARGS": jvm_args})

def configure_app_resources(cpus, memory):
    mem_lower = memory.lower()
    rc, _, err = run_cmd(["docker", "update", "--cpus", cpus, "--memory", mem_lower, APP_CONTAINER])
    if rc != 0:
        print(f"failed to update app resources: {err.strip()}")

def restart_sidecar(cpus, memory, heap, network):
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

def main():
    network = get_default_network()
    results = []
    for cpus in CPU_LIST:
        for memory in MEM_LIST:
            for heap in HEAP_LIST:
                if format_mem(heap) > format_mem(memory):
                    print(f"Skipping invalid combination: heap {heap} > memory {memory}")
                print(f"Testing: cpus={cpus}, memory={memory}, heap={heap}")
                stop_stack()
                jvm_args = f"-Xms{heap} -Xmx{heap}"
                start_stack(jvm_args)
                configure_app_resources(cpus, memory)

                restart_sidecar(cpus, memory, heap, network)
                if not wait_for_readiness(HEALTH_URL, READINESS_TIMEOUT):
                    print("Application did not become ready in time")
                    continue
                time.sleep(10)

                try:
                    value = query_prometheus(cpus, memory, heap)
                    print(f"Startup time: {value}")
                    results.append((cpus, memory, heap, value))
                except RuntimeError as e:
                    print(str(e))
                finally:
                    stop_stack()

    csv_file = "startup_data.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cpus", "memory", "heap", "startup_seconds"])
        for row in results:
            writer.writerow(row)
    print(f"Wrote {len(results)} measurements to {csv_file}.")


if __name__ == "__main__":
    main()