#!/usr/bin/env python3
"""
dual_controller.py

Use a trained DQN + PPO pair to tune startup time of a *new* application.

- DQN: chooses which resource to adjust (CPU, Memory, Heap).
- PPO: outputs a 3-D continuous action [Δ_cpu, Δ_mem, Δ_heap] in [-1, 1].
        We only apply the component corresponding to the resource
        selected by DQN.

At each step:
  1. Build observation from current configuration and last startup time.
  2. DQN picks a resource index: 0=CPU, 1=Memory, 2=Heap.
  3. PPO picks deltas for all three; we use only the chosen resource.
  4. Apply the change, respecting CPU/memory limits and heap <= memory.
  5. Restart Docker stack with new JVM heap, update app resource limits,
     restart sidecar, wait for readiness, query Prometheus for
     app_startup_seconds.
  6. Reward = baseline_startup - measured_startup.
  7. Repeat for N steps, remembering the best configuration observed.

Usage example:
    python dual_controller.py \
        --dqn dqn_newapp_finetuned.zip \
        --ppo ppo_newapp_finetuned.zip \
        --baseline 12.0 \
        --steps 10

You can also constrain max CPU / memory:
        --cpu-min 0.5 --cpu-max 4.0 --cpu-limit 2.0 \
        --mem-min 512M --mem-max 2G  --mem-limit 1G
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
from stable_baselines3 import DQN, PPO  # type: ignore

# ---------------------------------------------------------------------------
# Docker / Prometheus configuration (adapt if your new app differs)
# ---------------------------------------------------------------------------

APP_CONTAINER = "app"
SIDECAR_CONTAINER = "sidecar"

# Host-side health endpoint
HEALTH_URL_HOST = "http://localhost:9080/health/ready"
# In-container URL used by the sidecar
HEALTH_URL_CONTAINER = "http://app:9080/health/ready"

PROMETHEUS_QUERY_URL = "http://localhost:9090/api/v1/query"

RESOURCE_NAMES = ["CPU", "Memory", "Heap"]


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def docker_cmd(
    cmd: List[str], env: Optional[dict] = None, capture: bool = False
) -> Tuple[int, str, str]:
    """Run a docker-related command and return (rc, stdout, stderr)."""
    env_vars = os.environ.copy()
    if env:
        env_vars.update(env)
    try:
        result = subprocess.run(
            cmd,
            text=True,
            capture_output=capture,
            check=False,
            env=env_vars,
        )
        return result.returncode, result.stdout if capture else "", result.stderr if capture else ""
    except FileNotFoundError:
        print(f"[ERROR] Command not found: {cmd[0]}", file=sys.stderr)
        sys.exit(1)


def parse_memory_to_mb(mem: str) -> int:
    """Convert '512M', '768M', '1G', '2G', '1024' etc. to MB as int."""
    mem = str(mem).strip().upper()
    if mem.endswith("G"):
        return int(float(mem[:-1]) * 1024)
    if mem.endswith("M"):
        return int(float(mem[:-1]))
    # assume plain MB
    return int(float(mem))


def mb_to_str(mb: int) -> str:
    """Pretty-print MB as '768M' or '1G' etc."""
    if mb % 1024 == 0:
        return f"{mb // 1024}G"
    return f"{mb}M"


def get_default_network() -> str:
    """Return first docker network ending with '_default', or 'project_default'."""
    rc, out, err = docker_cmd(["docker", "network", "ls", "--format", "{{.Name}}"], capture=True)
    if rc != 0:
        print(f"[WARN] Failed to list docker networks: {err.strip()}", file=sys.stderr)
        return "project_default"
    for line in out.strip().splitlines():
        if line.endswith("_default"):
            return line
    return "project_default"


def wait_for_readiness(url: str, timeout: int = 300) -> bool:
    """Wait until the health endpoint returns HTTP 200 or timeout reached."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(url, timeout=3)
            if resp.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def query_prometheus_startup(cpus: str, memory: str, heap: str, timeout: int = 120) -> float:
    """
    Query Prometheus for app_startup_seconds{app="acmeair",cpus=...,memory=...,heap=...}.
    Adapt the 'app' label if your new app uses a different value.
    """
    query = (
        f'app_startup_seconds{{app="acmeair",'
        f'cpus="{cpus}",memory="{memory}",heap="{heap}"}}'
    )
    params = {"query": query}
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(PROMETHEUS_QUERY_URL, params=params, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("status") == "success":
                    results = data.get("data", {}).get("result", [])
                    if results:
                        value = results[0]["value"][1]
                        return float(value)
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError(f"No Prometheus metric for cpus={cpus}, memory={memory}, heap={heap}")


def stop_stack():
    """Stop docker compose stack and remove sidecar container."""
    print("[STACK] Stopping stack…")
    docker_cmd(["docker", "compose", "down", "-v"])
    docker_cmd(["docker", "rm", "-f", SIDECAR_CONTAINER])


def start_stack(jvm_args: str):
    """Start docker compose stack with JVM_ARGS env var for heap size."""
    print(f"[STACK] Starting stack with JVM_ARGS='{jvm_args}'…")
    docker_cmd(["docker", "compose", "up", "-d"], env={"JVM_ARGS": jvm_args})


def configure_app_resources(cpus: float, memory_mb: int):
    """Apply CPU and memory limits to the app container."""
    mem_str = mb_to_str(memory_mb)
    mem_lower = mem_str.lower()
    print(f"[STACK] Updating app resources: cpus={cpus}, memory={mem_lower}")
    rc, _, err = docker_cmd(
        ["docker", "update", "--cpus", str(cpus), "--memory", mem_lower, APP_CONTAINER]
    )
    if rc != 0:
        print(f"[WARN] docker update failed: {err.strip()}")


def restart_sidecar(cpus: float, memory_mb: int, heap_mb: int, network: str):
    """Restart sidecar container with labels for this configuration."""
    mem_str = mb_to_str(memory_mb)
    heap_str = mb_to_str(heap_mb)
    print(f"[STACK] Restarting sidecar: cpus={cpus}, memory={mem_str}, heap={heap_str}")
    docker_cmd(["docker", "rm", "-f", SIDECAR_CONTAINER])
    cmd = [
        "docker", "run", "-d",
        "--name", SIDECAR_CONTAINER,
        "-p", "9100:9100",
        "-e", "APP_NAME=acmeair",  # change label if needed
        "-e", f"HEALTH_URL={HEALTH_URL_CONTAINER}",
        "-e", f"CPUS={cpus}",
        "-e", f"MEMORY={mem_str}",
        "-e", f"HEAP={heap_str}",
        "--network", network,
        "project-sidecar:latest",
    ]
    rc, _, err = docker_cmd(cmd)
    if rc != 0:
        print(f"[WARN] Failed to start sidecar: {err.strip()}")


# ---------------------------------------------------------------------------
# Environment class (not a Gym env; just our control loop wrapper)
# ---------------------------------------------------------------------------

@dataclass
class ControllerConfig:
    cpu_min: float
    cpu_max: float
    cpu_limit: float

    mem_min_mb: int
    mem_max_mb: int
    mem_limit_mb: int

    heap_min_mb: int
    heap_max_mb: int  # logical max heap (usually <= mem_limit_mb)

    baseline_startup: float

    cpu_step: float = 0.5    # how much one normalized step can change CPU
    mem_step_mb: int = 256   # how much one step can change memory
    heap_step_mb: int = 128  # how much one step can change heap

    invalid_reward: float = -100.0


class NewAppDockerEnv:
    """
    Simple wrapper that holds the current configuration and interacts with Docker/Prometheus.

    Not a Gym env; we only use it for manual control with pre-trained models.
    """

    def __init__(self, config: ControllerConfig):
        self.cfg = config
        self.network = get_default_network()

        self.current_cpu: float = config.cpu_limit
        self.current_mem_mb: int = config.mem_limit_mb
        self.current_heap_mb: int = min(config.heap_max_mb, self.current_mem_mb // 2)
        self.last_startup: float = config.baseline_startup

        self.best_startup: float = float("inf")
        self.best_config: Dict[str, float] = {}

    # ------------- Observation -------------

    def _build_obs(self) -> np.ndarray:
        """Build normalized observation vector for RL models.

        NOTE: Must match the observation used during training:
              [cpu_norm, mem_norm, heap_norm, baseline_norm]  -> shape (4,)
        """
        cfg = self.cfg

        cpu_norm = (self.current_cpu - cfg.cpu_min) / (cfg.cpu_max - cfg.cpu_min + 1e-9)
        mem_norm = (self.current_mem_mb - cfg.mem_min_mb) / (cfg.mem_max_mb - cfg.mem_min_mb + 1e-9)
        heap_norm = (self.current_heap_mb - cfg.heap_min_mb) / (cfg.heap_max_mb - cfg.heap_min_mb + 1e-9)

        # Same as in OfflineStartupEnv / RealStartupEnv
        baseline_norm = min(cfg.baseline_startup / 60.0, 1.0)

        return np.array(
            [cpu_norm, mem_norm, heap_norm, baseline_norm],
            dtype=np.float32,
        )

    # ------------- Public API -------------

    def reset(self) -> np.ndarray:
        """Reset to a default configuration (within limits) and return first observation."""
        cfg = self.cfg
        self.current_cpu = min(max(1.0, cfg.cpu_min), cfg.cpu_limit)
        self.current_mem_mb = min(max(cfg.mem_min_mb * 2, cfg.mem_min_mb), cfg.mem_limit_mb)
        self.current_heap_mb = min(cfg.heap_max_mb, self.current_mem_mb // 2)
        self.last_startup = cfg.baseline_startup
        self.best_startup = float("inf")
        self.best_config = {}
        return self._build_obs()

    def step(self, resource_idx: int, delta_vec: np.ndarray) -> Tuple[np.ndarray, float, float, Dict]:
        """
        Apply the chosen adjustment, restart the stack, measure startup time.

        :param resource_idx: 0=CPU, 1=Memory, 2=Heap (chosen by DQN).
        :param delta_vec: ndarray shape (3,) in [-1,1], output of PPO.
        :returns: (next_obs, reward, startup_time, info_dict)
        """
        cfg = self.cfg

        # 1) Compute new configuration from delta
        delta_cpu, delta_mem, delta_heap = float(delta_vec[0]), float(delta_vec[1]), float(delta_vec[2])

        new_cpu = self.current_cpu
        new_mem_mb = self.current_mem_mb
        new_heap_mb = self.current_heap_mb

        if resource_idx == 0:
            # Adjust CPU only
            new_cpu = self.current_cpu + delta_cpu * cfg.cpu_step
        elif resource_idx == 1:
            # Adjust Memory only
            new_mem_mb = self.current_mem_mb + int(delta_mem * cfg.mem_step_mb)
        elif resource_idx == 2:
            # Adjust Heap only
            new_heap_mb = self.current_heap_mb + int(delta_heap * cfg.heap_step_mb)
        else:
            print(f"[WARN] Unknown resource index {resource_idx}, nothing changed.")

        # Clamp to allowed ranges and limits
        new_cpu = min(max(new_cpu, cfg.cpu_min), cfg.cpu_limit)
        new_mem_mb = min(max(new_mem_mb, cfg.mem_min_mb), cfg.mem_limit_mb)
        new_heap_mb = min(max(new_heap_mb, cfg.heap_min_mb), cfg.heap_max_mb)
        # Safety: heap cannot exceed memory
        new_heap_mb = min(new_heap_mb, new_mem_mb)

        # 2) Apply to Docker stack and measure startup time
        try:
            stop_stack()
            jvm_args = f"-Xms{mb_to_str(new_heap_mb)} -Xmx{mb_to_str(new_heap_mb)}"
            start_stack(jvm_args)
            configure_app_resources(new_cpu, new_mem_mb)
            restart_sidecar(new_cpu, new_mem_mb, new_heap_mb, self.network)

            ready = wait_for_readiness(HEALTH_URL_HOST, timeout=300)
            if not ready:
                print("[ENV] App did not become ready in time.")
                reward = cfg.invalid_reward
                startup = float("nan")
            else:
                print("[ENV] Waiting for Prometheus scrape…")
                time.sleep(10)
                startup = query_prometheus_startup(
                    cpus=str(new_cpu),
                    memory=mb_to_str(new_mem_mb),
                    heap=mb_to_str(new_heap_mb),
                )
                print(f"[ENV] Measured startup: {startup:.3f} seconds")
                reward = cfg.baseline_startup - startup

        except Exception as e:
            print(f"[ENV] Measurement failed: {e}")
            reward = cfg.invalid_reward
            startup = float("nan")

        # 3) Update current state
        self.current_cpu = new_cpu
        self.current_mem_mb = new_mem_mb
        self.current_heap_mb = new_heap_mb
        if not np.isnan(startup):
            self.last_startup = startup

            # Track best configuration
            if startup < self.best_startup:
                self.best_startup = startup
                self.best_config = {
                    "cpu": new_cpu,
                    "memory_mb": new_mem_mb,
                    "heap_mb": new_heap_mb,
                }

        next_obs = self._build_obs()
        info = {
            "startup_seconds": startup,
            "config": {
                "cpu": new_cpu,
                "memory_mb": new_mem_mb,
                "heap_mb": new_heap_mb,
            },
            "resource_index": resource_idx,
            "reward": reward,
        }
        return next_obs, reward, startup, info


# ---------------------------------------------------------------------------
# Main control loop: DQN + PPO working together
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Use DQN+PPO to tune startup time on a new app.")
    parser.add_argument("--dqn", type=str, required=True, help="Path to trained DQN model .zip")
    parser.add_argument("--ppo", type=str, required=True, help="Path to trained PPO model .zip")
    parser.add_argument("--baseline", type=float, required=True, help="Baseline startup time in seconds")
    parser.add_argument("--steps", type=int, default=10, help="Number of online adjustment steps")

    # Resource ranges and limits (adapt to your new app)
    parser.add_argument("--cpu-min", type=float, default=0.5)
    parser.add_argument("--cpu-max", type=float, default=4.0)
    parser.add_argument("--cpu-limit", type=float, default=2.0)

    parser.add_argument("--mem-min", type=str, default="512M")
    parser.add_argument("--mem-max", type=str, default="2G")
    parser.add_argument("--mem-limit", type=str, default="1G")

    parser.add_argument("--heap-min", type=str, default="256M")
    parser.add_argument("--heap-max", type=str, default="2G")

    args = parser.parse_args()

    # Build controller config
    cfg = ControllerConfig(
        cpu_min=args.cpu_min,
        cpu_max=args.cpu_max,
        cpu_limit=args.cpu_limit,
        mem_min_mb=parse_memory_to_mb(args.mem_min),
        mem_max_mb=parse_memory_to_mb(args.mem_max),
        mem_limit_mb=parse_memory_to_mb(args.mem_limit),
        heap_min_mb=parse_memory_to_mb(args.heap_min),
        heap_max_mb=parse_memory_to_mb(args.heap_max),
        baseline_startup=args.baseline,
    )

    print("[INFO] Controller configuration:")
    print(f"  CPU:  min={cfg.cpu_min}, max={cfg.cpu_max}, limit={cfg.cpu_limit}")
    print(f"  MEM:  min={mb_to_str(cfg.mem_min_mb)}, max={mb_to_str(cfg.mem_max_mb)}, "
          f"limit={mb_to_str(cfg.mem_limit_mb)}")
    print(f"  HEAP: min={mb_to_str(cfg.heap_min_mb)}, max={mb_to_str(cfg.heap_max_mb)}")
    print(f"  Baseline startup: {cfg.baseline_startup:.3f} s")

    # Load models
    print(f"[INFO] Loading DQN model from {args.dqn}")
    dqn_model = DQN.load(args.dqn)
    print(f"[INFO] Loading PPO model from {args.ppo}")
    ppo_model = PPO.load(args.ppo)

    # Environment
    env = NewAppDockerEnv(cfg)
    obs = env.reset()

    print("\n[RUN] Starting online adaptation loop…\n")
    for step in range(1, args.steps + 1):
        # 1) DQN chooses which resource to adjust
        resource_action, _ = dqn_model.predict(obs, deterministic=True)
        resource_idx = int(resource_action)

        # 2) PPO chooses continuous deltas (we use only the chosen dimension)
        ppo_action, _ = ppo_model.predict(obs, deterministic=False)
        ppo_action = np.asarray(ppo_action, dtype=np.float32).reshape(-1)
        if ppo_action.shape[0] != 3:
            raise ValueError(
                f"PPO action must be 3-D (Δ_cpu, Δ_mem, Δ_heap), got shape {ppo_action.shape}"
            )

        # 3) Apply step in environment
        obs, reward, startup, info = env.step(resource_idx, ppo_action)

        print(f"[STEP {step}] "
              f"resource={RESOURCE_NAMES[resource_idx]}, "
              f"reward={reward:.3f}, "
              f"startup={startup:.3f} s, "
              f"config={info['config']}")

    print("\n[RESULT] Best configuration found during this run:")
    if env.best_config:
        print(f"  CPU:   {env.best_config['cpu']}")
        print(f"  Mem:   {mb_to_str(env.best_config['memory_mb'])}")
        print(f"  Heap:  {mb_to_str(env.best_config['heap_mb'])}")
        print(f"  Startup time: {env.best_startup:.3f} s")
    else:
        print("  No valid measurement was obtained.")


if __name__ == "__main__":
    main()
