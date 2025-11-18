#!/usr/bin/env python3
"""
Hierarchical RL training for AcmeAir startup tuning with resource limits.

Hierarchy:
    - Low-level: PPO selects a full configuration (cpus, memory, heap).
    - High-level: DQN decides which parameter(s) to change
        0: CPU only
        1: Memory only
        2: Heap only
        3: Memory + Heap

Pipeline:
    Phase A (offline, fast):
        1) Train PPO on OfflineStartupEnv (CSV only).
        2) Train DQN on HierarchicalStartupEnv (wraps OfflineStartupEnv + PPO).

    Phase B (online, realistic, expensive):
        3) Fine-tune PPO on RealStartupEnv (Docker+Prometheus).
        4) Fine-tune DQN on HierarchicalStartupEnv wrapping RealStartupEnv + PPO.

Outputs (in ../model_results):
    - model.zip         : final high-level DQN after offline+online training
    - reward_curves.png : reward trajectories (all phases)
    - loss_curves.png   : loss proxy trajectories (all phases)
    - startup_time.png  : startup time trajectories (all phases)

Reward normalization:
    - By default, rewards are normalized by baseline startup time:
        normalized_reward = (baseline - startup) / baseline
    - You can disable this with --no-normalize-reward.
"""

from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import DQN, PPO  # type: ignore
from stable_baselines3.common.callbacks import BaseCallback  # type: ignore


# ==============================================================================
# Docker / Prometheus configuration
# ==============================================================================

APP_CONTAINER = "app"
SIDECAR_CONTAINER = "sidecar"

HEALTH_URL_HOST = "http://localhost:9080/health/ready"      # from host
HEALTH_URL_CONTAINER = "http://app:9080/health/ready"       # from sidecar
PROMETHEUS_QUERY_URL = "http://localhost:9090/api/v1/query"

# Typical upper bound for startup seconds used for simple normalization of
# the baseline feature in observations.
DEFAULT_STARTUP_MAX_SECONDS = 60.0


# ==============================================================================
# Typed action representation
# ==============================================================================

@dataclass(frozen=True)
class Action:
    """Single configuration choice for the app."""
    cpus: str    # e.g. "0.5", "1.0"
    memory: str  # e.g. "512M", "768M", "1G"
    heap: str    # e.g. "256M", "512M", "1G"


# ==============================================================================
# Utility helpers
# ==============================================================================

def parse_memory_to_mb(mem: str) -> int:
    """Convert memory string like '512M', '768M', '1G', '2G' to MB."""
    mem = str(mem).strip().upper()
    if mem.endswith("G"):
        return int(float(mem[:-1]) * 1024)
    if mem.endswith("M"):
        return int(float(mem[:-1]))
    # Assume raw MB number
    return int(float(mem))


def docker_cmd(
    cmd: List[str],
    env: Optional[dict] = None,
    capture: bool = False
) -> Tuple[int, str, str]:
    """
    Run a docker-related command and return (return_code, stdout, stderr).

    capture=True is useful when we want to read error messages.
    """
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


def get_default_network() -> str:
    """
    Return the first Docker network ending with '_default'.

    If nothing is found, fall back to 'project_default'.
    """
    rc, out, err = docker_cmd(["docker", "network", "ls", "--format", "{{.Name}}"], capture=True)
    if rc != 0:
        print(f"[WARN] Failed to list docker networks: {err.strip()}", file=sys.stderr)
        return "project_default"
    for line in out.strip().splitlines():
        if line.endswith("_default"):
            return line
    return "project_default"


def wait_for_readiness(url: str, timeout: int = 300) -> bool:
    """Poll the given health URL until it returns HTTP 200 or timeout is reached."""
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
    Query Prometheus for app_startup_seconds with the given labels.

    Raises:
        RuntimeError if no metric is found within 'timeout' seconds.
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


def stop_stack() -> None:
    """
    Stop the Docker Compose stack and remove the sidecar container.

    This is called before each new configuration to make sure we start from
    a clean state.
    """
    print("[STACK] Stopping stack…")
    docker_cmd(["docker", "compose", "down", "-v"])
    docker_cmd(["docker", "rm", "-f", SIDECAR_CONTAINER])


def start_stack(jvm_args: str) -> None:
    """
    Start Docker Compose stack with JVM_ARGS set for heap size.

    jvm_args example: "-Xms512M -Xmx512M"
    """
    print(f"[STACK] Starting stack with JVM_ARGS='{jvm_args}'…")
    docker_cmd(["docker", "compose", "up", "-d"], env={"JVM_ARGS": jvm_args})


def configure_app_resources(cpus: str, memory: str) -> None:
    """
    Apply CPU and memory limits to the app container using 'docker update'.

    Note: memory and memory-swap are set together to avoid Docker errors.
    """
    mem_lower = memory.lower()  # docker likes 512m / 1g
    print(f"[STACK] Updating app resources: cpus={cpus}, memory={mem_lower}")

    rc, _, err = docker_cmd(
        [
            "docker", "update",
            "--cpus", cpus,
            "--memory", mem_lower,
            "--memory-swap", mem_lower,  # or a bit larger than mem_lower
            APP_CONTAINER,
        ],
        capture=True,
    )

    if rc != 0:
        print(f"[WARN] 'docker update' failed: {err.strip()}")


def restart_sidecar(cpus: str, memory: str, heap: str, network: str) -> None:
    """
    Restart sidecar container with labels for this configuration.

    The sidecar:
        - watches the app's health endpoint
        - computes startup time
        - exports Prometheus metrics.
    """
    print(f"[STACK] Restarting sidecar: cpus={cpus}, memory={memory}, heap={heap}")
    docker_cmd(["docker", "rm", "-f", SIDECAR_CONTAINER])
    cmd = [
        "docker", "run", "-d",
        "--name", SIDECAR_CONTAINER,
        "-p", "9100:9100",
        "-e", "APP_NAME=acmeair",
        "-e", f"HEALTH_URL={HEALTH_URL_CONTAINER}",
        "-e", f"CPUS={cpus}",
        "-e", f"MEMORY={memory}",
        "-e", f"HEAP={heap}",
        "--network", network,
        "project-sidecar:latest",
    ]
    rc, _, err = docker_cmd(cmd)
    if rc != 0:
        print(f"[WARN] Failed to start sidecar: {err.strip()}")


# ==============================================================================
# Offline environment (CSV-only, no Docker)
# ==============================================================================

class OfflineStartupEnv(gym.Env):
    """
    Gym environment that uses only CSV data (offline sweep results).

    Observation (normalized to [0,1]):
        [cpu_norm, mem_norm, heap_norm, baseline_norm]

    Action:
        Discrete index into the list of Actions (cpus, memory, heap).

    Reward (by default normalized):
        (baseline - sampled_startup_seconds) / baseline
    """

    metadata = {"render.modes": []}

    def __init__(
        self,
        actions: List[Action],
        startup_df: pd.DataFrame,
        baseline_startup: float,
        normalize_reward: bool = True,
    ):
        super().__init__()

        self.actions = actions
        self.baseline_startup = baseline_startup
        self.normalize_reward = normalize_reward

        heap_col = "heap"
        if heap_col not in startup_df.columns and "heap_size" in startup_df.columns:
            heap_col = "heap_size"

        # Map each action index to all recorded startup times for that config.
        self.action_to_times: Dict[int, List[float]] = {}
        for i, action in enumerate(actions):
            mask = (
                startup_df["cpus"].astype(str) == action.cpus
            ) & (
                startup_df["memory"].astype(str) == action.memory
            ) & (
                startup_df[heap_col].astype(str) == action.heap
            )
            times = startup_df.loc[mask, "startup_seconds"].dropna().tolist()
            self.action_to_times[i] = times

        # Pre-compute mins/maxs for simple min-max normalization of features.
        cpu_vals = [float(a.cpus) for a in actions]
        mem_vals = [parse_memory_to_mb(a.memory) for a in actions]
        heap_vals = [parse_memory_to_mb(a.heap) for a in actions]

        self.cpu_min, self.cpu_max = min(cpu_vals), max(cpu_vals)
        self.mem_min, self.mem_max = min(mem_vals), max(mem_vals)
        self.heap_min, self.heap_max = min(heap_vals), max(heap_vals)

        # Observation space: 4 normalized features.
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(4,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(len(actions))

    # ----- internal helpers ---------------------------------------------------

    def _obs_from_action(self, idx: int) -> np.ndarray:
        """Build a normalized observation for a given action index."""
        action = self.actions[idx]
        cpu = float(action.cpus)
        mem_mb = parse_memory_to_mb(action.memory)
        heap_mb = parse_memory_to_mb(action.heap)

        cpu_norm = (cpu - self.cpu_min) / (self.cpu_max - self.cpu_min + 1e-9)
        mem_norm = (mem_mb - self.mem_min) / (self.mem_max - self.mem_min + 1e-9)
        heap_norm = (heap_mb - self.heap_min) / (self.heap_max - self.heap_min + 1e-9)
        baseline_norm = min(self.baseline_startup / DEFAULT_STARTUP_MAX_SECONDS, 1.0)

        return np.array(
            [cpu_norm, mem_norm, heap_norm, baseline_norm],
            dtype=np.float32,
        )

    # ----- Gym API ------------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        idx = random.randrange(len(self.actions))
        obs = self._obs_from_action(idx)
        return obs, {}

    def step(self, action_idx: int):
        idx = int(action_idx)
        times = self.action_to_times.get(idx, [])

        if times:
            startup = random.choice(times)
        else:
            # No CSV data for this configuration: assume it's worse than baseline.
            startup = self.baseline_startup * 1.5

        raw_reward = self.baseline_startup - startup
        if self.normalize_reward:
            # Reward in roughly [-?, 1], dimensionless.
            reward = raw_reward / max(self.baseline_startup, 1e-6)
        else:
            reward = raw_reward

        obs = self._obs_from_action(idx)
        terminated = True
        truncated = False
        info = {"startup_seconds": startup}
        return obs, reward, terminated, truncated, info


# ==============================================================================
# Online environment (real Docker + Prometheus)
# ==============================================================================

class RealStartupEnv(gym.Env):
    """
    Gym environment that uses the real Docker stack + Prometheus.

    Observation:
        Same 4 normalized features as OfflineStartupEnv.

    Action:
        Choose a (cpus, memory, heap) combination.

    Reward:
        Same formula as offline:
           normalized_reward = (baseline - measured_startup) / baseline
        or raw (baseline - startup) if normalization disabled.
    """

    INVALID_REWARD = -100.0
    metadata = {"render.modes": []}

    def __init__(
        self,
        actions: List[Action],
        baseline_startup: float,
        normalize_reward: bool = True,
    ):
        super().__init__()

        self.actions = actions
        self.baseline_startup = baseline_startup
        self.normalize_reward = normalize_reward
        self.network = get_default_network()

        cpu_vals = [float(a.cpus) for a in actions]
        mem_vals = [parse_memory_to_mb(a.memory) for a in actions]
        heap_vals = [parse_memory_to_mb(a.heap) for a in actions]

        self.cpu_min, self.cpu_max = min(cpu_vals), max(cpu_vals)
        self.mem_min, self.mem_max = min(mem_vals), max(mem_vals)
        self.heap_min, self.heap_max = min(heap_vals), max(heap_vals)

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(4,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(len(actions))

    # ----- internal helpers ---------------------------------------------------

    def _obs_from_action(self, idx: int) -> np.ndarray:
        action = self.actions[idx]
        cpu = float(action.cpus)
        mem_mb = parse_memory_to_mb(action.memory)
        heap_mb = parse_memory_to_mb(action.heap)

        cpu_norm = (cpu - self.cpu_min) / (self.cpu_max - self.cpu_min + 1e-9)
        mem_norm = (mem_mb - self.mem_min) / (self.mem_max - self.mem_min + 1e-9)
        heap_norm = (heap_mb - self.heap_min) / (self.heap_max - self.heap_min + 1e-9)
        baseline_norm = min(self.baseline_startup / DEFAULT_STARTUP_MAX_SECONDS, 1.0)

        return np.array(
            [cpu_norm, mem_norm, heap_norm, baseline_norm],
            dtype=np.float32,
        )

    # ----- Gym API ------------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        idx = random.randrange(len(self.actions))
        obs = self._obs_from_action(idx)
        return obs, {}

    def step(self, action_idx: int):
        idx = int(action_idx)
        action = self.actions[idx]
        cpus, memory, heap = action.cpus, action.memory, action.heap

        # Simple validity check: heap should not exceed container memory.
        if parse_memory_to_mb(heap) > parse_memory_to_mb(memory):
            print(f"[ENV] Invalid config heap={heap} > memory={memory}")
            obs = self._obs_from_action(idx)
            return obs, self.INVALID_REWARD, True, False, {"startup_seconds": None}

        try:
            # Restart whole stack with new JVM heap
            stop_stack()
            jvm_args = f"-Xms{heap} -Xmx{heap}"
            start_stack(jvm_args)

            # Apply CPU and memory limits to the app container
            configure_app_resources(cpus, memory)

            # Restart sidecar to attach new labels
            restart_sidecar(cpus, memory, heap, self.network)

            # Wait for the app to become healthy
            ready = wait_for_readiness(HEALTH_URL_HOST, timeout=300)
            if not ready:
                print(f"[ENV] App not ready for {action}")
                obs = self._obs_from_action(idx)
                return obs, self.INVALID_REWARD, True, False, {"startup_seconds": None}

            print("[ENV] Waiting for Prometheus scrape…")
            time.sleep(10)

            startup = query_prometheus_startup(cpus, memory, heap)
            print(f"[ENV] Measured startup: {startup:.3f} seconds")

            raw_reward = self.baseline_startup - startup
            if self.normalize_reward:
                reward = raw_reward / max(self.baseline_startup, 1e-6)
            else:
                reward = raw_reward

            obs = self._obs_from_action(idx)
            info = {"startup_seconds": startup}
        except Exception as exc:
            print(f"[ENV] Measurement failed for {action}: {exc}")
            obs = self._obs_from_action(idx)
            reward = self.INVALID_REWARD
            info = {"startup_seconds": None}

        return obs, reward, True, False, info


# ==============================================================================
# Hierarchical environment: high-level DQN + low-level PPO
# ==============================================================================

class HierarchicalStartupEnv(gym.Env):
    """
    High-level environment for DQN that sits on top of a base env
    (OfflineStartupEnv or RealStartupEnv) and a trained PPO model.

    - PPO (low-level) proposes a full configuration (cpus, memory, heap).
    - DQN (high-level) decides WHICH parameter(s) to adopt from PPO:
          0: update CPU only
          1: update Memory only
          2: update Heap only
          3: update Memory + Heap

    The resulting configuration index is then passed to the base env .step().
    """

    metadata = {"render.modes": []}

    def __init__(
        self,
        actions: List[Action],
        base_env: gym.Env,   # OfflineStartupEnv or RealStartupEnv
        ppo_model: PPO,
    ):
        super().__init__()
        self.actions = actions
        self.base_env = base_env
        self.ppo_model = ppo_model

        # High-level action space: which parameter(s) to change
        # 0: CPU, 1: Memory, 2: Heap, 3: Memory+Heap
        self.action_space = spaces.Discrete(4)
        # Observation space is the same as the base env
        self.observation_space = base_env.observation_space

        # Map config (cpus, memory, heap) -> index in actions list
        self.config_to_idx: Dict[Tuple[str, str, str], int] = {}
        for idx, a in enumerate(actions):
            key = (str(a.cpus), str(a.memory), str(a.heap))
            self.config_to_idx[key] = idx

        # Keep track of the current configuration index
        self.current_idx: int = 0

    # ---------------- internal helpers ----------------

    def _obs_from_idx(self, idx: int) -> np.ndarray:
        # Reuse the base env's normalization logic
        return self.base_env._obs_from_action(idx)

    def _apply_high_level_action(self, hi_action: int, curr: Action, ppo_cfg: Action) -> Action:
        """
        Combine current config and PPO suggested config according to the
        high-level decision.
        """
        if hi_action == 0:  # CPU only
            return Action(cpus=ppo_cfg.cpus, memory=curr.memory, heap=curr.heap)
        elif hi_action == 1:  # Memory only
            return Action(cpus=curr.cpus, memory=ppo_cfg.memory, heap=curr.heap)
        elif hi_action == 2:  # Heap only
            return Action(cpus=curr.cpus, memory=curr.memory, heap=ppo_cfg.heap)
        elif hi_action == 3:  # Memory + Heap
            return Action(cpus=curr.cpus, memory=ppo_cfg.memory, heap=ppo_cfg.heap)
        else:
            # Fallback: take PPO config directly
            return ppo_cfg

    # ---------------- Gym API ----------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        # Reset underlying env (for completeness)
        self.base_env.reset()
        # Start from a random configuration
        self.current_idx = random.randrange(len(self.actions))
        obs = self._obs_from_idx(self.current_idx)
        return obs, {}

    def step(self, hi_action: int):
        hi_action = int(hi_action)

        # 1) Get current observation
        obs = self._obs_from_idx(self.current_idx)

        # 2) Low-level PPO suggests a config
        ppo_action_idx, _ = self.ppo_model.predict(obs, deterministic=True)
        ppo_action_idx = int(ppo_action_idx)
        ppo_cfg = self.actions[ppo_action_idx]

        # 3) Combine current config and PPO suggestion as per high-level choice
        curr_cfg = self.actions[self.current_idx]
        new_cfg = self._apply_high_level_action(hi_action, curr_cfg, ppo_cfg)

        # 4) Map combined config to an index; fallback to PPO index if combo not in CSV
        key = (str(new_cfg.cpus), str(new_cfg.memory), str(new_cfg.heap))
        new_idx = self.config_to_idx.get(key, ppo_action_idx)

        # 5) Evaluate that config using the base environment
        obs2, reward, terminated, truncated, info = self.base_env.step(new_idx)

        # Make sure info is a dict we can extend
        if not isinstance(info, dict):
            info = {}
        info = dict(info)
        info.setdefault("config", (new_cfg.cpus, new_cfg.memory, new_cfg.heap))
        info.setdefault("config_idx", new_idx)

        # 6) Update internal state
        self.current_idx = new_idx

        return obs2, reward, terminated, truncated, info


# ==============================================================================
# Training metrics callback (used for OFFLINE + ONLINE)
# ==============================================================================

class TrainingMetricsCallback(BaseCallback):
    """
    Generic callback to track reward, startup time and a simple loss proxy.

    We approximate "loss" as -reward to give a decreasing curve when the model
    improves (higher reward -> lower loss_proxy).
    """

    def __init__(self, max_steps: Optional[int] = None, phase: str = ""):
        super().__init__()
        self.max_steps = max_steps
        self.phase = phase
        self.rewards: List[float] = []
        self.startup_times: List[Optional[float]] = []
        self.loss_proxy: List[float] = []

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards")
        infos = self.locals.get("infos")

        r = float(rewards[0]) if rewards is not None else float("nan")
        startup = None
        if infos is not None and len(infos) > 0 and isinstance(infos[0], dict):
            startup = infos[0].get("startup_seconds")

        self.rewards.append(r)
        self.startup_times.append(startup)
        self.loss_proxy.append(-r)  # simple proxy: lower is better

        if self.phase and self.num_timesteps % 1000 == 0:
            print(f"[{self.phase}] Step {self.num_timesteps}, reward={r:.3f}, startup={startup}")

        if self.max_steps is not None:
            return self.num_timesteps < self.max_steps
        return True


# ==============================================================================
# Action / baseline helpers
# ==============================================================================

def build_actions_from_csv(
    startup_df: pd.DataFrame,
    cpu_max: Optional[float],
    mem_max: Optional[str],
) -> Tuple[List[Action], pd.DataFrame]:
    """
    Filter CSV by resource limits and build the list of valid Actions.

    Returns:
        (actions, filtered_df)
    """
    heap_col = "heap"
    if heap_col not in startup_df.columns and "heap_size" in startup_df.columns:
        heap_col = "heap_size"

    filtered = startup_df.copy()

    if cpu_max is not None:
        filtered = filtered[filtered["cpus"].astype(float) <= cpu_max]

    if mem_max is not None:
        max_mb = parse_memory_to_mb(mem_max)
        mem_mb_series = filtered["memory"].astype(str).map(parse_memory_to_mb)
        filtered = filtered[mem_mb_series <= max_mb]

    # Ensure heap <= memory where heap column exists
    if heap_col in filtered.columns:
        heap_mb = filtered[heap_col].astype(str).map(parse_memory_to_mb)
        mem_mb = filtered["memory"].astype(str).map(parse_memory_to_mb)
        filtered = filtered[heap_mb <= mem_mb]

    combos = (
        filtered[["cpus", "memory", heap_col]]
        .astype(str)
        .drop_duplicates()
        .reset_index(drop=True)
    )

    actions: List[Action] = [
        Action(cpus=row["cpus"], memory=row["memory"], heap=row[heap_col])
        for _, row in combos.iterrows()
    ]
    return actions, filtered


def compute_baseline(startup_df: pd.DataFrame, mode: str) -> float:
    """
    Compute baseline startup time from the filtered CSV.

    mode can be:
        - "mean"
        - "median"
        - "min"
        - or a numeric string (e.g. "5.0")
    """
    if mode in ("mean", "avg"):
        return float(startup_df["startup_seconds"].mean())
    if mode == "median":
        return float(startup_df["startup_seconds"].median())
    if mode == "min":
        return float(startup_df["startup_seconds"].min())
    try:
        return float(mode)
    except ValueError:
        raise ValueError(f"Invalid baseline mode: {mode}")


# ==============================================================================
# Main entry point
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hierarchical RL training for AcmeAir with resource limits and plotting."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="startup_data.csv",
        help="CSV file with sweep data (must contain cpus, memory, startup_seconds).",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="median",
        help="Baseline mode: 'mean', 'median', 'min', or a numeric value.",
    )
    parser.add_argument(
        "--offline-steps",
        type=int,
        default=5000,
        help="Timesteps for offline training (CSV-only) for each level.",
    )
    parser.add_argument(
        "--online-steps",
        type=int,
        default=20,
        help="Timesteps for online fine-tuning (real Docker environment) for each level.",
    )
    parser.add_argument(
        "--cpu-max",
        type=float,
        default=None,
        help="Max CPU cores allowed (e.g. 2.0). Only configs with cpus <= cpu_max are used.",
    )
    parser.add_argument(
        "--mem-max",
        type=str,
        default=None,
        help="Max memory allowed (e.g. '1G', '768M'). Only configs with memory <= mem_max are used.",
    )
    parser.add_argument(
        "--no-normalize-reward",
        action="store_true",
        help=(
            "Disable reward normalization. By default, rewards are divided by "
            "the baseline startup time to keep them in a stable range."
        ),
    )

    args = parser.parse_args()
    normalize_reward = not args.no_normalize_reward

    # Where to save models / plots: ../model_results/
    project_dir = os.path.dirname(__file__)
    model_results_dir = os.path.abspath(os.path.join(project_dir, "..", "model_results"))
    os.makedirs(model_results_dir, exist_ok=True)

    # ----- Load CSV -----------------------------------------------------------

    default_csv_path = os.path.join(
        project_dir, "..", "data", "startup_data.csv"
    )

    csv_path = args.csv if args.csv != "startup_data.csv" else default_csv_path
    csv_path = os.path.abspath(csv_path)

    print(f"[INFO] Loading CSV from: {csv_path}")

    if not os.path.exists(csv_path):
        print(f"[ERROR] CSV file '{csv_path}' not found.", file=sys.stderr)
        sys.exit(1)

    startup_df = pd.read_csv(csv_path)

    required_cols = {"cpus", "memory", "startup_seconds"}
    if not required_cols.issubset(startup_df.columns):
        print(f"[ERROR] CSV must contain columns {required_cols}", file=sys.stderr)
        sys.exit(1)

    # ----- Build actions and baseline with limits ----------------------------

    actions, filtered_df = build_actions_from_csv(startup_df, args.cpu_max, args.mem_max)
    if filtered_df.empty or not actions:
        print("[ERROR] After applying limits, no valid configurations remain.", file=sys.stderr)
        sys.exit(1)

    if args.cpu_max is not None:
        print(f"[INFO] Applied CPU limit: {args.cpu_max} cores")
    if args.mem_max is not None:
        print(f"[INFO] Applied memory limit: {args.mem_max}")
    print(f"[INFO] Valid configurations after filtering: {len(actions)}")

    baseline_startup = compute_baseline(filtered_df, args.baseline)
    print(f"[INFO] Using baseline startup time: {baseline_startup:.3f} seconds")
    if normalize_reward:
        print("[INFO] Reward normalization is ENABLED (reward / baseline).")
    else:
        print("[INFO] Reward normalization is DISABLED (raw baseline - startup).")

    # =========================================================================
    # Phase A1: PPO low-level training on OfflineStartupEnv
    # =========================================================================

    offline_env = OfflineStartupEnv(
        actions=actions,
        startup_df=filtered_df,
        baseline_startup=baseline_startup,
        normalize_reward=normalize_reward,
    )

    ppo_model = PPO(
        "MlpPolicy",
        offline_env,
        learning_rate=1e-4,   
        n_steps=512,          
        batch_size=128,
        n_epochs=10,
        gamma=0.995,          
        gae_lambda=0.96,
        clip_range=0.1,       
        ent_coef=0.0,
        verbose=1,
    )


    ppo_offline_cb = TrainingMetricsCallback(phase="PPO_offline")

    print(f"[PHASE A1] Training PPO (low-level) offline for {args.offline_steps} timesteps…")
    ppo_model.learn(
        total_timesteps=args.offline_steps,
        progress_bar=False,
        callback=ppo_offline_cb,
    )

    # =========================================================================
    # Phase A2: DQN high-level training on hierarchical Offline env
    # =========================================================================

    hier_offline_env = HierarchicalStartupEnv(
        actions=actions,
        base_env=offline_env,
        ppo_model=ppo_model,
    )

    dqn_model = DQN(
        "MlpPolicy",
        hier_offline_env,
        learning_rate=2.5e-4,   
        buffer_size=100000,     
        batch_size=128,
        learning_starts=1000,
        train_freq=4,           
        exploration_fraction=0.15,
        exploration_final_eps=0.05,
        target_update_interval=1000,  
        gamma=0.995,
        verbose=1,
    )


    dqn_offline_cb = TrainingMetricsCallback(phase="DQN_offline")

    print(f"[PHASE A2] Training DQN (high-level) offline for {args.offline_steps} timesteps…")
    dqn_model.learn(
        total_timesteps=args.offline_steps,
        progress_bar=False,
        callback=dqn_offline_cb,
    )

    # =========================================================================
    # Phase B1: PPO low-level fine-tuning on RealStartupEnv
    # =========================================================================

    print(f"[PHASE B1] Switching PPO low-level to real Docker environment for {args.online_steps} timesteps…")

    real_env = RealStartupEnv(
        actions=actions,
        baseline_startup=baseline_startup,
        normalize_reward=normalize_reward,
    )

    ppo_model.set_env(real_env)
    # Make online PPO updates gentle (real runs are noisy & expensive)
    ppo_model.learning_rate = 5e-5
    ppo_model.n_steps = 32
    ppo_model.batch_size = 16
    ppo_model.gamma = 0.999
    ppo_model.clip_range = 0.1


    ppo_online_cb = TrainingMetricsCallback(
        max_steps=args.online_steps,
        phase="PPO_online",
    )

    ppo_model.learn(
        total_timesteps=args.online_steps,
        reset_num_timesteps=False,
        callback=ppo_online_cb,
        progress_bar=False,
    )

    # =========================================================================
    # Phase B2: DQN high-level fine-tuning on hierarchical Real env
    # =========================================================================

    print(f"[PHASE B2] Switching DQN high-level to hierarchical real env for {args.online_steps} timesteps…")

    hier_real_env = HierarchicalStartupEnv(
        actions=actions,
        base_env=real_env,
        ppo_model=ppo_model,
    )

    dqn_model.set_env(hier_real_env)
    # Gentle online DQN: lower LR, slower target updates, less exploration
    dqn_model.learning_rate = 1e-4
    dqn_model.buffer_size = 5000
    dqn_model.batch_size = 32
    dqn_model.exploration_fraction = 0.05
    dqn_model.exploration_final_eps = 0.02
    dqn_model.target_update_interval = 500
    dqn_model.gamma = 0.999


    dqn_online_cb = TrainingMetricsCallback(
        max_steps=args.online_steps,
        phase="DQN_online",
    )

    dqn_model.learn(
        total_timesteps=args.online_steps,
        reset_num_timesteps=False,
        callback=dqn_online_cb,
        progress_bar=False,
    )

    # =========================================================================
    # Save final high-level model (only one) to ../model_results/model.zip
    # =========================================================================

    final_model_file = os.path.join(model_results_dir, "model.zip")
    dqn_model.save(final_model_file)
    print(f"[SAVE] Saved final high-level DQN model to {final_model_file}")

    # =========================================================================
    # Plots: reward_curves.png, loss_curves.png, startup_time.png
    # =========================================================================

    try:
        import matplotlib.pyplot as plt

        series = [
            ("PPO offline", ppo_offline_cb),
            ("DQN offline", dqn_offline_cb),
            ("PPO online",  ppo_online_cb),
            ("DQN online",  dqn_online_cb),
        ]

        # Reward curves
        plt.figure(figsize=(10, 6))
        for label, cb in series:
            if cb.rewards:
                steps = list(range(1, len(cb.rewards) + 1))
                plt.plot(steps, cb.rewards, marker="o", markersize=2, linewidth=1, label=label)
        plt.xlabel("Timestep")
        plt.ylabel("Reward (normalized)" if normalize_reward else "Reward")
        plt.title("Reward curves (offline + online, hierarchical)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        reward_path = os.path.join(model_results_dir, "reward_curves.png")
        plt.savefig(reward_path)
        plt.close()
        print(f"[PLOT] Saved reward curves to {reward_path}")

        # Loss curves (proxy = -reward)
        plt.figure(figsize=(10, 6))
        for label, cb in series:
            if cb.loss_proxy:
                steps = list(range(1, len(cb.loss_proxy) + 1))
                plt.plot(steps, cb.rewards, marker="o", markersize=2, linewidth=1, label=label)
        plt.xlabel("Timestep")
        plt.ylabel("Loss proxy (-reward)")
        plt.title("Loss curves (offline + online, hierarchical)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        loss_path = os.path.join(model_results_dir, "loss_curves.png")
        plt.savefig(loss_path)
        plt.close()
        print(f"[PLOT] Saved loss curves to {loss_path}")

        # Startup time curves
        for label, cb in series:
            valid_points = []
            for i, s in enumerate(cb.startup_times):
                if s is None:
                    continue
                # only accept real scalar numbers
                try:
                    val = float(s)
                except Exception:
                    continue
                valid_points.append((i + 1, val))

            if valid_points:
                s_steps, s_vals = zip(*valid_points)
                plt.plot(s_steps, s_vals, marker="o", markersize=2, linewidth=1, label=label)

        plt.xlabel("Timestep")
        plt.ylabel("Startup time (s)")
        plt.title("Startup time curves (offline + online, hierarchical)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        startup_path = os.path.join(model_results_dir, "startup_time.png")
        plt.savefig(startup_path)
        plt.close()
        print(f"[PLOT] Saved startup time curves to {startup_path}")

    except Exception as exc:
        print(f"[PLOT] Failed to generate plots (maybe matplotlib missing?): {exc}")

    # =========================================================================
    # Final evaluation: run hierarchical policy once on real env
    # =========================================================================

    print("[EVAL] Evaluating final hierarchical policy once on real environment…")

    # Reuse hier_real_env (RealStartupEnv + PPO)
    eval_env = hier_real_env

    obs, _ = eval_env.reset()
    hi_action, _ = dqn_model.predict(obs, deterministic=True)
    obs2, reward, done, truncated, info = eval_env.step(int(hi_action))

    config = info.get("config")
    startup = info.get("startup_seconds")

    print("[EVAL] Final suggested configuration from hierarchical policy:")
    if config is not None:
        cpu, mem, heap = config
        print(f"  CPU:    {cpu}")
        print(f"  Memory: {mem}")
        print(f"  Heap:   {heap}")
    else:
        print("  Config: N/A")

    if startup is not None:
        print(f"  Measured startup time: {startup:.3f} seconds")
    else:
        print("  Measured startup time: N/A (invalid configuration or measurement failure)")

    print(f"  Final reward (normalized={normalize_reward}): {reward:.3f}")


if __name__ == "__main__":
    main()
