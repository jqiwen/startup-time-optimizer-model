#!/usr/bin/env python3
"""
Hybrid RL training for AcmeAir startup tuning with resource limits
and an online learning curve plot.

Phase A (offline):
    - Use startup_data.csv only.
    - Train RL agent on a Gym environment that samples startup_seconds
      directly from the CSV for each (cpu, memory, heap) action.
    - Fast: thousands of timesteps in seconds.

Phase B (online):
    - Reuse the same trained agent.
    - Switch to a real Docker+Prometheus environment.
    - Each step:
        * Restart Docker stack with chosen JVM heap.
        * Update CPU and memory limits on the 'app' container.
        * Restart sidecar with labels (cpus, memory, heap).
        * Wait for /health/ready.
        * Query Prometheus for app_startup_seconds{...}.
        * Reward = baseline_startup - measured_startup.
    - Slow but accurate: e.g. 3–10 timesteps.
    - Records reward and startup time per timestep and plots them.

New in this version:
    - Supports CPU and memory limits via CLI:
        --cpu-max 2.0   (only cpus <= 2.0)
        --mem-max 1G    (only memory <= 1G)
    - At the end of training, generates:
        online_training_curve.png
      showing reward and startup time vs online timestep.
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

# Gymnasium (preferred)
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import DQN, PPO  # type: ignore
from stable_baselines3.common.callbacks import BaseCallback  # type: ignore

# --------------------------------------------------------------------------------------
# Docker/Prometheus configuration
# --------------------------------------------------------------------------------------

APP_CONTAINER = "app"
SIDECAR_CONTAINER = "sidecar"

HEALTH_URL_HOST = "http://localhost:9080/health/ready"
HEALTH_URL_CONTAINER = "http://app:9080/health/ready"
PROMETHEUS_QUERY_URL = "http://localhost:9090/api/v1/query"

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Action:
    cpus: str   # e.g. "0.5", "1.0"
    memory: str # e.g. "512M", "768M", "1G"
    heap: str   # e.g. "256M", "512M", "1G"


def parse_memory_to_mb(mem: str) -> int:
    """Convert memory string like '512M', '768M', '1G', '2G' to MB."""
    mem = str(mem).strip().upper()
    if mem.endswith("G"):
        return int(float(mem[:-1]) * 1024)
    if mem.endswith("M"):
        return int(float(mem[:-1]))
    # Assume raw MB number
    return int(float(mem))


def docker_cmd(cmd: List[str], env: Optional[dict] = None, capture: bool = False) -> Tuple[int, str, str]:
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


def get_default_network() -> str:
    """Return the first docker network ending with '_default', or 'project_default'."""
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
    """Query Prometheus for app_startup_seconds with given labels."""
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


def configure_app_resources(cpus: str, memory: str):
    """Apply CPU and memory limits to the app container."""
    mem_lower = memory.lower()  # docker likes 512m / 1g
    print(f"[STACK] Updating app resources: cpus={cpus}, memory={mem_lower}")
    rc, _, err = docker_cmd(
        ["docker", "update", "--cpus", cpus, "--memory", mem_lower, APP_CONTAINER]
    )
    if rc != 0:
        print(f"[WARN] docker update failed: {err.strip()}")


def restart_sidecar(cpus: str, memory: str, heap: str, network: str):
    """Restart sidecar container with labels for this configuration."""
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


# --------------------------------------------------------------------------------------
# OFFLINE ENV (CSV-based)
# --------------------------------------------------------------------------------------


class OfflineStartupEnv(gym.Env):
    """Gym environment that uses only CSV data (no Docker)."""

    metadata = {"render.modes": []}

    def __init__(self, actions: List[Action], df: pd.DataFrame, baseline: float):
        super().__init__()
        self.actions = actions
        self.baseline = baseline

        heap_col = "heap"
        if heap_col not in df.columns and "heap_size" in df.columns:
            heap_col = "heap_size"

        # Mapping action_index -> list of measured startup times
        self.action_to_times: Dict[int, List[float]] = {}
        for i, a in enumerate(actions):
            mask = (
                df["cpus"].astype(str) == a.cpus
            ) & (
                df["memory"].astype(str) == a.memory
            ) & (
                df[heap_col].astype(str) == a.heap
            )
            times = df.loc[mask, "startup_seconds"].dropna().tolist()
            self.action_to_times[i] = times

        # For observation scaling
        cpu_vals = [float(a.cpus) for a in actions]
        mem_vals = [parse_memory_to_mb(a.memory) for a in actions]
        heap_vals = [parse_memory_to_mb(a.heap) for a in actions]

        self.cpu_min, self.cpu_max = min(cpu_vals), max(cpu_vals)
        self.mem_min, self.mem_max = min(mem_vals), max(mem_vals)
        self.heap_min, self.heap_max = min(heap_vals), max(heap_vals)

        # Observation: [cpu_norm, mem_norm, heap_norm, baseline_norm]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(4,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(len(actions))

    def _obs_from_action(self, idx: int) -> np.ndarray:
        a = self.actions[idx]
        cpu = float(a.cpus)
        mem = parse_memory_to_mb(a.memory)
        heap = parse_memory_to_mb(a.heap)

        cpu_norm = (cpu - self.cpu_min) / (self.cpu_max - self.cpu_min + 1e-9)
        mem_norm = (mem - self.mem_min) / (self.mem_max - self.mem_min + 1e-9)
        heap_norm = (heap - self.heap_min) / (self.heap_max - self.heap_min + 1e-9)
        baseline_norm = min(self.baseline / 60.0, 1.0)  # assume 0-60s typical

        return np.array(
            [cpu_norm, mem_norm, heap_norm, baseline_norm],
            dtype=np.float32,
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        idx = random.randrange(len(self.actions))
        return self._obs_from_action(idx), {}

    def step(self, action_idx: int):
        idx = int(action_idx)
        times = self.action_to_times.get(idx, [])
        if times:
            startup = random.choice(times)
        else:
            # No data: assign slightly worse than baseline
            startup = self.baseline * 1.5

        reward = self.baseline - startup
        obs = self._obs_from_action(idx)
        terminated = True
        truncated = False
        info = {"startup_seconds": startup}
        return obs, reward, terminated, truncated, info


# --------------------------------------------------------------------------------------
# ONLINE ENV (Real Docker + Prometheus)
# --------------------------------------------------------------------------------------


class RealStartupEnv(gym.Env):
    """Gym environment that uses real Docker stack + Prometheus."""

    INVALID_REWARD = -100.0

    metadata = {"render.modes": []}

    def __init__(self, actions: List[Action], baseline: float):
        super().__init__()
        self.actions = actions
        self.baseline = baseline
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

    def _obs_from_action(self, idx: int) -> np.ndarray:
        a = self.actions[idx]
        cpu = float(a.cpus)
        mem = parse_memory_to_mb(a.memory)
        heap = parse_memory_to_mb(a.heap)

        cpu_norm = (cpu - self.cpu_min) / (self.cpu_max - self.cpu_min + 1e-9)
        mem_norm = (mem - self.mem_min) / (self.mem_max - self.mem_min + 1e-9)
        heap_norm = (heap - self.heap_min) / (self.heap_max - self.heap_min + 1e-9)
        baseline_norm = min(self.baseline / 60.0, 1.0)

        return np.array(
            [cpu_norm, mem_norm, heap_norm, baseline_norm],
            dtype=np.float32,
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        idx = random.randrange(len(self.actions))
        return self._obs_from_action(idx), {}

    def step(self, action_idx: int):
        idx = int(action_idx)
        a = self.actions[idx]
        cpus, memory, heap = a.cpus, a.memory, a.heap

        # Basic validity: heap must not exceed memory
        if parse_memory_to_mb(heap) > parse_memory_to_mb(memory):
            print(f"[ENV] Invalid config heap={heap} > memory={memory}")
            reward = self.INVALID_REWARD
            obs = self._obs_from_action(idx)
            return obs, reward, True, False, {"startup_seconds": None}

        try:
            stop_stack()
            jvm_args = f"-Xms{heap} -Xmx{heap}"
            start_stack(jvm_args)
            configure_app_resources(cpus, memory)
            restart_sidecar(cpus, memory, heap, self.network)

            ready = wait_for_readiness(HEALTH_URL_HOST, timeout=300)
            if not ready:
                print(f"[ENV] App not ready for {a}")
                reward = self.INVALID_REWARD
                obs = self._obs_from_action(idx)
                return obs, reward, True, False, {"startup_seconds": None}

            print("[ENV] Waiting for Prometheus scrape…")
            time.sleep(10)

            startup = query_prometheus_startup(cpus, memory, heap)
            print(f"[ENV] Measured startup: {startup:.3f} seconds")

            reward = self.baseline - startup
            obs = self._obs_from_action(idx)
            info = {"startup_seconds": startup}
        except Exception as e:
            print(f"[ENV] Measurement failed for {a}: {e}")
            reward = self.INVALID_REWARD
            obs = self._obs_from_action(idx)
            info = {"startup_seconds": None}

        return obs, reward, True, False, info


# --------------------------------------------------------------------------------------
# Callback for online phase (prints + logs + stop at N steps)
# --------------------------------------------------------------------------------------


class OnlineStepPrinterCallback(BaseCallback):
    """Callback that:
       - prints timestep number
       - logs reward and startup_seconds
       - stops after max_steps timesteps
    """

    def __init__(self, max_steps: int):
        super().__init__()
        self.max_steps = max_steps
        self.reward_history: List[float] = []
        self.startup_history: List[Optional[float]] = []

    def _on_step(self) -> bool:
        # rewards and infos from SB3 VecEnv (n_envs=1)
        rewards = self.locals.get("rewards")
        infos = self.locals.get("infos")

        r = float(rewards[0]) if rewards is not None else float("nan")
        startup = None
        if infos is not None and len(infos) > 0 and isinstance(infos[0], dict):
            startup = infos[0].get("startup_seconds")

        self.reward_history.append(r)
        self.startup_history.append(startup)

        print(f"[ONLINE] Completed timestep {self.num_timesteps}, reward={r:.3f}, startup={startup}")
        # stop when reaching max_steps
        return self.num_timesteps < self.max_steps


# --------------------------------------------------------------------------------------
# Building actions & baselines with limits
# --------------------------------------------------------------------------------------


def build_actions_from_csv(df: pd.DataFrame, cpu_max: Optional[float], mem_max: Optional[str]) -> Tuple[List[Action], pd.DataFrame]:
    """Filter CSV by limits and build Action list. Returns (actions, filtered_df)."""
    heap_col = "heap"
    if heap_col not in df.columns and "heap_size" in df.columns:
        heap_col = "heap_size"

    filtered = df.copy()

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


def compute_baseline(df: pd.DataFrame, mode: str) -> float:
    if mode in ("mean", "avg"):
        return float(df["startup_seconds"].mean())
    if mode == "median":
        return float(df["startup_seconds"].median())
    if mode == "min":
        return float(df["startup_seconds"].min())
    try:
        return float(mode)
    except ValueError:
        raise ValueError(f"Invalid baseline mode: {mode}")


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Hybrid RL training for AcmeAir with limits + plotting.")
    parser.add_argument("--csv", type=str, default="startup_data.csv", help="CSV with sweep data.")
    parser.add_argument("--model", type=str, choices=["dqn", "ppo"], default="dqn")
    parser.add_argument("--baseline", type=str, default="median",
                        help="Baseline: 'mean', 'median', 'min', or a numeric value.")
    parser.add_argument("--offline-steps", type=int, default=5000,
                        help="Timesteps for offline training (CSV only).")
    parser.add_argument("--online-steps", type=int, default=5,
                        help="Timesteps for online fine-tuning (real Docker).")
    parser.add_argument("--cpu-max", type=float, default=None,
                        help="Max CPU cores allowed (e.g. 2.0). Only configs with cpus <= cpu_max are used.")
    parser.add_argument("--mem-max", type=str, default=None,
                        help="Max memory allowed (e.g. 1G, 768M). Only configs with memory <= mem_max are used.")
    args = parser.parse_args()

    # Load CSV
    if not os.path.exists(args.csv):
        print(f"[ERROR] CSV file '{args.csv}' not found.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.csv)
    required_cols = {"cpus", "memory", "startup_seconds"}
    if not required_cols.issubset(df.columns):
        print(f"[ERROR] CSV must contain columns {required_cols}", file=sys.stderr)
        sys.exit(1)

    # Build actions with limits applied
    actions, filtered_df = build_actions_from_csv(df, args.cpu_max, args.mem_max)
    if filtered_df.empty or not actions:
        print("[ERROR] After applying limits, no valid configurations remain.")
        sys.exit(1)

    if args.cpu_max is not None:
        print(f"[INFO] Applied CPU limit: {args.cpu_max} cores")
    if args.mem_max is not None:
        print(f"[INFO] Applied memory limit: {args.mem_max}")
    print(f"[INFO] Valid configurations after filtering: {len(actions)}")

    baseline_value = compute_baseline(filtered_df, args.baseline)
    print(f"[INFO] Using baseline startup time: {baseline_value:.3f} seconds")

    # ---------------- Phase A: Offline training ----------------
    offline_env = OfflineStartupEnv(actions, filtered_df, baseline_value)

    if args.model == "dqn":
        model = DQN(
            "MlpPolicy",
            offline_env,
            learning_rate=5e-4,
            buffer_size=50000,
            batch_size=64,
            learning_starts=500,
            train_freq=1,
            exploration_fraction=0.2,
            exploration_final_eps=0.05,
            target_update_interval=500,
            gamma=0.98,
            verbose=1
        )

    else:
        model = PPO(
            "MlpPolicy",
            offline_env,

            learning_rate=3e-4,     # slightly aggressive, trains fast
            n_steps=256,            # shorter rollouts for tabular-like data
            batch_size=64,          # good general-purpose setting
            n_epochs=10,            # enough optimization passes

            gamma=0.99,
            gae_lambda=0.95,

            clip_range=0.2,         # normal PPO clipping
            ent_coef=0.0,           # no entropy bonus (unnecessary)

            verbose=1
        )


    print(f"[PHASE A] Training {args.model.upper()} offline for {args.offline_steps} timesteps…")
    model.learn(total_timesteps=args.offline_steps, progress_bar=False)
    offline_model_file = f"{args.model}_acmeair_offline_limited.zip"
    model.save(offline_model_file)
    print(f"[PHASE A] Saved offline model to {offline_model_file}")

    # ---------------- Phase B: Online fine-tuning ----------------
    print(f"[PHASE B] Switching to real Docker environment for {args.online_steps} timesteps…")
    online_env = RealStartupEnv(actions, baseline_value)

    if args.model == "dqn":
        model = DQN.load(
            offline_model_file,
            env=online_env,
            learning_rate=1e-4,
            exploration_fraction=0.05,
            exploration_final_eps=0.01,
            gamma=0.99,
            buffer_size=2000,
            batch_size=32,
            target_update_interval=200,
            verbose=1
        )

    else:
        model = PPO(
            "MlpPolicy",
            online_env,

            learning_rate=1e-4,      # LOWER, avoids overshooting in noisy env
            n_steps=32,              # SMALL rollout (each step is expensive)
            batch_size=16,           # very small mini-batch
            n_epochs=4,              # few updates per rollout → conservative

            gamma=0.999,             # more weight on long-term improvement
            gae_lambda=0.90,         # slightly more bias → more stable

            clip_range=0.15,         # smaller clipping → safer updates
            ent_coef=0.0,            # deterministic is fine for tuning

            verbose=1
        )


    callback = OnlineStepPrinterCallback(max_steps=args.online_steps)
    model.learn(
        total_timesteps=args.online_steps,
        reset_num_timesteps=True,
        callback=callback,
        progress_bar=False,
    )

    finetuned_model_file = f"{args.model}_acmeair_hybrid_finetuned_limited.zip"
    model.save(finetuned_model_file)
    print(f"[PHASE B] Saved finetuned model to {finetuned_model_file}")

    # ---------------- Plot online learning curve ----------------
    try:
        import matplotlib.pyplot as plt

        steps = list(range(1, len(callback.reward_history) + 1))

        plt.figure(figsize=(10, 4))

        # Reward curve
        plt.subplot(1, 2, 1)
        plt.plot(steps, callback.reward_history, marker="o")
        plt.xlabel("Online timestep")
        plt.ylabel("Reward (baseline - startup)")
        plt.title("Online Reward vs Timestep")
        plt.grid(True)

        # Startup time curve (only where startup is not None)
        valid_points = [(i + 1, s) for i, s in enumerate(callback.startup_history) if s is not None]
        if valid_points:
            s_steps, s_vals = zip(*valid_points)
            plt.subplot(1, 2, 2)
            plt.plot(s_steps, s_vals, marker="o")
            plt.xlabel("Online timestep")
            plt.ylabel("Startup time (s)")
            plt.title("Measured Startup Time vs Timestep")
            plt.grid(True)

        plt.tight_layout()
        out_file = "online_training_curve.png"
        plt.savefig(out_file)
        plt.close()
        print(f"[PLOT] Saved online training curves to {out_file}")
    except Exception as e:
        print(f"[PLOT] Failed to generate plot (install matplotlib?): {e}")

    # ---------------- Evaluate final recommendation once ----------------
    print("[EVAL] Evaluating final policy once on real environment…")
    obs, _ = online_env.reset()
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = online_env.step(int(action))
    selected = actions[int(action)]

    print("[EVAL] Final suggested configuration (respecting limits):")
    print(f"  CPU:    {selected.cpus}")
    print(f"  Memory: {selected.memory}")
    print(f"  Heap:   {selected.heap}")

    startup = info.get("startup_seconds")
    if startup is not None:
        print(f"  Measured startup time: {startup:.3f} seconds")
    else:
        print("  Measured startup time: N/A (invalid configuration or measurement failure)")


if __name__ == "__main__":
    main()
