#!/usr/bin/env python3
"""
evaluate_model.py
===================

This script demonstrates how to use a trained reinforcement‑learning model
to interact with the AcmeAir startup environment.  It loads a saved model
(DQN or PPO) created by ``train_rl.py``, chooses a configuration (CPU cores,
memory limit and JVM heap size) based on the policy, applies that
configuration to your local Docker Compose stack, waits for the
application to become ready, measures the startup time via Prometheus,
and finally prints the result.

The environment is treated as a one‑shot bandit: each action corresponds
to a unique (CPU, memory, heap) combination.  When the agent selects an
action, the script restarts the stack with those settings and returns
the negative startup time as reward.  Because there is no persistent
state between episodes, the observation space is a dummy vector.

Usage:

    # Make sure you have stable‑baselines3, gymnasium and requests installed:
    #   pip install stable-baselines3 gymnasium requests pandas
    
    # Place this script in the same directory as startup_data.csv
    # and your saved models (dqn_acmeair_model.zip, ppo_acmeair_model.zip)
    
    python evaluate_model.py --model dqn --model-file dqn_acmeair_model.zip

    # Or for PPO:
    python evaluate_model.py --model ppo --model-file ppo_acmeair_model.zip

The script will output the selected configuration and the measured
startup time.

Note: This script is meant to be run on your own machine where you
have installed the necessary Python packages.  The current
container environment used by the assistant does not include
stable‑baselines3 or gymnasium, so you may see ImportError if you
attempt to run it here.  Install the required packages with::

    pip install stable-baselines3[extra] gymnasium pandas requests

and then run the script.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import pandas as pd  # type: ignore
import requests  # type: ignore

# Try to import gymnasium first, fallback to gym.  Stable‑baselines3 expects an
# environment that inherits from either gymnasium.Env or gym.Env.  If neither
# library is installed, the user must install one via pip.
try:
    import gymnasium as gym  # type: ignore
    _GYM_AVAILABLE = True
except ImportError:
    try:
        import gym  # type: ignore
        _GYM_AVAILABLE = True
    except ImportError:
        _GYM_AVAILABLE = False

# If gym or gymnasium is available, define a base environment class.  Otherwise
# default to object; instantiating the environment will raise ImportError.
if _GYM_AVAILABLE:
    base_env = getattr(gym, 'Env')
    spaces = getattr(gym, 'spaces')
else:
    base_env = object
    spaces = None

import numpy as np  # type: ignore


def run_cmd(cmd: List[str], env: Optional[dict] = None, capture: bool = False) -> Tuple[int, str, str]:
    """Run a subprocess command and return (returncode, stdout, stderr)."""
    env_vars = os.environ.copy()
    if env:
        env_vars.update(env)
    result = subprocess.run(
        cmd,
        check=False,
        text=True,
        capture_output=capture,
        env=env_vars,
    )
    return result.returncode, result.stdout if capture else "", result.stderr if capture else ""


def parse_memory_to_mb(value: str) -> float:
    """Convert a memory string like '512M' or '1G' to megabytes as float."""
    value = value.strip().lower()
    if value.endswith("g"):
        return float(value[:-1]) * 1024.0
    if value.endswith("m"):
        return float(value[:-1])
    # Assume bytes if no suffix
    return float(value) / (1024.0 * 1024.0)


def wait_for_readiness(url: str, timeout: int = 300) -> bool:
    """Poll an HTTP endpoint until it returns status code 200 or the timeout is reached."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def query_startup_time(cpus: str, memory: str, heap: str, query_url: str, timeout: int = 180) -> float:
    """Query Prometheus for app_startup_seconds and return the value as float."""
    query = (
        f'app_startup_seconds{{app="acmeair",cpus="{cpus}",memory="{memory}",heap="{heap}"}}'
    )
    params = {"query": query}
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            resp = requests.get(query_url, params=params, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("status") == "success":
                    results = data.get("data", {}).get("result", [])
                    if results:
                        value = float(results[0]["value"][1])
                        return value
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError(f"Metric not found for cpus={cpus}, memory={memory}, heap={heap}")


@dataclass
class Action:
    cpu: str
    memory: str
    heap: str

    def as_tuple(self) -> Tuple[str, str, str]:
        return self.cpu, self.memory, self.heap


class DockerStartupEnv(base_env):
    """Custom environment to apply resource settings and measure startup time.

    This environment emulates a one‑shot bandit: each step applies a resource
    configuration, restarts the stack, waits for readiness, queries Prometheus
    for the startup time, and returns negative startup time as reward.

    It derives from gym.Env or gymnasium.Env so that stable‑baselines3 can
    interface with it.  If gymnasium or gym is not installed, it raises
    ImportError during construction.
    """

    def __init__(
        self,
        actions: List[Action],
        health_url: str = "http://localhost:9080/health/ready",
        prometheus_query_url: str = "http://localhost:9090/api/v1/query",
    ) -> None:
        if not _GYM_AVAILABLE or spaces is None:
            raise ImportError(
                "Neither gymnasium nor gym is installed. Install via `pip install gymnasium` or `pip install gym`."
            )
        # Call parent constructor if available
        try:
            super().__init__()
        except Exception:
            pass

        self.actions = actions
        self.health_url = health_url
        self.prometheus_query_url = prometheus_query_url

        # Define action and observation spaces using imported spaces module
        self.action_space = spaces.Discrete(len(actions))
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset the environment and return an initial observation.

        This environment is stateless, so we return a fixed observation.
        """
        if seed is not None:
            # Use gym's seeding utilities if available
            if hasattr(gym.utils, 'seeding'):
                gym.utils.seeding.np_random(seed)
        observation = np.array([0.0], dtype=np.float32)
        info: Dict = {}
        return observation, info

    def step(self, action_idx: int):
        """Apply the chosen configuration and return (obs, reward, done, truncated, info)."""
        # Map action index to CPU, memory, heap strings
        action = self.actions[action_idx]

        # Penalise impossible configurations where heap > memory
        if parse_memory_to_mb(action.heap) > parse_memory_to_mb(action.memory):
            reward = -float('inf')
            observation = np.array([0.0], dtype=np.float32)
            return observation, reward, True, False, {}

        cpu = action.cpu
        memory = action.memory
        heap = action.heap

        # Build JVM arguments: match min and max heap sizes
        jvm_args = f"-Xms{heap} -Xmx{heap}"

        # Clean up any existing stack and sidecar
        run_cmd(["docker", "compose", "down", "-v"])
        run_cmd(["docker", "rm", "-f", "sidecar"])  # ignore if not found

        # Start stack with the given JVM arguments
        run_cmd(["docker", "compose", "up", "-d"], env={"JVM_ARGS": jvm_args})

        # Update CPU and memory limits for the app container
        mem_lower = memory.lower()
        run_cmd(["docker", "update", "--cpus", cpu, "--memory", mem_lower, "app"])

        # Remove any existing sidecar instance and start a fresh one with labels
        run_cmd(["docker", "rm", "-f", "sidecar"])  # ignore if not found
        # Determine the default network for compose
        network = self._get_default_network()
        run_cmd([
            "docker",
            "run",
            "-d",
            "--name",
            "sidecar",
            "-p",
            "9100:9100",
            "-e",
            "APP_NAME=acmeair",
            "-e",
            "HEALTH_URL=http://app:9080/health/ready",
            "-e",
            f"CPUS={cpu}",
            "-e",
            f"MEMORY={memory}",
            "-e",
            f"HEAP={heap}",
            "--network",
            network,
            "project-sidecar:latest",
        ])

        # Wait for the application to become ready
        ready = wait_for_readiness(self.health_url, timeout=300)
        if not ready:
            reward = -float('inf')
            observation = np.array([0.0], dtype=np.float32)
            return observation, reward, True, False, {}

        # Give Prometheus time to scrape
        time.sleep(10)

        # Query the startup time via Prometheus
        try:
            startup_time = query_startup_time(cpu, memory, heap, self.prometheus_query_url)
            reward = -startup_time
        except Exception:
            reward = -float('inf')

        # Environment is one‑step, so done=True
        observation = np.array([0.0], dtype=np.float32)
        done = True
        truncated = False
        info: Dict = {}
        return observation, reward, done, truncated, info

    def render(self):
        return  # No rendering

    def _get_default_network(self) -> str:
        """Return the first compose network ending with '_default'."""
        rc, out, err = run_cmd(["docker", "network", "ls", "--format", "{{.Name}}"], capture=True)
        if rc == 0:
            for name in out.strip().splitlines():
                if name.endswith("_default"):
                    return name
        return "project_default"


def load_actions_from_csv(csv_path: str) -> List[Action]:
    """Load unique (cpu, memory, heap) combinations from the CSV."""
    df = pd.read_csv(csv_path)
    required_cols = {"cpus", "memory", "heap"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}")
    actions = []
    # Drop duplicates to avoid repeated actions
    unique_rows = df.drop_duplicates(subset=["cpus", "memory", "heap"])
    for _, row in unique_rows.iterrows():
        actions.append(Action(cpu=str(row["cpus"]), memory=str(row["memory"]), heap=str(row["heap"])))
    return actions


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained RL model with the AcmeAir environment.")
    parser.add_argument(
        "--model",
        choices=["dqn", "ppo"],
        required=True,
        help="Type of model to load (dqn or ppo)",
    )
    parser.add_argument(
        "--model-file",
        required=True,
        help="Path to the saved model file (.zip) produced by train_rl.py",
    )
    parser.add_argument(
        "--csv",
        default="startup_data.csv",
        help="Path to the CSV file containing historical startup data",
    )
    args = parser.parse_args()

    # Load actions from CSV
    actions = load_actions_from_csv(args.csv)
    # Create environment
    env = DockerStartupEnv(actions)

    # Dynamically import stable‑baselines3 and gymnasium
    try:
        from stable_baselines3 import DQN, PPO  # type: ignore
    except ImportError:
        raise ImportError(
            "stable‑baselines3 is required to load and use the trained models.\n"
            "Install it via `pip install stable-baselines3[extra]`"
        )

    # Load the selected model
    if args.model == "dqn":
        model = DQN.load(args.model_file, env=env)
    else:
        model = PPO.load(args.model_file, env=env)

    # Reset environment (dummy)
    obs, _ = env.reset()
    # Ask the model to predict an action
    action, _ = model.predict(obs, deterministic=True)
    # Apply action in environment
    obs, reward, done, truncated, info = env.step(int(action))

    # Convert negative reward back to positive startup time
    startup_time = -float(reward)

    selected_action = actions[int(action)]
    print("Selected configuration:\n" f"  CPU cores: {selected_action.cpu}\n"
          f"  Memory: {selected_action.memory}\n"
          f"  Heap: {selected_action.heap}")
    print(f"Measured startup time: {startup_time:.3f} seconds")


if __name__ == "__main__":
    main()