#!/usr/bin/env python3
"""
k8s_rl_trainer.py

Online RL training on a real Kubernetes / IBM Cloud deployment using PPO
with a continuous 3D action space:

  action = [a_cpu, a_mem, a_heap] in [0,1]^3

mapped to:

  cpu  in [cpu_min,  cpu_max]  (cores)
  mem  in [mem_min,  mem_max]  (MB)
  heap in [heap_min, heap_max] (MB)

At each step:
  1) Map action -> (cpu, memory, heap).
  2) `kubectl set resources` + `kubectl set env`.
  3) `kubectl rollout restart` + `kubectl rollout status`.
  4) Measure startup_seconds = elapsed time.
  5) Reward = (baseline - startup) / baseline.

Baseline:
  - Automatically measured from the current production config when the script starts.

Env:
  - 1-state bandit-like environment (no temporal dynamics).
  - `done = False` each step so PPO sees a continuous stream of transitions.

Outputs:
  - Trained PPO model: model_results/model.zip
  - Best config + timings: /workspace/optimized_resources.json
  - Training plots in /workspace:
      * startup_time.png  (raw + moving average)
      * reward_curve.png  (raw + moving average)
      * loss_curve.png    (raw + moving average)
  - Evaluation plot in /workspace:
      * startup_time_eval.png  (deterministic policy after training)

Usage:

    python k8s_rl_trainer.py ^
      --config config/acmeair-mainservice.yaml ^
      --timesteps 50 ^
      --baseline-runs 3 ^
      --model-out model_results/model.zip
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import yaml

# Gym (old API) for SB3; SB3 will wrap with Gymnasium via shimmy internally
from gym import Env, spaces  # gym <= 0.21 recommended

from stable_baselines3 import PPO  # type: ignore

# matplotlib only for plots
try:
    import matplotlib.pyplot as plt  # type: ignore
except ImportError:
    plt = None


# ==============================================================================
# Config and helpers
# ==============================================================================

@dataclass
class AppConfig:
    app_name: str
    namespace: str
    deployment: str

    heap_env_name: str
    heap_template: str

    cpu_min: float
    cpu_max: float
    mem_min_mb: float
    mem_max_mb: float
    heap_min_mb: float
    heap_max_mb: float

    baseline_startup: float = 0.0  # will be overwritten by measurement

    @staticmethod
    def from_yaml(path: str) -> "AppConfig":
        """
        Expect YAML schema:

        app_name: acmeair
        namespace: acmeair-group11
        deployment: acmeair-mainservice

        heap_env:
          name: JVM_ARGS
          template: "-Xms{heap} -Xmx{heap}"

        resources:
          cpu:
            min: 0.5
            max: 2.0
          memory:
            min: "512Mi"
            max: "2Gi"
          heap:
            min: "256M"
            max: "1G"

        baseline_startup: 0.0   # optional, ignored for baseline measurement
        """
        with open(path, "r", encoding="utf-8") as f:
            docs = list(yaml.safe_load_all(f))
        if len(docs) != 1:
            raise ValueError(
                f"{path} must contain exactly one YAML document (found {len(docs)})"
            )
        cfg = docs[0]

        resources = cfg["resources"]
        cpu_range = resources["cpu"]
        mem_range = resources["memory"]
        heap_range = resources["heap"]

        cpu_min = float(cpu_range["min"])
        cpu_max = float(cpu_range["max"])
        mem_min_mb = parse_memory_to_mb(mem_range["min"])
        mem_max_mb = parse_memory_to_mb(mem_range["max"])
        heap_min_mb = parse_memory_to_mb(heap_range["min"])
        heap_max_mb = parse_memory_to_mb(heap_range["max"])

        return AppConfig(
            app_name=str(cfg.get("app_name", "")),
            namespace=str(cfg["namespace"]),
            deployment=str(cfg["deployment"]),
            heap_env_name=str(cfg["heap_env"]["name"]),
            heap_template=str(cfg["heap_env"]["template"]),
            cpu_min=cpu_min,
            cpu_max=cpu_max,
            mem_min_mb=mem_min_mb,
            mem_max_mb=mem_max_mb,
            heap_min_mb=heap_min_mb,
            heap_max_mb=heap_max_mb,
            baseline_startup=float(cfg.get("baseline_startup", 0.0)),
        )


def parse_memory_to_mb(s: str) -> float:
    """
    Convert strings like "512M", "512Mi", "1G", "2Gi" to MB.
    """
    x = s.strip().upper()
    x = x.replace("MI", "M").replace("GI", "G")
    if x.endswith("G"):
        return float(x[:-1]) * 1024.0
    if x.endswith("M"):
        return float(x[:-1])
    return float(x)


def mb_to_str(mb: float) -> str:
    """
    Convert MB float back to a Kubernetes memory string.
    Use Gi if it's a clean multiple of 1024, otherwise Mi.
    """
    if abs(mb / 1024.0 - round(mb / 1024.0)) < 1e-6:
        return f"{int(round(mb / 1024.0))}Gi"
    return f"{int(round(mb))}Mi"


def run(cmd: List[str]) -> None:
    print("[CMD]", " ".join(cmd))
    subprocess.run(cmd, check=True, text=True)


def capture(cmd: List[str]) -> str:
    print("[CAPTURE]", " ".join(cmd))
    out = subprocess.check_output(cmd, text=True)
    return out.strip()


def parse_heap_from_env(env_value: str) -> Optional[str]:
    """
    Try to extract heap size like '512M' or '1G' from something like '-Xms512M -Xmx512M'.
    """
    if not env_value:
        return None
    s = env_value.upper()
    m = re.search(r"-XM[X|S]([0-9]+[MG])", s)
    if m:
        return m.group(1)
    m2 = re.search(r"([0-9]+[MG])", s)
    if m2:
        return m2.group(1)
    return None


# ==============================================================================
# Baseline (production) config and measurements
# ==============================================================================

@dataclass
class Action:
    cpu: float      # cores
    mem_mb: float   # MB
    heap_mb: float  # MB


def get_production_config(cfg: AppConfig) -> Action:
    """
    Read the *current* Deployment's limits/env to infer production config.
    """
    def_mid = lambda lo, hi: 0.5 * (lo + hi)
    cpu = def_mid(cfg.cpu_min, cfg.cpu_max)
    mem_mb = def_mid(cfg.mem_min_mb, cfg.mem_max_mb)
    heap_mb = def_mid(cfg.heap_min_mb, cfg.heap_max_mb)

    # CPU limit
    try:
        cpu_str = capture([
            "kubectl", "-n", cfg.namespace,
            "get", "deploy", cfg.deployment,
            "-o", "jsonpath={.spec.template.spec.containers[0].resources.limits.cpu}",
        ])
        if cpu_str:
            cpu = float(cpu_str)
    except Exception as e:
        print(f"[WARN] Failed to read CPU limit; using default mid-range: {e}")

    # memory limit
    try:
        mem_str = capture([
            "kubectl", "-n", cfg.namespace,
            "get", "deploy", cfg.deployment,
            "-o", "jsonpath={.spec.template.spec.containers[0].resources.limits.memory}",
        ])
        if mem_str:
            mem_mb = parse_memory_to_mb(mem_str)
    except Exception as e:
        print(f"[WARN] Failed to read memory limit; using default mid-range: {e}")

    # heap env
    try:
        heap_env_val = capture([
            "kubectl", "-n", cfg.namespace,
            "get", "deploy", cfg.deployment,
            "-o", f"jsonpath={{.spec.template.spec.containers[0].env[?(@.name=='{cfg.heap_env_name}')].value}}",
        ])
        parsed_heap = parse_heap_from_env(heap_env_val)
        if parsed_heap:
            heap_mb = parse_memory_to_mb(parsed_heap)
    except Exception as e:
        print(f"[WARN] Failed to read heap env; using default mid-range: {e}")

    print(f"[INFO] Production config: cpu={cpu}, mem={mem_mb}MB, heap={heap_mb}MB")
    return Action(cpu=cpu, mem_mb=mem_mb, heap_mb=heap_mb)


def measure_startup(action: Action, cfg: AppConfig, timeout: int = 600) -> float:
    """
    Apply given (cpu, mem_mb, heap_mb) to deployment, restart, and time rollout.
    """
    cpu_str = f"{action.cpu}"
    mem_str = mb_to_str(action.mem_mb)
    heap_str = mb_to_str(action.heap_mb)

    # Set resources: limits and requests
    run([
        "kubectl", "-n", cfg.namespace,
        "set", "resources", "deploy", cfg.deployment,
        f"--limits=cpu={cpu_str},memory={mem_str}",
        f"--requests=cpu={cpu_str},memory={mem_str}",
    ])

    # Set heap env
    env_val = cfg.heap_template.format(heap=heap_str)
    run([
        "kubectl", "-n", cfg.namespace,
        "set", "env", "deploy", cfg.deployment,
        f"{cfg.heap_env_name}={env_val}",
    ])

    start = time.monotonic()
    run([
        "kubectl", "-n", cfg.namespace,
        "rollout", "restart", f"deploy/{cfg.deployment}",
    ])
    run([
        "kubectl", "-n", cfg.namespace,
        "rollout", "status", f"deploy/{cfg.deployment}",
        f"--timeout={timeout}s",
    ])
    end = time.monotonic()
    startup = end - start
    print(f"[K8S] Startup time: {startup:.3f}s")
    return startup


# ==============================================================================
# Gym environment for real K8s
# ==============================================================================

class K8sStartupEnv(Env):
    """
    Continuous bandit-like environment for PPO:

      action: Box(0,1, shape=(3,))  -> [a_cpu, a_mem, a_heap]
    mapped linearly to resource ranges.

      observation: Box(0,1, shape=(4,))
         [cpu_norm, mem_norm, heap_norm, baseline_norm]
    """

    metadata = {"render.modes": []}

    def __init__(self, cfg: AppConfig):
        super().__init__()
        self.cfg = cfg

        # Continuous action in [0,1]^3
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Observation: [cpu_norm, mem_norm, heap_norm, baseline_norm]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(4,), dtype=np.float32
        )

        self.baseline = cfg.baseline_startup  # set later

        self.history_startup: List[float] = []
        self.history_reward: List[float] = []

        self.best_action: Optional[Action] = None
        self.best_startup: float = float("inf")

        self.last_action: Optional[Action] = None

    # ---------- Gym / Gymnasium compatibility ----------

    def seed(self, seed: Optional[int] = None):
        """
        Called by Gymnasium shim (shimmy). We just seed NumPy.
        """
        if seed is not None:
            np.random.seed(seed)
        return [seed]

    # ---------- Helpers ----------

    def _scale_action(self, a: np.ndarray) -> Action:
        a = np.clip(a, 0.0, 1.0)
        cpu = self.cfg.cpu_min + a[0] * (self.cfg.cpu_max - self.cfg.cpu_min)
        mem_mb = self.cfg.mem_min_mb + a[1] * (self.cfg.mem_max_mb - self.cfg.mem_min_mb)
        heap_mb = self.cfg.heap_min_mb + a[2] * (self.cfg.heap_max_mb - self.cfg.heap_min_mb)
        return Action(cpu=cpu, mem_mb=mem_mb, heap_mb=heap_mb)

    def _build_obs(self, act: Action) -> np.ndarray:
        def norm(x, lo, hi):
            return float((x - lo) / (hi - lo + 1e-9))

        cpu_norm = norm(act.cpu, self.cfg.cpu_min, self.cfg.cpu_max)
        mem_norm = norm(act.mem_mb, self.cfg.mem_min_mb, self.cfg.mem_max_mb)
        heap_norm = norm(act.heap_mb, self.cfg.heap_min_mb, self.cfg.heap_max_mb)
        baseline_norm = min(self.baseline / 60.0, 1.0)

        return np.array(
            [cpu_norm, mem_norm, heap_norm, baseline_norm], dtype=np.float32
        )

    # ---------- Gym API ----------

    def reset(self):
        """
        Old-style Gym reset() signature.
        Gymnasium/shimmy will call env.seed(seed) separately.
        """
        prod = get_production_config(self.cfg)
        self.last_action = prod
        obs = self._build_obs(prod)
        return obs

    def step(self, action: np.ndarray):
        act = self._scale_action(action)
        self.last_action = act

        startup = measure_startup(act, self.cfg)
        reward = (self.baseline - startup) / max(self.baseline, 1e-6)

        self.history_startup.append(startup)
        self.history_reward.append(reward)

        if startup < self.best_startup:
            self.best_startup = startup
            self.best_action = act
            print("[BEST] New best config found.")

        obs = self._build_obs(act)

        # Keep episode running; PPO just needs a stream of transitions.
        done = False
        info = {"startup": startup, "reward": reward}
        return obs, reward, done, info


# ==============================================================================
# Evaluation (deterministic policy after training)
# ==============================================================================

def evaluate_policy_on_cluster(model: PPO, cfg: AppConfig, baseline: float, n_eval_steps: int = 100) -> List[float]:
    """
    Run the trained policy deterministically on the cluster for n_eval_steps,
    logging startup times. This does NOT update the model (pure evaluation).
    """
    env = K8sStartupEnv(cfg)
    env.baseline = baseline

    obs = env.reset()
    eval_startups: List[float] = []

    for i in range(n_eval_steps):
        print(f"[EVAL] Step {i + 1}/{n_eval_steps}")
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        startup = float(info["startup"])
        eval_startups.append(startup)
        print(f"[EVAL] Startup={startup:.3f}s, reward={reward:.3f}")

    return eval_startups


# ==============================================================================
# Plotting
# ==============================================================================

def moving_average(arr: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple moving average helper for smoothing plots.
    Returns (x_smooth, y_smooth).
    """
    if len(arr) < window:
        xs = np.arange(1, len(arr) + 1)
        return xs, arr
    kernel = np.ones(window) / window
    y = np.convolve(arr, kernel, mode="valid")
    xs = np.arange(window, len(arr) + 1)
    return xs, y


def save_training_plots(env: K8sStartupEnv, output_dir: str, ma_window: int = 5) -> None:
    if plt is None:
        print("[PLOT] matplotlib not installed; skipping training plots.")
        return

    os.makedirs(output_dir, exist_ok=True)
    xs = np.arange(1, len(env.history_startup) + 1)
    startups = np.array(env.history_startup, dtype=float)
    rewards = np.array(env.history_reward, dtype=float)
    losses = -rewards

    # ---------- helper: moving average ----------
    def moving_average(arr: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
        if len(arr) < window:
            return xs, arr
        kernel = np.ones(window) / window
        y = np.convolve(arr, kernel, mode="valid")
        x_new = np.arange(window, len(arr) + 1)
        return x_new, y

    # ---------- best-so-far curve ----------
    best_so_far = np.minimum.accumulate(startups)

    # Startup time (raw + MA + best-so-far)
    plt.figure()
    # raw
    plt.plot(xs, startups, marker=".", linewidth=0.5, alpha=0.3, label="Raw")
    # moving average
    xs_ma, startups_ma = moving_average(startups, ma_window)
    plt.plot(xs_ma, startups_ma, linewidth=2, label=f"MA (window={ma_window})")
    # best so far
    plt.plot(xs, best_so_far, linewidth=2, linestyle="-.", label="Best so far")
    # baseline
    plt.axhline(env.baseline, linestyle="--", label="Baseline")
    plt.xlabel("Step")
    plt.ylabel("Startup time (s)")
    plt.title("Startup time during PPO training")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, "startup_time.png")
    plt.savefig(path)
    plt.close()
    print(f"[PLOT] Saved training startup_time.png to {path}")

    # Reward (raw + MA)
    plt.figure()
    plt.plot(xs, rewards, marker=".", linewidth=0.5, alpha=0.3, label="Raw")
    xs_ma_r, rewards_ma = moving_average(rewards, ma_window)
    plt.plot(xs_ma_r, rewards_ma, linewidth=2, label=f"MA (window={ma_window})")
    plt.axhline(0.0, linestyle="--", label="Zero reward")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.title("Reward during PPO training")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, "reward_curve.png")
    plt.savefig(path)
    plt.close()
    print(f"[PLOT] Saved training reward_curve.png to {path}")

    # Loss (= -reward, raw + MA)
    plt.figure()
    plt.plot(xs, losses, marker=".", linewidth=0.5, alpha=0.3, label="Raw")
    xs_ma_l, losses_ma = moving_average(losses, ma_window)
    plt.plot(xs_ma_l, losses_ma, linewidth=2, label=f"MA (window={ma_window})")
    plt.axhline(0.0, linestyle="--", label="Zero loss")
    plt.xlabel("Step")
    plt.ylabel("Loss (= -reward)")
    plt.title("Loss during PPO training (lower is better)")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, "loss_curve.png")
    plt.savefig(path)
    plt.close()
    print(f"[PLOT] Saved training loss_curve.png to {path}")



def save_eval_plot(eval_startups: List[float], baseline: float, output_dir: str) -> None:
    if plt is None:
        print("[PLOT] matplotlib not installed; skipping eval plot.")
        return

    os.makedirs(output_dir, exist_ok=True)
    xs = np.arange(1, len(eval_startups) + 1)
    startups = np.array(eval_startups, dtype=float)

    plt.figure()
    plt.plot(xs, startups, marker="o", linewidth=1.5, label="Eval startup time")
    plt.axhline(baseline, linestyle="--", label="Baseline")
    plt.xlabel("Eval step")
    plt.ylabel("Startup time (s)")
    plt.title("Startup time during evaluation (deterministic policy)")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, "startup_time_eval.png")
    plt.savefig(path)
    plt.close()
    print(f"[PLOT] Saved evaluation startup_time_eval.png to {path}")


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Online PPO training for K8s startup optimization (continuous actions)."
    )
    parser.add_argument(
        "--config", required=True, help="Optimizer config YAML (ranges only)."
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=50,
        help="Total PPO training timesteps (each step = one rollout).",
    )
    parser.add_argument(
        "--baseline-runs",
        type=int,
        default=3,
        help="How many runs of the production config to estimate baseline.",
    )
    parser.add_argument(
        "--model-out",
        type=str,
        default="model_results/model.zip",
        help="Where to save the trained PPO model.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for PPO.",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=10,
        help="Number of deterministic evaluation rollouts after training.",
    )

    args = parser.parse_args()

    cfg = AppConfig.from_yaml(args.config)
    print(f"[INFO] Loaded config for app '{cfg.app_name}' "
          f"in namespace '{cfg.namespace}', deployment '{cfg.deployment}'")

    # 1) Determine production config and baseline
    prod_cfg = get_production_config(cfg)
    baseline_samples = []
    print(f"[INFO] Measuring baseline startup over {args.baseline_runs} run(s)...")
    for i in range(args.baseline_runs):
        print(f"[BASELINE] Run {i + 1}/{args.baseline_runs}")
        t = measure_startup(prod_cfg, cfg)
        baseline_samples.append(t)

    baseline = float(np.median(baseline_samples))
    cfg.baseline_startup = baseline
    print(f"[INFO] Baseline startup (median): {baseline:.3f}s")

    # 2) Create environment
    env = K8sStartupEnv(cfg)
    env.baseline = baseline

    # 3) PPO training with more conservative hyperparameters
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=1e-4,   # smaller LR for smoother updates
        gamma=0.0,            # bandit-like reward, no temporal credit
        n_steps=8,            # collect 8 steps per update (>=2)
        batch_size=8,         # match n_steps
        clip_range=0.1,       # smaller clip for more conservative policy updates
        verbose=1,
        seed=args.seed,
    )

    print(f"[TRAIN] Starting PPO training for {args.timesteps} timesteps...")
    model.learn(total_timesteps=args.timesteps)
    print("[TRAIN] Training finished.")

    # 4) Save model
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    model.save(args.model_out)
    print(f"[MODEL] Saved PPO model to {args.model_out}")

    # 5) Determine best config from training history
    if env.best_action is None:
        print("[WARN] No best_action recorded; using production config as best.")
        best = prod_cfg
        best_startup = float(baseline)
    else:
        best = env.best_action
        best_startup = float(env.best_startup)

    best_result = {
        "app": str(cfg.app_name),
        "cpu": float(best.cpu),
        "memory": str(mb_to_str(float(best.mem_mb))),
        "heap": str(mb_to_str(float(best.heap_mb))),
        "startup_seconds": float(best_startup),
        "baseline_startup_seconds": float(baseline),
    }

    output_dir = "model_results"
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "optimized_resources.json")
    print("_______________________________________________________________________")
    print(json_path)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(best_result, f, indent=2)

    print("[SUMMARY] Best configuration discovered (training phase):")
    print(json.dumps(best_result, indent=2))
    print(f"[OUTPUT] Saved optimized_resources.json to {json_path}")

    # 6) Save training plots (with smoothing)
    save_training_plots(env, output_dir, ma_window=5)

    # 7) Evaluation phase with deterministic policy
    if args.eval_steps > 0:
        print(f"[EVAL] Running deterministic evaluation for {args.eval_steps} step(s)...")
        eval_startups = evaluate_policy_on_cluster(
            model, cfg, baseline=baseline, n_eval_steps=args.eval_steps
        )
        save_eval_plot(eval_startups, baseline, output_dir)

        print("[EVAL] Evaluation startup times:", eval_startups)
        print(f"[EVAL] Mean eval startup: {float(np.mean(eval_startups)):.3f}s")
    else:
        print("[EVAL] Skipping evaluation (eval_steps <= 0).")


if __name__ == "__main__":
    main()