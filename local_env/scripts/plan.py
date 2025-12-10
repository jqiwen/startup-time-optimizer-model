import os
import subprocess
import sys
import time
import requests
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.utils import get_schedule_fn
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

DEFAULT_STARTUP_MAX_SECONDS = 60.0
APP_CONTAINER = "app"
SIDECAR_CONTAINER = "sidecar"

HEALTH_URL_HOST = "http://localhost:9080/local_app/"
HEALTH_URL_CONTAINER = "http://app:9080/local_app/"
PROMETHEUS_QUERY_URL = "http://localhost:9090/api/v1/query"

class Action:
    cpus: str 
    memory: str 
    heap: str

def format_mem(mem):
        mem = str(mem).strip().upper()
        if mem.endswith("G"):
            return int(float(mem[:-1]) * 1024)
        if mem.endswith("M"):
            return int(float(mem[:-1]))
        return int(float(mem))

def docker_cmd(cmd, env = None, capture = False):
    env_vars = os.environ.copy()
    if env:
        env_vars.update(env)
    try:
        result = subprocess.run(cmd, text=True, capture_output=capture, check=False, env=env_vars)
        if capture:
            stdout_result = result.stdout
            stderr_result = result.stderr
        else:
            stdout_result = ""
            stderr_result = ""
        return result.returncode, stdout_result, stderr_result
    except FileNotFoundError:
        sys.exit(1)

def get_default_network():
    rc, out = docker_cmd(["docker", "network", "ls", "--format", "{{.Name}}"], capture=True)
    if rc != 0:
        return "project_default"
    for line in out.strip().splitlines():
        if line.endswith("_default"):
            return line
    return "project_default"

def stop_stack():
    print("Stopping stack…")
    docker_cmd(["docker", "compose", "down", "-v"])
    docker_cmd(["docker", "rm", "-f", SIDECAR_CONTAINER])

def start_stack(jvm_args):
    print(f"Starting stack with JVM_ARGS='{jvm_args}'…")
    docker_cmd(["docker", "compose", "up", "-d"], env={"JVM_ARGS": jvm_args})


def configure_app_resources(cpus, memory):
    mem_lower = memory.lower()
    print(f"Updating app resources: cpus={cpus}, memory={mem_lower}")

    rc, _, err = docker_cmd(
        [
            "docker", "update",
            "--cpus", cpus,
            "--memory", mem_lower,
            "--memory-swap", mem_lower,
            APP_CONTAINER,
        ],
        capture=True,
    )

def restart_sidecar(cpus, memory, heap, network):
    print(f"Restarting sidecar: cpus={cpus}, memory={memory}, heap={heap}")
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
    docker_cmd(cmd)

def wait_for_readiness(url, timeout = 300):
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(url, timeout=3)
            if resp.status_code < 500:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False

def query_prometheus_startup(cpus, memory, heap, timeout = 120):
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

class OfflineStartupEnv(gym.Env):
    metadata = {"render.modes": []}
    def __init__(self, actions, startup_df, baseline_startup, normalize_reward,):
        super().__init__()

        self.actions = actions
        self.baseline_startup = baseline_startup
        self.normalize_reward = normalize_reward

        heap_col = "heap"
        if heap_col not in startup_df.columns and "heap_size" in startup_df.columns:
            heap_col = "heap_size"

        self.action_to_times= {}
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

        cpu_vals = [float(a.cpus) for a in actions]
        mem_vals = [format_mem(a.memory) for a in actions]
        heap_vals = [format_mem(a.heap) for a in actions]

        self.cpu_min, self.cpu_max = min(cpu_vals), max(cpu_vals)
        self.mem_min, self.mem_max = min(mem_vals), max(mem_vals)
        self.heap_min, self.heap_max = min(heap_vals), max(heap_vals)

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(4,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(len(actions))

    def get_obs(self, idx):
        action = self.actions[idx]
        cpu = float(action.cpus)
        mem_mb = format_mem(action.memory)
        heap_mb = format_mem(action.heap)

        cpu_norm = (cpu - self.cpu_min) / (self.cpu_max - self.cpu_min + 1e-9)
        mem_norm = (mem_mb - self.mem_min) / (self.mem_max - self.mem_min + 1e-9)
        heap_norm = (heap_mb - self.heap_min) / (self.heap_max - self.heap_min + 1e-9)
        baseline_norm = min(self.baseline_startup / DEFAULT_STARTUP_MAX_SECONDS, 1.0)

        return np.array(
            [cpu_norm, mem_norm, heap_norm, baseline_norm],
            dtype=np.float32,
        )

    def reset(self, *, seed = None):
        super().reset(seed=seed)
        idx = random.randrange(len(self.actions))
        obs = self.get_obs(idx)
        return obs, {}

    def step(self, action_idx):
        idx = int(action_idx)

        startup = self.baseline_startup * 1.5

        raw_reward = self.baseline_startup - startup
        if self.normalize_reward:
            reward = raw_reward / max(self.baseline_startup, 1e-6)
        else:
            reward = raw_reward

        obs = self.get_obs(idx)
        terminated = True
        truncated = False
        info = {"startup_seconds": startup}
        return obs, reward, terminated, truncated, info
    
class HierarchicalStartupEnv(gym.Env):
    metadata = {"render.modes": []}
    def __init__(self, actions, base_env, ppo_model):
        super().__init__()
        self.actions = actions
        self.base_env = base_env
        self.ppo_model = ppo_model

        self.action_space = spaces.Discrete(8)
        self.observation_space = base_env.observation_space

        self.config_to_idx = {}
        for idx, a in enumerate(actions):
            key = (str(a.cpus), str(a.memory), str(a.heap))
            self.config_to_idx[key] = idx

        self.current_idx = 0

    def high_level_action(self, hi_action: int, curr: Action, ppo_cfg: Action) -> Action:
        if hi_action == 0:  # CPU only
            return Action(cpus=ppo_cfg.cpus, memory=curr.memory, heap=curr.heap)
        elif hi_action == 1:  # Memory only
            return Action(cpus=curr.cpus, memory=ppo_cfg.memory, heap=curr.heap)
        elif hi_action == 2:  # Heap only
            return Action(cpus=curr.cpus, memory=curr.memory, heap=ppo_cfg.heap)
        elif hi_action == 3:  # Memory + Heap
            return Action(cpus=curr.cpus, memory=ppo_cfg.memory, heap=ppo_cfg.heap)
        elif hi_action == 4: #CPU + Memory
            return Action(cpus=ppo_cfg.cpus, memory=ppo_cfg.memory, heap=curr.heap)
        elif hi_action == 5: #CPU + Heap
            return Action(cpus=ppo_cfg.cpus, memory=curr.memory, heap=ppo_cfg.heap)
        elif hi_action == 6: #CPU + Memory + Heap
            return Action(cpus=ppo_cfg.cpus, memory=ppo_cfg.memory, heap=ppo_cfg.heap)
        elif hi_action == 7: #None
            return Action(cpus=curr.cpus, memory=curr.memory, heap=curr.heap)
        else:
            # take PPO config directly
            return ppo_cfg

    def reset(self, *, seed = None):
        super().reset(seed=seed)
        self.base_env.reset()
        self.current_idx = random.randrange(len(self.actions))
        obs = self.base_env.get_obs(self.current_idx)
        return obs, {}

    def step(self, hi_action):
        hi_action = int(hi_action)

        obs = self.base_env.get_obs(self.current_idx)

        ppo_action_idx, _ = self.ppo_model.predict(obs, deterministic=True)
        ppo_action_idx = int(ppo_action_idx)
        ppo_cfg = self.actions[ppo_action_idx]

        curr_cfg = self.actions[self.current_idx]
        new_cfg = self.high_level_action(hi_action, curr_cfg, ppo_cfg)

        key = (str(new_cfg.cpus), str(new_cfg.memory), str(new_cfg.heap))
        new_idx = self.config_to_idx.get(key, ppo_action_idx)

        obs2, reward, terminated, truncated, info = self.base_env.step(new_idx)

        if not isinstance(info, dict):
            info = {}
        info = dict(info)
        info.setdefault("config", (new_cfg.cpus, new_cfg.memory, new_cfg.heap))
        info.setdefault("config_idx", new_idx)

        self.current_idx = new_idx

        return obs2, reward, terminated, truncated, info
    
class RealStartupEnv(gym.Env):
    INVALID_REWARD = -100.0
    metadata = {"render.modes": []}
    def __init__(self, actions, baseline_startup, normalize_reward = True):
        super().__init__()

        self.actions = actions
        self.baseline_startup = baseline_startup
        self.normalize_reward = normalize_reward
        self.network = get_default_network()

        cpu_vals = [float(a.cpus) for a in actions]
        mem_vals = [format_mem(a.memory) for a in actions]
        heap_vals = [format_mem(a.heap) for a in actions]

        self.cpu_min, self.cpu_max = min(cpu_vals), max(cpu_vals)
        self.mem_min, self.mem_max = min(mem_vals), max(mem_vals)
        self.heap_min, self.heap_max = min(heap_vals), max(heap_vals)

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(actions))

    def get_obs(self, idx):
        action = self.actions[idx]
        cpu = float(action.cpus)
        mem_mb = format_mem(action.memory)
        heap_mb = format_mem(action.heap)

        cpu_norm = (cpu - self.cpu_min) / (self.cpu_max - self.cpu_min + 1e-9)
        mem_norm = (mem_mb - self.mem_min) / (self.mem_max - self.mem_min + 1e-9)
        heap_norm = (heap_mb - self.heap_min) / (self.heap_max - self.heap_min + 1e-9)
        baseline_norm = min(self.baseline_startup / DEFAULT_STARTUP_MAX_SECONDS, 1.0)

        return np.array([cpu_norm, mem_norm, heap_norm, baseline_norm], dtype=np.float32)

    def reset(self, *, seed = None):
        super().reset(seed=seed)
        idx = random.randrange(len(self.actions))
        obs = self.get_obs(idx)
        return obs, {}

    def step(self, action_idx):
        idx = int(action_idx)
        action = self.actions[idx]
        cpus, memory, heap = action.cpus, action.memory, action.heap

        if format_mem(heap) > format_mem(memory):
            obs = self.get_obs(idx)
            return obs, self.INVALID_REWARD, True, False, {"startup_seconds": None}

        try:
            stop_stack()
            jvm_args = f"-Xms{heap} -Xmx{heap}"
            start_stack(jvm_args)
            configure_app_resources(cpus, memory)
            restart_sidecar(cpus, memory, heap, self.network)
            ready = wait_for_readiness(HEALTH_URL_HOST, timeout=300)
            if not ready:
                print(f"Application not ready for {action}")
                obs = self.get_obs(idx)
                return obs, self.INVALID_REWARD, True, False, {"startup_seconds": None}

            time.sleep(10)

            startup = query_prometheus_startup(cpus, memory, heap)
            print(f"Measured startup: {startup:.3f} seconds")

            raw_reward = self.baseline_startup - startup
            if self.normalize_reward:
                reward = raw_reward / max(self.baseline_startup, 1e-6)
            else:
                reward = raw_reward

            obs = self.get_obs(idx)
            info = {"startup_seconds": startup}
        except Exception:
            obs = self.get_obs(idx)
            reward = self.INVALID_REWARD
            info = {"startup_seconds": None}

        return obs, reward, True, False, info

class Plan:
    def build_offline_envs(self, actions, filtered_df, baseline_startup,normalize_reward):
        offline_env = OfflineStartupEnv(
            actions=actions,
            startup_df=filtered_df,
            baseline_startup=baseline_startup,
            normalize_reward=normalize_reward,
        )

        dummy_ppo = PPO(
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
            verbose=0,
        )

        hier_offline_env = HierarchicalStartupEnv(
            actions=actions,
            base_env=offline_env,
            ppo_model=dummy_ppo,
        )

        return offline_env, hier_offline_env

    def build_models(self, offline_env, hier_offline_env):
        ppo_model = PPO(
            "MlpPolicy",
            offline_env,
            learning_rate=5e-5,
            n_steps=512,
            batch_size=256,
            n_epochs=10,
            gamma=0.995,
            gae_lambda=0.96,
            clip_range=get_schedule_fn(0.1),
            ent_coef=0.0,
            verbose=1,
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

        return ppo_model, dqn_model

    def build_online_envs(self, actions, baseline_startup, normalize_reward, ppo_model):
        real_env = RealStartupEnv(
            actions=actions,
            baseline_startup=baseline_startup,
            normalize_reward=normalize_reward,
        )

        hier_real_env = HierarchicalStartupEnv(
            actions=actions,
            base_env=real_env,
            ppo_model=ppo_model,
        )

        return real_env, hier_real_env
