# execute.py
import os
from typing import List

import matplotlib.pyplot as plt

from stable_baselines3 import PPO, DQN

# import train  # 复用 TrainingMetricsCallback, moving_average 等
from monitor import Monitor
from analysis import Analysis
from plan import Plan
from stable_baselines3.common.callbacks import BaseCallback 
import numpy as np

WINDOW_SIZE = 50


class TrainingMetricsCallback(BaseCallback):

    def __init__(self, max_steps, phase):
        super().__init__()
        self.max_steps = max_steps
        self.phase = phase
        self.rewards = []
        self.startup_times = []
        self.loss_proxy = []

    def _on_step(self):
        rewards = self.locals.get("rewards")
        infos = self.locals.get("infos")

        r = float(rewards[0]) if rewards is not None else float("nan")
        startup = None
        if infos is not None and len(infos) > 0 and isinstance(infos[0], dict):
            startup = infos[0].get("startup_seconds")

        self.rewards.append(r)
        self.startup_times.append(startup)
        self.loss_proxy.append(-r)

        if self.phase and self.num_timesteps % 1000 == 0:
            print(f"[{self.phase}] Step {self.num_timesteps}, reward={r:.3f}, startup={startup}")

        if self.max_steps is not None:
            return self.n_calls < self.max_steps
        return True



class Execute:

    def __init__( self, monitor, analysis, planner, model_results_dir, offline_steps, online_steps ) :
        self.monitor = monitor
        self.analysis = analysis
        self.planner = planner
        self.model_results_dir = model_results_dir
        self.offline_steps = offline_steps
        self.online_steps = online_steps


    def run(self, csv_arg) :
        startup_df = self.monitor.load_csv(csv_arg)
        actions, filtered_df, baseline_startup = self.analysis.build_actions_and_baseline(startup_df)
        offline_env, hier_offline_env = self.planner.build_offline_envs(
            actions,
            filtered_df,
            baseline_startup,
            self.analysis.normalize_reward,
        )

        ppo_model, dqn_model = self.planner.build_models(
            offline_env, hier_offline_env
        )
        hier_offline_env.ppo_model = ppo_model


        (
            ppo_offline_cb,
            dqn_offline_cb,
        ) = self.run_offline_training(
            ppo_model,
            dqn_model,
            offline_env,
            hier_offline_env,
        )

        real_env, hier_real_env = self.planner.build_online_envs(
            actions,
            baseline_startup,
            self.analysis.normalize_reward,
            ppo_model,
        )

        (
            ppo_online_cb,
            dqn_online_cb,
        ) = self.run_online_training(
            ppo_model,
            dqn_model,
            real_env,
            hier_real_env,
        )

        self._save_models(ppo_model, dqn_model)
        self._plot_curves(
            ppo_offline_cb,
            dqn_offline_cb,
            ppo_online_cb,
            dqn_online_cb,
            self.analysis.normalize_reward,
        )

        self._evaluate_final_policy(dqn_model, hier_real_env, self.analysis.normalize_reward)

    def run_offline_training(self, ppo_model, dqn_model,offline_env, hier_offline_env):
        print( f"Training PPO (low-level) offline for {self.offline_steps} timesteps…" )

        ppo_offline_cb = TrainingMetricsCallback(phase="PPO_offline")
        ppo_model.set_env(offline_env)
        ppo_model.learn(
            total_timesteps=self.offline_steps,
            progress_bar=False,
            callback=ppo_offline_cb,
        )

        print( f"Training DQN (high-level) offline for {self.offline_steps} timesteps…" )
        dqn_offline_cb = TrainingMetricsCallback(phase="DQN_offline")
        dqn_model.set_env(hier_offline_env)
        dqn_model.learn(
            total_timesteps=self.offline_steps,
            progress_bar=False,
            callback=dqn_offline_cb,
        )

        return ppo_offline_cb, dqn_offline_cb


    def run_online_training( self, ppo_model, dqn_model, real_env, hier_real_env ):
        print(f"Switching PPO low-level to real Docker environment for {self.online_steps} timesteps…" )

        ppo_model.set_env(real_env)
        ppo_model.learning_rate = 5e-5
        ppo_model.n_steps = 32
        ppo_model.batch_size = 16
        ppo_model.gamma = 0.999
        ppo_model.clip_range = 0.1

        ppo_online_cb = TrainingMetricsCallback(
            max_steps=self.online_steps,
            phase="PPO_online",
        )

        ppo_model.learn(
            total_timesteps=self.online_steps,
            reset_num_timesteps=False,
            callback=ppo_online_cb,
            progress_bar=False,
        )

        print( f"[PHASE B2] Switching DQN high-level to hierarchical real env for {self.online_steps} timesteps…" )

        dqn_model.set_env(hier_real_env)
        dqn_model.learning_rate = 1e-4
        dqn_model.buffer_size = 5000
        dqn_model.batch_size = 32
        dqn_model.exploration_fraction = 0.05
        dqn_model.exploration_final_eps = 0.02
        dqn_model.target_update_interval = 500
        dqn_model.gamma = 0.999

        dqn_online_cb = TrainingMetricsCallback(
            max_steps=self.online_steps,
            phase="DQN_online",
        )

        dqn_model.learn(
            total_timesteps=self.online_steps,
            reset_num_timesteps=False,
            callback=dqn_online_cb,
            progress_bar=False,
        )

        return ppo_online_cb, dqn_online_cb


    def _save_models(self, ppo_model, dqn_model):
        os.makedirs(self.model_results_dir, exist_ok=True)
        ppo_model_file = os.path.join(self.model_results_dir, "ppo_model.zip")
        dqn_model_file = os.path.join(self.model_results_dir, "dqn_model.zip")

        ppo_model.save(ppo_model_file)
        print(f"Saved final low-level PPO model to {ppo_model_file}")

        dqn_model.save(dqn_model_file)
        print(f"Saved final high-level DQN model to {dqn_model_file}")

    def moving_average(values, window):
        if not values:
            return []
        arr = np.asarray(values, dtype=float)
        if window <= 1 or window > len(arr):
            return arr
        kernel = np.ones(window) / window
        return np.convolve(arr, kernel, mode="valid")


    def _plot_curves( self, ppo_offline_cb, dqn_offline_cb, ppo_online_cb, dqn_online_cb, normalize_reward,):
        os.makedirs(self.model_results_dir, exist_ok=True)

        offline_series = [
            ("PPO", ppo_offline_cb),
            ("Hierarchical Model(PPO+DQN)", dqn_offline_cb),
        ]
        online_series = [
            ("PPO", ppo_online_cb),
            ("Hierarchical Model(PPO+DQN)", dqn_online_cb),
        ]

        def _smooth(values):
            if not values:
                return []
            window = max(5, len(values) // 100)
            return self.moving_average(values, window=window)

        plt.figure(figsize=(10, 6))
        for label, cb in offline_series:
            if cb.rewards:
                smoothed = _smooth(cb.rewards)
                steps = list(range(1, len(smoothed) + 1))
                plt.plot( steps, smoothed, linewidth=1.5, label=label )
        plt.xlabel("Timestep")
        plt.ylabel("Reward (normalized)" if normalize_reward else "Reward")
        plt.title("Reward curves(DB)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        reward_offline_path = os.path.join(self.model_results_dir, "reward_offline.png")
        plt.savefig(reward_offline_path)
        plt.close()
        print(f"Saved offline reward curves to {reward_offline_path}")

        plt.figure(figsize=(10, 6))
        for label, cb in online_series:
            if cb.rewards:
                smoothed = _smooth(cb.rewards)
                steps = list(range(1, len(smoothed) + 1))
                plt.plot(steps, smoothed, linewidth=1.5, label=label, )
        plt.xlabel("Timestep")
        plt.ylabel("Reward (normalized)" if normalize_reward else "Reward")
        plt.title("Reward curves(Loacl Env)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        reward_online_path = os.path.join(self.model_results_dir, "reward_online.png")
        plt.savefig(reward_online_path)
        plt.close()
        print(f"Saved online reward curves to {reward_online_path}")

        plt.figure(figsize=(10, 6))
        for label, cb in offline_series:
            if cb.loss_proxy:
                smoothed = _smooth(cb.loss_proxy)
                steps = list(range(1, len(smoothed) + 1))
                plt.plot(steps,smoothed, linewidth=1.5,label=label, )
        plt.xlabel("Timestep")
        plt.ylabel("Loss proxy (-reward)")
        plt.title("Loss curves(DB)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        loss_offline_path = os.path.join(self.model_results_dir, "loss_offline.png")
        plt.savefig(loss_offline_path)
        plt.close()
        print(f"Saved offline loss curves to {loss_offline_path}")

        plt.figure(figsize=(10, 6))
        for label, cb in online_series:
            if cb.loss_proxy:
                smoothed = _smooth(cb.loss_proxy)
                steps = list(range(1, len(smoothed) + 1))
                plt.plot( steps, smoothed, linewidth=1.5, label=label)
        plt.xlabel("Timestep")
        plt.ylabel("Loss proxy (-reward)")
        plt.title("Loss curves(Loacl Env)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        loss_online_path = os.path.join(self.model_results_dir, "loss_online.png")
        plt.savefig(loss_online_path)
        plt.close()
        print(f"Saved online loss curves to {loss_online_path}")

        plt.figure(figsize=(10, 6))
        for label, cb in offline_series:
            valid_points = []
            for i, s in enumerate(cb.startup_times):
                if s is None:
                    continue
                try:
                    val = float(s)
                except Exception:
                    continue
                valid_points.append((i + 1, val))

            if valid_points:
                _, s_vals = zip(*valid_points)
                smoothed_vals = _smooth(list(s_vals))
                smoothed_steps = list(range(1, len(smoothed_vals) + 1))
                plt.plot(
                    smoothed_steps,
                    smoothed_vals,
                    linewidth=1.5,
                    label=label,
                )
        plt.xlabel("Timestep")
        plt.ylabel("Startup time (s)")
        plt.title("Startup time curves(DB)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        startup_offline_path = os.path.join( self.model_results_dir, "startup_time_offline.png" )
        plt.savefig(startup_offline_path)
        plt.close()
        print(f"Saved offline startup time curves to {startup_offline_path}")

        plt.figure(figsize=(10, 6))
        for label, cb in online_series:
            valid_points = []
            for i, s in enumerate(cb.startup_times):
                if s is None:
                    continue
                try:
                    val = float(s)
                except Exception:
                    continue
                valid_points.append((i + 1, val))

            if valid_points:
                _, s_vals = zip(*valid_points)
                smoothed_vals = _smooth(list(s_vals))
                smoothed_steps = list(range(1, len(smoothed_vals) + 1))
                plt.plot( smoothed_steps,smoothed_vals,linewidth=1.5,label=label,)
        plt.xlabel("Timestep")
        plt.ylabel("Startup time (s)")
        plt.title("Startup time curves(Loacl Env)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        startup_online_path = os.path.join( self.model_results_dir, "startup_time_online.png" )
        plt.savefig(startup_online_path)
        plt.close()
        print(f"Saved online startup time curves to {startup_online_path}")


    def _evaluate_final_policy(self, dqn_model, eval_env, normalize_reward):
        print("Evaluating final hierarchical policy once on real environment…")

        obs, _ = eval_env.reset()
        hi_action, _ = dqn_model.predict(obs, deterministic=True)
        obs2, reward, done, truncated, info = eval_env.step(int(hi_action))

        config = info.get("config")
        startup = info.get("startup_seconds")

        print("Final suggested configuration from hierarchical policy:")
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
