# execute.py
import os
from typing import List

import matplotlib.pyplot as plt

from stable_baselines3 import PPO, DQN

import train  # 复用 TrainingMetricsCallback, moving_average 等
from monitor import Monitor
from analysis import Analysis
from plan import Plan

WINDOW_SIZE = 50

class Execute:
    """
    Execute: 负责实际“执行”整个自适应训练流程。

    - 调用 Monitor 收集数据
    - 调用 Analysis 生成 action 集和 baseline
    - 调用 Plan 创建环境和模型
    - 运行四个阶段：
        A1: PPO offline
        A2: DQN offline
        B1: PPO online (Real env)
        B2: DQN online (Hierarchical + Real env)
    - 保存模型和生成图表，并做一次最终评估
    """

    def __init__(
        self,
        monitor: Monitor,
        analysis: Analysis,
        planner: Plan,
        model_results_dir: str,
        offline_steps: int,
        online_steps: int,
    ) -> None:
        self.monitor = monitor
        self.analysis = analysis
        self.planner = planner
        self.model_results_dir = model_results_dir
        self.offline_steps = offline_steps
        self.online_steps = online_steps

    # ------------------------------------------------------------------ #
    # 核心入口
    # ------------------------------------------------------------------ #

    def run(self, csv_arg: str) -> None:
        # 1) Monitor: load CSV
        startup_df = self.monitor.load_csv(csv_arg)

        # 2) Analysis: 构建 actions + baseline
        actions, filtered_df, baseline_startup = self.analysis.build_actions_and_baseline(
            startup_df
        )

        # 3) Plan: 构建 offline envs & models
        offline_env, hier_offline_env = self.planner.build_offline_envs(
            actions,
            filtered_df,
            baseline_startup,
            self.analysis.normalize_reward,
        )

        # 重新用真正的 PPO 替换掉 Plan 中 dummy_ppo
        ppo_model, dqn_model = self.planner.build_models(
            offline_env, hier_offline_env
        )
        # 需要让层次环境使用真正的 PPO 模型
        hier_offline_env.ppo_model = ppo_model

        # 4) Execute: Phase A1 & A2 (offline)
        (
            ppo_offline_cb,
            dqn_offline_cb,
        ) = self._run_offline_training(
            ppo_model,
            dqn_model,
            offline_env,
            hier_offline_env,
        )

        # 5) Plan: 构建 online envs
        real_env, hier_real_env = self.planner.build_online_envs(
            actions,
            baseline_startup,
            self.analysis.normalize_reward,
            ppo_model,
        )

        # 6) Execute: Phase B1 & B2 (online fine-tuning)
        (
            ppo_online_cb,
            dqn_online_cb,
        ) = self._run_online_training(
            ppo_model,
            dqn_model,
            real_env,
            hier_real_env,
        )

        # 7) Execute: 保存模型 + 作图
        self._save_models(ppo_model, dqn_model)
        self._plot_curves(
            ppo_offline_cb,
            dqn_offline_cb,
            ppo_online_cb,
            dqn_online_cb,
            self.analysis.normalize_reward,
        )

        # 8) Execute: 最终评估一次
        self._evaluate_final_policy(dqn_model, hier_real_env, self.analysis.normalize_reward)

    # ------------------------------------------------------------------ #
    # Offline 训练
    # ------------------------------------------------------------------ #

    def _run_offline_training(
        self,
        ppo_model: PPO,
        dqn_model: DQN,
        offline_env,
        hier_offline_env,
    ):
        print(
            f"[PHASE A1] Training PPO (low-level) offline for {self.offline_steps} timesteps…"
        )

        ppo_offline_cb = train.TrainingMetricsCallback(phase="PPO_offline")
        ppo_model.set_env(offline_env)
        ppo_model.learn(
            total_timesteps=self.offline_steps,
            progress_bar=False,
            callback=ppo_offline_cb,
        )

        print(
            f"[PHASE A2] Training DQN (high-level) offline for {self.offline_steps} timesteps…"
        )
        dqn_offline_cb = train.TrainingMetricsCallback(phase="DQN_offline")
        dqn_model.set_env(hier_offline_env)
        dqn_model.learn(
            total_timesteps=self.offline_steps,
            progress_bar=False,
            callback=dqn_offline_cb,
        )

        return ppo_offline_cb, dqn_offline_cb

    # ------------------------------------------------------------------ #
    # Online 微调
    # ------------------------------------------------------------------ #

    def _run_online_training(
        self,
        ppo_model: PPO,
        dqn_model: DQN,
        real_env,
        hier_real_env,
    ):
        print(
            f"[PHASE B1] Switching PPO low-level to real Docker environment for {self.online_steps} timesteps…"
        )

        # Gentler PPO online
        ppo_model.set_env(real_env)
        ppo_model.learning_rate = 5e-5
        ppo_model.n_steps = 32
        ppo_model.batch_size = 16
        ppo_model.gamma = 0.999
        ppo_model.clip_range = 0.1

        ppo_online_cb = train.TrainingMetricsCallback(
            max_steps=self.online_steps,
            phase="PPO_online",
        )

        ppo_model.learn(
            total_timesteps=self.online_steps,
            reset_num_timesteps=False,
            callback=ppo_online_cb,
            progress_bar=False,
        )

        print(
            f"[PHASE B2] Switching DQN high-level to hierarchical real env for {self.online_steps} timesteps…"
        )

        # Gentler DQN online
        dqn_model.set_env(hier_real_env)
        dqn_model.learning_rate = 1e-4
        dqn_model.buffer_size = 5000
        dqn_model.batch_size = 32
        dqn_model.exploration_fraction = 0.05
        dqn_model.exploration_final_eps = 0.02
        dqn_model.target_update_interval = 500
        dqn_model.gamma = 0.999

        dqn_online_cb = train.TrainingMetricsCallback(
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

    # ------------------------------------------------------------------ #
    # 保存模型
    # ------------------------------------------------------------------ #

    def _save_models(self, ppo_model: PPO, dqn_model: DQN) -> None:
        os.makedirs(self.model_results_dir, exist_ok=True)
        ppo_model_file = os.path.join(self.model_results_dir, "ppo_model.zip")
        dqn_model_file = os.path.join(self.model_results_dir, "dqn_model.zip")

        ppo_model.save(ppo_model_file)
        print(f"[SAVE] Saved final low-level PPO model to {ppo_model_file}")

        dqn_model.save(dqn_model_file)
        print(f"[SAVE] Saved final high-level DQN model to {dqn_model_file}")

    # ------------------------------------------------------------------ #
    # 画图
    # ------------------------------------------------------------------ #

        # ------------------------------------------------------------------ #
    # 画图（离线 / 在线分开）
    # ------------------------------------------------------------------ #

        # ------------------------------------------------------------------ #
    # 画图（离线 / 在线分开）
    # ------------------------------------------------------------------ #
    def _plot_curves(
        self,
        ppo_offline_cb,
        dqn_offline_cb,
        ppo_online_cb,
        dqn_online_cb,
        normalize_reward: bool,
    ) -> None:
        os.makedirs(self.model_results_dir, exist_ok=True)

        offline_series = [
            ("PPO", ppo_offline_cb),
            ("Hierarchical Model(PPO+DQN)", dqn_offline_cb),
        ]
        online_series = [
            ("PPO", ppo_online_cb),
            ("Hierarchical Model(PPO+DQN)", dqn_online_cb),
        ]

        # Helper to choose window ~= timesteps / 100 and smooth
        def _smooth(values: List[float]) -> List[float]:
            if not values:
                return []
            # e.g. 5000 steps -> window 50; also make sure window >= 5
            window = max(5, len(values) // 100)
            return train.moving_average(values, window=window)

        # ---------- Reward curves: Offline ----------
        plt.figure(figsize=(10, 6))
        for label, cb in offline_series:
            if cb.rewards:
                smoothed = _smooth(cb.rewards)
                steps = list(range(1, len(smoothed) + 1))
                plt.plot(
                    steps,
                    smoothed,
                    linewidth=1.5,
                    label=label,
                )
        plt.xlabel("Timestep")
        plt.ylabel("Reward (normalized)" if normalize_reward else "Reward")
        plt.title("Reward curves(DB)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        reward_offline_path = os.path.join(self.model_results_dir, "reward_offline.png")
        plt.savefig(reward_offline_path)
        plt.close()
        print(f"[PLOT] Saved offline reward curves to {reward_offline_path}")

        # ---------- Reward curves: Online ----------
        plt.figure(figsize=(10, 6))
        for label, cb in online_series:
            if cb.rewards:
                smoothed = _smooth(cb.rewards)
                steps = list(range(1, len(smoothed) + 1))
                plt.plot(
                    steps,
                    smoothed,
                    linewidth=1.5,
                    label=label,
                )
        plt.xlabel("Timestep")
        plt.ylabel("Reward (normalized)" if normalize_reward else "Reward")
        plt.title("Reward curves(Loacl Env)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        reward_online_path = os.path.join(self.model_results_dir, "reward_online.png")
        plt.savefig(reward_online_path)
        plt.close()
        print(f"[PLOT] Saved online reward curves to {reward_online_path}")

        # ---------- Loss curves (-reward): Offline ----------
        plt.figure(figsize=(10, 6))
        for label, cb in offline_series:
            if cb.loss_proxy:
                smoothed = _smooth(cb.loss_proxy)
                steps = list(range(1, len(smoothed) + 1))
                plt.plot(
                    steps,
                    smoothed,
                    linewidth=1.5,
                    label=label,
                )
        plt.xlabel("Timestep")
        plt.ylabel("Loss proxy (-reward)")
        plt.title("Loss curves(DB)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        loss_offline_path = os.path.join(self.model_results_dir, "loss_offline.png")
        plt.savefig(loss_offline_path)
        plt.close()
        print(f"[PLOT] Saved offline loss curves to {loss_offline_path}")

        # ---------- Loss curves (-reward): Online ----------
        plt.figure(figsize=(10, 6))
        for label, cb in online_series:
            if cb.loss_proxy:
                smoothed = _smooth(cb.loss_proxy)
                steps = list(range(1, len(smoothed) + 1))
                plt.plot(
                    steps,
                    smoothed,
                    linewidth=1.5,
                    label=label,
                )
        plt.xlabel("Timestep")
        plt.ylabel("Loss proxy (-reward)")
        plt.title("Loss curves(Loacl Env)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        loss_online_path = os.path.join(self.model_results_dir, "loss_online.png")
        plt.savefig(loss_online_path)
        plt.close()
        print(f"[PLOT] Saved online loss curves to {loss_online_path}")

        # ---------- Startup time curves: Offline ----------
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
        startup_offline_path = os.path.join(
            self.model_results_dir, "startup_time_offline.png"
        )
        plt.savefig(startup_offline_path)
        plt.close()
        print(f"[PLOT] Saved offline startup time curves to {startup_offline_path}")

        # ---------- Startup time curves: Online ----------
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
                plt.plot(
                    smoothed_steps,
                    smoothed_vals,
                    linewidth=1.5,
                    label=label,
                )
        plt.xlabel("Timestep")
        plt.ylabel("Startup time (s)")
        plt.title("Startup time curves(Loacl Env)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        startup_online_path = os.path.join(
            self.model_results_dir, "startup_time_online.png"
        )
        plt.savefig(startup_online_path)
        plt.close()
        print(f"[PLOT] Saved online startup time curves to {startup_online_path}")



    # ------------------------------------------------------------------ #
    # 最终评估
    # ------------------------------------------------------------------ #

    def _evaluate_final_policy(self, dqn_model: DQN, eval_env, normalize_reward: bool) -> None:
        print("[EVAL] Evaluating final hierarchical policy once on real environment…")

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
