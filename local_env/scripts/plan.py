# plan.py
from typing import Tuple, List

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.utils import get_schedule_fn
import train  # 复用 OfflineStartupEnv, RealStartupEnv, HierarchicalStartupEnv, Action


class Plan:
    """
    Plan: 负责“规划”训练结构 —— 创建环境和模型，但不真正执行训练。

    - 创建 Offline 环境和 Online 环境
    - 创建 PPO (low-level) 与 DQN (high-level) 模型
    """

    def build_offline_envs(
        self,
        actions: List[train.Action],
        filtered_df,
        baseline_startup: float,
        normalize_reward: bool,
    ) -> Tuple[train.OfflineStartupEnv, train.HierarchicalStartupEnv]:
        """
        创建 OfflineStartupEnv 和基于它的 HierarchicalStartupEnv。
        """
        offline_env = train.OfflineStartupEnv(
            actions=actions,
            startup_df=filtered_df,
            baseline_startup=baseline_startup,
            normalize_reward=normalize_reward,
        )

        # 先用一个 dummy PPO 占位，稍后真正训练前会用它
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

        hier_offline_env = train.HierarchicalStartupEnv(
            actions=actions,
            base_env=offline_env,
            ppo_model=dummy_ppo,
        )

        return offline_env, hier_offline_env

    def build_models(
        self,
        offline_env: train.OfflineStartupEnv,
        hier_offline_env: train.HierarchicalStartupEnv,
    ) -> Tuple[PPO, DQN]:
        """
        根据 offline 环境创建 PPO 和 DQN 模型（参数和原 train.py 中一致）。
        """
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

    def build_online_envs(
        self,
        actions: List[train.Action],
        baseline_startup: float,
        normalize_reward: bool,
        ppo_model: PPO,
    ) -> tuple:
        """
        创建 RealStartupEnv 和基于它的 HierarchicalStartupEnv，用于 online 微调。
        """
        real_env = train.RealStartupEnv(
            actions=actions,
            baseline_startup=baseline_startup,
            normalize_reward=normalize_reward,
        )

        hier_real_env = train.HierarchicalStartupEnv(
            actions=actions,
            base_env=real_env,
            ppo_model=ppo_model,
        )

        return real_env, hier_real_env
