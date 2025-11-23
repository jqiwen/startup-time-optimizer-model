# analysis.py
from typing import Optional, Tuple, List
import pandas as pd

import train  # 复用你原来的实现：build_actions_from_csv, compute_baseline, Action


class Analysis:
    """
    Analysis: 对监控到的数据进行“分析”。

    - 主要功能：
      1) 根据 CPU / 内存限制过滤配置并构建 Action 列表
      2) 计算 baseline startup time
    """

    def __init__(
        self,
        cpu_max: Optional[float],
        mem_max: Optional[str],
        baseline_mode: str,
        normalize_reward: bool,
    ) -> None:
        self.cpu_max = cpu_max
        self.mem_max = mem_max
        self.baseline_mode = baseline_mode
        self.normalize_reward = normalize_reward

    def build_actions_and_baseline(
        self, startup_df: pd.DataFrame
    ) -> Tuple[list, pd.DataFrame, float]:
        """
        使用 train.py 中已有的工具函数：
          - build_actions_from_csv
          - compute_baseline
        """
        actions, filtered_df = train.build_actions_from_csv(
            startup_df, self.cpu_max, self.mem_max
        )

        if filtered_df.empty or not actions:
            raise RuntimeError(
                "[Analysis] After applying limits, no valid configurations remain."
            )

        if self.cpu_max is not None:
            print(f"[ANALYSIS] Applied CPU limit: {self.cpu_max} cores")
        if self.mem_max is not None:
            print(f"[ANALYSIS] Applied memory limit: {self.mem_max}")
        print(
            f"[ANALYSIS] Valid configurations after filtering: {len(actions)}"
        )

        baseline_startup = train.compute_baseline(filtered_df, self.baseline_mode)
        print(
            f"[ANALYSIS] Using baseline startup time: {baseline_startup:.3f} seconds"
        )
        if self.normalize_reward:
            print("[ANALYSIS] Reward normalization is ENABLED (reward / baseline).")
        else:
            print(
                "[ANALYSIS] Reward normalization is DISABLED (raw baseline - startup)."
            )

        return actions, filtered_df, baseline_startup
