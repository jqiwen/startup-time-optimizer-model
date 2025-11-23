# monitor.py
import os
import sys
import pandas as pd
from typing import Optional


class Monitor:
    """
    Monitor: 负责“监控”并收集原始数据（例如 CSV 启动时间数据）。

    - 主要功能：加载 CSV，做基本的存在性检查。
    """

    def __init__(self, default_relative_csv: str = "../data/startup_data.csv") -> None:
        project_dir = os.path.dirname(__file__)
        self.default_csv_path = os.path.abspath(
            os.path.join(project_dir, default_relative_csv)
        )

    def resolve_csv_path(self, csv_arg: str) -> str:
        """
        根据命令行参数解析 CSV 路径：
        - 如果传入的是默认名 'startup_data.csv'，则使用 ../data/startup_data.csv
        - 否则按用户给的路径解析
        """
        if csv_arg == "startup_data.csv":
            csv_path = self.default_csv_path
        else:
            csv_path = os.path.abspath(csv_arg)
        return csv_path

    def load_csv(self, csv_arg: str) -> pd.DataFrame:
        """
        加载 CSV 并检查是否存在。
        """
        csv_path = self.resolve_csv_path(csv_arg)
        print(f"[MONITOR] Loading CSV from: {csv_path}")

        if not os.path.exists(csv_path):
            print(f"[ERROR] CSV file '{csv_path}' not found.", file=sys.stderr)
            sys.exit(1)

        startup_df = pd.read_csv(csv_path)

        required_cols = {"cpus", "memory", "startup_seconds"}
        if not required_cols.issubset(startup_df.columns):
            print(f"[ERROR] CSV must contain columns {required_cols}", file=sys.stderr)
            sys.exit(1)

        return startup_df
