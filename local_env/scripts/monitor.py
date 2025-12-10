import os
import sys
import pandas as pd


class Monitor:
    def __init__(self, default_relative_csv: str = "../data/startup_data.csv"):
        project_dir = os.path.dirname(__file__)
        self.default_csv_path = os.path.abspath(os.path.join(project_dir, default_relative_csv))

    def resolve_csv_path(self, csv_arg: str):
        if csv_arg == "startup_data.csv":
            csv_path = self.default_csv_path
        else:
            csv_path = os.path.abspath(csv_arg)
        return csv_path

    def load_csv(self, csv_arg: str):
        csv_path = self.resolve_csv_path(csv_arg)
        print(f"Loading CSV from: {csv_path}")

        if not os.path.exists(csv_path):
            print(f"CSV file '{csv_path}' not found.", file=sys.stderr)
            sys.exit(1)

        startup_df = pd.read_csv(csv_path)

        required_cols = {"cpus", "memory", "startup_seconds"}
        if not required_cols.issubset(startup_df.columns):
            print(f"CSV must contain columns {required_cols}", file=sys.stderr)
            sys.exit(1)

        return startup_df
