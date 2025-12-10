import pandas as pd

class Action:
    cpus: str 
    memory: str 
    heap: str

class Analysis:
    def __init__(self, cpu_max, mem_max, baseline_mode, normalize_reward):
        self.cpu_max = cpu_max
        self.mem_max = mem_max
        self.baseline_mode = baseline_mode
        self.normalize_reward = normalize_reward

    def format_mem(mem):
        mem = str(mem).strip().upper()
        if mem.endswith("G"):
            return int(float(mem[:-1]) * 1024)
        if mem.endswith("M"):
            return int(float(mem[:-1]))
        return int(float(mem))

    def build_actions(self, startup_df, cpu_max, mem_max):
        heap_col = "heap"
        if heap_col not in startup_df.columns and "heap_size" in startup_df.columns:
            heap_col = "heap_size"

        filtered = startup_df.copy()

        if cpu_max is not None:
            filtered = filtered[filtered["cpus"].astype(float) <= cpu_max]

        if mem_max is not None:
            max_mb = self.format_mem(mem_max)
            mem_mb_series = filtered["memory"].astype(str).map(self.format_mem)
            filtered = filtered[mem_mb_series <= max_mb]

        if heap_col in filtered.columns:
            heap_mb = filtered[heap_col].astype(str).map(self.format_mem)
            mem_mb = filtered["memory"].astype(str).map(self.format_mem)
            filtered = filtered[heap_mb <= mem_mb]

        combos = (
            filtered[["cpus", "memory", heap_col]]
            .astype(str)
            .drop_duplicates()
            .reset_index(drop=True)
        )

        actions = [
            Action(cpus=row["cpus"], memory=row["memory"], heap=row[heap_col])
            for _, row in combos.iterrows()
        ]
        return actions, filtered
    
    def compute_baseline(startup_df, mode):
        if mode in ("mean"):
            return float(startup_df["startup_seconds"].mean())
        if mode == "median":
            return float(startup_df["startup_seconds"].median())
        if mode == "min":
            return float(startup_df["startup_seconds"].min())
        return float(mode)


    def build_actions_and_baseline(self, startup_df):
        actions, filtered_df = self.build_actions(startup_df, self.cpu_max, self.mem_max)

        if self.cpu_max is not None:
            print(f"Applied CPU limit: {self.cpu_max} cores")
        if self.mem_max is not None:
            print(f"Applied memory limit: {self.mem_max}")
        print(f"Valid configurations after filtering: {len(actions)}")

        baseline_startup = self.compute_baseline(filtered_df, self.baseline_mode)
        print(f"Using baseline startup time: {baseline_startup:.3f} seconds")
        return actions, filtered_df, baseline_startup
