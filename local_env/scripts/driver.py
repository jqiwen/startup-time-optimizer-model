# driver.py
import argparse
import os

from monitor import Monitor
from analysis import Analysis
from plan import Plan
from execute import Execute


def main():
    parser = argparse.ArgumentParser(
        description="Hierarchical RL training for AcmeAir with MAPE-style decomposition."
    )

    parser.add_argument(
        "--csv",
        type=str,
        default="startup_data.csv",
        help="CSV file with sweep data (must contain cpus, memory, startup_seconds).",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="median",
        help="Baseline mode: 'mean', 'median', 'min', or a numeric value.",
    )
    parser.add_argument(
        "--offline-steps",
        type=int,
        default=5000,
        help="Timesteps for offline training (CSV-only) for each level.",
    )
    parser.add_argument(
        "--online-steps",
        type=int,
        default=20,
        help="Timesteps for online fine-tuning (real Docker environment) for each level.",
    )
    parser.add_argument(
        "--cpu-max",
        type=float,
        default=None,
        help="Max CPU cores allowed (e.g. 2.0). Only configs with cpus <= cpu_max are used.",
    )
    parser.add_argument(
        "--mem-max",
        type=str,
        default=None,
        help="Max memory allowed (e.g. '1G', '768M'). Only configs with memory <= mem_max are used.",
    )
    parser.add_argument(
        "--no-normalize-reward",
        action="store_true",
        help=(
            "Disable reward normalization. By default, rewards are divided by "
            "the baseline startup time to keep them in a stable range."
        ),
    )

    args = parser.parse_args()
    normalize_reward = not args.no_normalize_reward

    # model_results 目录：../model_results
    project_dir = os.path.dirname(__file__)
    model_results_dir = os.path.abspath(os.path.join(project_dir, "..", "model_results"))

    monitor = Monitor()
    analysis = Analysis(
        cpu_max=args.cpu_max,
        mem_max=args.mem_max,
        baseline_mode=args.baseline,
        normalize_reward=normalize_reward,
    )
    planner = Plan()

    executor = Execute(
        monitor=monitor,
        analysis=analysis,
        planner=planner,
        model_results_dir=model_results_dir,
        offline_steps=args.offline_steps,
        online_steps=args.online_steps,
    )

    executor.run(csv_arg=args.csv)


if __name__ == "__main__":
    main()
