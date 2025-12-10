import argparse
import os
from monitor import Monitor
from analysis import Analysis
from plan import Plan
from execute import Execute


def main():
    parser = argparse.ArgumentParser(description="Hierarchical RL training with MAPE-K Architecture.")

    parser.add_argument(
        "--csv",
        type=str,
        default="startup_data.csv"
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="median"
    )
    parser.add_argument(
        "--offline-steps",
        type=int,
        default=5000
    )
    parser.add_argument(
        "--online-steps",
        type=int,
        default=20
    )
    parser.add_argument(
        "--cpu-max",
        type=float,
        default=None
    )
    parser.add_argument(
        "--mem-max",
        type=str,
        default=None
    )
    parser.add_argument(
        "--no-normalize-reward",
        action="store_true"
    )

    args = parser.parse_args()
    normalize_reward = not args.no_normalize_reward
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
