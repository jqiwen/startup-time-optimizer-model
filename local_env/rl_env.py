"""
Simple reinforcement‑learning environment and Q‑learning agent for
container startup optimisation.

This module defines a small environment that loads startup time data
from a CSV file (created by the sweep script) and exposes a discrete
action space.  Each action corresponds to selecting a particular
combination of CPU cores, memory limit and heap size for the application.

The environment returns a reward equal to the negative start‑up time
(because shorter times are better).  There is no real notion of state
transition here – each episode consists of a single action and reward.

An example usage is provided at the end of this file.  You can run
this module directly to see the best configuration learned via
Q‑learning.  The Q‑learning algorithm iteratively updates values for
each action based on the observed reward and gradually shifts towards
selecting the action with the lowest mean start‑up time.

Note:  This is a simplified environment suitable for small data sets.
For larger or continuous parameter spaces you may wish to fit a
surrogate model (regression) and/or use more sophisticated RL
algorithms.  However, this example illustrates the basic pattern of
exploration and exploitation on a finite set of choices.
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class Action:
    """Represents a single parameter combination (CPU, memory, heap)."""

    cpu: float
    memory: float
    heap: float

    def to_tuple(self) -> Tuple[float, float, float]:
        return self.cpu, self.memory, self.heap


class StartupEnv:
    """
    A simple environment that maps each action to a reward based on
    measured start‑up times.  The state space is trivial (single dummy
    state), and the action space is discrete.  The reward is the
    negative of the start‑up time, so faster start‑up yields higher
    reward.
    """

    def __init__(self, csv_file: str) -> None:
        self.data: Dict[Tuple[float, float, float], float] = {}
        self.actions: List[Action] = []
        self.load_data(csv_file)
        self.num_actions = len(self.actions)

    def load_data(self, csv_file: str) -> None:
        """Load start‑up times from a CSV file produced by the sweep."""
        with open(csv_file, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cpu = float(row["cpus"])
                # Convert memory strings like "1G" or "512M" to gigabytes for simplicity
                mem_str = row["memory"].strip().upper()
                if mem_str.endswith("G"):
                    memory = float(mem_str.rstrip("G"))
                elif mem_str.endswith("M"):
                    memory = float(mem_str.rstrip("M")) / 1024.0
                else:
                    # Default: assume gigabytes
                    memory = float(mem_str)
                heap_str = row.get("heap", row.get("heap_size", "0")).strip().upper()
                if heap_str.endswith("G"):
                    heap = float(heap_str.rstrip("G"))
                elif heap_str.endswith("M"):
                    heap = float(heap_str.rstrip("M")) / 1024.0
                else:
                    heap = float(heap_str) if heap_str else 0.0
                startup = float(row["startup_seconds"])
                key = (cpu, memory, heap)
                # In case of duplicate entries for the same key, keep the smallest time
                self.data[key] = min(self.data.get(key, float("inf")), startup)
        # Build a list of actions
        for combo in sorted(self.data.keys()):
            cpu, memory, heap = combo
            self.actions.append(Action(cpu=cpu, memory=memory, heap=heap))

    def reset(self) -> int:
        """Reset the environment (no real state, so always returns 0)."""
        return 0

    def step(self, action_idx: int) -> Tuple[int, float, bool, Dict[str, float]]:
        """
        Execute the given action and return the resulting state, reward and done flag.
        Because each action corresponds to a one‑shot experiment, the state is
        always 0 and `done` is True.
        """
        action = self.actions[action_idx]
        reward = -self.data[action.to_tuple()]
        return 0, reward, True, {}

    def sample_action(self) -> int:
        """Return a random action index."""
        return random.randrange(self.num_actions)


def q_learning(env: StartupEnv, episodes: int = 1000, alpha: float = 0.1,
               gamma: float = 0.95, epsilon: float = 0.2,
               epsilon_decay: float = 0.995) -> Tuple[Action, float, List[List[float]]]:
    """
    Train a simple Q‑learning agent on the provided environment.

    Parameters
    ----------
    env: StartupEnv
        The environment containing start‑up time data.
    episodes: int
        Number of training episodes.
    alpha: float
        Learning rate for Q‑value updates.
    gamma: float
        Discount factor (not particularly meaningful here since each
        episode is one step).
    epsilon: float
        Initial exploration rate; the agent chooses a random action
        with probability `epsilon`.
    epsilon_decay: float
        Factor by which `epsilon` decays after each episode.

    Returns
    -------
    best_action: Action
        The action (CPU, memory, heap) with the highest estimated reward
        (i.e., minimal start‑up time).
    best_time: float
        The corresponding start‑up time in seconds.
    q_values: List[List[float]]
        The final Q‑table values (single state × actions).
    """
    # Q‑table has shape (1, num_actions) because there is only one state
    q_table: List[List[float]] = [[0.0 for _ in range(env.num_actions)]]
    current_epsilon = epsilon
    for episode in range(episodes):
        state = env.reset()
        # Choose action via epsilon‑greedy strategy
        if random.random() < current_epsilon:
            action = env.sample_action()
        else:
            # Exploit: choose action with highest Q value
            max_q = max(q_table[state])
            # If multiple actions have the same Q value, pick randomly among them
            candidates = [i for i, q in enumerate(q_table[state]) if q == max_q]
            action = random.choice(candidates)
        # Take the step and get reward
        next_state, reward, done, _ = env.step(action)
        # Update Q value for the taken action
        old_q = q_table[state][action]
        next_max = max(q_table[next_state]) if not done else 0.0
        q_table[state][action] = old_q + alpha * (reward + gamma * next_max - old_q)
        # Decay exploration rate
        current_epsilon *= epsilon_decay
        current_epsilon = max(0.01, current_epsilon)  # minimum epsilon
    # After training, choose the best action
    state = 0
    best_action_idx = max(range(env.num_actions), key=lambda i: q_table[state][i])
    best_action = env.actions[best_action_idx]
    best_time = env.data[best_action.to_tuple()]
    return best_action, best_time, q_table


def main() -> None:
    """Demonstrate training the agent using the data from startup_data.csv."""
    import pathlib
    data_path = pathlib.Path(__file__).parent / "startup_data.csv"
    if not data_path.exists():
        print(f"Data file {data_path} not found. Run sweep.py to generate it.")
        return
    env = StartupEnv(str(data_path))
    best_action, best_time, q_table = q_learning(env, episodes=2000)
    print("Best configuration found:")
    print(f"  CPU cores: {best_action.cpu}")
    print(f"  Memory (GB): {best_action.memory}")
    print(f"  Heap (GB): {best_action.heap}")
    print(f"  Startup time (s): {best_time:.3f}")


if __name__ == "__main__":
    main()
