"""
Strategy Comparison Framework

This module provides functionality to compare the RL agent's performance against
baseline strategies on the same races. It runs counterfactual simulations and
computes statistical comparisons.
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

# Optional: scipy for statistical tests
try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not installed. Statistical significance tests will be skipped.")

from f1_env_final import create_f1_env, TireCompound
from f1_agent import F1PPOAgent
from baseline_strategies import create_baseline_strategy, BaselineStrategy


class StrategyComparator:
    """
    Compare RL agent against baseline strategies.

    Runs the same race scenarios with different strategies and collects metrics
    for statistical comparison.
    """

    def __init__(self, env, agent: Optional[F1PPOAgent] = None, data_dir: Path = Path("data")):
        self.env = env
        self.agent = agent
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / "processed"

        # Load data for historical strategies
        with open(self.processed_dir / "22-24_pitstops.json", "r") as f:
            self.pitstops_data = pd.DataFrame(json.load(f))

        self.results = []

    def run_rl_agent(self, race_id: int, driver_id: int, starting_position: int = None, render: bool = False) -> Dict:
        """Run the RL agent on a specific race"""
        if self.agent is None:
            raise ValueError("No RL agent provided")

        reset_options = {"race_id": race_id, "driver_id": driver_id}
        if starting_position is not None:
            reset_options["starting_position"] = starting_position

        state, info = self.env.reset(options=reset_options)

        total_reward = 0
        lap_times = []
        positions = []
        tire_compounds = []
        tire_degradations = []
        fuel_loads = []
        pit_laps = []
        actions = []
        done = False
        lap = 0

        with torch.no_grad():
            while not done:
                # Select action (deterministic)
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.agent.device)
                action_probs, _ = self.agent.policy(state_tensor)
                action = torch.argmax(action_probs, dim=-1).item()

                actions.append(action)

                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                # Record metrics
                total_reward += reward
                lap_times.append(info.get("total_race_time", 0) - sum(lap_times))
                positions.append(info["position"])
                tire_compounds.append(info["tire_compound"])
                tire_degradations.append(info["tire_degradation"])
                fuel_loads.append(info["fuel_load"])

                if action >= 3:  # Pit stop
                    pit_laps.append(lap + 1)

                state = next_state
                lap += 1

                if render:
                    self.env.render()

        return {
            "strategy": "RL Agent",
            "race_id": race_id,
            "driver_id": driver_id,
            "starting_position": self.env.starting_position,
            "final_position": info["position"],
            "total_reward": total_reward,
            "total_race_time": info["total_race_time"],
            "num_pitstops": info["num_pitstops"],
            "pit_laps": pit_laps,
            "lap_times": lap_times,
            "positions": positions,
            "tire_compounds": tire_compounds,
            "tire_degradations": tire_degradations,
            "fuel_loads": fuel_loads,
            "actions": actions
        }

    def run_baseline(
        self,
        strategy: BaselineStrategy,
        race_id: int,
        driver_id: int,
        starting_position: int = None,
        render: bool = False
    ) -> Dict:
        """Run a baseline strategy on a specific race"""
        reset_options = {"race_id": race_id, "driver_id": driver_id}
        if starting_position is not None:
            reset_options["starting_position"] = starting_position

        state, info = self.env.reset(options=reset_options)

        obs_array = state
        total_reward = 0
        lap_times = []
        positions = []
        tire_compounds = []
        tire_degradations = []
        fuel_loads = []
        pit_laps = []
        actions = []
        done = False
        lap = 0
        total_laps = self.env.total_laps

        while not done:
            # Convert state to dict for baseline strategy
            state_dict = {
                "tire_age": self.env.tire_age,
                "tire_degradation": self.env.tire_degradation,
                "tire_compound": self.env.tire_compound,
                "position": self.env.position,
                "fuel_load": self.env.fuel_load,
                "lap": self.env.current_lap
            }

            # Get action from baseline strategy
            action = strategy.decide_action(state_dict, lap + 1, total_laps)
            actions.append(action)

            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # Record metrics
            total_reward += reward
            lap_times.append(info.get("total_race_time", 0) - sum(lap_times))
            positions.append(info["position"])
            tire_compounds.append(info["tire_compound"])
            tire_degradations.append(info["tire_degradation"])
            fuel_loads.append(info["fuel_load"])

            if action >= 3:  # Pit stop
                pit_laps.append(lap + 1)

            obs_array = next_state
            lap += 1

            if render:
                self.env.render()

        return {
            "strategy": strategy.name,
            "race_id": race_id,
            "driver_id": driver_id,
            "starting_position": self.env.starting_position,
            "final_position": info["position"],
            "total_reward": total_reward,
            "total_race_time": info["total_race_time"],
            "num_pitstops": info["num_pitstops"],
            "pit_laps": pit_laps,
            "lap_times": lap_times,
            "positions": positions,
            "tire_compounds": tire_compounds,
            "tire_degradations": tire_degradations,
            "fuel_loads": fuel_loads,
            "actions": actions
        }

    def compare_on_races(
        self,
        race_ids: List[int],
        driver_id: int,
        baseline_types: List[str] = ["2-stop", "3-stop", "adaptive"],
        num_runs_per_race: int = 1
    ) -> pd.DataFrame:
        """
        Compare RL agent vs baselines across multiple races.

        Args:
            race_ids: List of race IDs to test on
            driver_id: Driver ID to use
            baseline_types: List of baseline strategy types
            num_runs_per_race: Number of runs per race (for stochastic environments)

        Returns:
            DataFrame with comparison results
        """
        all_results = []

        total_comparisons = len(race_ids) * num_runs_per_race * (1 + len(baseline_types))
        pbar = tqdm(total=total_comparisons, desc="Running comparisons", unit="run")

        for race_id in race_ids:
            for run in range(num_runs_per_race):
                starting_position = None

                # Run RL agent
                if self.agent is not None:
                    rl_result = self.run_rl_agent(race_id, driver_id)
                    rl_result["run"] = run
                    starting_position = rl_result["starting_position"]  # Capture starting position
                    all_results.append(rl_result)
                    pbar.update(1)

                # Run each baseline with the same starting position
                for baseline_type in baseline_types:
                    # Create baseline strategy
                    if baseline_type == "historical":
                        strategy = create_baseline_strategy(
                            baseline_type,
                            pitstops_data=self.pitstops_data,
                            race_id=race_id,
                            driver_id=driver_id
                        )
                    else:
                        strategy = create_baseline_strategy(baseline_type)

                    baseline_result = self.run_baseline(strategy, race_id, driver_id, starting_position=starting_position)
                    baseline_result["run"] = run
                    all_results.append(baseline_result)
                    pbar.update(1)

        pbar.close()

        self.results = all_results
        return pd.DataFrame(all_results)

    def compute_statistics(self, results_df: Optional[pd.DataFrame] = None) -> Dict:
        """
        Compute statistical comparisons between strategies.

        Returns:
            Dictionary with statistical metrics
        """
        if results_df is None:
            results_df = pd.DataFrame(self.results)

        if len(results_df) == 0:
            return {}

        stats_summary = {}

        # Group by strategy
        for strategy_name in results_df["strategy"].unique():
            strategy_results = results_df[results_df["strategy"] == strategy_name]

            stats_summary[strategy_name] = {
                "avg_position": strategy_results["final_position"].mean(),
                "std_position": strategy_results["final_position"].std(),
                "avg_race_time": strategy_results["total_race_time"].mean(),
                "std_race_time": strategy_results["total_race_time"].std(),
                "avg_reward": strategy_results["total_reward"].mean(),
                "std_reward": strategy_results["total_reward"].std(),
                "avg_pitstops": strategy_results["num_pitstops"].mean(),
                "std_pitstops": strategy_results["num_pitstops"].std(),
                "wins": (strategy_results["final_position"] == 1).sum(),
                "podiums": (strategy_results["final_position"] <= 3).sum(),
                "points_finishes": (strategy_results["final_position"] <= 10).sum(),
                "num_races": len(strategy_results)
            }

        # Statistical tests: RL Agent vs each baseline
        if "RL Agent" in results_df["strategy"].values and HAS_SCIPY:
            rl_positions = results_df[results_df["strategy"] == "RL Agent"]["final_position"]
            rl_times = results_df[results_df["strategy"] == "RL Agent"]["total_race_time"]

            stats_summary["statistical_tests"] = {}

            for strategy_name in results_df["strategy"].unique():
                if strategy_name == "RL Agent":
                    continue

                baseline_positions = results_df[results_df["strategy"] == strategy_name]["final_position"]
                baseline_times = results_df[results_df["strategy"] == strategy_name]["total_race_time"]

                # Wilcoxon signed-rank test (paired test)
                if len(rl_positions) == len(baseline_positions) and len(rl_positions) > 0:
                    try:
                        pos_stat, pos_pval = stats.wilcoxon(rl_positions, baseline_positions)
                        time_stat, time_pval = stats.wilcoxon(rl_times, baseline_times)

                        stats_summary["statistical_tests"][strategy_name] = {
                            "position_wilcoxon_pval": pos_pval,
                            "position_significant": pos_pval < 0.05,
                            "time_wilcoxon_pval": time_pval,
                            "time_significant": time_pval < 0.05,
                            "position_improvement": float(baseline_positions.mean() - rl_positions.mean()),
                            "time_improvement": float(baseline_times.mean() - rl_times.mean())
                        }
                    except Exception as e:
                        stats_summary["statistical_tests"][strategy_name] = {
                            "error": str(e)
                        }
        elif "RL Agent" in results_df["strategy"].values and not HAS_SCIPY:
            stats_summary["statistical_tests"] = {"note": "scipy not installed - statistical tests skipped"}

        return stats_summary

    def print_comparison_report(self, stats_summary: Optional[Dict] = None):
        """Print a formatted comparison report"""
        if stats_summary is None:
            results_df = pd.DataFrame(self.results)
            stats_summary = self.compute_statistics(results_df)

        print("\n" + "=" * 80)
        print("STRATEGY COMPARISON REPORT")
        print("=" * 80)

        # Summary table
        print("\nPERFORMANCE SUMMARY:")
        print("-" * 80)
        print(f"{'Strategy':<20} {'Avg Pos':<10} {'Avg Time':<12} {'Avg Stops':<10} {'Podiums':<10}")
        print("-" * 80)

        for strategy_name, metrics in stats_summary.items():
            if strategy_name == "statistical_tests":
                continue

            print(f"{strategy_name:<20} "
                  f"{metrics['avg_position']:>6.2f} ± {metrics['std_position']:.2f}  "
                  f"{metrics['avg_race_time']:>8.1f}s ± {metrics['std_race_time']:.1f}  "
                  f"{metrics['avg_pitstops']:>5.2f} ± {metrics['std_pitstops']:.2f}  "
                  f"{metrics['podiums']:>8}/{metrics['num_races']}")

        # Statistical tests
        if "statistical_tests" in stats_summary:
            print("\n" + "=" * 80)
            print("STATISTICAL SIGNIFICANCE TESTS (RL Agent vs Baselines)")
            print("=" * 80)

            for baseline_name, test_results in stats_summary["statistical_tests"].items():
                if "error" in test_results:
                    print(f"\n{baseline_name}: Error - {test_results['error']}")
                    continue

                print(f"\nRL Agent vs {baseline_name}:")
                print(f"  Position improvement: {test_results['position_improvement']:.2f} positions")
                print(f"  Position test p-value: {test_results['position_wilcoxon_pval']:.4f} "
                      f"{'✓ Significant' if test_results['position_significant'] else '✗ Not significant'}")
                print(f"  Time improvement: {test_results['time_improvement']:.2f} seconds")
                print(f"  Time test p-value: {test_results['time_wilcoxon_pval']:.4f} "
                      f"{'✓ Significant' if test_results['time_significant'] else '✗ Not significant'}")

        print("\n" + "=" * 80)

    def save_results(self, output_dir: Path = Path("results")):
        """Save comparison results to files"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save detailed results
        results_df = pd.DataFrame(self.results)
        results_df.to_csv(output_dir / "strategy_comparison_results.csv", index=False)

        # Save statistics
        stats_summary = self.compute_statistics(results_df)
        with open(output_dir / "strategy_comparison_stats.json", "w") as f:
            json.dump(stats_summary, f, indent=2)

        print(f"\n✓ Results saved to {output_dir}")


def plot_strategy_comparison(results_df: pd.DataFrame, save_path: Optional[Path] = None):
    """
    Create visualization comparing strategies.

    Args:
        results_df: DataFrame with comparison results
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Final Position Distribution
    ax = axes[0, 0]
    results_df.boxplot(column="final_position", by="strategy", ax=ax)
    ax.set_title("Final Position Distribution by Strategy")
    ax.set_xlabel("Strategy")
    ax.set_ylabel("Final Position")
    ax.get_figure().suptitle("")  # Remove default title

    # 2. Total Race Time
    ax = axes[0, 1]
    results_df.boxplot(column="total_race_time", by="strategy", ax=ax)
    ax.set_title("Total Race Time by Strategy")
    ax.set_xlabel("Strategy")
    ax.set_ylabel("Race Time (seconds)")
    ax.get_figure().suptitle("")

    # 3. Number of Pit Stops
    ax = axes[1, 0]
    pit_counts = results_df.groupby("strategy")["num_pitstops"].value_counts().unstack(fill_value=0)
    pit_counts.plot(kind="bar", ax=ax, stacked=False)
    ax.set_title("Pit Stop Distribution by Strategy")
    ax.set_xlabel("Strategy")
    ax.set_ylabel("Count")
    ax.legend(title="Num Pit Stops")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 4. Win/Podium/Points Rate
    ax = axes[1, 1]
    strategy_names = results_df["strategy"].unique()
    wins = [(results_df[results_df["strategy"] == s]["final_position"] == 1).sum() for s in strategy_names]
    podiums = [(results_df[results_df["strategy"] == s]["final_position"] <= 3).sum() for s in strategy_names]
    points = [(results_df[results_df["strategy"] == s]["final_position"] <= 10).sum() for s in strategy_names]

    x = np.arange(len(strategy_names))
    width = 0.25

    ax.bar(x - width, wins, width, label="Wins")
    ax.bar(x, podiums, width, label="Podiums")
    ax.bar(x + width, points, width, label="Points")

    ax.set_xlabel("Strategy")
    ax.set_ylabel("Count")
    ax.set_title("Success Metrics by Strategy")
    ax.set_xticks(x)
    ax.set_xticklabels(strategy_names, rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Comparison plot saved to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    from f1_env_final import create_f1_env
    from f1_agent import create_f1_agent

    print("Testing Strategy Comparison Framework...")

    # Create environment
    env = create_f1_env()

    # Test without agent first (baseline only)
    print("\nTesting baseline strategies only...")
    comparator = StrategyComparator(env)

    # Get a sample race and driver
    race_id = env.races_data["raceId"].iloc[0]
    driver_id = env.drivers_data["driverId"].iloc[0]

    print(f"Race ID: {race_id}, Driver ID: {driver_id}")

    # Run baselines
    for baseline_type in ["2-stop", "3-stop", "adaptive"]:
        strategy = create_baseline_strategy(baseline_type)
        result = comparator.run_baseline(strategy, race_id, driver_id)
        print(f"\n{strategy.name}: Position {result['final_position']}, "
              f"Stops: {result['num_pitstops']}, Time: {result['total_race_time']:.1f}s")

    print("\n✅ Strategy comparison test complete!")
