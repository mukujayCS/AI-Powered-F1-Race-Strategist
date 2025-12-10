"""
Run Comparison: RL Agent vs Baseline Strategy

Simple command-line script to compare RL agent against a baseline strategy
and generate a comparison visualization.

Usage:
    python run_comparison.py --model models/f1_agent_final.pt --driver-id 1
    python run_comparison.py --model models/f1_agent_final.pt --race-id 1074 --driver-id 1 --output results/comparison.png
"""

import argparse
from pathlib import Path
import sys

from f1_env_final import create_f1_env
from f1_agent import create_f1_agent
from baseline_strategies import create_baseline_strategy
from compare_strategies import StrategyComparator
from visualize_race import create_comparison_visualization


def main():
    parser = argparse.ArgumentParser(
        description="Compare RL Agent vs Baseline Strategy"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model"
    )
    parser.add_argument(
        "--race-id",
        type=int,
        default=None,
        help="Specific race ID (None = random race)"
    )
    parser.add_argument(
        "--driver-id",
        type=int,
        default=None,
        help="Specific driver ID (None = random driver)"
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="2-stop",
        choices=["2-stop", "3-stop", "adaptive"],
        help="Baseline strategy type (default: 2-stop)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/rl_vs_baseline.png",
        help="Output path for comparison visualization"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("F1 RACE STRATEGY COMPARISON")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Race ID: {args.race_id if args.race_id else 'Random'}")
    print(f"Driver ID: {args.driver_id if args.driver_id else 'Random'}")
    print(f"Baseline: {args.baseline}")
    print(f"Output: {args.output}")
    print("=" * 60)

    # Load environment and trained agent
    print("\nInitializing environment...")
    env = create_f1_env()

    print("Loading trained model...")
    agent = create_f1_agent(env)
    agent.load_model(Path(args.model))

    # Create comparator
    comparator = StrategyComparator(env, agent)

    # Select race and driver
    if args.race_id is None:
        race_id = env.races_data['raceId'].iloc[0]
        print(f"Using first race in dataset: {race_id}")
    else:
        race_id = args.race_id

    if args.driver_id is None:
        driver_id = env.drivers_data['driverId'].iloc[0]
        print(f"Using first driver in dataset: {driver_id}")
    else:
        driver_id = args.driver_id

    # Get race and driver names for visualization
    race_name = f"Race {race_id}"
    driver_name = f"Driver {driver_id}"

    print(f"\nRunning RL Agent on Race {race_id}, Driver {driver_id}...")
    rl_result = comparator.run_rl_agent(race_id, driver_id)

    # Use the same starting position for fair comparison
    starting_position = rl_result["starting_position"]
    print(f"Starting position: P{starting_position}")

    print(f"Running {args.baseline} Baseline...")
    baseline_strategy = create_baseline_strategy(args.baseline)
    baseline_result = comparator.run_baseline(baseline_strategy, race_id, driver_id, starting_position=starting_position)

    # Print results
    print(f"\n{'=' * 60}")
    print("COMPARISON RESULTS")
    print(f"{'=' * 60}")
    print(f"RL Agent:  P{rl_result['final_position']}, "
          f"{rl_result['total_race_time']:.1f}s, "
          f"{rl_result['num_pitstops']} stops")
    print(f"Baseline:  P{baseline_result['final_position']}, "
          f"{baseline_result['total_race_time']:.1f}s, "
          f"{baseline_result['num_pitstops']} stops")
    print(f"{'=' * 60}")

    # Create comparison visualization
    print("\nCreating comparison visualization...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    create_comparison_visualization(
        [rl_result, baseline_result],
        race_name=race_name,
        driver_name=driver_name,
        save_path=output_path
    )

    print(f"\n✅ Comparison complete!")
    print(f"✅ Visualization saved to: {args.output}")


if __name__ == "__main__":
    main()
