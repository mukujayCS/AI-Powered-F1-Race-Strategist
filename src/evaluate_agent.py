"""
Evaluation Script for F1 Race Strategy Agent

This script evaluates a trained agent on multiple races and generates
performance metrics and comparisons.

Usage:
    python evaluate_agent.py --model models/f1_agent_final.pt --episodes 20
    python evaluate_agent.py --model models/f1_agent_final.pt --driver-id 1 --render
    python evaluate_agent.py --model models/f1_agent_final.pt --episodes 5 --visualize
"""

import argparse
from pathlib import Path
import json
from f1_env_final import create_f1_env
from f1_agent import create_f1_agent
from visualize_race import run_and_visualize_race


def main():
    parser = argparse.ArgumentParser(description="Evaluate F1 Race Strategy Agent")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--driver-id",
        type=int,
        default=None,
        help="Specific driver ID to evaluate (None = random drivers)"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render environment during evaluation"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.json",
        help="Output file for evaluation results"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate race visualizations for each episode"
    )
    parser.add_argument(
        "--viz-dir",
        type=str,
        default="race_visualizations",
        help="Directory to save race visualizations"
    )

    args = parser.parse_args()

    print("="*60)
    print("F1 RACE STRATEGY AGENT EVALUATION")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Evaluation Episodes: {args.episodes}")
    print(f"Driver ID: {args.driver_id if args.driver_id else 'Random'}")
    print("="*60)

    # Create environment
    print("\nInitializing environment...")
    render_mode = "human" if args.render else None
    env = create_f1_env(render_mode=render_mode)

    # Create agent and load model
    print("Loading trained model...")
    agent = create_f1_agent(env)
    agent.load_model(Path(args.model))

    # Evaluate agent
    print("\nStarting evaluation...\n")

    if args.visualize:
        # Create visualization directory
        viz_dir = Path(args.viz_dir)
        viz_dir.mkdir(parents=True, exist_ok=True)

        print(f"Visualizations will be saved to: {viz_dir}\n")

        # Run episodes with visualization
        eval_results = {
            'rewards': [],
            'positions': [],
            'pitstops': [],
            'race_times': []
        }

        for episode in range(args.episodes):
            print(f"\n{'='*60}")
            print(f"Episode {episode + 1}/{args.episodes}")
            print(f"{'='*60}")

            # Run race and create visualization
            viz_path = viz_dir / f"race_{episode + 1}.png"
            viz = run_and_visualize_race(
                agent, env,
                race_id=None,  # Random race
                driver_id=args.driver_id,
                render=args.render,
                save_path=viz_path
            )

            # Record results
            eval_results['positions'].append(viz.race_data['positions'][-1])
            eval_results['pitstops'].append(len(viz.race_data['pit_laps']))
            eval_results['rewards'].append(sum(viz.race_data['rewards']))

        # Compute statistics
        import numpy as np
        results = {
            'avg_reward': np.mean(eval_results['rewards']),
            'std_reward': np.std(eval_results['rewards']),
            'avg_position': np.mean(eval_results['positions']),
            'std_position': np.std(eval_results['positions']),
            'avg_pitstops': np.mean(eval_results['pitstops']),
            'wins': sum(1 for p in eval_results['positions'] if p == 1),
            'podiums': sum(1 for p in eval_results['positions'] if p <= 3),
            'points_finishes': sum(1 for p in eval_results['positions'] if p <= 10),
        }

        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        for key, value in results.items():
            print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")
        print("="*60)

    else:
        # Standard evaluation without visualizations
        results = agent.evaluate(
            num_episodes=args.episodes,
            driver_id=args.driver_id,
            render=args.render
        )

    # Save results
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\nResults saved to: {output_path}")
    if args.visualize:
        print(f"Visualizations saved to: {viz_dir}/")


if __name__ == "__main__":
    main()
