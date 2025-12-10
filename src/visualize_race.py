"""
Race Strategy Visualization

Creates comprehensive visualizations of F1 race strategies including:
- Lap times over the race
- Tire compound usage
- Pit stop timing
- Pace mode decisions (push/neutral/conserve)
- Tire degradation
- Position changes
- Fuel load

Usage:
    python visualize_race.py --model models/f1_agent_final.pt --race-id 1074 --driver-id 1
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import argparse
from pathlib import Path
from typing import List, Dict
from f1_env_final import create_f1_env
from f1_agent import create_f1_agent


class RaceVisualizer:
    """Visualizes F1 race strategy"""

    # Color schemes
    PACE_COLORS = {
        'conserve': '#3498db',  # Blue
        'neutral': '#95a5a6',   # Gray
        'push': '#e74c3c'       # Red
    }

    TIRE_COLORS = {
        'SOFT': '#e74c3c',      # Red
        'MEDIUM': '#f39c12',    # Yellow/Orange
        'HARD': '#ecf0f1'       # White/Light gray
    }

    ACTION_NAMES = {
        0: 'conserve',
        1: 'neutral',
        2: 'push',
        3: 'pit_soft',
        4: 'pit_medium',
        5: 'pit_hard',
        6: 'pit_same'
    }

    def __init__(self, figsize=(16, 12)):
        self.figsize = figsize
        self.race_data = {
            'laps': [],
            'lap_times': [],
            'positions': [],
            'tire_compounds': [],
            'tire_ages': [],
            'tire_degradations': [],
            'fuel_loads': [],
            'actions': [],
            'pace_modes': [],
            'pit_laps': [],
            'weather': [],
            'rewards': []
        }

    def record_lap(self, lap: int, lap_time: float, info: dict, action: int, reward: float):
        """Record data for one lap"""
        self.race_data['laps'].append(lap)
        self.race_data['lap_times'].append(lap_time)
        self.race_data['positions'].append(info['position'])
        self.race_data['tire_compounds'].append(info['tire_compound'])
        self.race_data['tire_ages'].append(info['tire_age'])
        self.race_data['tire_degradations'].append(info['tire_degradation'])
        self.race_data['fuel_loads'].append(info['fuel_load'])
        self.race_data['actions'].append(action)
        self.race_data['weather'].append(info['weather'])
        self.race_data['rewards'].append(reward)

        # Determine pace mode from action
        if action < 3:
            pace_mode = ['conserve', 'neutral', 'push'][action]
        else:
            pace_mode = 'pit'
        self.race_data['pace_modes'].append(pace_mode)

        # Record pit stops
        if action >= 3:
            self.race_data['pit_laps'].append(lap)

    def create_visualization(self, driver_name: str = "Driver", race_name: str = "Race",
                           save_path: Path = None):
        """Create comprehensive race strategy visualization"""

        fig = plt.figure(figsize=self.figsize)
        gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)

        # Main plot: Lap times with strategy overlay
        ax_main = fig.add_subplot(gs[0:2, :])
        self._plot_lap_times_with_strategy(ax_main, driver_name, race_name)

        # Secondary plots
        ax_position = fig.add_subplot(gs[2, 0])
        self._plot_position_changes(ax_position)

        ax_tires = fig.add_subplot(gs[2, 1])
        self._plot_tire_degradation(ax_tires)

        ax_fuel = fig.add_subplot(gs[3, 0])
        self._plot_fuel_load(ax_fuel)

        ax_strategy = fig.add_subplot(gs[3, 1])
        self._plot_strategy_summary(ax_strategy)

        plt.suptitle(f'{driver_name} - {race_name}\nRL Agent Strategy Visualization',
                    fontsize=16, fontweight='bold', y=0.995)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        else:
            plt.show()

        plt.close()

    def _plot_lap_times_with_strategy(self, ax, driver_name: str, race_name: str):
        """Main plot: Lap times with color-coded pace modes and tire stints"""

        laps = np.array(self.race_data['laps'])
        lap_times = np.array(self.race_data['lap_times'])
        pace_modes = self.race_data['pace_modes']
        tire_compounds = self.race_data['tire_compounds']

        # Create segments for colored line
        points = np.array([laps, lap_times]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Color each segment by pace mode
        colors = []
        for pace in pace_modes[:-1]:  # One less than points
            if pace == 'pit':
                colors.append('#2c3e50')  # Dark color for pit laps
            else:
                colors.append(self.PACE_COLORS[pace])

        # Create line collection
        lc = LineCollection(segments, colors=colors, linewidths=3, alpha=0.8)
        ax.add_collection(lc)

        # Add tire compound shading in background
        self._add_tire_stint_shading(ax, laps, lap_times)

        # Mark pit stops
        for pit_lap in self.race_data['pit_laps']:
            idx = self.race_data['laps'].index(pit_lap)
            ax.axvline(x=pit_lap, color='black', linestyle='--', linewidth=2, alpha=0.7)
            ax.scatter(pit_lap, lap_times[idx], s=200, color='black', marker='v',
                      zorder=5, edgecolors='white', linewidths=2)
            ax.text(pit_lap, lap_times[idx] + 1, 'PIT', ha='center', va='bottom',
                   fontsize=9, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3',
                   facecolor='black', edgecolor='white', alpha=0.8))

        # Set limits and labels
        ax.set_xlim(laps.min() - 1, laps.max() + 1)
        y_range = lap_times.max() - lap_times.min()
        ax.set_ylim(lap_times.min() - 0.1 * y_range, lap_times.max() + 0.15 * y_range)

        ax.set_xlabel('Lap Number', fontsize=12, fontweight='bold')
        ax.set_ylabel('Lap Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_title('Lap Times & Strategy Evolution', fontsize=14, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, linestyle='--')

        # Create legend
        pace_legend = [
            mpatches.Patch(color=self.PACE_COLORS['push'], label='Push (Aggressive)'),
            mpatches.Patch(color=self.PACE_COLORS['neutral'], label='Neutral (Balanced)'),
            mpatches.Patch(color=self.PACE_COLORS['conserve'], label='Conserve (Tire Saving)'),
            mpatches.Patch(color='#2c3e50', label='Pit Stop Lap')
        ]

        tire_legend = [
            mpatches.Patch(facecolor=self.TIRE_COLORS['SOFT'], edgecolor='black', label='Soft Tires'),
            mpatches.Patch(facecolor=self.TIRE_COLORS['MEDIUM'], edgecolor='black', label='Medium Tires'),
            mpatches.Patch(facecolor=self.TIRE_COLORS['HARD'], edgecolor='black', label='Hard Tires')
        ]

        # Two-column legend
        leg1 = ax.legend(handles=pace_legend, loc='upper left', fontsize=10,
                        title='Pace Mode', title_fontsize=11, framealpha=0.9)
        ax.add_artist(leg1)
        ax.legend(handles=tire_legend, loc='upper right', fontsize=10,
                 title='Tire Compound (Background)', title_fontsize=11, framealpha=0.9)

    def _add_tire_stint_shading(self, ax, laps, lap_times):
        """Add colored background regions for tire stints"""

        stint_start = laps[0]
        current_compound = self.race_data['tire_compounds'][0]

        for i in range(1, len(laps)):
            if self.race_data['tire_compounds'][i] != current_compound:
                # End of stint - add shading
                ax.axvspan(stint_start, laps[i-1],
                          facecolor=self.TIRE_COLORS[current_compound],
                          alpha=0.2, zorder=0)

                # Add compound label
                mid_lap = (stint_start + laps[i-1]) / 2
                ax.text(mid_lap, ax.get_ylim()[1] * 0.99, current_compound,
                       ha='center', va='top', fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.4',
                                facecolor=self.TIRE_COLORS[current_compound],
                                edgecolor='black', alpha=0.7))

                stint_start = laps[i]
                current_compound = self.race_data['tire_compounds'][i]

        # Last stint
        ax.axvspan(stint_start, laps[-1],
                  facecolor=self.TIRE_COLORS[current_compound],
                  alpha=0.2, zorder=0)
        mid_lap = (stint_start + laps[-1]) / 2
        ax.text(mid_lap, ax.get_ylim()[1] * 0.99, current_compound,
               ha='center', va='top', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.4',
                        facecolor=self.TIRE_COLORS[current_compound],
                        edgecolor='black', alpha=0.7))

    def _plot_position_changes(self, ax):
        """Plot position changes throughout race"""

        laps = self.race_data['laps']
        positions = self.race_data['positions']

        ax.plot(laps, positions, linewidth=2.5, color='#2c3e50', marker='o',
               markersize=4, markerfacecolor='#3498db', markeredgecolor='white',
               markeredgewidth=1)

        # Mark pit stops
        for pit_lap in self.race_data['pit_laps']:
            idx = self.race_data['laps'].index(pit_lap)
            ax.scatter(pit_lap, positions[idx], s=150, color='#e74c3c',
                      marker='v', zorder=5, edgecolors='white', linewidths=2)

        ax.set_xlabel('Lap Number', fontsize=11, fontweight='bold')
        ax.set_ylabel('Position', fontsize=11, fontweight='bold')
        ax.set_title('Race Position Over Time', fontsize=12, fontweight='bold')
        ax.invert_yaxis()  # Lower position is better
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_yticks(range(1, 21))

        # Add start and finish annotations
        ax.text(laps[0], positions[0], f' P{positions[0]} START',
               va='center', ha='left', fontsize=9, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.7))
        ax.text(laps[-1], positions[-1], f' P{positions[-1]} FINISH',
               va='center', ha='right', fontsize=9, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7))

    def _plot_tire_degradation(self, ax):
        """Plot tire degradation over race"""

        laps = self.race_data['laps']
        degradations = [d * 100 for d in self.race_data['tire_degradations']]  # Convert to %
        compounds = self.race_data['tire_compounds']

        # Color by tire compound
        for i in range(len(laps) - 1):
            ax.plot(laps[i:i+2], degradations[i:i+2],
                   color=self.TIRE_COLORS[compounds[i]],
                   linewidth=2.5, alpha=0.8)

        # Mark pit stops (degradation resets to 0)
        for pit_lap in self.race_data['pit_laps']:
            ax.axvline(x=pit_lap, color='black', linestyle='--', alpha=0.5)

        ax.set_xlabel('Lap Number', fontsize=11, fontweight='bold')
        ax.set_ylabel('Tire Degradation (%)', fontsize=11, fontweight='bold')
        ax.set_title('Tire Degradation by Stint', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim(0, 100)

        # Add danger zone shading
        ax.axhspan(80, 100, facecolor='red', alpha=0.1)
        ax.text(laps[-1] * 0.98, 90, 'DANGER ZONE', ha='right', va='center',
               fontsize=9, color='red', fontweight='bold', alpha=0.6)

    def _plot_fuel_load(self, ax):
        """Plot fuel consumption over race"""

        laps = self.race_data['laps']
        fuel = [f * 100 for f in self.race_data['fuel_loads']]  # Convert to %

        ax.fill_between(laps, 0, fuel, alpha=0.3, color='#27ae60')
        ax.plot(laps, fuel, linewidth=2.5, color='#27ae60', marker='o',
               markersize=3, markerfacecolor='white', markeredgecolor='#27ae60')

        ax.set_xlabel('Lap Number', fontsize=11, fontweight='bold')
        ax.set_ylabel('Fuel Load (%)', fontsize=11, fontweight='bold')
        ax.set_title('Fuel Consumption', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim(0, 100)

        # Add fuel status annotations
        if fuel[-1] < 5:
            ax.text(laps[-1], fuel[-1], ' LOW FUEL', ha='left', va='bottom',
                   fontsize=9, color='red', fontweight='bold')

    def _plot_strategy_summary(self, ax):
        """Plot strategy statistics and summary"""

        # Count actions
        action_counts = {}
        for action in self.race_data['actions']:
            action_name = self.ACTION_NAMES[action]
            if 'pit' in action_name:
                action_name = 'pit'
            action_counts[action_name] = action_counts.get(action_name, 0) + 1

        # Separate pace modes and pits
        pace_actions = ['conserve', 'neutral', 'push']
        pace_counts = [action_counts.get(a, 0) for a in pace_actions]
        pit_count = action_counts.get('pit', 0)

        # Create bar chart
        colors = [self.PACE_COLORS[a] for a in pace_actions]
        bars = ax.bar(pace_actions, pace_counts, color=colors, alpha=0.8,
                     edgecolor='black', linewidth=2)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)} laps',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.set_ylabel('Number of Laps', fontsize=11, fontweight='bold')
        ax.set_title('Pace Mode Distribution', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')

        # Add text summary in corner
        total_laps = len(self.race_data['laps'])
        summary_text = (
            f"Total Laps: {total_laps}\n"
            f"Pit Stops: {pit_count}\n"
            f"Final Position: P{self.race_data['positions'][-1]}\n"
            f"Avg Lap Time: {np.mean(self.race_data['lap_times']):.2f}s"
        )
        ax.text(0.98, 0.97, summary_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8))


def run_and_visualize_race(agent, env, race_id=None, driver_id=None,
                           render=False, save_path=None):
    """Run a race and create visualization"""

    # Create visualizer
    viz = RaceVisualizer()

    # Reset environment
    options = {}
    if race_id:
        options['race_id'] = race_id
    if driver_id:
        options['driver_id'] = driver_id

    state, info = env.reset(options=options)

    # Get race info
    race_name = f"Race {info.get('race_id', 'Unknown')}"
    driver_name = f"Driver {info.get('driver_id', 'Unknown')}"

    print(f"\nRunning race: {race_name}, {driver_name}")
    print("="*60)

    done = False
    lap_times_list = []
    prev_time = 0

    while not done:
        # Select action
        import torch
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_probs, _ = agent.policy(state_tensor)
            action = torch.argmax(action_probs, dim=-1).item()

        # Take step
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Calculate lap time (total time this lap)
        lap_time = info['total_race_time'] - prev_time
        prev_time = info['total_race_time']

        # Record lap data
        viz.record_lap(info['lap'], lap_time, info, action, reward)

        if render:
            env.render()

        print(f"Lap {info['lap']:2d}: P{info['position']:2d} | "
              f"{info['tire_compound']:6s} (age {info['tire_age']:2d}) | "
              f"Action: {viz.ACTION_NAMES[action]:12s} | "
              f"Time: {lap_time:6.2f}s")

        state = next_state

    print("="*60)
    print(f"Race Complete!")
    print(f"Final Position: P{info['position']}")
    print(f"Total Pitstops: {info['num_pitstops']}")
    print(f"Total Race Time: {info['total_race_time']:.2f}s")
    print("="*60)

    # Create visualization
    print("\nGenerating visualization...")
    viz.create_visualization(driver_name, race_name, save_path)

    return viz


def main():
    parser = argparse.ArgumentParser(description="Visualize F1 Race Strategy")
    parser.add_argument('--model', type=str, required=True, help="Path to trained model")
    parser.add_argument('--race-id', type=int, default=None, help="Specific race ID")
    parser.add_argument('--driver-id', type=int, default=None, help="Specific driver ID")
    parser.add_argument('--render', action='store_true', help="Render race in console")
    parser.add_argument('--output', type=str, default=None,
                       help="Output path for visualization (default: race_visualization.png)")

    args = parser.parse_args()

    print("="*60)
    print("F1 RACE STRATEGY VISUALIZATION")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Race ID: {args.race_id if args.race_id else 'Random'}")
    print(f"Driver ID: {args.driver_id if args.driver_id else 'Random'}")
    print("="*60)

    # Create environment and load agent
    env = create_f1_env(render_mode="human" if args.render else None)
    agent = create_f1_agent(env)
    agent.load_model(Path(args.model))

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path("race_visualization.png")

    # Run race and create visualization
    viz = run_and_visualize_race(
        agent, env,
        race_id=args.race_id,
        driver_id=args.driver_id,
        render=args.render,
        save_path=output_path
    )

    print(f"\n✅ Visualization complete!")


def create_comparison_visualization(
    results: List[Dict],
    race_name: str = "Race",
    driver_name: str = "Driver",
    save_path: Path = None
):
    """
    Create a comparison visualization showing RL agent vs baseline strategy.

    Args:
        results: List of result dictionaries from compare_strategies.py
                 Each dict should have: strategy, lap_times, positions, pit_laps, etc.
        race_name: Name of the race
        driver_name: Name of the driver
        save_path: Optional path to save the visualization
    """
    # Check if we have RL Agent results
    has_rl_agent = any(r['strategy'] == 'RL Agent' for r in results)

    # Use 3-panel layout: Lap Times, Positions, Summary Table
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Color scheme for different strategies
    strategy_colors = {
        'RL Agent': '#e74c3c',
        'Fixed 2-Stop': '#3498db'
    }

    # 1. Lap Time Comparison
    ax = axes[0]
    for result in results:
        strategy_name = result['strategy']
        laps = range(1, len(result['lap_times']) + 1)
        lap_times = result['lap_times']
        color = strategy_colors.get(strategy_name, '#34495e')

        # Plot lap times
        ax.plot(laps, lap_times, label=strategy_name, linewidth=2.5,
               alpha=0.8, color=color)

        # Mark pit stops
        for pit_lap in result['pit_laps']:
            if pit_lap <= len(lap_times):
                ax.scatter(pit_lap, lap_times[pit_lap - 1], s=100,
                          color=color, marker='v', zorder=5,
                          edgecolors='white', linewidths=1.5)

    ax.set_xlabel('Lap Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('Lap Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Lap Time Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')

    # 2. Position Comparison
    ax = axes[1]
    for result in results:
        strategy_name = result['strategy']
        laps = range(1, len(result['positions']) + 1)
        positions = result['positions']
        color = strategy_colors.get(strategy_name, '#34495e')

        ax.plot(laps, positions, label=strategy_name, linewidth=2.5,
               alpha=0.8, color=color, marker='o', markersize=3)

    ax.set_xlabel('Lap Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('Position', fontsize=12, fontweight='bold')
    ax.set_title('Race Position Comparison', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.legend(fontsize=10, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')

    # 3. Strategy Summary Table
    ax = axes[2]
    ax.axis('off')

    # Create summary table
    table_data = []
    headers = ['Strategy', 'Final Pos', 'Total Time', 'Pit Stops', 'Avg Lap Time']

    for result in results:
        avg_lap_time = np.mean(result['lap_times'])
        row = [
            result['strategy'],
            f"P{result['final_position']}",
            f"{result['total_race_time']:.1f}s",
            str(result['num_pitstops']),
            f"{avg_lap_time:.2f}s"
        ]
        table_data.append(row)

    # Sort by final position
    table_data.sort(key=lambda x: int(x[1][1:]))  # Extract position number

    # Create table
    table = ax.table(cellText=table_data, colLabels=headers,
                    cellLoc='center', loc='center',
                    bbox=[0, 0, 1, 1])

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    # Color header row
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Color rows by strategy
    for i, row in enumerate(table_data, start=1):
        strategy_name = row[0]
        color = strategy_colors.get(strategy_name, '#ecf0f1')

        for j in range(len(headers)):
            table[(i, j)].set_facecolor(color)
            table[(i, j)].set_alpha(0.3)

            # Bold the RL Agent row
            if strategy_name == 'RL Agent':
                table[(i, j)].set_text_props(weight='bold')

    ax.set_title('Strategy Performance Summary', fontsize=14,
                fontweight='bold', pad=20)

    # Overall title
    if has_rl_agent:
        plt.suptitle(f'{driver_name} - {race_name}\nRL Agent vs Fixed 2-Stop Baseline',
                    fontsize=16, fontweight='bold', y=1.02)
    else:
        plt.suptitle(f'{driver_name} - {race_name}\nBaseline Strategy Performance',
                    fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison visualization saved to: {save_path}")
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    main()
