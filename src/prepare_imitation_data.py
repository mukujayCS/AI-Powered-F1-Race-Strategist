"""
Prepare Imitation Learning Dataset from Historical F1 Data

This script processes historical lap times and pit stop data to create
a supervised learning dataset for behavior cloning.

For each lap, we extract:
- State features (lap number, tire age, position, degradation estimate, etc.)
- Action label (0 = stay out, 1 = pit)

Output: imitation_dataset.npz containing (states, actions, weights)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


def load_data(data_dir: Path):
    """Load historical F1 data"""
    print("Loading historical data...")

    with open(data_dir / "22-24_lap_times.json") as f:
        lap_times = pd.DataFrame(json.load(f))

    with open(data_dir / "22-24_pitstops.json") as f:
        pitstops = pd.DataFrame(json.load(f))

    with open(data_dir / "22-24_races.json") as f:
        races = pd.DataFrame(json.load(f))

    print(f"✅ Loaded {len(lap_times):,} lap times")
    print(f"✅ Loaded {len(pitstops):,} pit stops")
    print(f"✅ Loaded {len(races):,} races")

    return lap_times, pitstops, races


def extract_state_features(row, tire_age, race_laps, position_normalized):
    """
    Extract state features for a given lap.

    These features mirror what the RL agent sees in the environment.
    """
    lap = row['lap']
    position = row['position']

    # Normalize features to [0, 1] range
    lap_progress = lap / race_laps
    tire_age_normalized = tire_age / 40.0  # Assuming max stint ~40 laps
    position_norm = position / 20.0  # Assuming 20 cars

    # Estimate tire degradation (simple linear model for now)
    # In reality, this comes from the environment's degradation model
    tire_degradation_estimate = min(1.0, tire_age * 0.025)  # ~40 laps to full degradation

    # State features (simplified version of environment state)
    state = np.array([
        lap_progress,           # Race progress (0-1)
        tire_age_normalized,    # Tire age (0-1)
        tire_degradation_estimate,  # Estimated degradation (0-1)
        position_norm,          # Current position (0-1)
        1.0 if tire_age < 5 else 0.0,  # Fresh tires indicator
        1.0 if tire_degradation_estimate > 0.7 else 0.0,  # Old tires indicator
        1.0 if lap_progress < 0.3 else 0.0,  # Early race
        1.0 if lap_progress > 0.7 else 0.0,  # Late race
    ], dtype=np.float32)

    return state


def create_imitation_dataset(lap_times, pitstops, races, min_laps=10):
    """
    Create supervised learning dataset from historical data.

    Returns:
        states: (N, state_dim) array of state features
        actions: (N,) array of pit decisions (0=stay out, 1=pit)
        weights: (N,) array of sample weights (more weight to pit decisions)
    """
    states_list = []
    actions_list = []
    weights_list = []

    # Create pit stop lookup: {(raceId, driverId, lap): True}
    pit_lookup = set(
        (row['raceId'], row['driverId'], row['lap'])
        for _, row in pitstops.iterrows()
    )

    # Create race laps lookup: {raceId: total_laps}
    race_laps = {}
    for _, race in races.iterrows():
        race_id = race['raceId']
        # Get max lap for this race from lap_times
        max_lap = lap_times[lap_times['raceId'] == race_id]['lap'].max()
        race_laps[race_id] = max_lap

    # Group by race and driver to track tire age
    print("Processing laps and extracting pit decisions...")

    grouped = lap_times.groupby(['raceId', 'driverId'])

    for (race_id, driver_id), driver_laps in tqdm(grouped, desc="Processing drivers"):
        driver_laps = driver_laps.sort_values('lap')

        # Skip if too few laps (DNF, crash, etc.)
        if len(driver_laps) < min_laps:
            continue

        # Track tire age for this driver
        tire_age = 0
        total_laps = race_laps.get(race_id, 50)

        for _, row in driver_laps.iterrows():
            lap = row['lap']

            # Skip first few laps (no pit decisions made in lap 1-3 typically)
            if lap < 4:
                tire_age += 1
                continue

            # Skip last lap (no pit decisions made on final lap)
            if lap >= total_laps:
                continue

            # Extract state features
            position_norm = row['position'] / 20.0
            state = extract_state_features(row, tire_age, total_laps, position_norm)

            # Check if driver pitted this lap
            did_pit = (race_id, driver_id, lap) in pit_lookup
            action = 1 if did_pit else 0

            # Sample weight: Give more weight to pit decisions (class imbalance)
            # Most laps are "stay out", few are "pit"
            weight = 10.0 if did_pit else 1.0

            states_list.append(state)
            actions_list.append(action)
            weights_list.append(weight)

            # Update tire age
            if did_pit:
                tire_age = 0  # Reset after pit stop
            else:
                tire_age += 1

    states = np.array(states_list, dtype=np.float32)
    actions = np.array(actions_list, dtype=np.int64)
    weights = np.array(weights_list, dtype=np.float32)

    print(f"\n✅ Created dataset:")
    print(f"   Total samples: {len(states):,}")
    print(f"   Pit decisions: {actions.sum():,} ({100*actions.mean():.2f}%)")
    print(f"   Stay out decisions: {(len(actions)-actions.sum()):,} ({100*(1-actions.mean()):.2f}%)")
    print(f"   State dimension: {states.shape[1]}")

    return states, actions, weights


def main():
    data_dir = Path("data/processed")
    output_file = Path("data/processed/imitation_dataset.npz")

    # Load data
    lap_times, pitstops, races = load_data(data_dir)

    # Create dataset
    states, actions, weights = create_imitation_dataset(lap_times, pitstops, races)

    # Save dataset
    print(f"\nSaving dataset to {output_file}...")
    np.savez_compressed(
        output_file,
        states=states,
        actions=actions,
        weights=weights
    )
    print("✅ Dataset saved!")

    # Print statistics
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    print(f"Total samples: {len(states):,}")
    print(f"Pit decisions: {actions.sum():,} ({100*actions.mean():.2f}%)")
    print(f"State features: {states.shape[1]}")
    print(f"Class balance (weighted): {weights[actions==1].sum() / weights.sum():.2%}")
    print("="*60)


if __name__ == "__main__":
    main()
