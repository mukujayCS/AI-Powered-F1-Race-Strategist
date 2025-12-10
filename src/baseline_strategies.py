"""
Baseline Strategy Implementations for F1 Race Strategy

This module implements various baseline strategies to compare against the RL agent:
1. Historical Strategy Replay: Replay actual pit strategies from historical data
2. Fixed 2-Stop Strategy: Simple heuristic with 2 pit stops at fixed laps
3. Fixed 3-Stop Strategy: Simple heuristic with 3 pit stops at fixed laps
4. Adaptive Tire Degradation Strategy: Pit when tire degradation exceeds threshold

These baselines serve as counterfactual comparisons for evaluating the RL agent's
performance improvements.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from enum import IntEnum
from tqdm import tqdm


class TireCompound(IntEnum):
    """Tire compound types"""
    SOFT = 0
    MEDIUM = 1
    HARD = 2


class BaselineStrategy:
    """Base class for all baseline strategies"""

    def __init__(self, name: str):
        self.name = name

    def decide_action(self, state: Dict, lap: int, total_laps: int) -> int:
        """
        Decide what action to take given the current state.

        Args:
            state: Dictionary containing race state information
            lap: Current lap number
            total_laps: Total number of laps in the race

        Returns:
            action: Integer action (0-6 as defined in F1RaceEnv)
                0: Stay out, conserve pace
                1: Stay out, neutral pace
                2: Stay out, push pace
                3: Pit for soft tires
                4: Pit for medium tires
                5: Pit for hard tires
                6: Pit for same compound
        """
        raise NotImplementedError


class HistoricalReplayStrategy(BaselineStrategy):
    """
    Replay actual pit stop strategies from historical race data.

    This strategy extracts the pit stop decisions made by a specific driver
    in a specific race and replays them exactly.
    """

    def __init__(self, pitstops_data: pd.DataFrame, race_id: int, driver_id: int):
        super().__init__("Historical Replay")

        # Extract pit stops for this specific race and driver
        self.pit_stops = pitstops_data[
            (pitstops_data["raceId"] == race_id) &
            (pitstops_data["driverId"] == driver_id)
        ].sort_values("lap")

        self.pit_laps = self.pit_stops["lap"].tolist() if len(self.pit_stops) > 0 else []
        self.current_pit_index = 0

    def decide_action(self, state: Dict, lap: int, total_laps: int) -> int:
        """Replay historical pit stop decisions"""
        # Check if we should pit on this lap
        if self.current_pit_index < len(self.pit_laps):
            if lap >= self.pit_laps[self.current_pit_index]:
                self.current_pit_index += 1
                # Pit for medium tires (default strategy)
                return 4

        # Otherwise, maintain neutral pace
        return 1


class Fixed2StopStrategy(BaselineStrategy):
    """
    Simple 2-stop strategy with fixed pit stop laps.

    Pits on laps that divide the race into roughly equal thirds.
    Uses a tire rotation: MEDIUM -> SOFT -> MEDIUM
    """

    def __init__(self):
        super().__init__("Fixed 2-Stop")
        self.pit_count = 0

    def decide_action(self, state: Dict, lap: int, total_laps: int) -> int:
        """Execute fixed 2-stop strategy"""
        # Calculate pit stop laps (1/3 and 2/3 through race)
        first_stop = int(total_laps * 0.33)
        second_stop = int(total_laps * 0.67)

        # First pit stop: switch to soft tires
        if lap == first_stop and self.pit_count == 0:
            self.pit_count += 1
            return 3  # Pit for soft

        # Second pit stop: switch to medium tires
        elif lap == second_stop and self.pit_count == 1:
            self.pit_count += 1
            return 4  # Pit for medium

        # Manage pace based on tire age
        tire_age = state.get("tire_age", 0)
        if tire_age < 5:
            return 2  # Push on fresh tires
        elif tire_age > 15:
            return 0  # Conserve on old tires
        else:
            return 1  # Neutral pace


class Fixed3StopStrategy(BaselineStrategy):
    """
    Aggressive 3-stop strategy with fixed pit stop laps.

    Pits on laps that divide the race into roughly equal quarters.
    Uses a tire rotation: MEDIUM -> SOFT -> SOFT -> MEDIUM
    """

    def __init__(self):
        super().__init__("Fixed 3-Stop")
        self.pit_count = 0

    def decide_action(self, state: Dict, lap: int, total_laps: int) -> int:
        """Execute fixed 3-stop strategy"""
        # Calculate pit stop laps (1/4, 1/2, 3/4 through race)
        first_stop = int(total_laps * 0.25)
        second_stop = int(total_laps * 0.50)
        third_stop = int(total_laps * 0.75)

        # First pit stop: switch to soft tires
        if lap == first_stop and self.pit_count == 0:
            self.pit_count += 1
            return 3  # Pit for soft

        # Second pit stop: switch to soft tires again
        elif lap == second_stop and self.pit_count == 1:
            self.pit_count += 1
            return 3  # Pit for soft

        # Third pit stop: switch to medium tires
        elif lap == third_stop and self.pit_count == 2:
            self.pit_count += 1
            return 4  # Pit for medium

        # Always push with this aggressive strategy
        return 2


class AdaptiveTireDegradationStrategy(BaselineStrategy):
    """
    Adaptive strategy that pits based on tire degradation threshold.

    Monitors tire degradation and pits when it exceeds a threshold.
    Also considers race progress and minimum stint length.
    """

    def __init__(self, degradation_threshold: float = 0.70, min_stint_length: int = 8):
        super().__init__("Adaptive Tire Deg")
        self.degradation_threshold = degradation_threshold
        self.min_stint_length = min_stint_length
        self.pit_count = 0
        self.last_pit_lap = 0

    def decide_action(self, state: Dict, lap: int, total_laps: int) -> int:
        """Execute adaptive tire degradation strategy"""
        tire_degradation = state.get("tire_degradation", 0.0)
        tire_age = state.get("tire_age", 0)
        laps_remaining = total_laps - lap

        # Don't pit too frequently
        laps_since_pit = lap - self.last_pit_lap

        # Decision logic
        should_pit = False

        # Pit if degradation exceeds threshold and minimum stint length met
        if tire_degradation >= self.degradation_threshold and laps_since_pit >= self.min_stint_length:
            should_pit = True

        # Emergency pit if degradation is critical
        if tire_degradation >= 0.95:
            should_pit = True

        # Don't pit if too close to end of race
        if laps_remaining < 8:
            should_pit = False

        # Execute pit stop
        if should_pit:
            self.pit_count += 1
            self.last_pit_lap = lap

            # Choose tire compound based on race progress
            race_progress = lap / total_laps

            if race_progress < 0.40:
                # Early race: use medium tires for longevity
                return 4
            elif race_progress < 0.70:
                # Mid race: use soft tires for pace
                return 3
            else:
                # Late race: use soft tires for final push
                return 3

        # Pace management based on tire degradation
        if tire_degradation < 0.40:
            return 2  # Push on fresh tires
        elif tire_degradation < 0.65:
            return 1  # Neutral pace
        else:
            return 0  # Conserve on worn tires


class RandomStrategy(BaselineStrategy):
    """
    Random strategy for establishing a lower bound baseline.

    Makes random decisions with some constraints to avoid completely absurd strategies.
    """

    def __init__(self, pit_probability: float = 0.05):
        super().__init__("Random")
        self.pit_probability = pit_probability
        self.pit_count = 0
        self.last_pit_lap = 0

    def decide_action(self, state: Dict, lap: int, total_laps: int) -> int:
        """Make random decisions with basic constraints"""
        laps_since_pit = lap - self.last_pit_lap
        laps_remaining = total_laps - lap

        # Don't pit too frequently or too close to end
        can_pit = laps_since_pit >= 5 and laps_remaining >= 5

        if can_pit and np.random.random() < self.pit_probability:
            self.pit_count += 1
            self.last_pit_lap = lap
            # Random tire choice (3-5)
            return np.random.randint(3, 6)

        # Random pace choice (0-2)
        return np.random.randint(0, 3)


def create_baseline_strategy(
    strategy_type: str,
    pitstops_data: Optional[pd.DataFrame] = None,
    race_id: Optional[int] = None,
    driver_id: Optional[int] = None,
    **kwargs
) -> BaselineStrategy:
    """
    Factory function to create baseline strategies.

    Args:
        strategy_type: Type of strategy ("historical", "2-stop", "3-stop", "adaptive", "random")
        pitstops_data: Historical pit stop data (required for historical strategy)
        race_id: Race ID (required for historical strategy)
        driver_id: Driver ID (required for historical strategy)
        **kwargs: Additional parameters for specific strategies

    Returns:
        BaselineStrategy instance
    """
    strategy_type = strategy_type.lower()

    if strategy_type == "historical":
        if pitstops_data is None or race_id is None or driver_id is None:
            raise ValueError("Historical strategy requires pitstops_data, race_id, and driver_id")
        return HistoricalReplayStrategy(pitstops_data, race_id, driver_id)

    elif strategy_type == "2-stop" or strategy_type == "fixed-2-stop":
        return Fixed2StopStrategy()

    elif strategy_type == "3-stop" or strategy_type == "fixed-3-stop":
        return Fixed3StopStrategy()

    elif strategy_type == "adaptive" or strategy_type == "tire-deg":
        return AdaptiveTireDegradationStrategy(**kwargs)

    elif strategy_type == "random":
        return RandomStrategy(**kwargs)

    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")


if __name__ == "__main__":
    # Test baseline strategies
    print("Testing Baseline Strategies...")

    # Test state
    test_state = {
        "tire_age": 10,
        "tire_degradation": 0.5,
        "position": 5,
        "lap": 20
    }

    # Test each strategy
    strategies = [
        Fixed2StopStrategy(),
        Fixed3StopStrategy(),
        AdaptiveTireDegradationStrategy(),
        RandomStrategy()
    ]

    total_laps = 50

    for strategy in strategies:
        print(f"\n{strategy.name} Strategy:")
        pit_laps = []

        for lap in range(1, total_laps + 1):
            action = strategy.decide_action(test_state, lap, total_laps)
            if action >= 3:  # Pit stop actions
                pit_laps.append(lap)
                print(f"  Lap {lap}: PIT (action={action})")

        print(f"  Total pit stops: {len(pit_laps)}")
        print(f"  Pit laps: {pit_laps}")

    print("\n✅ Baseline strategies test complete!")
