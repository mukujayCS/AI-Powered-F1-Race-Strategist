"""
F1 Race Strategy Environment - Final Version

A comprehensive Gymnasium environment for training reinforcement learning agents
to make optimal F1 race strategy decisions including pit stops, tire management,
and pace control.

Key Features:
- Driver-specific performance profiles (consistency, aggression, tire/fuel management)
- Tire compound modeling (Soft, Medium, Hard) with degradation
- Weather transitions (Dry, Wet, Mixed)
- Track-specific characteristics (pit loss, overtaking difficulty, degradation)
- Lap time prediction model with statistical basis
- Opponent modeling for position changes
- Pit stop mechanics with realistic time penalties
"""

import json
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from enum import IntEnum
from tqdm import tqdm

# Import curriculum learning module (optional)
try:
    from f1_curriculum import F1Curriculum
except ImportError:
    F1Curriculum = None


class TireCompound(IntEnum):
    """Tire compound types"""
    SOFT = 0
    MEDIUM = 1
    HARD = 2


class WeatherCondition(IntEnum):
    """Weather condition types"""
    DRY = 0
    MIXED = 1
    WET = 2


class F1RaceEnv(gym.Env):
    """
    F1 Race Strategy Environment

    State Space:
        - current_lap (normalized)
        - position (normalized)
        - tire_age (laps on current tire)
        - tire_compound (one-hot: soft, medium, hard)
        - tire_degradation (0-1, normalized wear)
        - fuel_load (normalized, decreases each lap)
        - weather_condition (one-hot: dry, mixed, wet)
        - gap_to_leader (seconds, normalized)
        - gap_to_car_ahead (seconds, normalized)
        - gap_to_car_behind (seconds, normalized)
        - driver_profile (5 features: consistency, aggression, pit_eff, tire_mgmt, fuel_mgmt)
        - track_profile (3 features: pit_loss, overtake_diff, degradation)
        - num_pitstops_made (normalized)

    Action Space:
        - Discrete(7):
            0: Stay out, conserve pace
            1: Stay out, neutral pace
            2: Stay out, push pace
            3: Pit for soft tires
            4: Pit for medium tires
            5: Pit for hard tires
            6: Pit for same compound

    Reward:
        - Primary: Position improvement/maintenance
        - Penalties: Excessive pitstops, tire damage, poor finishing position
        - Bonuses: Optimal pit timing, finishing in points (top 10)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, data_dir: Path = Path("data"), render_mode: Optional[str] = None, curriculum: Optional['F1Curriculum'] = None):
        super().__init__()

        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / "processed"
        self.render_mode = render_mode
        self.curriculum = curriculum  # Optional curriculum learning

        # Load all data
        self._load_data()

        # Compute driver and track features
        self._compute_driver_features()
        self._compute_track_features()

        # Define action and observation spaces
        self.action_space = spaces.Discrete(7)

        # Observation space: detailed state vector
        # [lap, pos, tire_age, tire_one_hot(3), tire_deg, fuel, weather_one_hot(3),
        #  gaps(3), driver_profile(5), track_profile(3), num_stops]
        obs_dim = 1 + 1 + 1 + 3 + 1 + 1 + 3 + 3 + 5 + 3 + 1  # = 23
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32
        )

        # Environment state variables
        self.reset()

    def _load_data(self):
        """Load all processed data files"""
        data_files = [
            ("qualifying", "22-24_qualifying.json"),
            ("lap_times", "22-24_lap_times.json"),
            ("races", "22-24_races.json"),
            ("circuits", "22-24_circuits.json"),
            ("drivers", "22-24_drivers.json"),
            ("pitstops", "22-24_pitstops.json")
        ]

        with tqdm(data_files, desc="Loading data files", unit="file") as pbar:
            for attr_name, filename in pbar:
                pbar.set_postfix_str(filename)
                with open(self.processed_dir / filename, "r") as f:
                    setattr(self, f"{attr_name}_data", pd.DataFrame(json.load(f)))

        print("✅ Loaded all processed data files")

    def _compute_driver_features(self):
        """
        Compute driver-specific features from historical data.

        Features (normalized 0-1):
        - consistency: inverse of lap time std deviation
        - aggression: number of overtakes per race
        - pit_efficiency: inverse of average pit stop time
        - tire_management: inverse of lap time degradation slope
        - fuel_management: lap time consistency in final stint
        """
        driver_ids = self.drivers_data["driverId"].unique()
        self.driver_features = {}

        for driver_id in tqdm(driver_ids, desc="Computing driver features", unit="driver"):
            driver_laps = self.lap_times_data[self.lap_times_data["driverId"] == driver_id]
            driver_pitstops = self.pitstops_data[self.pitstops_data["driverId"] == driver_id]

            if len(driver_laps) < 10:  # Not enough data
                self.driver_features[driver_id] = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
                continue

            # Consistency: lower std = more consistent
            lap_std = driver_laps["lapTime_seconds"].std()
            consistency = 1.0 / (1.0 + lap_std / 10.0)  # normalize

            # Aggression: position changes (overtakes)
            position_changes = driver_laps.groupby("raceId")["position"].apply(
                lambda x: (x.diff() < 0).sum()  # negative diff = overtake
            )
            aggression = np.clip(position_changes.mean() / 10.0, 0, 1)

            # Pit efficiency: faster stops = better
            if len(driver_pitstops) > 0:
                avg_pit_time = driver_pitstops["stop_time_seconds"].mean()
                pit_efficiency = np.clip(1.0 - (avg_pit_time - 20.0) / 10.0, 0, 1)
            else:
                pit_efficiency = 0.5

            # Tire management: slower degradation = better
            # Compute lap time slope per race
            slopes = []
            for race_id in driver_laps["raceId"].unique():
                race_laps = driver_laps[driver_laps["raceId"] == race_id]
                if len(race_laps) > 5:
                    slope = np.polyfit(range(len(race_laps)), race_laps["lapTime_seconds"], 1)[0]
                    slopes.append(slope)
            avg_slope = np.mean(slopes) if slopes else 0
            tire_management = np.clip(1.0 - avg_slope / 0.5, 0, 1)

            # Fuel management: consistency in final laps
            final_lap_std = driver_laps.groupby("raceId", group_keys=False).apply(
                lambda x: x.tail(10)["lapTime_seconds"].std() if len(x) > 10 else 0
            ).mean()
            fuel_management = 1.0 / (1.0 + final_lap_std / 5.0)

            self.driver_features[driver_id] = np.array([
                consistency, aggression, pit_efficiency, tire_management, fuel_management
            ])

        print(f"✅ Computed features for {len(self.driver_features)} drivers")

    def _compute_track_features(self):
        """
        Compute track-specific features.

        Features (normalized 0-1):
        - pit_loss: estimated time loss for pit stop (track-dependent)
        - overtaking_difficulty: how hard it is to overtake (based on position changes)
        - degradation_index: tire degradation rate on this track
        """
        circuit_ids = self.circuits_data["circuitId"].unique()
        self.track_features = {}

        for circuit_id in tqdm(circuit_ids, desc="Computing track features", unit="track"):
            # Get races at this circuit
            circuit_races = self.races_data[self.races_data["circuitId"] == circuit_id]
            race_ids = circuit_races["raceId"].values

            if len(race_ids) == 0:
                self.track_features[circuit_id] = np.array([0.5, 0.5, 0.5])
                continue

            # Pit loss: average pit stop duration for this circuit
            circuit_pitstops = self.pitstops_data[self.pitstops_data["raceId"].isin(race_ids)]
            if len(circuit_pitstops) > 0:
                avg_pit_time = circuit_pitstops["stop_time_seconds"].mean()
                pit_loss = np.clip((avg_pit_time - 20.0) / 10.0, 0, 1)
            else:
                pit_loss = 0.5

            # Overtaking difficulty: fewer position changes = harder
            circuit_laps = self.lap_times_data[self.lap_times_data["raceId"].isin(race_ids)]
            position_changes = circuit_laps.groupby("raceId")["position"].apply(
                lambda x: (x.diff() != 0).sum()
            )
            overtaking_difficulty = 1.0 - np.clip(position_changes.mean() / 100.0, 0, 1)

            # Degradation: lap time increase over race
            lap_time_slopes = []
            for race_id in race_ids:
                race_laps = circuit_laps[circuit_laps["raceId"] == race_id]
                for driver_id in race_laps["driverId"].unique():
                    driver_race_laps = race_laps[race_laps["driverId"] == driver_id]
                    if len(driver_race_laps) > 10:
                        slope = np.polyfit(
                            range(len(driver_race_laps)),
                            driver_race_laps["lapTime_seconds"],
                            1
                        )[0]
                        lap_time_slopes.append(slope)

            avg_slope = np.mean(lap_time_slopes) if lap_time_slopes else 0.1
            degradation_index = np.clip(avg_slope / 0.5, 0, 1)

            self.track_features[circuit_id] = np.array([
                pit_loss, overtaking_difficulty, degradation_index
            ])

        print(f"✅ Computed features for {len(self.track_features)} tracks")

    def reset(self, seed=None, options=None):
        """Reset environment to start of a new race"""
        super().reset(seed=seed)

        # Select race and driver
        if options and "race_id" in options:
            self.current_race_id = options["race_id"]
        else:
            self.current_race_id = self.np_random.choice(self.races_data["raceId"].values)

        if options and "driver_id" in options:
            self.current_driver_id = options["driver_id"]
        else:
            self.current_driver_id = self.np_random.choice(self.drivers_data["driverId"].values)

        # Get race info
        race_info = self.races_data[self.races_data["raceId"] == self.current_race_id].iloc[0]
        self.current_circuit_id = race_info["circuitId"]

        # Get total laps for this race
        race_laps = self.lap_times_data[self.lap_times_data["raceId"] == self.current_race_id]
        self.total_laps = int(race_laps["lap"].max()) if len(race_laps) > 0 else 50

        # Get starting position from options, qualifying, or random
        if options and "starting_position" in options:
            self.starting_position = int(options["starting_position"])
        else:
            quali_results = self.qualifying_data[
                (self.qualifying_data["raceId"] == self.current_race_id) &
                (self.qualifying_data["driverId"] == self.current_driver_id)
            ]
            if len(quali_results) > 0:
                self.starting_position = int(quali_results.iloc[0]["position"])
            else:
                self.starting_position = self.np_random.integers(1, 21)

        # Initialize race state
        self.current_lap = 0
        self.position = self.starting_position
        self.tire_compound = TireCompound.MEDIUM  # Start on medium tires
        self.tire_age = 0
        self.tire_degradation = 0.0
        self.fuel_load = 1.0  # Full tank
        self.weather = WeatherCondition.DRY
        self.num_pitstops = 0
        self.total_race_time = 0.0

        # Track gaps (initialized with some randomness based on qualifying)
        self.gap_to_leader = (self.position - 1) * 0.5  # ~0.5s per position
        self.gap_to_car_ahead = 0.5 if self.position > 1 else 0.0
        self.gap_to_car_behind = 0.5 if self.position < 20 else 0.0

        # History tracking
        self.lap_times = []
        self.positions = [self.position]
        self.pit_laps = []

        return self._get_observation(), {"starting_position": self.starting_position}

    def _get_observation(self) -> np.ndarray:
        """Construct observation vector"""
        # Normalize values
        lap_norm = self.current_lap / self.total_laps
        position_norm = self.position / 20.0
        tire_age_norm = np.clip(self.tire_age / 30.0, 0, 1)  # Max 30 laps on tires
        fuel_norm = self.fuel_load

        # One-hot encode tire compound
        tire_one_hot = np.zeros(3)
        tire_one_hot[self.tire_compound] = 1.0

        # One-hot encode weather
        weather_one_hot = np.zeros(3)
        weather_one_hot[self.weather] = 1.0

        # Gaps (normalized by typical race gap)
        gap_leader_norm = np.clip(self.gap_to_leader / 60.0, 0, 1)
        gap_ahead_norm = np.clip(self.gap_to_car_ahead / 10.0, 0, 1)
        gap_behind_norm = np.clip(self.gap_to_car_behind / 10.0, 0, 1)

        # Driver and track profiles
        driver_profile = self.driver_features.get(
            self.current_driver_id, np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        )
        track_profile = self.track_features.get(
            self.current_circuit_id, np.array([0.5, 0.5, 0.5])
        )

        # Number of pitstops (normalized)
        pitstops_norm = np.clip(self.num_pitstops / 3.0, 0, 1)

        # Concatenate all features
        obs = np.concatenate([
            [lap_norm],
            [position_norm],
            [tire_age_norm],
            tire_one_hot,
            [self.tire_degradation],
            [fuel_norm],
            weather_one_hot,
            [gap_leader_norm, gap_ahead_norm, gap_behind_norm],
            driver_profile,
            track_profile,
            [pitstops_norm]
        ]).astype(np.float32)

        return obs

    def step(self, action: int):
        """Execute one step of the environment"""
        # Advance lap
        self.current_lap += 1
        self.tire_age += 1

        # Fuel consumption
        self.fuel_load = max(0.0, self.fuel_load - 1.0 / self.total_laps)

        # Apply curriculum constraints (if curriculum learning is enabled)
        if self.curriculum is not None:
            # Check if we should force a pit stop
            if self.curriculum.should_force_pit(self.num_pitstops, self.current_lap, self.total_laps):
                action = 4  # Force pit for medium tires
            else:
                # Enforce curriculum action constraints
                action = self.curriculum.get_forced_action(action, self.num_pitstops, self.current_lap, self.total_laps)

        # Parse action
        pit_decision = action >= 3  # Actions 3-6 are pit stops
        pace_mode = action % 3 if action < 3 else 1  # 0=conserve, 1=neutral, 2=push

        # Handle pit stop
        pit_time_loss = 0.0
        if pit_decision:
            self.num_pitstops += 1
            self.pit_laps.append(self.current_lap)

            # Determine new tire compound
            if action == 3:
                self.tire_compound = TireCompound.SOFT
            elif action == 4:
                self.tire_compound = TireCompound.MEDIUM
            elif action == 5:
                self.tire_compound = TireCompound.HARD
            elif action == 6:
                pass  # Keep same compound

            # Reset tire state
            self.tire_age = 0
            self.tire_degradation = 0.0

            # Pit stop time loss (20-25 seconds typical)
            track_pit_loss = self.track_features.get(self.current_circuit_id, np.array([0.5, 0.5, 0.5]))[0]
            pit_time_loss = 20.0 + track_pit_loss * 5.0 + self.np_random.uniform(-1, 1)

            # Position loss from pit stop (typically 2-4 positions)
            self.position = min(20, self.position + self.np_random.integers(2, 5))

        # EXPONENTIAL tire degradation (more realistic - tires degrade faster as they wear)
        # Formula: deg(t) = 1 - exp(-k * t^1.5)
        # This creates accelerating degradation as tire age increases
        track_deg_factor = self.track_features.get(self.current_circuit_id, np.array([0.5, 0.5, 0.5]))[2]
        base_deg_rate = 0.08  # Single compound (no soft/medium/hard distinction in data)

        # Exponential degradation: deg_rate increases with tire age
        # Early laps: slow degradation (e.g., lap 1-10: ~2% per lap)
        # Mid laps: moderate (e.g., lap 10-20: ~3-4% per lap)
        # Late laps: fast (e.g., lap 20+: ~5-8% per lap)
        degradation_factor = (self.tire_age / 30.0) ** 1.5  # Exponential curve
        lap_degradation = base_deg_rate * degradation_factor * (1 + track_deg_factor)

        self.tire_degradation = min(1.0, self.tire_degradation + lap_degradation)

        # Weather transitions (simple Markov chain)
        if self.np_random.random() < 0.05:  # 5% chance of weather change per lap
            self.weather = WeatherCondition(self.np_random.integers(0, 3))

        # Compute lap time
        lap_time = self._compute_lap_time(pace_mode, pit_time_loss)
        self.lap_times.append(lap_time)
        self.total_race_time += lap_time

        # Update position based on pace and competitors
        self._update_position(pace_mode)
        self.positions.append(self.position)

        # Update gaps (simplified)
        self._update_gaps()

        # Compute reward
        reward = self._compute_reward(action, pit_decision)

        # Check if race is done
        terminated = self.current_lap >= self.total_laps
        truncated = False

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _compute_lap_time(self, pace_mode: int, pit_time_loss: float) -> float:
        """
        Compute lap time based on current state.

        Factors:
        - Base lap time (track-specific)
        - Tire degradation
        - Fuel load
        - Weather
        - Pace mode (conserve/neutral/push)
        - Driver skill
        """
        # Get base lap time from historical data
        race_laps = self.lap_times_data[
            (self.lap_times_data["raceId"] == self.current_race_id) &
            (self.lap_times_data["driverId"] == self.current_driver_id)
        ]

        if len(race_laps) > 0:
            base_time = race_laps["lapTime_seconds"].median()
        else:
            # Fallback: use circuit average
            circuit_laps = self.lap_times_data[
                self.lap_times_data["raceId"].isin(
                    self.races_data[self.races_data["circuitId"] == self.current_circuit_id]["raceId"]
                )
            ]
            base_time = circuit_laps["lapTime_seconds"].median() if len(circuit_laps) > 0 else 90.0

        # Tire degradation effect (up to +3 seconds for fully worn tires)
        tire_effect = self.tire_degradation * 3.0

        # Compound effect (soft is faster but degrades quicker)
        compound_effect = {
            TireCompound.SOFT: -0.5,
            TireCompound.MEDIUM: 0.0,
            TireCompound.HARD: 0.5
        }[self.tire_compound]

        # Fuel effect (heavier = slower, up to +0.5s at full tank)
        fuel_effect = self.fuel_load * 0.5

        # Weather effect
        weather_effect = {
            WeatherCondition.DRY: 0.0,
            WeatherCondition.MIXED: 2.0,
            WeatherCondition.WET: 5.0
        }[self.weather]

        # Pace mode effect
        pace_effect = {
            0: 0.5,   # Conserve: slower but saves tires
            1: 0.0,   # Neutral: baseline
            2: -0.3   # Push: faster but hurts tires
        }[pace_mode]

        # Driver consistency (random variation based on driver skill)
        driver_profile = self.driver_features.get(self.current_driver_id, np.array([0.5, 0.5, 0.5, 0.5, 0.5]))
        consistency = driver_profile[0]
        random_variation = self.np_random.normal(0, (1 - consistency) * 2.0)

        # Total lap time
        lap_time = (
            base_time +
            tire_effect +
            compound_effect +
            fuel_effect +
            weather_effect +
            pace_effect +
            random_variation +
            pit_time_loss
        )

        return max(lap_time, base_time * 0.8)  # Can't be too fast

    def _update_position(self, pace_mode: int):
        """Update position based on pace relative to competitors"""
        # Simplified position change model
        # Faster pace = more likely to gain positions
        # Tire degradation = more likely to lose positions

        pace_factor = {
            0: -0.1,  # Conserve: slight position loss risk
            1: 0.0,   # Neutral: maintain
            2: 0.2    # Push: position gain opportunity
        }[pace_mode]

        tire_factor = -self.tire_degradation * 0.3

        position_change_prob = pace_factor + tire_factor

        if self.np_random.random() < abs(position_change_prob):
            if position_change_prob > 0:
                # Gain position (move forward)
                self.position = max(1, self.position - 1)
            else:
                # Lose position (move backward)
                self.position = min(20, self.position + 1)

    def _update_gaps(self):
        """Update gaps to cars around (simplified)"""
        # This is a simplified model - in reality, gaps depend on all drivers' lap times
        self.gap_to_leader = max(0, self.gap_to_leader + self.np_random.uniform(-0.5, 0.5))
        self.gap_to_car_ahead = max(0, self.gap_to_car_ahead + self.np_random.uniform(-0.3, 0.3))
        self.gap_to_car_behind = max(0, self.gap_to_car_behind + self.np_random.uniform(-0.3, 0.3))

    def _compute_reward(self, action: int, pit_decision: bool) -> float:
        """
        DENSE REWARD STRUCTURE - Feedback every single lap!

        Key principle: Agent should know EVERY lap if it's on track for a good strategy
        """
        reward = 0.0
        race_progress = self.current_lap / self.total_laps  # 0.0 to 1.0

        # ===== LAP-BY-LAP DENSE REWARDS =====

        # 1. POSITION TRACKING (every lap)
        if len(self.positions) >= 2:
            position_change = self.positions[-2] - self.positions[-1]
            reward += position_change * 2.0  # Strong position signal

        # 2. TIRE MANAGEMENT (every lap) - Critical for strategy
        if self.tire_degradation < 0.4:
            reward += 0.5  # Good: Fresh tires
        elif 0.4 <= self.tire_degradation <= 0.7:
            reward += 1.0  # Excellent: Optimal tire window
        elif 0.7 < self.tire_degradation <= 0.85:
            reward += 0.3  # OK: Tires getting old, should pit soon
        else:  # > 0.85
            reward -= 1.0  # Bad: Tires too degraded

        # 3. PIT STOP STRATEGY FEEDBACK (every lap after lap 15)
        if self.current_lap > 15:
            expected_stops_by_now = race_progress * 1.5  # Should have ~1.5 stops by end

            if self.num_pitstops == 0:
                # No pits yet - penalize increasingly
                reward -= (self.current_lap - 15) * 0.5  # Growing penalty
            elif self.num_pitstops == 1:
                # Good: 1 stop made
                if race_progress < 0.7:
                    reward += 1.0  # On track for 1-2 stop strategy
                else:
                    reward += 0.5  # Might need second stop
            elif self.num_pitstops == 2:
                # Good: 2 stops made
                reward += 1.0  # Classic 2-stop strategy
            else:
                # Too many stops
                reward -= 2.0 * (self.num_pitstops - 2)  # Penalize excess

        # 4. STINT LENGTH REWARD (encourage reasonable stints)
        if self.tire_age > 5:  # After 5 laps on tires
            if self.tire_age <= 25:
                reward += 0.3  # Good stint length
            else:
                reward -= 0.5  # Stint too long

        # 5. STRATEGIC PITTING REWARD (when actually pitting)
        if pit_decision:
            # Reward pitting at right time
            if 0.5 < self.tire_degradation < 0.9 and self.tire_age >= 10:
                reward += 10.0  # Excellent pit timing!
            elif 0.3 < self.tire_degradation < 0.5:
                reward += 3.0  # Decent timing
            elif self.tire_degradation < 0.2:
                reward -= 8.0  # Way too early

            # Encourage 1-2 stops total
            if self.num_pitstops <= 2:
                reward += 3.0  # Good: Building toward optimal strategy
            else:
                reward -= 10.0 * (self.num_pitstops - 2)  # Bad: Too many stops

        # 6. RACE PROGRESS CHECKPOINTS (dense feedback at milestones)
        if self.current_lap == 20:  # ~40% through race
            if self.num_pitstops >= 1:
                reward += 5.0  # Good: Made first stop
            else:
                reward -= 10.0  # Bad: Should have pitted by now

        if self.current_lap == 35:  # ~70% through race
            if 1 <= self.num_pitstops <= 2:
                reward += 5.0  # Good: On track
            else:
                reward -= 10.0  # Bad: Wrong number of stops

        # ===== END OF RACE FINAL EVALUATION =====
        if self.current_lap >= self.total_laps:
            # Disqualification for no stops
            if self.num_pitstops == 0:
                self.position = 20
                return -50.0  # Reduced from -100 since we have dense penalties

            # Final strategy evaluation
            if self.num_pitstops == 1 or self.num_pitstops == 2:
                reward += 15.0  # Bonus for optimal stops

            # Position-based rewards (scaled down since we reward positions every lap)
            position_reward = (21 - self.position) ** 1.5 / 3.0
            reward += position_reward

            # Points/podium bonuses
            if self.position <= 10:
                reward += 5.0
            if self.position <= 3:
                reward += 10.0
            if self.position == 1:
                reward += 15.0

        # Apply curriculum learning reward modifications (if enabled)
        if self.curriculum is not None:
            is_race_end = self.current_lap >= self.total_laps
            reward = self.curriculum.modify_reward(reward, self.num_pitstops, is_race_end, self.current_lap)

        return reward

    def _get_info(self) -> dict:
        """Return additional information about the current state"""
        return {
            "lap": self.current_lap,
            "position": self.position,
            "tire_compound": self.tire_compound.name,
            "tire_age": self.tire_age,
            "tire_degradation": self.tire_degradation,
            "fuel_load": self.fuel_load,
            "weather": self.weather.name,
            "num_pitstops": self.num_pitstops,
            "total_race_time": self.total_race_time,
            "driver_id": self.current_driver_id,
            "race_id": self.current_race_id,
        }

    def render(self):
        """Render the current state (optional)"""
        if self.render_mode == "human":
            print(f"\n{'='*60}")
            print(f"Lap {self.current_lap}/{self.total_laps} | Position: {self.position}")
            print(f"Tire: {self.tire_compound.name} (Age: {self.tire_age}, Deg: {self.tire_degradation:.2f})")
            print(f"Fuel: {self.fuel_load:.2%} | Weather: {self.weather.name}")
            print(f"Gaps: Leader +{self.gap_to_leader:.2f}s, Ahead +{self.gap_to_car_ahead:.2f}s")
            print(f"Pitstops: {self.num_pitstops}")
            print(f"{'='*60}")


def create_f1_env(data_dir: Path = Path("data"), render_mode: Optional[str] = None, curriculum=None) -> F1RaceEnv:
    """Factory function to create F1 environment with optional curriculum learning"""
    return F1RaceEnv(data_dir=data_dir, render_mode=render_mode, curriculum=curriculum)


if __name__ == "__main__":
    # Test the environment
    print("Testing F1 Race Environment...")
    env = create_f1_env(render_mode="human")

    obs, info = env.reset()
    print(f"Initial state shape: {obs.shape}")
    print(f"Initial info: {info}")

    # Run a few steps
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        print(f"Action: {action}, Reward: {reward:.2f}")

        if terminated or truncated:
            break

    print("\n✅ Environment test complete!")
