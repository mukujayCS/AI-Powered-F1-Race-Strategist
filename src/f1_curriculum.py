"""
F1 Race Strategy Curriculum Learning

Progressive training curriculum that teaches the agent pit stop strategy in stages:
1. Stage 1: Learn WHEN to pit (forced exactly 1 stop)
2. Stage 2: Learn HOW MANY stops (forced 1-2 stops)
3. Stage 3: Full freedom (learn optimal strategy)

Based on research showing curriculum learning reduces training episodes by 40%+
and improves convergence stability.
"""

class F1Curriculum:
    """
    Curriculum learning for F1 pit stop strategy.

    Stages:
    - Stage 1 (episodes 1-200): Must pit exactly once, learn timing
    - Stage 2 (episodes 201-500): Must pit 1-2 times, learn optimization
    - Stage 3 (episodes 501+): Full freedom, learn strategy
    """

    def __init__(self):
        self.current_episode = 0

        # Stage boundaries
        self.stage1_end = 200
        self.stage2_end = 500

    def update_episode(self, episode: int):
        """Update current episode number"""
        self.current_episode = episode

    def get_stage(self) -> int:
        """Get current curriculum stage (1, 2, or 3)"""
        if self.current_episode <= self.stage1_end:
            return 1
        elif self.current_episode <= self.stage2_end:
            return 2
        else:
            return 3

    def get_stage_name(self) -> str:
        """Get human-readable stage name"""
        stage = self.get_stage()
        if stage == 1:
            return "Stage 1: Learning WHEN to pit (1 stop required)"
        elif stage == 2:
            return "Stage 2: Learning optimization (1-2 stops required)"
        else:
            return "Stage 3: Full strategy (all actions allowed)"

    def is_action_allowed(self, action: int, current_pitstops: int, current_lap: int, total_laps: int) -> bool:
        """
        Check if action is allowed in current curriculum stage.

        Args:
            action: Action to take (0-6)
            current_pitstops: Number of pitstops made so far
            current_lap: Current lap number
            total_laps: Total laps in race

        Returns:
            True if action is allowed, False otherwise
        """
        stage = self.get_stage()
        is_pit_action = action >= 3  # Actions 3-6 are pit stops

        if stage == 1:
            # Stage 1: Must pit exactly once
            if current_pitstops == 0:
                # Haven't pitted yet - allow both pit and non-pit
                return True
            elif current_pitstops == 1:
                # Already pitted once - only allow non-pit actions
                return not is_pit_action
            else:
                # Shouldn't happen, but block pits if >1 stops
                return not is_pit_action

        elif stage == 2:
            # Stage 2: Must pit 1-2 times
            if current_pitstops < 2:
                # 0 or 1 stops - allow everything
                return True
            else:
                # Already 2 stops - only allow non-pit actions
                return not is_pit_action

        else:
            # Stage 3: Full freedom
            return True

    def get_forced_action(self, action: int, current_pitstops: int, current_lap: int, total_laps: int) -> int:
        """
        Force agent to comply with curriculum constraints.

        If agent chooses disallowed action, replace with allowed action.
        """
        if self.is_action_allowed(action, current_pitstops, current_lap, total_laps):
            return action

        stage = self.get_stage()
        is_pit_action = action >= 3

        if stage == 1 or stage == 2:
            if is_pit_action and current_pitstops >= (1 if stage == 1 else 2):
                # Agent tried to pit but already hit limit - force to neutral pace
                return 1  # Neutral pace, no pit

        return action

    def should_force_pit(self, current_pitstops: int, current_lap: int, total_laps: int) -> bool:
        """
        Check if we should force the agent to pit (ensure minimum pitstops).

        Returns True if agent MUST pit on this lap to meet curriculum requirements.
        """
        stage = self.get_stage()

        if stage == 1:
            # Stage 1: Must pit exactly once
            if current_pitstops == 0 and current_lap >= total_laps - 5:
                # Less than 5 laps left and haven't pitted - FORCE PIT
                return True

        elif stage == 2:
            # Stage 2: Must pit 1-2 times
            if current_pitstops == 0 and current_lap >= total_laps - 5:
                # Less than 5 laps left and haven't pitted - FORCE PIT
                return True

        return False

    def modify_reward(self, reward: float, num_pitstops: int, is_race_end: bool, current_lap: int = 0) -> float:
        """
        Modify reward based on curriculum compliance.

        Give bonus for meeting curriculum requirements, penalty for violating.
        """
        stage = self.get_stage()

        # CONTINUOUS PENALTIES during the race (not just at end!)
        if not is_race_end:
            if stage == 1:
                # Stage 1: Must have pitted by lap 30
                if num_pitstops == 0 and current_lap > 25:
                    penalty = (current_lap - 25) ** 2  # Exponentially increasing
                    reward -= penalty
            elif stage == 2:
                # Stage 2: Must have pitted by lap 30
                if num_pitstops == 0 and current_lap > 25:
                    penalty = (current_lap - 25) ** 2
                    reward -= penalty
            return reward

        # End-of-race curriculum rewards
        if stage == 1:
            # Stage 1: Reward exactly 1 stop
            if num_pitstops == 1:
                reward += 50.0  # Big bonus for compliance
            elif num_pitstops == 0:
                reward -= 100.0  # Massive penalty for 0 stops
            elif num_pitstops > 1:
                reward -= 20.0 * (num_pitstops - 1)  # Penalty for excess stops

        elif stage == 2:
            # Stage 2: Reward 1-2 stops
            if num_pitstops == 1 or num_pitstops == 2:
                reward += 30.0  # Good bonus for compliance
            elif num_pitstops == 0:
                reward -= 100.0  # Massive penalty for 0 stops
            elif num_pitstops > 2:
                reward -= 15.0 * (num_pitstops - 2)  # Penalty for excess stops

        # Stage 3: Use normal reward structure (no curriculum modification)

        return reward

    def get_progress_string(self) -> str:
        """Get progress string for logging"""
        stage = self.get_stage()
        if stage == 1:
            progress = (self.current_episode / self.stage1_end) * 100
            return f"Stage 1: {progress:.0f}% ({self.current_episode}/{self.stage1_end})"
        elif stage == 2:
            progress = ((self.current_episode - self.stage1_end) / (self.stage2_end - self.stage1_end)) * 100
            return f"Stage 2: {progress:.0f}% ({self.current_episode}/{self.stage2_end})"
        else:
            return f"Stage 3: Episode {self.current_episode}"
