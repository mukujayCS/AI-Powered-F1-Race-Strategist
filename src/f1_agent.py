"""
F1 Race Strategy Agent

Reinforcement learning agent for F1 race strategy using Proximal Policy Optimization (PPO).
This agent learns optimal pit stop timing, tire compound selection, and pace management
based on driver-specific profiles and race conditions.

Features:
- PPO algorithm with actor-critic architecture
- Driver-personalized training
- Multi-race training with experience replay
- Model checkpointing and evaluation
- Visualization of training progress
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from collections import deque
import matplotlib.pyplot as plt
from tqdm import tqdm


class ActorCritic(nn.Module):
    """
    Actor-Critic neural network for PPO.

    Actor: Outputs action probabilities
    Critic: Outputs state value estimation
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super(ActorCritic, self).__init__()

        # Shared feature extraction layers
        self.shared_layers = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Actor head (policy network)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Softmax(dim=-1)
        )

        # Critic head (value network)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, state):
        """Forward pass through the network"""
        shared_features = self.shared_layers(state)
        action_probs = self.actor(shared_features)
        state_value = self.critic(shared_features)
        return action_probs, state_value

    def act(self, state):
        """Select action based on current policy"""
        state = torch.FloatTensor(state).unsqueeze(0)
        action_probs, state_value = self.forward(state)

        # Sample action from probability distribution
        dist = Categorical(action_probs)
        action = dist.sample()

        # Return action, log probability, and value
        return action.item(), dist.log_prob(action), state_value

    def evaluate(self, states, actions):
        """Evaluate actions for PPO update"""
        action_probs, state_values = self.forward(states)

        dist = Categorical(action_probs)
        action_log_probs = dist.log_prob(actions)
        dist_entropy = dist.entropy()

        return action_log_probs, state_values, dist_entropy


class PPOMemory:
    """Memory buffer for storing experience"""

    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def add(self, state, action, log_prob, reward, value, done):
        """Add experience to memory"""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        """Clear memory"""
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def get_batches(self, batch_size: int):
        """Get randomized batches for training"""
        n_samples = len(self.states)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        for start_idx in range(0, n_samples, batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]

            yield (
                torch.FloatTensor(np.array(self.states)[batch_indices]),
                torch.LongTensor(np.array(self.actions)[batch_indices]),
                torch.FloatTensor(np.array(self.log_probs)[batch_indices]),
                torch.FloatTensor(np.array(self.rewards)[batch_indices]),
                torch.FloatTensor(np.array(self.values)[batch_indices]),
                torch.FloatTensor(np.array(self.dones)[batch_indices])
            )


class F1PPOAgent:
    """
    PPO Agent for F1 Race Strategy

    Hyperparameters tuned for F1 racing context where:
    - Episodes are long (50+ laps)
    - Actions have delayed consequences
    - Reward is sparse (concentrated at race end)
    """

    def __init__(
        self,
        env: gym.Env,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        c1: float = 0.5,  # Value loss coefficient
        c2: float = 0.01,  # Entropy coefficient
        epochs: int = 10,
        batch_size: int = 64,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.env = env
        self.device = device

        # Hyperparameters
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.c1 = c1
        self.c2 = c2
        self.epochs = epochs
        self.batch_size = batch_size

        # Network
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        self.policy = ActorCritic(obs_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Memory
        self.memory = PPOMemory()

        # Metrics tracking
        self.episode_rewards = []
        self.episode_positions = []
        self.episode_pitstops = []
        self.training_losses = []

    def select_action(self, state):
        """Select action using current policy"""
        with torch.no_grad():
            action, log_prob, value = self.policy.act(state)
        return action, log_prob.item(), value.item()

    def compute_gae(self, rewards, values, dones):
        """Compute Generalized Advantage Estimation"""
        advantages = []
        gae = 0

        # Reverse iteration for temporal difference
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        return torch.FloatTensor(advantages)

    def update_policy(self):
        """Update policy using PPO algorithm"""
        # Convert memory to tensors
        states = torch.FloatTensor(np.array(self.memory.states)).to(self.device)
        actions = torch.LongTensor(np.array(self.memory.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.memory.log_probs)).to(self.device)
        rewards = self.memory.rewards
        values = self.memory.values
        dones = self.memory.dones

        # Compute advantages using GAE
        advantages = self.compute_gae(rewards, values, dones).to(self.device)

        # Compute returns (targets for value function)
        returns = advantages + torch.FloatTensor(values).to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update for multiple epochs
        total_loss = 0
        for _ in tqdm(range(self.epochs), desc="PPO Update", leave=False, disable=self.epochs < 5):
            # Evaluate actions with current policy
            log_probs, state_values, entropy = self.policy.evaluate(states, actions)

            # Compute ratio for PPO
            ratios = torch.exp(log_probs - old_log_probs)

            # Compute surrogate losses
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages

            # Total loss
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(state_values.squeeze(), returns)
            entropy_loss = -entropy.mean()

            loss = actor_loss + self.c1 * critic_loss + self.c2 * entropy_loss

            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / self.epochs
        self.training_losses.append(avg_loss)

        # Clear memory
        self.memory.clear()

        return avg_loss

    def train(
        self,
        num_episodes: int,
        driver_id: Optional[int] = None,
        save_dir: Path = Path("models"),
        save_freq: int = 100,
        verbose: bool = True
    ):
        """
        Train the agent

        Args:
            num_episodes: Number of races to train on
            driver_id: Specific driver to train for (None = train on all)
            save_dir: Directory to save model checkpoints
            save_freq: Save model every N episodes
            verbose: Print training progress
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"Starting training for {num_episodes} episodes...")
        print(f"Device: {self.device}")

        # Create progress bar
        pbar = tqdm(range(1, num_episodes + 1), desc="Training", unit="episode")

        for episode in pbar:
            # Update curriculum stage (if curriculum learning is enabled)
            if hasattr(self.env, 'curriculum') and self.env.curriculum is not None:
                self.env.curriculum.update_episode(episode)

            # Reset environment
            options = {"driver_id": driver_id} if driver_id else None
            state, info = self.env.reset(options=options)

            episode_reward = 0
            done = False
            step = 0

            # Collect experience for one episode
            while not done:
                # Select action
                action, log_prob, value = self.select_action(state)

                # Take action in environment
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                # Store experience
                self.memory.add(state, action, log_prob, reward, value, done)

                episode_reward += reward
                state = next_state
                step += 1

            # Update policy after episode
            loss = self.update_policy()

            # Track metrics
            self.episode_rewards.append(episode_reward)
            self.episode_positions.append(info["position"])
            self.episode_pitstops.append(info["num_pitstops"])

            # Update progress bar with latest metrics
            if len(self.episode_rewards) >= 10:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_position = np.mean(self.episode_positions[-10:])
                avg_pitstops = np.mean(self.episode_pitstops[-10:])
                pbar.set_postfix({
                    'Reward': f'{avg_reward:.2f}',
                    'Pos': f'{avg_position:.2f}',
                    'Stops': f'{avg_pitstops:.2f}',
                    'Loss': f'{loss:.4f}'
                })
            else:
                pbar.set_postfix({
                    'Reward': f'{episode_reward:.2f}',
                    'Pos': f'{info["position"]}',
                    'Stops': f'{info["num_pitstops"]}',
                    'Loss': f'{loss:.4f}'
                })

            # Logging
            if verbose and episode % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_position = np.mean(self.episode_positions[-10:])
                avg_pitstops = np.mean(self.episode_pitstops[-10:])

                log_msg = (f"Episode {episode}/{num_episodes} | "
                          f"Avg Reward: {avg_reward:.2f} | "
                          f"Avg Position: {avg_position:.2f} | "
                          f"Avg Pitstops: {avg_pitstops:.2f} | "
                          f"Loss: {loss:.4f}")

                # Add curriculum stage info if enabled
                if hasattr(self.env, 'curriculum') and self.env.curriculum is not None:
                    stage_info = self.env.curriculum.get_progress_string()
                    log_msg += f" | {stage_info}"

                tqdm.write(log_msg)

            # Save checkpoint
            if episode % save_freq == 0:
                self.save_model(save_dir / f"f1_agent_episode_{episode}.pt")

        pbar.close()

        print("Training complete!")
        self.save_model(save_dir / "f1_agent_final.pt")

    def evaluate(
        self,
        num_episodes: int = 10,
        driver_id: Optional[int] = None,
        render: bool = False
    ) -> Dict:
        """
        Evaluate trained agent

        Returns:
            Dictionary with evaluation metrics
        """
        print(f"Evaluating agent for {num_episodes} episodes...")

        eval_rewards = []
        eval_positions = []
        eval_pitstops = []
        eval_race_times = []

        pbar = tqdm(range(num_episodes), desc="Evaluation", unit="episode")

        for episode in pbar:
            options = {"driver_id": driver_id} if driver_id else None
            state, info = self.env.reset(options=options)

            episode_reward = 0
            done = False

            while not done:
                # Select action (use sampling like training for consistent behavior)
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    action_probs, _ = self.policy(state_tensor)
                    # Sample from distribution instead of argmax
                    dist = Categorical(action_probs)
                    action = dist.sample().item()

                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                if render:
                    self.env.render()

                episode_reward += reward
                state = next_state

            eval_rewards.append(episode_reward)
            eval_positions.append(info["position"])
            eval_pitstops.append(info["num_pitstops"])
            eval_race_times.append(info["total_race_time"])

            pbar.set_postfix({
                'Pos': info['position'],
                'Reward': f'{episode_reward:.2f}',
                'Stops': info['num_pitstops']
            })

            tqdm.write(f"Episode {episode + 1}: "
                      f"Position {info['position']}, "
                      f"Reward {episode_reward:.2f}, "
                      f"Pitstops {info['num_pitstops']}")

        pbar.close()

        results = {
            "avg_reward": np.mean(eval_rewards),
            "std_reward": np.std(eval_rewards),
            "avg_position": np.mean(eval_positions),
            "std_position": np.std(eval_positions),
            "avg_pitstops": np.mean(eval_pitstops),
            "avg_race_time": np.mean(eval_race_times),
            "wins": sum(1 for p in eval_positions if p == 1),
            "podiums": sum(1 for p in eval_positions if p <= 3),
            "points_finishes": sum(1 for p in eval_positions if p <= 10),
        }

        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        for key, value in results.items():
            print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")
        print("="*60)

        return results

    def save_model(self, path: Path):
        """Save model checkpoint"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'episode_positions': self.episode_positions,
            'episode_pitstops': self.episode_pitstops,
            'training_losses': self.training_losses,
            'hyperparameters': {
                'lr': self.lr,
                'gamma': self.gamma,
                'gae_lambda': self.gae_lambda,
                'clip_epsilon': self.clip_epsilon,
                'c1': self.c1,
                'c2': self.c2,
            }
        }, path)

        print(f"Model saved to {path}")

    def load_model(self, path: Path):
        """Load model checkpoint"""
        print(f"Loading model from {path}...")
        with tqdm(total=4, desc="Loading model", unit="component") as pbar:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            pbar.update(1)

            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            pbar.update(1)

            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            pbar.update(1)

            self.episode_rewards = checkpoint.get('episode_rewards', [])
            self.episode_positions = checkpoint.get('episode_positions', [])
            self.episode_pitstops = checkpoint.get('episode_pitstops', [])
            self.training_losses = checkpoint.get('training_losses', [])
            pbar.update(1)

        print(f"✓ Model loaded successfully!")

    def plot_training_progress(self, save_path: Optional[Path] = None):
        """Plot training metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Episode rewards
        axes[0, 0].plot(self.episode_rewards, alpha=0.6, label='Episode Reward')
        axes[0, 0].plot(
            np.convolve(self.episode_rewards, np.ones(50)/50, mode='valid'),
            label='Moving Average (50 episodes)',
            linewidth=2
        )
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].set_title('Training Rewards')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Final positions
        axes[0, 1].plot(self.episode_positions, alpha=0.6, label='Final Position')
        axes[0, 1].plot(
            np.convolve(self.episode_positions, np.ones(50)/50, mode='valid'),
            label='Moving Average (50 episodes)',
            linewidth=2
        )
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Position')
        axes[0, 1].set_title('Final Race Positions')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        axes[0, 1].invert_yaxis()  # Lower position is better

        # Pitstops
        axes[1, 0].plot(self.episode_pitstops, alpha=0.6, label='Pitstops')
        axes[1, 0].plot(
            np.convolve(self.episode_pitstops, np.ones(50)/50, mode='valid'),
            label='Moving Average (50 episodes)',
            linewidth=2
        )
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Number of Pitstops')
        axes[1, 0].set_title('Pitstop Strategy')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Training loss
        if self.training_losses:
            axes[1, 1].plot(self.training_losses, linewidth=2)
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].set_title('Training Loss')
            axes[1, 1].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training plot saved to {save_path}")
        else:
            plt.show()


def create_f1_agent(env: gym.Env, **kwargs) -> F1PPOAgent:
    """Factory function to create F1 agent"""
    return F1PPOAgent(env, **kwargs)


if __name__ == "__main__":
    from f1_env_final import create_f1_env

    print("Creating F1 environment and agent...")

    # Create environment
    env = create_f1_env()

    # Create agent
    agent = create_f1_agent(env)

    print(f"\nAgent created with {sum(p.numel() for p in agent.policy.parameters())} parameters")
    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.n}")

    # Quick test
    print("\nTesting agent with random actions...")
    state, info = env.reset()
    action, log_prob, value = agent.select_action(state)
    print(f"Sample action: {action}, log_prob: {log_prob:.4f}, value: {value:.4f}")

    print("\n✅ Agent test complete!")
