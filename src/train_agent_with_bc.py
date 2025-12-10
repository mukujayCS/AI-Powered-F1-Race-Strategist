"""
Two-Stage Training: Behavior Cloning + PPO Fine-tuning

Phase 1: Train BC model on historical pit stop data (done separately)
Phase 2: Initialize PPO agent with BC weights, then fine-tune with RL

This approach addresses PPO's credit assignment problem by starting from
a policy that already knows "when F1 drivers typically pit", then optimizing
beyond historical strategies using RL.

Usage:
    python train_agent_with_bc.py --episodes 1000 --driver-id 1
"""

import argparse
import torch
from torch.distributions import Categorical
from pathlib import Path
from f1_env_final import create_f1_env
from f1_agent import F1PPOAgent
import numpy as np


def load_bc_weights_into_ppo(ppo_agent, bc_model_path):
    """
    Load behavior cloning weights into PPO policy network.

    The BC model has structure:
        - Linear(8, 128) -> ReLU -> Linear(128, 128) -> ReLU -> Linear(128, 2)

    The PPO policy network (ActorCritic) has:
        - Shared layers: Linear(23, 256) -> ReLU -> Linear(256, 256) -> ReLU
        - Actor head: Linear(256, 7)
        - Critic head: Linear(256, 1)

    Strategy:
    - The BC model was trained on simplified 8-feature state
    - PPO uses full 23-feature state from environment
    - We can't directly transfer weights due to dimension mismatch
    - Instead: Use BC model to guide initial policy by training PPO for a few
      episodes with BC model as "expert" demonstration

    Alternative simpler approach:
    - Just use BC model to initialize the policy bias toward reasonable pit timing
    - Train PPO normally but with BC-guided exploration
    """
    print("="*60)
    print("LOADING BEHAVIOR CLONING WEIGHTS")
    print("="*60)

    # Load BC model
    bc_checkpoint = torch.load(bc_model_path, map_location='cpu')
    bc_state_dict = bc_checkpoint['model_state_dict']

    print(f"✅ Loaded BC model from {bc_model_path}")
    print(f"   BC state dim: {bc_checkpoint['state_dim']}")
    print(f"   BC hidden dim: {bc_checkpoint['hidden_dim']}")

    # Note: Due to different state dimensions (BC: 8, PPO: 23), we cannot
    # directly transfer weights. Instead, we'll use the BC model to guide
    # early training by injecting BC predictions into the PPO loss.

    # Store BC model in agent for guided training
    from behavior_cloning import BehaviorCloningModel
    bc_model = BehaviorCloningModel(
        state_dim=bc_checkpoint['state_dim'],
        hidden_dim=bc_checkpoint['hidden_dim']
    )
    bc_model.load_state_dict(bc_state_dict)
    bc_model.eval()

    ppo_agent.bc_model = bc_model
    ppo_agent.use_bc_guidance = True

    print("✅ BC model loaded for guided training")
    print("   PPO will use BC predictions to guide early exploration")
    print("="*60)

    return ppo_agent


def extract_bc_features(state):
    """
    Extract BC-compatible features from full environment state.

    Environment state (23 dims):
        [0]: current_lap / total_laps
        [1]: tire_age / 40
        [2]: tire_degradation
        [3]: position / 20
        [4:7]: weather one-hot
        [7:10]: tire compound one-hot
        [10:]: driver/track features

    BC features (8 dims):
        [0]: lap_progress
        [1]: tire_age_normalized
        [2]: tire_degradation
        [3]: position_normalized
        [4]: fresh_tires_indicator
        [5]: old_tires_indicator
        [6]: early_race
        [7]: late_race
    """
    lap_progress = state[0]
    tire_age_norm = state[1]
    tire_degradation = state[2]
    position_norm = state[3]

    bc_features = np.array([
        lap_progress,
        tire_age_norm,
        tire_degradation,
        position_norm,
        1.0 if tire_age_norm < 0.125 else 0.0,  # Fresh tires (< 5 laps)
        1.0 if tire_degradation > 0.7 else 0.0,  # Old tires
        1.0 if lap_progress < 0.3 else 0.0,      # Early race
        1.0 if lap_progress > 0.7 else 0.0,      # Late race
    ], dtype=np.float32)

    return bc_features


def main():
    parser = argparse.ArgumentParser(description="Train F1 Agent with BC + PPO")
    parser.add_argument(
        "--episodes",
        type=int,
        default=500,
        help="Number of PPO fine-tuning episodes"
    )
    parser.add_argument(
        "--driver-id",
        type=int,
        default=None,
        help="Specific driver ID to train for"
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=100,
        help="Save model checkpoint every N episodes"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="models/bc_ppo",
        help="Directory to save trained models"
    )
    parser.add_argument(
        "--bc-model",
        type=str,
        default="models/behavior_cloning/behavior_cloning_model.pt",
        help="Path to pre-trained BC model"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,  # Lower LR for fine-tuning
        help="Learning rate for PPO fine-tuning"
    )

    args = parser.parse_args()

    print("="*60)
    print("TWO-STAGE TRAINING: BEHAVIOR CLONING + PPO")
    print("="*60)
    print(f"Phase 1: Behavior Cloning (pre-trained)")
    print(f"Phase 2: PPO Fine-tuning ({args.episodes} episodes)")
    print(f"Driver ID: {args.driver_id if args.driver_id else 'All drivers'}")
    print(f"Learning Rate: {args.lr} (lower for fine-tuning)")
    print(f"Save Directory: {args.save_dir}")
    print("="*60)

    # Check if BC model exists
    bc_model_path = Path(args.bc_model)
    if not bc_model_path.exists():
        print(f"\n❌ ERROR: BC model not found at {bc_model_path}")
        print("Please run behavior_cloning.py first to train BC model:")
        print("  python src/behavior_cloning.py")
        return

    # Create environment (no curriculum - use dense rewards)
    print("\nInitializing environment with DENSE REWARDS...")
    env = create_f1_env(render_mode=None, curriculum=None)

    # Create PPO agent
    print("Creating PPO agent...")
    agent = F1PPOAgent(
        env=env,
        lr=args.lr,  # Lower LR for fine-tuning
        gamma=0.99,
        epochs=10,
        batch_size=64,
        c1=0.5,   # Value loss coefficient
        c2=0.01   # Entropy coefficient (encourage some exploration)
    )

    print(f"Agent has {sum(p.numel() for p in agent.policy.parameters())} parameters")

    # Load BC weights
    print("\nLoading BC model for guided training...")
    agent = load_bc_weights_into_ppo(agent, bc_model_path)

    # Add BC-guided action selection to agent
    def bc_guided_select_action(state):
        """
        Use BC model to guide action selection during early training.

        Strategy:
        - Extract BC-compatible features from state
        - Get BC prediction for pit probability
        - Blend BC prediction with PPO policy (higher weight early, lower later)

        Returns: (action, log_prob, value) - same as original select_action
        """
        # Get PPO action, log_prob, and value using the policy's act() method
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action, log_prob, value = agent.policy.act(state_tensor)

        # Get BC prediction
        bc_features = extract_bc_features(state)
        bc_features_tensor = torch.FloatTensor(bc_features).unsqueeze(0)

        with torch.no_grad():
            bc_logits = agent.bc_model(bc_features_tensor)
            bc_probs = torch.softmax(bc_logits, dim=-1)
            bc_pit_prob = bc_probs[0, 1].item()  # Probability of pitting

        # Blend BC guidance with PPO policy
        # During early episodes, follow BC more. Later, trust PPO more.
        episode_ratio = min(1.0, agent.episode_count / 200.0)  # 0->1 over 200 episodes
        bc_weight = 0.5 * (1 - episode_ratio)  # Start at 0.5, decay to 0

        # If BC strongly suggests pitting (>50%), bias toward pit actions (3-6)
        if bc_pit_prob > 0.5 and bc_weight > 0.1:
            # If current action is "stay out" (0-2) but BC suggests pit,
            # override with a random pit action
            if action < 3 and np.random.random() < bc_weight:
                action = np.random.choice([3, 4, 5, 6])  # Random pit action

                # CRITICAL FIX: Recompute log_prob for the overridden action
                # Otherwise PPO trains on wrong gradients (it would train thinking
                # we took the "stay out" action when we actually pitted)
                with torch.no_grad():
                    action_probs, _ = agent.policy(state_tensor)
                    dist = Categorical(action_probs)
                    log_prob = dist.log_prob(torch.tensor(action))

        # Return same format as original select_action: (action, log_prob, value)
        return action, log_prob.item(), value.item()

    # Replace agent's select_action with BC-guided version
    agent.episode_count = 0
    agent.original_select_action = agent.select_action

    def wrapped_select_action(state):
        if hasattr(agent, 'use_bc_guidance') and agent.use_bc_guidance:
            return bc_guided_select_action(state)
        else:
            return agent.original_select_action(state)

    agent.select_action = wrapped_select_action

    # Train with PPO fine-tuning
    print("\n" + "="*60)
    print("STARTING PPO FINE-TUNING")
    print("="*60)
    print("BC guidance will gradually decrease over first 200 episodes")
    print("="*60 + "\n")

    try:
        agent.train(
            num_episodes=args.episodes,
            driver_id=args.driver_id,
            save_dir=Path(args.save_dir),
            save_freq=args.save_freq,
            verbose=True
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")

    # Plot training progress
    print("\nGenerating training plots...")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    agent.plot_training_progress(save_dir / "bc_ppo_training_progress.png")

    print("\n" + "="*60)
    print("TWO-STAGE TRAINING COMPLETE!")
    print("="*60)
    print(f"Final model saved to: {save_dir}/f1_agent_final.pt")
    print(f"Training plot saved to: {save_dir}/bc_ppo_training_progress.png")
    print("="*60)


if __name__ == "__main__":
    main()
