"""
Behavior Cloning for F1 Pit Stop Strategy

Trains a supervised model to imitate historical pit stop decisions.
This model is then used to initialize the PPO agent for faster learning.

Architecture:
- Input: 8 state features (lap progress, tire age, degradation, position, etc.)
- Hidden: 2 layers of 128 units each
- Output: Binary classification (pit vs stay out)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt


class PitStopDataset(Dataset):
    """Dataset for pit stop decisions"""

    def __init__(self, states, actions, weights):
        self.states = torch.FloatTensor(states)
        self.actions = torch.LongTensor(actions)
        self.weights = torch.FloatTensor(weights)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx], self.weights[idx]


class BehaviorCloningModel(nn.Module):
    """
    Neural network for behavior cloning.

    This architecture matches the PPO policy network so we can
    transfer weights easily.
    """

    def __init__(self, state_dim=8, hidden_dim=128):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # Binary: pit or stay out
        )

    def forward(self, state):
        return self.network(state)


def train_behavior_cloning(
    dataset_path: Path,
    save_dir: Path,
    epochs: int = 20,
    batch_size: int = 512,
    lr: float = 3e-4,
    val_split: float = 0.2,
    device: str = "cpu"
):
    """
    Train behavior cloning model on historical pit stop data.

    Args:
        dataset_path: Path to imitation_dataset.npz
        save_dir: Directory to save trained model
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        val_split: Validation split fraction
        device: "cpu" or "cuda"
    """
    print("="*60)
    print("BEHAVIOR CLONING TRAINING")
    print("="*60)

    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    data = np.load(dataset_path)
    states = data['states']
    actions = data['actions']
    weights = data['weights']

    print(f"Dataset size: {len(states):,} samples")
    print(f"State dimension: {states.shape[1]}")
    print(f"Pit decisions: {actions.sum():,} ({100*actions.mean():.2f}%)")

    # Train/val split
    n_samples = len(states)
    n_val = int(n_samples * val_split)
    n_train = n_samples - n_val

    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    train_states = states[train_idx]
    train_actions = actions[train_idx]
    train_weights = weights[train_idx]

    val_states = states[val_idx]
    val_actions = actions[val_idx]

    print(f"\nTrain samples: {len(train_states):,}")
    print(f"Val samples: {len(val_states):,}")

    # Create datasets
    train_dataset = PitStopDataset(train_states, train_actions, train_weights)
    val_dataset = PitStopDataset(val_states, val_actions, np.ones(len(val_states)))

    # Create samplers for handling class imbalance
    # Use weighted sampling to balance pit vs stay-out decisions
    sampler = WeightedRandomSampler(
        weights=train_weights,
        num_samples=len(train_weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    # Create model
    model = BehaviorCloningModel(state_dim=states.shape[1], hidden_dim=128)
    model = model.to(device)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_pit_precision': [],
        'val_pit_recall': []
    }

    print("\n" + "="*60)
    print("TRAINING START")
    print("="*60)

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_states, batch_actions, batch_weights in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{epochs}"
        ):
            batch_states = batch_states.to(device)
            batch_actions = batch_actions.to(device)

            # Forward pass
            outputs = model(batch_states)
            loss = criterion(outputs, batch_actions)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_actions.size(0)
            train_correct += (predicted == batch_actions).sum().item()

        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_tp = 0  # True positives (correctly predicted pits)
        val_fp = 0  # False positives (predicted pit, actually stayed out)
        val_fn = 0  # False negatives (predicted stay out, actually pitted)

        with torch.no_grad():
            for batch_states, batch_actions, _ in val_loader:
                batch_states = batch_states.to(device)
                batch_actions = batch_actions.to(device)

                outputs = model(batch_states)
                loss = criterion(outputs, batch_actions)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_actions.size(0)
                val_correct += (predicted == batch_actions).sum().item()

                # Compute pit decision metrics
                val_tp += ((predicted == 1) & (batch_actions == 1)).sum().item()
                val_fp += ((predicted == 1) & (batch_actions == 0)).sum().item()
                val_fn += ((predicted == 0) & (batch_actions == 1)).sum().item()

        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total

        # Precision and recall for pit decisions
        val_pit_precision = 100 * val_tp / (val_tp + val_fp) if (val_tp + val_fp) > 0 else 0
        val_pit_recall = 100 * val_tp / (val_tp + val_fn) if (val_tp + val_fn) > 0 else 0

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_pit_precision'].append(val_pit_precision)
        history['val_pit_recall'].append(val_pit_recall)

        # Print progress
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"  Val Pit Precision: {val_pit_precision:.2f}% | Recall: {val_pit_recall:.2f}%")
        print()

    # Save model
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_dir / "behavior_cloning_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'state_dim': states.shape[1],
        'hidden_dim': 128,
        'history': history
    }, model_path)

    print("="*60)
    print(f"✅ Model saved to {model_path}")

    # Plot training curves
    plot_training_curves(history, save_dir / "bc_training_curves.png")

    return model, history


def plot_training_curves(history, save_path):
    """Plot training and validation curves"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Val')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training & Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Accuracy
    axes[0, 1].plot(history['train_acc'], label='Train')
    axes[0, 1].plot(history['val_acc'], label='Val')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Training & Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Pit decision precision
    axes[1, 0].plot(history['val_pit_precision'])
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision (%)')
    axes[1, 0].set_title('Pit Decision Precision')
    axes[1, 0].grid(True)

    # Pit decision recall
    axes[1, 1].plot(history['val_pit_recall'])
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall (%)')
    axes[1, 1].set_title('Pit Decision Recall')
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Training curves saved to {save_path}")
    plt.close()


def main():
    dataset_path = Path("data/processed/imitation_dataset.npz")
    save_dir = Path("models/behavior_cloning")

    print("Training behavior cloning model on historical pit stop data...")
    model, history = train_behavior_cloning(
        dataset_path=dataset_path,
        save_dir=save_dir,
        epochs=20,
        batch_size=512,
        lr=3e-4
    )

    print("\n" + "="*60)
    print("BEHAVIOR CLONING TRAINING COMPLETE!")
    print("="*60)
    print(f"Final validation accuracy: {history['val_acc'][-1]:.2f}%")
    print(f"Final pit precision: {history['val_pit_precision'][-1]:.2f}%")
    print(f"Final pit recall: {history['val_pit_recall'][-1]:.2f}%")
    print("="*60)


if __name__ == "__main__":
    main()
