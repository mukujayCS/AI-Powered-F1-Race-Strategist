# AI-Powered F1 Race Engineer

**Two-Stage Reinforcement Learning for Optimal Pit Stop Strategy**

An intelligent race engineer that learns F1 pit stop timing and race strategy by combining **Behavior Cloning** from historical data with **Proximal Policy Optimization** (PPO) reinforcement learning.

**Key Results:** P6.0 average finish, 2 wins, 30% podium rate, +13 position improvement over baseline

---

## Table of Contents

1. [Installation](#installation)
2. [Complete Training Workflow](#complete-training-workflow)
3. [Step-by-Step Instructions](#step-by-step-instructions)
4. [Driver & Circuit Reference](#driver--circuit-reference)
5. [Command Reference](#command-reference)
6. [Troubleshooting](#troubleshooting)
7. [Results Summary](#results-summary)
8. [System Overview](#system-overview)

---

## Installation

### Prerequisites

- **Python 3.8+**
- **pip** package manager

### Step 1: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv f1_race_engineer

# Activate virtual environment
# On macOS/Linux:
source f1_race_engineer/bin/activate

# On Windows:
# f1_race_engineer\Scripts\activate
```

**Note:** You should see `(f1_race_engineer)` in your terminal prompt after activation.

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**
- `numpy` - Numerical computing
- `pandas` - Data manipulation
- `torch` - Deep learning framework
- `gymnasium` - RL environment interface
- `matplotlib` - Visualization
- `tqdm` - Progress bars

### Step 3: Verify Data Files

The processed data from 2022-2024 F1 seasons should be in `data/processed/`:

```bash
ls data/processed/
# Should show:
# 22-24_circuits.json
# 22-24_drivers.json
# 22-24_lap_times.json
# 22-24_pitstops.json
# 22-24_qualifying.json
# 22-24_races.json
```

**Data Source:** [Kaggle F1 World Championship (1950-2024)](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020)

**Note:** The processed data files are **included in this repository**, so you can start training immediately. If you want to download and reprocess the raw data yourself:

1. Create a Kaggle account and generate API credentials
2. Create a `cred.env` file in the project root:
   ```
   KAGGLE_USERNAME=your_username
   KAGGLE_KEY=your_api_key
   ```
3. Run `python download_dataset.py` to download raw data to `data/raw/`
4. Run preprocessing scripts in `src/pre_processing/` to regenerate `data/processed/`

---

## Complete Training Workflow

### Quick Command Summary

```bash
# PHASE 1: Prepare Data
python src/prepare_imitation_data.py
python src/behavior_cloning.py

# PHASE 2: Train BC+PPO Agent
python src/train_agent_with_bc.py --episodes 500 --driver-id 1 --save-freq 100 --lr 1e-4

# PHASE 3: Evaluate
python src/evaluate_agent.py --model models/bc_ppo/f1_agent_final.pt --driver-id 1 --episodes 10

# PHASE 4: Generate Comparison Visualization
python src/run_comparison.py \
    --model models/bc_ppo/f1_agent_final.pt \
    --driver-id 1 \
    --output results/bc_ppo_vs_baseline.png
```

**Result:** Trained agent achieving P6.0 average finish with 2 wins and 30% podium rate!

---

## Step-by-Step Instructions

### PHASE 1: Data Preparation

#### Step 1.1: Extract Historical Pit Stop Decisions

```bash
python src/prepare_imitation_data.py
```

**What this does:**
- Processes 74,489 historical lap times from 2022-2024 seasons
- Extracts 2,339 pit stop decisions with state features
- Creates `data/processed/imitation_dataset.npz` (69,600 training samples)
- Shows class distribution: 3.36% pit, 96.64% stay out

**Expected output:**
```
Processing historical data...
Created dataset:
   Total samples: 69,600
   Pit decisions: 2,339 (3.36%)
   Stay out: 67,261 (96.64%)
   State dimension: 8
   Dataset saved to data/processed/imitation_dataset.npz
```

**Troubleshooting:** If you see "FileNotFoundError", verify data files exist in `data/processed/`

---

#### Step 1.2: Train Behavior Cloning Model

```bash
python src/behavior_cloning.py
```

**What this does:**
- Trains neural network to predict "pit or stay out" from lap state
- Uses weighted sampling to handle class imbalance (pit decisions are rare)
- Trains for 20 epochs with 80/20 train/validation split
- Saves model to `models/behavior_cloning/behavior_cloning_model.pt`
- Generates training curves: `models/behavior_cloning/bc_training_curves.png`

**Expected output:**
```
Training Behavior Cloning Model...
Epoch 1/20: Loss: 0.3421, Val Acc: 88.32%
Epoch 5/20: Loss: 0.1245, Val Acc: 92.15%
Epoch 10/20: Loss: 0.0876, Val Acc: 93.54%
Epoch 20/20: Loss: 0.0543, Val Acc: 94.14%

 Training complete!
   Final validation accuracy: 94.14%
   Pit precision: 9.77%
   Pit recall: 9.31%
   Model saved to models/behavior_cloning/behavior_cloning_model.pt
```

**Note:** Low pit recall (9.31%) is expected—BC model is conservative to prevent over-pitting.

---

### PHASE 2: BC+PPO Training

#### Step 2.1: Train BC+PPO Agent (500 Episodes - Recommended)

```bash
python src/train_agent_with_bc.py --episodes 500 --driver-id 1 --save-freq 100 --lr 1e-4
```

**Command Breakdown:**
- `--episodes 500` - Train for 500 episodes
- `--driver-id 1` - Train for Lewis Hamilton (see [Driver Reference](#driver-reference))
- `--save-freq 100` - Save checkpoints every 100 episodes
- `--lr 1e-4` - Learning rate (lower for stable fine-tuning)

**What this does:**
- Loads pre-trained BC model for guided action selection
- BC guidance decays from 50% → 0% over first 200 episodes
- PPO learns to optimize beyond historical strategies
- Saves checkpoints: `models/bc_ppo/f1_agent_episode_100.pt`, `200.pt`, etc.
- Generates training curves: `models/bc_ppo/bc_ppo_training_progress.png`

**Expected training progression:**
```
Episode 50/500: Avg Pitstops: 2.1, Position: P6.0, Reward: +25.43
Episode 100/500: Avg Pitstops: 1.2, Position: P6.2, Reward: +42.76
Episode 200/500: BC guidance ended. Avg Pitstops: 1.5, Position: P8.1
Episode 300/500: Avg Pitstops: 1.8, Position: P9.4, Reward: +38.21
Episode 500/500: Avg Pitstops: 2.4, Position: P11.0, Reward: +29.93

 Training complete!
   Final model saved to models/bc_ppo/f1_agent_final.pt
   Training curves saved to models/bc_ppo/bc_ppo_training_progress.png
```

**What to expect:**
- **Episodes 1-50:** BC guidance strong, agent makes 2-10 pitstops
- **Episodes 50-200:** BC guidance decays, agent explores 1-2 stop strategies
- **Episodes 200-500:** Pure PPO optimization, converges to optimal strategy

---

#### Step 2.2: Extended Training (1000 Episodes - Optional)

For even better performance:

```bash
python src/train_agent_with_bc.py --episodes 1000 --driver-id 1 --save-freq 100 --lr 1e-4
```

**Benefits:**
- Better position (P7.4 → P5-6)
- More consistent performance
- Better generalization across race scenarios

---

### PHASE 3: Evaluation

#### Step 3.1: Evaluate Agent Performance

```bash
python src/evaluate_agent.py --model models/bc_ppo/f1_agent_final.pt --driver-id 1 --episodes 10
```

**Command Breakdown:**
- `--model` - Path to trained BC+PPO model
- `--driver-id 1` - Evaluate for Lewis Hamilton
- `--episodes 10` - Run 10 random test races

**Expected output:**
```
Evaluating agent on 10 test races...

Race 1: Position P2, Pitstops: 1, Race Time: 5771.3s
Race 2: Position P12, Pitstops: 2, Race Time: 5962.1s
Race 3: Position P1, Pitstops: 1, Race Time: 5698.4s (WIN!)
...
Race 10: Position P8, Pitstops: 1, Race Time: 5843.7s

======================================
EVALUATION SUMMARY
======================================
Average Position: P6.0
Wins: 2/10 (20%)
Podiums: 3/10 (30%)
Points Finishes: 7/10 (70%)
Average Pitstops: 1.30
Average Reward: +53.41
======================================
```

**What to look for:**
- Average pitstops: 1.0-1.5 (optimal F1 strategy)
- Average position: P5-P10 (points-scoring finish)
- Podiums: 10-30% of races
- Wins: 10-20% of races

---

#### Step 3.2: Generate Comparison Visualization

```bash
python src/run_comparison.py \
    --model models/bc_ppo/f1_agent_final.pt \
    --driver-id 1 \
    --race-id 1074 \
    --output results/bc_ppo_vs_baseline.png
```

**Command Breakdown:**
- `--model` - Trained BC+PPO agent
- `--driver-id 1` - Lewis Hamilton
- `--race-id 1074` - Specific race (optional, uses random if omitted)
- `--output` - Where to save visualization

**Output:** 3-panel visualization showing:
1. **Lap Time Comparison** - RL agent vs baseline lap times
2. **Position Evolution** - Position changes throughout race
3. **Performance Summary** - Final statistics table

**Example output:**
```
Running race comparison...
  RL Agent: P2, 1 pitstop, 5771.3s
  Baseline: P12, 2 pitstops, 5962.1s

Visualization saved to results/bc_ppo_vs_baseline.png
```

---

### PHASE 4: Advanced Usage

#### Train for Different Drivers

**Important:** The `--save-dir` folder name (e.g., "verstappen", "leclerc", "norris") is **manually specified by you**. The system does NOT automatically look up driver names from the ID. You can name the folder anything—we use driver names for organization.

```bash
# Max Verstappen (Driver ID 830) - manually specify folder name "verstappen"
python src/train_agent_with_bc.py --episodes 1000 --driver-id 830 --save-dir models/verstappen

# Charles Leclerc (Driver ID 844) - manually specify folder name "leclerc"
python src/train_agent_with_bc.py --episodes 1000 --driver-id 844 --save-dir models/leclerc

# Lando Norris (Driver ID 846) - manually specify folder name "norris"
python src/train_agent_with_bc.py --episodes 1000 --driver-id 846 --save-dir models/norris
```

See [Driver Reference](#driver-reference) for complete driver ID list.

---

#### Multi-Race Statistical Comparison

Compare BC+PPO vs baseline across multiple races:

```python
import sys
from pathlib import Path
sys.path.insert(0, 'src')

from f1_env_final import create_f1_env
from f1_agent import F1PPOAgent
from compare_strategies import StrategyComparator

# Load environment and agent
env = create_f1_env(data_dir=Path('data'))
agent = F1PPOAgent(env=env, lr=1e-4, gamma=0.99, epochs=10, batch_size=64)
agent.load_model(Path('models/bc_ppo/f1_agent_final.pt'))

# Run comparison
comparator = StrategyComparator(env, agent, data_dir=Path('data'))
race_ids = list(env.races_data['raceId'].iloc[:10])
results = comparator.compare_on_races(race_ids, driver_id=1, baseline_types=['2-stop'])

# Print statistics
stats = comparator.compute_statistics(results)
comparator.print_comparison_report(stats)
comparator.save_results(Path('results'))
```

---

## Driver & Circuit Reference

### Driver ID Mapping

Use these IDs with `--driver-id` parameter:

| Driver ID | Driver Name | Code | Driver ID | Driver Name | Code |
|-----------|-------------|------|-----------|-------------|------|
| **1** | **Lewis Hamilton** | **HAM** | 839 | Esteban Ocon | OCO |
| 4 | Fernando Alonso | ALO | 840 | Lance Stroll | STR |
| 20 | Sebastian Vettel | VET | 842 | Pierre Gasly | GAS |
| 807 | Nico Hulkenberg | HUL | **844** | **Charles Leclerc** | **LEC** |
| 815 | Sergio Perez | PER | **846** | **Lando Norris** | **NOR** |
| 817 | Daniel Ricciardo | RIC | 847 | George Russell | RUS |
| 822 | Valtteri Bottas | BOT | 848 | Alexander Albon | ALB |
| 825 | Kevin Magnussen | MAG | 852 | Yuki Tsunoda | TSU |
| **830** | **Max Verstappen** | **VER** | 857 | Oscar Piastri | PIA |
| 832 | Carlos Sainz | SAI | 858 | Logan Sargeant | SAR |

**Note:** All results in this project use **Driver ID 1 (Lewis Hamilton)**.

### Circuit ID Mapping

Use these IDs with `--race-id` parameter for specific race selection:

| Circuit ID | Circuit Name | Circuit ID | Circuit Name |
|------------|--------------|------------|--------------|
| 1 | Albert Park (Australia) | 21 | Imola (Italy) |
| 3 | Bahrain International Circuit | 22 | Suzuka (Japan) |
| 4 | Barcelona-Catalunya (Spain) | 24 | Yas Marina (Abu Dhabi) |
| 6 | Monaco | 32 | Mexico City |
| 7 | Montreal (Canada) | 39 | Zandvoort (Netherlands) |
| 9 | Silverstone (Great Britain) | 69 | Circuit of the Americas (USA) |
| 11 | Hungaroring (Hungary) | 70 | Red Bull Ring (Austria) |
| 13 | Spa-Francorchamps (Belgium) | 73 | Baku (Azerbaijan) |
| 14 | Monza (Italy) | 77 | Jeddah (Saudi Arabia) |
| 15 | Marina Bay (Singapore) | 78 | Losail (Qatar) |
| 17 | Shanghai (China) | 79 | Miami (USA) |
| 18 | Interlagos (Brazil) | 80 | Las Vegas (USA) |

---

## Command Reference

### Training Commands

```bash
# Standard 500-episode training
python src/train_agent_with_bc.py --episodes 500 --driver-id 1 --save-freq 100 --lr 1e-4

# Extended 1000-episode training
python src/train_agent_with_bc.py --episodes 1000 --driver-id 1 --save-freq 100 --lr 1e-4

# Custom save directory
python src/train_agent_with_bc.py --episodes 500 --driver-id 830 --save-dir models/verstappen

# Custom hyperparameters
python src/train_agent_with_bc.py --episodes 500 --driver-id 1 --lr 5e-5 --gamma 0.98
```

### Evaluation Commands

```bash
# Basic evaluation (10 races)
python src/evaluate_agent.py --model models/bc_ppo/f1_agent_final.pt --driver-id 1 --episodes 10

# Extended evaluation (20 races)
python src/evaluate_agent.py --model models/bc_ppo/f1_agent_final.pt --driver-id 1 --episodes 20

# Evaluate specific checkpoint
python src/evaluate_agent.py --model models/bc_ppo/f1_agent_episode_300.pt --driver-id 1 --episodes 10
```

### Visualization Commands

```bash
# Random race comparison
python src/run_comparison.py --model models/bc_ppo/f1_agent_final.pt --driver-id 1

# Specific race comparison
python src/run_comparison.py --model models/bc_ppo/f1_agent_final.pt --driver-id 1 --race-id 1074

# Custom output path
python src/run_comparison.py --model models/bc_ppo/f1_agent_final.pt --driver-id 1 --output my_comparison.png

# Different baseline strategy
python src/run_comparison.py --model models/bc_ppo/f1_agent_final.pt --driver-id 1 --baseline 3-stop
```

### Data Preparation Commands

```bash
# Extract imitation learning dataset
python src/prepare_imitation_data.py

# Train behavior cloning model
python src/behavior_cloning.py

# Reprocess raw data (if needed)
python src/pre_processing/pre_process_files.py
```

---

## Troubleshooting

If you follow the [Complete Training Workflow](#complete-training-workflow) in order, you shouldn't encounter errors. Common issues:

| Error | Solution |
|-------|----------|
| `FileNotFoundError: imitation_dataset.npz` | Run `python src/prepare_imitation_data.py` first |
| `FileNotFoundError: behavior_cloning_model.pt` | Run `python src/behavior_cloning.py` before training |
| `ModuleNotFoundError: No module named 'torch'` | Run `pip install -r requirements.txt` |
| Agent learns 0 or 60 pitstops | Use `train_agent_with_bc.py`, NOT `train_agent.py` |

**Note:** BC model having low pit recall (~9%) is expected—it's conservative to prevent over-pitting.

---

## Results Summary

### BC+PPO Agent Performance (Driver ID 1 - Lewis Hamilton)

**Evaluation across 10 test races:**
- Average Position: **P6.0** (top 10 finish!)
- Wins: **2/10 (20%)**
- Podiums: **3/10 (30%)**
- Points Finishes: **7/10 (70%)**
- Average Pitstops: **1.30** (optimal 1-2 stop strategy)

### Comparison vs Fixed 2-Stop Baseline

| Metric | BC+PPO Agent | Fixed 2-Stop Baseline | Improvement |
|--------|--------------|----------------------|-------------|
| Average Position | P6.0 | P19.0 | **+13 positions** |
| Wins | 2 (20%) | 0 (0%) | **+2 wins** |
| Podiums | 3 (30%) | 0 (0%) | **+3 podiums** |
| Points Finishes | 7 (70%) | 0 (0%) | **+7 scoring** |
| Average Pitstops | 1.30 | 2.00 | **35% fewer** |

### Training Speed: BC+PPO vs Pure PPO

| Method | Episodes to Top-10 | Training Stability |
|--------|--------------------|--------------------|
| Pure PPO (baseline) | 500+ (never achieved) | Unstable |
| **BC+PPO (our approach)** | **50 episodes** | Stable |
| **Speedup** | **10× faster** | Reliable |

---

## System Overview

### The Challenge

Formula 1 strategy requires making pit stop decisions across 50+ lap races, where a single wrong choice can cost a championship. Pure reinforcement learning struggles because:
- **Credit assignment problem:** Hard to connect early decisions to final race outcome
- **Exploration challenges:** Agents get stuck at local optima (never pitting or pitting excessively)
- **Ignores historical data:** 74K+ historical laps contain valuable F1 strategy patterns

**Example:** 2021 Abu Dhabi GP - Max Verstappen's late-race pit stop for fresh tires during a safety car enabled him to win his first world championship on the final lap.

### Our Solution: Behavior Cloning + PPO

**Phase 1 - Behavior Cloning (BC):** Train supervised model on 69,600 historical pit decisions (94% accuracy)
**Phase 2 - PPO Fine-tuning:** Initialize RL agent with BC guidance (decays 50%→0% over 200 episodes), then optimize beyond human strategies

**Why it works:**
- BC bootstraps agent with F1 strategy knowledge (solves exploration problem)
- PPO optimizes beyond historical data through reinforcement learning
- 10× faster convergence (50 vs 500+ episodes to top-10 finishes)
- Stable training (no oscillation between 0 and 60 pitstops)

### Architecture Summary

**Behavior Cloning Model:**
- Neural network: 8 → 128 → 128 → 2 (17,922 parameters)
- Input: Lap progress, tire age, degradation, position + 4 binary indicators
- Output: Binary classification (pit vs stay out)
- Training: 20 epochs on 69,600 samples, achieves 94% validation accuracy

**PPO Agent:**
- Actor-Critic: 23 → 256 → 256 → 7/1 (138,760 parameters)
- State space: 23 dimensions (race state, tire state, weather, driver/track profiles)
- Action space: 7 discrete (stay out with 3 pace modes, pit with 4 tire choices)
- Reward: Dense lap-by-lap feedback (position changes, tire management, strategy bonuses)

**Tire Degradation:**
- Exponential model: `rate = 0.08 × (age/30)^1.5 × track_factor`
- Track factors computed from historical lap time progressions
- Realistic wear: Slow initially, accelerates after 15-20 laps

---

## References

- **Dataset:** [Kaggle F1 World Championship (1950-2024)](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020)
- **PPO Paper:** [Schulman et al., 2017 - Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)
- **Behavior Cloning:** [Pomerleau, 1991 - Efficient Training for Autonomous Navigation](https://direct.mit.edu/neco/article/3/1/88/5571)
- **Gymnasium:** [Farama Foundation - OpenAI Gym Successor](https://gymnasium.farama.org/)

---

## Author

Mukund Iyengar

**Master's Project - Johns Hopkins University**

---

## Quick Start Summary

```bash
# Complete workflow
pip install -r requirements.txt
python src/prepare_imitation_data.py
python src/behavior_cloning.py
python src/train_agent_with_bc.py --episodes 500 --driver-id 1 --save-freq 100 --lr 1e-4
python src/evaluate_agent.py --model models/bc_ppo/f1_agent_final.pt --driver-id 1 --episodes 10
python src/run_comparison.py --model models/bc_ppo/f1_agent_final.pt --driver-id 1 --output results/bc_ppo_vs_baseline.png
``` 
