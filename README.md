# DPPO for Real-Time PID Parameter Tuning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Project Goal

To use a Deep Reinforcement Learning (DRL) agent, specifically based on **Proximal Policy Optimization (PPO)**, to learn an optimal policy for adjusting the three PID gains (K_p, K_i, K_d) in real-time, such that a simulated system (the "Plant") accurately and robustly tracks a target trajectory or setpoint.

The core idea is to train a **Meta-Controller** (The DPPO Agent) that manipulates the classical **Inner-Controller** (The PID Loop).

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run demo to test environment
python demo.py

# Train the agent
python train.py

# Evaluate trained model
python evaluate.py --model models/dppo_pid_final_*.zip
```
## Table of Contents

- [Project Structure](#project-structure)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [System Architecture](#system-architecture)
- [Usage](#usage)
- [Configuration](#configuration)
- [Implementation Details](#implementation-details)
- [Results and Visualization](#results-and-visualization)
- [Advanced Topics](#advanced-topics)

## Project Structure

```
DPPO_PID_controller/
├── dppo_pid_env.py          # Custom Gymnasium environment
├── train.py                 # Training script with PPO
├── evaluate.py              # Evaluation and visualization
├── demo.py                  # Demo script for testing
├── config.yaml              # Hyperparameter configuration
├── PPO_HYPERPARAMETERS.md   # Detailed hyperparameter guide (中英文)
├── requirements.txt         # Python dependencies
├── .gitignore               # Git ignore file
└── README.md                # This file
```

## Technology Stack

| Component | Technology | Role |
|-----------|-----------|------|
| RL Framework | Stable-Baselines3 (PPO) | Implements the DRL algorithm and training loop |
| Simulation Environment | Farama Gymnasium (custom) | Defines state, action, reward, and transition dynamics |
| System Dynamics (Plant) | NumPy | High-speed numerical integration for physics model |
| Logging/Visualization | TensorBoard & Matplotlib | Tracks learning progress and performance metrics |

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd DPPO_PID_controller

# Install dependencies
pip install -r requirements.txt
```

## System Architecture

The system implements a **dual-loop control architecture**:

### Two-Timescale Control Loop

```
┌──────────────────────────────────────────────────────┐
│                   Meta-Controller                     │
│              (PPO/DPPO RL Agent @ 20 Hz)             │
│                                                       │
│  Inputs: [error, error_dot, integral, x, x_dot,     │
│           reference, Kp, Ki, Kd]                     │
│  Outputs: [Kp_new, Ki_new, Kd_new]                  │
└─────────────────┬────────────────────────────────────┘
                  │ Updates every 0.05s
                  ↓
┌──────────────────────────────────────────────────────┐
│               Inner PID Controller                    │
│                    (@ 200 Hz)                        │
│                                                       │
│  u(t) = Kp·e(t) + Ki·∫e(t)dt + Kd·de(t)/dt         │
└─────────────────┬────────────────────────────────────┘
                  │ Control signal u(t)
                  ↓
┌──────────────────────────────────────────────────────┐
│            2nd-Order Plant System                     │
│                                                       │
│          J·ẍ + B·ẋ = u(t) + d(t)                    │
│                                                       │
│  J: Inertia (1.0)                                    │
│  B: Damping (0.5)                                    │
│  d: External disturbance                             │
└──────────────────────────────────────────────────────┘
```

### Timing Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Inner Loop Δt | 0.005s (200 Hz) | PID calculation and physics integration |
| Outer Loop Δt | 0.05s (20 Hz) | RL agent updates PID gains |
| Steps per RL Action | 10 | Inner loop steps per outer loop step |

### Plant Dynamics

The environment simulates a **2nd-order linear system**:

```
J·ẍ + B·ẋ = u(t) + d(t)
```

Where:
- **x**: Position/angle of the system
- **u(t)**: Control output from PID controller
- **d(t)**: External disturbance (random, time-limited)
- **J**: Inertia coefficient (default: 1.0)
- **B**: Damping coefficient (default: 0.5)

### PID Controller

Standard parallel-form PID:

```
u(t) = Kp·e(t) + Ki·∫e(t)dt + Kd·(e(t) - e(t-1))/Δt
```

Where:
- **e(t) = r(t) - x(t)**: Tracking error
- **Kp, Ki, Kd**: Gains adjusted by the RL agent
- **r(t)**: Reference setpoint (changes periodically)
## Usage

### 1. Test the Environment (Demo)

Run the demo script to verify the environment works correctly:

```bash
python demo.py
```

This will:
- Test the environment API
- Run an episode with random PID gains
- Generate a visualization (`demo_results.png`)

### 2. Train the Agent

Start training with default configuration:

```bash
python train.py
```

With custom configuration:

```bash
python train.py --config my_config.yaml
```

Resume training from a checkpoint:

```bash
python train.py --resume --model models/dppo_pid_checkpoint_1000000_steps.zip
```

Monitor training progress with TensorBoard:

```bash
tensorboard --logdir ./ppo_pid_logs/
```

### 3. Evaluate Trained Model

Evaluate and visualize performance:

```bash
python evaluate.py --model models/dppo_pid_final_TIMESTAMP.zip --episodes 10
```

This generates:
- Performance plots for best/worst episodes
- Summary statistics across all episodes
- Saved in `./evaluation_results/`

## Configuration

All hyperparameters are defined in `config.yaml`. Key sections:

### Plant Parameters

```yaml
plant:
  J: 1.0      # Inertia
  B: 0.5      # Damping
```

### Reward Weights

```yaml
reward:
  lambda_error: 5.0       # Tracking error weight
  lambda_velocity: 0.5    # Velocity penalty weight
  lambda_control: 0.01    # Control effort weight
  lambda_overshoot: 0.2   # Overshoot penalty weight
```

### PPO Training Parameters

```yaml
training:
  total_timesteps: 5000000
  learning_rate: 0.0003
  n_steps: 2048
  batch_size: 64
  gamma: 0.99
```

See `config.yaml` for all available parameters.

**📖 For detailed explanations of PPO hyperparameters, see [PPO_HYPERPARAMETERS.md](PPO_HYPERPARAMETERS.md)**

The current configuration uses **recommended settings** for a 9-dimensional state space:
- **Policy Network**: 2 layers × 128 units (suitable for 9D state)
- **Learning Rate**: 3×10⁻⁴ (standard PPO starting value)
- **Batch Size**: 64 (< n_steps for mini-batch training)
- **VecNormalize**: Enabled (critical for training stability)

## Implementation Details

### Markov Decision Process (MDP)

#### Action Space (𝐀)

**Dimensions**: 3 (continuous)
**Range**: [0.0, K_max] (default: [0, 10])
**Components**: [K_p, K_i, K_d]

The agent directly outputs new PID gain values, bounded to prevent instability.

#### Observation Space (𝐒)

**Dimensions**: 9 (continuous, normalized to [-1, 1])
**Components**:

1. Current position error (e)
2. Error derivative (ė)
3. Accumulated error (integral term)
4. System position (x)
5. System velocity (ẋ)
6. Target reference (r)
7. Current Kp
8. Current Ki
9. Current Kd

Including current gains enables the agent to learn **relative adjustments**.

#### Reward Function (𝐑)

Multi-objective reward combining four components:

```python
R = -λ₁·e² - λ₂·ẋ² - λ₃·u² - λ₄·max(0, -e·ẋ)
```

| Component | Weight (λ) | Purpose |
|-----------|------------|---------|
| Tracking Error | λ₁ = 5.0 | Minimize deviation from setpoint |
| Velocity Penalty | λ₂ = 0.5 | Reduce oscillations |
| Control Effort | λ₃ = 0.01 | Energy efficiency |
| Overshoot | λ₄ = 0.2 | Penalize moving away from setpoint |

#### Episode Termination

An episode ends when:
- Position exceeds safety bounds (|x| > 10.0) → system instability
- Maximum steps reached (1000 steps = 50 seconds)

### Reference Signal

To encourage adaptation, the setpoint changes periodically:

- Changes every 2 seconds
- Random value in [-2, 2]
- Forces agent to re-tune gains for different operating conditions

### Disturbances

Random external disturbances test robustness:

- Magnitude: ±0.2
- Duration: 0.1 seconds
- Occurs randomly with low probability
## Results and Visualization

### Training Metrics (TensorBoard)

Monitor during training:
- Episode reward (cumulative)
- Episode length
- Policy loss / Value loss
- Entropy (exploration)

### Evaluation Plots

The evaluation script generates:

1. **Position Tracking**: Shows system response vs. reference signal
2. **Tracking Error**: Error over time
3. **Control Input**: PID output force/torque
4. **PID Gains Evolution**: How Kp, Ki, Kd change in real-time
5. **Summary Statistics**: Rewards, errors, and gain distributions

## Advanced Topics

### Curriculum Learning

For faster training, consider implementing curriculum learning:

1. **Phase 1**: Simple step references, no disturbances
2. **Phase 2**: Introduce reference changes
3. **Phase 3**: Add external disturbances
4. **Phase 4**: Increase disturbance magnitude

Modify `config.yaml` between phases or implement automatic curriculum in the environment.

### Hyperparameter Tuning

Key parameters to tune:

**Reward Weights**: Balance between tracking, stability, and efficiency
- Increase `lambda_error` if tracking is poor
- Increase `lambda_overshoot` if oscillations occur
- Adjust `lambda_velocity` for smoother control

**PPO Parameters**:
- `learning_rate`: Lower (1e-4) for stability, higher (5e-4) for faster learning
- `n_steps`: More steps = more data per update (but slower)
- `batch_size`: Larger batches = more stable gradients

### Extensions

Potential improvements:

1. **Multi-axis control**: Extend to 3D systems (quadrotors, robot arms)
2. **Model-based approaches**: Incorporate system identification
3. **Domain randomization**: Vary plant parameters (J, B) during training
4. **Real-world transfer**: Deploy on hardware with sim-to-real techniques
5. **Hierarchical control**: Add higher-level trajectory planning

### C++ Integration (Optional)

For high-fidelity simulation or hardware deployment:

1. **Plant in C++**: Use Eigen for dynamics, expose via pybind11
2. **Low-latency PID**: C++ inner loop for realistic timing
3. **ROS/Gazebo**: Integrate with robotics middleware
4. **Hardware-in-the-loop**: Test on actual systems

## Evaluation Metrics

Compare against baselines:

| Metric | Description | Better |
|--------|-------------|--------|
| ISE | Integrated Squared Error | Lower |
| RMSE | Root Mean Square Error | Lower |
| Settling Time | Time to reach ±2% of setpoint | Lower |
| Overshoot % | Maximum overshoot percentage | Lower |
| Control Effort | Sum of squared control inputs | Lower |

## Citation

If you use this code in your research, please cite:

```bibtex
@software{dppo_pid_controller,
  title = {DPPO for Real-Time PID Parameter Tuning},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/DPPO_PID_controller}
}
```

## License

This project is licensed under the MIT License.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## Acknowledgments

- **Stable-Baselines3**: Robust PPO implementation
- **Gymnasium**: Clean RL environment interface
- **OpenAI**: Original PPO algorithm

## References

1. Schulman et al. (2017). "Proximal Policy Optimization Algorithms"
2. Lillicrap et al. (2015). "Continuous control with deep reinforcement learning"
3. Åström & Murray (2008). "Feedback Systems: An Introduction for Scientists and Engineers"

## Contact

For questions or issues, please open an issue on GitHub.
