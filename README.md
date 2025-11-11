# DPPO for Real-Time Adaptive PID Tuning
# DPPO 實時自適應 PID 調參系統

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Research](https://img.shields.io/badge/research-DPPO-red.svg)](RESEARCH_PLAN.md)

---

## Project Goal / 項目目標

**English:**
To implement a Deep Reinforcement Learning (DRL) agent using **Diffusion Policy Policy Optimization (DPPO)** as a Meta-Controller to learn optimal strategies for real-time adjustment of PID gains (K_p, K_i, K_d) for multi-axis control systems, specifically targeting inner-loop attitude rate control of a simulated quadrotor.

The project achieves **adaptive control** that is:
- **Robust** to disturbances and parameter variations
- **High-performance** in tracking diverse trajectories
- **Adaptive** compared to fixed controllers

**中文：**
本項目旨在實現一個基於**擴散策略策略優化（DPPO）**的深度強化學習（DRL）智能體，作為元控制器學習最優策略，實時調整多軸控制系統的 PID 增益（K_p, K_i, K_d），特別針對模擬四旋翼飛行器的內環姿態角速率控制。

項目實現的**自適應控制**具有以下特點：
- 對擾動和參數變化具有**魯棒性**
- 在跟蹤多樣化軌跡時具有**高性能**
- 相比固定控制器更具**自適應性**

---

## Research Focus / 研究重點

**English:**
**Primary Focus**: Phase 3 - DPPO Policy Model Implementation (see [RESEARCH_PLAN.md](RESEARCH_PLAN.md))

This project represents a novel combination of:
- Diffusion Models for action generation
- PPO objectives for policy optimization
- Real-time PID parameter tuning

**中文：**
**主要研究重點**：第三階段 - DPPO 策略模型實現（詳見 [RESEARCH_PLAN.md](RESEARCH_PLAN.md)）

本項目代表了以下技術的創新結合：
- 用於動作生成的擴散模型
- 用於策略優化的 PPO 目標
- 實時 PID 參數調整

---

## Development Phases / 開發階段

```
Phase 1: Single-Axis Foundation ✓ IMPLEMENTED
第一階段：單軸基礎 ✓ 已實現
   ├─ 2nd-order system dynamics / 二階系統動力學
   ├─ RK4 integration / RK4 積分
   ├─ PID inner loop (200 Hz) / PID 內環（200 Hz）
   └─ PPO meta-controller (20 Hz) / PPO 元控制器（20 Hz）

Phase 2: DPPO MDP Definition ✓ COMPLETE
第二階段：DPPO MDP 定義 ✓ 完成
   ├─ 9D observation space / 9 維觀測空間
   ├─ 3D action space / 3 維動作空間
   └─ Multi-objective reward function / 多目標獎勵函數

Phase 3: DPPO Implementation 🚧 IN PROGRESS (CORE RESEARCH)
第三階段：DPPO 實現 🚧 進行中（核心研究）
   ├─ Diffusion model policy / 擴散模型策略
   ├─ PPO-weighted training / PPO 加權訓練
   └─ Fast inference (<50ms) / 快速推理（<50ms）

Phase 4: 6-DOF Quadrotor 📋 PLANNED
第四階段：6-DOF 四旋翼 📋 規劃中
   ├─ Full nonlinear dynamics / 完整非線性動力學
   ├─ Cascaded control (position → attitude → rate)
   │  級聯控制（位置 → 姿態 → 角速率）
   ├─ 27+ dimensional state space / 27+ 維狀態空間
   └─ 9D action space (3 axes × 3 gains)
      9 維動作空間（3 軸 × 3 增益）

Phase 5: Evaluation & Deployment 📋 PLANNED
第五階段：評估與部署 📋 規劃中
   ├─ Baseline comparisons (Manual PID, LQR, Fixed RL-PID)
   │  基準對比（手動 PID、LQR、固定 RL-PID）
   ├─ Performance metrics (RMSE, settling time, robustness)
   │  性能指標（RMSE、穩定時間、魯棒性）
   └─ Real-time deployment considerations
      實時部署考慮
```

📖 **See [RESEARCH_PLAN.md](RESEARCH_PLAN.md) for complete specifications**
📖 **詳見 [RESEARCH_PLAN.md](RESEARCH_PLAN.md) 獲取完整規格說明**

---

## Quick Start (Phase 1 - Current) / 快速開始（第一階段 - 當前）

```bash
# Install dependencies / 安裝依賴
pip install -r requirements.txt

# Run demo to test environment / 運行演示測試環境
python demo.py

# Train PPO agent (Phase 1 baseline before DPPO)
# 訓練 PPO 智能體（第一階段基準，DPPO 之前）
python train.py

# Evaluate trained model / 評估訓練模型
python evaluate.py --model models/dppo_pid_final_*.zip

# Test DPPO model structure (Phase 3)
# 測試 DPPO 模型結構（第三階段）
python dppo_model.py
```

---

## Table of Contents / 目錄

- [Project Structure / 項目結構](#project-structure--項目結構)
- [Technology Stack / 技術棧](#technology-stack--技術棧)
- [Installation / 安裝](#installation--安裝)
- [System Architecture / 系統架構](#system-architecture--系統架構)
- [Usage / 使用方法](#usage--使用方法)
- [Configuration / 配置](#configuration--配置)
- [Implementation Details / 實現細節](#implementation-details--實現細節)
- [Results and Visualization / 結果與可視化](#results-and-visualization--結果與可視化)
- [Advanced Topics / 進階主題](#advanced-topics--進階主題)

---

## Project Structure / 項目結構

```
DPPO_PID_controller/
├── Phase 1 & 2: Single-Axis with PPO
│   第一、二階段：基於 PPO 的單軸系統
│   ├── dppo_pid_env.py          # Gymnasium environment (Phase 1)
│   │                            # Gymnasium 環境（第一階段）
│   ├── train.py                 # PPO training script / PPO 訓練腳本
│   ├── evaluate.py              # Evaluation and visualization
│   │                            # 評估與可視化
│   ├── demo.py                  # Demo/testing script / 演示/測試腳本
│   └── config.yaml              # Phase 1 configuration / 第一階段配置
│
├── Phase 3: DPPO Implementation (CORE)
│   第三階段：DPPO 實現（核心）
│   ├── dppo_model.py            # DPPO model (🚧 skeleton)
│   │                            # DPPO 模型（🚧 骨架）
│   └── train_dppo.py            # DPPO training (TODO)
│                                # DPPO 訓練（待實現）
│
├── Phase 4: 6-DOF Quadrotor
│   第四階段：6-DOF 四旋翼
│   ├── quadrotor_6dof_env.py    # 6-DOF environment (📋 placeholder)
│   │                            # 6-DOF 環境（📋 佔位符）
│   └── config_6dof.yaml         # Phase 4 configuration (TODO)
│                                # 第四階段配置（待實現）
│
├── Documentation / 文檔
│   ├── README.md                # This file / 本文件
│   ├── RESEARCH_PLAN.md         # Complete research plan / 完整研究計劃
│   └── PPO_HYPERPARAMETERS.md   # Hyperparameter guide (中英文)
│                                # 超參數指南（中英文）
│
└── Configuration / 配置
    ├── requirements.txt         # Python dependencies / Python 依賴
    └── .gitignore               # Git ignore patterns / Git 忽略模式
```

---

## Technology Stack / 技術棧

| Component<br>組件 | Technology<br>技術 | Role<br>作用 |
|-----------|-----------|------|
| RL Framework<br>強化學習框架 | Stable-Baselines3 (PPO) | Implements the DRL algorithm and training loop<br>實現 DRL 算法和訓練循環 |
| Diffusion Model<br>擴散模型 | PyTorch (custom)<br>PyTorch（自定義） | DPPO policy network for action generation<br>用於動作生成的 DPPO 策略網絡 |
| Simulation Environment<br>模擬環境 | Farama Gymnasium (custom)<br>Farama Gymnasium（自定義） | Defines state, action, reward, and transition dynamics<br>定義狀態、動作、獎勵和轉移動力學 |
| System Dynamics (Plant)<br>系統動力學（被控對象） | NumPy | High-speed numerical integration for physics model<br>物理模型的高速數值積分 |
| Logging/Visualization<br>日誌/可視化 | TensorBoard & Matplotlib | Tracks learning progress and performance metrics<br>追蹤學習進度和性能指標 |

---

## Installation / 安裝

### Prerequisites / 前置要求

**English:**
- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for Phase 3 DPPO training

**中文：**
- Python 3.8 或更高版本
- pip 套件管理器
- （可選）支持 CUDA 的 GPU 用於第三階段 DPPO 訓練

### Setup / 設置

```bash
# Clone the repository / 克隆倉庫
git clone <repository-url>
cd DPPO_PID_controller

# Create a virtual environment
python -m venv dppo  

# Activate the virtual environment
.\dppo\Scripts\activate 

# Install dependencies / 安裝依賴
pip install -r requirements.txt
```

---

## System Architecture / 系統架構

**English:**
The system implements a **dual-loop control architecture**:

**中文：**
系統實現了**雙迴路控制架構**：

### Two-Timescale Control Loop / 雙時間尺度控制迴路

```
┌──────────────────────────────────────────────────────┐
│              Meta-Controller / 元控制器               │
│        (PPO/DPPO RL Agent @ 20 Hz)                   │
│        (PPO/DPPO 強化學習智能體 @ 20 Hz)             │
│                                                       │
│  Inputs / 輸入: [error, error_dot, integral, x,      │
│                  x_dot, reference, Kp, Ki, Kd]       │
│  Outputs / 輸出: [Kp_new, Ki_new, Kd_new]           │
└─────────────────┬────────────────────────────────────┘
                  │ Updates every 0.05s / 每 0.05 秒更新
                  ↓
┌──────────────────────────────────────────────────────┐
│         Inner PID Controller / 內環 PID 控制器       │
│                  (@ 200 Hz)                          │
│                                                       │
│  u(t) = Kp·e(t) + Ki·∫e(t)dt + Kd·de(t)/dt         │
└─────────────────┬────────────────────────────────────┘
                  │ Control signal u(t) / 控制信號 u(t)
                  ↓
┌──────────────────────────────────────────────────────┐
│       2nd-Order Plant System / 二階被控系統          │
│                                                       │
│          J·ẍ + B·ẋ = u(t) + d(t)                    │
│                                                       │
│  J: Inertia (1.0) / 慣量 (1.0)                      │
│  B: Damping (0.5) / 阻尼 (0.5)                      │
│  d: External disturbance / 外部擾動                  │
└──────────────────────────────────────────────────────┘
```

### Timing Configuration / 時序配置

| Parameter<br>參數 | Value<br>數值 | Description<br>描述 |
|-----------|-------|-------------|
| Inner Loop Δt<br>內環 Δt | 0.005s (200 Hz) | PID calculation and physics integration<br>PID 計算和物理積分 |
| Outer Loop Δt<br>外環 Δt | 0.05s (20 Hz) | RL agent updates PID gains<br>RL 智能體更新 PID 增益 |
| Steps per RL Action<br>每個 RL 動作的步數 | 10 | Inner loop steps per outer loop step<br>每個外環步驟的內環步數 |

### Plant Dynamics / 被控對象動力學

**English:**
The environment simulates a **2nd-order linear system**:

**中文：**
環境模擬一個**二階線性系統**：

```
J·ẍ + B·ẋ = u(t) + d(t)
```

**Where / 其中：**
- **x**: Position/angle of the system / 系統的位置/角度
- **u(t)**: Control output from PID controller / PID 控制器的控制輸出
- **d(t)**: External disturbance (random, time-limited) / 外部擾動（隨機、時限）
- **J**: Inertia coefficient (default: 1.0) / 慣量係數（默認：1.0）
- **B**: Damping coefficient (default: 0.5) / 阻尼係數（默認：0.5）

### PID Controller / PID 控制器

**Standard parallel-form PID / 標準並聯式 PID：**

```
u(t) = Kp·e(t) + Ki·∫e(t)dt + Kd·(e(t) - e(t-1))/Δt
```

**Where / 其中：**
- **e(t) = r(t) - x(t)**: Tracking error / 跟蹤誤差
- **Kp, Ki, Kd**: Gains adjusted by the RL agent / 由 RL 智能體調整的增益
- **r(t)**: Reference setpoint (changes periodically) / 參考設定值（週期性變化）

---

## Usage / 使用方法

### 1. Test the Environment (Demo) / 測試環境（演示）

**English:**
Run the demo script to verify the environment works correctly:

**中文：**
運行演示腳本以驗證環境是否正常工作：

```bash
python demo.py
```

**This will / 這將：**
- Test the environment API / 測試環境 API
- Run an episode with random PID gains / 使用隨機 PID 增益運行一個回合
- Generate a visualization (`demo_results.png`) / 生成可視化結果（`demo_results.png`）

### 2. Train the Agent / 訓練智能體

**English:**
Start training with default configuration:

**中文：**
使用默認配置開始訓練：

```bash
python train.py
```

**With custom configuration / 使用自定義配置：**

```bash
python train.py --config my_config.yaml
```

**Resume training from a checkpoint / 從檢查點恢復訓練：**

```bash
python train.py --resume --model models/dppo_pid_checkpoint_1000000_steps.zip
```

**Monitor training progress with TensorBoard / 使用 TensorBoard 監控訓練進度：**

```bash
tensorboard --logdir ./ppo_pid_logs/
```

### 3. Evaluate Trained Model / 評估訓練模型

**English:**
Evaluate and visualize performance:

**中文：**
評估並可視化性能：

```bash
python evaluate.py --model models/dppo_pid_final_TIMESTAMP.zip --episodes 10
```

**This generates / 這將生成：**
- Performance plots for best/worst episodes / 最佳/最差回合的性能圖表
- Summary statistics across all episodes / 所有回合的統計摘要
- Saved in `./evaluation_results/` / 保存在 `./evaluation_results/`

---

## Configuration / 配置

**English:**
All hyperparameters are defined in `config.yaml`. Key sections:

**中文：**
所有超參數在 `config.yaml` 中定義。主要部分：

### Plant Parameters / 被控對象參數

```yaml
plant:
  J: 1.0      # Inertia / 慣量
  B: 0.5      # Damping / 阻尼
  u_min: -10.0  # Min control / 最小控制
  u_max: 10.0   # Max control / 最大控制
  integration_method: "rk4"  # "euler" or "rk4"
```

### Reward Weights / 獎勵權重

```yaml
reward:
  lambda_error: 5.0       # Tracking error weight / 跟蹤誤差權重
  lambda_velocity: 0.5    # Velocity penalty weight / 速度懲罰權重
  lambda_control: 0.01    # Control effort weight / 控制努力權重
  lambda_overshoot: 0.2   # Overshoot penalty weight / 超調懲罰權重
```

### PPO Training Parameters / PPO 訓練參數

```yaml
training:
  total_timesteps: 5000000  # Total training steps / 總訓練步數
  learning_rate: 0.0003     # Learning rate / 學習率
  n_steps: 2048             # Trajectory length / 軌跡長度
  batch_size: 64            # Mini-batch size / 小批量大小
  gamma: 0.99               # Discount factor / 折扣因子
```

**See `config.yaml` for all available parameters.**
**查看 `config.yaml` 獲取所有可用參數。**

---

## Implementation Details / 實現細節

### Markov Decision Process (MDP) / 馬爾可夫決策過程 (MDP)

#### Action Space (𝐀) / 動作空間 (𝐀)

**Dimensions / 維度**: 3 (continuous / 連續)
**Range / 範圍**: [0.0, K_max]
- Kp_max = 10.0
- Ki_max = 5.0
- Kd_max = 5.0

**Components / 組成**: [K_p, K_i, K_d]

**English:**
The agent directly outputs new PID gain values, bounded to prevent instability.

**中文：**
智能體直接輸出新的 PID 增益值，有界以防止不穩定。

#### Observation Space (𝐒) / 觀測空間 (𝐒)

**Dimensions / 維度**: 9 (continuous, normalized to [-1, 1] / 連續，歸一化到 [-1, 1])

**Components / 組成：**

1. Current position error (e) / 當前位置誤差 (e)
2. Error derivative (ė) / 誤差導數 (ė)
3. Accumulated error (integral term) / 累積誤差（積分項）
4. System position (x) / 系統位置 (x)
5. System velocity (ẋ) / 系統速度 (ẋ)
6. Target reference (r) / 目標參考值 (r)
7. Current Kp / 當前 Kp
8. Current Ki / 當前 Ki
9. Current Kd / 當前 Kd

**English:**
Including current gains enables the agent to learn **relative adjustments**.

**中文：**
包含當前增益使智能體能夠學習**相對調整**。

#### Reward Function (𝐑) / 獎勵函數 (𝐑)

**English:**
Multi-objective reward combining four components:

**中文：**
結合四個組成部分的多目標獎勵：

```python
R = -λ₁·e² - λ₂·ẋ² - λ₃·u² - λ₄·max(0, e·ė)
```

| Component<br>組成部分 | Weight (λ)<br>權重 (λ) | Purpose<br>目的 |
|-----------|------------|---------|
| Tracking Error<br>跟蹤誤差 | λ₁ = 5.0 | Minimize deviation from setpoint<br>最小化與設定值的偏差 |
| Velocity Penalty<br>速度懲罰 | λ₂ = 0.5 | Reduce oscillations<br>減少振盪 |
| Control Effort<br>控制努力 | λ₃ = 0.01 | Energy efficiency<br>能量效率 |
| Overshoot<br>超調 | λ₄ = 0.2 | Penalize moving away from setpoint<br>懲罰遠離設定值的移動 |

#### Episode Termination / 回合終止

**English:**
An episode ends when:

**中文：**
當以下情況發生時回合結束：

- Position exceeds safety bounds (|x| > 5.0) → system instability
  位置超出安全界限 (|x| > 5.0) → 系統不穩定
- Maximum steps reached (1000 steps = 50 seconds)
  達到最大步數（1000 步 = 50 秒）

### Reference Signal / 參考信號

**English:**
To encourage adaptation, the setpoint changes periodically:

**中文：**
為了鼓勵自適應，設定值週期性變化：

- Changes every 2 seconds / 每 2 秒變化一次
- Random value in [-2, 2] / [-2, 2] 範圍內的隨機值
- Forces agent to re-tune gains for different operating conditions
  強制智能體針對不同工作條件重新調整增益

### Disturbances / 擾動

**English:**
Random external disturbances test robustness:

**中文：**
隨機外部擾動測試魯棒性：

- Magnitude: ±0.5 / 幅度：±0.5
- Duration: 0.1 seconds / 持續時間：0.1 秒
- Occurs randomly with low probability / 以低概率隨機發生

---

## Results and Visualization / 結果與可視化

### Training Metrics (TensorBoard) / 訓練指標 (TensorBoard)

**Monitor during training / 訓練期間監控：**
- Episode reward (cumulative) / 回合獎勵（累積）
- Episode length / 回合長度
- Policy loss / Value loss / 策略損失 / 價值損失
- Entropy (exploration) / 熵（探索）

### Evaluation Plots / 評估圖表

**English:**
The evaluation script generates:

**中文：**
評估腳本生成：

1. **Position Tracking** / **位置跟蹤**: Shows system response vs. reference signal
   顯示系統響應與參考信號
2. **Tracking Error** / **跟蹤誤差**: Error over time / 隨時間變化的誤差
3. **Control Input** / **控制輸入**: PID output force/torque / PID 輸出力/力矩
4. **PID Gains Evolution** / **PID 增益演化**: How Kp, Ki, Kd change in real-time
   Kp, Ki, Kd 如何實時變化
5. **Summary Statistics** / **統計摘要**: Rewards, errors, and gain distributions
   獎勵、誤差和增益分佈

---

## Advanced Topics / 進階主題

### Curriculum Learning / 課程學習

**English:**
For faster training, consider implementing curriculum learning:

**中文：**
為了更快訓練，考慮實施課程學習：

1. **Phase 1** / **階段 1**: Simple step references, no disturbances
   簡單階躍參考，無擾動
2. **Phase 2** / **階段 2**: Introduce reference changes / 引入參考變化
3. **Phase 3** / **階段 3**: Add external disturbances / 添加外部擾動
4. **Phase 4** / **階段 4**: Increase disturbance magnitude / 增加擾動幅度

**Modify `config.yaml` between phases or implement automatic curriculum in the environment.**
**在階段之間修改 `config.yaml` 或在環境中實現自動課程。**

### Hyperparameter Tuning / 超參數調整

**English:**
Key parameters to tune:

**中文：**
需要調整的關鍵參數：

**Reward Weights / 獎勵權重**: Balance between tracking, stability, and efficiency
平衡跟蹤、穩定性和效率
- Increase `lambda_error` if tracking is poor / 如果跟蹤效果差則增加 `lambda_error`
- Increase `lambda_overshoot` if oscillations occur / 如果發生振盪則增加 `lambda_overshoot`
- Adjust `lambda_velocity` for smoother control / 調整 `lambda_velocity` 以獲得更平滑的控制

**PPO Parameters / PPO 參數**:
- `learning_rate`: Lower (1e-4) for stability, higher (5e-4) for faster learning
  降低（1e-4）以提高穩定性，提高（5e-4）以加快學習
- `n_steps`: More steps = more data per update (but slower)
  更多步數 = 每次更新更多數據（但更慢）
- `batch_size`: Larger batches = more stable gradients
  更大的批次 = 更穩定的梯度

### Extensions / 擴展

**English:**
Potential improvements:

**中文：**
潛在改進：

1. **Multi-axis control** / **多軸控制**: Extend to 3D systems (quadrotors, robot arms)
   擴展到 3D 系統（四旋翼、機器人手臂）
2. **Model-based approaches** / **基於模型的方法**: Incorporate system identification
   結合系統辨識
3. **Domain randomization** / **領域隨機化**: Vary plant parameters (J, B) during training
   訓練期間改變被控對象參數（J, B）
4. **Real-world transfer** / **實際應用遷移**: Deploy on hardware with sim-to-real techniques
   使用仿真到現實技術部署到硬件
5. **Hierarchical control** / **分層控制**: Add higher-level trajectory planning
   添加更高層次的軌跡規劃

### C++ Integration (Optional) / C++ 集成（可選）

**English:**
For high-fidelity simulation or hardware deployment:

**中文：**
用於高保真模擬或硬件部署：

1. **Plant in C++** / **C++ 被控對象**: Use Eigen for dynamics, expose via pybind11
   使用 Eigen 進行動力學計算，通過 pybind11 暴露
2. **Low-latency PID** / **低延遲 PID**: C++ inner loop for realistic timing
   C++ 內環實現真實時序
3. **ROS/Gazebo**: Integrate with robotics middleware / 與機器人中間件集成
4. **Hardware-in-the-loop** / **硬件在環**: Test on actual systems / 在實際系統上測試

---

## Evaluation Metrics / 評估指標

**English:**
Compare against baselines:

**中文：**
與基準進行比較：

| Metric<br>指標 | Description<br>描述 | Better<br>更好 |
|--------|-------------|--------|
| ISE | Integrated Squared Error<br>積分平方誤差 | Lower ↓<br>越低越好 |
| RMSE | Root Mean Square Error<br>均方根誤差 | Lower ↓<br>越低越好 |
| Settling Time<br>穩定時間 | Time to reach ±2% of setpoint<br>達到設定值 ±2% 的時間 | Lower ↓<br>越低越好 |
| Overshoot %<br>超調百分比 | Maximum overshoot percentage<br>最大超調百分比 | Lower ↓<br>越低越好 |
| Control Effort<br>控制努力 | Sum of squared control inputs<br>控制輸入平方和 | Lower ↓<br>越低越好 |

---

## Citation / 引用

**English:**
If you use this code in your research, please cite:

**中文：**
如果您在研究中使用此代碼，請引用：

```bibtex
@software{dppo_pid_controller,
  title = {DPPO for Real-Time Adaptive PID Tuning},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/DPPO_PID_controller}
}
```

---

## License / 許可證

This project is licensed under the MIT License.
本項目根據 MIT 許可證授權。

---

## Contributing / 貢獻

**English:**
Contributions are welcome! Please:

**中文：**
歡迎貢獻！請：

1. Fork the repository / 分叉倉庫
2. Create a feature branch / 創建功能分支
3. Commit your changes / 提交您的更改
4. Push to the branch / 推送到分支
5. Open a Pull Request / 打開拉取請求

---

## Acknowledgments / 致謝

- **Stable-Baselines3**: Robust PPO implementation / 強大的 PPO 實現
- **Gymnasium**: Clean RL environment interface / 清晰的 RL 環境接口
- **OpenAI**: Original PPO algorithm / 原始 PPO 算法

---

## References / 參考文獻

1. Schulman et al. (2017). "Proximal Policy Optimization Algorithms"
2. Lillicrap et al. (2015). "Continuous control with deep reinforcement learning"
3. Åström & Murray (2008). "Feedback Systems: An Introduction for Scientists and Engineers"
4. Ho et al. (2020). "Denoising Diffusion Probabilistic Models"
5. Song et al. (2020). "Denoising Diffusion Implicit Models"

---

## Contact / 聯繫方式

**English:**
For questions or issues, please open an issue on GitHub.

**中文：**
如有問題或疑問，請在 GitHub 上開啟 issue。

---

**📖 For detailed research plan, see [RESEARCH_PLAN.md](RESEARCH_PLAN.md)**
**📖 詳細研究計劃請見 [RESEARCH_PLAN.md](RESEARCH_PLAN.md)**

**📖 For PPO hyperparameter tuning guide, see [PPO_HYPERPARAMETERS.md](PPO_HYPERPARAMETERS.md)**
**📖 PPO 超參數調整指南請見 [PPO_HYPERPARAMETERS.md](PPO_HYPERPARAMETERS.md)**
