# Vision-DPPO: End-to-End Drone Control via Diffusion Policy
# 基於視覺與擴散策略的無人機端到端直接控制

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Research](https://img.shields.io/badge/research-Vision--DPPO-red.svg)](RESEARCH_PLAN.md)

---

## Project Goal / 項目目標

**English:**
To implement an **end-to-end vision-based drone controller** where a **Diffusion Policy** directly outputs 4D motor thrust commands from FPV (First-Person View) image sequences, completely bypassing traditional PID controllers. The system learns visuomotor control through a 5-phase pipeline:

1. Train a state-based PPO expert for direct motor control
2. Collect expert demonstrations with synthetic FPV images
3. Train a Vision Diffusion Policy via imitation learning
4. Fine-tune with DPPO (Diffusion Policy Policy Optimization)
5. Deploy on NVIDIA Orin + PX4 via ROS 2

**中文：**
本項目旨在實現一個**基於視覺的端到端無人機控制器**，其中**擴散策略（Diffusion Policy）**直接從 FPV（第一人稱視角）圖像序列輸出 4D 電機推力命令，完全繞過傳統 PID 控制器。系統通過 5 階段管線學習視覺運動控制：

1. 訓練基於狀態的 PPO 專家進行直接電機控制
2. 使用合成 FPV 圖像收集專家示範數據
3. 通過模仿學習訓練視覺擴散策略
4. 使用 DPPO（擴散策略策略優化）進行微調
5. 部署於 NVIDIA Orin + PX4（通過 ROS 2）

---

## Key Innovation / 核心創新

```
Traditional Approach / 傳統方法:
  Sensor → State Estimation → Position PID → Attitude PID → Rate PID → Motor Mixing → Motors

Vision-DPPO (This Project) / 本項目:
  FPV Image Sequence → Vision Encoder (CNN) → Diffusion Policy (1D U-Net) → Motor Thrusts
```

The Diffusion Policy replaces the entire cascaded PID control stack with a single learned policy that maps visual observations directly to motor commands.

擴散策略用單一學習策略替代整個級聯 PID 控制堆棧，直接從視覺觀測映射到電機命令。

---

## Development Phases / 開發階段

```
Phase 1: Quadrotor Environment + PPO Expert  ✅ IMPLEMENTED
第一階段：四旋翼環境 + PPO 專家                ✅ 已實現
   ├─ 6-DOF quadrotor dynamics (quaternion, RK4 @ 200Hz)
   │  6-DOF 四旋翼動力學（四元數、RK4 @ 200Hz）
   ├─ 15D observation, 4D motor thrust action
   │  15 維觀測、4 維電機推力動作
   └─ PPO expert with TanhNormal distribution
      PPO 專家（TanhNormal 分佈）

Phase 2: Expert Data Collection  ✅ IMPLEMENTED
第二階段：專家數據收集            ✅ 已實現
   ├─ Synthetic FPV rendering (64×64 RGB)
   │  合成 FPV 渲染（64×64 RGB）
   ├─ HDF5 dataset (images + actions + states)
   │  HDF5 數據集（圖像 + 動作 + 狀態）
   └─ Sliding window: T_obs=2 frames → T_pred=8 actions
      滑動窗口：T_obs=2 幀 → T_pred=8 動作

Phase 3: Vision Diffusion Policy  ✅ IMPLEMENTED
第三階段：視覺擴散策略              ✅ 已實現
   ├─ Vision Encoder (4-layer CNN → 256D features)
   │  視覺編碼器（4 層 CNN → 256D 特徵）
   ├─ Conditional 1D U-Net (FiLM conditioning)
   │  條件 1D U-Net（FiLM 調制）
   ├─ DDPM training / DDIM fast inference (10 steps)
   │  DDPM 訓練 / DDIM 快速推理（10 步）
   └─ DPPO advantage-weighted fine-tuning
      DPPO 優勢加權微調

Phase 4: Closed-Loop RHC Evaluation  ✅ IMPLEMENTED
第四階段：閉環 RHC 評估                ✅ 已實現
   ├─ Receding Horizon Control (predict 8, execute 4)
   │  滾動時域控制（預測 8 步、執行 4 步）
   ├─ Performance comparison vs PPO expert
   │  性能對比 vs PPO 專家
   └─ Metrics: RMSE, crash rate, inference latency
      指標：RMSE、崩潰率、推理延遲

Phase 5: Hardware Deployment  📋 PLANNED
第五階段：硬件部署              📋 規劃中
   ├─ NVIDIA Jetson Orin (TensorRT optimization)
   ├─ PX4 via ROS 2 uXRCE-DDS
   └─ Real FPV camera integration
      真實 FPV 攝像頭集成
```

---

## Quick Start / 快速開始

### Prerequisites / 前置要求

- Python 3.8+
- PyTorch 2.0+ (with CUDA recommended / 建議使用 CUDA)
- pip package manager

### Installation / 安裝

```bash
# Clone the repository / 克隆倉庫
git clone <repository-url>
cd DPPO_PID_controller

# Create virtual environment / 創建虛擬環境
python -m venv dppo
# Windows:
.\dppo\Scripts\activate
# Linux/macOS:
source dppo/bin/activate

# Install dependencies / 安裝依賴
pip install -r requirements.txt

# Install PyTorch with CUDA (recommended)
# 安裝 PyTorch CUDA 版（建議）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### Training Pipeline / 訓練管線

```bash
# Phase 1: Train PPO expert (state-based, ~5M timesteps)
# 第一階段：訓練 PPO 專家（基於狀態，約 500 萬步）
python -m scripts.train_ppo_expert

# Phase 2: Collect expert demonstrations (FPV images + motor actions)
# 第二階段：收集專家示範（FPV 圖像 + 電機動作）
python -m scripts.collect_data \
    --ppo-model checkpoints/ppo_expert/.../best_model.pt \
    --ppo-norm checkpoints/ppo_expert/.../best_obs_rms.npz

# Phase 3: Train Vision Diffusion Policy (supervised)
# 第三階段：訓練視覺擴散策略（監督學習）
python -m scripts.train_diffusion --config configs/diffusion_policy.yaml

# Phase 3b: DPPO fine-tuning (optional, RL-based)
# 第三階段 b：DPPO 微調（可選，基於 RL）
python -m scripts.train_dppo \
    --pretrained checkpoints/diffusion_policy/.../best_model.pt

# Phase 4: Evaluate with Receding Horizon Control
# 第四階段：使用滾動時域控制評估
python -m scripts.evaluate_rhc \
    --diffusion-model checkpoints/diffusion_policy/.../best_model.pt \
    --ppo-model checkpoints/ppo_expert/.../best_model.pt \
    --ppo-norm checkpoints/ppo_expert/.../best_obs_rms.npz
```

### Monitor Training / 監控訓練

```bash
tensorboard --logdir ./runs/
```

---

## Project Structure / 項目結構

```
DPPO_PID_controller/
├── configs/                          # Configuration files / 配置文件
│   ├── quadrotor.yaml                # Quadrotor physics + reward / 四旋翼物理 + 獎勵
│   ├── ppo_expert.yaml               # PPO expert training / PPO 專家訓練
│   └── diffusion_policy.yaml         # Diffusion + U-Net + vision / 擴散 + U-Net + 視覺
│
├── envs/                             # Gymnasium environments / Gymnasium 環境
│   ├── __init__.py
│   ├── quadrotor_dynamics.py         # Pure 6-DOF physics (quaternion, RK4, motors)
│   │                                 # 純 6-DOF 物理（四元數、RK4、電機）
│   ├── quadrotor_env.py              # Gymnasium wrapper (15D obs, 4D action)
│   │                                 # Gymnasium 包裝器（15D 觀測、4D 動作）
│   └── quadrotor_visual_env.py       # FPV image rendering wrapper
│                                     # FPV 圖像渲染包裝器
│
├── models/                           # Neural network models / 神經網路模型
│   ├── __init__.py
│   ├── ppo_expert.py                 # Actor-Critic PPO (direct motor control)
│   │                                 # Actor-Critic PPO（直接電機控制）
│   ├── vision_encoder.py             # CNN (image stack → 256D features)
│   │                                 # CNN（圖像堆疊 → 256D 特徵）
│   ├── conditional_unet1d.py         # 1D U-Net for action denoising
│   │                                 # 用於動作去噪的 1D U-Net
│   ├── diffusion_process.py          # DDPM/DDIM forward + reverse process
│   │                                 # DDPM/DDIM 正向 + 反向過程
│   └── diffusion_policy.py           # Full Vision Diffusion Policy (glue)
│                                     # 完整視覺擴散策略（膠合模組）
│
├── scripts/                          # Training & evaluation scripts / 訓練與評估腳本
│   ├── train_ppo_expert.py           # Phase 1: Train state-based PPO expert
│   ├── collect_data.py               # Phase 2: Collect FPV demos → HDF5
│   ├── train_diffusion.py            # Phase 3: Supervised diffusion training
│   ├── train_dppo.py                 # Phase 3b: DPPO RL fine-tuning
│   └── evaluate_rhc.py              # Phase 4: Closed-loop RHC evaluation
│
├── utils/                            # Utilities / 工具模組
│   ├── training_metrics.py           # Training metric tracking
│   └── visualization.py              # 3D trajectory visualization
│
├── data/                             # Expert demo datasets (HDF5)
│   └── .gitkeep                      # 專家示範數據集（HDF5）
│
├── requirements.txt                  # Python dependencies
├── README.md                         # This file / 本文件
└── RESEARCH_PLAN.md                  # Detailed research plan / 詳細研究計劃
```

---

## System Architecture / 系統架構

### End-to-End Vision-to-Motor Pipeline / 端到端視覺到電機管線

```
┌──────────────────────────────────────────────────────────────┐
│                  FPV Image Sequence / FPV 圖像序列            │
│              (T_obs=2 frames, 64×64 RGB each)                │
│              (T_obs=2 幀，每幀 64×64 RGB)                     │
└──────────────────┬───────────────────────────────────────────┘
                   │ (B, 6, 64, 64)
                   ↓
┌──────────────────────────────────────────────────────────────┐
│              Vision Encoder (CNN) / 視覺編碼器                │
│  Conv2d(6,32,3,2) → Conv2d(32,64,3,2) → Conv2d(64,128,3,2) │
│  → Conv2d(128,256,3,2) → AdaptiveAvgPool → Linear(256)      │
│  GroupNorm + Mish activation throughout                       │
└──────────────────┬───────────────────────────────────────────┘
                   │ (B, 256) visual features / 視覺特徵
                   ↓
┌──────────────────────────────────────────────────────────────┐
│          Conditional 1D U-Net / 條件 1D U-Net                 │
│                                                               │
│  Condition: visual_features(256) + time_embed(128) = 384D    │
│  Input: noisy action sequence (B, 4, 8)                      │
│  輸入：含噪動作序列 (B, 4, 8)                                │
│                                                               │
│  Encoder: ResBlock(4→256) ↓ ResBlock(256→512) ↓              │
│  Mid:     ResBlock(512→512)                                   │
│  Decoder: ResBlock(512→256) ↑ ResBlock(256→4) ↑              │
│  FiLM conditioning (scale + shift) at each block             │
│  FiLM 調制（縮放 + 偏移）在每個區塊                          │
└──────────────────┬───────────────────────────────────────────┘
                   │ predicted noise ε_θ / 預測噪聲 ε_θ
                   ↓
┌──────────────────────────────────────────────────────────────┐
│         DDIM Reverse Process (10 steps) / DDIM 反向過程       │
│                                                               │
│  Iteratively denoise: a_T → a_{T-1} → ... → a_0             │
│  迭代去噪：a_T → a_{T-1} → ... → a_0                        │
│  Output: clean action sequence (B, 8, 4) clamped to [-1,1]  │
│  輸出：乾淨動作序列 (B, 8, 4) 裁剪到 [-1,1]                 │
└──────────────────┬───────────────────────────────────────────┘
                   │ (T_pred=8, action_dim=4)
                   ↓
┌──────────────────────────────────────────────────────────────┐
│      Receding Horizon Control (RHC) / 滾動時域控制            │
│                                                               │
│  Execute first T_action=4 actions (out of 8 predicted)       │
│  執行前 T_action=4 個動作（共預測 8 個）                      │
│  Then re-observe and re-plan / 然後重新觀測並重新規劃         │
│                                                               │
│  Decision frequency: 50Hz / 4 = 12.5 Hz                     │
│  決策頻率：50Hz / 4 = 12.5 Hz                                │
└──────────────────┬───────────────────────────────────────────┘
                   │ 4D motor thrusts / 4D 電機推力
                   ↓
┌──────────────────────────────────────────────────────────────┐
│              6-DOF Quadrotor / 6-DOF 四旋翼                   │
│                                                               │
│  m·dv/dt = R(q)·[0,0,F_total]^T - [0,0,m·g]^T + F_drag     │
│  I·dω/dt = τ - ω × (I·ω)                                    │
│  dq/dt = 0.5 · q ⊗ [0, ω]                                   │
│                                                               │
│  RK4 integration @ 200Hz, motor lag τ=0.02s                  │
│  RK4 積分 @ 200Hz，電機滯後 τ=0.02s                          │
└──────────────────────────────────────────────────────────────┘
```

### Timing Configuration / 時序配置

| Parameter / 參數 | Value / 數值 | Description / 描述 |
|---|---|---|
| Physics Δt | 0.005s (200 Hz) | RK4 integration / RK4 積分 |
| RL Decision Δt | 0.02s (50 Hz) | Motor command update / 電機命令更新 |
| Inner Steps per Decision | 4 | Physics steps per RL step / 每個 RL 步的物理步數 |
| RHC Replan Rate | ~12.5 Hz | Every T_action=4 RL steps / 每 T_action=4 個 RL 步 |

---

## MDP Definition / MDP 定義

### Observation Space (15D) / 觀測空間 (15D)

| Dims | Content / 內容 | Description / 描述 |
|------|---------|-------------|
| 0-2 | Position error (body frame) | R^T · (p_target - p) / 體座標系位置誤差 |
| 3-8 | 6D rotation representation | First 2 columns of R / 旋轉矩陣前 2 列 |
| 9-11 | Linear velocity (body frame) | R^T · v / 體座標系線速度 |
| 12-14 | Angular velocity (body frame) | ω / 體座標系角速度 |

### Action Space (4D) / 動作空間 (4D)

Normalized motor thrusts in [-1, 1], mapped to [0, f_max].

歸一化電機推力 [-1, 1]，映射到 [0, f_max]。

### Reward Function / 獎勵函數

Gaussian-based bounded reward / 基於高斯的有界獎勵：

```
R = w_pos · exp(-||pos_err||² / σ_pos)    # Position tracking / 位置跟蹤
  + w_vel · exp(-||vel||² / σ_vel)         # Velocity penalty / 速度懲罰
  + w_ang · exp(-||ang_vel||² / σ_ang)     # Angular rate penalty / 角速率懲罰
  - w_action · ||action||²                 # Control effort / 控制努力
  + alive_bonus                            # Survival bonus / 存活獎勵
```

### Termination / 終止條件

- Position out of bounds (> 5m) / 位置越界
- Tilt angle > 60° / 傾斜角 > 60°
- Ground contact (Z > 0 in NED frame) / 地面碰撞（NED 座標系 Z > 0）

---

## Quadrotor Physics / 四旋翼物理

### Coordinate Frame / 座標系

- **World frame**: NED (North-East-Down) / 世界座標系：NED（北-東-下）
- **Body frame**: X-forward, Y-right, Z-down / 體座標系：X-前、Y-右、Z-下
- **Attitude**: Quaternion [qw, qx, qy, qz] / 姿態：四元數

### Motor Configuration / 電機配置

X-configuration with motor mixing matrix / X 構型電機混合矩陣：

```
Motor Layout (top view):  1(CW)  ×  2(CCW)
                          3(CCW) ×  4(CW)

F_total = f1 + f2 + f3 + f4
τ_x = L · (f1 - f2 - f3 + f4)      # Roll torque
τ_y = L · (f1 + f2 - f3 - f4)      # Pitch torque
τ_z = c_τ · (-f1 + f2 - f3 + f4)   # Yaw torque
```

### Default Parameters / 默認參數

| Parameter / 參數 | Value / 數值 | Description / 描述 |
|---|---|---|
| Mass / 質量 | 0.5 kg | Total mass / 總質量 |
| Arm length / 臂長 | 0.17 m | Motor to center / 電機到中心距離 |
| Inertia / 慣量 | diag(2.5e-3, 2.5e-3, 4.5e-3) | kg·m² |
| Max thrust / 最大推力 | 3.0 N per motor | 每電機最大推力 |
| Motor time constant | 0.02 s | First-order lag / 一階滯後 |

---

## Diffusion Policy Details / 擴散策略詳情

### Training (DDPM) / 訓練

- **Forward process**: Cosine beta schedule, 100 timesteps
- **Loss**: MSE between predicted noise ε_θ and actual noise ε
- **Optimizer**: AdamW, cosine LR with warmup

### Inference (DDIM) / 推理

- **Steps**: 10 DDIM steps (accelerated from 100)
- **Output**: Action sequence (T_pred=8, action_dim=4)
- **Target latency**: < 50ms per inference / 目標延遲：< 50ms

### DPPO Fine-Tuning / DPPO 微調

Advantage-weighted diffusion loss / 優勢加權擴散損失：

```
L = E[ exp(β · A_normalized) · ||ε_θ(a_t, t, s) - ε||² ]
```

Where A_normalized is the GAE advantage from closed-loop rollouts.

---

## Configuration / 配置

All hyperparameters are defined in YAML config files:

所有超參數在 YAML 配置文件中定義：

- **`configs/quadrotor.yaml`** — Physics, timing, reward, termination, disturbances
- **`configs/ppo_expert.yaml`** — PPO training (learning rate, hidden dim, GAE, etc.)
- **`configs/diffusion_policy.yaml`** — Vision encoder, U-Net, diffusion, DPPO, action horizon

---

## Evaluation Metrics / 評估指標

| Metric / 指標 | Description / 描述 | Goal / 目標 |
|---|---|---|
| Position RMSE | Root mean square position error / 均方根位置誤差 | Lower ↓ |
| Crash Rate | Episodes ending in crash / 崩潰回合比率 | Lower ↓ |
| Episode Reward | Cumulative reward per episode / 每回合累積獎勵 | Higher ↑ |
| Inference Latency | DDIM sampling time / DDIM 採樣時間 | < 50ms |
| Diffusion/PPO Ratio | Performance vs expert baseline / 性能 vs 專家基準 | > 80% |

---

## References / 參考文獻

1. Ho et al. (2020). "Denoising Diffusion Probabilistic Models"
2. Song et al. (2020). "Denoising Diffusion Implicit Models"
3. Chi et al. (2023). "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"
4. Schulman et al. (2017). "Proximal Policy Optimization Algorithms"
5. Kaufmann et al. (2023). "Champion-level drone racing using deep reinforcement learning" (Swift)
6. Beard & McLain (2012). "Small Unmanned Aircraft: Theory and Practice"

---

## License / 許可證

This project is licensed under the MIT License.
本項目根據 MIT 許可證授權。

---

**For detailed research plan, see [RESEARCH_PLAN.md](RESEARCH_PLAN.md)**
**詳細研究計劃請見 [RESEARCH_PLAN.md](RESEARCH_PLAN.md)**
