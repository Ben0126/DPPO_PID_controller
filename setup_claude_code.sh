#!/usr/bin/env bash
# =============================================================================
# Vision-DPPO v4.0 — Claude Code 自動化設定腳本
# 執行方式：bash setup_claude_code.sh
# 在專案根目錄執行，會建立所有 Claude Code 所需的配置
# =============================================================================

set -e  # 任何錯誤立即停止

# ── 顏色輸出 ──────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; RED='\033[0;31m'; NC='\033[0m'
info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[OK]${NC}   $1"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
header()  { echo -e "\n${GREEN}━━━ $1 ━━━${NC}"; }

# =============================================================================
header "1 / 7  目錄結構建立"
# =============================================================================

DIRS=(
  ".claude/skills/flow-matching"
  ".claude/skills/ctbr-control"
  ".claude/skills/experiment-protocol"
  ".claude/skills/jetson-deploy"
  ".claude/skills/debug-pytorch"
  ".claude/rules"
  "envs" "models" "scripts" "configs" "docs" "deployment/jetson" "utils"
)

for d in "${DIRS[@]}"; do
  mkdir -p "$d"
done
success "目錄結構建立完成"

# =============================================================================
header "2 / 7  CLAUDE.md（根目錄）"
# =============================================================================

cat > CLAUDE.md << 'EOF'
# Vision-DPPO v4.0 — End-to-End FPV Drone Control via Flow Matching

## 研究定位
學習演算法與技能為首要目標，硬體部署成功為 aspirational goal。
**無發表截止壓力。每個 phase 的 gate criteria 未達標，不進入下一階段。**

## RESEARCH_PLAN_v4.0 核心轉換
| 組件         | v3.3（已棄用）          | v4.0（當前）                    |
|------------|----------------------|-------------------------------|
| 動作空間      | SRT (4D motor)       | **CTBR** (thrust + body rates) |
| 生成模型      | DDPM/DDIM            | **Flow Matching**              |
| RL 微調      | DPPO                 | **ReinFlow**                  |
| 內迴路        | 無                   | **500Hz PID → INDI**           |
| 硬體部署      | optional             | **必做 (Jetson Orin Nano)**    |

## 技術棧
- Python 3.10+, PyTorch 2.x, CUDA
- 模擬器：Isaac Gym / custom QuadrotorEnv
- 監控：WandB（必須記錄所有實驗）
- 部署：Jetson Orin Nano + ROS 2 + PX4

## 目錄結構
```
envs/             — 模擬環境（quadrotor_env_v4.py = 當前版本）
models/           — flow_policy.py, ppo_expert.py, vision_encoder.py
scripts/          — train_*.py, collect_data.py, evaluate.py
configs/          — quadrotor_v4.yaml, flow_matching.yaml
docs/             — RESEARCH_PLAN_v4.md, DEV_LOG.md, GATE_CRITERIA.md
deployment/jetson — TensorRT, ROS 2 node
utils/            — metrics.py, wandb_logger.py
```

## 常用指令
```bash
python scripts/train_ppo_expert.py --config configs/quadrotor_v4.yaml
python scripts/collect_data.py     --episodes 1000
python scripts/train_flow_policy.py
python scripts/train_reinflow.py
python scripts/evaluate.py         --checkpoint <path> --n_episodes 50
```

## Phase Gate Criteria（嚴格執行，未達標不進入下一 phase）
### Phase 0: CTBR 環境
- PID inner-loop hover 穩定 500 步，crash rate = 0
- 隨機 CTBR 指令 crash rate < 10%

### Phase 1: PPO Expert
- RMSE < 0.10m（目標 < 0.08m）
- Crash rate = 0/50 episodes
- Action smoothness: |Δaction| < 0.3

### Phase 2: 數據收集
- 1000 episodes，~4GB
- IMU + depth 資料完整
- Data quality check passed

### Phase 3: Flow Matching Policy
- BC Loss 穩定收斂
- RMSE < 0.12m（sim）
- Inference latency < 10ms（sim）

### Phase 4: ReinFlow RL 微調
- RMSE < 0.08m
- Crash rate < 5%（sim）

### Phase 5: Jetson 部署（必做）
- TensorRT 轉換成功，inference < 30ms
- Real-world hover 成功

## Tensor 規範（CRITICAL）
- 所有 forward() 修改前必須先推演並列出 Input/Output Tensor Shape
- Shape 格式：`(Batch, Obs_Dim)` `(Batch, T, C, H, W)` 等
- 必須在 Type Hint 或注釋中標明：`# (B, T, action_dim)`
- Device：一律用 `.to(self.device)`，禁止 hardcode `cuda:0`
- OOM 防禦：for-loop 內累積 loss 必須 `.detach()`，eval 必須 `torch.no_grad()`

## 觀測空間（15D）
| 維度   | 內容                              |
|------|---------------------------------|
| 0-2  | 體座標系位置誤差 R^T(p_target - p)   |
| 3-8  | 6D 旋轉（R 前兩列，避免 Gimbal Lock）  |
| 9-11 | 體座標系線速度 R^T v                 |
| 12-14| 體座標系角速度 ω                    |

## WandB 記錄規範
每次 run 必須記錄（缺少則視為無效實驗）：
- `train/position_rmse`, `train/crash_rate`
- `train/policy_loss`
- `eval/position_rmse`, `eval/crash_rate`
- `system/inference_latency_ms`
- run tags: phase 名稱（e.g., `phase1`, `phase3a`）

## 失敗記錄原則（重要）
歷史失敗 NEVER 刪除，只標記 deprecated。
DEV_LOG.md 必須記錄：現象（附數字）、根本原因、修復方案。

## 嚴格禁止（NEVER）
- NEVER 修改帶 `_v3` 後綴的歷史檔案
- NEVER 在沒有 WandB 的情況下跑超過 100 epoch 的訓練
- NEVER 跳過 gate criteria 進入下一 phase
- NEVER 在 Jetson 上跑未經 sim 驗證的 policy
- NEVER 宣稱「應該可以跑」，必須附上實驗數據（Evidence Before Assertions）
- NEVER hardcode device、batch size 或 checkpoint 路徑

## Skills
See .claude/skills/ for domain-specific workflows
EOF

success "CLAUDE.md 建立完成"

# =============================================================================
header "3 / 7  Skills 建立"
# =============================================================================

# ── Skill 1: Flow Matching ──────────────────────────────────────────────────
cat > .claude/skills/flow-matching/SKILL.md << 'EOF'
---
name: flow-matching
description: Flow Matching 數學推導與 PyTorch 實作，包含 ODE 積分、1-step inference、ReinFlow RL fine-tuning。當需要實作或修改生成模型、推理加速、RL微調時使用。
---
# Flow Matching for Drone Policy

## 核心 ODE
dx = v_θ(x,t)dt，t∈[0,1]
從 x_0 ~ N(0,I) 流向 x_1 ~ p_data(action)

## 訓練（Conditional Flow Matching Loss）
```python
def flow_matching_loss(model, action, condition):
    B = action.shape[0]
    t = torch.rand(B, device=action.device)           # t ~ U[0,1]
    x0 = torch.randn_like(action)                     # 噪聲
    xt = (1 - t[:,None]) * x0 + t[:,None] * action   # 線性插值
    v_target = action - x0                            # 目標向量場
    v_pred = model(xt, t, condition)                  # 預測向量場
    return F.mse_loss(v_pred, v_target)
```

## 推理（N-step ODE Euler Solver）
```python
@torch.no_grad()
def sample(model, condition, n_steps=10, action_dim=4):
    B = condition.shape[0]
    x = torch.randn(B, action_dim, device=condition.device)
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = torch.full((B,), i * dt, device=x.device)
        v = model(x, t, condition)
        x = x + dt * v
    return x  # shape: (B, action_dim)
```

## 1-Step 推理（直接近似，latency 最低）
```python
@torch.no_grad()
def sample_1step(model, condition, action_dim=4):
    B = condition.shape[0]
    x = torch.randn(B, action_dim, device=condition.device)
    t = torch.zeros(B, device=x.device)   # t=0 直接積分到 t=1
    v = model(x, t, condition)
    return x + v   # 近似：忽略曲率
```

## vs DDPM/DDIM 關鍵差異
| 特性          | DDPM/DDIM          | Flow Matching   |
|-------------|-------------------|----------------|
| ε-space 放大  | 最大 ~64× (t=T-1)  | 無，直接預測向量場 |
| 1-step 品質   | 差（需蒸餾）         | 直接可用          |
| RL 相容性     | 需特殊處理           | ReinFlow 原生支援  |
| 訓練穩定性    | 依賴 noise schedule | 線性插值，穩定    |

## ReinFlow 整合要點
- Policy 輸出 v_θ(x_t, t | obs)（條件向量場）
- Action log_prob 用 change-of-variables 計算（需 ODE jacobian）
- KL divergence 在向量場空間計算，避免 ε-space 放大

## Tensor Shape 規範
- condition (obs): (B, obs_dim) = (B, 15)
- action:         (B, action_dim) = (B, 4)  # CTBR
- t:              (B,) float in [0,1]
- v_pred:         (B, action_dim)
EOF

# ── Skill 2: CTBR Control ──────────────────────────────────────────────────
cat > .claude/skills/ctbr-control/SKILL.md << 'EOF'
---
name: ctbr-control
description: CTBR 動作空間設計、PID/INDI inner-loop 實作、與 SRT 的比較。當修改動作空間、調整控制器增益、或診斷墜毀問題時使用。
---
# CTBR Action Space & PID Inner-Loop

## 為何 SRT → CTBR（根本原因）
SRT 問題：12-25Hz 馬達指令直接控制 → 開迴路不穩定 → 100% crash rate
CTBR 解法：Policy 輸出高層指令，500Hz PID 處理姿態穩定

## CTBR 動作定義
```
a = [f_collective, ω_x_cmd, ω_y_cmd, ω_z_cmd]，歸一化到 [-1, 1]
f_collective: 總推力 / (mg)，= 1.0 為懸停
ω_cmd:        目標角速度，±ω_max（典型 ω_max = 5 rad/s）
```

## PID Inner-Loop 實作
```python
class PIDController:
    def __init__(self, kp=(6.5,6.5,5.0), ki=(0.1,0.1,0.05), kd=(0.3,0.3,0.2)):
        self.kp = np.array(kp)
        self.ki = np.array(ki)
        self.kd = np.array(kd)
        self.integral = np.zeros(3)
        self.prev_error = np.zeros(3)

    def step(self, omega_cmd: np.ndarray, omega_actual: np.ndarray, dt=0.002):
        # omega_cmd/actual: shape (3,) = [roll_rate, pitch_rate, yaw_rate]
        error = omega_cmd - omega_actual
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        torque = self.kp*error + self.ki*self.integral + self.kd*derivative
        self.prev_error = error
        return torque  # → 轉換為 4 個 motor thrusts

    def reset(self):
        self.integral[:] = 0.0
        self.prev_error[:] = 0.0
```

## 環境整合（QuadrotorEnv v4 核心邏輯）
```python
def step(self, action):
    # action shape: (4,) = [f_collective, wx, wy, wz]，range [-1,1]
    f_total = (action[0] + 1) / 2 * self.f_max   # 反歸一化
    omega_cmd = action[1:] * self.omega_max

    # 500Hz inner-loop（每個 RL step 執行 N 次）
    for _ in range(self.inner_loop_steps):         # inner_loop_steps = 8
        omega_actual = self.get_angular_velocity()
        torque = self.pid.step(omega_cmd, omega_actual, dt=self.inner_dt)
        motor_thrusts = self.mixer(f_total, torque) # shape: (4,)
        self.physics_step(motor_thrusts)
```

## Phase 0 Gate Criteria
1. 手動固定 CTBR 指令 [1.0, 0, 0, 0] → hover 500 步不墜毀
2. Step response：ω_cmd 到 steady-state < 50ms
3. 隨機 CTBR 指令 crash rate < 10%（PID 應能穩定大部分指令）
EOF

# ── Skill 3: Experiment Protocol ────────────────────────────────────────────
cat > .claude/skills/experiment-protocol/SKILL.md << 'EOF'
---
name: experiment-protocol
description: WandB 初始化規範、DEV_LOG 失敗記錄模板、Evidence Before Assertions 原則。每次開始新實驗或記錄結果時使用。
---
# 實驗記錄規範 (Evidence Before Assertions)

## WandB 初始化標準模板
```python
import wandb, time, os

def init_wandb(phase: int, experiment_name: str, config: dict):
    run = wandb.init(
        project="vision-dppo-v4",
        name=f"phase{phase}_{experiment_name}_{int(time.time())}",
        tags=[f"phase{phase}", config.get("action_space","CTBR"),
              config.get("model","flow_matching")],
        config=config,
        resume="allow",
    )
    return run
```

## 必須記錄的 Metrics（缺少 = 無效實驗）
```python
# 每個 epoch 必須記錄
wandb.log({
    "train/policy_loss":      loss.item(),
    "train/position_rmse":    rmse,
    "train/crash_rate":       crashes / total,
    "eval/position_rmse":     eval_rmse,   # 每 N epoch
    "eval/crash_rate":        eval_crash,
    "system/inference_latency_ms": latency_ms,
    "epoch": epoch,
})
```

## DEV_LOG.md 失敗記錄模板
```markdown
### [YYYY-MM-DD] Run: {wandb_run_id}
**Phase**: Phase X
**現象**: [具體觀察，必須有數字，例如：crash rate = 100%, RMSE = inf]
**假設**: [可能的根本原因，列出 1-3 個]
**驗證方法**: [如何確認哪個假設正確]
**根本原因**: [確認後的原因]
**修復方案**: [採取的行動]
**結果**: [修復後的數字]
**狀態**: RESOLVED ✅ / OPEN 🔍 / DEPRECATED 🗄️
```

## Evidence Before Assertions 原則
宣稱成功前，必須提供：
- eval episode 的 RMSE 數字（≥ 50 episodes）
- crash_rate = n_crashes / n_episodes（附具體數字）
- WandB run ID（可追溯）
- inference latency（若聲稱達到 real-time）

❌ 禁止：「這應該可以跑」「看起來收斂了」
✅ 要求：「Run phase1_ctbr_001: RMSE=0.087m, crash=0/50, latency=8.2ms」
EOF

# ── Skill 4: Debug PyTorch ───────────────────────────────────────────────────
cat > .claude/skills/debug-pytorch/SKILL.md << 'EOF'
---
name: debug-pytorch
description: PyTorch 常見問題診斷：CUDA OOM、Tensor shape mismatch、梯度消失/爆炸、訓練發散。遇到報錯或訓練異常時使用。
---
# PyTorch Debug Protocol for Vision-DPPO

## CUDA Out of Memory (OOM) 診斷
```python
# 1. 找出哪裡在累積計算圖
# ❌ 錯誤：在 for-loop 中累積 loss
total_loss += loss   # loss 帶著整個計算圖！

# ✅ 正確：detach 後累積
total_loss += loss.item()   # 只保留數值

# 2. 確認 eval 用 no_grad
with torch.no_grad():        # 評估時必須
    pred = model(obs)

# 3. 顯示顯存使用
print(f"GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB / {torch.cuda.max_memory_allocated()/1e9:.2f}GB")
```

## Tensor Shape Mismatch 診斷
```python
# 修改任何 forward() 前，先推演 shape
# 例：Vision Encoder → Policy head
# Input:  obs_img  (B, 3, 84, 84)
# CNN:    features (B, 256)        # after AdaptiveAvgPool
# concat: obs_full (B, 256+15)     # + state obs
# Output: action   (B, 4)          # CTBR

# Debug 工具：在 forward 中加 assert
assert x.shape == (B, 256), f"Expected (B,256), got {x.shape}"
```

## 訓練發散診斷清單
1. Loss = NaN？→ 檢查 log(0) 或除以 0；加 `torch.nan_to_num()`
2. Gradient explosion？→ `torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)`
3. Policy collapse？→ 檢查 KL divergence 是否過大；降低 learning rate
4. RMSE 不下降？→ 確認 action 有正確 unnormalize；確認 obs normalization

## 常用診斷指令（貼進 shell）
```bash
# 顯存使用
nvidia-smi --query-gpu=memory.used,memory.free --format=csv

# 監看訓練進度
tail -f logs/train.log | grep "RMSE\|crash\|loss"

# 快速測試 forward pass（不跑訓練）
python -c "
import torch
from models.flow_policy import FlowPolicy
m = FlowPolicy().cuda()
obs = torch.randn(4, 15).cuda()
a = m.sample(obs)
print('action shape:', a.shape)   # 預期 (4, 4)
"
```

## Flow Matching 特有問題
- v_pred 全為 0？→ 檢查 t sampling 是否有包含 t=0 附近
- Sample 品質差？→ 增加 n_steps；確認 condition 有正確傳入
- ReinFlow KL 爆炸？→ 確認 log_prob 計算用 change-of-variables，非 ε-space
EOF

# ── Skill 5: Jetson Deploy ─────────────────────────────────────────────────
cat > .claude/skills/jetson-deploy/SKILL.md << 'EOF'
---
name: jetson-deploy
description: Jetson Orin Nano 部署流程：TensorRT 轉換、ROS 2 node 撰寫、PX4 MAVLink 接口、latency 測試。Phase 5 使用。
---
# Jetson Orin Nano 部署流程

## Phase 5 Gate Criteria（必須全過）
- [ ] TensorRT engine 轉換成功
- [ ] Inference latency < 30ms（Jetson Orin Nano, 1-step FM）
- [ ] ROS 2 node 發布 CTBR 指令，頻率 > 30Hz
- [ ] PX4 接收到 OFFBOARD 模式指令
- [ ] Real-world hover 成功 > 10 秒

## TensorRT 轉換流程
```python
# Step 1: Export to ONNX（在訓練機執行）
torch.onnx.export(
    model,
    (sample_obs, sample_t),
    "deployment/jetson/flow_policy.onnx",
    input_names=["obs", "t"],
    output_names=["v_pred"],
    dynamic_axes={"obs": {0: "batch"}, "t": {0: "batch"}},
    opset_version=17,
)

# Step 2: Convert to TensorRT（在 Jetson 上執行）
# trtexec --onnx=flow_policy.onnx \
#         --saveEngine=flow_policy.engine \
#         --fp16 \
#         --optShapes=obs:1x15,t:1

# Step 3: Latency 測試
# trtexec --loadEngine=flow_policy.engine --iterations=1000
```

## ROS 2 Node 基礎結構
```python
# deployment/jetson/ros2_policy_node.py
import rclpy
from rclpy.node import Node
from mavros_msgs.msg import AttitudeTarget

class PolicyNode(Node):
    def __init__(self):
        super().__init__("vision_dppo_policy")
        self.pub = self.create_publisher(AttitudeTarget, "/mavros/setpoint_raw/attitude", 10)
        self.timer = self.create_timer(1/30, self.policy_step)  # 30Hz

    def policy_step(self):
        obs = self.get_observation()           # RGB + IMU
        action = self.engine.infer(obs)        # CTBR from TensorRT
        msg = self.ctbr_to_attitude_target(action)
        self.pub.publish(msg)
```

## 安全規則
- NEVER 在未測試的 policy 上嘗試飛行
- 必須先在 sim 完成 Phase 4 才能進入 Phase 5
- 首次真機測試必須有系留繩（tether）
- Kill switch 必須在手邊
EOF

success "所有 Skills 建立完成（5 個）"

# =============================================================================
header "4 / 7  .claudeignore"
# =============================================================================

cat > .claudeignore << 'EOF'
# Model weights & checkpoints
*.pt
*.pth
*.ckpt
*.safetensors
*.engine
*.onnx
/checkpoints/
/saved_models/

# Datasets
/datasets/
/data/*.h5
/data/*.hdf5
*.h5
*.hdf5
*.npz

# WandB & Logs
/wandb/
/logs/
/outputs/
/runs/
*.log
*.jsonl

# Isaac Gym / Sim assets
/assets/
*.urdf
*.mjcf

# Videos & Images
*.mp4
*.avi
*.gif
*.png
*.jpg
/videos/
/renders/

# Python cache
__pycache__/
*.pyc
*.pyo
.pytest_cache/
*.egg-info/

# Jupyter
.ipynb_checkpoints/
*.ipynb

# Virtual env
.venv/
venv/
env/
EOF

success ".claudeignore 建立完成"

# =============================================================================
header "5 / 7  Hooks（settings.json）"
# =============================================================================

cat > .claude/settings.json << 'EOF'
{
  "hooks": {
    "PreToolUse": [
      {
        "description": "保護 v3 歷史檔案",
        "script": "if echo \"$CLAUDE_FILE_PATH\" | grep -qE '_v3\\.(py|yaml|md)$'; then echo '⛔ 禁止修改 v3 歷史檔案！請建立 v4 版本。' && exit 1; fi"
      },
      {
        "description": "防止未命名的長訓練",
        "script": "if echo \"$CLAUDE_COMMAND\" | grep -qE 'train.*--epochs [0-9]{3,}'; then echo '⚠️  長訓練請確認 WandB 已初始化，繼續？(y/N): ' && read -r ans && [ \"$ans\" = 'y' ] || exit 1; fi"
      }
    ],
    "PostToolUse": [
      {
        "description": "Python 語法快速檢查",
        "script": "if echo \"$CLAUDE_FILE_PATH\" | grep -q '\\.py$'; then python -m py_compile \"$CLAUDE_FILE_PATH\" 2>&1 | head -5; fi"
      }
    ],
    "Stop": [
      {
        "description": "Session 結束 checklist",
        "script": "echo '\n📋 Session 結束 Checklist：\n  1. 實驗結果有在 WandB 記錄嗎？\n  2. 需要更新 docs/DEV_LOG.md 嗎？\n  3. 當前 Phase gate criteria 完成幾項？\n  4. 有未 commit 的重要變更嗎？'"
      }
    ]
  }
}
EOF

success "Hooks (settings.json) 建立完成"

# =============================================================================
header "6 / 7  文件框架建立"
# =============================================================================

# DEV_LOG.md
cat > docs/DEV_LOG.md << 'EOF'
# Vision-DPPO v4.0 — 開發日誌

> 失敗記錄 NEVER 刪除。只標記 RESOLVED ✅ / OPEN 🔍 / DEPRECATED 🗄️

## 記錄模板
```
### [YYYY-MM-DD] Run: {wandb_run_id}
**Phase**: Phase X
**現象**: [具體觀察，必須有數字]
**假設**: [可能原因]
**根本原因**: [確認後]
**修復方案**: [行動]
**結果**: [修復後數字]
**狀態**: RESOLVED ✅
```

---

### [2024-XX-XX] 歷史：v3.3 100% Crash Rate
**Phase**: Phase 3c_v33
**現象**: crash rate = 100%，所有 episode 在 < 5 步內墜毀
**根本原因**: SRT 動作空間在 12-25Hz 輸出 4D motor thrust，無內迴路穩定，開迴路不穩定
**修復方案**: RESEARCH_PLAN_v4.0，切換至 CTBR + PID inner-loop
**狀態**: DEPRECATED 🗄️（v3 保留為歷史）

### [2024-XX-XX] 歷史：ε-space 放大問題
**Phase**: Phase 3d
**現象**: 單步蒸餾品質嚴重下降，RMSE 從 0.10m 退化至 0.26m
**根本原因**: DDPM ε-space 在 t=T-1 放大係數 1/√ᾱ ≈ 64×，單步誤差被放大
**修復方案**: 改用 Flow Matching（無放大問題，原生支援 1-step）
**狀態**: DEPRECATED 🗄️（v4 改用 Flow Matching）
EOF

# GATE_CRITERIA.md
cat > docs/GATE_CRITERIA.md << 'EOF'
# Phase Gate Criteria — Vision-DPPO v4.0

未達標 = 不進入下一 phase（無例外）

## Phase 0: CTBR 環境驗證
- [ ] hover 500 步，crash = 0（手動固定指令 [1,0,0,0]）
- [ ] 隨機 CTBR 指令，crash rate < 10%（1000 steps）
- [ ] Step response：ω_cmd → steady-state < 50ms

## Phase 1: PPO Expert (CTBR)
- [ ] RMSE < 0.10m（評估 50 episodes）
- [ ] Crash rate = 0/50
- [ ] Action smoothness: mean |Δaction| < 0.3
- [ ] WandB run ID: _______________

## Phase 2: Data Collection
- [ ] 1000 episodes 收集完成
- [ ] Data size ~ 4GB
- [ ] IMU 資料完整（無 NaN）
- [ ] Depth 資料完整

## Phase 3a: Flow Matching BC
- [ ] Training loss 穩定下降（無 NaN）
- [ ] Eval RMSE < 0.12m（sim，50 episodes）
- [ ] Inference latency < 10ms（單步，sim）
- [ ] WandB run ID: _______________

## Phase 3b: ReinFlow RL Fine-tuning
- [ ] RMSE < 0.08m
- [ ] Crash rate < 5%（sim）
- [ ] KL divergence stable（無爆炸）
- [ ] WandB run ID: _______________

## Phase 5: Jetson Deployment（必做）
- [ ] ONNX export 成功
- [ ] TensorRT engine 轉換成功
- [ ] Inference latency < 30ms（Jetson Orin Nano）
- [ ] ROS 2 node 發布頻率 > 30Hz
- [ ] Real-world hover > 10 秒 ✈️
EOF

success "文件框架建立完成"

# =============================================================================
header "7 / 7  CLAUDE.local.md（個人設定，已加入 .gitignore）"
# =============================================================================

cat > CLAUDE.local.md << 'EOF'
# 本機環境（不 commit，個人覆蓋設定）

## 硬體環境
- GPU: RTX 3090, CUDA 12.x
- 訓練機 hostname: [填入你的機器名稱]
- WandB project: vision-dppo-v4
- WandB entity: [填入你的 WandB username]

## 路徑設定
- Isaac Gym path: /path/to/isaacgym
- Checkpoint save: /path/to/checkpoints
- Dataset path: /path/to/data

## 部署（Phase 5）
- Jetson IP: [填入 Jetson 的 IP]
- Jetson user: [填入 username]
- PX4 機型: [填入機型]

## 當前狀態
- 進行中 Phase: Phase 0
- 當前 blocker: [填入當前問題]
- 上次 WandB run: [填入 run ID]
EOF

# 加入 .gitignore
if ! grep -q "CLAUDE.local.md" .gitignore 2>/dev/null; then
  echo -e "\n# Claude Code 個人設定\nCLAUDE.local.md\n.claude/settings.local.json" >> .gitignore
fi

success "CLAUDE.local.md 建立完成（已加入 .gitignore）"

# =============================================================================
# 完成摘要
# =============================================================================

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║         Vision-DPPO Claude Code 設定完成！                  ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "建立的檔案："
echo -e "  ${BLUE}CLAUDE.md${NC}                          — 根目錄系統提示"
echo -e "  ${BLUE}.claudeignore${NC}                      — 排除 weights/datasets/logs"
echo -e "  ${BLUE}.claude/settings.json${NC}              — Hooks（v3保護/Python語法/Session checklist）"
echo -e "  ${BLUE}.claude/skills/flow-matching/${NC}       — Flow Matching 數學與實作"
echo -e "  ${BLUE}.claude/skills/ctbr-control/${NC}        — CTBR + PID inner-loop"
echo -e "  ${BLUE}.claude/skills/experiment-protocol/${NC} — WandB 規範 + 失敗記錄模板"
echo -e "  ${BLUE}.claude/skills/debug-pytorch/${NC}       — OOM / shape / 發散診斷"
echo -e "  ${BLUE}.claude/skills/jetson-deploy/${NC}       — TensorRT + ROS 2 部署"
echo -e "  ${BLUE}docs/DEV_LOG.md${NC}                    — 失敗記錄日誌（含歷史記錄）"
echo -e "  ${BLUE}docs/GATE_CRITERIA.md${NC}              — Phase gate 通過標準清單"
echo -e "  ${BLUE}CLAUDE.local.md${NC}                    — 個人環境設定（已 gitignore）"
echo ""
echo -e "${YELLOW}下一步：${NC}"
echo -e "  1. 在 Claude Code 執行："
echo -e "     ${GREEN}/plugin install superpowers@claude-plugins-official${NC}"
echo -e "     ${GREEN}claude skills add abagames/criticalthink${NC}"
echo -e "     ${GREEN}claude skills add K-Dense-AI/claude-scientific-skills${NC}"
echo -e "  2. 填寫 CLAUDE.local.md 的個人設定（GPU/路徑/Jetson IP）"
echo -e "  3. 開始新 session，輸入 ${GREEN}/brainstorm${NC} 讓 Superpowers 引導 Phase 0"
echo ""
