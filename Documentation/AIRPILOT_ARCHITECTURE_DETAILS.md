# AirPilot 神經網路架構與獎勵函數詳細對比與實作指南

## 📋 概述

本文檔基於 AirPilot 論文的詳細架構資訊，提供與當前 DPPO_PID_controller 專案的對比分析，並給出具體的實作建議。

---

## 一、神經網路架構對比

### 1.1 架構規格對比

| 項目 | DPPO_PID_controller (當前) | AirPilot (論文) | 建議 |
|------|---------------------------|----------------|------|
| **共享層架構** | [128, 128] | [64, 64] | ✅ 可選：添加 [64, 64] 快速模式 |
| **Actor Head** | 3-dim (Kp, Ki, Kd) | 9-dim (Kp, Ki, Kd × 3軸) | ✅ 當前設計正確（單軸） |
| **Critic Head** | 1-dim (V(s)) | 1-dim (V(s)) | ✅ 一致 |
| **激活函數** | ReLU (推測) | ReLU (推測) | ✅ 一致 |
| **參數數量** | ~15K | ~10K | ✅ 當前稍大，但可接受 |
| **共享參數** | ✅ 是 (SB3 預設) | ✅ 是 | ✅ 一致 |

### 1.2 Stable-Baselines3 的共享架構

**重要發現**：SB3 的 `MlpPolicy` **預設就是共享參數架構**！

```python
# SB3 內部實現（簡化版）
class ActorCriticPolicy:
    def __init__(self, ...):
        # 共享特徵提取層
        self.features_extractor = ...
        
        # 分離的輸出頭
        self.action_net = ...  # Actor head
        self.value_net = ...   # Critic head
```

**當前配置已經正確**：
```python
# train.py 中的配置
policy_kwargs=dict(
    net_arch=[dict(pi=policy_net, vf=value_net)]
)
# SB3 會自動共享前幾層（如果 pi 和 vf 相同）
```

### 1.3 具體配置建議

#### 選項 1: 保持當前架構（推薦）

```yaml
# config.yaml
training:
  policy_net_arch: [128, 128]  # 當前設定
  value_net_arch: [128, 128]   # 當前設定
```

**優點**：
- 更大的容量，可能學習更複雜的策略
- 已經驗證可工作

#### 選項 2: 添加 AirPilot 風格的快速模式

```yaml
# config.yaml
training:
  # 標準模式（當前）
  policy_net_arch: [128, 128]
  value_net_arch: [128, 128]
  
  # 快速驗證模式（參考 AirPilot）
  quick_test_mode: false
  quick_test_net_arch: [64, 64]  # AirPilot 的架構
  quick_test_timesteps: 20000
```

**實作**（在 `train.py` 中）：
```python
# 檢查快速模式
if config['training'].get('quick_test_mode', False):
    net_arch = config['training']['quick_test_net_arch']
    total_timesteps = config['training']['quick_test_timesteps']
    print("⚠️ 快速訓練模式（AirPilot 風格）")
else:
    net_arch = config['training']['policy_net_arch']
    total_timesteps = config['training']['total_timesteps']

# 創建模型（SB3 自動共享參數）
model = PPO(
    policy="MlpPolicy",
    env=train_env,
    policy_kwargs=dict(
        net_arch=[dict(pi=net_arch, vf=net_arch)]  # 相同 = 共享
    ),
    # ... 其他參數
)
```

---

## 二、獎勵函數設計對比

### 2.1 核心差異

| 特性 | DPPO_PID_controller | AirPilot | 適用場景 |
|------|---------------------|----------|---------|
| **類型** | 連續型（每步都有獎勵） | 任務完成型（到達目標後給大獎勵） | 不同 |
| **主要獎勵** | `-λ₁e² - λ₂ẋ² - λ₃u² - λ₄max(0,e·ė)` | `e^(EffectiveSpeed × 10)` | 不同 |
| **穩定性要求** | 隱式（通過懲罰振盪） | 顯式（50 timesteps 穩定） | 不同 |
| **適用任務** | 連續跟蹤任務 | 點對點導航任務 | 不同 |

### 2.2 當前獎勵函數分析

```python
# dppo_pid_env.py 中的當前實現
reward = -λ_error * error²           # 追蹤誤差
        - λ_velocity * velocity²     # 速度懲罰
        - λ_control * u²             # 控制努力
        - λ_overshoot * max(0, e·ė)  # 超調懲罰
```

**優點**：
- ✅ 每步都有信號，適合連續控制
- ✅ 多目標平衡（精度、穩定性、效率）
- ✅ 適合當前任務（連續跟蹤）

**缺點**：
- ⚠️ 可能不夠激勵快速完成任務
- ⚠️ 沒有明確的「任務完成」概念

### 2.3 AirPilot 獎勵函數分析

```python
# AirPilot 的獎勵邏輯（簡化版）
if stable_counter >= 50:  # 穩定 50 timesteps
    effective_speed = distance / (0.04 * (timestep - 50))
    reward = np.exp(effective_speed * 10)  # 指數獎勵
    # 重置任務
else:
    reward = -np.linalg.norm(position_error)  # 接近獎勵
```

**優點**：
- ✅ 強烈激勵快速完成（指數放大）
- ✅ 明確的穩定性要求
- ✅ 適合點對點導航

**缺點**：
- ⚠️ 不適合連續跟蹤任務
- ⚠️ 訓練初期可能沒有獎勵信號（無法達到穩定）

### 2.4 混合獎勵函數設計（建議）

**目標**：結合兩者優點，支持兩種任務模式

#### 實作方案：在 `config.yaml` 添加獎勵類型選項

```yaml
# config.yaml
reward:
  # 獎勵函數類型
  reward_type: "continuous"  # "continuous" 或 "task_completion"
  
  # 連續型獎勵（當前實現）
  lambda_error: 5.0
  lambda_velocity: 0.5
  lambda_control: 0.01
  lambda_overshoot: 0.2
  
  # 任務完成型獎勵（AirPilot 風格，可選）
  task_completion:
    stable_threshold: 0.1      # 穩定閾值（米）
    stable_timesteps: 50       # 穩定時間步數
    effective_speed_multiplier: 10.0  # 有效速度乘數
    distance_scale: 1.0         # 距離縮放
```

#### 在環境中實現

```python
# dppo_pid_env.py 修改
def _calculate_reward(self, error: float, error_dot: float, u: float) -> float:
    """
    計算獎勵（支持兩種模式）
    """
    reward_type = self.config['reward'].get('reward_type', 'continuous')
    
    if reward_type == 'task_completion':
        return self._calculate_task_completion_reward(error)
    else:
        return self._calculate_continuous_reward(error, error_dot, u)

def _calculate_continuous_reward(self, error: float, error_dot: float, u: float) -> float:
    """當前實現的連續型獎勵"""
    error_penalty = -self.lambda_error * error**2
    velocity_penalty = -self.lambda_velocity * self.x_dot**2
    control_penalty = -self.lambda_control * u**2
    overshoot_penalty = -self.lambda_overshoot * max(0, error * error_dot)
    return error_penalty + velocity_penalty + control_penalty + overshoot_penalty

def _calculate_task_completion_reward(self, error: float) -> float:
    """
    任務完成型獎勵（AirPilot 風格）
    
    注意：需要額外的狀態追蹤
    """
    # 初始化狀態（在 __init__ 或 reset 中）
    if not hasattr(self, 'task_start_pos'):
        self.task_start_pos = self.x
        self.task_target_pos = self.reference
        self.task_distance = abs(self.task_target_pos - self.task_start_pos)
        self.task_stable_counter = 0
        self.task_timestep = 0
    
    self.task_timestep += 1
    abs_error = abs(error)
    
    # 檢查穩定性
    stable_threshold = self.config['reward']['task_completion']['stable_threshold']
    if abs_error < stable_threshold:
        self.task_stable_counter += 1
    else:
        self.task_stable_counter = 0
    
    # 計算獎勵
    stable_timesteps = self.config['reward']['task_completion']['stable_timesteps']
    if self.task_stable_counter >= stable_timesteps:
        # 達到穩定 - 計算有效速度
        time_taken = self.dt_outer * (self.task_timestep - stable_timesteps)
        if time_taken > 0:
            effective_speed = self.task_distance / time_taken
            multiplier = self.config['reward']['task_completion']['effective_speed_multiplier']
            reward = np.exp(effective_speed * multiplier)
            
            # 重置任務（生成新目標）
            self.reference = self.np_random.uniform(self.r_min, self.r_max)
            self.task_start_pos = self.x
            self.task_target_pos = self.reference
            self.task_distance = abs(self.task_target_pos - self.task_start_pos)
            self.task_stable_counter = 0
            self.task_timestep = 0
            
            return reward
        else:
            return 0.0
    else:
        # 未達穩定 - 接近獎勵
        return -abs_error
```

**注意**：任務完成型獎勵需要修改環境邏輯，可能影響現有功能。建議作為**實驗性功能**。

---

## 三、訓練指標可視化

### 3.1 AirPilot 的訓練指標

論文中的 Fig.14-16 展示了三個關鍵指標：
- **Effective Speed** vs Training Timesteps
- **Settling Time** vs Training Timesteps
- **Overshoot** vs Training Timesteps

### 3.2 實作訓練指標追蹤

#### 步驟 1: 創建指標追蹤工具

```python
# utils/training_metrics.py
import numpy as np
from typing import Dict, List
import json
import os

class TrainingMetricsTracker:
    """
    追蹤訓練過程中的關鍵指標（參考 AirPilot Fig.14-16）
    """
    
    def __init__(self, log_dir: str = "./training_metrics/"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # 指標存儲
        self.timesteps = []
        self.effective_speeds = []
        self.settling_times = []
        self.overshoots = []
        self.mean_errors = []
        
    def log_episode(self, 
                    timestep: int,
                    episode_history: Dict,
                    target_positions: List[float] = None):
        """
        記錄一個回合的指標
        
        Args:
            timestep: 當前訓練步數
            episode_history: 回合歷史（從 env.get_history()）
            target_positions: 目標位置列表（用於計算 settling time）
        """
        if not episode_history or not episode_history.get('position'):
            return
        
        positions = np.array(episode_history['position'])
        references = np.array(episode_history['reference'])
        errors = np.array(episode_history['error'])
        times = np.array(episode_history['time'])
        
        # 1. 計算平均誤差
        mean_error = np.mean(np.abs(errors))
        self.mean_errors.append(mean_error)
        
        # 2. 計算有效速度（如果適用）
        # 注意：這需要任務完成型獎勵，否則為 NaN
        effective_speed = self._calculate_effective_speed(
            positions, references, times
        )
        self.effective_speeds.append(effective_speed)
        
        # 3. 計算穩定時間
        settling_time = self._calculate_settling_time(
            errors, times, threshold=0.02  # 2% 誤差
        )
        self.settling_times.append(settling_time)
        
        # 4. 計算超調
        overshoot = self._calculate_overshoot(
            positions, references
        )
        self.overshoots.append(overshoot)
        
        # 5. 記錄時間步
        self.timesteps.append(timestep)
    
    def _calculate_effective_speed(self, positions, references, times):
        """
        計算有效速度（AirPilot Eq.9）
        
        注意：這需要任務完成型場景
        """
        # 簡化實現：計算平均速度
        if len(positions) < 2:
            return np.nan
        
        distances = np.diff(positions)
        time_diffs = np.diff(times)
        
        if np.sum(time_diffs) > 0:
            avg_speed = np.sum(np.abs(distances)) / np.sum(time_diffs)
            return avg_speed
        return np.nan
    
    def _calculate_settling_time(self, errors, times, threshold=0.02):
        """
        計算穩定時間（達到 ±threshold 誤差內的時間）
        """
        abs_errors = np.abs(errors)
        target_error = threshold * np.max(np.abs(errors)) if np.max(np.abs(errors)) > 0 else threshold
        
        # 找到最後一次超過閾值的時間
        above_threshold = abs_errors > target_error
        if np.any(above_threshold):
            last_above_idx = np.where(above_threshold)[0][-1]
            if last_above_idx < len(times) - 1:
                return times[last_above_idx + 1] - times[0]
        
        return times[-1] - times[0]  # 整個回合時間
    
    def _calculate_overshoot(self, positions, references):
        """
        計算超調量
        """
        errors = positions - references
        max_overshoot = np.max(np.abs(errors))
        return max_overshoot
    
    def save(self, filename: str = "training_metrics.json"):
        """保存指標到 JSON"""
        data = {
            'timesteps': self.timesteps,
            'effective_speeds': [float(x) if not np.isnan(x) else None for x in self.effective_speeds],
            'settling_times': [float(x) if not np.isnan(x) else None for x in self.settling_times],
            'overshoots': [float(x) if not np.isnan(x) else None for x in self.overshoots],
            'mean_errors': [float(x) for x in self.mean_errors]
        }
        
        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✓ 訓練指標已保存到: {filepath}")
    
    def load(self, filename: str = "training_metrics.json"):
        """從 JSON 載入指標"""
        filepath = os.path.join(self.log_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
            self.timesteps = data['timesteps']
            self.effective_speeds = [x if x is not None else np.nan for x in data['effective_speeds']]
            self.settling_times = [x if x is not None else np.nan for x in data['settling_times']]
            self.overshoots = [x if x is not None else np.nan for x in data['overshoots']]
            self.mean_errors = data['mean_errors']
            return True
        return False
```

#### 步驟 2: 創建可視化函數

```python
# utils/visualization.py 添加
import matplotlib.pyplot as plt
import numpy as np
from utils.training_metrics import TrainingMetricsTracker

def plot_airpilot_style_metrics(metrics_tracker: TrainingMetricsTracker, 
                                output_dir: str = "./training_metrics/"):
    """
    繪製 AirPilot 風格的訓練指標圖表（Fig.14-16）
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    timesteps = np.array(metrics_tracker.timesteps)
    
    # Fig.14: Effective Speed vs Timesteps
    effective_speeds = np.array(metrics_tracker.effective_speeds)
    valid_mask = ~np.isnan(effective_speeds)
    
    axes[0].plot(timesteps[valid_mask], effective_speeds[valid_mask], 
                 'b-', linewidth=2, label='Effective Speed')
    axes[0].axhline(y=0.92, color='r', linestyle='--', 
                    label='Fine-tuned PID baseline', linewidth=2)
    axes[0].set_xlabel('Training Timesteps', fontsize=12)
    axes[0].set_ylabel('Effective Speed (m/s)', fontsize=12)
    axes[0].set_title('Effective Speed vs Training Timesteps', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Fig.15: Settling Time vs Timesteps
    settling_times = np.array(metrics_tracker.settling_times)
    valid_mask = ~np.isnan(settling_times)
    
    axes[1].plot(timesteps[valid_mask], settling_times[valid_mask], 
                 'g-', linewidth=2, label='Settling Time')
    axes[1].set_xlabel('Training Timesteps', fontsize=12)
    axes[1].set_ylabel('Settling Time (s)', fontsize=12)
    axes[1].set_title('Settling Time vs Training Timesteps', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Fig.16: Overshoot vs Timesteps
    overshoots = np.array(metrics_tracker.overshoots)
    valid_mask = ~np.isnan(overshoots)
    
    axes[2].plot(timesteps[valid_mask], overshoots[valid_mask], 
                 'r-', linewidth=2, label='Overshoot')
    axes[2].set_xlabel('Training Timesteps', fontsize=12)
    axes[2].set_ylabel('Overshoot (m)', fontsize=12)
    axes[2].set_title('Overshoot vs Training Timesteps', fontsize=14, fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存
    filepath = os.path.join(output_dir, 'airpilot_style_metrics.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ 訓練指標圖表已保存到: {filepath}")
    plt.close()
```

#### 步驟 3: 在訓練腳本中集成

```python
# train.py 修改
from utils.training_metrics import TrainingMetricsTracker

def train(config_path: str = "config.yaml", ...):
    # ... 現有代碼 ...
    
    # 創建指標追蹤器
    metrics_tracker = TrainingMetricsTracker()
    
    # 自定義回調（在 EvalCallback 中）
    class MetricsCallback(EvalCallback):
        def __init__(self, *args, metrics_tracker=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.metrics_tracker = metrics_tracker
        
        def _on_step(self) -> bool:
            # 在評估時記錄指標
            if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
                # 獲取評估環境的歷史
                # ... 實現細節 ...
                pass
            return super()._on_step()
    
    # 使用自定義回調
    eval_callback = MetricsCallback(
        eval_env,
        metrics_tracker=metrics_tracker,
        # ... 其他參數
    )
    
    # 訓練後保存指標
    model.learn(...)
    metrics_tracker.save()
    
    # 繪製圖表
    from utils.visualization import plot_airpilot_style_metrics
    plot_airpilot_style_metrics(metrics_tracker)
```

---

## 四、實施優先級建議

### 🔴 高優先級（立即實施）

1. **添加 [64, 64] 快速訓練模式**
   - 時間：30 分鐘
   - 價值：快速驗證，對比 AirPilot 性能

2. **添加訓練指標可視化**
   - 時間：2-3 小時
   - 價值：更好的訓練監控

### 🟡 中優先級（短期考慮）

3. **實現任務完成型獎勵（實驗性）**
   - 時間：3-4 小時
   - 價值：對比不同獎勵函數的效果
   - **注意**：需要謹慎測試，可能影響現有功能

### 🟢 低優先級（長期考慮）

4. **完整實現 AirPilot 風格的環境**
   - 僅當需要點對點導航任務時

---

## 五、關鍵要點總結

### 神經網路架構

✅ **當前配置已經正確**：SB3 預設共享參數  
✅ **可選優化**：添加 [64, 64] 快速模式  
✅ **無需大幅修改**：架構設計已經符合最佳實踐

### 獎勵函數

✅ **當前設計適合連續跟蹤任務**  
⚠️ **AirPilot 設計適合點對點導航**  
💡 **建議**：保持當前設計，任務完成型作為實驗性功能

### 訓練指標

✅ **建議添加**：Effective Speed, Settling Time, Overshoot 追蹤  
✅ **價值**：更好的訓練監控和對比分析

---

**文件版本**：1.0  
**建立日期**：2025-01-XX  
**最後更新**：2025-01-XX

