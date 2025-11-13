# AirPilot vs DPPO_PID_controller 比較分析與改進建議

## 📋 執行摘要

本文檔比較了 **AirPilot PPO-PID 控制器計畫書** 與 **DPPO_PID_controller 專案**的架構設計，識別可借鑑的設計模式，並提供具體的改進建議以加速開發進程。

---

## 一、核心差異對比

### 1.1 算法選擇

| 項目 | DPPO_PID_controller | AirPilot |
|------|---------------------|----------|
| **核心算法** | DPPO (Diffusion Policy Policy Optimization) | 標準 PPO |
| **創新點** | 擴散模型生成動作 + PPO 優化 | 非線性 PID + 標準 PPO |
| **複雜度** | 高（需要實現擴散模型） | 中（直接使用 SB3） |
| **訓練效率** | 待驗證 | 20,000 timesteps（高效） |
| **當前狀態** | Phase 3 骨架實現 | 完整實作計畫 |

**建議**：
- ✅ **保留 DPPO 作為長期目標**（核心研究價值）
- ✅ **短期可先實現標準 PPO 版本**作為基準對比
- ✅ 借鑑 AirPilot 的訓練效率優化策略

### 1.2 系統規模

| 項目 | DPPO_PID_controller | AirPilot |
|------|---------------------|----------|
| **當前階段** | Phase 1: 單軸系統 | 3D 無人機（完整系統） |
| **動作空間** | 3D (Kp, Ki, Kd) | 9D (Kp, Ki, Kd × 3軸) |
| **觀測空間** | 9D | 9D (但結構不同) |
| **控制頻率** | 20 Hz RL + 200 Hz PID | 25 Hz 統一頻率 |

**建議**：
- ✅ 當前單軸設計是正確的漸進式開發策略
- ✅ 可參考 AirPilot 的 3D 擴展方案作為 Phase 4 的設計指南

---

## 二、可借鑑的設計模式

### 2.1 非線性 PID 控制器設計 ⭐⭐⭐

**AirPilot 的創新點**：
```python
# 正規化速度輸出（Eq.7）
normalized_velocity = velocity / (np.abs(velocity) + 1.0)
normalized_velocity = np.clip(normalized_velocity, -max_velocity, max_velocity)
```

**當前專案**：
- 使用標準線性 PID
- 控制輸入直接飽和到 [-10, 10]

**改進建議**：
1. **可選實現非線性 PID 模組**（作為實驗性功能）
   - 創建 `controllers/nonlinear_pid.py`
   - 在 `config.yaml` 中添加 `pid_type: "linear" | "nonlinear"` 選項
   - 保留向後兼容性

2. **實作位置**：
   - 在 `dppo_pid_env.py` 中添加 `NonlinearPID` 類別（可選）
   - 或創建獨立的 `controllers/` 目錄

### 2.2 獎勵函數設計 ⭐⭐

**AirPilot 的設計**：
```python
# 基於 Effective Speed 的獎勵（Eq.8-10）
if stable_counter >= 50:  # 穩定到達目標
    effective_speed = distance / time_taken
    reward = np.exp(effective_speed * 10)
else:
    reward = -np.linalg.norm(position_error)  # 持續接近
```

**當前專案**：
```python
# 多目標懲罰型獎勵
reward = -λ_error * error² - λ_velocity * velocity² - λ_control * u² - λ_overshoot * max(0, e·ė)
```

**比較分析**：
- ✅ **當前設計更適合連續控制任務**（每步都有信號）
- ✅ AirPilot 的設計更適合**任務完成型**場景（到達目標後重置）
- ⚠️ 當前設計已經很完善，**不需要大幅修改**

**建議**：
- 可選：在 `config.yaml` 中添加 `reward_type: "continuous" | "task_completion"` 選項
- 保留當前設計作為默認（更通用）

### 2.3 訓練效率優化 ⭐⭐⭐

**AirPilot 的優勢**：
- 僅需 **20,000 timesteps**（約 1.5 小時）
- 使用較小的網路架構 `[64, 64]`
- 明確的超參數設定

**當前專案**：
- 設定 `total_timesteps: 5,000,000`（但 config 中已改為 100,000）
- 網路架構 `[128, 128]`

**改進建議**：

1. **添加快速訓練模式**：
```yaml
# config.yaml 新增
training:
  # 快速驗證模式（參考 AirPilot）
  quick_test_mode: false
  quick_test_timesteps: 20000
  quick_test_net_arch: [64, 64]
```

2. **實現訓練階段切換**：
```python
# train.py 修改
if config['training'].get('quick_test_mode', False):
    total_timesteps = config['training']['quick_test_timesteps']
    net_arch = config['training']['quick_test_net_arch']
else:
    total_timesteps = config['training']['total_timesteps']
    net_arch = config['training']['policy_net_arch']
```

### 2.4 模組化架構設計 ⭐⭐⭐

**AirPilot 的目錄結構**：
```
airpilot_ppo/
├── envs/
│   └── drone_env.py
├── controllers/
│   └── nonlinear_pid.py
├── utils/
│   ├── reward_functions.py
│   └── visualization.py
└── configs/
    └── training_config.yaml
```

**當前專案結構**：
```
DPPO_PID_controller/
├── dppo_pid_env.py
├── train.py
├── evaluate.py
├── demo.py
└── config.yaml
```

**改進建議**：

1. **重構為模組化結構**（向後兼容）：
```
DPPO_PID_controller/
├── envs/
│   ├── __init__.py
│   └── dppo_pid_env.py          # 移動現有檔案
├── controllers/
│   ├── __init__.py
│   ├── linear_pid.py            # 提取 PID 邏輯
│   └── nonlinear_pid.py          # 新增（可選）
├── utils/
│   ├── __init__.py
│   ├── reward_functions.py      # 提取獎勵函數
│   └── visualization.py          # 提取可視化
├── train.py
├── evaluate.py
├── demo.py
└── config.yaml
```

2. **實施步驟**（零破壞性）：
   - Step 1: 創建新目錄結構
   - Step 2: 移動檔案並更新 import
   - Step 3: 添加 `__init__.py` 保持向後兼容
   - Step 4: 測試所有腳本仍可運行

### 2.5 可視化改進 ⭐⭐

**AirPilot 的特色**：
- **PID Gains vs Position Error** 圖表（Fig.17）
- 展示增益如何隨誤差自適應調整

**當前專案**：
- 已有 PID Gains vs Time 圖表
- 缺少 Gains vs Error 的關聯分析

**改進建議**：

在 `evaluate.py` 中添加新圖表：

```python
def plot_gains_vs_error(history, output_dir):
    """
    繪製 PID 增益 vs 位置誤差（參考 AirPilot Fig.17）
    """
    error = np.array(history['error'])
    kp = np.array(history['kp'])
    ki = np.array(history['ki'])
    kd = np.array(history['kd'])
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].scatter(np.abs(error), kp, alpha=0.5, s=10)
    axes[0].set_xlabel('|Position Error|')
    axes[0].set_ylabel('Kp')
    axes[0].set_title('Kp vs Position Error')
    axes[0].grid(True, alpha=0.3)
    
    # 類似地繪製 Ki 和 Kd
    # ...
```

---

## 三、具體實施建議

### 3.1 優先級排序

#### 🔴 高優先級（立即實施）

1. **添加快速訓練模式**
   - 檔案：`config.yaml`, `train.py`
   - 時間：30 分鐘
   - 價值：快速驗證環境和訓練流程

2. **提取 PID 控制器為獨立模組**
   - 檔案：新建 `controllers/linear_pid.py`
   - 時間：1 小時
   - 價值：提高代碼可維護性，為未來擴展做準備

3. **添加 Gains vs Error 可視化**
   - 檔案：`evaluate.py`
   - 時間：30 分鐘
   - 價值：更好的性能分析

#### 🟡 中優先級（短期實施）

4. **模組化目錄重構**
   - 檔案：整個專案結構
   - 時間：2-3 小時
   - 價值：提高專業度和可擴展性

5. **實現非線性 PID（實驗性）**
   - 檔案：`controllers/nonlinear_pid.py`
   - 時間：2 小時
   - 價值：對比實驗，驗證設計選擇

#### 🟢 低優先級（長期考慮）

6. **ROS/MAVROS 整合**（僅當需要實體測試時）
7. **6-DOF 擴展**（Phase 4 計畫）

### 3.2 實施檢查清單

#### Phase 1: 快速改進（本週）

- [ ] 在 `config.yaml` 添加 `quick_test_mode` 選項
- [ ] 修改 `train.py` 支持快速訓練模式
- [ ] 在 `evaluate.py` 添加 `plot_gains_vs_error()` 函數
- [ ] 測試快速訓練模式（20,000 timesteps）

#### Phase 2: 模組化重構（下週）

- [ ] 創建 `controllers/` 目錄
- [ ] 提取 PID 邏輯到 `controllers/linear_pid.py`
- [ ] 創建 `utils/` 目錄
- [ ] 提取可視化函數到 `utils/visualization.py`
- [ ] 更新所有 import 語句
- [ ] 測試所有腳本仍可運行

#### Phase 3: 實驗性功能（可選）

- [ ] 實現 `controllers/nonlinear_pid.py`
- [ ] 在環境中添加 PID 類型選擇
- [ ] 對比實驗：線性 vs 非線性 PID

---

## 四、關鍵設計決策對比

### 4.1 控制架構

| 設計決策 | DPPO_PID_controller | AirPilot | 建議 |
|---------|---------------------|----------|------|
| **雙層控制** | ✅ 20 Hz RL + 200 Hz PID | ❌ 25 Hz 統一 | ✅ **保留當前設計**（更符合實際控制系統） |
| **時間尺度分離** | ✅ 明確分離 | ❌ 無分離 | ✅ **優勢設計**，無需改變 |

### 4.2 觀測空間設計

**DPPO_PID_controller**：
```
[error, error_dot, integral, position, velocity, reference, Kp, Ki, Kd]
```

**AirPilot**：
```
[PE_x, PE_y, PE_z, dPE_x, dPE_y, dPE_z, ∫PE_x, ∫PE_y, ∫PE_z]
```

**分析**：
- ✅ 當前設計**包含當前增益**，使智能體能學習相對調整
- ✅ AirPilot 設計更簡潔，但缺少增益資訊
- **建議**：**保留當前設計**（更適合自適應控制）

### 4.3 動作空間設計

**DPPO_PID_controller**：
- 直接輸出絕對增益值 `[Kp, Ki, Kd]`
- 範圍：`[0, K_max]`

**AirPilot**：
- 直接輸出絕對增益值（3D 擴展到 9D）
- 範圍：每軸獨立設定

**分析**：
- ✅ 兩者設計一致
- ✅ 當前設計已足夠

---

## 五、程式碼改進範例

### 5.1 快速訓練模式實現

```python
# config.yaml 新增
training:
  # 快速驗證模式（參考 AirPilot 的 20,000 timesteps）
  quick_test_mode: false
  quick_test_timesteps: 20000
  quick_test_net_arch: [64, 64]
  
  # 原有設定
  total_timesteps: 5000000
  policy_net_arch: [128, 128]
```

```python
# train.py 修改
def train(config_path: str = "config.yaml", ...):
    # ...
    config = yaml.safe_load(open(config_path))
    
    # 檢查快速訓練模式
    if config['training'].get('quick_test_mode', False):
        total_timesteps = config['training']['quick_test_timesteps']
        net_arch = config['training']['quick_test_net_arch']
        print("⚠️ 快速訓練模式啟用：", total_timesteps, "timesteps")
    else:
        total_timesteps = config['training']['total_timesteps']
        net_arch = config['training']['policy_net_arch']
    
    # 使用 net_arch 創建模型
    model = PPO(
        # ...
        policy_kwargs=dict(net_arch=[dict(pi=net_arch, vf=net_arch)])
    )
```

### 5.2 PID 控制器模組化

```python
# controllers/linear_pid.py
import numpy as np

class LinearPID:
    """
    標準線性 PID 控制器（當前實現）
    """
    def __init__(self, kp=5.0, ki=0.1, kd=0.2, integral_max=100.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.last_error = 0.0
        self.integral_max = integral_max
    
    def compute(self, error: float, dt: float) -> float:
        """
        計算 PID 控制輸出
        
        Args:
            error: 當前誤差
            dt: 時間步長
        
        Returns:
            control_output: 控制輸入 u
        """
        # 積分項（含 anti-windup）
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.integral_max, self.integral_max)
        
        # 微分項
        error_dot = (error - self.last_error) / dt if dt > 0 else 0.0
        
        # PID 輸出
        u = self.kp * error + self.ki * self.integral + self.kd * error_dot
        
        self.last_error = error
        return u
    
    def update_gains(self, kp: float, ki: float, kd: float):
        """更新 PID 增益"""
        self.kp = kp
        self.ki = ki
        self.kd = kd
    
    def reset(self):
        """重置積分器和歷史"""
        self.integral = 0.0
        self.last_error = 0.0
```

```python
# controllers/nonlinear_pid.py（可選）
import numpy as np

class NonlinearPID:
    """
    非線性 PID 控制器（參考 AirPilot Eq.6-7）
    """
    def __init__(self, kp=5.0, ki=0.1, kd=0.2, max_velocity=1.0, integral_max=100.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_velocity = max_velocity
        self.integral = 0.0
        self.last_error = 0.0
        self.integral_max = integral_max
    
    def compute(self, error: float, dt: float) -> float:
        """
        計算非線性 PID 控制輸出（正規化速度）
        """
        # 積分項
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.integral_max, self.integral_max)
        
        # 微分項
        error_dot = (error - self.last_error) / dt if dt > 0 else 0.0
        
        # PID 輸出（Eq.6）
        velocity = self.kp * error + self.ki * self.integral + self.kd * error_dot
        
        # 正規化（Eq.7）
        normalized_velocity = velocity / (np.abs(velocity) + 1.0)
        normalized_velocity = np.clip(normalized_velocity, -self.max_velocity, self.max_velocity)
        
        self.last_error = error
        return normalized_velocity
    
    def update_gains(self, kp: float, ki: float, kd: float):
        """更新 PID 增益"""
        self.kp = kp
        self.ki = ki
        self.kd = kd
    
    def reset(self):
        """重置積分器和歷史"""
        self.integral = 0.0
        self.last_error = 0.0
```

### 5.3 可視化改進

```python
# evaluate.py 新增函數
def plot_gains_vs_error(history, output_dir, episode_idx):
    """
    繪製 PID 增益 vs 位置誤差（參考 AirPilot Fig.17）
    """
    error = np.array(history['error'])
    kp = np.array(history['kp'])
    ki = np.array(history['ki'])
    kd = np.array(history['kd'])
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Kp vs Error
    axes[0].scatter(np.abs(error), kp, alpha=0.5, s=10, color='red')
    axes[0].set_xlabel('|Position Error|', fontsize=12)
    axes[0].set_ylabel('Kp', fontsize=12)
    axes[0].set_title('Kp vs Position Error', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Ki vs Error
    axes[1].scatter(np.abs(error), ki, alpha=0.5, s=10, color='green')
    axes[1].set_xlabel('|Position Error|', fontsize=12)
    axes[1].set_ylabel('Ki', fontsize=12)
    axes[1].set_title('Ki vs Position Error', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # Kd vs Error
    axes[2].scatter(np.abs(error), kd, alpha=0.5, s=10, color='blue')
    axes[2].set_xlabel('|Position Error|', fontsize=12)
    axes[2].set_ylabel('Kd', fontsize=12)
    axes[2].set_title('Kd vs Position Error', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, f'gains_vs_error_ep{episode_idx + 1}.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")
```

---

## 六、總結與建議

### 6.1 核心結論

1. **✅ 當前架構設計優秀**
   - 雙層控制架構更符合實際系統
   - 觀測空間包含增益資訊，有利於自適應學習
   - 獎勵函數設計完善

2. **✅ 可借鑑的改進點**
   - 快速訓練模式（提高開發效率）
   - 模組化架構（提高可維護性）
   - 可視化增強（更好的分析工具）

3. **⚠️ 不需要改變的核心設計**
   - 雙層控制架構
   - 觀測空間設計
   - 基本獎勵函數

### 6.2 實施路線圖

```
Week 1: 快速改進
  ├─ 添加快速訓練模式
  ├─ 添加 Gains vs Error 可視化
  └─ 測試驗證

Week 2: 模組化重構
  ├─ 創建 controllers/ 目錄
  ├─ 提取 PID 邏輯
  ├─ 創建 utils/ 目錄
  └─ 更新 import

Week 3+: 實驗性功能（可選）
  ├─ 實現非線性 PID
  └─ 對比實驗
```

### 6.3 風險評估

| 改進項目 | 風險等級 | 緩解措施 |
|---------|---------|---------|
| 快速訓練模式 | 🟢 低 | 添加配置選項，默認關閉 |
| 模組化重構 | 🟡 中 | 逐步遷移，保持向後兼容 |
| 非線性 PID | 🟢 低 | 作為可選功能，不影響現有代碼 |

---

## 七、參考資源

- **AirPilot 論文計畫書**：提供的實作指南
- **當前專案文檔**：
  - `RESEARCH_PLAN.md` - 完整研究計畫
  - `PROGRAM_ARCHITECTURE.md` - 程式架構說明
  - `README.md` - 專案概述

---

**文件版本**：1.0  
**建立日期**：2025-01-XX  
**最後更新**：2025-01-XX

