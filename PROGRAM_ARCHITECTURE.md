# DPPO PID 控制器程式架構說明

## 📋 目錄
1. [整體架構概覽](#整體架構概覽)
2. [核心模組詳細說明](#核心模組詳細說明)
3. [程式執行流程](#程式執行流程)
4. [資料流向圖](#資料流向圖)
5. [模組間依賴關係](#模組間依賴關係)

---

## 整體架構概覽

本專案實現了一個基於強化學習（RL）的自適應 PID 控制器系統，採用雙層控制架構：
- **外層（20 Hz）**：RL 代理調整 PID 參數（Kp, Ki, Kd）
- **內層（200 Hz）**：PID 控制器執行實際控制動作

```
┌─────────────────────────────────────────────────────────┐
│                   應用層腳本                              │
├──────────────┬──────────────┬──────────────────────────┤
│   demo.py    │   train.py   │      evaluate.py         │
│  (環境測試)   │  (模型訓練)   │      (模型評估)          │
└──────┬───────┴──────┬───────┴──────────┬───────────────┘
       │              │                  │
       └──────────────┼──────────────────┘
                      │
       ┌──────────────▼──────────────┐
       │    dppo_pid_env.py          │
       │  (自定義 Gymnasium 環境)     │
       └──────────────┬──────────────┘
                      │
       ┌──────────────▼──────────────┐
       │    dppo_model.py            │
       │  (DPPO 模型 - 骨架實現)      │
       └────────────────────────────┘
```

---

## 核心模組詳細說明

### 1. `dppo_pid_env.py` - 環境核心

**角色**：自定義 Gymnasium 環境，定義了完整的控制系統模擬

**主要功能**：

#### 1.1 環境初始化 (`__init__`)
```python
- 載入 config.yaml 配置檔
- 定義動作空間：Box([0,0,0], [Kp_max, Ki_max, Kd_max])  # 3維PID增益
- 定義觀測空間：Box(-1, 1, shape=(9,))  # 9維正規化狀態向量
- 初始化狀態變數（位置、速度、PID狀態等）
```

#### 1.2 狀態向量組成（9維觀測）
```python
obs = [
    error / error_scale,           # 追蹤誤差
    error_dot / error_dot_scale,   # 誤差變化率
    integral / integral_scale,    # 積分項
    position / position_scale,     # 當前位置
    velocity / velocity_scale,     # 當前速度
    reference / reference_scale,   # 參考信號
    Kp / gain_scale,              # 當前Kp值
    Ki / gain_scale,              # 當前Ki值
    Kd / gain_scale               # 當前Kd值
]
```

#### 1.3 雙層控制架構 (`step` 方法)

**外層循環（20 Hz）**：
```python
1. 接收 RL 動作：[Kp, Ki, Kd]
2. 更新 PID 增益（限制在允許範圍內）
3. 執行 n_inner_steps 次內層循環
4. 更新參考信號（每 change_interval 秒）
5. 檢查終止條件
6. 返回觀測、累積獎勵、終止標誌
```

**內層循環（200 Hz）**：
```python
for inner_step in range(n_inner_steps):
    1. 計算誤差：error = reference - position
    2. 計算 PID 控制輸入：u = Kp*error + Ki*integral + Kd*error_dot
    3. 限制控制輸入：u ∈ [-10, 10]
    4. 應用干擾（如果啟用）
    5. 積分系統動態（RK4 或 Euler）
    6. 更新 PID 狀態（積分項、誤差歷史）
    7. 計算獎勵並累積
```

#### 1.4 系統動態模型
```python
# 二階系統：J * x_ddot + B * x_dot = u(t) + d(t)
# 其中：
#   J = 慣性矩（1.0）
#   B = 阻尼係數（0.5）
#   u = 控制輸入
#   d = 外部干擾
```

#### 1.5 獎勵函數
```python
reward = -λ_error * error²           # 追蹤誤差懲罰
        - λ_velocity * velocity²     # 速度懲罰（防止振盪）
        - λ_control * u²             # 控制努力懲罰
        - λ_overshoot * max(0, e·ė)  # 超調懲罰
```

---

### 2. `train.py` - 訓練腳本

**角色**：使用 Stable-Baselines3 的 PPO 算法訓練代理

**執行流程**：

#### 2.1 環境準備
```python
1. 載入 config.yaml
2. 創建訓練環境（帶 Monitor 包裝）
3. 創建評估環境（獨立，用於驗證）
4. 應用 VecNormalize（觀測/獎勵正規化）
```

#### 2.2 模型初始化
```python
# 從配置載入或創建新模型
PPO(
    policy="MlpPolicy",
    learning_rate=3e-4,
    n_steps=2048,          # 每次更新收集的步數
    batch_size=64,         # 小批次大小
    n_epochs=10,           # 每次更新的訓練輪數
    gamma=0.99,            # 折扣因子
    gae_lambda=0.95,       # GAE 參數
    clip_range=0.2,        # PPO 裁剪範圍
    policy_kwargs={
        'net_arch': [dict(pi=[128,128], vf=[128,128])]
    }
)
```

#### 2.3 回調設置
```python
1. CheckpointCallback：定期保存模型檢查點
2. EvalCallback：定期評估並保存最佳模型
```

#### 2.4 訓練循環
```python
model.learn(
    total_timesteps=5,000,000,
    callback=[checkpoint_callback, eval_callback],
    progress_bar=True
)
```

**訓練過程**：
1. 收集 2048 步的軌跡資料
2. 計算優勢估計（GAE）
3. 進行 10 輪小批次更新
4. 應用 PPO 裁剪損失
5. 更新價值函數
6. 重複直到達到總步數

---

### 3. `evaluate.py` - 評估腳本

**角色**：評估訓練好的模型並生成視覺化結果

**執行流程**：

#### 3.1 模型載入
```python
1. 載入訓練好的 PPO 模型
2. 載入正規化統計資料（如果存在）
3. 創建評估環境（與訓練環境相同配置）
```

#### 3.2 評估循環
```python
for episode in range(n_episodes):
    obs = env.reset()
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        # 記錄歷史資料
    # 從環境獲取完整歷史
    history = env.get_history()
    all_histories.append(history)
```

#### 3.3 視覺化生成
```python
1. 最佳回合圖（最高獎勵）
   - 位置追蹤
   - 誤差和控制輸入
   - PID 增益演化

2. 最差回合圖（最低獎勵）

3. 統計摘要圖
   - 回合獎勵分佈
   - 回合長度分佈
   - 平均絕對誤差分佈
   - 最終 PID 增益分佈
```

---

### 4. `demo.py` - 演示腳本

**角色**：使用隨機動作測試環境，驗證實現正確性

**執行流程**：

#### 4.1 環境 API 測試
```python
test_environment_api():
    - 測試 reset()：驗證觀測空間形狀和範圍
    - 測試 step()：驗證返回格式
    - 測試多步執行
    - 測試動作空間邊界
```

#### 4.2 隨機動作演示
```python
run_demo_episode():
    - 創建環境
    - 執行 n_steps 步隨機動作
    - 每 50 步打印狀態資訊
    - 繪製結果圖表
```

**用途**：
- 驗證環境實現正確性
- 檢查系統動態是否合理
- 確認觀測和獎勵計算無誤

---

### 5. `dppo_model.py` - DPPO 模型（骨架實現）

**角色**：實現 Diffusion Policy + PPO 的混合算法（目前為骨架）

**架構組成**：

#### 5.1 核心組件
```python
1. SinusoidalPositionEmbeddings
   - 將擴散時間步編碼為向量

2. ConditionalMLP
   - 將觀測編碼為條件向量

3. DenoisingNetwork (ε_θ)
   - 預測噪聲的網路
   - 輸入：噪聲動作、時間步、觀測
   - 輸出：預測的噪聲

4. ValueNetwork (V_φ)
   - 價值函數網路
   - 估計狀態價值

5. DiffusionProcess
   - 實現前向和反向擴散過程
   - q_sample：添加噪聲（訓練）
   - p_sample：去噪（採樣）
   - ddim_sample：快速採樣（推理）
```

#### 5.2 DPPO 算法流程（設計）
```python
1. 收集軌跡（使用擴散策略採樣動作）
2. 計算 GAE 優勢
3. 更新去噪網路（加權擴散損失）
   L_policy = E[exp(β·A_t) · ||ε_θ(S, A_t, t) - ε||²]
4. 更新價值網路（MSE 損失）
```

**目前狀態**：骨架實現，主要方法標記為 `NotImplementedError`

---

## 程式執行流程

### 典型工作流程

```
┌─────────────────────────────────────────────────────────┐
│ 步驟 1：驗證環境 (demo.py)                               │
├─────────────────────────────────────────────────────────┤
│ $ python demo.py                                        │
│                                                          │
│ → 測試環境 API                                           │
│ → 執行隨機動作演示                                        │
│ → 生成 demo_results.png                                  │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ 步驟 2：訓練模型 (train.py)                              │
├─────────────────────────────────────────────────────────┤
│ $ python train.py --config config.yaml                 │
│                                                          │
│ → 創建訓練/評估環境                                       │
│ → 初始化 PPO 模型                                        │
│ → 訓練 5,000,000 步                                     │
│ → 定期保存檢查點和最佳模型                                │
│ → 生成 TensorBoard 日誌                                  │
│                                                          │
│ 輸出：                                                    │
│   - models/dppo_pid_checkpoint_*.zip                    │
│   - models/best_model.zip                               │
│   - models/dppo_pid_final_*.zip                         │
│   - ppo_pid_logs/ (TensorBoard)                         │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ 步驟 3：評估模型 (evaluate.py)                           │
├─────────────────────────────────────────────────────────┤
│ $ python evaluate.py --model models/best_model.zip     │
│                                                          │
│ → 載入訓練好的模型                                        │
│ → 執行 5 個評估回合                                       │
│ → 生成視覺化圖表                                         │
│                                                          │
│ 輸出：                                                    │
│   - evaluation_results/best_episode_*.png              │
│   - evaluation_results/worst_episode_*.png              │
│   - evaluation_results/evaluation_summary.png           │
└─────────────────────────────────────────────────────────┘
```

---

## 資料流向圖

### 訓練階段的資料流

```
┌─────────────┐
│  config.yaml│
└──────┬──────┘
       │
       ▼
┌──────────────────┐      ┌──────────────┐
│  dppo_pid_env.py │◄─────│   train.py   │
│                  │      │              │
│  - reset()       │      │  - 創建環境   │
│  - step(action)  │      │  - 初始化PPO │
│  - get_history() │      │  - 訓練循環   │
└──────┬───────────┘      └──────┬───────┘
       │                         │
       │ action: [Kp, Ki, Kd]    │
       │◄────────────────────────┘
       │                         │
       │ obs: [9維狀態向量]       │
       │ reward: 標量獎勵         │
       │─────────────────────────►
       │
       ▼
┌──────────────────┐
│  系統動態模擬      │
│  - 二階系統       │
│  - PID 控制       │
│  - 干擾注入       │
└──────────────────┘
```

### 評估階段的資料流

```
┌──────────────────┐
│  best_model.zip   │
└──────┬────────────┘
       │
       ▼
┌──────────────────┐      ┌──────────────┐
│   evaluate.py    │─────►│ dppo_pid_env │
│                  │      │              │
│  - 載入模型       │      │  - reset()    │
│  - 預測動作       │      │  - step()     │
│  - 收集歷史       │      │  - get_history│
│  - 生成圖表       │      └──────────────┘
└──────────────────┘
       │
       ▼
┌──────────────────┐
│  evaluation_     │
│  results/        │
│  - *.png         │
└──────────────────┘
```

---

## 模組間依賴關係

### 依賴圖

```
demo.py
  ├── dppo_pid_env.py (make_env)
  └── matplotlib (視覺化)

train.py
  ├── dppo_pid_env.py (make_env)
  ├── stable_baselines3 (PPO, VecNormalize, Monitor)
  └── config.yaml

evaluate.py
  ├── dppo_pid_env.py (make_env)
  ├── stable_baselines3 (PPO, VecNormalize)
  ├── matplotlib (視覺化)
  └── config.yaml

dppo_pid_env.py
  ├── gymnasium (基礎環境類)
  ├── numpy (數值計算)
  └── config.yaml

dppo_model.py
  ├── torch (神經網路)
  └── numpy (數值計算)
  └── (目前未與其他模組整合)
```

### 關鍵介面

#### 1. 環境介面（Gymnasium 標準）
```python
# dppo_pid_env.py 實現
class DPPOPIDEnv(gym.Env):
    def reset(seed, options) -> (obs, info)
    def step(action) -> (obs, reward, terminated, truncated, info)
    def get_history() -> Dict  # 自定義方法
```

#### 2. 模型介面（Stable-Baselines3）
```python
# train.py 使用
model = PPO.load(model_path)
model.learn(total_timesteps, callback)
model.predict(obs, deterministic=True)

# evaluate.py 使用
model = PPO.load(model_path)
action, _ = model.predict(obs, deterministic=True)
```

#### 3. 配置介面（YAML）
```python
# 所有腳本共用 config.yaml
config = yaml.safe_load(open('config.yaml'))
# 包含：plant, timing, reference, disturbance, pid, 
#       observation, reward, episode, training, logging
```

---

## 關鍵設計決策

### 1. 雙層控制架構
- **外層（20 Hz）**：RL 代理調整 PID 參數
- **內層（200 Hz）**：PID 控制器執行控制
- **優勢**：分離時間尺度，符合實際控制系統

### 2. 觀測空間正規化
- 所有觀測值正規化到 [-1, 1]
- 使用配置檔定義的縮放因子
- **優勢**：提高訓練穩定性

### 3. VecNormalize 包裝
- 動態正規化觀測和獎勵
- 訓練時更新統計，評估時凍結
- **優勢**：自動適應資料分佈

### 4. 歷史記錄機制
- 環境內部維護完整歷史
- 僅在需要時（評估/演示）提取
- **優勢**：不影響訓練效率

### 5. 模組化設計
- 環境、訓練、評估分離
- 配置檔集中管理
- **優勢**：易於維護和擴展

---

## 使用建議

### 開發階段
1. **先運行 `demo.py`**：驗證環境實現
2. **檢查 `config.yaml`**：確認參數合理
3. **小規模訓練**：修改 `total_timesteps` 為較小值測試

### 訓練階段
1. **監控 TensorBoard**：`tensorboard --logdir ppo_pid_logs/`
2. **定期檢查檢查點**：確保模型正常保存
3. **調整超參數**：根據訓練曲線調整學習率等

### 評估階段
1. **使用最佳模型**：`models/best_model.zip`
2. **多回合評估**：增加 `--episodes` 參數
3. **分析視覺化**：檢查追蹤性能和 PID 增益演化

---

## 未來擴展方向

### 短期（Phase 2）
- [ ] 實現 `dppo_model.py` 的完整功能
- [ ] 整合 DPPO 訓練流程
- [ ] 性能對比（PPO vs DPPO）

### 長期（Phase 3+）
- [ ] 多軸控制（6-DOF）
- [ ] 更複雜的系統動態
- [ ] 實時硬體部署

---

## 總結

本專案採用清晰的模組化架構：

1. **`dppo_pid_env.py`**：核心環境，定義控制問題
2. **`train.py`**：訓練流程，使用 PPO 學習策略
3. **`evaluate.py`**：評估工具，驗證模型性能
4. **`demo.py`**：測試工具，驗證環境正確性
5. **`dppo_model.py`**：未來擴展，實現 DPPO 算法

所有模組通過標準介面（Gymnasium、YAML 配置）連接，確保良好的可維護性和擴展性。

