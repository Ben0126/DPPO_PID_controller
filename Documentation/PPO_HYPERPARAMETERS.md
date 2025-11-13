# PPO Hyperparameter Guide / PPO 超參數指南

## Recommended Settings / 建議設定

### 1. Policy Network (策略網路)

**Value / 值**: MLP with 2 layers, 128 units each
**Chinese / 中文**: MLP，2 層，每層 128 units
**Configuration**:
```yaml
policy_net_arch: [128, 128]
value_net_arch: [128, 128]
```

**Explanation / 說明**:
- For a 9-dimensional state space, a medium-sized network is sufficient
- 對於 9 維狀態空間，使用中等大小的網路即可
- 128 units per layer provides good capacity without overfitting
- 每層 128 個單元提供良好的容量而不會過度擬合

**Alternatives / 其他選擇**:
- Smaller: `[64, 64]` - faster training, may underfit
- Larger: `[256, 256]` - more capacity, slower training

---

### 2. Learning Rate (學習率)

**Value / 值**: 3 × 10⁻⁴ (0.0003)
**Configuration**:
```yaml
learning_rate: 0.0003
```

**Explanation / 說明**:
- Standard starting value for PPO
- 標準起始值，訓練過程中可能需要遞減
- May need to decrease during training if loss oscillates
- 如果損失振盪，訓練過程中可能需要降低

**Tuning Tips / 調整建議**:
- Too high (>1e-3): Training unstable, loss oscillates
- Too low (<1e-5): Training too slow, may get stuck
- Recommended range: 1e-4 to 5e-4

---

### 3. Gamma (γ) - Discount Factor

**Value / 值**: 0.99
**Configuration**:
```yaml
gamma: 0.99
```

**Explanation / 說明**:
- High discount factor for long-term planning
- 用於長期規劃的高折扣率
- Important for PID integral term learning
- 對於學習 PID 積分項很重要

**Why 0.99? / 為什麼是 0.99？**:
- Episode length: 1000 steps (50 seconds)
- With γ=0.99, effective horizon ≈ 100 steps (5 seconds)
- Balances immediate and future rewards

---

### 4. GAE Lambda (λ)

**Value / 值**: 0.95
**Configuration**:
```yaml
gae_lambda: 0.95
```

**Explanation / 說明**:
- Balances bias and variance in advantage estimation
- 在優勢估計中平衡偏差 / 變異
- 0.95 is a well-tested default value

**Understanding GAE / 理解 GAE**:
- λ=0: Low variance, high bias (uses only 1-step returns)
- λ=1: High variance, low bias (uses Monte Carlo returns)
- λ=0.95: Good balance for most problems

---

### 5. n_steps - Trajectory Length

**Value / 值**: 2048
**Configuration**:
```yaml
n_steps: 2048
```

**Explanation / 說明**:
- Number of steps collected before each PPO update
- 代理人與環境互動的步數（影響 batch size）
- At 20 Hz outer loop: 2048 steps ≈ 102 seconds of simulation

**Memory Requirements / 記憶體需求**:
- Buffer size: n_steps × observation_dim = 2048 × 9
- Larger n_steps → more stable gradients but more memory

---

### 6. Batch Size (批次大小)

**Value / 值**: 64
**Configuration**:
```yaml
batch_size: 64
```

**Explanation / 說明**:
- Must be smaller than n_steps (currently 2048)
- 應小於 nₛₜₑₚₛ
- Number of samples used in each gradient update

**Calculation / 計算**:
- n_steps = 2048, batch_size = 64
- Number of mini-batches = 2048 / 64 = 32
- With n_epochs=10: 320 gradient updates per PPO update

**Tuning / 調整**:
- Larger (128, 256): More stable gradients, slower updates
- Smaller (32): Faster updates, noisier gradients

---

### 7. VecNormalize - Observation/Reward Normalization

**Value / 值**: Enabled
**Configuration**:
```yaml
use_vec_normalize: true
```

**Implementation / 實作**:
```python
train_env = VecNormalize(
    train_env,
    norm_obs=True,      # Normalize observations
    norm_reward=True,   # Normalize rewards
    clip_obs=10.0,      # Clip normalized obs to [-10, 10]
    clip_reward=10.0    # Clip normalized rewards to [-10, 10]
)
```

**Explanation / 說明**:
- **CRITICAL for stability** / **對穩定性至關重要**
- Normalizes observations to zero mean, unit variance
- 將觀察值正規化為零均值、單位變異數
- Prevents gradient explosion with different scales

**Why It's Important / 為什麼重要**:
- State dimensions have different scales:
  - Error: [-2, 2]
  - Velocity: [-10, 10]
  - Gains: [0, 10]
- Without normalization, learning is unstable

---

## Summary Table / 總結表格

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Policy Network | [128, 128] | State → Action mapping |
| Value Network | [128, 128] | State → Value estimation |
| Learning Rate | 3×10⁻⁴ | Step size for gradient descent |
| n_steps | 2048 | Samples per PPO update |
| Batch Size | 64 | Samples per gradient update |
| Gamma (γ) | 0.99 | Future reward discount |
| GAE Lambda (λ) | 0.95 | Advantage estimation balance |
| VecNormalize | Enabled | Stabilize training |

---

## Training Progress Monitoring / 訓練進度監控

### Expected Training Time / 預期訓練時間

- Total timesteps: 5,000,000
- At ~1000 steps/second: **~90 minutes on CPU**
- At ~5000 steps/second: **~15 minutes on GPU**

### Key Metrics to Monitor / 關鍵指標

1. **Episode Reward** (ep_rew_mean)
   - Should increase over time
   - Target: > -500 (good control)

2. **Episode Length** (ep_len_mean)
   - Should increase (fewer early terminations)
   - Target: Close to 1000 (full episode)

3. **Policy Loss** (policy_loss)
   - Should decrease then stabilize
   - Large fluctuations indicate instability

4. **Value Loss** (value_loss)
   - Should decrease over time
   - Measures value function accuracy

5. **Explained Variance**
   - Should be > 0.5
   - Indicates value function quality

---

## Troubleshooting / 疑難排解

### Problem: Training is unstable
**Solution / 解決方案**:
- ✓ Enable VecNormalize (already enabled)
- Lower learning rate to 1e-4
- Reduce clip_range to 0.1

### Problem: Learning too slow
**Solution / 解決方案**:
- Increase learning rate to 5e-4
- Increase n_steps to 4096
- Check reward function weights

### Problem: Agent doesn't improve
**Solution / 解決方案**:
- Check if episodes are terminating early (unstable PID)
- Adjust reward weights (increase lambda_error)
- Verify environment implementation

### Problem: Overfitting
**Solution / 解決方案**:
- Add entropy coefficient: `ent_coef: 0.01`
- Use smaller network: `[64, 64]`
- Add domain randomization

---

## Advanced Configuration / 進階配置

### Learning Rate Schedule

For longer training, use learning rate decay:

```python
from stable_baselines3.common.utils import linear_schedule

model = PPO(
    learning_rate=linear_schedule(3e-4, 1e-5),  # Decay from 3e-4 to 1e-5
    ...
)
```

### Adaptive KL Penalty

For more stable training:

```yaml
use_kl_penalty: true
target_kl: 0.01
```

### Curriculum Learning

Progressively increase difficulty:

1. Phase 1 (0-1M steps): Simple setpoints, no disturbances
2. Phase 2 (1M-3M steps): Add setpoint changes
3. Phase 3 (3M-5M steps): Add disturbances

---

## References / 參考資料

1. Schulman et al. (2017). "Proximal Policy Optimization Algorithms"
2. Stable-Baselines3 Documentation: https://stable-baselines3.readthedocs.io/
3. PPO Hyperparameter Tuning Guide: https://medium.com/aureliantactics/ppo-hyperparameters-and-ranges-6fc2d29bccbe

---

## Quick Reference / 快速參考

```yaml
# Copy this to your config.yaml
training:
  policy_net_arch: [128, 128]    # ✓ Recommended
  value_net_arch: [128, 128]     # ✓ Recommended
  learning_rate: 0.0003          # ✓ 3×10⁻⁴
  n_steps: 2048                  # ✓ Standard
  batch_size: 64                 # ✓ Should be < n_steps
  gamma: 0.99                    # ✓ High for long-term planning
  gae_lambda: 0.95               # ✓ Balanced bias/variance
  use_vec_normalize: true        # ✓ CRITICAL
```
