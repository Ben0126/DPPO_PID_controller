# Training Parameters Modification Guide
# 訓練參數修改指南

## Overview / 概述

All training parameters are configured in `config.yaml`. This guide explains how to modify training steps and related parameters.

所有訓練參數都在 `config.yaml` 文件中配置。本指南說明如何修改訓練步數及相關參數。

---

## Main Training Parameters / 主要訓練參數

### Location / 位置
Edit the `training:` section in `config.yaml` (lines 70-89)

編輯 `config.yaml` 文件中的 `training:` 部分（第 70-89 行）

### Key Parameters / 關鍵參數

#### 1. **total_timesteps** (Line 76)
- **Description**: Total number of training steps (environment steps)
- **Current Value**: `5000000` (5 million steps)
- **How to Modify**: Change the number directly
- **Example**:
  ```yaml
  total_timesteps: 1000000  # Train for 1 million steps
  ```
- **Training Time Estimate**:
  - ~1000 steps/second: ~83 minutes for 5M steps
  - ~5000 steps/second: ~17 minutes for 5M steps

#### 2. **n_steps** (Line 78)
- **Description**: Number of steps to collect before each PPO update (trajectory length)
- **Current Value**: `2048`
- **How to Modify**: Change the number (typically powers of 2: 512, 1024, 2048, 4096)
- **Example**:
  ```yaml
  n_steps: 1024  # Collect 1024 steps before updating
  ```
- **Note**: Larger values = more stable but slower updates

#### 3. **batch_size** (Line 79)
- **Description**: Mini-batch size for gradient updates
- **Current Value**: `64`
- **How to Modify**: Change the number (must be < n_steps)
- **Example**:
  ```yaml
  batch_size: 32  # Smaller batches for more frequent updates
  ```
- **Note**: Should divide evenly into n_steps

#### 4. **learning_rate** (Line 77)
- **Description**: Learning rate for policy and value networks
- **Current Value**: `0.0003` (3×10⁻⁴)
- **How to Modify**: Change the decimal value
- **Example**:
  ```yaml
  learning_rate: 0.0001  # Lower learning rate for more stable training
  ```
- **Common Values**: 1e-4 to 3e-4

#### 5. **n_epochs** (Line 86)
- **Description**: Number of training epochs per PPO update
- **Current Value**: `10`
- **How to Modify**: Change the number
- **Example**:
  ```yaml
  n_epochs: 5  # Fewer epochs for faster training
  ```

---

## Episode Configuration / Episode 配置

### Location / 位置
Edit the `episode:` section in `config.yaml` (lines 65-67)

編輯 `config.yaml` 文件中的 `episode:` 部分（第 65-67 行）

### Parameters / 參數

#### **max_steps** (Line 66)
- **Description**: Maximum RL steps per episode
- **Current Value**: `1000` steps (50 seconds at 20 Hz)
- **How to Modify**: Change the number
- **Example**:
  ```yaml
  episode:
    max_steps: 500  # Shorter episodes (25 seconds)
  ```

---

## Logging Configuration / 日誌配置

### Location / 位置
Edit the `logging:` section in `config.yaml` (lines 92-95)

編輯 `config.yaml` 文件中的 `logging:` 部分（第 92-95 行）

### Parameters / 參數

#### **checkpoint_freq** (Line 95)
- **Description**: Frequency of model checkpoint saves (in steps)
- **Current Value**: `100000` (save every 100k steps)
- **How to Modify**: Change the number
- **Example**:
  ```yaml
  logging:
    checkpoint_freq: 50000  # Save checkpoints more frequently
  ```

---

## Quick Modification Examples / 快速修改範例

### Example 1: Shorter Training (Quick Test)
```yaml
training:
  total_timesteps: 100000    # Only 100k steps for quick test
  n_steps: 1024              # Smaller trajectory
  batch_size: 32             # Smaller batches
  checkpoint_freq: 20000     # Save every 20k steps
```

### Example 2: Longer Training (Better Results)
```yaml
training:
  total_timesteps: 10000000  # 10 million steps
  n_steps: 4096              # Larger trajectory
  batch_size: 128            # Larger batches
  checkpoint_freq: 200000    # Save every 200k steps
```

### Example 3: Faster Learning (Higher Learning Rate)
```yaml
training:
  total_timesteps: 5000000
  learning_rate: 0.0005      # Higher learning rate
  n_epochs: 15               # More epochs per update
```

### Example 4: More Stable Learning (Lower Learning Rate)
```yaml
training:
  total_timesteps: 5000000
  learning_rate: 0.0001      # Lower learning rate
  n_epochs: 5                # Fewer epochs
```

---

## How to Apply Changes / 如何應用更改

1. **Edit config.yaml**: Open `config.yaml` and modify the desired parameters
   - 編輯 config.yaml：打開 `config.yaml` 並修改所需參數

2. **Run Training**: Start training with the modified configuration
   - 運行訓練：使用修改後的配置開始訓練
   ```bash
   python train.py
   ```

3. **Use Custom Config**: Specify a different config file if needed
   - 使用自定義配置：如果需要，指定不同的配置文件
   ```bash
   python train.py --config my_custom_config.yaml
   ```

---

## Parameter Relationships / 參數關係

### Important Constraints / 重要約束

1. **batch_size < n_steps**
   - Batch size must be smaller than trajectory length
   - Batch size 必須小於 trajectory length

2. **n_steps should divide evenly by batch_size**
   - For efficient training: `n_steps % batch_size == 0`
   - 為了高效訓練：`n_steps % batch_size == 0`

3. **Total training time**
   - Estimated time = `total_timesteps / steps_per_second`
   - 估計時間 = `total_timesteps / steps_per_second`

### Recommended Combinations / 推薦組合

| Use Case | total_timesteps | n_steps | batch_size | learning_rate |
|----------|----------------|---------|------------|---------------|
| Quick Test | 100,000 | 1024 | 32 | 0.0003 |
| Standard | 5,000,000 | 2048 | 64 | 0.0003 |
| Long Training | 10,000,000 | 4096 | 128 | 0.0003 |
| Fast Learning | 5,000,000 | 2048 | 64 | 0.0005 |
| Stable Learning | 5,000,000 | 2048 | 64 | 0.0001 |

---

## Notes / 注意事項

1. **Backup Original Config**: Always backup `config.yaml` before making changes
   - 備份原始配置：修改前務必備份 `config.yaml`

2. **Start Small**: For testing, use smaller `total_timesteps` first
   - 從小開始：測試時先使用較小的 `total_timesteps`

3. **Monitor Training**: Use TensorBoard to monitor training progress
   - 監控訓練：使用 TensorBoard 監控訓練進度
   ```bash
   tensorboard --logdir ./ppo_pid_logs/
   ```

4. **Checkpoint Frequency**: Set `checkpoint_freq` based on your `total_timesteps`
   - 檢查點頻率：根據 `total_timesteps` 設置 `checkpoint_freq`

