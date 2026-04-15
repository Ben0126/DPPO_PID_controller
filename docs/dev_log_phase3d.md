# Phase 3d Dev Log: OneDP Single-Step Distillation (v3.3)

**Started:** 2026-04-15
**Status:** Run 1 complete. Inference target achieved (9.9ms). Quality insufficient (RMSE 0.271m).

---

## 1. Motivation & Root Cause

Phase 3c v3.3 確認 50/50 crash 是硬上限，根本原因：

| 指標 | 數值 |
|------|------|
| DDIM 10-step 推理時間 | **74ms** |
| 控制週期 | 20ms（50Hz） |
| 延遲倍率 | **3.7×** |
| 最佳 RMSE（v3.3 Run 1） | 0.1039m |
| Crash 率 | 50/50（所有 v3.3 run） |

Policy 從未在 74ms 滯後下訓練，covariate shift 結構性存在，超參無法解決。

**Phase 3d 目標：** 蒸餾出 1-step student，推理 <16ms，突破延遲瓶頸。

Teacher: `checkpoints/diffusion_policy/dppo_v33_20260413_033647/best_dppo_v33_model.pt`

---

## 2. 實作（2026-04-15）

### 2.1 新增 `models/diffusion_process.py::sample_onestep()`

在 `ddim_sample` 之後新增（不改動現有程式碼）：

```python
def sample_onestep(self, denoise_fn, condition, shape):
    """OneDP 1-step: x_T -> x_0 via single UNet call at t=T-1=99."""
    device = condition.device
    B = shape[0]
    t_idx = self.num_timesteps - 1   # 99
    x_t = torch.randn(shape, device=device)
    t_batch = torch.full((B,), t_idx, device=device, dtype=torch.long)
    eps_pred = denoise_fn(x_t, t_batch, condition)
    alpha_bar = self.alphas_cumprod[t_idx].to(device)
    x0_pred = (x_t - sqrt(1-alpha_bar) * eps_pred) / sqrt(alpha_bar)
    return torch.clamp(x0_pred, -5.0, 5.0)
```

**關鍵數學：** α̅_99 ≈ 0.000243（cosine schedule），放大係數 1/sqrt(α̅_99) ≈ **64.2×**。

### 2.2 `models/vision_dppo_v31.py` 修改

- `predict_action()`: 加入 `steps==1` 分支呼叫 `sample_onestep()`
- 新增 `compute_distillation_loss()` 方法（見下方演變）

### 2.3 `scripts/train_onedp_v33.py`（新建）

Teacher（frozen）生成 10-step DDIM x0 → Student（trainable）蒸餾學習。

### 2.4 `scripts/evaluate_rhc_v33.py` 修改

新增 `--ddim-steps` CLI 參數（3 處改動）。

---

## 3. Smoke Tests（通過）

```
sample_onestep shape: (2, 4, 8) ✓
1-step inference:  8.1ms avg ✓ (<16ms 門檻)
10-step inference: 70.9ms avg
加速: 8.8×
compute_distillation_loss: 可微 ✓
```

---

## 4. 蒸餾訓練嘗試 1：x0-space（失敗）

**設計：** `L = MSE(x0_student, x0_teacher)`

`x0_student` 由 `sample_onestep()` 在訓練中計算，gradient 可回傳。

**結果（Run 4，2026-04-15）：**

| Epoch | loss_distill |
|-------|-------------|
| 1–22 | **24.7476（常數）** |

**失敗根本原因：**

```
x0_student = (x_T - sqrt(1-α̅_99) * eps_pred) / sqrt(α̅_99)
           ≈ eps_pred × 64.2 + noise × 64.2   ← 放大 64.2×
clamp(-5, 5) → 幾乎所有元素在 clamp 邊界
∂clamp/∂x = 0 at boundary → gradient = 0 → 完全無法學習
```

Aborted at epoch 22。

---

## 5. 蒸餾訓練嘗試 2：ε-space（成功收斂，品質不足）

**設計改動：** 改用標準 diffusion loss，以 teacher x0 為目標：

```python
# Teacher x0 當作乾淨 action（no_grad）
t = random_t ∈ [0, 99]
x_t = sqrt(α̅_t) * teacher_x0 + sqrt(1-α̅_t) * ε
eps_pred = student.noise_pred_net(x_t, t, global_cond)
L_distill = MSE(eps_pred, ε)
```

完全避開 64.2× 放大係數，在 ε-space 訓練。

**超參數：**

| 參數 | 值 |
|------|-----|
| num_epochs | 50 |
| batch_size | 128 |
| learning_rate | 3e-4 |
| warmup_epochs | 5 |
| lambda_dispersive | 0.05 |
| lambda_depth | 0.1 |

**訓練曲線（Run 5，2026-04-15~16）：**

Save dir: `checkpoints/diffusion_policy/onedp_v33_20260415_134933/`

| Epoch | loss_distill | total loss | LR |
|-------|-------------|------------|-----|
| 1 | 0.0022 | -0.726 | 6e-5（warmup） |
| 5 | 0.0069 | -0.723 | 3e-4（peak） |
| 10 | 0.0047 | -0.730 | 2.91e-4 |
| 20 | 0.0028 | -0.740 | 2.25e-4 |
| 35 | 0.0010 | -0.749 | 7.5e-5 |
| 45 | 0.0003 | -0.751 | 9.1e-6 |
| **50** | **0.0003** | **-0.751** | 0 |

收斂良好，loss_distill 下降 23×。Best checkpoint: epoch 49。

---

## 6. RHC 評估結果（2026-04-16）

```bash
python -m scripts.evaluate_rhc_v33 \
    --diffusion-model checkpoints/diffusion_policy/onedp_v33_20260415_134933/best_onedp_model.pt \
    --ddim-steps 1
```

| 指標 | OneDP 1-step | Teacher（v3.3 Run 1） | 差距 |
|------|-------------|----------------------|------|
| RMSE | **0.2713m** | 0.1039m | 2.6× 差 |
| Crashes | **50/50** | 50/50 | = |
| Inference | **9.9ms** ✓ | 74ms | 7.5× 快 |
| Mean steps | ~17 | ~37 | 2.2× 少 |
| Mean reward | 3.13 | 24.51 | — |

**推理延遲目標達成（9.9ms < 16ms ✓）**
**飛行品質目標未達成（RMSE 0.271m >> 0.104m）**

---

## 7. Phase 3d 失敗分析

### 7.1 為什麼 ε-space 蒸餾品質不足？

**訓練目標 vs 推理路徑不匹配：**

```
訓練：對 t ∈ [0,99] 均勻取樣，MSE(eps_student(x_t, t), ε)
推理：只在 t=99 做一次 eps prediction，然後直接輸出 x0

→ 訓練讓 student 學習「平均所有 t 的 ε 預測」
→ 推理只依賴「t=99 的 ε 預測準確性」
→ t=99 佔訓練 loss 的 1/100，並未被重點優化
```

**與 teacher 的質量差距：**
- Teacher：10 步 DDIM 從 t=99 逐步除噪，每步糾正誤差
- Student：1 步直接從 t=99→x0，無糾正機會
- 即使 t=99 ε 預測完美，64.2× 放大使任何 ε 預測誤差都被放大

### 7.2 本質限制

| 限制 | 說明 |
|------|------|
| 放大係數 64.2× | 任何 t=99 ε 誤差直接放大 → x0 誤差大 |
| 訓練/推理 mismatch | ε-space 訓練均勻，1-step 推理只用 t=99 |
| Teacher 本身不完美 | Teacher 10-step 輸出作為 target，已有近似誤差 |
| 無 RL 閉環信號 | 蒸餾是純監督，student 未見過自己 1-step 執行的軌跡 |

---

## 8. Phase 3d-v2 策略選項

### 選項 A：僅在 t=99 做蒸餾（focused distillation）

```python
# 只在 t=99 計算蒸餾 loss
t_idx = 99
x_T = torch.randn(shape)
# target: reverse from teacher
eps_target = (x_T - sqrt(alpha_bar_99) * teacher_x0) / sqrt(1-alpha_bar_99)
eps_student = student.noise_pred_net(x_T, t=99, global_cond)
L = MSE(eps_student, eps_target)
```

優點：完全針對推理路徑訓練
風險：teacher x0 + 隨機 x_T 的組合不在任何模態中心，target 不穩定

### 選項 B：DPPO 閉環 fine-tune 1-step student

將 student 用於 RHC rollout，用 RL advantage 更新：
```
L = exp(β × A) × MSE(eps_student(x_T, t=99), eps_target)
```
優點：閉環訓練，見到 1-step 執行的真實後果
風險：訓練穩定性（需要 value net）

### 選項 C：改回 10-step，接受 13Hz，專注解決 crash

放棄延遲目標，轉而研究為何 DPPO 後仍 50/50 crash。
分析 crash 模式：是初始偏差？還是累積誤差？是否有保存期間的好 checkpoint？

### 選項 D：減少 DDIM steps（4-5 步）

```
4-step DDIM ≈ 30ms → 比 50Hz 慢但比 10-step 快
5-step DDIM ≈ 37ms → 大幅優於 74ms
```
仍有 1.5-2× 延遲，但比 10-step 改善

---

## 9. 當前 Checkpoint 總覽

| Checkpoint | 用途 | 推理 |
|-----------|------|------|
| `dppo_v33_20260413_033647/best_dppo_v33_model.pt` | Teacher（DPPO Run 1） | 10-step 74ms |
| `onedp_v33_20260415_134933/best_onedp_model.pt` | OneDP student（有 depth_decoder） | 1-step 9.9ms |
| `onedp_v33_20260415_134933/deploy_onedp_model.pt` | OneDP student（無 depth_decoder） | 1-step 9.9ms |
