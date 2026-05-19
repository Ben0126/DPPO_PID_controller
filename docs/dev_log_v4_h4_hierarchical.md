# v4.0 Dev Log — H4 Architecture & Hierarchical Evaluation Era

**繼續自：** [dev_log_v4_post20.md](dev_log_v4_post20.md)（Runs 1–20 + Hypothesis 1–3a）
**起始日期：** 2026-05-13
**狀態（截至 2026-05-19，最終）：** **H4 BC 確認為 v4.0 最終 SOTA**（hierarchical score 0.167–0.171）。RL fine-tuning 27 runs 全部 AWR mode-collapse。v5 pipeline Stages A→D 全部失敗——encoder-action alignment 在分離預訓練下結構性不可解。研究結論記錄於 §14。

---

## 重大發現摘要

| 發現 | 日期 | 影響 |
|------|------|------|
| **RMSE 是誤導性指標** — 偏袒短命 policy（窗口越短 RMSE 越小） | 2026-05-15 | 推翻過去 24 個 runs 的「最佳」排名 |
| **Disturbance 不是 crash 原因** — 移除後 steps 完全沒變 | 2026-05-14 | 否定 Run 24「domain gap」假設 |
| **Phase lag (RHC) 是次要因子** — T_action=4→1 只 +13% steps | 2026-05-14 | 不是控制頻寬瓶頸 |
| **Flow inference 1-step 不夠** — n_steps=3 給 +35% steps | 2026-05-14 | 挑戰 OT 線性插值「1-step 最優」理論 |
| **H4 (IMU 512D dominance) = v4.0 真正 SOTA** | 2026-05-15 | 觀測層才是瓶頸；BC 突破 +55% steps |
| **AWR mode-collapse**：advantage normalization 抹平 crash penalty | 2026-05-16 | 解釋 Runs 25/26/27 的單調退化 |
| **v5 蒸餾 Gradient conflict** — flow_loss 與 state_loss 共享 vision encoder，梯度方向對抗 | 2026-05-18 | 在 vision-only 學物理的設計上失敗；DAgger 蒸餾本身對 OOD 影像無解 |
| **Hover-only BC 對 OOD swift perturbation 無覆蓋** — rot6D std≈0.04 使 25° tilt = ±5σ | 2026-05-18 | State prediction 在 normalized 空間結構性爆炸 |
| **Encoder-action alignment 不可解（分離預訓練）** — state regression 最優特徵 ≠ action generation 最優特徵 | 2026-05-19 | v5 pipeline Stages A→D 全部失敗；H4 BC 確認 v4.0 最終 SOTA |
| **DAgger 在 100% crash 下結構性失敗** — teacher label 基於 crash trajectory，flow_net 學習墜落時的動作 | 2026-05-19 | 任何基於 on-policy rollout 的蒸餾都需要 >0% 存活率作為前提 |

---

## 實驗索引（Run 21+ 對應 dev_log_v4_post20.md 之 Hypothesis 3a 之後）

| Run | 假設 | 日期 | 結果 |
|-----|------|------|------|
| 21 | "Unshackled RL" (sigma_pos=0.30, lambda_bc=0.01) | 2026-05-12 | **DENIED** — VLoss gate 從未開啟（10.0 結構性不可達） |
| 22 | vloss_gate disabled (999) | 2026-05-13 | 部分成功 — peak 0.657 @u147 後崩潰，curriculum ramp 致命 |
| 23 | hover-only RL (n_hover_updates=600) | 2026-05-14 | **DENIED** — peak 0.686 @u186，純 hover 也衰退 |
| 24 | disturbance matched (0.3N) | 2026-05-14 | **CANCELLED** — Exp A 證實 disturbance 不是 crash 原因 |
| 25 | **H4 architecture** (IMU 512D dominance) | 2026-05-15 | BC +55% steps（202→ vs H3a 130）；RL 仍崩潰 86 steps |
| 26 | **Linear IAE reward** (alive_bonus=1, crash=100) | 2026-05-16 | **DENIED** — 同 Run 25 崩潰模式（peak u274 後衰退） |
| 27 | **Dense risk** (tilt² + ω²) | 2026-05-16 | **DENIED** — 同樣 peak u259 → 崩潰 |
| 28 | **Positive-advantage mask** (硬遮罩 adv<0) | 2026-05-17 | 中斷（v5 架構轉向） |
| v5-BC | **Cross-Attn + State Aux** BC 預訓練 | 2026-05-18 | ✓ 成功 — val/flow=0.0627 @ ep22；H4 weights transfer 94 tensors |
| v5-D1 | **DAgger Distill** (λ_state=0.1, lr=1e-5) | 2026-05-18 | **FAILED** — killed u66/200，state_loss 振盪 0.24–5.46，crash 100% |
| v5-D2 | **DAgger Distill** (λ_state=1.0, lr=1e-5) | 2026-05-18 | **FAILED** — killed u20/200，state_loss 上升 1.34→1.91，crash 100% |
| v5-A | **Partial norm** (rot6D raw) | 2026-05-18 | 部分改善，OOD 仍崩；進 Stage B |
| v5-B | **OOB pretrain** (vision_enc + state_pred on recovery.h5) | 2026-05-19 | val_state=0.194 ✓；eval score=0.140 < H4 BC 0.171 |
| v5-C | **Frozen encoder + λ=0 DAgger** (Run 5) | 2026-05-19 | flow_loss 收斂，crash_rate 99.6%；DAgger 結構性失敗確認 |
| v5-D | **BC retrain flow_net** (frozen OOB encoder, ep1 best) | 2026-05-19 | **FAILED** — score=0.073 < OOB 0.140；encoder-action alignment 不可解 |

---

## 1. Runs 21-23：傳統 RL 修正嘗試

### Run 21 — VLoss Gate 結構性不可達

**設定：** sigma_pos=0.30, lambda_bc=0.01, curriculum 0.1→2.0m, vloss_gate=10.0

**結果：** 700 updates 全部失敗 — vloss 始終在 40-75 之間，gate 從未開啟，policy_loss=0 全程。

**根因：** 50/50 crash 環境下，return 有結構性隨機性：episode 在 step 50 vs step 300 crash 的 return 差距 ≈ 15 units（std ≈ 8）。VLoss = std² ≈ 64，**結構上無法降到 10.0**。

**修正：** `vloss_gate: 10.0 → 999`（純時間 warmup）。

### Run 22 — Curriculum Ramp 致命

**設定：** Run 21 + vloss_gate=999, n_hover_updates=100。

**結果：** Peak reward 0.657 @u147（hover 階段），u150 ramp 開始後立即崩潰至 0.083 @u699。BC gate eval: RMSE 0.600m, 50/50 crash, steps ~80。

**修正：** `n_hover_updates: 100 → 600`（純 hover）。

### Run 23 — 純 Hover 仍然衰退

**設定：** Run 22 + n_hover_updates=600（700 update 內無 ramp）。

**結果：** Peak reward 0.686 @u186。但在純 hover（pos_range=0.1m 全程）下仍從 u186 衰退到 0.271 @u699。BC gate: RMSE 0.529m, 50/50, steps ~70。

**新發現：** 崩潰**不是 ramp 造成的**，是更深層的問題。

---

## 2. 三個 Zero-Cost 診斷實驗（2026-05-14）

對 Run 23 best checkpoint 做 inference-time 變化，**不重訓**。

| 設定 | Steps avg | RMSE | 改善 vs baseline |
|------|-----------|------|------------------|
| T_action=4, n_steps=1 (baseline) | 70 | 0.529m | — |
| **Disturbance=OFF** | 73 | 0.534m | **+4%（噪聲級別）** |
| T_action=1, n_steps=1 | 79 | 0.549m | +13% |
| T_action=1, n_steps=2 | 91 | 0.606m | +30% |
| T_action=1, n_steps=3 | 107 | 0.647m | +53% |

**結論：**

1. **Disturbance 完全不是 crash 原因**（Run 24 假設否定）
2. **Phase lag (T_action)** 是次要因子（+13%）
3. **Flow inference quality**（n_inference_steps）比 OT 理論預期更重要（+35%）
4. **架構性 ceiling 確認**：53% 改善壓榨完所有 inference-time 招數，仍然 50/50 crash → 觀測層才是瓶頸

---

## 3. Hypothesis 4：H4 IMU-Dominant Fusion（Run 25）

### 假設

Vision-only policy 對「高頻狀態」（2° tilt、0.05 m/s 微擾）有結構性盲點。FPV 影像在 CNN 感受野下小變化無法被感知 → policy 認為「自己沒事」→ 偏差累積至 ~step 75 達到 max_tilt 60° crash。

### 架構修改

| 元件 | H3a | **H4** |
|------|-----|--------|
| IMU encoder hidden_dim | 256 | **1024** |
| IMU feature_dim | 128 | **512** |
| global_cond_dim | 256 + 128 = 384 | **256 + 512 = 768** |
| IMU 比例 | 33% | **67%** |
| IMU params | 34,816 | **531,968** (15×) |
| **V/I 梯度比** | **9.9×** | **3.22×** |

**Verification：** 梯度比從 H3a 的 9.9× 降到 3.22×（vs original 46.8×），IMU 真正獲得主導地位。

### 結果

**H4 BC（無 RL）：**
- Val loss: 0.0632 @ epoch 76（與 H3a 相同；BC 訓練資料是 stable hover，IMU 訊號弱）
- **BC gate eval: steps avg 202** (vs H3a BC 130, **+55%**)
- RMSE 1.15m, 50/50 crash
- 最佳 episode 撐到 step 304/500

**H4 + RL (Run 25)：**
- Peak training reward 0.703 @u375（v4.0 新紀錄）
- 但 BC gate eval 反而退化：steps 86, RMSE 0.54m
- 與 Run 23 相同的「train↑ eval↓」訓練-評估鴻溝

---

## 4. RMSE 偏誤發現（2026-05-15）

`evaluate_rhc_v4.py` 計算 RMSE 時：

```python
pos_errors = np.array(targets) - np.array(positions)
rmse = np.sqrt(np.mean(np.sum(pos_errors**2, axis=1)))   # 平均只在 ep_length 內
```

**關鍵：** T = `ep_length`（實際存活步數），不是 `max_episode_steps=500`。

→ Crash 越早，視窗越短，drone 沒時間飄遠 → RMSE 越小。

**後果：** 過去整個 v4.0 研究方向被誤導性指標牽著走：

| Run | Steps | RMSE | 過去詮釋 | 真實狀況 |
|-----|-------|------|---------|---------|
| Run 10 | 36 | **0.30m** | 「最佳結果」 | 死得最快的精準 hover |
| Run 23 | 70 | 0.53m | 「次佳」 | 死得快+尚算精準 |
| H4 BC | 202 | 1.15m | 「最差」 | **活得最久**，僅在較長視窗內看起來偏離 |

---

## 5. Hierarchical Evaluation Framework（飛→穩→準）

實作於 [scripts/evaluate_hierarchical.py](../scripts/evaluate_hierarchical.py)。

### Tier 1 — 飛 (Survival)
$$\text{survival\_rate} = \text{ep\_length} / \text{max\_episode\_steps}$$
通過閾值：> 0.5（撐過半場才有資格談穩定）

### Tier 2 — 穩 (Stability，跳過暫態)
$$\text{IAE}_{\text{steady}} = \frac{1}{T/2}\sum_{t=T/2}^{T} |\mathbf{e}_t|$$
（後半段平均累計誤差，避開起飛暫態）

### Tier 3 — 準 (Accuracy，終值)
$$\text{terminal\_err} = \frac{1}{T/10}\sum_{t=0.9T}^{T} |\mathbf{e}_t|$$

### 複合分數

```
若 survival_rate < 0.5：score = 0.5 × survival_rate  (Tier 1 fail，<0.25)
否則：
  stability_score = max(0, 1 - IAE_steady / 2.0)
  accuracy_score  = max(0, 1 - terminal_err / 2.0)
  score = survival_rate × (0.6 × stability_score + 0.4 × accuracy_score)
```

### 7-Checkpoint 完整重評（2026-05-15）

| 排名 | Checkpoint | Score | Survive | IAE_st | Term | RMSE | 舊 RMSE 排名 |
|------|-----------|-------|---------|--------|------|------|-------------|
| **1** | **H4_BC** | **0.165** | 44.0% | 1.473m | 2.510m | 1.165m | 第 5 |
| 2 | Run25_u50 | 0.158 | 43.2% | 1.650m | 3.022m | 1.334m | 第 7（最差）|
| 3 | H3a_BC | 0.151 | 43.6% | 1.656m | 2.839m | 1.310m | 第 6 |
| 4 | Run25_u200 | 0.128 | 26.6% | 0.976m | 1.694m | 0.773m | 第 4 |
| 5 | Run23_RL | 0.093 | 18.6% | 0.770m | 1.272m | 0.601m | 第 3 |
| 6 | Run25_best | 0.086 | 17.1% | 0.713m | 1.119m | 0.551m | 第 2 |
| 7 | **Run19_RL** | 0.078 | 15.6% | 0.710m | 1.113m | 0.548m | 第 1（最佳）|

**排名與 RMSE 完全反轉。** 過去「最佳」runs 在新指標下是「最差」。

---

## 6. Runs 26-27：Reward Function 重設計（仍失敗）

### Run 26 — Linear IAE Reward

修改 `envs/quadrotor_env_v4.py` 加 `_calculate_reward_iae`：

$$r_t = \underbrace{1.0}_{\text{alive\_bonus}} - \underbrace{0.5|\mathbf{e}_t|}_{\text{IAE}} - \underbrace{0.05|\mathbf{v}_t|}_{\text{vel}} - \underbrace{0.1|\boldsymbol{\omega}_t|}_{\text{ang}} - \underbrace{0.005|\Delta\mathbf{a}_t|^2}_{\text{action}}$$

Crash 額外扣 -100。

**結果：** Best reward -0.16 @u274，然後**同樣的單調崩潰**到 -1.87 @u699。Hierarchical score 0.106（仍輸 H4 BC 的 0.165）。

### Run 27 — Dense Risk Signal

加入「crash precursor」每步懲罰：

$$r_t = \dots - w_{\text{tilt}} \cdot \text{tilt}^2 - w_{\omega} \cdot |\boldsymbol{\omega}|^2$$

`w_tilt=1.0`（tilt 60° 給 -1.10/step）, `w_ang: 0.1 → 0.3`（quadratic）。

**設計理由：** 假設 PPO advantage normalization 抹平了 sparse crash_penalty（常數偏移）。Dense per-step risk 訊號在 episode 內變化 → 在 normalization 後仍保留方向資訊。

**結果：** 同樣 peak u259 (-0.17)，然後崩潰至 -1.78。**Reward 設計不是瓶頸。**

---

## 7. AWR Mode-Collapse 診斷（核心發現）

`models/flow_policy_v4.py:compute_weighted_loss`：

```python
weights = torch.exp(beta * advantages).clamp(max=20.0).detach()  # [0.74, 1.35×]
mse = F.mse_loss(v_pred, v_target, reduction='none').mean(dim=[1, 2])
return (weights * mse).mean()
```

**這是 Advantage-Weighted Regression (AWR)：**
- 對 **自己 rollout 的 action** 做加權 MSE 模仿
- Crash 軌跡 → 低權重 (0.74×)，**但仍會學**
- Policy 漸進吸收所有 rollout 行為，**包括 crash 行為**

**證據：** Runs 25/26/27 都呈現相同 pattern：
- u200-u280：reward 緩慢爬升（policy 學會「比平均好」的軌跡）
- u280-u500：reward 持續崩潰（policy mode-collapse 到自己的退化分佈）

無論 reward 怎麼改（Gaussian, Linear IAE, +Dense Risk），這個 pattern 都重現。

---

## 8. Run 28：Positive-Advantage Mask（進行中）

### 修正

`flow_policy_v4.py`：加 `positive_mask: bool = False` 參數。

```python
if positive_mask:
    mask = (advantages > 0).float().detach()
    num_positive = mask.sum().clamp(min=1.0)
    return (weights * mse * mask).sum() / num_positive
```

**核心改變：** 完全不在負 advantage 樣本上訓練。Policy 不再被迫模仿自己的 crash 行為，只從「比平均好」的軌跡 segment 學習。

### Config

`configs/reinflow_v4.yaml`:
```yaml
positive_advantage_mask: true
```

### 預期判讀

| 結果 | 解讀 |
|------|------|
| Training reward 持續上升不崩 | mode-collapse 已修復 |
| Hierarchical score > 0.165 (H4 BC) | **首次 RL 真正提升 v4.0 ceiling** |
| Survival > 50% | 通過 Tier 1，飛機真的會飛 |
| 仍同 Runs 25/26/27 崩潰 | 連 positive mask 都救不了 → AWR 範式對此任務無解 |

---

## 跨 Run 結果總表（2026-05-17）

| Run | 架構 | Reward 類型 | Loss 類型 | Peak Train | Peak Update | Hierarchical Score |
|-----|------|------------|-----------|-----------|-------------|-------------------|
| 19 | Original | Gaussian | weighted MSE | 0.6948 | u200 | 0.078 |
| 23 | H3a | Gaussian | weighted MSE | 0.686 | u186 | 0.093 |
| H4 BC | H4 | — | BC only | — | — | **0.165** |
| 25 | H4 | Gaussian | weighted MSE | 0.703 | u375 | 0.086 |
| 26 | H4 | Linear IAE | weighted MSE | -0.16 | u274 | 0.106 |
| 27 | H4 | + Dense Risk | weighted MSE | -0.17 | u259 | TBD |
| 28 | H4 | + Dense Risk | **+ positive mask** | — | — | **進行中** |

**目前 SOTA：** H4 BC（純 BC，無 RL），score 0.165。

---

## 關鍵 Lessons

1. **指標決定方向。** RMSE 偏誤把 v4.0 帶到錯誤方向 25 個 runs。新 hierarchical metric 之前的所有 RL 工作都需要在新框架下重新審視。

2. **架構修正比 RL 修正更有效。** H4（IMU dominance）給 +55% steps 的 BC 改善，超過任何 RL hyperparameter 調整。

3. **AWR 對 noisy rollout 脆弱。** 在 50/50 crash 環境下，AWR 必然 mode-collapse 到 rollout 分佈。需要 hard filter（positive mask）或徹底換 algorithm。

4. **PPO advantage normalization 抹平稀疏訊號。** crash_penalty=-100 在 normalization 後變成「常數」消失。Dense per-step risk 訊號才能存活。

5. **OT Flow 1-step 推論不一定最優。** 與線性插值理論預期相反，2-3 步 Euler 給 +35% steps。RL fine-tune 可能破壞 velocity field 的線性結構。

---

## 待 Run 28 完成後計畫（已中斷，轉向 v5）

| 結果 | 後續行動 |
|------|---------|
| Hierarchical score > 0.165 | 寫論文：H4 + Positive-mask AWR 是 v4.0 突破 |
| 0.10 < score ≤ 0.165 | Tune positive mask threshold，繼續調 |
| score ≤ 0.10 | 接受 H4 BC 為 SOTA，寫 negative result paper：vision-only 在此任務無法 RL improve |

---

## 9. v5 架構：Cross-Attention IMU→Vision + State Prediction Aux Loss（2026-05-18）

### 9.1 動機

**核心診斷（user + model 一致）：Information Asymmetry + Vision Feature Collapse**

- Teacher (PPO Expert) 看 15D state（pos/vel/quat/ω）；Student 只看 6×64×64 FPV image
- H4 BC 在 hover-only 1000 episodes 訓練 → vision encoder 從未見過 swift perturbation 下的 OOD 影像
- Swift perturbation 啟動時，student 把「沒看過的傾斜畫面」映射到 teacher 的「完美修正動作」，缺乏物理理解 → 訊號斷裂
- H4 IMU 主導（V/I gradient ratio 3.22×）讓 vision encoder 進一步 causal confusion（沒梯度壓力學物理）

### 9.2 兩項架構升級

**Task 1 — Cross-Attention (IMU → Vision)：** IMU 決定「看畫面哪裡」

```python
# CrossAttentionIMU2Vision (n_heads=8)
Q = Linear(imu_feat_512, 256).unsqueeze(1)            # (B, 1, 256)
K = V = spatial_map.flatten(2).T                       # (B, 16, 256) ← 4×4 spatial
attended = MultiHeadAttention(Q, K, V) → (B, 256)
global_cond = cat([attended(256), imu_feat(512)])     # 768D — 與 H4 同維度
```

**Task 2 — State Prediction Auxiliary Loss：** 強迫 vision encoder 學物理

```python
state_pred = StatePredictor(vis_pooled)  # MLP 256→256→15，僅接 pooled vision
L = L_flow + λ_state × MSE(state_pred, state_15d_normalized) + λ_tilt × MSE(tilt_pred, tilt_gt)
```

關鍵設計：
- `global_cond` 維度刻意維持 768D → H4 `flow_net` 權重 1:1 transfer，加速 BC 收斂
- `state_predictor` **僅接 `vis_pooled`（不接 IMU/attended）** → 梯度純粹施壓 vision encoder
- State target 經 PPO `obs_rms.npz` normalize → 與 teacher 輸入空間一致

### 9.3 新檔案

| 檔案 | 用途 |
|------|------|
| `models/vision_encoder_v5.py` | VisionEncoderV5 — `forward(images, return_spatial=False)` 暴露 spatial map |
| `models/flow_policy_v5.py` | FlowMatchingPolicyV5 — cross_attn + state_predictor + compute_loss(states_gt=...) |
| `configs/flow_policy_v5.yaml` | BC 設定，新增 cross_attn.n_heads / state_predictor.lambda_state |
| `configs/distillation_v5.yaml` | 蒸餾設定（lambda_state）|
| `configs/distillation_v5_smoke.yaml` | 2-update smoke test |
| `scripts/train_flow_v5.py` | BC 預訓練（支援 --transfer-from-h4） |
| `scripts/train_distillation_v5.py` | DAgger 蒸餾（rollout 多收 states_norm） |
| `scripts/evaluate_hierarchical.py`（已修改） | 新增 v5 偵測分支（檢查 `cross_attn.q_proj.weight` 鍵） |

### 9.4 v5 BC 預訓練（2026-05-18）

**Command：**
```bash
dppo/Scripts/python.exe -m scripts.train_flow_v5 \
    --transfer-from-h4 checkpoints/flow_policy_v4/20260514_175219/best_model.pt \
    --hover-only --hover-episodes 1000
```

**Transfer 結果：** 94 tensors 成功傳遞（vision_encoder.encoder.*, vision_encoder.fc.*, imu_encoder.*, flow_net.*, tilt_head.*）；`cross_attn` 與 `state_predictor` 隨機初始化。

**訓練結果：**
- Checkpoint: `checkpoints/flow_policy_v5/20260518_072501/best_model.pt`
- Best val/flow_loss = **0.06273 @ epoch 22**（與 H4 BC 相當）
- 80 epochs 完整跑完；val_loss 早期下降很快（前 30 ep），後期略有 overfit 但 best ckpt 保留 ep22
- state_aux_loss 從 ~0.8 收斂至 ~0.4（hover-only 分布內可學）

### 9.5 v5 Distillation Run 1（λ_state=0.1，2026-05-18）

**Config：** `configs/distillation_v5.yaml`，`learning_rate: 1.0e-5`，`n_rollout_steps: 4096`，`sde_noise_std: 0.05`。

**結果：** killed @ u66/200

| 指標 | u1–5 | u30 | u66 |
|------|------|-----|------|
| flow_loss | 0.32 | 0.30 | 0.29 |
| state_loss | 1.34 | 2.41 | 3.94（avg）|
| teacher_student_action_mse | 0.30 | 0.18 | 0.13 |
| crash_rate | 100% | 100% | 100% |
| mean_episode_steps | 35 | 38 | 36 |

**現象：** state_loss 在 update 19 出現極值 5.46，整體呈劇烈振盪且 trend 上升。flow_loss 收斂正常，但 vision encoder 預測的 state 越來越偏離 teacher 觀察的真實 state。

**初步診斷：** λ_state=0.1 太小，state gradient 被 flow gradient 蓋過；rot6D normalized dim std≈0.04，25° OOD tilt → ±5σ outlier，MSE 結構性大。

### 9.6 v5 Distillation Run 2（λ_state=1.0，2026-05-18）

**修正：** 確認 state target 已用 `obs_rms.normalize` 處理（[scripts/train_distillation_v5.py:90](DPPO_PID_controller/scripts/train_distillation_v5.py#L90)），將 `lambda_state` 從 0.1 升到 1.0。

**Command：**
```bash
dppo/Scripts/python.exe -m scripts.train_distillation_v5 \
    --pretrained checkpoints/flow_policy_v5/20260518_072501/best_model.pt \
    --flow-config configs/flow_policy_v5.yaml \
    --rl-config configs/distillation_v5.yaml
```

**結果：** killed @ u20/200

| 指標 | u1–5 avg | u20 avg | trend |
|------|---------|---------|-------|
| flow_loss | 0.31 | 0.34 | 略上升 |
| state_loss | 1.336 | **1.913** | **上升 43%** |
| crash_rate | 100% | 100% | flat |

**結論：拉大 λ 也救不回。**

### 9.7 根因確診：Gradient Conflict + Hover-only OOD Gap

```
                  ┌─→ flow_net (學動作)
vis_pooled (256) ─┤
                  └─→ state_predictor (學物理)
```

- `flow_loss` 和 `state_loss` 都 backprop 過 `vis_pooled → vision_encoder`
- Teacher 行動目標（flow）↔ Teacher 物理狀態（state）兩個目標的最佳 representation **不一致**
- 加上 BC 從未見過 swift perturbation 影像 → state predictor 在 OOD 影像上預測誤差**結構性無法降低**
- rot6D 在 normalized 空間 std≈0.04，OOD 傾角 25° 對應的 normalized 值 ≈ ±5σ → MSE 結構性 ~5–25 量級
- 拉大 λ_state 只會讓 vision encoder 試圖學「不可學」的目標，破壞 flow representation

### 9.8 後續方向（候選）

| 方向 | 內容 | 風險 |
|------|------|------|
| A | 先收集 swift perturbation 下的 OOD demo（hover_anchor + tilt_max=30°），混合重訓 BC，再做蒸餾 | 與 Hypothesis 2 DAgger Recovery 同類，過去曾失敗（49/50 crash） |
| B | v5 BC + RL（沿用 Run 28 positive-advantage mask），跳過蒸餾 | 仍受 AWR mode-collapse 影響 |
| C | 放大 vision encoder（feature_dim 256→512），freeze flow_net 僅蒸餾 vision/cross_attn | 改善梯度衝突，但仍受 OOD gap 限制 |
| D | 接受 v4.0 H4 BC 作為 SOTA，轉寫 negative result：「vision-only end-to-end + DAgger 蒸餾在 hover-only BC 上不可行」 | 論文難度高，但結論已紮實 |

### 9.9 關鍵 Lessons

1. **DAgger 對 OOD 影像無解。** Teacher 看 state 給的「正確」動作，需要 vision encoder 倒推物理才能學。但 BC 只在 hover 分布訓練，OOD 影像沒覆蓋 → 無 representation 可倒推。

2. **共享 encoder 的多目標 aux loss 有 gradient conflict 風險。** 設計時假設 state prediction 會「強化」vision encoder，實際上兩個 loss 的最佳 representation 不同，反而互相破壞。

3. **Normalized 空間放大 OOD 訊號。** rot6D 用單位向量分量，hover 分布下 std≈0.04，任何 tilt OOD 自動變成 ±5σ outlier，使 MSE loss 結構性無法收斂。設計 aux loss 時必須考慮目標分佈的尺度。

4. **架構 transfer 成功 ≠ 訓練成功。** 94 tensors 完美 transfer + BC val loss 達標，仍然蒸餾失敗。Architecture 對了不代表訓練 setup 對了。

---

## 10. Stage A — Normalization Fix 實驗（2026-05-19）

### 10.1 假設與動機

v5 Run 1/2 state_loss 持續上升的兩個候選原因：
- **(a) Gradient conflict** — flow / state loss 共享 vision encoder
- **(b) Normalization scale** — rot6D 在 hover-fit obs_rms 下 std≈0.04，OOD tilt = ±5σ outlier

**Plan：** 先用最便宜方式驗證 (b)。實作 `state_target_norm='partial'`：pos/vel/omega 仍用 `obs_rms.normalize`，rot6D[3:9] 保留 raw（本身已在 [-1, 1]）。Teacher 動作目標仍吃完整 normalize（保護其輸入空間不變）。

### 10.2 修改清單

| 檔案 | 修改 |
|------|------|
| `scripts/train_distillation_v5.py` | 新增 `_normalise_state(mode)` helper；`collect_rollout` 接受 `state_target_norm` 參數；teacher 呼叫維持 `obs_rms.normalize` |
| `models/flow_policy_v5.py` | `compute_loss` 新增 `state_loss_type: 'mse'|'huber'` 參數 |
| `configs/distillation_v5.yaml` | 新增 `state_target_norm: 'partial'`、`state_loss_type: 'mse'` |

### 10.3 結果（Run 3，完整 200 updates，`distillation_v5_20260518_155508`）

| Metric | u0 | u20 | u30 (swift on) | u80 | u140 | u199 | Trend |
|--------|-----|-----|----------------|-----|------|------|-------|
| state_loss | 1.10 | **0.89** | 1.84 | 3.05 | 3.96 | 2.53 | ↑ +130% |
| flow_loss | 1.94 | 0.38 | 0.34 | 0.35 | 0.29 | 0.33 | ↓ 收斂正常 |
| crash_rate | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 全程 100% |
| mean_steps | 220 | 64 | 131 | 56 | 88 | 103 | 無趨勢，震盪 |

**Curriculum 時間線：** u0–u29 hover-only（swift_prob=0）；u30+ swift_prob=0.2 啟動。

### 10.4 關鍵觀察

1. **u0–u20 hover phase：** state_loss 1.10 → 0.89（-19%）。**Partial normalization 在 in-distribution 確實有效**，rot6D scale 修正是正確方向。

2. **u30 swift_prob 啟動：** state_loss 立即從 0.89 跳到 1.84（+107% in one update），然後持續振盪上升到 ~4.0。**OOD 影像一出現立刻引爆**。

3. **flow_loss 收斂正常**（1.94→0.33），說明 flow matching 本身沒問題；state predictor 的爆炸是局部的，但足以破壞 vision encoder → crash 100%。

### 10.5 §A.3 矩陣判定

**→「仍上升」（state_loss 1.10 → 2.53）→ Stage B**

Stage A 診斷結論：
- 問題 (b) 中的 **normalization scale** 是**部分**原因（hover-only 時可改善）
- **問題 (b) 的核心是 OOD 影像覆蓋**，不是 normalization — hover-only BC 從未見過傾斜畫面，partial normalization 對 OOD phase 毫無幫助
- 問題 (a)（gradient conflict）可能仍存在，但目前從資料無法分離；OOD gap 是更顯著的失敗因子

### 10.6 Stage B 決策：Vision OOD Pretrain

確認需要進入 Stage B。OOD 影像來源待 user 確認後另開 plan，候選如下：

| 來源 | 優點 | 缺點 |
|------|------|------|
| PPO expert + swift perturbation rollout | 覆蓋傾斜畫面；expert 不會 crash | 需要跑新資料收集（~1hr） |
| `data/expert_demos_v4_recovery.h5`（已有） | 零成本；500 ep OOD rollout | Hypothesis 2 時已確認 BC 行為失敗（但**只取影像**不取 action 可能可行） |
| Random / noise policy rollout | 最快，不需 expert | 影像分布可能不夠代表真實 OOD tilt |

→ **採用 `expert_demos_v4_recovery.h5`（只取影像 + states，不用 actions）**

---

## 11. Stage B — Vision OOB Pretrain（2026-05-19）

### 11.1 設計

新腳本 `scripts/pretrain_vision_oob.py`：
- 載入 recovery.h5（500 ep × 500 steps，OOD tilt 影像 + states）
- 載入 v5 BC checkpoint 作為起點
- **凍結**：`imu_encoder`、`cross_attn`、`flow_net`、`tilt_head`
- **只更新**：`vision_encoder`（456k params）+ `state_predictor`（70k params）= 526k params
- Loss: `MSE(state_predictor(vis_pooled), _normalise_state(s, obs_rms, 'partial'))`
  - rot6D[3:9] 保留 raw（Stage A 確認此 fix 在 hover 分布有效）
  - pos/vel/omega 仍用 obs_rms.normalize
- 90/10 train/val split；60 epochs；lr=1e-4 cosine decay

### 11.2 結果（`pretrain_vision_oob/oob_20260519_002605`）

| 指標 | ep1 | ep16 (best) | ep60 (final) |
|------|-----|-------------|--------------|
| val_state_loss | 0.229 | **0.194** | 0.219 |
| train_state_loss | 0.308 | 0.132 | 0.057 |
| val_pos_loss | 0.028 | 0.017 | — |
| **val_rot6d_loss** | 0.0062 | **0.0032** | 0.0032 |
| val_vel_loss | 0.168 | 0.151 | — |
| **val_omega_loss** | 0.934 | **0.920** | 0.920 |

**val_state_loss = 0.194，遠低於 0.5 門檻 → Stage B pretrain 成功。**

關鍵發現：
- `val_rot6d_loss = 0.003` — 這是 Stage A 和 Run 1/2 爆炸的維度，現在幾乎完美。Vision encoder 已學會從 OOD 傾斜影像讀出旋轉資訊。
- `val_omega_loss ≈ 0.92` — 高，但在預期中：靜態影像估角速度需要 temporal motion，而且 IMU 本來就直接測量 omega，distillation 不需要 vision encoder 解決這個維度。
- ep16 之後 val loss 略微 overfit（val 0.194→0.219，train 繼續下降至 0.057）— best_model.pt 保存 ep16 ✓

**Best checkpoint：** `checkpoints/pretrain_vision_oob/oob_20260519_002605/best_model.pt`

### 11.3 Distillation Run 4（已啟動，2026-05-19）

用 OOB pretrained encoder 熱啟動蒸餾：
```
dppo/Scripts/python.exe -m scripts.train_distillation_v5 \
    --pretrained checkpoints/pretrain_vision_oob/oob_20260519_002605/best_model.pt \
    --flow-config configs/flow_policy_v5.yaml \
    --rl-config   configs/distillation_v5.yaml
```
Config：`state_target_norm='partial'`、`lambda_state=1.0`、curriculum `swift_prob=0.2`

**預期判讀矩陣：**

| 結果 | 解讀 | 下一步 |
|------|------|--------|
| state_loss 前 30 updates 保持 <0.5 | pretrain 遷移成功，OOD 覆蓋有效 | 跑完 200 updates，做 hierarchical eval |
| state_loss 前 30 updates 仍上升 | OOD pretrain 夠，但 DAgger rollout 分布更 OOD | 考慮 curriculum 延長 hover phase（n_hover_updates: 30→60）|
| crash_rate 降到 <50% 且 score > 0.165 | **v5 突破 H4 BC SOTA** | 寫 Stage B 勝利結論 |
| crash_rate 仍 100%，state_loss OK | state_predictor 解碼成功但不影響 action 品質 | 考慮 stop-gradient（此刻才有意義）作為 Stage C |

### 11.4 Distillation Run 4 完整結果（200 updates）

| 指標 | 全程均值 | 最小 | 最大 | 最後 |
|------|---------|------|------|------|
| state_loss raw | 2.367 | **0.192** | 4.116 | 1.910 |
| flow_loss | 0.347 | 0.223 | 1.939 | 0.351 |
| crash_rate | 99.8% | **85.7%** | 100% | 100% |
| mean_episode_steps | 108 | 34 | **284.5** | 171 |

**最佳 update：** u92（crash=0.857，steps=284.5）；`checkpoints/distillation_v5/distillation_v5_20260519_004011/best_distillation_model.pt` 保存。

**Rolling-10 state_loss 趨勢：** 0.98(u0) → 1.32(u20) → 2.44(u80) → 2.92(u180)
**判定：state_loss 仍持續上升**（與 Run 1/2/3 同類），OOB pretrain 的知識被 DAgger 的 state_loss 振盪逐漸侵蝕。

**關鍵觀察：**
- u10 觸及 state_loss=0.192（matching pretrain val=0.194）→ **pretrain 遷移確實成功**
- u92 steps=284 > H4 BC 202 → **OOB pretrained 特徵比原始 v5 BC 更好**
- 但 state_loss 振盪 0.19↔4.1 → rolling mean 持續上升 → 逐漸侵蝕 vision encoder

**根因：** λ_state=1.0 的 state_loss 在每次 DAgger rollout 後持續破壞 OOB pretrained 的 vision encoder representation。每次 rollout 分布不同（有時 34 steps crash，有時 284 steps）→ optimizer 把 vision encoder 拉向矛盾方向。

**決策：→ Stage C（freeze vision_encoder + λ_state=0）**

### 11.5 v5 OOB Pretrain BC Evaluation（2026-05-19）

**插入評估：** Run 4/5 起點 u0 的 mean_steps=232–235 超過 H4 BC 202，提示 OOB pretrained checkpoint 本身作為 BC 模型可能已有改善。立刻對 `pretrain_vision_oob/best_model.pt` 做 hierarchical eval（20 episodes）。

| 排名 | Model | Score | Survive | IAE_st | Term | Steps avg |
|------|-------|-------|---------|--------|------|-----------|
| 1 | **H4 BC** | **0.171** | 39.9% | 1.418m | 2.544m | ~197 |
| 2 | v5 OOB pretrain BC | 0.112 | **49.0%** | 1.773m | 3.074m | ~244 |

**OOB pretrain BC：score=0.112 < H4 BC 0.171，但 survive=49% > H4 BC 40%**

**關鍵發現——行為改變了：**
- v5 OOB 有 7/20 episodes score=0.000（survive > 50%，但 IAE_st > 2.0m → stability_score 被 clip 到 0）
- H4 BC 有 2/20 episodes score=0.000
- OOB pretrain 讓飛機**飛更久**（49% survive > 40%），但**漂移更多**（IAE_st 1.77 > 1.42）

**根因診斷：**

```
OOB pretrain 只訓練了 vision_encoder + state_predictor
flow_net 仍然是 v5 BC 的 hover-only hover weights
→ vision encoder 學會「感知傾斜」
→ 但 flow_net 不知道「遇到傾斜時要做什麼」
→ drone 存活更久（偵測到危險）但無法精確恢復（action 仍為 hover-tuned）
```

這個行為分裂（survive↑, stability↓）提示：**OOB pretrained encoder 是正確方向，但 flow_net 需要對應更新**。DAgger 嘗試做這件事，但因為 100% crash rollout 的 distribution mismatch 而失敗。

---

## 12. Stage C — Freeze Vision Encoder + Pure DAgger（2026-05-19）

### 12.1 假設

OOB pretrain 已給 vision encoder 足夠的 OOD 物理解碼能力（val=0.194，rot6D_loss=0.003）。
問題是 DAgger 訓練期間 λ_state=1.0 的振盪梯度在侵蝕這個能力。

**修正：** 蒸餾期間完全凍結 `vision_encoder`，禁用 `state_loss`（λ=0）。
- Flow matching loss 只更新 `flow_net` + `cross_attn`
- Vision encoder 的 OOB pretrained weights 保持不動
- 這是「OOB pretrained encoder 提供特徵，flow_net 學習用這些特徵生成動作」的最乾淨測試

### 12.2 實作

`configs/distillation_v5.yaml` 新增：
```yaml
freeze_vision_encoder: true
lambda_state: 0.0
```

`scripts/train_distillation_v5.py` 在載入 pretrained 後：
```python
if distill_cfg.get('freeze_vision_encoder', False):
    for p in policy.vision_encoder.parameters():
        p.requires_grad = False
    print("vision_encoder FROZEN")
policy_opt = AdamW(filter(lambda p: p.requires_grad, policy.parameters()), lr=...)
```

### 12.3 Run 5 完整結果（200 updates）

| 指標 | 全程均值 | 最小 | 最大 | rm10_last |
|------|---------|------|------|-----------|
| flow_loss | 0.356 | 0.255 | 1.920 | **0.337** |
| crash_rate | 99.6% | **85.7%** | 100% | 100% |
| mean_steps | 115 | 33 | **292.6** | 135 |

**最佳 updates：** u78（crash=0.857，steps=263），u187（crash=0.857，steps=292）

**對比 Run 4：** mean_steps 均值 108→115（略升），最大 284→292，crash_rate min 兩者都是 0.857。**Stage C 沒有打破 crash 循環**。

### 12.4 DAgger 範式失敗結論

五次蒸餾（Run 1–5）+ 三個階段優化，crash_rate 全程 99%+，結果總結：

| 問題 | 嘗試過的修正 | 結果 |
|------|------------|------|
| Normalization ±5σ | partial norm（Stage A） | 在 hover 分布有效，OOD 仍炸 |
| OOD 影像無覆蓋 | OOB pretrain on recovery.h5（Stage B） | pretrain val=0.194，蒸餾 state_loss 仍振盪 |
| Gradient conflict | freeze encoder + λ=0（Stage C） | flow_loss 收斂最乾淨，但 crash 100% |

**根本瓶頸：DAgger 的 100% crash rollout distribution。** 學生全程崩潰 → teacher 標記的全是 crash 狀態下的「正確動作」→ flow_net 學會「在 crash 時怎麼做」，但從未學到「不要 crash」。這不是 normalization、encoder、或 gradient 問題——是 DAgger 的 distribution mismatch 在 100% crash 環境下的結構性失敗。

**但發現了一個新方向：**
OOB pretrained BC 評估（§11.5）顯示 v5 OOB encoder 讓 survive 從 40% 升到 49%，但 stability 下降（IAE_st 1.42→1.77）。這個分裂清楚指出：**vision encoder 對了（OOD-aware），但 flow_net 的 hover action 和新 encoder 特徵空間不對齊**。

**→ Stage D：BC retrain flow_net only with frozen OOB encoder**

---

## 13. Stage D — BC Retrain flow_net + cross_attn（凍結 OOB Encoder）（2026-05-19）

### 13.1 假設

OOB pretrained encoder 的特徵空間已從 v5 BC 時代的「hover-only visual features」轉變為「OOD-aware physical features」。原有的 flow_net（v5 BC weights）是在舊特徵空間下訓練的，兩者已不對齊，導致 action quality 下降（stability↓）。

**修正：** 凍結 OOB encoder，在 hover-only 資料上重新對齊 flow_net + cross_attn。
- vision_encoder：凍結（保護 OOD 物理解碼能力）
- flow_net + cross_attn：從頭 BC fine-tune，對齊新 encoder 特徵
- 訓練資料：hover-only 1000 episodes（同原始 v5 BC）

**預期效果：** survive 維持 OOB encoder 帶來的 49%（甚至更高），同時 IAE_st 回到 1.42m 以下，hierarchical score 超過 H4 BC 0.171。

### 13.2 實作

**修改：** `scripts/train_flow_v5.py` 新增 `--pretrained` + `--freeze-vision` flag：
- `--pretrained`：載入完整 checkpoint（不限於 H4 transfer）
- `--freeze-vision`：凍結 `vision_encoder.parameters()`，optimizer 只更新 `requires_grad=True` 的參數

**凍結後參數分配：**
| 模組 | 狀態 | 參數量 |
|------|------|--------|
| vision_encoder | FROZEN | 456,032 |
| imu_encoder | trainable | ~532k |
| cross_attn | trainable | — |
| flow_net (UNet) | trainable | — |
| state_predictor | trainable | — |
| tilt_head | trainable | — |
| **trainable total** | | **13,047,320** |

### 13.3 啟動設定

```
dppo/Scripts/python.exe -u -m scripts.train_flow_v5 \
    --pretrained checkpoints/pretrain_vision_oob/oob_20260519_002605/best_model.pt \
    --hover-only --hover-episodes 1000 --freeze-vision
```

- Config: `configs/flow_policy_v5.yaml`（num_epochs=80, batch=256, LR=1e-4）
- lambda_state=0.1, lambda_tilt=0.2（state predictor 同步對齊）
- Save dir: `checkpoints/flow_policy_v5/20260519_115655`
- 啟動時間：2026-05-19，task `bho3muv55`

### 13.4 判讀矩陣

| val_flow_loss 趨勢 | survive | IAE_st | 結論 |
|---|---|---|---|
| 下降 → ≤0.065（原 v5 BC 水準） | ≥49% | ≤1.42m | ✓ 對齊成功，score > 0.171 → v5 突破 |
| 下降但 >0.065 | ≥40% | 1.4-2.0m | 部分成功，可能需要更長訓練 |
| 不收斂 / 上升 | — | — | OOB encoder 與 hover 資料不相容 |

### 13.5 Stage D 結果（2026-05-19）

訓練在 ep40/80 提前停止（val_flow 振盪無改善，best_model.pt 鎖在 ep1）。

**val_flow 訓練歷程：**

| Epoch | val_flow | train_flow | LR |
|-------|----------|------------|----|
| 1 | **0.06098** (best) | 0.06164 | 2e-5 (warmup) |
| 5 | 0.06329 | 0.06151 | 1e-4 (peak) |
| 10 | 0.06288 | 0.06166 | 9.89e-5 |
| 20 | 0.06349 | 0.06039 | 9.05e-5 |
| 30 | 0.06240 | 0.05921 | 7.50e-5 |
| 40 | 0.06407 | 0.05775 | 5.52e-5 |

val_flow 在 0.062-0.065 振盪；train_flow 持續下降但 val 無跟隨 → best_model.pt = ep1（LR=2e-5，幾乎未訓練）。

**Hierarchical Eval（20 episodes）：**

| 模型 | Score | Survive | IAE_st | Term |
|------|-------|---------|--------|------|
| **H4_BC** | **0.167** | 39.1% | 1.450m | 2.617m |
| v5_OOB_pretrain | 0.140 | 51.5% | 1.407m | 2.441m |
| **v5_StageD_best** | **0.073** | 55.2% | 2.259m | 3.852m |

**Stage D 假設否定。**

### 13.6 v5 Pipeline 完整失敗分析

**Stages A→D 核心教訓：**

| Stage | 修正目標 | 結果 |
|-------|---------|------|
| A (partial norm) | rot6D ±5σ 爆炸 | state_loss 改善但 OOD 仍崩 |
| B (OOB pretrain) | vision encoder 無傾斜覆蓋 | val_state=0.194 ✓，但流 matching 不對齊 |
| C (frozen + λ=0) | gradient conflict | flow_loss 收斂，但 DAgger 100% crash |
| D (BC retrain flow_net) | flow_net 與 encoder 不對齊 | val_flow=0.061，但 score=0.073 < OOB 0.140 |

**根本問題：兩個特徵空間不相容**

OOB encoder 在 state regression 任務上優化（輸出應對應物理狀態），而 flow_net 在 action generation 任務上需要不同的特徵分布。即使 BC fine-tuning，兩者的最優 representation 仍然不同。ep1 的 1 個 epoch（LR=2e-5）不足以對齊 flow_net，反而比純 OOB pretrain 更差（IAE_st 1.407→2.259m）。

**架構層面的瓶頸：**
- H4 的 IMU-dominant 設計（512D IMU, 256D vision）天然適合 hover 穩定
- v5 的 cross-attention 設計試圖讓 vision 更主動參與，但當 encoder features 不是「action-optimal」時，cross-attention 無法補救
- DAgger 蒸餾依賴「student 能生成合理 trajectory 以收集有效 teacher label」，在 100% crash 環境下結構性無解

**v5 架構未能超越 H4 BC 的結論：**
- 在本研究的訓練資料範圍（hover-only + recovery）與架構選擇（Cross-Attention + State Aux）下，v5 pipeline 無法解決「vision encoder 特徵空間與 action generation 的 alignment 問題」
- H4 BC（score 0.167-0.171）確認為 v4.0 SOTA

---

## 14. 最終研究結論（2026-05-19）

### 14.1 v4.0 SOTA 確認

**H4 BC（IMU 512D dominant）= v4.0 最終 SOTA**

| 指標 | 值 |
|------|-----|
| Hierarchical score | 0.167–0.171 |
| Survive rate | ~39-44% |
| IAE_st | ~1.42-1.47m |
| 架構 | IMU 512D + Vision 256D + flow matching |
| Checkpoint | `checkpoints/flow_policy_v4/20260514_175219/best_model.pt` |

### 14.2 五條核心發現（研究貢獻）

1. **RMSE 是誤導性指標**：episode 長度不歸一化造成系統性偏差，推翻過去 24 runs 的排名。
2. **IMU 主導優先於 Vision**：H4（V/I grad ratio 3.22×）比 H3a（9.9×）步數多 55%（202 vs 130）；越平衡越好。
3. **AWR mode-collapse 診斷**：Advantage normalization 在 50/50 crash 環境下抹平 crash penalty，所有 RL 嘗試（27 runs）系統性退化。
4. **DAgger 在 100% crash 環境下結構性失敗**：teacher label 基於 crash trajectory，flow_net 學習「如何在墜落中行動」而非「如何避免墜落」。
5. **Encoder 特徵空間與 Action Generation 的 Alignment 難題**：state prediction 最優特徵 ≠ action generation 最優特徵；分離預訓練無法解決此問題。

### 14.3 後續方向建議

若要繼續突破 H4 BC 0.171：
- **E2E 訓練**：從頭訓練 v5 架構（含 OOD 傾斜影像）而非分離預訓練，讓 encoder 和 flow_net 在同一任務下聯合優化
- **模型蒸餾替代方案**：直接在 BC 訓練中引入 expert 動作標籤（而非 DAgger rollout），避免 crash distribution 污染
- **RL 在 H4 BC 上的新嘗試**：解決 AWR mode-collapse 的根本問題（positive-advantage mask 或 PPO-style clip）
