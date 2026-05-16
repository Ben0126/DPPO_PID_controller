# v4.0 Dev Log — H4 Architecture & Hierarchical Evaluation Era

**繼續自：** [dev_log_v4_post20.md](dev_log_v4_post20.md)（Runs 1–20 + Hypothesis 1–3a）
**起始日期：** 2026-05-13
**狀態（截至 2026-05-17）：** **H4 BC = v4.0 真正 SOTA**（hierarchical score 0.165）。RL fine-tuning 在 27 個 runs 中全部 systematically destroy BC policy。Run 28 進行中。

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
| 28 | **Positive-advantage mask** (硬遮罩 adv<0) | 2026-05-17 | **進行中** — 修正 AWR mode-collapse |

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

## 待 Run 28 完成後計畫

| 結果 | 後續行動 |
|------|---------|
| Hierarchical score > 0.165 | 寫論文：H4 + Positive-mask AWR 是 v4.0 突破 |
| 0.10 < score ≤ 0.165 | Tune positive mask threshold，繼續調 |
| score ≤ 0.10 | 接受 H4 BC 為 SOTA，寫 negative result paper：vision-only 在此任務無法 RL improve |
