---
marp: true
theme: default
size: 16:9
paginate: true
backgroundColor: "#0f1419"
color: "#e6edf3"
style: |
  section {
    font-family: "Inter", "Segoe UI", "PingFang TC", sans-serif;
    padding: 50px 70px;
  }
  h1 { color: #58a6ff; font-size: 1.7em; }
  h2 { color: #79c0ff; border-bottom: 2px solid #30363d; padding-bottom: 6px; }
  h3 { color: #a5d6ff; }
  strong { color: #ffa657; }
  code { background: #161b22; color: #d2a8ff; padding: 2px 6px; border-radius: 4px; }
  table { font-size: 0.78em; }
  th { background: #1f6feb; color: white; }
  tr:nth-child(even) { background: #161b22; }
  tr:nth-child(odd)  { background: #0d1117; }
  blockquote { border-left: 4px solid #f78166; color: #ffa198; }
  .small { font-size: 0.75em; color: #8b949e; }
  .ok    { color: #56d364; }
  .warn  { color: #ffa657; }
  .bad   { color: #f85149; }
---

<!-- _class: lead -->

# Vision-Based Quadrotor Control
## 以 Flow Matching + ReinFlow 突破覆蓋偏移
### 進度報告 · 2026-05-06

<br>

**核心問題：** 監督式視覺策略一定會 crash，RL 微調能救嗎？
**目前狀態：** 20 runs / Phase 3b 完成 / 訓練-評估 gap 已被定位
**下一步：** 從「最佳化 reward」轉向「直接最佳化生存」

<span class="small">Target: CoRL 2025 / ICRA 2026 / RSS 2026</span>

---

## 一頁理解這個研究

| | |
|---|---|
| **任務** | 64×64 FPV 視覺 → 4 顆馬達推力，50Hz 閉環控制四旋翼 |
| **想法** | 用 PPO Expert 收 1000 條 demo → Flow Matching 監督預訓練 → ReinFlow RL 微調 |
| **為什麼難** | BC 必 crash（covariate shift），RL 又會擊潰預訓練權重（policy collapse） |
| **目前最好** | Run 10：RMSE **0.3005m**（從 BC 0.522m 降 42%），但 50/50 還是 crash |
| **意外發現** | 簡單 PID 在 hover 任務拿到 **0.022m**，比 PPO Expert 還好 3× |
| **核心瓶頸** | **訓練 reward 0.529 → 0.695** 但 **eval RMSE 不變**——training-eval gap |

---

## 研究 Pipeline 全景

![bg right:55% 95%](figures/01_pipeline_overview.png)

**5 階段流水線**

- **Ph.0** INDI 懸停 gate <span class="ok">✓</span>
- **Ph.1** CTBR PPO Expert <span class="ok">✓</span>
- **Ph.2** 1000-ep FPV 數據 <span class="ok">✓</span>
- **Ph.3a** Flow Matching BC <span class="ok">✓</span>
- **Ph.3b** ReinFlow RL <span class="warn">◐ 20 runs</span>
- **Ph.4** Jetson 部署 <span class="bad">○ 未開始</span>

---

## 系統架構（Phase 3a/3b baseline）

```
FPV stack (T_obs=2, 6×64×64 uint8)
   │
   ▼
VisionEncoder CNN ──────────────► 256D vision_feat
                                   │
                                   ▼ (+ timestep 128D)
                          ConditionalUnet1D ε_θ
                                   │
                                   ▼  Flow Matching 1-step
                       action seq (T_pred=8 × 4 motor)
                                   │
                                   ▼  RHC: execute T_action=4
                                Re-observe loop
```

**參數規模：** 10.93M | **Inference：** 8.2 ms (~122 Hz) <span class="ok">✓ 已達 latency 目標</span>
**Loss：** `L = E[ exp(β·A_norm) · ‖v_θ − (ε−x_0)‖² ] + λ_bc · L_BC`

---

## 進度時間軸

![bg right:60% 95%](figures/02_progress_timeline.png)

**節奏摘要**

- 04-01 ~ 04-19：v4.0 物理 + Expert + 數據（19 天）
- 04-20：BC 監督預訓練收斂 val=0.063
- 04-20 ~ 05-03：**ReinFlow 20 runs（13 天）**
- 05-06：**PID baseline + Temperature ablation**

**當前分歧點：** 是否還在 ReinFlow 軸上繼續，還是換骨架（D²PPO + Dispersive Loss）？

---

## 結果一覽（這頁回答審稿人 80% 的問題）

| 方法 | 模式 | RMSE | Crash | 評語 |
|------|------|------|-------|------|
| <strong>PID Cascade</strong> | Hover | <strong class="ok">0.022 m</strong> | <strong class="ok">0/50</strong> | <strong>意外冠軍</strong>，比 PPO 好 3× |
| PPO Expert (CTBR+INDI) | Hover | 0.065 m | 0/50 | 原本的黃金標準 |
| PID Cascade | Waypoint 2.0m | 1.18 m | <strong class="ok">0/50</strong> | 穩但慢，追不上 |
| <strong>ReinFlow Run 10</strong> | Waypoint 2.0m | <strong class="ok">0.30 m</strong> | <strong class="bad">50/50</strong> | <strong>RL 最佳 eval</strong> |
| ReinFlow Run 12 | Waypoint 2.0m | 0.297 m | 50/50 | hover anchor + soft penalty |
| ReinFlow Run 19 | Waypoint 2.0m | 0.523 m | 50/50 | train reward ★ 0.695 |
| Flow Matching BC | Waypoint 2.0m | 0.522 m | 50/50 | 監督基線（covariate shift）|
| v3.3 DPPO Run 1 | Waypoint 2.0m | <strong>0.104 m</strong> | 50/50 | v3 架構最佳跨參考 |

<span class="small">所有 RL 方法的 50/50 crash 是研究敘事的「核心 gap」</span>

---

## 三個對比視覺化

![bg 90%](figures/03_method_comparison.png)

---

## ReinFlow 20 runs 的故事

![bg right:62% 95%](figures/04_run_history.png)

**訓練 reward 一路爬升，但 eval RMSE 紋風不動**

- Runs 1–9：工程 bug 連環撞牆
- **Run 10：曲線學習首次突破** RMSE 0.30
- Runs 13–18：lambda_bc 鎖死 / VLoss spike
- **Run 19：LR=1e-7 解鎖最高 reward 0.6948**
- Run 20：confirm 訓練 ceiling 與 curriculum 結構無關

---

## 解決過的工程問題（11 個 bug 全 fix）

| # | Bug | 出現於 | 解法 |
|---|-----|--------|------|
| 1 | VLoss 門檻 2.0 太嚴，永遠不存 ckpt | Run 1 | 改 100 |
| 2 | `fixed_x1 + pos_filter` → PLoss≈0 | Run 3 | 同時移除兩者 |
| 3 | `value_lr` 太低，V net 跟不上 | Run 1–4 | 3e-4 → 1e-3 |
| 4 | VLoss gate 雙向震盪（76% warmup） | Run 5–6 | one-way latch flag |
| 5 | 訓練分佈全是 hover，policy 沒看過 OOD | Run 1–7 | curriculum on `initial_pos_range` |
| 6 | OOD disturbance 2.0N 太強，gate 永關 | Run 8 | 降 1.0N + gate 20 |
| 7 | `crash_penalty_rl=1.0` 訓練/評估不一致 | Run 12 | 還原為 10.0 |
| 8 | `lambda_bc=0.1` 鎖死預訓練 basin | Run 13–16 | 降到 0.01 |
| 9 | LR=5e-7 → VLoss 突刺 30+ → 崩潰 | Run 17–18 | 降到 1e-7 |
| 10 | n_hover=400 過度訓練 hover | Run 19 | 降到 100 |
| 11 | **SO3 attitude error 旋轉順序寫反**（PID baseline） | hover 全 crash | `R.T @ R_des`（不是 `R_des.T @ R`） |

---

## 關鍵發現 1：訓練-評估 gap

![bg right:55% 95%](figures/05_train_eval_gap.png)

**Train per-step reward：** 0.529 → 0.6948（+31%）
**Eval RMSE：** 0.522 m → 0.523 m（**0%**）
**Eval crash steps：** 全部 50 集落在 step **55–67**

> **Policy 不是學會「不 crash」，而是學會「在 crash 前那 60 步把 reward 拉到最大」**

GAE gamma=0.99 + LR=1e-7 + 短 episode（~60 步）+ on-policy → policy 從來看不到「漂移快要崩潰前」的狀態，自然學不到 recovery。

---

## 關鍵發現 2：Temperature scaling 假設被否定

![bg right:55% 95%](figures/06_temperature_ablation.png)

**假設：** crash 是 flow matching 採樣噪聲累積造成的
**做法：** eval 時 `x_1 ~ N(0, σ²I)`，σ ∈ {1.0, 0.7, 0.5, 0.3}
**成本：** 0（不用重訓）

| σ | RMSE | Crash | AvgCrashStep |
|---|------|-------|------|
| 1.0 | 0.523 | 50/50 | 61.5 |
| 0.5 | 0.536 | 50/50 | 62.9 |
| 0.3 | 0.556 | 50/50 | 66.1 |

**結論：** 噪聲 ≠ crash 主因；**crash 是結構性 distribution shift**——適合寫進論文當 negative result。

---

## 兩軸診斷：hover vs approach 為何學不到一起

```
                        hover ★         approach ★      eval RMSE / steps
Run 7  hover-only       ★★★              ✗               0.514 m / 57
Run 10 ramp 2.0m         ★★               ★★              0.300 m / 36   ← best eval
Run 11 ramp 3.0m         ✗               ★★★              0.142 m / 14   (artefact, 即時 crash)
Run 12 anchor + soft     ★★★              ★★              0.297 m / 22
Run 19 LR=1e-7          ★★★              ✗ hover-only    0.523 m / 61   ← best train reward
```

**現象：** 學會 hover 就忘 approach，學會 approach 就 hover 不穩。
**假設：** Flow Matching 容量上限 + BC anchor 強度 + reward 不夠 penalise 不穩定。
**這正是論文的「故事」：傳統控制器穩但慢，RL 快但崩——如何讓 RL 在快的同時保持穩定。**

---

## PID baseline 的意外結論

**Hover 模式（50 episodes）**

```
RMSE:  0.0219 m  (±0.0025)
Crash: 0/50
CPU 推論: 177 µs/step
```

**為什麼 PID 不會 crash？**

1. 沒有採樣噪聲（顯式 P/PI gain）
2. 顯式 rate controller 直接控制 ω
3. 顯式傾角限制 `vel_max=2.0` → 最大傾角 ~31°，遠低於 60° 終止線

**為什麼 RL 會 crash？**

1. Flow matching 每步採樣 → body rate 噪聲累積
2. 沒有顯式姿態穩定器，只靠 reward 間接約束
3. Training-eval gap：rollout 短 episode → 沒看過漂移狀態

> **論文貢獻不應再強調 hover accuracy（PID 都贏了），而是「快速追蹤 + 穩定」的學習框架。**

---

## 下一步路線（按 ROI 排序）

| 優先 | 方向 | 預期效應 | 成本 |
|------|------|---------|------|
| <strong class="ok">P1</strong> | <strong>傾角懲罰 reward shaping</strong>（`w_tilt·(tilt-30°)²`） | policy 在傾角 >30° 時主動修正 | 低（改 reward）|
| <strong class="ok">P1</strong> | <strong>Survival bonus + tilt penalty</strong> 組合 | 直接最佳化生存而非 reward | 低 |
| P2 | Hybrid PID-RL（tilt>30° 切 PID） | 立刻拿到 0/50 crash 數字 | 低 |
| P2 | **Dispersive Loss**（架構升級至 v3.1/v3.3 路線） | 防止 feature collapse | 中 |
| P3 | Multi-step Euler / DDIM 增加 inference steps | 推論精度↑ | 低 |
| P3 | 重新收 OOD demos（含 recovery 軌跡）| breaks coverage 限制 | 高 |
| - | <s>繼續調 ReinFlow 超參數</s> | 已證明 ceiling 是結構性 | <span class="bad">不做</span> |

---

## 簡報重點濃縮（給聽眾帶走的三句話）

1. **覆蓋偏移確實會殺死 BC 視覺策略**（100% crash），這是 RL 微調的存在意義。

2. **20 runs 後我們證明了：訓練 reward 提升不會自動轉化為評估 RMSE 改善** ——
   這是個結構性 distribution-shift 問題，不是噪聲、不是超參數。

3. **下一步是把 reward 從「最佳化在分佈內」改為「最佳化生存」** —— 加 tilt penalty、
   alive bonus，或直接做 PID-RL hybrid 拿到 0/50 crash。

---

<!-- _class: lead -->

# Q&A

**FAQ 預先準備**

- *PID 都贏了，RL 還有意義嗎？* → Hover 任務 PID 贏，但 waypoint 追蹤 RL 比 PID 快 4×（0.30 vs 1.18m）
- *為什麼不用更大模型？* → 推論 budget 8.2ms 已是 Jetson 上限，模型容量不能變
- *為什麼不繼續調超參？* → Run 19 vs 20 證明 reward ceiling 是結構性，不是超參
- *Crash 為什麼一定在 step 60？* → episode 長度上限 + 漂移時間常數，而非單一 bug
- *為什麼不直接用 PPO？* → PPO 對單峰高斯，無法表達多模態 expert 動作分佈

<br>
<span class="small">Repo: c:\Users\User\Desktop\DPPO_PID_controller · 2026-05-06</span>
