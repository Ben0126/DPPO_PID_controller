# Gemini / Nano Banana / ChatGPT 圖像生成 Prompts

把這些 prompt 直接貼給多模態圖像生成模型，可以獲得高品質的系統圖。
所有 prompt 已對齊本專案實際數據（路徑、checkpoint、超參數）。

---

## A. 完整研究藍圖（建議橫式 16:9 海報，可作為簡報封面）

```
Generate a high-resolution "research project blueprint" diagram, 16:9 horizontal,
dark navy background with neon-green/cyan accent borders, monospaced code snippets
inside each box. The diagram has a 4-tier vertical layout (Phase 0..4) with the
title at the top center.

=== TITLE ===
"DPPO PID Controller — Research Roadmap (2026-04 to 2026-05)"
Subtitle: "Vision-based quadrotor control · Flow Matching + ReinFlow · CoRL 2025 / ICRA 2026"

=== TIER 1 · Foundation (Phase 0-2) ===
[BOX] Phase 0  INDI Hover Gate    [DONE]
   tilt 0.00°, omega 0 rad/s
   gate criterion before data collection

[BOX] Phase 1  CTBR PPO Expert    [DONE]
   stable-baselines3 PPO + RunningMeanStd
   RMSE 0.065m | 0/50 crashes
   ckpt: ppo_expert_v4/20260419_142245

[BOX] Phase 2  FPV Data Collection    [DONE]
   1000 episodes, 500k steps, 3.9 GB, 0 crashes
   data/expert_demos_v4.h5

=== TIER 2 · Supervised Pre-training (Phase 3a) ===
[BOX] Flow Matching Policy v4    [DONE]
   VisionEncoder CNN → 256D
   ConditionalUnet1D + timestep(128D)
   1-step linear flow: x_t = (1-t)·x_0 + t·ε
   val loss 0.063 → eval RMSE 0.522m, 50/50 crash (covariate shift)
   ckpt: flow_policy_v4/20260420_034314

=== TIER 3 · ReinFlow RL Fine-tune (Phase 3b)  HIGHLIGHTED ===
[BIG BOX] 20 runs across 13 days (04-20 → 05-03)
   L_RL = exp(β·A_norm) × ||v_θ(x_t,t,c) − (ε−x_0)||²  +  λ_bc · L_BC
   GAE: γ=0.99, λ=0.95
   Best train reward: 0.6948 @ u200 (Run 19)
   Best eval RMSE: 0.3005m (Run 10, curriculum 2.0m)
   Crash rate: 50/50 universal across all 20 runs

   Sub-tags showing key inflection runs:
     Run 1   β=0.1 LR=5e-6     collapse
     Run 7   one-way VLoss gate    fixed
     Run 10  curriculum pos→2.0m  ★ first real eval drop
     Run 12  hover anchor + soft penalty
     Run 19  LR=1e-7 → reward ceiling 0.6948

=== TIER 4 · Diagnostics & Baselines (2026-05-06) ===
[BOX] PID Cascade (Surprise winner)
   Hover: 0.022m / 0/50 crash  (beats PPO Expert!)
   Waypoint: 1.18m / 0/50 (stable but slow)
   Bug fixed during impl: SO3 attitude error (R.T @ R_des)

[BOX] Temperature Scaling (negative result)
   σ ∈ {1.0, 0.7, 0.5, 0.3}, all 50/50 crash
   Hypothesis "noise causes crash" rejected
   Crash is structural distribution shift, not noise

=== ARROWS ===
Ph.0 → Ph.1 → Ph.2 → Ph.3a → Ph.3b
Ph.3a (dashed red) → "covariate shift, BC fails"
Ph.3b (solid orange) → Ph.4 "Jetson deploy (planned, 122Hz target met)"
Diagnostics box (dotted green) ↑ "informs next reward shaping"

=== STYLE ===
- Background: deep navy (#0f1419) with subtle grid
- Phase labels rotated sideways on the left margin
- Status badges per box: [DONE] green, [20 runs] amber, [PLANNED] red
- Code/config snippets in monospaced light-purple
- Aspect 16:9, sharp text, high DPI, no decorative drones
```

---

## B. 訓練-評估 gap 概念圖（用於展示核心問題）

```
Generate a clean academic-style schematic illustrating "the training-evaluation
distribution gap in short-horizon RL fine-tuning", 16:9 dark navy background.

Left panel (Train rollout):
- A horizontal trajectory of ~60 dots representing time steps
- Steps 0-58 colored bright cyan (positive reward 0.86/step)
- Step 60 colored deep red (crash, reward -10)
- Annotation arrows: "policy never sees what comes after step 60"
- Label: "Train rollout — short, hover-dominated"

Right panel (Eval episode):
- Same 60-step trajectory, but with a sub-trajectory drawn ABOVE showing
  drift accumulation (slowly diverging path)
- Steps 50-60 marked with progressively larger red glow
- Crash circle at step 60 with explosion marker
- Annotation: "Drift compounds for 60 steps → crash window 55–67"
- Label: "Eval episode — same crash signature regardless of training reward"

Center vertical divider with text:
"Training reward:  0.529 → 0.6948  (+31%)"
"Eval RMSE:        0.522 → 0.523  (+0%)"
"Crash count:      50/50 → 50/50"

Bottom caption (yellow):
"Policy learns to maximise pre-crash reward, not to avoid crashing."

Style: minimal, schematic, high contrast, monospaced labels, no 3D rendering.
```

---

## C. 兩軸診斷散布圖（hover ability × approach ability）

```
Generate a 2D quadrant diagnostic plot, dark navy background, with axes:
  X-axis: "Approach ability →"   (0=can't move, 5=fast tracking)
  Y-axis: "Hover stability →"    (0=falls immediately, 5=rock solid)

Plot the following data points as labeled circles:

  PID Cascade (hover)        @ (0, 5)   green   — perfect hover, no movement
  PID Cascade (waypoint)     @ (1, 5)   green   — stable but too slow
  PPO Expert                 @ (2, 5)   blue    — solid hover, modest tracking
  Flow Matching BC           @ (1, 1)   gray    — fails everywhere (covariate shift)
  ReinFlow Run 7             @ (0, 4)   purple  — hover-only RL
  ReinFlow Run 10            @ (3, 2)   orange  — best eval, can't stabilise
  ReinFlow Run 11            @ (4, 0)   red     — instant crash artefact
  ReinFlow Run 12            @ (3, 3)   yellow  — anchor hover + crash mismatch
  ReinFlow Run 19            @ (1, 4)   cyan    — train ★ but hover-only
  Goal region                shaded green box at upper-right (4-5, 4-5)
                             label: "research target: stable + fast"

Add a dotted curve from (0,5) → (3,2) → (4,0) labeled
"observed Pareto frontier: trade-off, not co-improvement"

Bottom caption: "After 20 runs: each policy can do hover OR approach, not both.
The diagonal trade-off is the gap to close."

Style: clean academic scatter, no clutter, label each point with its name and
RMSE in small text, color-coded by method family.
```

---

## D. 工程 Bug 解決時間軸（讓聽眾看見進步）

```
Generate a horizontal timeline diagram showing 11 engineering bugs identified
and fixed during 20 ReinFlow runs (04-20 to 05-03). Dark navy background.

X-axis: Run 1 → Run 20
Each bug as a horizontal bar:
  - Bar starts at the run where the bug first appeared (red zone)
  - Bar ends at the run where the fix landed (green zone)
  - Bug name as label on the bar

Bugs to plot (from earliest to latest):
  [Run 1 → Run 1]   "VLoss threshold 2.0 too strict"        fix: raised to 100
  [Run 3 → Run 4]   "PLoss=0 (fixed_x1 + pos_filter bug)"   fix: removed both
  [Run 1 → Run 5]   "value_lr too low (3e-4)"               fix: 1e-3
  [Run 5 → Run 7]   "VLoss gate two-sided oscillation"      fix: one-way latch
  [Run 1 → Run 10]  "Hover-only training distribution"      fix: curriculum
  [Run 8 → Run 9]   "OOD disturbance 2.0N too strong"       fix: 1.0N
  [Run 12 → Run 13] "crash_penalty_rl=1.0 train/eval mismatch"  fix: revert to 10
  [Run 13 → Run 18] "lambda_bc=0.1 locks pretrained basin"  fix: 0.01
  [Run 17 → Run 19] "LR=5e-7 → VLoss spike 30+"             fix: 1e-7
  [Run 19 → Run 20] "n_hover=400 over-trains"               fix: 100
  [Day-of-impl]     "PID SO3 rotation order swapped (R_des.T @ R)"  fix: R.T @ R_des

Above the bars, two summary metric lines:
  Blue line  = best train reward (rises from 0.668 to 0.6948)
  Orange line = eval RMSE (flat at 0.51m except dips at Run 10/12 to 0.30m)

Title: "20 ReinFlow Runs — 11 Bugs Found, 11 Fixed; Core Problem (50/50 crash) Remains Structural"

Style: Gantt-chart aesthetic, monospaced labels, color-coded severity,
red text for "still open" issues.
```

---

## 使用建議

| 用途 | 建議使用 prompt |
|------|----------------|
| 簡報封面 / 研究全景 | A |
| Lab meeting Q&A 解釋核心問題 | B |
| 寫論文 method section 配圖 | C |
| 給 advisor 看「我做了多少苦工」 | D |

把 prompt 餵給 Gemini 2.5 / Nano Banana 後，存成 PNG，命名為
`figures/extra_A_blueprint.png` 等，再自行嵌入 [slides.md](slides.md) 即可。
