# DPPO PID Controller — Development Log Index (Phase 2–3c)

> Continuation of [dev_log.md](dev_log.md) (Phase 1 documented there)
> Phase 2 start: 2026-04-01 | Phase 3c v3.2 ongoing: 2026-04-11
> Predecessor: PPO Expert Run 6 (`checkpoints/ppo_expert/20260401_103107/`)

---

## Log Files

| 檔案 | 涵蓋範圍 | 大小 |
|------|----------|------|
| [dev_log_phase2.md](dev_log_phase2.md) | Phase 2：Expert demo 收集、HDF5 結構、pre-collection bug fixes | ~70 行 |
| [dev_log_phase3a.md](dev_log_phase3a.md) | Phase 3a：監督預訓練、Bug Audit × 2、Domain Randomization、Re-run 1 & 2 | ~870 行 |
| [dev_log_phase3b.md](dev_log_phase3b.md) | Phase 3b：RHC baseline eval、DPPO Runs 1–3、Key Lessons、Results Summary | ~280 行 |
| [dev_log_phase3c_v31.md](dev_log_phase3c_v31.md) | Phase 3c v3.1：IMU Late Fusion + FCN Depth 架構、DPPO v3.1 Runs 1–2 post-mortem | ~400 行 |
| [dev_log_phase3c_v32.md](dev_log_phase3c_v32.md) | Phase 3c v3.2：DPPO Run 4、物理 IMU 實作、v3.2 supervised eval、v3.2 DPPO Run 1 | ~2170 行 |

---

## Results Snapshot

| Run | Model | RMSE | Crashes | 備註 |
|-----|-------|------|---------|------|
| Phase 1 | PPO Expert Run 6 | **0.069m** | **0/50** | 黃金標準 |
| Phase 3a | Supervised DP (original) | 0.286m | 50/50 | Covariate shift — 預期 |
| Phase 3b Run 1 | DPPO (β=1.0) | 0.378m | 50/50 | Collapse @ update ~100 |
| Phase 3b Run 2 | DPPO (β=0.1, u11) | **0.168m** | 50/50 | 歷史最佳 RMSE |
| Phase 3b Run 3 | DPPO (β=0.15) | 0.488m | 50/50 | β 稍高即退化 |
| Phase 3c v3.1 Run 1 | DPPO v3.1 | 0.518m | 50/50 | Value net lag |
| Phase 3c v3.1 Run 2 | DPPO v3.1 | 0.466m | 50/50 | finite-diff IMU 不穩定 |
| Phase 3b Run 4 | DPPO (原始架構改良) | TBD | TBD | β=0.05, warm-up 50 |
| Phase 3c v3.2 Run 1 | DPPO v3.2 | TBD | TBD | **進行中** (`dppo_v32_20260411_114141`) |

---

## Key Checkpoints

| Artifact | Path |
|----------|------|
| PPO Expert Run 6 | `checkpoints/ppo_expert/20260401_103107/` |
| Supervised DP (original) | `checkpoints/diffusion_policy/20260402_032701/best_model.pt` |
| Supervised DP Re-run 2 | `checkpoints/diffusion_policy/20260405_044808/best_model.pt` |
| DPPO Run 2 best (歷史最佳) | `checkpoints/diffusion_policy/dppo_20260404_044552/best_dppo_model.pt` |
| v3.1 supervised | `checkpoints/diffusion_policy/v31_20260406_185128/best_model.pt` |
| v3.2 supervised | `checkpoints/diffusion_policy/v32_20260410_120042/best_model.pt` |
| v3.2 DPPO Run 1 (進行中) | `checkpoints/diffusion_policy/dppo_v32_20260411_114141/` |

---

## Active Run

**`dppo_v32_20260411_114141`** — Phase 3c v3.2 DPPO Run 1
- Started: 2026-04-11 11:41
- Log: `logs/train_dppo_v32_20260411_114133.log`
- 詳細配置與監控方式見 [dev_log_phase3c_v32.md §17](dev_log_phase3c_v32.md)

---

## Known Failure Modes (跨版本通用)

| 問題 | 症狀 | 解法 |
|------|------|------|
| Covariate shift | 監督模型 100% crash | D²PPO closed-loop 訓練 (必須) |
| Policy collapse | per-step reward 正→負 | β 減小 + LR 降低 |
| Value net lag | VLoss > 5 至 update ~150 | value warmup (凍結 policy 50 updates) |
| IMU 未歸一化 | v3.2 supervised RMSE 1.985m | gyro/2.0, sf 中心化後/5.0 |
| 推論速度 14Hz | DDIM 10步 = 74ms, 無法達 50Hz | Phase 3d OneDP 單步蒸餾 |
