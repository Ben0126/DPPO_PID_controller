# Vision-DPPO 視覺無人機端到端控制研究 — 進度與狀態統整

本文件旨在統整當前項目的研究狀態、最新進度、核心洞察以及未來的研究方向，作為開發與學術論文寫作的指引。

---

## 📌 研究背景與核心目標
* **研究課題**：利用**擴散策略 (Diffusion Policy)** 實現基於第一人稱視角 (FPV) 圖像序列輸入，直接映射至 4D 馬達推力指令 (CTBR) 的端到端無人機控制。
* **主要貢獻**：
  1. 提出 **D²PPO (Dispersive PPO)** 演算法，引入 **色散損失 (Dispersive Loss)** 來防止高頻視覺特徵崩潰 (Feature Collapse)。
  2. 藉由 **OneDP 單步蒸餾技術** 將 10-step DDIM 降至 1-step，滿足端到端實時控制要求（>60Hz，延遲 <16ms）。
* **目標學術會議**：CoRL 2025 / ICRA 2026 / RSS 2026

---

## 🚀 階段性研究狀態 (Research Pipeline Status)

當前專案已演進至 **v4.0 階段**，並在嘗試 **v5.0 架構** 後得出結構性結論。以下為各階段狀態摘要：

| 階段 | 描述 | 當前狀態 | 關鍵數據 / Checkpoint |
|---|---|---|---|
| **Phase 0** | INDI 內環 Hover 測試 | **Done** | 姿態傾角 0.00°, 角速度 0.000 rad/s |
| **Phase 1** | 狀態輸入 PPO 專家策略 (CTBR) | **Done** | RMSE 0.0649m, 0/50 墜毀 (`20260419_142245`) |
| **Phase 2** | FPV 專家演示數據收集 | **Done** | `data/expert_demos_v4.h5` (1000 ep, 3.9GB, 0 墜毀) |
| **Phase 3a** | Flow Matching 監督預訓練 | **Done** | 最佳驗證損失 0.0630 (`flow_policy_v4/20260420_034314`) |
| **Phase 3b** | ReinFlow RL 微調 (AWR) | **已結束** | 27 個 runs 全部面臨 **AWR 模態崩潰** (AWR mode-collapse) |
| **Phase 3c** | DAgger 恢復軌跡訓練 (DAgger Recovery) | **否決** | 恢復數據毒化了 Hover 軌跡，飛行器表現惡化 |
| **Phase 3d** | **H4 架構 + 多層次指標評估 (飛穩準)** | **當前 SOTA** | H4 BC: 存活步數 202 步, 分數 **0.171** (`flow_policy_v4/20260514_175219`) |
| **v5.0 BC** | Joint E2E 聯合預訓練 + Phase B&C（任務調節 + 端到端色散損失） | **Done** | Joint E2E val_flow **0.0642** (`20260603_171316`); Phase B&C val_flow **0.0663** (`20260604_141454`) |
| **v5.0 RL** | ReinFlow + 優勢硬遮罩微調（Phase D，positive_advantage_mask） | **已結束（Phase D 結論）** | best: score **0.130**, survive 27.9%（⚠️ 短存活偽象，tier1 通過率 1/30）；final: survive **4.8%**（課程崩潰，reward -0.22→-3.08） |
| **Phase 4** | 實機部署 (Jetson Orin Nano) | **未來計畫** | 實時推理目標時間 < 30ms (含相機採集與命令輸出) |

---

## 💡 六大核心研究發現 (Core Research Discoveries)

### 1. 傳統 RMSE 指標之系統性偏誤 (RMSE Metric Bias)
在過去 24 個 Runs 的評估中，評估代碼在計算 Position RMSE 時，是以策略的**實際存活時間**為分母。這導致**死得越快（存活步數極短）的策略反而能取得極小的 RMSE（因飛機沒時間漂移）**。此發現徹底推翻了過去將 Run 10（RMSE 0.3005m，僅存活 36 步）視為最佳成果的結論。

### 2. 「飛 ➔ 穩 ➔ 準」多層次評估指標 (Hierarchical Evaluation Framework)
為了解決 RMSE 的偏誤與 $SR=0.5$ 邊界上的 10x 斷崖式不連續（Discontinuity Cliff），項目在 **Phase A** 引入了平滑化與歸一化尺度修正（實作於 [evaluate_hierarchical.py](file:///c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/scripts/evaluate_hierarchical.py)）：
* **Tier 1 (飛 - Survival)**：存活率 $\text{SR} = T / T_{max}$。
* **Tier 2 (穩 - Stability)**：採用引入物理容差尺度因子 $\sigma$（預設 $\sigma=2.0\text{m}$）的平滑指數衰減：$\text{stability\_score} = e^{-\text{IAE}_{steady} / \sigma}$，其中 $\text{IAE}_{steady} = \text{mean}(|\mathbf{e}_t|)_{t \in [T/2, T]}$。
* **Tier 3 (準 - Accuracy)**：採用引入物理容差尺度因子 $\sigma$ 的平滑指數衰減：$\text{accuracy\_score} = e^{-\text{terminal\_err} / \sigma}$，其中 $\text{terminal\_err}$ 為最後 10% 步數的位置誤差。
* **複合分數 (Composite Score)**：
  $$\text{Score} = \text{SR} \times (0.6 \times \text{stability\_score} + 0.4 \times \text{accuracy\_score})$$
  移除了 Tier 1 gate 的硬性劃分，使評分函數在整個生存範圍內均平滑、連續且單調。

### 3. IMU 主導優先於視覺 (IMU-Dominant Fusion)
在無人機高頻穩定控制中，純視覺編碼器難以區分微小姿態變化，導致特徵塌陷。**H4 架構**（定義於 [flow_policy_v4.py](file:///c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/models/flow_policy_v4.py#L48)）將 IMU 特徵從 128D 擴大至 512D，使視覺與 IMU 的梯度比率從 46.8× 降至 **3.22×**，讓高頻 IMU 訊號主導控制。
> **結果**：在純監督學習下，H4 BC 將平均存活步數從 130 步提升 55% 至 **202 步**，複合評估分數達 **0.171**，超越過去所有 RL 微調模型。

### 4. Advantage-Weighted Regression (AWR) 模態崩潰
所有 Phase 3b 的 RL 微調嘗試均以失敗告終。深層原因在於 AWR 的優化機制：
$$\mathcal{L} = \mathbb{E}\left[ e^{\beta \cdot A} ||v_\theta - v^*||^2 \right]$$
在 50/50 墜毀的高噪聲環境下，GAE 優化器計算出的 Advantage 正規化後，負 Advantage 樣本權重僅降至 ~0.74×。這意味著策略**仍在模仿自己產生的墜毀動作**，導致訓練在 200 次更新後發生嚴重的單調退化與 mode-collapse。

### 5. 特徵空間與動作生成的不對齊 (Encoder-Action Alignment Issue)
為了解決視覺特徵漂移，**v5.0 架構**設計了 IMU-guided 交叉注意力機制 ([CrossAttentionIMU2Vision](file:///c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/models/flow_policy_v5.py#L42))，並加上輔助狀態預測頭 ([StatePredictor](file:///c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/models/flow_policy_v5.py#L64)) 強迫視覺編碼器學習物理規律。
然而，在 Stages A→D 迭代測試中證實：**用於預測物理狀態（State Regression）的最優視覺特徵，不等於生成控制動作（Action Generation）的最優特徵**。分開預訓練會導致梯度衝突，且 DAgger 在 100% 墜毀的 Rollout 分布下無法有效蒸餾。本項目後續通過**端到端聯合訓練 (Joint E2E Training)** 解決了特徵對齊問題，使生存率提升至 60.1%。

### 6. 優勢硬遮罩的效果與局限——課程誘發退化與容量衝突（2026-06-04）

在 v5.0 Phase D 中，透過 `positive_advantage_mask: true` 將負優勢樣本從梯度更新中完全排除。這讓 700 次 RL update 的 VLoss 從初始 **348** 正常收斂至 **56**，克服了 v4.0 Phase 3b 中 VLoss spike（30+）導致訓練即時崩潰的問題——**正優勢遮罩對即時崩潰的抑制作用確認有效**。

然而，隨著課程位置邊界從 0.1m 逐步推展至 **0.57m**，訓練 reward 從峰值 **-0.2161** 惡化至 **-3.08**，最終策略（`v5_RL_final`）存活率僅 **4.8%**。此崩潰與 v4.0 AWR 崩潰形式相同，只是發作時機延後至課程擴展階段。

`v5_RL_best`（複合分數 0.130，σ=2.0）的低穩態誤差（IAE **1.224m**）幾乎確定是**短存活偽象**：30 集中僅 **1 集通過 tier1**（平均 139 步），這是本項目 2026-05-15 揭露的 RMSE 偏誤的 IAE 版重演——「飛行時間短 → 沒時間漂移 → IAE 人工偏低」。

此結果揭示了一個更深層的結構性限制：**「高精度懸停穩定（需精細 sub-meter 動作輸出）」與「廣域危機恢復（需大幅度推力）」之間存在根本性的參數容量衝突（Robustness-Precision Capacity Conflict）**，在給定網路規模下，課程學習無法同時優化兩者。

---

## 📊 各控制器性能對照表 (Controller Benchmark)

在**飛 ➔ 穩 ➔ 準**多層次評估指標下（評估 30 episodes，最大 500 步），各控制器的真實排名如下。標注「σ=2.0」者為 Phase A 更新後的平滑指數衰減指標（`exp(-e/2.0)`）；標注「舊指標」者尚未以 σ=2.0 重評。

| 排名 | 模型 / 核心設定 | 複合分數 (Score) | 存活率 (Survive) | 穩定誤差 (IAE_steady) | 終端誤差 (Term) | 備註 |
|:---:|---|:---:|:---:|:---:|:---:|---|
| **-** | **Cascade PID Baseline** | **-** | **100%** | **0.022m** (Hover)<br>**1.177m** (Waypoint) | **0.022m**<br>**1.177m** | 穩定但反應極慢，無法即時追蹤高頻動態 Waypoint |
| **1** | **v4.0 H4 BC** | **0.166** (σ=2.0) | **42.2%** | **1.599m** | **2.720m** | **複合分數 SOTA**。無 RL，2-step Euler，高頻精準度最高 |
| **2** | **v5_RL_best** (Phase D 最優保存點) | **0.130** (σ=2.0) | 27.9% | **1.224m** | **1.902m** | ⚠️ **短存活偽象**：僅 1/30 集通過 tier1，平均 139 步；IAE 低係因飛行時間短而非真正精度提升。勿視作「精度突破」 |
| **3** | **v5_BC** (Phase B&C，任務調節+色散損失) | **0.126** (σ=2.0) | **54.9%** | 2.505m | 4.454m | 精簡資料（200 hover+100 recovery），val_flow 0.0663；存活率高於 H4 BC |
| **4** | **Joint_E2E_v5** (E2E BC, 20260603) | **0.110** (σ=2.0) | **55.3%** | 2.763m | 4.693m | **存活率最高（另一輪評估 60.1%）**。端到端聯合訓練，視覺特徵與動作對齊 |
| **5** | v5.0 OOB Pretrain BC | 0.112 (舊指標) | 49.0% | 1.773m | 3.074m | 存活率高，但特徵與動作未對齊 |
| **6** | v4.0 H3a BC | 0.151 (舊指標) | 43.6% | 1.656m | 2.839m | 舊版 IMU 128D，視覺主導 |
| **7** | v5.0 Stage D Best | 0.073 (舊指標) | 55.2% | 2.259m | 3.852m | 分離預訓練，flow_net 重訓失敗 |
| **8** | Run 23 RL (Best RL, v4.0) | 0.093 (舊指標) | 18.6% | 0.770m | 1.272m | AWR mode-collapse，存活率極差 |
| **9** | Run 19 RL | 0.078 (舊指標) | 15.6% | 0.710m | 1.113m | 舊 RMSE「最佳」，實際死得最快 |
| **10** | **v5_RL_final** (課程崩潰) | 0.032 (σ=2.0) | 4.8% | 0.656m | 1.045m | Phase D 末端，課程推展 0.57m 後災難性退化；同 AWR 崩潰形式 |

---

## 🛠️ 代碼與配置檔案指南 (Codebase Directory & Configurations)

所有源代碼與研究進度開發日誌均存放在內層 Git 倉庫 `DPPO_PID_controller/` 中：

### 關鍵模組路徑與連結
* **物理與動力學環境**：
  * [quadrotor_dynamics.py](file:///c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/envs/quadrotor_dynamics.py)：6-DOF 物理運動方程式與 RK4 積分器。
  * [quadrotor_env.py](file:///c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/envs/quadrotor_env.py)：15D 狀態觀測與高斯 Reward 定義。
* **策略網路模型**：
  * [flow_policy_v4.py](file:///c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/models/flow_policy_v4.py)：H4 架構定義（IMU 512D Dominance + Tilt 輔支頭）。
  * [flow_policy_v5.py](file:///c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/models/flow_policy_v5.py)：v5.0 交叉注意力與狀態預測器定義。
  * [vision_encoder_v5.py](file:///c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/models/vision_encoder_v5.py)：提供 spatial map 的 CNN 視覺特徵提取器。
* **經典控制器**：
  * [pid_controller.py](file:///c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/controllers/pid_controller.py)：串級 PID 基準控制器（NED 坐標系，SO3 誤差）。
* **訓練與評估腳本**：
  * [evaluate_hierarchical.py](file:///c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/scripts/evaluate_hierarchical.py)：「飛穩準」新評估指標腳本。
  * [train_flow_v5.py](file:///c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/scripts/train_flow_v5.py)：v5.0 監督預訓練腳本。
  * [train_distillation_v5.py](file:///c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/scripts/train_distillation_v5.py)：v5.0 DAgger 蒸餾學習腳本。
  * [pretrain_vision_oob.py](file:///c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/scripts/pretrain_vision_oob.py)：OOB 狀態預估視覺預訓練。
  * [train_reinflow_v5.py](file:///c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/scripts/train_reinflow_v5.py)：v5.0 ReinFlow RL 強化微調腳本（含 ValueNetworkV5、`positive_advantage_mask`、課程學習 curriculum）。

### 配置檔案
* [quadrotor_v4.yaml](file:///c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/configs/quadrotor_v4.yaml)：無人機物理與環境常數配置。
* [flow_policy_v5.yaml](file:///c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/configs/flow_policy_v5.yaml)：v5 BC 訓練配置。
* [distillation_v5.yaml](file:///c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/configs/distillation_v5.yaml)：v5 DAgger 蒸餾配置。
* [reinflow_v4.yaml](file:///c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/configs/reinflow_v4.yaml)：v4 ReinFlow 強化學習配置。
* [reinflow_v5.yaml](file:///c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/configs/reinflow_v5.yaml)：v5.0 ReinFlow RL 配置（`positive_advantage_mask: true`、課程學習起始/終止位置邊界、`lambda_bc`、`lambda_disp`）。

### 開發歷史日誌
* [dev_log_v4_h4_hierarchical.md](file:///c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/docs/dev_log_v4_h4_hierarchical.md) — 包含 H4 誕生、新指標引入與 v5.0 完整測試日誌。
* [dev_log_v4_post20.md](file:///c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/docs/dev_log_v4_post20.md) — Runs 21–28 與 DAgger Recovery 的假設驗證鏈。

---

## 🔮 後續研究方向與建議 (Next Steps)

若要突破當前 H4 BC 的複合分數瓶頸 (0.171)，建議朝向以下幾點進行改進：

1. **聯合端到端訓練 (Joint End-to-End Training)**：
   * 捨棄分開預訓練（Separate Pre-training）的範式，讓視覺編碼器與 `flow_net` 針對同一個軌跡生成任務進行联合優化，從而天然對齊特徵空間與控制輸出。
2. **在 BC 階段直接引入動作標籤的蒸餾**：
   * DAgger 在 100% 墜毀環境下之所以失效，是因為其 rollout 收集的均是墜毀狀態的標籤。後續可考慮只利用專家生成的穩定 hover 與 waypoint 追蹤數據進行離線的行為蒸餾 (Offline Distillation)。
3. **解決 AWR 模態崩潰的 RL 微調**：
   * If仍希望利用強化學習微調，必須引入強硬的篩選機制（如 `positive_advantage_mask`），完全不讓策略去模仿負向優勢的動作，或者引入類似 PPO 的 Clipped 信任區域限制，避免梯度爆炸。

---
*本文件由 Antigravity 彙整，最後更新於 2026-06-12。所有實驗日誌詳情可參閱 [docs/dev_log_v4_h4_hierarchical.md](file:///c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/docs/dev_log_v4_h4_hierarchical.md) 與 [docs/experiment_report_joint_e2e.md](file:///c:/Users/User/Desktop/DPPO_PID_controller/DPPO_PID_controller/docs/experiment_report_joint_e2e.md)。*
