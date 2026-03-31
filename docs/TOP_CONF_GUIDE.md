# Vision-DPPO 頂會論文研究指南

> 目標會議：CoRL 2025 / ICRA 2026 / RSS 2026  
> 文件用途：研究流程說明、審稿人攻擊點防禦、架構升級路線

---

## 一、學術研究的正確流程

很多人以為研究流程是「發現問題 → 提出方法 → 驗證 → 重複」，但頂會研究在每個步驟的實際內容要細很多。

### 1.1 問題定義（最難，也最重要）

這不是「發現有個問題」，而是精確回答三件事：

1. **現有方法在哪個具體情境下失敗？**（不是模糊的「效果不好」）
2. **失敗的根本原因是什麼？**（不是表象症狀）
3. **這個失敗重要到值得一篇論文解決嗎？**

**以本計畫為例：**
- ❌ 錯誤框架：「擴散策略太慢」
- ✅ 正確框架：「擴散策略的迭代降噪架構與即時控制的延遲需求存在根本性矛盾，導致在欠驅動非線性系統（四旋翼）上無法達到足夠的控制頻率」

這兩個框架指向完全不同的解決方案和論文貢獻。

### 1.2 核心假設（Hypothesis）

你需要一個**可以被推翻**的假設，而不只是「我提出一個新方法」。

好的假設範例：
> 「我們假設，將色散損失（Dispersive Loss）引入擴散策略的表徵學習，能夠緩解高速視覺控制中的特徵崩塌，從而提升策略在分佈外視覺輸入下的穩定性。」

這個假設可以被消融實驗獨立驗證或推翻。有可能被推翻的假設才是科學假設。

### 1.3 方法設計

不只是「設計演算法」，還需要從理論上說明**為什麼這個方法應該有效**。

頂會審稿人區分的兩種論文：
- **工程報告**：「我們試了這個，它有效」
- **研究論文**：「我們有理由相信這應該有效（理論），實驗印證了這個理由（實證）」

後者才能發頂會。

### 1.4 受控實驗（三個層次缺一不可）

| 層次 | 目的 | 對應到本計畫 |
|------|------|------|
| **消融實驗（Ablation）** | 獨立證明每個設計決策的貢獻 | 有/無色散損失；CNN vs ViT；多步 vs 單步 |
| **基準對比（Baselines）** | 在公平條件下證明整體優越性 | BC-LSTM、VTD3、Standard DP |
| **分析（Analysis）** | 解釋為什麼有效，什麼時候會失效 | 特徵空間可視化、失敗案例 |

### 1.5 失敗案例分析（最常被忽略）

**主動展示你的方法在哪些情況下不好用**，並解釋原因。這不是弱點，這是你對自己方法理解深度的指標。頂會審稿人最怕的是「作者不知道自己的弱點在哪」。

---

## 二、為什麼用擴散策略，而不是直接用 PPO 或 VIO+PPO

### 2.1 PPO 的根本限制

PPO 假設「在給定狀態下，最好的動作服從高斯分佈（Unimodal）」。

問題：**很多飛行情境的最優動作是多峰的（Multimodal）。**

想像無人機面前有柱子，往左或往右都合理。PPO 的高斯分佈會把左繞和右繞平均，輸出「直接撞上柱子」的動作。這是數學上必然發生的事，不是調參能解決的。

擴散策略學習的是動作的**完整分佈**，可以同時維持多個峰，這次往左、下次往右，兩個都是有效策略。

### 2.2 VIO + PPO 的結構性問題

VIO+PPO 確實可以用，很多實際系統就是這樣。但有三個問題：

**問題一：誤差累積**  
VIO 的狀態估計誤差會傳遞給控制器。你很難判斷「是控制策略不好，還是狀態估計不準」。

**問題二：優化目標不對齊**  
VIO 的目標是「狀態估計準確」，PPO 的目標是「控制性能好」。這兩個目標不一定對齊。端到端系統讓特徵提取直接服務於最終控制性能。

**問題三：機載資源競爭**  
在 Jetson 上同時跑 VIO 和神經網路控制器，兩者競爭 CPU/GPU 資源。

### 2.3 擴散策略特有的優勢

擴散策略學習動作「分佈」而非確定性映射，當視覺輸入有雜訊時，模型以分佈的不確定性吸收干擾，而不是硬失敗。這在 Sim-to-Real 遷移中是重要優勢。

---

## 三、審稿人會如何攻擊這篇論文

### 攻擊 #1：推理延遲（致命）

> **預測審稿人原文：** "The control frequency of 12.5Hz is insufficient for a quadrotor with highly nonlinear, underactuated dynamics. Standard PID operates at 400Hz+. This system would fail catastrophically in real-world wind gusts."

**根本問題：** 10步 DDIM 在 Jetson 上約需 50-80ms，對應 12-20Hz。四旋翼在受側向陣風時，姿態可在 50ms 內達到危險角度。

**防禦方案：** OneDP 單步蒸餾 → 62Hz+，延遲 <16ms

### 攻擊 #2：表徵崩塌（技術性）

> **預測審稿人原文：** "The 4-layer CNN will produce degenerate embeddings for rapid yaw changes. CNN's inductive bias over-smooths high-frequency visual signals. There is no analysis of representation quality."

**根本問題：** CNN 對微小姿態差異產生幾乎相同的特徵向量，擴散模型無法區分不同飛行狀態。

**注意：** 「過度平滑」在技術上不夠精確，更準確的說法是「缺乏全局上下文建模能力，跨域遷移泛化性弱」。如果用「過度平滑」這個說法，要準備好反駁論據。

**防禦方案：** D²PPO 色散損失 → 強制特徵向量在空間中保持分散性

### 攻擊 #3：Sim-to-Real 差距（系統性）

> **預測審稿人原文：** "The 64×64 synthetic renderer has no analysis of domain gap. How does this transfer to a real camera with motion blur, lens distortion, and varying lighting?"

**根本問題：** 合成渲染的光線模型、材質反射、感測器雜訊、運動模糊都與真實相機不同。特徵提取器學到的是模擬圖片的統計規律，換到真實相機分佈不同就失效。

**防禦方案：** Flightmare/Agilicious 高保真模擬器 + 強力域隨機化

### 攻擊 #4：基準測試不公平（方法論）

> **預測審稿人原文：** "Comparing against a state-based PPO expert is unfair—it has privileged state information. Where is the comparison against BC-LSTM, VTD3, or the original Diffusion Policy?"

**根本問題：** PPO Expert 有完整真實狀態，視覺系統沒有。這不是 fair comparison。

**防禦方案：** 加入相同輸入條件下的視覺基準對比

---

## 四、四大架構升級方案

### 升級一：CNN → Vision Transformer + 特權資訊解碼

**為什麼 ViT 更好（前提條件重要）：**  
ViT 的優勢只在使用大規模預訓練權重（如 ImageNet）進行微調的情境下成立。從頭訓練（from scratch）的 ViT 反而需要更多資料。

**特權資訊解碼（核心貢獻）：**  
在預訓練階段，在 ViT 隱藏層加入輔助解碼頭，強制學習對應物理量的表徵（3D 位置誤差、線速度、角速度）。部署時移除解碼頭，隱藏層已隱式學到等同狀態估計器的能力。

```
FPV Image → Pretrained ViT → [Hidden Features]
                                    ↓ (train only)
                              Auxiliary Decoder → [Position, Velocity, Attitude]
                                    ↓ (deploy)
                              Conditional 1D U-Net → Action Sequence
```

**參考：** SkyDreamer (arXiv:2510.14783), NavDP

### 升級二：DPPO → D²PPO（色散損失 + 雙層 MDP）

**色散損失的數學直覺：**
```
L_total = L_diffusion + λ × L_dispersive

L_dispersive = -log( ||h_i - h_j|| / (||h_i - h_j|| + margin) )
```

將同一 mini-batch 內所有樣本的隱藏特徵向量均視為負對，強制彼此遠離，避免語義上不同的狀態被編碼成相同特徵。

**雙層 MDP 框架（論文核心理論）：**
- 內層 MDP：擴散模型的迭代降噪過程
- 外層 MDP：無人機與物理環境的真實交互
- 因為每步都涉及高斯似然性，策略梯度在數學上是 well-defined 的

**D²PPO 的數字（注意：來自操縱任務，非無人機）：**  
94% 平均成功率，比基準提升 26.1%。這些數字只能作為方法可行性的先例，**不能**直接宣稱你的系統會達到相同效果。

**參考：** D²PPO (arXiv:2508.02644)

### 升級三：多步 DDIM → OneDP 單步蒸餾

| | 現有系統 | 升級目標 |
|---|---|---|
| 步數 | 10步 DDIM | 1步 |
| 控制頻率 | 12.5 Hz | 62 Hz+ |
| 延遲 | ~80ms | <16ms |
| Jetson 實際 | 可能 <5Hz | 30~50Hz |

**蒸餾流程：**
1. 完成 D²PPO 微調的多步策略作為教師模型
2. 最小化 KL 散度：`L_distill = KL( q_teacher(a|s) || q_student(a|s) )`
3. 自一致性訓練 + 高密度專家模式的自我引導正則化

**重要順序：** 必須先有好的教師模型（D²PPO 微調後）才能蒸餾。用差的教師蒸餾，學生也會很差。

**額外成本：** 約 2~10% 預訓練算力。

**參考：** OneDP (arXiv:2410.21257), OFP (arXiv:2603.12480)

### 升級四：合成渲染 → 高保真模擬器 + 域隨機化

**推薦模擬器：Flightmare**
- Unity/Unreal Engine 高保真渲染
- 物理引擎與渲染器解耦，支持數千 FPS 採樣
- 精確 6-DOF 四旋翼動力學
- 支持 ROS 2 接口

**域隨機化清單（最低要求）：**
- 隨機光源位置與強度（3種以上光照模式）
- 相機運動模糊（Motion Blur）
- 鏡頭耀光與徑向畸變
- 部分畫面遮蔽（Visual Occlusions，隨機矩形/圓形遮蔽）
- 隨機背景紋理替換

---

## 五、完整基準測試矩陣

### 公平比較的原則

**「PPO Expert vs Vision-DPPO」不是公平比較。** PPO Expert 有完整狀態資訊，視覺系統沒有。必須在相同輸入條件下對比。

### 基準矩陣

| 方法 | 輸入 | 對比目的 | 預期結果 |
|------|------|---------|---------|
| BC-LSTM | RGB 圖像 | 證明 BC 的多模態崩塌問題 | BC 在岔路口動作抖動；DPPO 平滑 |
| VTD3（Vision-based TD3） | RGB 圖像 | 對比連續動作空間的樣本效率 | DPPO 在多模態任務更穩定 |
| Standard DP（CNN+DDPM） | RGB 圖像 | 消融：證明各升級的貢獻 | 延遲下降、成功率提升 |
| VIO + 幾何控制 | RGB + IMU | 證明端到端消除誤差累積 | Vision-DPPO 在高速時 RMSE 更低 |
| PPO Expert（僅作上界） | 完整狀態 | 確立性能上界（Oracle） | Vision-DPPO 達 PPO 的 80%+ |

### 必要評估指標

**飛行性能：**
- Position RMSE（目標：<0.15m）
- Crash Rate（目標：<10%）
- Episode Reward 相對 PPO Oracle 比率（目標：>80%）
- 干擾後恢復時間（Settling Time）

**系統性能：**
- Inference Latency ms（目標：<20ms）
- Control Frequency Hz（目標：60+）
- Sim-to-Real 零樣本遷移成功率
- 抗陣風干擾存活率（真實戶外測試）

---

## 六、各頂會對齊策略

| 會議 | 錄取率 | 核心要求 | 本計畫的重點 |
|------|--------|---------|------------|
| CoRL 2025 | ~15% | 生成式策略的泛化能力、實體機器人驗證 | D²PPO 理論創新、多模態分佈的必要性論述 |
| ICRA 2026 | ~40% | 系統整合、硬體實驗、閉環魯棒性 | 真實飛行數據、Jetson 延遲測試、陣風抗擾 |
| RSS 2026 | ~12% | 演算法深度、數學嚴謹性 | DPPO 收斂性證明、消融實驗、流形探索分析 |

**策略建議：** 如果時間緊，先做 D²PPO 的消融實驗拿到理論貢獻，投 CoRL。硬體部署留給 ICRA 的完整版本。

---

## 七、論文寫作實戰 Tips

### 摘要黃金法則（4 句話）

1. **問題（1句）**：現有方法的根本缺陷是什麼？（要有具體情境）
2. **洞見（1句）**：你的核心直覺是什麼？（為什麼這個方向對）
3. **方法（1句）**：具體提出了什麼？（技術名詞要精準）
4. **結果（1句）**：量化結果是什麼？（必須有數字）

❌ 不要：「我們提出了一個新的端到端無人機控制框架...」  
✓ 要：「現有擴散策略因迭代降噪導致 12Hz 的控制頻率，無法應對四旋翼的非線性動態；我們提出 D²PPO，透過色散損失解決表徵崩塌，並以單步蒸餾將控制頻率提升至 62Hz，在 X 任務上達到 Y% 成功率，比基準提升 Z%。」

### 消融實驗設計（最低要求）

每個設計決策都需要一個消融實驗獨立證明其貢獻：

| 消融組合 | 目的 |
|---------|------|
| CNN vs ViT（純替換） | 視覺編碼器的獨立貢獻 |
| 有/無色散損失 | D²PPO 核心貢獻 |
| 色散損失在早期/晚期層 | 超參數敏感性分析 |
| 多步 DDIM vs 單步蒸餾 | 延遲與性能取捨曲線 |
| 有/無特權資訊解碼預訓練 | 輔助任務的必要性 |

### 主動防禦弱點

在 Limitation 節中主動點出你的限制，比等審稿人攻擊更有說服力：

```
"Our method assumes stable lighting conditions. Under extreme 
illumination changes (e.g., direct sunlight into camera), the 
ViT encoder may produce unreliable features. Future work will 
incorporate adaptive exposure compensation."
```

### 影片製作（CoRL/ICRA 重視）

- **開頭 10 秒**：最震撼的飛行片段（先勾住注意力）
- **中段**：系統架構動畫說明
- **對比段**：並排展示 Vision-DPPO vs 最強 baseline
- **結尾**：真實硬體戶外飛行（有陣風更有說服力）

### Reference 選擇策略

**必引（缺一不可）：**
- Chi et al. RSS 2023（Diffusion Policy 原論文）
- DPPO (OpenReview:mEpqHvbD2h)
- D²PPO (arXiv:2508.02644)
- OneDP (arXiv:2410.21257)
- Kaufmann et al. Nature 2023（Swift，無人機領域 SOTA）

**加分：** 引用目標會議 2024 年的最新工作，顯示你跟得上前沿。

**注意：** 未正式發表的 arXiv 論文不要引超過 3 篇，否則會被質疑文獻嚴謹性。

---

## 八、指引文件的重要勘誤

以下內容在原指引文件中有誤導性，使用前請注意：

### 勘誤一：ViT 的適用前提

**原文說法：** ViT 能以減少 40% 訓練數據適應新領域，收斂快 3~5 倍。

**正確理解：** 這個優勢只在使用大規模預訓練權重（ImageNet 等）進行微調時成立。若從頭訓練（from scratch），ViT 反而需要比 CNN 更多的訓練資料。你的計畫必須使用預訓練 ViT，不能自己訓一個。

### 勘誤二：CNN「過度平滑」的說法

**原文說法：** CNN 的時間卷積歸納偏置傾向偏好低頻訊號，導致過度平滑。

**正確理解：** CNN 本身可以捕捉高頻空間特徵。問題更準確的說法是「缺乏全局上下文建模能力」以及「跨域遷移時泛化性弱」。如果在論文中用「過度平滑」，要準備好被審稿人質疑。

### 勘誤三：效能數字的適用範圍

**原文說法：** OneDP 達 62Hz，D²PPO 達 94% 成功率、提升 26.1%。

**正確理解：** 這些數字來自各論文在操縱任務（manipulation tasks）上的實驗，**不是無人機控制任務**。可以引用這些數字作為方法可行性的先例，但不能宣稱你的系統會達到相同效果。你的系統需要自己測量這些數字。

---

*最後更新：2026-03*  
*基於：RESEARCH_PLAN.md、D²PPO arXiv:2508.02644、OneDP arXiv:2410.21257、SkyDreamer arXiv:2510.14783*
