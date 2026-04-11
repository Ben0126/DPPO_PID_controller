# Claude Code 功能使用指南
# DPPO PID Controller 專案專用

> 更新：2026-04-04
> 對應設定檔：`../.claude/settings.local.json`、`.claude/commands/`

---

## 一、自動運作（不需要你做任何事）

這些功能在背景靜默執行，無需手動觸發。

---

### SessionStart hook — 開啟 Claude Code 時自動顯示訓練狀態

**觸發時機：** 每次開啟 Claude Code session（包含重啟）

**效果：** Claude 的第一句話就已知道目前 run 的狀態，不用你重新說明：
```
=== DPPO Status at Session Start ===
Active run : dppo_20260404_044552
Health     : Healthy
Update     : 78
Reward/step: +0.5064  (healthy range: +0.3 to +0.6)
Value loss : 5.0671 ← still warming up
Best reward: +0.5827 at update 11
Trend      : declining
```

**背後腳本：** `scripts/session_summary.py`（讀 TensorBoard events）

---

### StatusLine — 底部狀態列即時顯示

**觸發時機：** 持續更新（Claude Code 介面底部）

**顯示格式：**
```
DPPO:4_044552 u=78 r=+0.506→ vl=5.07
```
- `u=` ：最新 update 編號
- `r=` ：最新 reward/step（+0.3 以上健康，負數 = collapse）
- `→↑↓` ：趨勢（穩定 / 上升 / 下降）
- `vl=` ：value loss（< 1.0 為收斂）
- `⚠COLLAPSE` ：出現代表需要立刻介入

**更新方式：** session 結束時由 Stop hook 更新 `dppo_status.json` cache。
若想立刻重新整理，執行 `/check-run`。

**背後腳本：** `scripts/status_line.py`

---

### PreToolUse guard — 保護關鍵檔案

**觸發時機：** Claude 每次準備執行 Bash 指令前自動檢查

**保護對象：**
| 檔案 | 重建代價 |
|------|---------|
| `data/expert_demos.h5` | 33 分鐘重新收集 |
| `checkpoints/diffusion_policy/20260402_032701/best_model.pt` | 14 小時重新訓練 |
| `checkpoints/ppo_expert/20260401_103107/` | PPO Run 6 金標準 |
| 任何 `best_dppo_model.pt` | 當前最佳 DPPO checkpoint |

**攔截行為：** 若 Claude 嘗試 `rm`、`del` 等刪除操作，直接 BLOCK 並說明原因。
**手動刪除：** 若真的需要刪除，請在 Claude Code 外的終端機手動執行。

**背後腳本：** `scripts/guard_files.py`

---

### PostToolUse auto_log — 自動寫 dev log

觸發來源分三種，全部寫入 `docs/dev_log_phase2_3.md`：

**① Bash — 訓練/評估指令執行紀錄**

| 指令 | Log 標題 |
|------|---------|
| `python check_device.py` | Device Check |
| `train_dppo` | DPPO Training — Started |
| `train_diffusion` | Diffusion Training — Started |
| `evaluate_rhc` | RHC Evaluation |
| `collect_data` | Expert Data Collection |
| `evaluate_ppo_expert` | PPO Expert Evaluation |

記錄：執行指令 + 輸出（前 600 字元）

---

**② Edit — config 超參數變動 / bug fix diff**

| 被修改的檔案 | Log 標題 |
|-------------|---------|
| `configs/*.yaml` | Config / HP Change |
| `scripts/*.py` | Script Fix |
| `models/*.py` | Model Fix |
| `envs/*.py` | Env Fix |
| `utils/*.py` | Utils Fix |

記錄：檔案路徑 + **Before / After diff**（最多 800 字元各側）

**跳過不記錄：** `docs/`、`.claude/`、`dppo_status.json`（雜訊）

---

**③ Write — 新建 script/config 檔案**

同上分類，記錄：檔案路徑 + 內容片段（前 800 字元）

---

**實際 log 範例（Config HP Change）：**
```markdown
### [Auto-Log] 2026-04-05 09:12:33 — Config / HP Change

**File:** `configs/diffusion_policy.yaml`

**Before:**
  advantage_beta: 0.1   # Run2

**After:**
  advantage_beta: 0.15  # Run3: reward stabilized in Run2
```

**實際 log 範例（Script Fix）：**
```markdown
### [Auto-Log] 2026-04-05 09:14:01 — Script Fix

**File:** `scripts/train_dppo.py`

**Before:**
  value_loss = mse_loss(value_pred, returns_t)

**After:**
  value_loss = mse_loss(value_pred.squeeze(-1), returns_t)  # fix: shape mismatch
```

**背後腳本：** `scripts/auto_log.py`

---

### Stop hook — Session 結束自動備份 metrics

**觸發時機：** 關閉 Claude Code、`/clear`、`/exit` 時

**效果：** 更新 `dppo_status.json`（被 StatusLine 和 SessionStart 讀取的 cache）。
確保下次開啟 session 時顯示最新數據。

**背後腳本：** `scripts/snapshot_metrics.py`

---

### PreCompact hook — 長對話壓縮前保留關鍵 context

**觸發時機：** Claude Code context 快滿時自動壓縮前

**效果：** 提示 Claude 在壓縮摘要中保留：
- 當前 run 的 timestamp 與 checkpoint 路徑
- 最新 update 編號與 reward/step
- Value loss 收斂狀態
- 本次 session 觀察到的 failure mode 或異常
- 已做或計劃中的 HP 修改

**為何重要：** 若不設定此 hook，Claude 壓縮後可能遺忘「目前跑哪個 run」等關鍵資訊。

---

## 二、Slash Commands（輸入後觸發）

在 Claude Code 對話框輸入指令名稱即可。

---

### `/check-run` — 查看訓練進度與健康狀態

**使用場景：**
- 早上起來看夜間訓練的結果
- 手動觸發一次 run 健康檢查
- StatusLine 顯示異常時，想看詳細資訊

**執行內容：**
1. 找最新 DPPO checkpoint 目錄
2. 讀 `training_metrics.json` 並報告 update 數、reward 趨勢、value loss
3. 給出一段健康評估（是否有 collapse 跡象）

**輸入方式：** 直接輸入 `/check-run`

---

### `/eval-best` — 對最佳 checkpoint 執行 50-episode 評估

**使用場景：**
- Run 結束後想知道實際表現（RMSE、crash rate）
- 想和 supervised baseline（0.286m, 50/50 crashes）比較
- 準備寫 dev log 的結果段落時

**執行內容：**
1. 自動找 `best_dppo_model.pt`（或 fallback 到 supervised best）
2. 執行 `scripts/evaluate_rhc.py`（50 episodes）
3. 輸出對比表：mean reward、position RMSE、crash count、inference time

**輸入方式：** 直接輸入 `/eval-best`

---

### `/plan-run` — 高風險修改前的強制規劃流程

**使用場景：**
- 要修改 `configs/diffusion_policy.yaml`（尤其是 HP）
- Run N 結束，準備設計 Run N+1
- 想系統性分析「為什麼這個 run 表現不佳」

**執行內容：**
1. 讀現有 config 與當前 run metrics
2. 要你說明修改動機（解決什麼問題、依據什麼證據）
3. 輸出 HP diff 表（舊值 → 新值 + 理由）
4. 風險評估：若太激進會出現什麼 failure mode
5. 定義成功指標（幾個 update 後、達到什麼 reward 才算改對了）
6. 定義 rollback 方案
7. **等你說「確認」才修改 config 並啟動訓練**

**輸入範例：**
```
/plan-run
/plan-run Run 2 reward declining，考慮調整 advantage_beta 到 0.15
/plan-run value loss 5.07 仍偏高，想提高 value_lr
```

---

### `/new-dppo-run` — 設定並啟動新 DPPO run

**使用場景：**
- `/plan-run` 完成並確認後的執行步驟
- 快速啟動新 run（HP 直接在指令後面說明）

**執行內容：**
1. 顯示目前 config 作為 baseline
2. 修改你指定的 HP（加上版本注解）
3. 在 dev log 新增 Run N 條目
4. 執行 `train_dppo.py`

**輸入範例：**
```
/new-dppo-run
/new-dppo-run advantage_beta=0.15, value_lr=5e-4
```

---

### `/monitor` — 訓練期間定時監控

**使用場景：**
- 剛啟動 `train_dppo.py` 之後
- 夜間或長時間無人看守的訓練
- Update 50–100 的關鍵收斂期

**實際輸入方式（使用內建 /loop skill）：**
```
/loop 15m /check-run        ← 每 15 分鐘檢查一次（推薦）
/loop 10m /check-run        ← 收斂關鍵期用（update 50-100）
/loop 30m /check-run        ← 夜間長時間監控
```

**停止監控：** 按 `Escape` 或輸入 `/stop`

**警戒訊號：**
| 訊號 | 意義 | 建議行動 |
|------|------|---------|
| reward < 0 連續 3 次 | Policy collapse | 立即停止，回到 best checkpoint |
| value loss > 10 且 update > 80 | Value net 未收斂 | 考慮調高 value_lr，規劃 Run N+1 |
| reward 峰值後持續下降 | 早期 collapse 跡象 | 先確認 best ckpt 已存，觀察 10 個 update |
| reward 穩定 +0.3–0.6 | 健康 | 繼續等待 |

---

## 三、內建功能（直接使用，無需設定）

---

### Plan Mode — 只規劃不執行

**使用場景：**
- 任何「我想改 code 但不確定對不對」的時候
- 看到 value loss 異常，想分析根因但不想還沒想清楚就動手

**使用方式：** 按 `Shift + Tab` 切換到 Plan Mode（輸入框旁邊有標示）。
Claude 只會提出計劃，不會執行任何 Bash 或修改檔案，直到你切回執行模式。

**與 `/plan-run` 的差異：**
- `/plan-run`：專用於設計新 DPPO run 的結構化流程
- Plan Mode：通用，任何任務都可以先「只規劃」

---

### WebSearch / WebFetch — 即時查論文

**使用場景：**
- 查 D²PPO / Diffusion Policy / DDPO 最新 paper
- 確認 CoRL 2025 投稿 deadline 與格式
- 查 OneDP（Phase 3d）的實作細節
- 搜尋 advantage weighting 在 RL fine-tuning 的相關 ablation

**使用方式：** 直接在對話中說：
```
搜尋 arXiv 上 diffusion policy RL fine-tuning 的最新 paper
fetch 這個 URL 並總結 method section
查 DDPO 和 DPPO 在 advantage_beta 設定上的差異
```

**不需要任何額外設定**，已在 permissions 開放。

---

### Memory — 跨對話記憶

**位置：** `~/.claude/projects/c--Users-User-Desktop-DPPO-PID-controller/memory/`

**自動運作：** Claude 會自動記住重要的 project 狀態、你的偏好、feedback。

**手動要求記憶：**
```
記住：Run 2 在 update 80 之後趨勢轉 declining，可能是 value net lag
記住：我不希望每次都看到 torch 的 warning 輸出
```

**手動要求遺忘：**
```
忘掉 Run 1 的 collapse 細節，那些已記在 dev log 了
```

---

## 四、快速參考：什麼情況用什麼

| 情況 | 使用功能 |
|------|---------|
| 早上開電腦，想知道夜間訓練結果 | 自動：SessionStart hook 已顯示；手動：`/check-run` |
| 啟動訓練後想自動監控 | `/loop 15m /check-run` |
| StatusLine 顯示 `⚠COLLAPSE` | 立刻 `/check-run` 看詳細，考慮停止訓練 |
| Run 結束，想量化評估結果 | `/eval-best` |
| 準備修改 HP 啟動新 Run | `/plan-run` → 確認 → `/new-dppo-run` |
| 想改 code 但思路還沒清楚 | `Shift+Tab` 進 Plan Mode |
| 查某個 RL 技術的論文 | 直接問 Claude，它會用 WebSearch |
| Claude 似乎忘了 run 狀態 | `/check-run` 或告訴它「讀 dppo_status.json」 |
| 想確認某個保護檔案還在 | 直接問，guard 會在刪除時保護，不影響讀取 |
