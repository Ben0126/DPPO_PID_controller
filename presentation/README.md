# Presentation — DPPO PID Controller 進度報告

完整可演講材料。**目標：10–15 分鐘把研究想法、進度、卡點、解法講完。**

---

## 內容物

| 檔案 | 用途 |
|------|------|
| [slides.md](slides.md) | 16 張投影片（Marp 格式）—— 可輸出 PDF / PPTX / HTML |
| [generate_figures.py](generate_figures.py) | 自動從 dev log 數據產生 6 張關鍵圖 |
| [diagram_prompts.md](diagram_prompts.md) | 4 套給 Gemini / Nano Banana 的高階圖示 prompt |
| `figures/` | `generate_figures.py` 的輸出（6 張 PNG，已產生）|

---

## 快速開始（3 步驟）

```powershell
# 1. 確認在專案根目錄
cd C:\Users\User\Desktop\DPPO_PID_controller\DPPO_PID_controller

# 2. 產生圖（已內建在 slides.md 裡引用）
python -m presentation.generate_figures

# 3. 渲染投影片（任選一種）
```

### 渲染選項

#### A. Marp CLI（最快，輸出 PDF/PPTX/HTML）

```powershell
# 安裝（一次就好）
npm install -g @marp-team/marp-cli

# 輸出 PDF
marp presentation/slides.md --pdf --allow-local-files

# 輸出 PowerPoint
marp presentation/slides.md --pptx --allow-local-files

# 即時預覽（瀏覽器開啟，存檔自動 reload）
marp presentation/slides.md --watch --html
```

#### B. VS Code 擴充套件（最直覺）

裝 `Marp for VS Code`，開 [slides.md](slides.md)，右上角 Preview 按鈕即可。

#### C. 不想裝 Marp？用 Pandoc

```powershell
pandoc presentation/slides.md -o presentation.pptx
```

（樣式會比較陽春，但能直接編輯 PPTX。）

---

## 投影片骨架

| # | 標題 | 類型 |
|---|------|------|
| 1  | 封面 + 一句話總結 | 開場 |
| 2  | 一頁理解這個研究 | 速懂 |
| 3  | Pipeline 全景 | 圖（圖 01）|
| 4  | 系統架構（FPV → Flow Matching → RHC）| 文字+ASCII |
| 5  | 進度時間軸 | 圖（圖 02）|
| 6  | 結果一覽（PID vs PPO vs ReinFlow vs BC）| 表 |
| 7  | 三方法對比柱狀圖 | 圖（圖 03）|
| 8  | ReinFlow 20 runs 故事 | 圖（圖 04）|
| 9  | 解決過的 11 個工程 bug | 表 |
| 10 | **核心發現 1：訓練-評估 gap** | 圖（圖 05）|
| 11 | **核心發現 2：Temperature ablation 否定假設** | 圖（圖 06）|
| 12 | 兩軸診斷（hover × approach）| 文字 |
| 13 | PID baseline 意外結論 | 文字 |
| 14 | 下一步路線（按 ROI）| 表 |
| 15 | 三句話帶走 | 收尾 |
| 16 | Q&A + FAQ 預先準備 | 對答 |

---

## 演講節奏建議（10 分鐘版）

```
 0:00   Slide 1-2     開場，講核心問題（90 秒）
 1:30   Slide 3-5     Pipeline + 進度（90 秒）
 3:00   Slide 6-7     現有結果對比（90 秒）
 4:30   Slide 8-9     ReinFlow 20 runs + 已解決 bug（90 秒）
 6:00   Slide 10-11   兩個核心發現 ★ 重點區（120 秒）
 8:00   Slide 12-13   兩軸診斷 + PID 意外（60 秒）
 9:00   Slide 14-15   下一步 + 收尾（60 秒）
10:00   Slide 16      Q&A
```

---

## 不同聽眾的剪裁建議

### 對 advisor / 同 lab
- 保留全部 16 張
- Slide 10/11 可多停留：強調這是研究故事的「轉折點」

### 對 conference 評審 / 投稿前 dry run
- 砍掉 Slide 9（bug 列表太細）
- 強化 Slide 6/7/12：把對比與 trade-off 講清楚
- Slide 13 改寫成「為什麼 PID 不能是答案」

### 對非本領域聽眾（家人、業界）
- 砍 Slide 4（架構）和 Slide 11（temperature ablation）
- 把 Slide 12 改畫成更直觀的「飛行軌跡示意圖」
- Slide 14 改成「未來 3 個月路線」

---

## 維護

- `generate_figures.py` 內所有數據都從 dev log 直接抄來（hard-coded），新 run 後直接更新檔內變數即可。
- 想換主題色：改 [generate_figures.py](generate_figures.py) 上方的 `PALETTE` dict + [slides.md](slides.md) 的 `style:` 區塊。
- 新增投影片：直接在 [slides.md](slides.md) 用 `---` 分頁加。

---

## 需要更新的時機

| 觸發條件 | 要改什麼 |
|---------|----------|
| 跑完新 run（Run 21+）| `generate_figures.py` 的 `runs/train_peak/eval_rmse` 陣列 |
| 出新的 baseline 數據 | `slides.md` 的「結果一覽」表 |
| 換研究方向（例如改用 D²PPO） | Slide 14 路線表，並加新的 Phase 3c 章節 |
| 投稿目標期刊變更 | Slide 1 副標題的 venues |
