# Vision-DPPO v4.0 — 開發日誌

> 失敗記錄 NEVER 刪除。只標記 RESOLVED ✅ / OPEN 🔍 / DEPRECATED 🗄️

## 記錄模板
```
### [YYYY-MM-DD] Run: {wandb_run_id}
**Phase**: Phase X
**現象**: [具體觀察，必須有數字]
**假設**: [可能原因]
**根本原因**: [確認後]
**修復方案**: [行動]
**結果**: [修復後數字]
**狀態**: RESOLVED ✅
```

---

### [2024-XX-XX] 歷史：v3.3 100% Crash Rate
**Phase**: Phase 3c_v33
**現象**: crash rate = 100%，所有 episode 在 < 5 步內墜毀
**根本原因**: SRT 動作空間在 12-25Hz 輸出 4D motor thrust，無內迴路穩定，開迴路不穩定
**修復方案**: RESEARCH_PLAN_v4.0，切換至 CTBR + PID inner-loop
**狀態**: DEPRECATED 🗄️（v3 保留為歷史）

### [2024-XX-XX] 歷史：ε-space 放大問題
**Phase**: Phase 3d
**現象**: 單步蒸餾品質嚴重下降，RMSE 從 0.10m 退化至 0.26m
**根本原因**: DDPM ε-space 在 t=T-1 放大係數 1/√ᾱ ≈ 64×，單步誤差被放大
**修復方案**: 改用 Flow Matching（無放大問題，原生支援 1-step）
**狀態**: DEPRECATED 🗄️（v4 改用 Flow Matching）
