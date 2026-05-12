# Phase Gate Criteria — Vision-DPPO v4.0

未達標 = 不進入下一 phase（無例外）

## Phase 0: CTBR 環境驗證
- [ ] hover 500 步，crash = 0（手動固定指令 [1,0,0,0]）
- [ ] 隨機 CTBR 指令，crash rate < 10%（1000 steps）
- [ ] Step response：ω_cmd → steady-state < 50ms

## Phase 1: PPO Expert (CTBR)
- [ ] RMSE < 0.10m（評估 50 episodes）
- [ ] Crash rate = 0/50
- [ ] Action smoothness: mean |Δaction| < 0.3
- [ ] WandB run ID: _______________

## Phase 2: Data Collection
- [ ] 1000 episodes 收集完成
- [ ] Data size ~ 4GB
- [ ] IMU 資料完整（無 NaN）
- [ ] Depth 資料完整

## Phase 3a: Flow Matching BC
- [ ] Training loss 穩定下降（無 NaN）
- [ ] Eval RMSE < 0.12m（sim，50 episodes）
- [ ] Inference latency < 10ms（單步，sim）
- [ ] WandB run ID: _______________

## Phase 3b: ReinFlow RL Fine-tuning
- [ ] RMSE < 0.08m
- [ ] Crash rate < 5%（sim）
- [ ] KL divergence stable（無爆炸）
- [ ] WandB run ID: _______________

## Phase 5: Jetson Deployment（必做）
- [ ] ONNX export 成功
- [ ] TensorRT engine 轉換成功
- [ ] Inference latency < 30ms（Jetson Orin Nano）
- [ ] ROS 2 node 發布頻率 > 30Hz
- [ ] Real-world hover > 10 秒 ✈️
