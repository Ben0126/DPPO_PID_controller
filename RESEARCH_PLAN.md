# Comprehensive Research Plan: Real-Time Adaptive PID Tuning via DPPO

## Executive Summary

This document outlines the complete research and development plan for implementing a Deep Reinforcement Learning (DRL) agent using **Diffusion Policy Policy Optimization (DPPO)** as a Meta-Controller to learn optimal strategies for real-time adjustment of PID gains for multi-axis control systems.

**Primary Research Focus**: Phase 3 - DPPO Policy Model Implementation and PPO Strategy Gradient Integration

---

## Project Goal

To implement a DPPO-based Meta-Controller that learns optimal strategies for real-time adjustment of PID gains ($K_p, K_i, K_d$) for the inner-loop attitude rate control of a simulated multi-axis system (Quadrotor), achieving superior:
- **Robustness** to disturbances and parameter variations
- **Tracking Performance** across diverse reference trajectories
- **Adaptability** compared to fixed controllers

---

## Development Phases Overview

```
Phase 1: Single-Axis Foundation (Current)
   ↓
Phase 2: DPPO MDP Definition
   ↓
Phase 3: DPPO Implementation (CORE RESEARCH)
   ↓
Phase 4: 6-DOF Quadrotor Scaling
   ↓
Phase 5: Evaluation & Deployment
```

---

## Phase 1: Single-Axis Environment Setup (Foundation)

**Objective**: Establish a simple, stable Gymnasium environment to validate the core DPPO-PID tuning mechanism before scaling.

### 1.1 System Dynamics (Single-Axis Simplification)

**Model**: Second-Order System (Angular Rate Control)

$$J \ddot{x} + B \dot{x} = u(t) + d(t)$$

| Component | Parameter | Value | Notes |
|-----------|-----------|-------|-------|
| **Model Variable** | $x$ | - | Angular velocity (rad/s) |
| **Inertia** | $J$ | 1.0 | Moment of inertia |
| **Damping** | $B$ | 0.5 | Damping coefficient |
| **Control Input** | $u(t)$ | $[-10, 10]$ | Actuator output limits |
| **Disturbance** | $d(t)$ | $\pm 0.5$ | Intermittent pulse disturbance |

**Implementation Requirements**:
- Use **RK4 (Runge-Kutta 4th Order)** integration for superior numerical stability
- Disturbances applied randomly for short durations to force adaptation

### 1.2 Cascaded Control and Time Base

| Parameter | Value | Role | Description |
|-----------|-------|------|-------------|
| **Inner Loop Δt** | 0.005 s (200 Hz) | PID calculation & physics integration | Fast control loop |
| **Outer Loop Δt_RL** | 0.05 s (20 Hz) | DPPO agent action frequency | PID gain updates |
| **Ratio N** | 10 | Steps per RL action | Inner steps per outer step |

### 1.3 PID Inner-Loop Parameters

| Gain Type | Initial Value | Maximum Bound (K_max) | Notes |
|-----------|---------------|----------------------|-------|
| **Proportional (K_p)** | 5.0 | 10.0 | Error response |
| **Integral (K_i)** | 0.1 | 5.0 | Steady-state error elimination |
| **Derivative (K_d)** | 0.2 | 5.0 | Damping & overshoot reduction |

**Critical Implementation**:
- **Anti-windup**: Stop integral accumulation when control saturates
- Action clipping to $[0, K_{\max}]$ bounds

---

## Phase 2: DPPO Meta-Controller MDP Definition

### 2.1 Action Space $\mathbf{A}$ (DPPO Output)

**Vector**: $\mathbf{A} = [K_p, K_i, K_d]$

**Properties**:
- Dimensionality: 3 (continuous)
- Range: $[0, K_{\max}]$ (clipped)
- Execution: Applied for 10 inner loop steps (0.05 seconds)

### 2.2 Observation Space $\mathbf{S}$ (DPPO Input)

**9-Dimensional Normalized Vector**:

$$\mathbf{S} = [e, \dot{e}, \int e, x, \dot{x}, r, K_p^{\text{curr}}, K_i^{\text{curr}}, K_d^{\text{curr}}]$$

| Index | Component | Description | Normalization |
|-------|-----------|-------------|---------------|
| 0 | $e$ | Current error | Running mean/std |
| 1 | $\dot{e}$ | Error derivative | Running mean/std |
| 2 | $\int e$ | Integral of error | Running mean/std |
| 3 | $x$ | System position/velocity | Running mean/std |
| 4 | $\dot{x}$ | System acceleration | Running mean/std |
| 5 | $r$ | Reference setpoint | Running mean/std |
| 6 | $K_p^{\text{curr}}$ | Current P gain | Running mean/std |
| 7 | $K_i^{\text{curr}}$ | Current I gain | Running mean/std |
| 8 | $K_d^{\text{curr}}$ | Current D gain | Running mean/std |

**⚠️ CRITICAL**: Must implement **Running Mean/Std Dev** normalization for ALL 9 components.

### 2.3 Reward Function $\mathbf{R}$ (Accumulated per Δt_RL)

**Total Reward (Summed over 10 inner steps)**:

$$\mathbf{R}_{\text{total}} = \sum_{i=1}^{10} \left( -\lambda_1 e_i^2 - \lambda_2 \dot{x}_i^2 - \lambda_3 u_i^2 - \lambda_4 \cdot \text{Overshoot}_i \right)$$

**Overshoot Penalty**: $\text{max}(0, e \cdot \dot{e})$

| Component | Weight (λ) | Description | Goal |
|-----------|-----------|-------------|------|
| **Tracking Error** | λ₁ = 5.0 | $-\lambda_1 e^2$ | Minimize primary error |
| **Velocity Oscillation** | λ₂ = 0.5 | $-\lambda_2 \dot{x}^2$ | Prevent high-frequency jitter |
| **Control Effort** | λ₃ = 0.01 | $-\lambda_3 u^2$ | Energy efficiency |
| **Overshoot/Instability** | λ₄ = 0.2 | $-\lambda_4 \max(0, e \cdot \dot{e})$ | **Core stability term** |

### 2.4 Program Flow: `step(action)` Algorithm

```
1. Gain Update:
   - Clip action [Kp, Ki, Kd] to valid bounds
   - Apply to PID controller

2. Inner Loop (10 iterations):
   FOR i = 1 to 10:
     a. Calculate current error e = r - x
     b. Compute PID output u = Kp*e + Ki*∫e + Kd*ė
     c. Apply disturbance d(t)
     d. Integrate physics (RK4) → update x, ẋ
     e. Update PID state:
        - Integral with anti-windup
        - Derivative term
     f. Calculate and accumulate reward Ri

3. Termination Check:
   - Stability violation (|x| > 5 or |ẋ| > threshold)
   - OR max episode steps reached

4. Return:
   - S_new: Updated observation
   - R_total: Accumulated reward
   - terminated: Boolean
   - truncated: Boolean
   - info: Dictionary
```

---

## Phase 3: DPPO Model Implementation & Training (CORE RESEARCH)

**⚠️ PRIMARY FOCUS AREA**: This phase represents the main technical challenge and academic contribution.

### 3.1 DPPO Policy Network Architecture (ε_θ)

**Network Type**: U-Net variant or Transformer-based model for conditional diffusion

**Inputs**:
1. **Conditioning**: Normalized observation $\mathbf{S}_{\text{curr}}$ (9D)
2. **Noisy Action**: Action vector $\mathbf{A}_t$ with added noise (3D)
3. **Timestep**: Positional encoding of noise level $t$

**Output**: Predicted noise vector $\epsilon_{\theta}$ (3D)

**Diffusion Parameters**:
- Timesteps $T$: 100 (starting value)
- Noise schedule: Linear or cosine $\beta_t$ schedule

**Architecture Components**:
```
Conditioning Path:
  S_curr → MLP Encoder → Conditioning Vector

Diffusion Path:
  [A_t, t] → Time Embedding → U-Net/Transformer → ε_θ

Combination:
  Cross-Attention or Concatenation of conditioning
```

### 3.2 DPPO Training Algorithm

**Step 1: Data Collection (Policy Rollout)**
```
1. Use current ε_θ and Reverse Diffusion Sampler (DDIM)
2. Generate action candidates A_curr from noise
3. Execute A_curr in environment
4. Collect trajectories τ = (S, A, R)
```

**Step 2: Advantage Estimation**
```
1. Calculate Value Target (Discounted Return G_t)
2. Compute Generalized Advantage Estimation (GAE):
   A_t = Σ(γλ)^k δ_{t+k}
   where δ_t = r_t + γV(s_{t+1}) - V(s_t)
```

**Step 3: DPPO Loss Function (THE CORE)**

```python
# Value Loss
L_value = MSE(V_φ(S), G_t)

# Policy Loss (Weighted Diffusion Loss)
L_policy = E[exp(β * A_t) * ||ε_θ(S, A_t, t) - ε||²]

# Total Loss
L_total = L_policy + c_value * L_value
```

**Key Insight**: The weight $\exp(\beta \cdot A_t)$ ensures the model focuses on correcting actions that lead to high reward.

**Step 4: Optimization**
```
Optimizer: Adam or AdamW
Update: ε_θ and V_φ based on L_total
```

### 3.3 Critical Prerequisites for DPPO

| Requirement | Technical Detail | Status |
|-------------|-----------------|--------|
| **Diffusion Model** | DDPM or DDIM formulas for reverse sampling | To implement |
| **Noise Schedule** | Stable β_t schedule (linear/cosine) | To implement |
| **Conditioning** | Cross-Attention or Concatenation with S | To implement |
| **PPO-Loss Integration** | Correct GAE computation and advantage weighting | **MOST CRITICAL** |
| **Sampler** | DDIM sampler for fast inference (<50ms) | To implement |

---

## Phase 4: Scaling to 6-DOF Quadrotor Model

### 4.1 6-DOF Dynamics and Cascaded Structure

**Full Non-linear Quadrotor Dynamics**:
- Attitude dynamics (quaternion representation)
- Position dynamics (X, Y, Z)
- Motor dynamics and thrust allocation
- Gravity and aerodynamic effects

**Cascaded Control Loops**:
```
Outer Loop (Position):   X, Y, Z control → desired angles
   ↓
Middle Loop (Attitude):  φ, θ, ψ control → desired rates
   ↓
Inner Loop (Rate):       ω_φ, ω_θ, ω_ψ control ← DPPO TARGET
```

### 4.2 Expanded Action Space (9D)

$$\mathbf{A}_{\text{6DOF}} = [K_p^{\text{roll}}, K_i^{\text{roll}}, K_d^{\text{roll}}, K_p^{\text{pitch}}, K_i^{\text{pitch}}, K_d^{\text{pitch}}, K_p^{\text{yaw}}, K_i^{\text{yaw}}, K_d^{\text{yaw}}]$$

**Properties**:
- Dimensionality: 9 (continuous)
- 3 PID gains per axis (roll, pitch, yaw)
- Independent tuning for each axis

### 4.3 Expanded Observation Space (27+ D)

$$\mathbf{S}_{\text{6DOF}} = [\mathbf{e}_{\text{rates}}, \text{Angles}, \text{Rates}, \mathbf{A}_{\text{curr}}]$$

**Components**:
- Error states: 3 rate errors + derivatives + integrals (9D)
- Attitude: φ, θ, ψ (3D)
- Angular rates: ω_φ, ω_θ, ω_ψ (3D)
- Additional states: position, velocity (6D)
- Current gains: 9D
- **Total**: ~27-30 dimensions

### 4.4 Expanded Reward Function

$$\mathbf{R}_{\text{6DOF}} = \sum_{\text{axes}} \left( -\lambda_1 e_{\text{axis}}^2 - \lambda_2 \dot{\omega}_{\text{axis}}^2 - \lambda_3 u_{\text{axis}}^2 \right) - \text{Penalties}$$

**Additional Penalties**:
- **Motor Saturation**: Heavy penalty if thrust clipped
- **Coupling**: Penalize unwanted motion on non-commanded axes
- **Position Drift**: Minor penalty for position/altitude instability

### 4.5 Implementation Requirements

| Component | Requirement | Priority |
|-----------|-------------|----------|
| **Attitude Representation** | Quaternions (avoid gimbal lock) | Critical |
| **Thrust Allocator** | 4D thrust → 4 motor commands | Critical |
| **Integration** | RK4 for nonlinear dynamics | Critical |
| **Anti-windup** | Per-axis integral clamping | High |
| **Normalization** | Running stats for 27+ dimensions | Critical |

---

## Phase 5: Evaluation and Deployment Planning

### 5.1 Baseline Controllers

**Comparison Baselines**:

1. **Fixed Manual PID**
   - Best manually-tuned gains for nominal conditions
   - Represents traditional engineering approach

2. **LQR Controller**
   - Linear Quadratic Regulator
   - Optimal control benchmark for linear systems
   - Tests if DPPO learns non-linear robustness

3. **Fixed RL-Optimized PID**
   - Average/final K values from DPPO training
   - Run without real-time adaptation
   - Demonstrates value of adaptation vs. static optimization

### 5.2 Evaluation Metrics

| Category | Metric | Formula | Goal |
|----------|--------|---------|------|
| **Tracking** | RMSE | $\sqrt{\frac{1}{T} \sum (r_t - x_t)^2}$ | Lower ↓ |
| **Transient** | Overshoot % | Peak / Steady-state | Lower ↓ |
| **Transient** | Settling Time | Time to ±2% of target | Shorter ↓ |
| **Robustness** | Max Load Tolerance | Max disturbance before unstable | Higher ↑ |
| **Efficiency** | Control Effort | $\sum u_t^2$ | Lower ↓ |
| **Feasibility** | Inference Latency | Computation time for action | <50ms |

### 5.3 Test Scenarios

**Trajectory Types**:
1. Step responses (various amplitudes)
2. Sine wave tracking (varying frequencies)
3. Aggressive maneuvers (square waves, ramps)
4. Multi-axis coordinated movements

**Disturbance Profiles**:
1. No disturbance (baseline)
2. Intermittent pulses (as in training)
3. Continuous noise (wind simulation)
4. Parameter variations (inertia changes, mass changes)
5. Unseen disturbances (test generalization)

---

## Appendix A: Critical Implementation Checklist

### A.1 DRL and DPPO Algorithm (Phase 3)

- [ ] Select diffusion model variant (DDPM/DDIM)
- [ ] Implement noise schedule (β_t)
- [ ] Design conditioning mechanism (cross-attention/concat)
- [ ] Implement reverse diffusion sampler
- [ ] **Compute GAE advantages correctly**
- [ ] **Integrate advantages into diffusion loss**
- [ ] Implement value network V_φ
- [ ] Set up training loop with proper sampling
- [ ] Profile inference time (<50ms requirement)

### A.2 Environment Stability (Phase 1 & 4)

- [ ] **Implement Running Mean/Std normalization**
- [ ] Implement RK4 integration
- [ ] Add PID anti-windup logic
- [ ] Validate numerical stability
- [ ] Test edge cases (saturation, large errors)
- [ ] Implement proper termination conditions

### A.3 6-DOF Implementation (Phase 4)

- [ ] Implement quaternion attitude representation
- [ ] Implement quaternion integration
- [ ] Design thrust allocation matrix
- [ ] Implement motor dynamics
- [ ] Add gravity and aerodynamic forces
- [ ] Validate against known quadrotor models
- [ ] Test all control loops independently

---

## Appendix B: Technical Specifications

### B.1 Diffusion Model Details

**Forward Process** (Adding Noise):
$$q(A_t | A_{t-1}) = \mathcal{N}(A_t; \sqrt{1-\beta_t} A_{t-1}, \beta_t I)$$

**Reverse Process** (Denoising):
$$p_\theta(A_{t-1} | A_t, S) = \mathcal{N}(A_{t-1}; \mu_\theta(A_t, S, t), \Sigma_\theta(A_t, S, t))$$

**Noise Schedule**:
- Linear: $\beta_t = \beta_{\min} + \frac{t}{T}(\beta_{\max} - \beta_{\min})$
- Cosine: $\beta_t = 1 - \frac{\alpha_t}{\alpha_{t-1}}$, where $\alpha_t = \frac{f(t)}{f(0)}$

### B.2 GAE (Generalized Advantage Estimation)

$$A_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

where $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$

**Parameters**:
- γ (gamma): 0.99 (discount factor)
- λ (lambda): 0.95 (GAE parameter)

### B.3 DPPO Policy Loss (Detailed)

```python
def dppo_policy_loss(epsilon_theta, states, actions, timesteps,
                     advantages, beta=1.0):
    """
    Compute DPPO policy loss.

    Args:
        epsilon_theta: Denoising network
        states: Batch of states S
        actions: Batch of noisy actions A_t
        timesteps: Diffusion timesteps t
        advantages: Computed advantages A_t
        beta: Advantage weighting coefficient

    Returns:
        Weighted diffusion loss
    """
    # Predict noise
    pred_noise = epsilon_theta(states, actions, timesteps)

    # Actual noise (from forward process)
    actual_noise = compute_actual_noise(actions, timesteps)

    # MSE loss
    mse_loss = (pred_noise - actual_noise) ** 2

    # Advantage weighting
    weights = torch.exp(beta * advantages)
    weights = weights / weights.sum()  # Normalize

    # Weighted loss
    loss = (weights * mse_loss).mean()

    return loss
```

---

## Appendix C: Development Roadmap

### Timeline Estimate

| Phase | Duration | Effort | Priority |
|-------|----------|--------|----------|
| Phase 1 | 1-2 weeks | Medium | High |
| Phase 2 | 1 week | Low | High |
| **Phase 3** | **4-6 weeks** | **Very High** | **CRITICAL** |
| Phase 4 | 3-4 weeks | High | Medium |
| Phase 5 | 2-3 weeks | Medium | Medium |
| **Total** | **11-16 weeks** | | |

### Resource Requirements

**Computational**:
- GPU with 8GB+ VRAM (for DPPO training)
- Multi-core CPU for parallel environment sampling
- TensorBoard for monitoring

**Software**:
- PyTorch (for DPPO implementation)
- Gymnasium (environment)
- NumPy/SciPy (dynamics)
- Matplotlib (visualization)

**Knowledge**:
- Diffusion models theory
- Reinforcement learning (PPO, GAE)
- Control theory (PID, LQR)
- Quadrotor dynamics

---

## Appendix D: Risk Mitigation

### Technical Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| DPPO training instability | High | Medium | Start with PPO, gradually add diffusion |
| Inference latency >50ms | High | Medium | Optimize with DDIM, reduce T, model compression |
| Poor transfer to 6-DOF | Medium | Low | Thorough Phase 1 validation |
| Diffusion model implementation complexity | High | High | Use existing libraries (diffusers) |

### Schedule Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Phase 3 takes longer | High | Allocate buffer time, simplify if needed |
| Debugging 6-DOF dynamics | Medium | Use validated models, unit tests |
| Evaluation takes too long | Low | Automate benchmarking scripts |

---

## References

1. **Diffusion Models**:
   - Ho et al. (2020). "Denoising Diffusion Probabilistic Models"
   - Song et al. (2020). "Denoising Diffusion Implicit Models"

2. **Policy Optimization**:
   - Schulman et al. (2017). "Proximal Policy Optimization"
   - Wang et al. (2022). "Diffusion Policy" (inspiration for DPPO)

3. **Control Theory**:
   - Åström & Murray (2008). "Feedback Systems"
   - Beard & McLain (2012). "Small Unmanned Aircraft"

---

## Document Version

- **Version**: 1.0
- **Date**: 2025-11-07
- **Status**: Active Development Plan
- **Next Review**: After Phase 1 completion
