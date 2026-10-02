# Vision-DPPO: Flow-Matching Visuomotor Policies for Quadrotor Hover (Simulation Study, Negative Result)
### 以 flow-matching 視覺策略做四旋翼懸停的模擬研究（負面結果與診斷）

> **About the name:** the repo name is historical. The project started as a diffusion policy with PPO fine-tuning (DPPO). The results reported here come from behaviour-cloned flow-matching policies.

**At a glance**
- **What I built:** a 6-DOF quadrotor simulator with a 64×64 FPV renderer, a state-based PPO expert, and a 13.5 M-parameter flow-matching vision + IMU policy, all in PyTorch.
- **What I found:** my original metric made policies that crashed early look precise. I replaced it with a frozen, multi-seed evaluation protocol.
- **Result:** an anti-collapse regulariser (Dispersive Loss) gave no gain. Under every intervention I tried, hover error stayed at about 2.4–3.0 m. The state-based oracle reaches 0.068 m.

This is a **simulation-only** study. The policy sees a 64×64 first-person camera, an IMU and a 2-value hover/recovery mode tag, and has to hover a quadrotor. At evaluation the mode tag is computed from the true state (position error, tilt, body rate), so it is one bit of privileged information. The study asks two things:
1. Can a flow-matching visuomotor policy do this hover task?
2. Does **Dispersive Loss**, a contrastive regulariser meant to prevent feature collapse, improve the policy's closed-loop control?

**The result is negative.** The data did not support Dispersive Loss. Hover precision stayed far from the oracle under every fix that was tested.

---

## Status

| | |
|---|---|
| Project | Finished. Feb–Jun 2026. Independent, self-directed project with no advisor. I studied the concepts and algorithms in February and wrote the code from March to June. |
| Write-up | A draft (`docs/paper_negative_result_draft.*`). **It has not been submitted, and I do not plan to submit it.** The venue lines inside the draft are left over from an earlier plan. |
| Hardware | **No real-robot claim.** Every result comes from the simulator in `envs/`. |
| Result files | `evaluation_results/`, `checkpoints/` and `data/*.h5` are gitignored, so they are **not in this repository**. The numbers below are copied from the write-up and from the experiment reports in `docs/`. |

---

## TL;DR

**Question.** A flow-matching policy sees a 2-frame 64×64 FPV stack, an IMU and a hover/recovery mode tag. Can it hover a quadrotor in closed loop at 50 Hz? And does Dispersive Loss improve that control?

**What was built.** Everything below runs in simulation:
- a 6-DOF quadrotor simulator: RK4 integration, a 200 Hz INDI rate loop and a 50 Hz policy loop.
- a state-based PPO expert. It serves as the oracle. It also produced the demonstrations for the initial BC policy and for the Dispersive study.
- a cascade PID-CTBR teacher. It produced both the hover and the far-range demonstrations for the Teacher × Observation study and the capacity study.
- a vision + IMU flow-matching policy, trained by behaviour cloning.
- three controlled ablations, each cell trained with 3 seeds.

**Metrics used below** (defined in full under *Evaluation protocol*):
- **Tier-1:** the % of episodes that fly at least half of the 500-step horizon.
- **Survival:** the mean fraction of the horizon flown.
- **cond-IAE:** the mean position error in metres. It is measured over the second half of each episode, and only over episodes that reached Tier-1.
- **pp:** percentage points.

**The data did not support Dispersive Loss.** I implemented it to match the official reference code: InfoNCE-L2 on the `flow_net` mid-block, λ=0.5, τ=0.5, with the `/d` normalisation.
- With a trainable encoder, adding it changed Tier-1 by −2.2 pp. That is inside the across-seed noise of 6.3 pp. Survival changed by −2.1 pp.
- With a frozen encoder, adding it lowered Tier-1 from 87.8 to 74.4.

**There is a precision floor.**
- Under the sensing, far-range-coverage and capacity interventions, cond-IAE stayed at about **2.4–3.0 m**.
- Across every configuration in the write-up, it spans roughly **2.4–3.3 m**.
- The state-based oracle reaches **0.068 m**.

**I tested several explanations. None brought precision near the oracle.** None got below the 1.5 m target set in the research plan. The factors tested were:
- feature collapse in the representation;
- sensing, which included handing the policy the true range;
- data coverage, using a far-range teacher together with a range-encoding render;
- model capacity: making the action network 3.3× larger.

The two largest effects:
- **Capacity:** about 0.3 m better with 3.3× more parameters (XL vs S), only just above seed noise.
- **Oracle range:** about 0.5 m better when the policy was given the true range. This cost 6.7 pp of survival, and 0.15 m of sensor noise removed the gain.

Three seeds are only enough to say these explanations were *not supported*. They are not enough to rule them out.

**What remains is a robustness–precision trade-off, and the data show it directly.** Adding far-range recovery data raises survival. On the perspective cell, it also makes precision worse (2.48 → 2.93 m). I have not tested *why* the trade-off exists. Two candidate causes remain:
- how model capacity is allocated;
- a limit of closed-loop control under partial observability.

**Main methodology contribution: a frozen evaluation protocol.**
- **The problem.** The older RMSE metric averaged error only over the steps a policy was still flying (`scripts/evaluate_rhc_v4.py:91–92`). Policies that crashed early therefore looked precise.
- **The fix.** The frozen protocol (`scripts/evaluate_frozen_p0.py`):
  - gives every model the same paired initial conditions;
  - reports precision only over episodes that survived;
  - adds bootstrap confidence intervals;
  - reports mean ± std across training seeds;
  - normalises against a *measured* oracle.
- **The effect.** Under this protocol, two earlier claims did not hold: "H4 is best" and "RL gives a 51 % precision gain".

---

## Final architecture (v5 flow policy)

```
 64x64 RGB FPV x 2 frames (6 ch) ──► CNN encoder (VisionEncoderV5) ──► spatial map 256x4x4
                                                                        │ (keys/values)
 6-D IMU (gyro + specific force) ──► IMU MLP 6→1024→512 ──► imu_feat 512 ─┤ (query)
                                                                        ▼
                                           IMU→vision cross-attention ──► attended 256
                                                                        │
          global_cond = [attended 256 ; imu_feat 512 ; task tag 2 (hover / recovery)]
                                                                        ▼
               flow-matching 1-D conditional U-Net (flow_net, down_dims = (256, 512))
               2 Euler steps at eval ──► chunk of 8 CTBR commands; execute the first, re-plan
                                                                        ▼
            CTBR = collective thrust + body rates, 50 Hz  (envs/quadrotor_env_v4.py)
                                                                        ▼
                  INDI rate controller, 200 Hz ──► per-motor commands ──► 6-DOF rigid body (RK4)
```

**Policy**
- Code: `models/flow_policy_v5.py` (`FlowMatchingPolicyV5`) and `models/vision_encoder_v5.py`.
- A training-only auxiliary head predicts the 15-D state from the pooled vision feature.
- The final evaluated policy has 13.5 M trainable parameters (`docs/experiment_report_p2cap_capacity.md`).

**Dispersive Loss variants** (all in `models/flow_policy_v5.py`)
- `_dispersive_loss_infonce`: the faithful version.
- `_dispersive_loss_cosine`
- `_dispersive_loss_vicreg`

**Simulator**
- Dynamics: `envs/quadrotor_dynamics.py` (RK4, `dt = 0.005`).
- CTBR action space and INDI loop: `envs/quadrotor_env_v4.py`.
- FPV renderer: `envs/quadrotor_visual_env.py`.
- Loop rates: `configs/quadrotor_v4.yaml` (`dt_inner: 0.005`, `dt_outer: 0.02`).

**Experts**
- **State-based PPO expert** (`models/ppo_expert.py`, `scripts/train_ppo_expert_v4.py`):
  - the oracle;
  - the demonstrator for the initial "H4" BC policy and for the Dispersive study.
- **Cascade PID-CTBR teacher** (`controllers/pid_controller.py`):
  - produced all hover and far-range demonstrations for Tables 6–7, through `scripts/collect_data_v7_pidctbr.py`;
  - one teacher was used for both, so the teacher axis changes only data coverage.

---

## Key results

How to read the tables:
- Every row is the **mean ± std over 3 training seeds**.
- Each seed is scored with the frozen protocol: 30 paired episodes.
- cond-IAE is in metres; lower is better.
- State oracle: 0.068 m.

**1. Dispersive × encoder (Table 2 in the draft).**
- The decisive comparison is D1E1 vs D0E1.
- Sources: `docs/paper_negative_result_draft.md` §5 and `docs/experiment_report_faithful_dispersive.md`.

| Cell | Dispersive / encoder | Tier-1 % | Survival % | cond-IAE |
|---|---|---:|---:|---:|
| D0E0 | off / frozen | 87.8 ± 3.1 | 66.1 ± 3.7 | 2.93 |
| D1E0 | on / frozen | 74.4 ± 8.7 | 60.4 ± 1.6 | 2.81 |
| D0E1 | off / trainable | 92.2 ± 3.1 | 65.0 ± 2.8 | 2.91 |
| D1E1 | on / trainable | 90.0 ± 5.4 | 62.9 ± 2.4 | 2.89 |

**2. Teacher coverage × observation (Table 6).**
- **T1** adds far-range recovery demonstrations from the PID-CTBR teacher.
- **O1** replaces the crosshair target with a perspective target. The crosshair stops changing size at far range; the perspective target encodes range.
- Sources: §6.4 and `docs/experiment_report_p2to_decisive.md`.

| Cell | Teacher / observation | cond-IAE | Survival % | Tier-1 % |
|---|---|---:|---:|---:|
| T0O0 | hover only / crosshair | 2.69 ± 0.15 | 83.2 ± 5.8 | 98.9 ± 1.6 |
| T0O1 | hover only / perspective | 2.48 ± 0.14 | 76.7 ± 10.5 | 86.7 ± 9.8 |
| T1O0 | + far-range / crosshair | 2.71 ± 0.22 | 91.4 ± 1.5 | 100.0 ± 0.0 |
| T1O1 | + far-range / perspective | 2.93 ± 0.18 | 90.6 ± 3.3 | 98.9 ± 1.6 |

What the table shows:
- **Floor:** T1O1 vs T0O0 differs by Δ = +0.24 m, against a pooled std of 0.23 m. The floor did not break.
- **Coverage:** adding far-range data raised survival but did not improve precision.
- **Perspective column:** adding coverage made precision worse (2.48 → 2.93 m).

**3. Capacity ladder (Table 7).**
- The T1O1 recipe is held fixed. Only `flow_net` `down_dims` changes.
- Sources: §6.5 and `docs/experiment_report_p2cap_capacity.md`.

| Rung | `down_dims` | Trainable params | cond-IAE | Survival % | Tier-1 % |
|---|---|---:|---:|---:|---:|
| S (= T1O1) | (256, 512) | 13.5 M | 2.93 ± 0.18 | 90.6 ± 3.3 | 98.9 |
| M | (256, 512, 768) | 35.0 M | 2.67 ± 0.26 | 86.7 ± 2.3 | 100.0 |
| XL | (512, 1024) | 44.0 M | 2.63 ± 0.19 | 86.6 ± 6.7 | 94.4 |

What the table shows:
- **Size of the gain:** 3.3× more parameters lowered cond-IAE by about 0.3 m. The gain is already flattening: M → XL improves by only 0.04 m. XL is still about 39× the oracle error.
- **Confound:** XL's wider `flow_net` could not reuse the transferred weights, so it was trained from scratch. The XL-vs-S gap therefore mixes capacity with initialisation.
- **M vs S:** M was partially transferred. Its gap to S (−0.26 m) is inside seed noise (pooled std 0.32 m) (`docs/experiment_report_p2cap_capacity.md`, rung table and verdict).

**Also in the write-up:**
- feature-geometry measurements: Dispersive Loss inflates the feature norm instead of raising the effective rank (§6.1, Table 5);
- the sensing gate and the range-cue intervention (§6.3);
- two baselines, BC-vision-only and PPO-from-pixels (§4, Table 1).

---

## Evaluation protocol

`scripts/evaluate_frozen_p0.py` fixes one protocol so that numbers stay comparable across runs.

- **Paired initial conditions.** Episode *i* uses `seed = 12345 + i` for three things: the environment, the global NumPy RNG (visual domain randomisation) and torch (flow noise). Every model therefore starts from the same conditions.
- **Survival and Tier-1.**
  - Survival is the mean fraction of the 500-step episode flown.
  - Tier-1 is the fraction of episodes that fly at least half the horizon (≥ 250 steps), defined at `scripts/evaluate_hierarchical.py:166`.
- **cond-IAE.**
  - It is the mean position-error norm over the second half of each episode (`scripts/evaluate_hierarchical.py:134`).
  - It is averaged **only over episodes that survived ≥ 250 steps**, and reported together with `n_cond`, the number of such episodes.
  - Despite the name, it is a mean distance in metres, not an integral.
  - The all-episode version is still reported but flagged, because early crashes make it look better than it is.
- **Bootstrap 95 % CI.** Percentile CIs over episodes, for the composite score and for survival.
- **Seed variation.** Every retrained model is reported as the mean ± std over 3 training seeds. On a single seed, the PPO-from-pixels baseline swung between 0 % and 47 % Tier-1.
- **Decision rule and "pooled std".**
  - "Pooled std" is the root-sum-square of the two cells' across-seed stds, i.e. the std of the difference of means (`scripts/evaluate_p2_ablation.py:138`, `scripts/evaluate_p2to_ablation.py`).
  - The stds use ddof=0 over 3 seeds, so they understate the sample std by about 18 %.
- **Measured oracle.**
  - The state-based PPO oracle runs through the same protocol (`--oracle-ckpt` / `--oracle-norm`): 100 % survival, 0.068 m, composite 0.9668.
  - This replaces an older hard-coded 0.85. The script still falls back to 0.85 when no oracle checkpoint is given.

---

## Reproduce

> **Not included:** datasets (`data/*.h5`), checkpoints (`checkpoints/`) and result JSONs (`evaluation_results/`) are gitignored. You have to collect the data and train the models yourself.
>
> **Original environment:** a single RTX 3090 on Windows, with a venv in `dppo/` (also gitignored). Run every command from the repo root.

```bash
pip install -r requirements.txt
python check_device.py                      # confirms CUDA is visible

# 1) State-based PPO expert (= oracle; demonstrator for H4 and the Dispersive study)
python -m scripts.train_ppo_expert_v4       # -> checkpoints/ppo_expert_v4/<ts>/best_model.pt, best_obs_rms.npz
EXP=checkpoints/ppo_expert_v4/<ts>

# 2) FPV demonstrations
python -m scripts.collect_data_v4          --model $EXP/best_model.pt --norm $EXP/best_obs_rms.npz   # data/expert_demos_v4.h5
python -m scripts.collect_data_v4_recovery --model $EXP/best_model.pt --norm $EXP/best_obs_rms.npz   # data/expert_demos_v4_recovery.h5
python -m scripts.collect_data_v7_pidctbr --mode hover   # PID-CTBR teacher -> data/expert_demos_v7_hover_{crosshair,persp}.h5
python -m scripts.collect_data_v7_pidctbr --mode far     # PID-CTBR teacher -> data/expert_demos_v7_far_{crosshair,persp}.h5

# 3) Shared initialisation (the "H4" BC policy, trained on data/expert_demos_v4.h5)
python -m scripts.train_flow_v4 --config configs/flow_policy_v4.yaml
H4=checkpoints/flow_policy_v4/<ts>/best_model.pt
ORACLE="--oracle-ckpt $EXP/best_model.pt --oracle-norm $EXP/best_obs_rms.npz"

# 4a) Dispersive x encoder 2x2 (faithful Dispersive)
python -m scripts.run_p2_ablation --faithful --h4-ckpt $H4
python -m scripts.evaluate_p2_ablation --manifest evaluation_results/p2f_ablation_manifest.json \
    --output evaluation_results/p2f_ablation_leaderboard.json $ORACLE

# 4b) Teacher x Observation 2x2
python -m scripts.run_p2to_ablation --h4-ckpt $H4
python -m scripts.evaluate_p2to_ablation $ORACLE

# 4c) Capacity ladder (rung S reuses the T1O1 checkpoints from 4b)
python -m scripts.run_p2cap_ablation --h4-ckpt $H4
python -m scripts.evaluate_p2cap_ablation $ORACLE

# Figures and the HTML/PDF write-up (both read evaluation_results/; see notes below)
python scripts/make_paper_figures.py
python scripts/export_paper.py
```

Notes on these commands:
- **Driver flags.** Each `run_*` driver accepts two flags:
  - `--dry-run` prints the training commands without launching them;
  - `--quick` runs a short smoke test.
- **`make_paper_figures.py` needs every study's results.** It reads the result JSONs of *all* the studies listed under *Reproducibility / Artifacts* in the draft: the baselines, feature geometry, sensing probes and the legacy P2 run. Steps 4a–4c alone are not enough, and the script stops with `FileNotFoundError`.
- **`export_paper.py` has extra requirements.**
  - It needs `pip install markdown`, which is not in `requirements.txt`.
  - It needs a Chromium binary for the PDF. It finds one automatically only at the Windows Chrome/Edge install paths. On any other system, pass `--browser /path/to/chrome`, or it produces HTML only.
- **Other scripts.** The draft's *Reproducibility / Artifacts* section lists the scripts for every table and figure. That includes the baselines, the feature-geometry measurements, the sensing probes and the scale-invariant regulariser runs.

---

## Repo map

| Path | What it is |
|---|---|
| `envs/` | 6-DOF dynamics (RK4), CTBR + INDI environment (`quadrotor_env_v4.py`), FPV renderer (`quadrotor_visual_env.py`) |
| `models/` | Current: `flow_policy_v5.py`, `vision_encoder_v5.py`, `conditional_unet1d.py`, `ppo_expert.py`, `ppo_pixel.py`. Older generations: `flow_policy_v4.py`, `diffusion_policy.py`, `vision_dppo_v31.py`, … |
| `controllers/` | `pid_controller.py`: the cascade PID-CTBR teacher |
| `configs/` | YAML for the simulator, the PPO expert, the flow policies and RL fine-tuning |
| `scripts/` | Training, data collection, evaluation (`evaluate_frozen_p0.py`), ablation drivers (`run_p2*` / `evaluate_p2*`), diagnostics (`measure_*`), paper tools (`make_paper_figures.py`, `export_paper.py`). Also holds scripts from earlier versions (`*_v31`, `*_v33`, `train_dppo*.py`, `train_reinflow_*.py`). |
| `docs/` | The write-up draft (`.md` / `.html` / `.pdf`), `experiment_report_*.md`, `dev_log*.md`. `architecture.md` is the v4.0 diagram from May 2026: it shows the earlier RL fine-tuning pipeline, not the final v5 policy above. |
| `utils/` | Helpers for metric logging and plotting |
| `presentation/` | Progress-report slides from May 2026, made before the final results |
| `RESEARCH_PLAN*.md` | The plan for each research stage |
| `data/` | Empty placeholder (`.gitkeep`). Datasets are not committed. |
| `CLAUDE.md`, `gemini.md`, `setup_claude_code.sh`, `docs/*guide*.md` | AI-assistant working notes and setup files. They are internal notes, not results. Some lines in them are out of date (for example, target venues and hardware deployment). Where they disagree with this README, this README is current. |

---

## Write-up

**Draft**
- Read the PDF: [`docs/paper_negative_result_draft.pdf`](docs/paper_negative_result_draft.pdf). It is also available as [HTML](docs/paper_negative_result_draft.html). Both have the figures embedded.
- The [Markdown source](docs/paper_negative_result_draft.md) links to `docs/figures/*.png`, but `*.png` is gitignored, so **the images do not render in the `.md` view on GitHub.**

**Known overstatements in the draft.** Where the draft and this README differ, use this README:
- The draft says "pre-registered". The plans and the results were committed together, so the order in which they were written cannot be proven.
- The title and abstract say the tested factors are ruled out, for example "…Is Not the Bottleneck — and Neither Is Coverage or Sensing". Three seeds can only show that they were not supported.
- The draft says the results and artifacts are released, and that "every number is reproducible from the cited artifact". The result JSONs and checkpoints are not in this repo.

**Experiment reports** (one per study)
- [faithful Dispersive](docs/experiment_report_faithful_dispersive.md)
- [feature collapse](docs/experiment_report_feature_collapse.md)
- [scale-invariant forms](docs/experiment_report_p6_scale_invariant.md)
- [survival movers](docs/experiment_report_survival_movers.md)
- [OOD coverage](docs/experiment_report_ood_coverage.md)
- [image–distance information](docs/experiment_report_image_distance_info.md)
- [sensing ablation](docs/experiment_report_sensing_ablation.md)
- [teacher/renderer gates](docs/experiment_report_p0_teacher_renderer_gates.md)
- [dataset collection](docs/experiment_report_p3_dataset_collection.md)
- [Teacher × Observation](docs/experiment_report_p2to_decisive.md)
- [capacity ladder](docs/experiment_report_p2cap_capacity.md)
- [joint training](docs/experiment_report_joint_e2e.md)

---

## History

- **The original plan:**
  - a diffusion policy with a ViT encoder (never implemented; the encoder was always a CNN);
  - one-step distillation;
  - per-motor commands as the action space. These were later replaced by thrust + body-rate (CTBR) commands over an INDI rate loop.
- **How it ended:**
  - More than two dozen ReinFlow/AWR-style RL fine-tuning runs degraded (`docs/dev_log_v4_h4_hierarchical.md`).
  - A later masked-advantage RL run (v5_RL_best, Table 1) turned out to be an artifact of short survival under the frozen protocol: only 4 of 30 episodes reached the cond-IAE threshold.
  - The work then moved to BC flow matching and the frozen evaluation protocol.
- **Old README:** in the git history (the version of `README.md` before this one). It describes the original plan, not the results.
- **Plans:** [`RESEARCH_PLAN_v6.md`](RESEARCH_PLAN_v6.md) through [`RESEARCH_PLAN_v9.md`](RESEARCH_PLAN_v9.md). v9 is a follow-up plan, and the write-up does not report its results.
- **Development logs:** `docs/dev_log*.md`.
- **Earliest commits:** the Nov 2025 commits belong to an earlier PID-tuning experiment in the same repository. The vision work starts in March 2026.

---

## How this was built

I used Claude Code as a pair programmer for implementation and for drafting documents. The research questions, the experimental decisions and the conclusions are mine. `CLAUDE.md` holds the working rules the assistant followed.

---

## License

All rights reserved: no open-source license has been added.
