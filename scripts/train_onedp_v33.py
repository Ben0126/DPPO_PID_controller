"""
Phase 3d: OneDP Single-Step Distillation (v3.3)

Teacher  : VisionDPPOv31 (DPPO fine-tuned, frozen, 10-step DDIM)
Student  : VisionDPPOv31 (1-step inference, trainable)

Distillation loss:
    L = MSE(x0_student, x0_teacher)
      + lambda_dispersive * L_dispersive
      + lambda_depth * L_depth

Root cause of Phase 3c bottleneck: DDIM 10-step = 74ms >> 20ms control
period (50Hz). 1-step distillation targets ~13ms inference.

Usage:
    python -m scripts.train_onedp_v33 \\
        --teacher checkpoints/diffusion_policy/dppo_v33_20260413_033647/best_dppo_v33_model.pt \\
        --config  configs/diffusion_policy.yaml \\
        2>&1 | tee logs/train_onedp_v33_$(date +%Y%m%d_%H%M%S).log
"""

import os
import sys
import argparse
import yaml
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vision_dppo_v31 import VisionDPPOv31, DemoDatasetV31


# ----------------------------------------------------------------------------
# v3.3 dataset alias (reads from data/v33_mmap/)
# ----------------------------------------------------------------------------
class DemoDatasetV33(DemoDatasetV31):
    MMAP_DIR = 'data/v33_mmap'


def _ensure_v33_cache(hdf5_path: str):
    """Build data/v33_mmap/ on first run (one-shot ~1 min)."""
    cache_dir = DemoDatasetV33.MMAP_DIR
    if os.path.isfile(f'{cache_dir}/images.dat'):
        return
    print(f"[v33] Building memmap cache at {cache_dir}/ from {hdf5_path} ...")
    DemoDatasetV33.build_mmap_cache(hdf5_path, out_dir=cache_dir)


def train(args):
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # -----------------------------------------------------------------------
    # Hyperparameters
    # -----------------------------------------------------------------------
    num_epochs        = args.num_epochs       # 50
    batch_size        = args.batch_size       # 128
    learning_rate     = args.lr               # 3e-4
    weight_decay      = args.weight_decay     # 0.01
    warmup_epochs     = args.warmup_epochs    # 5
    grad_clip         = args.grad_clip        # 1.0
    lambda_dispersive = args.lambda_disp      # 0.05
    lambda_depth      = args.lambda_depth     # 0.1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    vision_cfg = config['vision']
    diff_cfg   = config['diffusion']
    unet_cfg   = config['unet']
    action_cfg = config['action']
    log_cfg    = config['logging']

    # -----------------------------------------------------------------------
    # Dataset
    # -----------------------------------------------------------------------
    dataset_path = args.dataset or 'data/expert_demos_v33.h5'
    _ensure_v33_cache(dataset_path)

    dataset = DemoDatasetV33(
        hdf5_path=dataset_path,
        T_obs=vision_cfg['T_obs'],
        T_pred=action_cfg['T_pred'],
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Dataset: {len(dataset)} samples, {len(dataloader)} batches/epoch")
    print(f"λ_dispersive={lambda_dispersive}  λ_depth={lambda_depth}")

    # -----------------------------------------------------------------------
    # Model constructor args (shared by teacher and student)
    # -----------------------------------------------------------------------
    model_kwargs = dict(
        action_dim=action_cfg['action_dim'],
        T_obs=vision_cfg['T_obs'],
        T_pred=action_cfg['T_pred'],
        image_channels=vision_cfg['channels'],
        image_size=vision_cfg['image_size'],
        time_embed_dim=unet_cfg['time_embed_dim'],
        down_dims=tuple(unet_cfg['down_dims']),
        num_diffusion_steps=diff_cfg['num_timesteps'],
        beta_schedule=diff_cfg['beta_schedule'],
        ddim_steps=diff_cfg['ddim_steps'],
    )

    # -----------------------------------------------------------------------
    # Teacher: frozen DPPO-fine-tuned model (10-step DDIM)
    # -----------------------------------------------------------------------
    teacher = VisionDPPOv31(**model_kwargs, use_depth_decoder=False).to(device)
    teacher.load(args.teacher)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    print(f"Teacher loaded from: {args.teacher}")

    # -----------------------------------------------------------------------
    # Student: trainable, initialised from teacher weights
    # -----------------------------------------------------------------------
    student = VisionDPPOv31(**model_kwargs, use_depth_decoder=(lambda_depth > 0)).to(device)
    student.load(args.teacher)   # copies encoder + noise_pred_net from DPPO ckpt

    # depth_decoder cold-start: load from supervised v33 checkpoint if available
    supervised_ckpt = args.supervised_ckpt or \
        'checkpoints/diffusion_policy/v33_20260412_052333/best_model.pt'
    if lambda_depth > 0 and student.depth_decoder is not None and \
            os.path.exists(supervised_ckpt):
        ckpt = torch.load(supervised_ckpt, map_location='cpu')
        dd_state = ckpt.get('depth_decoder')
        if dd_state is not None:
            student.depth_decoder.load_state_dict(dd_state)
            print(f"depth_decoder initialised from: {supervised_ckpt}")
        else:
            print(f"[warn] 'depth_decoder' key not found in {supervised_ckpt}; random init")
    elif lambda_depth > 0 and student.depth_decoder is not None:
        print(f"[warn] Supervised ckpt not found at {supervised_ckpt}; depth_decoder random init")

    n_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"Student trainable parameters: {n_params:,}")

    # -----------------------------------------------------------------------
    # Optimizer + cosine-with-warmup scheduler
    # -----------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, student.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    total_steps  = num_epochs * len(dataloader)
    warmup_steps = warmup_epochs * len(dataloader)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # -----------------------------------------------------------------------
    # Logging / checkpoints
    # -----------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag   = f"onedp_v33_{timestamp}"
    log_dir   = os.path.join(log_cfg['tensorboard_log'], run_tag)
    save_dir  = os.path.join(log_cfg['save_path'], run_tag)
    os.makedirs(log_dir,  exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    # File logger
    log_file = f"logs/train_onedp_v33_{timestamp}.log"
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger(__name__)

    writer = SummaryWriter(log_dir)
    log_freq = log_cfg.get('log_freq', 10)

    logger.info(f"{'='*60}")
    logger.info(f"Phase 3d: OneDP 1-step distillation (v3.3)")
    logger.info(f"Teacher : {args.teacher}")
    logger.info(f"Epochs  : {num_epochs}  Batch: {batch_size}  LR: {learning_rate:.1e}")
    logger.info(f"Save dir: {save_dir}")
    logger.info(f"{'='*60}")

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    global_step = 0
    best_loss   = float('inf')

    for epoch in range(num_epochs):
        student.train()
        epoch_losses  = []
        epoch_l_dist  = []
        epoch_l_disp  = []
        epoch_l_depth = []

        for batch_idx, (img_stack, _, imu, depth_gt) in enumerate(dataloader):
            img_stack = img_stack.to(device, non_blocking=True)
            imu       = imu.to(device, non_blocking=True)
            depth_gt  = depth_gt.to(device, non_blocking=True)
            B = img_stack.shape[0]

            # On-GPU augmentation (brightness + contrast jitter)
            brightness = 1.0 + (torch.rand(B, 1, 1, 1, device=device) - 0.5) * 0.6
            img_mean   = img_stack.mean(dim=(-2, -1), keepdim=True)
            contrast   = 1.0 + (torch.rand(B, 1, 1, 1, device=device) - 0.5) * 0.4
            img_stack  = torch.clamp(
                (img_stack - img_mean) * contrast + img_mean * brightness,
                0.0, 255.0
            )

            # ------------------------------------------------------------------
            # Teacher: generate x0_teacher (10-step DDIM, no grad)
            # ------------------------------------------------------------------
            with torch.no_grad():
                global_cond_t, _ = teacher._encode(img_stack, imu)
                shape = (B, teacher.action_dim, teacher.T_pred)
                x0_teacher = teacher.diffusion.ddim_sample(
                    denoise_fn=lambda x, t, c: teacher.noise_pred_net(x, t, c),
                    condition=global_cond_t,
                    shape=shape,
                    ddim_steps=10,
                )

            # ------------------------------------------------------------------
            # Student: 1-step distillation loss
            # ------------------------------------------------------------------
            loss, metrics = student.compute_distillation_loss(
                img_stack, imu,
                teacher_x0=x0_teacher,
                depth_gt=depth_gt,
                lambda_dispersive=lambda_dispersive,
                lambda_depth=lambda_depth,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()

            epoch_losses.append(loss.item())
            epoch_l_dist.append(metrics['loss_distill'])
            epoch_l_disp.append(metrics['loss_dispersive'])
            epoch_l_depth.append(metrics['loss_depth'])
            global_step += 1

            if batch_idx % log_freq == 0:
                writer.add_scalar('train/loss',            loss.item(),                  global_step)
                writer.add_scalar('train/loss_distill',    metrics['loss_distill'],      global_step)
                writer.add_scalar('train/loss_dispersive', metrics['loss_dispersive'],   global_step)
                writer.add_scalar('train/loss_depth',      metrics['loss_depth'],        global_step)
                writer.add_scalar('train/lr',              scheduler.get_last_lr()[0],   global_step)

        mean_loss  = np.mean(epoch_losses)
        mean_dist  = np.mean(epoch_l_dist)
        mean_disp  = np.mean(epoch_l_disp)
        mean_depth = np.mean(epoch_l_depth)

        writer.add_scalar('train/epoch_loss',         mean_loss,  epoch)
        writer.add_scalar('train/epoch_loss_distill', mean_dist,  epoch)

        logger.info(
            f"Epoch {epoch+1:>3}/{num_epochs} | "
            f"loss={mean_loss:.6f} | "
            f"distill={mean_dist:.4f} "
            f"disp={mean_disp:.4f} "
            f"depth={mean_depth:.4f} | "
            f"LR={scheduler.get_last_lr()[0]:.2e}"
        )

        if mean_loss < best_loss:
            best_loss = mean_loss
            student.save(os.path.join(save_dir, "best_onedp_model.pt"))
            student.save_deployable(os.path.join(save_dir, "deploy_onedp_model.pt"))
            logger.info(f"  *** New best: loss={best_loss:.6f} → saved best_onedp_model.pt")

    student.save(os.path.join(save_dir, "final_onedp_model.pt"))
    writer.close()

    logger.info(f"\nDistillation complete! Best loss: {best_loss:.6f}")
    logger.info(f"Models saved to: {save_dir}")
    print(f"\nBest checkpoint: {save_dir}/best_onedp_model.pt")
    print(f"Deploy model:    {save_dir}/deploy_onedp_model.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OneDP 1-step distillation from v3.3 DPPO teacher")
    parser.add_argument('--teacher', type=str, required=True,
                        help='Path to DPPO fine-tuned teacher checkpoint (best_dppo_v33_model.pt)')
    parser.add_argument('--config',  type=str, default='configs/diffusion_policy.yaml')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Override path to expert_demos_v33.h5')
    parser.add_argument('--supervised-ckpt', type=str, default=None,
                        help='Supervised v33 ckpt for depth_decoder init (default: auto-detect)')
    parser.add_argument('--num-epochs',    type=int,   default=50)
    parser.add_argument('--batch-size',    type=int,   default=128)
    parser.add_argument('--lr',            type=float, default=3e-4)
    parser.add_argument('--weight-decay',  type=float, default=0.01)
    parser.add_argument('--warmup-epochs', type=int,   default=5)
    parser.add_argument('--grad-clip',     type=float, default=1.0)
    parser.add_argument('--lambda-disp',   type=float, default=0.05,
                        help='Dispersive loss weight (default: 0.05, lower than supervised 0.1)')
    parser.add_argument('--lambda-depth',  type=float, default=0.1,
                        help='Depth auxiliary loss weight')
    args = parser.parse_args()
    train(args)
