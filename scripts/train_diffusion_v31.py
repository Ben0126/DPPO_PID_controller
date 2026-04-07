"""
Phase 3a v3.1: Supervised Diffusion Policy Pre-training (IMU Late Fusion + Depth Aux)

Trains VisionDPPOv31 on expert_demos_v31.h5 using:
    L = L_diffusion + λ_disp × L_dispersive + λ_depth × MSE(depth_pred, depth_gt)

Requires: data/expert_demos_v31.h5 (collected with scripts/collect_data.py --v31)

Usage:
    python -m scripts.train_diffusion_v31 --config configs/diffusion_policy.yaml
"""

import os
import sys
import argparse
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vision_dppo_v31 import VisionDPPOv31, DemoDatasetV31


def train(args):
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    vision_cfg = config['vision']
    diff_cfg   = config['diffusion']
    unet_cfg   = config['unet']
    action_cfg = config['action']
    train_cfg  = config['training']
    log_cfg    = config['logging']
    v31_cfg    = config.get('v31', {})

    lambda_dispersive = v31_cfg.get('lambda_dispersive', 0.1)
    lambda_depth      = v31_cfg.get('lambda_depth', 0.1)
    dataset_path      = v31_cfg.get('dataset_path_v31', 'data/expert_demos_v31.h5')

    # Dataset
    dataset = DemoDatasetV31(
        hdf5_path=dataset_path,
        T_obs=vision_cfg['T_obs'],
        T_pred=action_cfg['T_pred'],
    )
    # num_workers=0: all data is pre-loaded into RAM in DemoDatasetV31.__init__,
    # so __getitem__ is a pure memory-copy; workers would each duplicate the full
    # ~8 GB dataset under Windows spawn, causing MemoryError.
    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Dataset: {len(dataset)} samples, {len(dataloader)} batches/epoch")
    print(f"λ_dispersive={lambda_dispersive}  λ_depth={lambda_depth}")

    # Model
    policy = VisionDPPOv31(
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
        use_depth_decoder=(lambda_depth > 0),
    ).to(device)

    n_params = sum(p.numel() for p in policy.parameters())
    print(f"Model parameters: {n_params:,}")

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=train_cfg['learning_rate'],
        weight_decay=train_cfg['weight_decay'],
    )
    num_epochs    = train_cfg['num_epochs']
    warmup_epochs = train_cfg.get('warmup_epochs', 10)
    total_steps   = num_epochs * len(dataloader)
    warmup_steps  = warmup_epochs * len(dataloader)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Logging
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag    = f"v31_{timestamp}"
    log_dir    = os.path.join(log_cfg['tensorboard_log'], run_tag)
    save_dir   = os.path.join(log_cfg['save_path'], run_tag)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    grad_clip        = train_cfg.get('grad_clip', 1.0)
    log_freq         = log_cfg.get('log_freq', 10)
    checkpoint_freq  = log_cfg.get('checkpoint_freq', 50)

    print(f"\n{'='*60}")
    print(f"Training VisionDPPOv31 (supervised pre-training)")
    print(f"Epochs: {num_epochs}, Batch: {train_cfg['batch_size']}")
    print(f"{'='*60}\n")

    global_step = 0
    best_loss   = float('inf')

    for epoch in range(num_epochs):
        policy.train()
        epoch_losses      = []
        epoch_l_diff      = []
        epoch_l_disp      = []
        epoch_l_depth     = []

        for batch_idx, (img_stack, action_seq, imu, depth_gt) in enumerate(dataloader):
            img_stack  = img_stack.to(device, non_blocking=True)
            action_seq = action_seq.to(device, non_blocking=True)
            imu        = imu.to(device, non_blocking=True)
            depth_gt   = depth_gt.to(device, non_blocking=True)

            # Option B: GPU brightness + contrast jitter (same as train_diffusion.py)
            B = img_stack.shape[0]
            brightness = 1.0 + (torch.rand(B, 1, 1, 1, device=device) - 0.5) * 0.6
            img_mean   = img_stack.mean(dim=(-2, -1), keepdim=True)
            contrast   = 1.0 + (torch.rand(B, 1, 1, 1, device=device) - 0.5) * 0.4
            img_stack  = torch.clamp(
                (img_stack - img_mean) * contrast + img_mean * brightness,
                0.0, 255.0
            )

            loss, metrics = policy.compute_loss(
                img_stack, action_seq, imu,
                depth_gt=depth_gt,
                lambda_dispersive=lambda_dispersive,
                lambda_depth=lambda_depth,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()

            epoch_losses.append(loss.item())
            epoch_l_diff.append(metrics['loss_diffusion'])
            epoch_l_disp.append(metrics['loss_dispersive'])
            epoch_l_depth.append(metrics['loss_depth'])
            global_step += 1

            if batch_idx % log_freq == 0:
                writer.add_scalar('train/loss',            loss.item(),                  global_step)
                writer.add_scalar('train/loss_diffusion',  metrics['loss_diffusion'],    global_step)
                writer.add_scalar('train/loss_dispersive', metrics['loss_dispersive'],   global_step)
                writer.add_scalar('train/loss_depth',      metrics['loss_depth'],        global_step)
                writer.add_scalar('train/lr',              scheduler.get_last_lr()[0],   global_step)

        mean_loss  = np.mean(epoch_losses)
        writer.add_scalar('train/epoch_loss', mean_loss, epoch)

        print(f"Epoch {epoch+1:>4}/{num_epochs} | "
              f"Loss: {mean_loss:.6f} | "
              f"diff={np.mean(epoch_l_diff):.4f} "
              f"disp={np.mean(epoch_l_disp):.4f} "
              f"depth={np.mean(epoch_l_depth):.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")

        if (epoch + 1) % checkpoint_freq == 0:
            ckpt_path = os.path.join(save_dir, f"v31_epoch_{epoch+1}.pt")
            policy.save(ckpt_path)
            print(f"  Checkpoint: {ckpt_path}")

        if mean_loss < best_loss:
            best_loss = mean_loss
            policy.save(os.path.join(save_dir, "best_model.pt"))

    policy.save(os.path.join(save_dir, "final_model.pt"))
    policy.save_deployable(os.path.join(save_dir, "deploy_model.pt"))

    writer.close()
    print(f"\nTraining complete! Best loss: {best_loss:.6f}")
    print(f"Models saved to: {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VisionDPPOv31 (supervised)")
    parser.add_argument('--config', type=str, default='configs/diffusion_policy.yaml')
    args = parser.parse_args()
    train(args)
