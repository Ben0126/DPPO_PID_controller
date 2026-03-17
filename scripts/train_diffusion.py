"""
Phase 3: Supervised Diffusion Policy Training

Trains the Vision Diffusion Policy using expert demonstrations.
Loss = MSE between predicted noise and actual noise added to actions.

Usage:
    python -m scripts.train_diffusion
    python -m scripts.train_diffusion --config configs/diffusion_policy.yaml
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

from models.diffusion_policy import VisionDiffusionPolicy, DemoDataset


def train(args):
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    vision_cfg = config['vision']
    diff_cfg = config['diffusion']
    unet_cfg = config['unet']
    action_cfg = config['action']
    train_cfg = config['training']
    log_cfg = config['logging']

    # Create dataset
    dataset = DemoDataset(
        hdf5_path=train_cfg['dataset_path'],
        T_obs=vision_cfg['T_obs'],
        T_pred=action_cfg['T_pred'],
    )
    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Dataset: {len(dataset)} samples, {len(dataloader)} batches/epoch")

    # Create model
    policy = VisionDiffusionPolicy(
        action_dim=action_cfg['action_dim'],
        T_obs=vision_cfg['T_obs'],
        T_pred=action_cfg['T_pred'],
        image_channels=vision_cfg['channels'],
        image_size=vision_cfg['image_size'],
        feature_dim=vision_cfg['feature_dim'],
        time_embed_dim=unet_cfg['time_embed_dim'],
        down_dims=tuple(unet_cfg['down_dims']),
        num_diffusion_steps=diff_cfg['num_timesteps'],
        beta_schedule=diff_cfg['beta_schedule'],
        ddim_steps=diff_cfg['ddim_steps'],
    ).to(device)

    n_params = sum(p.numel() for p in policy.parameters())
    print(f"Model parameters: {n_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=train_cfg['learning_rate'],
        weight_decay=train_cfg['weight_decay'],
    )

    # Learning rate scheduler
    num_epochs = train_cfg['num_epochs']
    warmup_epochs = train_cfg.get('warmup_epochs', 10)
    total_steps = num_epochs * len(dataloader)
    warmup_steps = warmup_epochs * len(dataloader)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(log_cfg['tensorboard_log'], timestamp)
    save_dir = os.path.join(log_cfg['save_path'], timestamp)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    grad_clip = train_cfg.get('grad_clip', 1.0)
    log_freq = log_cfg.get('log_freq', 10)
    checkpoint_freq = log_cfg.get('checkpoint_freq', 50)

    print(f"\n{'='*60}")
    print(f"Training Vision Diffusion Policy")
    print(f"Epochs: {num_epochs}, Batch size: {train_cfg['batch_size']}")
    print(f"LR: {train_cfg['learning_rate']}, Warmup: {warmup_epochs} epochs")
    print(f"{'='*60}\n")

    # Training loop
    global_step = 0
    best_loss = float('inf')

    for epoch in range(num_epochs):
        policy.train()
        epoch_losses = []

        for batch_idx, (img_stack, action_seq) in enumerate(dataloader):
            img_stack = img_stack.to(device)
            action_seq = action_seq.to(device)

            loss = policy.compute_loss(img_stack, action_seq)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()

            epoch_losses.append(loss.item())
            global_step += 1

            if batch_idx % log_freq == 0:
                writer.add_scalar('train/loss', loss.item(), global_step)
                writer.add_scalar('train/lr', scheduler.get_last_lr()[0], global_step)

        # Epoch summary
        mean_loss = np.mean(epoch_losses)
        writer.add_scalar('train/epoch_loss', mean_loss, epoch)

        print(f"Epoch {epoch+1:>4}/{num_epochs} | "
              f"Loss: {mean_loss:.6f} | "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")

        # Checkpoint
        if (epoch + 1) % checkpoint_freq == 0:
            ckpt_path = os.path.join(save_dir, f"diffusion_epoch_{epoch+1}.pt")
            policy.save(ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

        # Best model
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_path = os.path.join(save_dir, "best_model.pt")
            policy.save(best_path)

    # Save final model
    final_path = os.path.join(save_dir, "final_model.pt")
    policy.save(final_path)

    writer.close()
    print(f"\nTraining complete! Best loss: {best_loss:.6f}")
    print(f"Models saved to: {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Vision Diffusion Policy")
    parser.add_argument('--config', type=str, default='configs/diffusion_policy.yaml')
    args = parser.parse_args()
    train(args)
