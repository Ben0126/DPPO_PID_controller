"""
可視化工具模組

包含所有評估和訓練相關的可視化函數。
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
from typing import Dict, List, Optional

from .training_metrics import TrainingMetricsTracker


def plot_airpilot_style_metrics(metrics_tracker: TrainingMetricsTracker, 
                                output_dir: str = "./training_metrics/"):
    """
    繪製 AirPilot 風格的訓練指標圖表（Fig.14-16）
    
    Args:
        metrics_tracker: TrainingMetricsTracker 實例
        output_dir: 輸出目錄
    """
    if len(metrics_tracker.timesteps) == 0:
        print("⚠️ 無法繪製訓練指標：沒有數據")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    timesteps = np.array(metrics_tracker.timesteps)
    
    # Fig.14: Effective Speed vs Timesteps
    effective_speeds = np.array(metrics_tracker.effective_speeds)
    valid_mask = ~np.isnan(effective_speeds)
    
    if np.any(valid_mask):
        axes[0].plot(timesteps[valid_mask], effective_speeds[valid_mask], 
                     'b-', linewidth=2, label='Effective Speed')
        axes[0].axhline(y=0.92, color='r', linestyle='--', 
                        label='Fine-tuned PID baseline', linewidth=2)
    axes[0].set_xlabel('Training Timesteps', fontsize=12)
    axes[0].set_ylabel('Effective Speed (m/s)', fontsize=12)
    axes[0].set_title('Effective Speed vs Training Timesteps', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Fig.15: Settling Time vs Timesteps
    settling_times = np.array(metrics_tracker.settling_times)
    valid_mask = ~np.isnan(settling_times)
    
    if np.any(valid_mask):
        axes[1].plot(timesteps[valid_mask], settling_times[valid_mask], 
                     'g-', linewidth=2, label='Settling Time')
    axes[1].set_xlabel('Training Timesteps', fontsize=12)
    axes[1].set_ylabel('Settling Time (s)', fontsize=12)
    axes[1].set_title('Settling Time vs Training Timesteps', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Fig.16: Overshoot vs Timesteps
    overshoots = np.array(metrics_tracker.overshoots)
    valid_mask = ~np.isnan(overshoots)
    
    if np.any(valid_mask):
        axes[2].plot(timesteps[valid_mask], overshoots[valid_mask], 
                     'r-', linewidth=2, label='Overshoot')
    axes[2].set_xlabel('Training Timesteps', fontsize=12)
    axes[2].set_ylabel('Overshoot (m)', fontsize=12)
    axes[2].set_title('Overshoot vs Training Timesteps', fontsize=14, fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'airpilot_style_metrics.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ 訓練指標圖表已保存到: {filepath}")
    plt.close()


def plot_episode(history, episode_idx, reward, output_dir, title_prefix="Episode"):
    """
    繪製單個回合的詳細圖表
    
    Args:
        history: Dictionary containing episode history
        episode_idx: Episode index
        reward: Total episode reward
        output_dir: Directory to save plots
        title_prefix: Prefix for plot title
    """
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(3, 2, figure=fig)

    time = np.array(history['time'])
    position = np.array(history['position'])
    reference = np.array(history['reference'])
    error = np.array(history['error'])
    control = np.array(history['control'])
    kp = np.array(history['kp'])
    ki = np.array(history['ki'])
    kd = np.array(history['kd'])
    velocity = np.array(history['velocity'])

    # Plot 1: Position Tracking
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(time, position, 'b-', label='Position', linewidth=2)
    ax1.plot(time, reference, 'r--', label='Reference', linewidth=2)
    ax1.set_ylabel('Position', fontsize=12)
    ax1.set_title(f'{title_prefix} Episode {episode_idx + 1} - Total Reward: {reward:.2f}', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Tracking Error
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(time, error, 'g-', linewidth=2)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.set_ylabel('Tracking Error', fontsize=12)
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Control Input
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(time, control, 'purple', linewidth=2)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax3.set_ylabel('Control Input (u)', fontsize=12)
    ax3.set_xlabel('Time (s)', fontsize=12)
    ax3.grid(True, alpha=0.3)

    # Plot 4: PID Gains Evolution
    ax4 = fig.add_subplot(gs[2, :])
    ax4.plot(time, kp, 'r-', label='Kp', linewidth=2)
    ax4.plot(time, ki, 'g-', label='Ki', linewidth=2)
    ax4.plot(time, kd, 'b-', label='Kd', linewidth=2)
    ax4.set_ylabel('PID Gains', fontsize=12)
    ax4.set_xlabel('Time (s)', fontsize=12)
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    filename = f"{title_prefix.lower()}_episode_{episode_idx + 1}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"  Saved: {filepath}")
    
    # 顯示圖表
    plt.show()
    plt.close()


def plot_summary(episode_rewards, episode_lengths, all_histories, output_dir):
    """
    繪製統計摘要圖表
    
    Args:
        episode_rewards: List of episode rewards
        episode_lengths: List of episode lengths
        all_histories: List of episode histories
        output_dir: Directory to save plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: Episode Rewards
    axes[0, 0].bar(range(1, len(episode_rewards) + 1), episode_rewards, color='steelblue')
    axes[0, 0].axhline(y=np.mean(episode_rewards), color='r', linestyle='--',
                       label=f'Mean: {np.mean(episode_rewards):.2f}')
    axes[0, 0].set_xlabel('Episode', fontsize=12)
    axes[0, 0].set_ylabel('Total Reward', fontsize=12)
    axes[0, 0].set_title('Episode Rewards', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Episode Lengths
    axes[0, 1].bar(range(1, len(episode_lengths) + 1), episode_lengths, color='coral')
    axes[0, 1].axhline(y=np.mean(episode_lengths), color='r', linestyle='--',
                       label=f'Mean: {np.mean(episode_lengths):.1f}')
    axes[0, 1].set_xlabel('Episode', fontsize=12)
    axes[0, 1].set_ylabel('Steps', fontsize=12)
    axes[0, 1].set_title('Episode Lengths', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Mean Absolute Error Distribution
    mean_errors = [np.mean(np.abs(h['error'])) for h in all_histories if h.get('error')]
    axes[1, 0].hist(mean_errors, bins=10, color='green', alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(x=np.mean(mean_errors), color='r', linestyle='--',
                       label=f'Mean: {np.mean(mean_errors):.4f}')
    axes[1, 0].set_xlabel('Mean Absolute Error', fontsize=12)
    axes[1, 0].set_ylabel('Frequency', fontsize=12)
    axes[1, 0].set_title('Tracking Error Distribution', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Final PID Gains Distribution
    final_kp = [h['kp'][-1] for h in all_histories if h.get('kp') and len(h['kp']) > 0]
    final_ki = [h['ki'][-1] for h in all_histories if h.get('ki') and len(h['ki']) > 0]
    final_kd = [h['kd'][-1] for h in all_histories if h.get('kd') and len(h['kd']) > 0]

    x_pos = np.arange(3)
    means = [np.mean(final_kp), np.mean(final_ki), np.mean(final_kd)]
    stds = [np.std(final_kp), np.std(final_ki), np.std(final_kd)]

    axes[1, 1].bar(x_pos, means, yerr=stds, color=['red', 'green', 'blue'],
                   alpha=0.7, capsize=5, edgecolor='black')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(['Kp', 'Ki', 'Kd'])
    axes[1, 1].set_ylabel('Gain Value', fontsize=12)
    axes[1, 1].set_title('Final PID Gains (Mean ± Std)', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save figure
    filepath = os.path.join(output_dir, 'evaluation_summary.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"  Saved: {filepath}")
    
    # 顯示圖表
    plt.show()
    plt.close()


def plot_gains_vs_error(history, output_dir, episode_idx, title_prefix="Episode"):
    """
    繪製 PID 增益 vs 位置誤差（參考 AirPilot Fig.17）
    
    Args:
        history: 回合歷史字典
        output_dir: 輸出目錄
        episode_idx: 回合索引
        title_prefix: 圖表標題前綴
    """
    if not history or not history.get('error') or len(history['error']) == 0:
        print(f"  ⚠️ 無法繪製 Gains vs Error：歷史資料為空")
        return
    
    error = np.abs(np.array(history['error']))  # 使用絕對誤差
    kp = np.array(history['kp'])
    ki = np.array(history['ki'])
    kd = np.array(history['kd'])
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Kp vs Error
    axes[0].scatter(error, kp, alpha=0.5, s=10, color='red', edgecolors='darkred', linewidths=0.5)
    axes[0].set_xlabel('|Position Error|', fontsize=12)
    axes[0].set_ylabel('Kp', fontsize=12)
    axes[0].set_title('Kp vs Position Error', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Ki vs Error
    axes[1].scatter(error, ki, alpha=0.5, s=10, color='green', edgecolors='darkgreen', linewidths=0.5)
    axes[1].set_xlabel('|Position Error|', fontsize=12)
    axes[1].set_ylabel('Ki', fontsize=12)
    axes[1].set_title('Ki vs Position Error', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # Kd vs Error
    axes[2].scatter(error, kd, alpha=0.5, s=10, color='blue', edgecolors='darkblue', linewidths=0.5)
    axes[2].set_xlabel('|Position Error|', fontsize=12)
    axes[2].set_ylabel('Kd', fontsize=12)
    axes[2].set_title('Kd vs Position Error', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle(f'{title_prefix} {episode_idx + 1}: PID Gains vs Position Error', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # 保存
    filename = f'{title_prefix.lower()}_gains_vs_error_ep{episode_idx + 1}.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"  Saved: {filepath}")
    plt.close()


def plot_demo_results(history):
    """
    繪製演示結果
    
    Args:
        history: Dictionary containing episode history
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    time = np.array(history['time'])
    position = np.array(history['position'])
    reference = np.array(history['reference'])
    error = np.array(history['error'])
    control = np.array(history['control'])
    kp = np.array(history['kp'])
    ki = np.array(history['ki'])
    kd = np.array(history['kd'])

    # Plot 1: Position Tracking
    axes[0].plot(time, position, 'b-', label='Position', linewidth=2)
    axes[0].plot(time, reference, 'r--', label='Reference', linewidth=2)
    axes[0].set_ylabel('Position', fontsize=12)
    axes[0].set_title('Demo Episode - Random PID Gains', fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Tracking Error and Control
    ax2_twin = axes[1].twinx()
    axes[1].plot(time, error, 'g-', label='Error', linewidth=2)
    ax2_twin.plot(time, control, 'purple', label='Control', linewidth=2, alpha=0.7)
    axes[1].set_ylabel('Tracking Error', fontsize=12, color='g')
    ax2_twin.set_ylabel('Control Input (u)', fontsize=12, color='purple')
    axes[1].tick_params(axis='y', labelcolor='g')
    ax2_twin.tick_params(axis='y', labelcolor='purple')
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1].grid(True, alpha=0.3)

    # Plot 3: PID Gains
    axes[2].plot(time, kp, 'r-', label='Kp', linewidth=2)
    axes[2].plot(time, ki, 'g-', label='Ki', linewidth=2)
    axes[2].plot(time, kd, 'b-', label='Kd', linewidth=2)
    axes[2].set_ylabel('PID Gains', fontsize=12)
    axes[2].set_xlabel('Time (s)', fontsize=12)
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    plt.savefig('demo_results.png', dpi=150, bbox_inches='tight')
    print("\n✓ Demo plot saved to: demo_results.png")

    plt.show()

